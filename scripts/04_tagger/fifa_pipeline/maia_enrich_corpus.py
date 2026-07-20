#!/usr/bin/env python3
"""Maia enrichment pass over the FIFA band corpus (Gauntlet drill ranking, issue #69).

Same 186K positions the FIFA scripts consume (fifa_sweep.json x fifa_enrich.json) — this adds
the ONE missing field the frontier ranking needs: popByElo per candidate move, plus a
consistent-depth SF eval for every candidate.

Per position:
  - Query Elos: 5 points [-400,-200,0,+200,+400] around the position's BAND CENTER, converted
    Lichess-rapid -> Maia's Lichess-blitz scale (same anchors as frontend maiaEloConversion.ts),
    rounded to 100, clamped to Maia 3's trained range [1100, 2600]. Index 2 = own band,
    index 3 = band above — matching frontierRanking.ts PLAYER_BAND_IDX/ABOVE_BAND_IDX.
  - Candidates: moves with >=2% Maia pop at ANY of the 5 Elos, plus the played move, plus the
    d16 best move (first move of enrich top_3_best lines). Capped at 12 by max pop.
  - SF: ONE searchmoves multipv call at depth 16 over all candidates -> cp (mover POV) each.
    d16 (Sam's call): these cp values grade pass/fail (heart loss) AND get displayed as
    "which moves were good" — d12 misjudging a sharp candidate = false heart loss. d16 also
    matches the corpus re-detect depth, so cp_loss/eval_before stay depth-consistent.
    ~4-5h on 60 procs (re-detect benchmark: 5k pos/48s at d16 MultiPV=1 on 48 procs, x12 PVs).

Output: fifa_maia_enrich_<band>.json per band, restartable (skips bands whose file exists).
Row: {fen, played_uci, cp_loss, eval_before, eval_after, elos:[5], cands:[{uci, pop:[5], cp}]}

Run on chess-poc (GPU for Maia batch, 60 procs for SF):
  screen -dmS maia_enrich bash -c 'cd ~/SageMaker && python3 fifa_pipeline/maia_enrich_corpus.py > maia_enrich.log 2>&1'
"""
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import chess
import chess.engine
import onnxruntime as ort

BASE = os.path.expanduser("~/SageMaker")
SWEEP = os.path.join(BASE, "fifa_sweep.json")
ENRICH = os.path.join(BASE, "fifa_enrich.json")
MODEL = os.path.join(BASE, "maia3_models/maia3_simplified.onnx")
STOCKFISH = os.path.join(BASE, "stockfish_compiled")
OUT_DIR = os.path.join(BASE, "fifa_maia")

MAIA_MIN, MAIA_MAX = 1100, 2600
ELO_OFFSETS = [-400, -200, 0, 200, 400]
POP_THRESHOLD = 2.0      # % at any band -> candidate
MAX_CANDS = 12
SF_DEPTH = 16
SF_PROCS = 60
MAIA_BATCH = 1024        # (fens x 5 elos) rows per ONNX run
# Maia stage parallelism: 12 procs x 4 ORT threads ≈ 48 cores. Single-proc measured 22 pos/s
# at ~8 cores (87% box idle) — sharding is what actually uses the machine.
MAIA_PROCS = 12
ORT_THREADS = 4
# Band order: Sam's bands first so the product is playable while the rest cook.
BAND_PRIORITY = ["1800-2000", "1600-1800", "2000-2200", "1400-1600"]

# Lichess Rapid -> Lichess Blitz (frontend maiaEloConversion.ts LI_RAPID_TO_LI_BLITZ).
LI_RAPID_TO_BLITZ = [(1205, 1030), (1270, 1075), (1340, 1145), (1400, 1200), (1515, 1335),
                     (1615, 1420), (1690, 1475), (1765, 1565), (1825, 1635), (1880, 1705),
                     (1930, 1780), (1990, 1850), (2035, 1910), (2085, 1970), (2135, 2050),
                     (2185, 2100), (2240, 2170), (2285, 2235), (2330, 2295), (2380, 2370),
                     (2445, 2445), (2510, 2560), (2595, 2625), (2630, 2695), (2705, 2780),
                     (2735, 2850)]


def rapid_to_blitz(r):
    t = LI_RAPID_TO_BLITZ
    if r <= t[0][0]:
        return t[0][1]
    if r >= t[-1][0]:
        return t[-1][1]
    for i in range(1, len(t)):
        x1, y1 = t[i]
        if r <= x1:
            x0, y0 = t[i - 1]
            return round(y0 + (r - x0) / (x1 - x0) * (y1 - y0))
    return t[-1][1]


def band_elos(band):
    """5 clamped Maia query Elos for a band like '1600-1800'."""
    lo, hi = band.split("-")
    mid = (int(lo) + int(hi)) // 2
    center = round(rapid_to_blitz(mid) / 100) * 100
    return [max(MAIA_MIN, min(MAIA_MAX, center + o)) for o in ELO_OFFSETS]


# ---- Maia 3 ONNX plumbing (port of backend/mcp/maia3_engine.py, batched) ----
PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
FILES = 'abcdefgh'
RANKS = '12345678'
PROMOS = 'qrbn'


def _build_vocab():
    out = []
    for r in RANKS:
        for f in FILES:
            for r2 in RANKS:
                for f2 in FILES:
                    out.append(f + r + f2 + r2)
    for ff in FILES:
        for tf in FILES:
            if abs(ord(ff) - ord(tf)) <= 1:
                for p in PROMOS:
                    out.append(ff + '7' + tf + '8' + p)
            else:
                out.extend([''] * 4)
    return out


MOVE_VOCAB = _build_vocab()
MOVE_INDEX = {m: i for i, m in enumerate(MOVE_VOCAB) if m}


def mirror_move(uci):
    def ms(sq):
        return sq[0] + str(9 - int(sq[1]))
    return ms(uci[:2]) + ms(uci[2:4]) + (uci[4:] if len(uci) > 4 else '')


def mirror_fen(fen):
    parts = fen.split()
    pos, active, castling, ep = parts[0], parts[1], parts[2], parts[3]
    ranks = []
    for rank in reversed(pos.split('/')):
        ranks.append(''.join(c.lower() if c.isupper() else c.upper() if c.islower() else c
                             for c in rank))
    mc = ''
    if castling != '-':
        has = set(castling)
        if 'k' in has:
            mc += 'K'
        if 'q' in has:
            mc += 'Q'
        if 'K' in has:
            mc += 'k'
        if 'Q' in has:
            mc += 'q'
    mep = ep if ep == '-' else ep[0] + str(9 - int(ep[1]))
    return f"{'/'.join(ranks)} {'b' if active == 'w' else 'w'} {mc or '-'} {mep} 0 1"


def fen_to_tokens(fen):
    tokens = np.zeros((64, 12), dtype=np.float32)
    pm = {c: i for i, c in enumerate(PIECE_CHARS)}
    for row_idx, rank_str in enumerate(fen.split()[0].split('/')):
        rank = 7 - row_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                if ch in pm:
                    tokens[rank * 8 + file, pm[ch]] = 1.0
                file += 1
    return tokens


def prep_position(fen):
    """tokens + legal vocab indices + uci list, in the model's white-POV frame."""
    board = chess.Board(fen)
    mirrored = board.turn == chess.BLACK
    eff_fen = mirror_fen(fen) if mirrored else fen
    eff_board = chess.Board(eff_fen)
    legal_idx, legal_uci = [], []
    for mv in eff_board.legal_moves:
        u = mv.uci()
        i = MOVE_INDEX.get(u)
        if i is not None:
            legal_idx.append(i)
            legal_uci.append(mirror_move(u) if mirrored else u)
    return fen_to_tokens(eff_fen), np.array(legal_idx), legal_uci


def _pops_from_logits(logits_5, legal_idx, legal_uci, played):
    """5 logit rows -> {uci: [5 pops]} for kept moves. Returns None if any prob is NaN."""
    probs_by_k = []
    for k in range(5):
        lg = logits_5[k][legal_idx].astype(np.float64)
        lg -= lg.max()
        p = np.exp(lg)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        p /= s
        if not np.all(np.isfinite(p)):
            return None
        probs_by_k.append(p)
    pops = {}
    for k, p in enumerate(probs_by_k):
        for u, prob in zip(legal_uci, p):
            v = float(prob) * 100
            if v >= POP_THRESHOLD or u == played:
                pops.setdefault(u, [0.0] * 5)
    for u in pops:
        ui = legal_uci.index(u)
        for k, p in enumerate(probs_by_k):
            pops[u][k] = round(float(p[ui]) * 100, 1)
    return pops


def maia_stage(rows, elos_by_band):
    """rows: [{fen, uci, band, ...}] -> {fen|uci: {uci: [5 pops]}} for candidate moves.

    FRESH ONNX SESSION PER CHUNK: a long-lived CPU session degrades over many batched
    runs and starts emitting NaN logits (hit 56% of band 600-800 on the first run; the
    research activations script works per-chunk for the same reason). Any position whose
    probs still come out NaN is retried once, alone, on another fresh session.
    """
    preps = []
    for r in rows:
        try:
            preps.append(prep_position(r["fen"]))
        except Exception:
            preps.append(None)

    out = {}
    retry = []  # (i,) positions whose chunk produced NaN
    fens_per_chunk = max(1, MAIA_BATCH // 5)
    t0 = time.time()
    first = True
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    for start in range(0, len(rows), fens_per_chunk):
        sess = ort.InferenceSession(MODEL, sess_options=so, providers=ort.get_available_providers())
        if first:
            print(f"  ONNX providers: {sess.get_providers()}", flush=True)
            first = False
        in_names = [i.name for i in sess.get_inputs()]
        chunk = list(range(start, min(start + fens_per_chunk, len(rows))))
        idxs = [i for i in chunk if preps[i] is not None]
        if not idxs:
            continue
        tokens = np.concatenate([np.repeat(preps[i][0][None], 5, axis=0) for i in idxs])
        e_self = np.concatenate([np.array(elos_by_band[rows[i]["band"]], dtype=np.float32)
                                 for i in idxs])
        feeds = {in_names[0]: tokens, in_names[1]: e_self, in_names[2]: e_self}
        logits = sess.run(None, feeds)[0]  # (n*5, 4352)

        for j, i in enumerate(idxs):
            _, legal_idx, legal_uci = preps[i]
            played = rows[i]["uci"]
            pops = _pops_from_logits(logits[j * 5:(j + 1) * 5], legal_idx, legal_uci, played)
            if pops is None:
                retry.append(i)
            else:
                out[f'{rows[i]["fen"]}|{played}'] = pops
        del sess
        if (start // fens_per_chunk) % 20 == 0:
            done = min(start + fens_per_chunk, len(rows))
            print(f"  maia {done}/{len(rows)} ({time.time()-t0:.0f}s, {len(retry)} retries)",
                  flush=True)

    # Retry NaN positions one-by-one on fresh sessions; drop them if still NaN (logged).
    dropped = 0
    for i in retry:
        sess = ort.InferenceSession(MODEL, providers=ort.get_available_providers())
        in_names = [inp.name for inp in sess.get_inputs()]
        tokens = np.repeat(preps[i][0][None], 5, axis=0)
        e_self = np.array(elos_by_band[rows[i]["band"]], dtype=np.float32)
        logits = sess.run(None, {in_names[0]: tokens, in_names[1]: e_self,
                                 in_names[2]: e_self})[0]
        _, legal_idx, legal_uci = preps[i]
        played = rows[i]["uci"]
        pops = _pops_from_logits(logits, legal_idx, legal_uci, played)
        if pops is None:
            dropped += 1
        else:
            out[f'{rows[i]["fen"]}|{played}'] = pops
        del sess
    if retry:
        print(f"  maia retries: {len(retry)}, dropped after retry: {dropped}", flush=True)
    return out


def _maia_shard(args):
    """Pool worker: run maia_stage on a shard of rows. Own ONNX sessions per chunk."""
    rows_shard, elos_by_band = args
    return maia_stage(rows_shard, elos_by_band)


def maia_stage_parallel(rows, elos_by_band):
    """Shard rows across MAIA_PROCS processes (each ORT_THREADS intra-op threads).
    Single-proc ONNX only drove ~13% of the 64-core box; 12x4 uses it properly."""
    if len(rows) < 2000:  # small band — sharding overhead not worth it
        return maia_stage(rows, elos_by_band)
    shard_size = (len(rows) + MAIA_PROCS - 1) // MAIA_PROCS
    shards = [rows[i:i + shard_size] for i in range(0, len(rows), shard_size)]
    out = {}
    t0 = time.time()
    with Pool(MAIA_PROCS) as pool:
        for n, part in enumerate(pool.imap_unordered(
                _maia_shard, [(s, elos_by_band) for s in shards])):
            out.update(part)
            print(f"  maia shard {n + 1}/{len(shards)} merged "
                  f"({len(out)}/{len(rows)}, {time.time()-t0:.0f}s)", flush=True)
    return out


# ---- SF stage ----
_num = re.compile(r"^\d+\.+$")


def line_first_uci(fen, line):
    """First move of an enrich top_3_best SAN line -> uci ('' on failure)."""
    for tok in (line or "").replace("...", ". ").split():
        if _num.match(tok) or (tok and tok[0].isdigit()):
            continue
        try:
            return chess.Board(fen).parse_san(tok).uci()
        except Exception:
            return ""
    return ""


_engine = None


def _sf_init():
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _engine.configure({"Threads": 1, "Hash": 64})


def _sf_job(job):
    """(fen, [ucis]) -> {uci: cp mover-POV}."""
    fen, ucis = job
    board = chess.Board(fen)
    root = []
    for u in ucis:
        try:
            mv = chess.Move.from_uci(u)
            if mv in board.legal_moves:
                root.append(mv)
        except Exception:
            pass
    if not root:
        return fen, {}
    try:
        infos = _engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH),
                                root_moves=root, multipv=len(root))
    except Exception:
        return fen, {}
    evals = {}
    for info in infos:
        pv = info.get("pv")
        if pv:
            evals[pv[0].uci()] = info["score"].pov(board.turn).score(mate_score=10000)
    return fen, evals


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading sweep + enrich...", flush=True)
    sweep = json.load(open(SWEEP))
    enrich = json.load(open(ENRICH))
    by_band = {}
    for r in sweep:
        by_band.setdefault(r["band"], []).append(r)
    elos_by_band = {b: band_elos(b) for b in by_band}
    for b in sorted(by_band):
        print(f"{b}: {len(by_band[b])} positions, elos {elos_by_band[b]}", flush=True)

    ordered = BAND_PRIORITY + sorted(
        (b for b in by_band if b not in BAND_PRIORITY), key=lambda b: int(b.split("-")[0]))
    for band in ordered:
        if band not in by_band:
            continue
        out_path = os.path.join(OUT_DIR, f"fifa_maia_enrich_{band}.json")
        if os.path.exists(out_path):
            print(f"== {band}: exists, skipping", flush=True)
            continue
        rows = by_band[band]
        print(f"== {band}: {len(rows)} positions — Maia stage", flush=True)
        pops_by_key = maia_stage_parallel(rows, elos_by_band)

        # Build SF jobs: Maia cands (capped) + played + d16 best
        jobs, metas = [], []
        for r in rows:
            key = f'{r["fen"]}|{r["uci"]}'
            pops = pops_by_key.get(key)
            if pops is None:
                continue
            ce = enrich.get(key, {})
            best_uci = ""
            if ce.get("top_3_best"):
                best_uci = line_first_uci(r["fen"], ce["top_3_best"][0].get("line", ""))
            ranked = sorted(pops.items(), key=lambda kv: -max(kv[1]))
            cands = [u for u, _ in ranked[:MAX_CANDS]]
            for must in (r["uci"], best_uci):
                if must and must not in cands:
                    cands.append(must)
                    pops.setdefault(must, [0.0] * 5)
            jobs.append((r["fen"], cands))
            metas.append((r, key, pops, cands))

        print(f"== {band}: SF stage, {len(jobs)} positions x depth {SF_DEPTH}", flush=True)
        t0 = time.time()
        evals_by_fen = {}
        with Pool(SF_PROCS, initializer=_sf_init) as pool:
            for n, (fen, evals) in enumerate(pool.imap_unordered(_sf_job, jobs, chunksize=16)):
                evals_by_fen[fen] = evals
                if n % 2000 == 0:
                    print(f"  sf {n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

        out_rows = []
        for r, key, pops, cands in metas:
            evals = evals_by_fen.get(r["fen"], {})
            out_rows.append({
                "fen": r["fen"],
                "played_uci": r["uci"],
                "cp_loss": r.get("cp_loss"),
                "eval_before": r.get("eval_before"),
                "eval_after": r.get("eval_after"),
                "elos": elos_by_band[band],
                "cands": [{"uci": u, "pop": pops[u], "cp": evals.get(u)} for u in cands],
            })
        json.dump(out_rows, open(out_path, "w"))
        print(f"== {band}: wrote {len(out_rows)} rows -> {out_path}", flush=True)

    print("ALL BANDS DONE", flush=True)


if __name__ == "__main__":
    main()
