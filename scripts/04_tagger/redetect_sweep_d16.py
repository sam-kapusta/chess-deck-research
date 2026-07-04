#!/usr/bin/env python3
"""Phase 2 of the depth-16 sweep re-detection.

Re-detect rating-band blunders with OUR Stockfish at DEPTH 16, MultiPV=1, over the SAME fixed
game set the original sweep used (sweep_blunders_2000.json). The original set was selected from
Lichess's own [%eval] annotations (variable depth ~20+); this re-detects with the engine/depth the
PRODUCT runs, so the per-band baselines match what a live user's mistakes are measured against.

Why MultiPV=1 is correct here: the band tagger (run_rating_bands.py -> build_mistake ->
tag_mistake_full) reads ONLY PV1 — best_line = top_3_best[0].line, refutation = top_3_refutations[0]
.line — plus eval_before/after/cp_loss/played_san/best_san. The Mistake dataclass has no field for
n_good_moves or punish_type (those need MultiPV but are only read by the SEPARATE tag_game.py deep
path, not the sweep). So MPV=1 produces an enrichment record the tagger consumes UNCHANGED, at ~3.35x
less engine cost than MPV=3.

Method (one pass, exact cp_loss parity with enrich_all_positions.enrich_position):
  For a game with n moves -> positions p_0..p_{n-1} (before each move) + final board (after last).
  Analyze each of those n+1 positions ONCE at depth 16 -> (white-POV score, PV).
  For move i by the side whose Elo falls in a valid band:
    eval_before = score(p_i); eval_after = score(position after move i) = score(p_{i+1}|final)
    cp_loss     = (before-after) if white else (after-before), clamped >=0   [== enrich_position]
    best line   = PV of p_i ; refutation line = PV of the position AFTER move i
  Blunder iff cp_loss >= 200 (same threshold as the original generator).

Banding: per-player, reproducing the original generator — white's blunders bucketed by WhiteElo,
black's by BlackElo, via get_band (half-open [lo,hi), only 1000-2200). Run on the fixed game set this
reproduces the original band memberships; only WHICH moves qualify changes (depth 16 < Lichess depth).

Outputs (depth16/MPV1, parallel to the originals):
  sweep_blunders_d16.json            rows: {band, game_id, outcome, cp_loss, fen, uci}
  position_enrichment_cache_d16.json keyed "fen|uci", schema == position_enrichment_cache.json
                                     (position_features omitted -> [] ; band tagger never reads it)
Resumable: skips games already in the done-set; checkpoints every CHECKPOINT_EVERY games.
"""
import json, io, os, time, threading
import chess, chess.engine, chess.pgn
from concurrent.futures import ThreadPoolExecutor

SF       = "/home/ec2-user/SageMaker/stockfish_compiled"   # Stockfish 16.1
SWEEP    = "/home/ec2-user/SageMaker/sweep_blunders_2000.json"
PGNS     = "/home/ec2-user/SageMaker/sweep_pgns_cache.json"
OUT_BL   = "/home/ec2-user/SageMaker/sweep_blunders_d16.json"
OUT_EN   = "/home/ec2-user/SageMaker/position_enrichment_cache_d16.json"
OUT_DONE = "/home/ec2-user/SageMaker/sweep_d16_done.json"

DEPTH = 16
WORKERS = 48
THRESH = 200
CHECKPOINT_EVERY = 200   # games

BANDS = [("1000-1200",1000,1200),("1200-1400",1200,1400),("1400-1600",1400,1600),
         ("1600-1800",1600,1800),("1800-2000",1800,2000),("2000-2200",2000,2200)]

def get_band(elo):
    for name, lo, hi in BANDS:
        if lo <= elo < hi:
            return name
    return None

def parse_result(result, is_white):
    if result == "1-0":     return "win"  if is_white else "loss"
    if result == "0-1":     return "loss" if is_white else "win"
    if result == "1/2-1/2": return "draw"
    return None

def format_line(board, pv):
    """'30. Rf3 Re6+ 31. Re3' — identical to enrich_all_positions.format_line."""
    temp = board.copy(); parts = []
    for i, m in enumerate(pv[:5]):
        san = temp.san(m)
        if temp.turn == chess.WHITE:
            parts.append(f"{temp.fullmove_number}. {san}")
        else:
            parts.append(f"{temp.fullmove_number}... {san}" if i == 0 else san)
        temp.push(m)
    return " ".join(parts)

def get_phase(board):
    pc = len(board.piece_map())
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    w_c = wk is not None and wk != chess.E1
    b_c = bk is not None and bk != chess.E8
    if pc <= 14: return "endgame"
    if not w_c and not b_c and board.fullmove_number <= 10: return "opening"
    if (w_c and b_c) or board.fullmove_number > 12: return "middlegame"
    return "opening"

# ---- shared state ----
lock = threading.Lock()
blunders = []          # list of rows
enrich   = {}          # "fen|uci" -> record
done_ids = set()
counter  = {"games": 0, "t0": time.time()}

def checkpoint():
    for path, obj in ((OUT_BL, blunders), (OUT_EN, enrich), (OUT_DONE, sorted(done_ids))):
        tmp = path + ".new"
        json.dump(obj, open(tmp, "w"))
        os.replace(tmp, path)

def detect_game(engine, game_id, pgn_text, white_band, black_band):
    """Return (rows, records) for one game's blunders by in-band players."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None: return [], {}
    result = game.headers.get("Result", "*")
    moves = list(game.mainline_moves())
    if not moves: return [], {}

    # boards before each move + final board
    pre = []; b = game.board()
    for mv in moves:
        pre.append(b.copy()); b.push(mv)
    final_board = b
    boards = pre + [final_board]                      # len n+1

    # analyze each position once (MPV=1: omit multipv -> single InfoDict)
    infos = [engine.analyse(bd, chess.engine.Limit(depth=DEPTH)) for bd in boards]
    scores = [inf["score"].white() for inf in infos]  # POV white
    pvs    = [inf.get("pv", []) for inf in infos]

    rows = []; recs = {}
    for i, mv in enumerate(moves):
        pb = pre[i]
        mover_white = pb.turn == chess.WHITE
        band = white_band if mover_white else black_band
        if band is None:
            continue
        eb = scores[i].score(mate_score=10000)
        ea = scores[i+1].score(mate_score=10000)
        cp_loss = (eb - ea) if mover_white else (ea - eb)
        cp_loss = max(int(cp_loss), 0)
        if cp_loss < THRESH:
            continue
        fen = pb.fen(); uci = mv.uci()
        played_san = pb.san(mv)
        best_line = format_line(pb, pvs[i])
        after_board = boards[i+1]
        refut_line = format_line(after_board, pvs[i+1])
        # match enrich_position: 2nd token if the line carries a move number, else 1st; guard empty
        if best_line:
            toks = best_line.split()
            best_san = toks[1] if ("." in best_line and len(toks) > 1) else toks[0]
        else:
            best_san = ""
        outcome = parse_result(result, mover_white)
        rows.append({"band": band, "game_id": game_id, "outcome": outcome,
                     "cp_loss": cp_loss, "fen": fen, "uci": uci})
        recs[f"{fen}|{uci}"] = {
            "fen": fen, "uci": uci, "played_san": played_san, "best_san": best_san,
            "eval_before": str(scores[i]), "eval_after": str(scores[i+1]),
            "cp_loss": cp_loss, "side": "White" if mover_white else "Black",
            "phase": get_phase(pb), "n_good_moves": None, "punish_type": "",
            "top_3_best": [{"line": best_line, "eval": str(scores[i])}],
            "top_3_refutations": [{"line": refut_line, "eval": str(scores[i+1])}],
            "position_features": [],   # band tagger never reads this; omitted to save the heavy port
        }
    return rows, recs

def worker(chunk):
    engine = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        engine.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass
    for game_id, pgn_text, wb, bb in chunk:
        try:
            rows, recs = detect_game(engine, game_id, pgn_text, wb, bb)
        except Exception as e:
            rows, recs = [], {}
            print(f"  ERR {game_id}: {str(e)[:120]}", flush=True)
        with lock:
            blunders.extend(rows)
            enrich.update(recs)
            done_ids.add(game_id)
            counter["games"] += 1
            n = counter["games"]
            if n % 100 == 0:
                el = (time.time() - counter["t0"]) / 60
                print(f"  {n} games | {len(blunders)} blunders | {el:.1f}min", flush=True)
            if n % CHECKPOINT_EVERY == 0:
                checkpoint()
    engine.quit()

def main():
    rows = json.load(open(SWEEP))
    pgns = json.load(open(PGNS))

    # per game_id -> the set of bands it appears under (drives which side(s) we detect)
    # reproduce generator banding from the PGN's own Elos via get_band, intersected with the
    # bands this game is actually present in (the original cap already fixed membership).
    present = {}
    for r in rows:
        present.setdefault(r["game_id"], set()).add(r["band"])

    # resume
    if os.path.exists(OUT_DONE):
        for gid in json.load(open(OUT_DONE)):
            done_ids.add(gid)
        if os.path.exists(OUT_BL): blunders.extend(json.load(open(OUT_BL)))
        if os.path.exists(OUT_EN): enrich.update(json.load(open(OUT_EN)))
        print(f"resuming: {len(done_ids)} games done, {len(blunders)} blunders so far", flush=True)

    work = []
    missing = 0
    for game_id, bands_here in present.items():
        if game_id in done_ids:
            continue
        pgn_text = pgns.get(game_id)
        if not pgn_text:
            missing += 1
            continue
        g = chess.pgn.read_game(io.StringIO(pgn_text))
        if g is None:
            missing += 1
            continue
        we = int(g.headers.get("WhiteElo") or 0)
        be = int(g.headers.get("BlackElo") or 0)
        wb = get_band(we); bb = get_band(be)
        # only detect for sides whose band the game is actually present under (parity w/ original set)
        wb = wb if wb in bands_here else None
        bb = bb if bb in bands_here else None
        if wb is None and bb is None:
            continue
        work.append((game_id, pgn_text, wb, bb))

    print(f"games to process: {len(work)} | missing PGNs: {missing} | depth {DEPTH} MPV=1 | {WORKERS} engines", flush=True)
    if not work:
        checkpoint(); print("nothing to do.", flush=True); return

    chunks = [work[i::WORKERS] for i in range(WORKERS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(worker, chunks))

    checkpoint()
    el = (time.time() - counter["t0"]) / 60
    # per-band summary
    from collections import Counter
    by_band = Counter(r["band"] for r in blunders)
    print(f"\nDONE {counter['games']} games | {len(blunders)} blunders | {el:.1f}min", flush=True)
    for b, _, _ in BANDS:
        print(f"  {b}: {by_band.get(b,0)} blunders", flush=True)
    print(f"-> {OUT_BL}\n-> {OUT_EN}", flush=True)

if __name__ == "__main__":
    main()
