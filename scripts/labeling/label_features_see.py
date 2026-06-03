#!/usr/bin/env python3
"""Feature-level Opus labeler WITH SEE stats on BOTH moves as raw data.

The right method (per Sam): Opus reads each feature's top-N positions AS A SET and names
the shared mistake holistically — but it's also handed SEE numbers computed on BOTH the
played (blunder) move AND the best move, as raw data to ground the judgment.

SEE-on-both makes the f127 case unambiguous: "best move wins a hanging enemy piece (SEE>0)
in 9/10, player played a non-capture" => "Missed Hanging Piece". Per-position SEE lines +
an aggregate block are injected into the prompt alongside the existing Opus per-position
analysis + Stockfish enrichment.

Usage (on chess-poc):
  AWS_PROFILE=default python label_features_see.py \
    --profiles d1024_k4_profiles.json --positions all_positions_labeled_opus.json \
    --enrichment position_enrichment_cache.json --cache chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
    --output feature_labels_see_d1024_k4.json --resume
"""
import argparse, json, time, boto3, chess
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"; REGION = "us-east-1"
MAX_CONCURRENT = 20; REQUEST_TIMEOUT = 120
client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=REQUEST_TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"times": [], "throttles": 0}

VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
PIECE = {chess.KNIGHT:'N', chess.BISHOP:'B', chess.ROOK:'R', chess.QUEEN:'Q', chess.PAWN:'P', chess.KING:'K'}
def see(bd, t, stm):
    aa = bd.attackers(stm, t)
    if not aa: return 0
    lva = min(aa, key=lambda s: VAL.get(bd.piece_type_at(s), 99)); cv = VAL.get(bd.piece_type_at(t), 0)
    b2 = bd.copy(); b2.remove_piece_at(t); b2.set_piece_at(t, bd.piece_at(lva)); b2.remove_piece_at(lva)
    return max(0, cv - see(b2, t, not stm))

def worst_own_hang(board, mover):
    """After a move (board = position after), the largest SEE-loss the mover leaves en prise."""
    opp = not mover; worst = 0; wp = None
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == mover and board.is_attacked_by(opp, sq):
            l = see(board, sq, opp)
            if l > worst: worst = l; wp = p.piece_type
    return worst, (PIECE.get(wp) if wp else None)

def see_both(fen, blunder_uci, best_uci):
    """SEE facts on BOTH moves. Returns dict of raw signals for one position."""
    try:
        b = chess.Board(fen); mover = b.turn
        bm = chess.Move.from_uci(blunder_uci)
        out = {'played': b.san(bm), 'played_is_capture': b.is_capture(bm)}
        # what an enemy piece is winnable in the ORIGINAL position (the thing you could've taken)
        best_win = 0
        if best_uci and len(best_uci) >= 4:
            bmv = chess.Move.from_uci(best_uci)
            out['best'] = b.san(bmv)
            out['best_is_capture'] = b.is_capture(bmv)
            out['best_is_check'] = b.gives_check(bmv)
            # material the best move wins: SEE of its target square (captures) or the enemy piece it wins
            if b.is_capture(bmv):
                tgt = bmv.to_square; cap = b.piece_at(tgt)
                capval = VAL.get(cap.piece_type, 1) if cap else 1  # en passant -> pawn
                # net after recapture: capval - SEE recapture on that square by opponent after best move
                bb = b.copy(); bb.push(bmv)
                best_win = max(0, capval - see(bb, tgt, not mover))
            else:
                # non-capture best (e.g. check that wins): approximate by largest enemy piece hanging after best
                bb = b.copy(); bb.push(bmv)
                w, _ = worst_own_hang(bb, not mover)  # enemy's hanging piece after our best move
                best_win = w
        out['best_wins_material'] = best_win
        # own piece left hanging AFTER the blunder
        bb = b.copy(); bb.push(bm)
        w, wp = worst_own_hang(bb, mover)
        out['blunder_hangs_own'] = w; out['blunder_hangs_piece'] = wp
        return out
    except Exception:
        return None

PROMPT = """You are an elite chess analyst labeling an SAE (Sparse Autoencoder) feature.

The feature fires on positions sharing ONE type of mistake. Below are its top activating
positions. For each you get: the move played, the engine's best move, that position's prior
analysis, and SEE (static-exchange-eval) numbers computed on BOTH moves as raw data.

READ THE SEE DATA. It disambiguates the mistake:
- best_wins_material > 0 + played_is_capture = false  => the player MISSED capturing/winning a
  hanging enemy piece (a "missed hanging piece" feature) — even if the best move is a check that
  then wins it.
- blunder_hangs_own > 0  => the player's OWN move left a piece en prise (a "hung piece" feature).
- These are OPPOSITE mistakes; do not confuse them. Look at which one is consistent across positions.

=== FEATURE: {n} positions ===
Moves played: {moves}
Phases: {phases} | Sides: {sides} | Avg cp_loss: {avg_cp:.0f}

=== SEE AGGREGATE (across {n} positions) ===
best move wins material (enemy piece winnable): {agg_best_win}/{n}   [median value won: {med_win}]
player's move was a capture: {agg_played_cap}/{n}
player's move left OWN piece hanging: {agg_own_hang}/{n}   [median value: {med_own}]
best move is a check: {agg_best_check}/{n} | best move is a capture: {agg_best_cap}/{n}

{positions_text}

=== INSTRUCTIONS ===
Identify the SINGLE shared mistake. Use the SEE aggregate to decide direction (missed-winning
vs hung-own). Name it concisely. Reference counts X/N from the SEE data in your description.

Respond in JSON:
{{
  "chip": "<2-4 word title, e.g. 'Missed Hanging Piece'>",
  "label": "<one sentence>",
  "description": "<paragraph citing SEE counts X/N>",
  "why_bad": "<why these moves fail>",
  "confidence": <0-100>
}}"""

def build_prompt(fid, examples, enrichment, analyses, meta_by_key):
    moves, phases, sides, cps = [], [], [], []
    see_rows = []; ptext = ""
    agg = Counter()
    win_vals = []; own_vals = []
    n = 0
    for i, ex in enumerate(examples):
        key = f"{ex['fen']}|{ex['uci']}"
        enr = enrichment.get(key, {})
        ana = analyses.get(key, {}).get("analysis", {}) if isinstance(analyses.get(key), dict) else {}
        sb = see_both(ex['fen'], ex['uci'], enr.get('best_uci') or _best_from_meta(key, meta_by_key))
        if sb is None: continue
        n += 1
        moves.append(sb['played']); phases.append(enr.get('phase','?')); sides.append(enr.get('side','?'))
        cps.append(enr.get('cp_loss',0) or 0)
        if sb.get('best_wins_material',0) > 0: agg['best_win'] += 1; win_vals.append(sb['best_wins_material'])
        if sb.get('played_is_capture'): agg['played_cap'] += 1
        if sb.get('blunder_hangs_own',0) > 0: agg['own_hang'] += 1; own_vals.append(sb['blunder_hangs_own'])
        if sb.get('best_is_check'): agg['best_check'] += 1
        if sb.get('best_is_capture'): agg['best_cap'] += 1
        seeline = (f"SEE: best={sb.get('best','?')} wins {sb.get('best_wins_material',0)} "
                   f"| played={sb['played']} {'(capture)' if sb['played_is_capture'] else '(quiet)'} "
                   f"| own piece hung after: {sb.get('blunder_hangs_own',0)}"
                   + (f" ({sb['blunder_hangs_piece']})" if sb.get('blunder_hangs_piece') else ""))
        feats = "\n".join(f"    - {f}" for f in enr.get("position_features", [])) or "    - (standard)"
        ptext += f"""
=== POSITION {i+1} ===
{seeline}
Move: {sb['played']} | Side: {enr.get('side','?')} | Phase: {enr.get('phase','?')} | cp_loss: {enr.get('cp_loss','?')}
Eval: {enr.get('eval_before','?')} -> {enr.get('eval_after','?')}
Position features:
{feats}
Analysis: {json.dumps(ana)[:600] if ana else '(none)'}
"""
    if n < 3: return None
    med = lambda L: sorted(L)[len(L)//2] if L else 0
    return PROMPT.format(n=n, moves=", ".join(moves),
        phases=", ".join(f"{p}({c})" for p,c in Counter(phases).most_common()),
        sides=", ".join(f"{s}({c})" for s,c in Counter(sides).most_common()),
        avg_cp=sum(cps)/len(cps),
        agg_best_win=agg['best_win'], med_win=med(win_vals),
        agg_played_cap=agg['played_cap'], agg_own_hang=agg['own_hang'], med_own=med(own_vals),
        agg_best_check=agg['best_check'], agg_best_cap=agg['best_cap'],
        positions_text=ptext)

_META = {}
def _best_from_meta(key, meta_by_key):
    return meta_by_key.get(key, {}).get('best_uci', '')

def parse_json(text):
    text = text.strip()
    for p in ["```json", "```"]:
        if text.startswith(p): text = text[len(p):]
    if text.endswith("```"): text = text[:-3]
    s, e = text.find("{"), text.rfind("}")+1
    if s >= 0 and e > s:
        try: return json.loads(text[s:e])
        except: return None
    return None

def invoke(prompt):
    body = json.dumps({"anthropic_version":"bedrock-2023-05-31","max_tokens":8000,
        "thinking":{"type":"enabled","budget_tokens":4096},"messages":[{"role":"user","content":prompt}]})
    r = client.invoke_model(modelId=MODEL_ID, body=body)
    res = json.loads(r["body"].read())
    for blk in res["content"]:
        if blk.get("type") == "text": return blk["text"]
    return res["content"][-1].get("text","")

def process(item):
    fid, prompt = item; t0 = time.time()
    for att in range(3):
        try:
            raw = invoke(prompt); p = parse_json(raw)
            return (fid, {"analysis": p, "time_s": round(time.time()-t0,1)} if p else {"error":"parse"})
        except (ReadTimeoutError, ClientError) as e:
            if isinstance(e, ClientError) and e.response["Error"]["Code"] != "ThrottlingException":
                return (fid, {"error": str(e)[:150]})
            stats["throttles"] += 1; time.sleep(2**(att+1))
    return (fid, {"error":"retries"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True); ap.add_argument("--positions", required=True)
    ap.add_argument("--enrichment", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    profiles = json.load(open(a.profiles)); analyses = json.load(open(a.positions))
    enrichment = json.load(open(a.enrichment))
    import torch
    meta = torch.load(a.cache, map_location='cpu', weights_only=False)['metadata']
    meta_by_key = {m['fen']+'|'+m['blunder_uci']: m for m in meta}
    results = {}
    if a.resume:
        try:
            results = json.load(open(a.output))
            print(f"Resuming: {sum(1 for v in results.values() if 'error' not in v)} done")
        except (FileNotFoundError, json.JSONDecodeError): pass
    work = []; skipped = 0
    for fid in sorted(profiles.keys(), key=int):
        if fid in results and "error" not in results[fid]: continue
        prompt = build_prompt(fid, profiles[fid].get("examples", [])[:15], enrichment, analyses, meta_by_key)
        if prompt is None: results[fid] = {"error":"insufficient"}; skipped += 1; continue
        work.append((fid, prompt))
    print(f"To label: {len(work)} (skipped {skipped}) | Opus 4.6 | conc={MAX_CONCURRENT}", flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futs = {pool.submit(process, it): it for it in work}
        for fu in as_completed(futs):
            fid, res = fu.result(); results[fid] = res; done += 1
            if "time_s" in res: stats["times"].append(res["time_s"])
            if done % 25 == 0:
                json.dump(results, open(a.output,"w"))
                at = sum(stats["times"])/max(1,len(stats["times"]))
                eta = (len(work)-done)*at/MAX_CONCURRENT/60
                print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | avg {at:.0f}s | ETA {eta:.0f}min", flush=True)
    json.dump(results, open(a.output,"w"), indent=1)
    print(f"Done. {sum(1 for v in results.values() if 'error' not in v)} labeled -> {a.output}", flush=True)

if __name__ == "__main__":
    main()
