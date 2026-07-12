#!/usr/bin/env python3
"""Full Stockfish (depth 16, matching the prod worker) over the ~60k l7-cache positions that have an
Opus analysis. Per position computes the two lines the tagger needs:
  pv_uci         = best line from fen_before (the "missed" line)   [6 plies]
  refutation_uci = best line from the position AFTER the blunder   [6 plies]
Also carries best_uci/bestMoveSan + eval_before/eval_after (cp, white-POV) so the tagger's win_drop
gate + Missed/Allowed/Failed motif detectors all fire at full strength.

Parallel: N worker processes, each its own single-thread Stockfish. Output JSONL keyed "fen|blunder".
Resumable (skips keys already in the output). ~5 min on 48 workers.

Run: python3 sf_batch_60k.py --workers 48 --depth 16 --out /home/ec2-user/SageMaker/jr_canon_out/sf_lines_60k.jsonl
"""
import argparse, json, os, time
import chess, chess.engine, torch
from multiprocessing import Process, Queue

SF = "/home/ec2-user/SageMaker/stockfish_compiled"
CACHE = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt"
OPUS = "/home/ec2-user/SageMaker/all_positions_labeled_opus.json"


def cp_white(score, board_turn):
    """POV score -> white-POV centipawns (mate = ±10000)."""
    v = score.white()
    if v.is_mate():
        return 10000 if v.mate() > 0 else -10000
    return v.score()


def worker(wid, depth, jobs, out_q):
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 64})
    lim = chess.engine.Limit(depth=depth)
    for key, fen, blunder in jobs:
        rec = {"key": key}
        try:
            b = chess.Board(fen)
            info = eng.analyse(b, lim, multipv=1)
            info = info[0] if isinstance(info, list) else info
            pv = info["pv"]
            rec["pv_uci"] = [m.uci() for m in pv[:6]]
            rec["best_uci"] = pv[0].uci() if pv else ""
            rec["bestMoveSan"] = b.san(pv[0]) if pv else ""
            rec["eval_before"] = cp_white(info["score"], b.turn)
            # refutation: after the blunder
            bm = chess.Move.from_uci(blunder)
            rec["san"] = b.san(bm) if bm in b.legal_moves else blunder
            b.push(bm)
            info2 = eng.analyse(b, lim, multipv=1)
            info2 = info2[0] if isinstance(info2, list) else info2
            rec["refutation_uci"] = [m.uci() for m in info2["pv"][:6]]
            rec["eval_after"] = cp_white(info2["score"], b.turn)
        except Exception as e:
            rec["_error"] = str(e)[:120]
        out_q.put(rec)
    eng.quit()
    out_q.put(("_DONE", wid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap positions (smoke test)")
    args = ap.parse_args()

    opus = set(json.load(open(OPUS)).keys())
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    meta = c["metadata"]
    # only positions WITH an Opus analysis (the labeling universe)
    todo = []
    for m in meta:
        k = f"{m['fen']}|{m['blunder_uci']}"
        if k in opus:
            todo.append((k, m["fen"], m["blunder_uci"]))
    print(f"{len(todo)} analyzed positions to Stockfish", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if line:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
    todo = [t for t in todo if t[0] not in done]
    if args.limit: todo = todo[:args.limit]
    print(f"{len(todo)} to do ({len(done)} already done)", flush=True)
    if not todo:
        print("nothing to do", flush=True); return

    # shard round-robin across workers
    W = args.workers
    shards = [[] for _ in range(W)]
    for i, t in enumerate(todo):
        shards[i % W].append(t)
    out_q = Queue(maxsize=10000)
    procs = [Process(target=worker, args=(w, args.depth, shards[w], out_q)) for w in range(W)]
    for p in procs: p.start()

    t0 = time.time(); n = 0; finished = 0
    with open(args.out, "a") as f:
        while finished < W:
            item = out_q.get()
            if isinstance(item, tuple) and item[0] == "_DONE":
                finished += 1; continue
            f.write(json.dumps(item) + "\n"); n += 1
            if n % 2000 == 0:
                f.flush()
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.0f}/s  eta {(len(todo)-n)/rate/60:.1f}min", flush=True)
    for p in procs: p.join()
    print(f"DONE {n} positions in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
