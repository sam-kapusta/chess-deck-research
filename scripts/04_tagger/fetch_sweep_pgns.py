#!/usr/bin/env python3
"""Phase 1 of the depth-16 sweep re-detection: fetch + CACHE the raw PGNs.

The original sweep (sweep_blunders_2000.json) selected blunders from Lichess's own [%eval]
annotations (variable depth ~20+), and never cached the raw games — so re-detecting at OUR
Stockfish depth requires re-fetching. This script fetches the exact same distinct game IDs
(holding the game set FIXED so the only thing that changes is the detection engine/depth) via
the Lichess bulk-export API and caches them to disk, keyed by full game_id URL.

Idempotent/resumable: skips IDs already in the cache. Run once; redetect_sweep_d16.py reads it.

Output: /home/ec2-user/SageMaker/sweep_pgns_cache.json  {game_id_url: pgn_text}
"""
import json, io, time, urllib.request

SWEEP = "/home/ec2-user/SageMaker/sweep_blunders_2000.json"
OUT   = "/home/ec2-user/SageMaker/sweep_pgns_cache.json"
BATCH = 200
# moves + tags (need WhiteElo/BlackElo/Result), no evals/clocks/opening — smaller payload, and we
# compute our own evals anyway.
URL = "https://lichess.org/api/games/export/_ids?moves=true&tags=true&clocks=false&evals=false&opening=false"

def short_id(game_id):
    return game_id.rstrip("/").split("/")[-1]

def main():
    rows = json.load(open(SWEEP))
    all_ids = sorted({r["game_id"] for r in rows})  # full URLs; sorted for deterministic order
    print(f"{len(all_ids)} distinct games to fetch", flush=True)

    cache = {}
    try:
        cache = json.load(open(OUT))
        print(f"resuming: {len(cache)} already cached", flush=True)
    except FileNotFoundError:
        pass

    todo = [g for g in all_ids if g not in cache]
    print(f"to fetch: {len(todo)}", flush=True)
    short_to_full = {short_id(g): g for g in todo}

    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i+BATCH]
        ids = ",".join(short_id(g) for g in batch)
        try:
            req = urllib.request.Request(URL, data=ids.encode(),
                headers={"User-Agent": "chess-deck-research d16-redetect",
                         "Content-Type": "text/plain"})
            txt = urllib.request.urlopen(req, timeout=90).read().decode()
        except Exception as e:
            print(f"  batch {i} err: {e} — retry in 10s", flush=True); time.sleep(10); continue

        # split the concatenated PGN stream into individual games by the [Event tag boundary
        import chess.pgn
        stream = io.StringIO(txt)
        n_in_batch = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            site = game.headers.get("Site", "").rstrip("/")
            sid = short_id(site)
            full = short_to_full.get(sid) or site
            cache[full] = str(game)
            n_in_batch += 1

        # checkpoint every batch (atomic write)
        tmp = OUT + ".new"
        json.dump(cache, open(tmp, "w"))
        import os; os.replace(tmp, OUT)
        el = (time.time() - t0) / 60
        print(f"  {min(i+BATCH, len(todo))}/{len(todo)} requested | +{n_in_batch} | cache {len(cache)} | {el:.1f}min", flush=True)
        time.sleep(1.0)  # politeness

    print(f"\nDONE {len(cache)} games cached -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
