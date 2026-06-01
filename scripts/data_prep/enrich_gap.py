"""Stockfish-enrich the k=16 v2 profile gap positions.
Reuses enrich_position() from enrich_all_positions.py."""
import json, sys, importlib.util, threading, time, chess, chess.engine
from concurrent.futures import ThreadPoolExecutor

B = "/home/ec2-user/SageMaker"
spec = importlib.util.spec_from_file_location("enr", B + "/enrich_all_positions.py")
enr = importlib.util.module_from_spec(spec); spec.loader.exec_module(enr)
print("Stockfish:", enr.STOCKFISH_PATH, "depth:", enr.DEPTH, flush=True)

CACHE = B + "/position_enrichment_cache.json"
gap = json.load(open(B + "/l7only_enrich_gap_k16v2.json"))
cache = json.load(open(CACHE))
todo = [(k, k.rsplit("|", 1)[0], k.rsplit("|", 1)[1]) for k in gap if k not in cache]
print(f"gap={len(gap)} | already cached={len(gap)-len(todo)} | todo={len(todo)}", flush=True)
if not todo:
    print("Nothing to enrich."); sys.exit(0)

lock = threading.Lock(); done = {"n": 0}; t0 = time.time()

def worker(items):
    eng = chess.engine.SimpleEngine.popen_uci(enr.STOCKFISH_PATH)
    res = {}
    for key, fen, uci in items:
        try: res[key] = enr.enrich_position(eng, fen, uci)
        except Exception as e: res[key] = {"error": str(e)[:200]}
        with lock:
            done["n"] += 1
            if done["n"] % 200 == 0:
                el = time.time() - t0; r = done["n"] / max(el, 1)
                print(f"  {done['n']}/{len(todo)} | {el/60:.1f}min | ETA {(len(todo)-done['n'])/max(r,1)/60:.0f}min", flush=True)
    eng.quit(); return res

WORKERS = 48
sz = (len(todo) + WORKERS - 1) // WORKERS
parts = [todo[i:i+sz] for i in range(0, len(todo), sz)]
allres = {}
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for r in pool.map(worker, parts): allres.update(r)

cache.update(allres)
json.dump(cache, open(CACHE, "w"))
ok = sum(1 for k in todo if "error" not in allres.get(k, {"error": 1}))
print(f"DONE enriched {ok}/{len(todo)} | cache now {len(cache)}", flush=True)
