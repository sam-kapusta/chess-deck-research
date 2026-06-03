"""Generic Stockfish-enrichment of a gap key-list (FEN|move), merging into the shared
position_enrichment_cache.json. Reuses the proven enrich_position() from
enrich_all_positions.py. Resumable, multi-worker, depth from that module.
Usage: python enrich_gap.py --gap union_gap_top10_needenr.json [--workers 48]"""
import json,time,threading,chess,chess.engine,argparse,importlib.util
B='/home/ec2-user/SageMaker'
ap=argparse.ArgumentParser();ap.add_argument('--gap',required=True);ap.add_argument('--workers',type=int,default=48)
ap.add_argument('--cache',default=B+'/position_enrichment_cache.json');a=ap.parse_args()
spec=importlib.util.spec_from_file_location("enr",B+"/enrich_all_positions.py")
enr=importlib.util.module_from_spec(spec);spec.loader.exec_module(enr)
STOCKFISH=enr.STOCKFISH_PATH;DEPTH=enr.DEPTH
print('stockfish=',STOCKFISH,'depth=',DEPTH,flush=True)
gap=json.load(open(a.gap));cache=json.load(open(a.cache))
todo=[k for k in gap if k not in cache]
print(f'gap={len(gap)} cached={len(gap)-len(todo)} TODO={len(todo)}',flush=True)
if not todo: print('nothing to do');raise SystemExit
work=[(k,)+tuple(k.rsplit('|',1)) for k in todo]
lock=threading.Lock();done={'n':0};t0=time.time();W=a.workers
def chunk(l,n):s=(len(l)+n-1)//n;return[l[i:i+s] for i in range(0,len(l),s)]
def worker(items):
    eng=chess.engine.SimpleEngine.popen_uci(STOCKFISH);res={}
    for key,fen,uci in items:
        try:res[key]=enr.enrich_position(eng,fen,uci)
        except Exception as e:res[key]={'error':str(e)[:200]}
        with lock:
            done['n']+=1;n=done['n']
            if n%200==0:
                el=time.time()-t0;r=n/max(el,1)
                print(f'  {n}/{len(todo)} | {el/60:.1f}min | {r:.0f}/s | ETA {(len(todo)-n)/max(r,1)/60:.0f}min',flush=True)
    eng.quit();return res
allres={}
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=W) as pool:
    for fut in [pool.submit(worker,p) for p in chunk(work,W)]:
        allres.update(fut.result())
cache.update(allres);json.dump(cache,open(a.cache,'w'))
print(f'done. enriched {len(allres)}, cache now {len(cache)}',flush=True)
