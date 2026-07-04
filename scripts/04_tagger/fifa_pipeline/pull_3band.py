"""3-band rapid-only proof pull (mover-Elo banding, matches original redetect_sweep_d16 get_band).
A blunder is bucketed by the MOVER's Elo, not requiring both players in-band. Rapid only, >=200cp."""
import os, re, json, io, time
# HF_TOKEN required (unauthenticated streaming gets rate-limited and crashes). Set it in the env
# before running: export HF_TOKEN=hf_...  (a working token lives in ~/SageMaker/hf_download.py on chess-poc).
assert os.environ.get("HF_TOKEN"), "set HF_TOKEN env var (see ~/SageMaker/hf_download.py on chess-poc)"
import chess, chess.pgn
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Lichess/standard-chess-games"
BANDS = [('800-1000',800,1000), ('1600-1800',1600,1800), ('2600-2800',2600,2800)]
TARGET = 2500; MIN_LOSS = 200

def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m: return False
    base, inc = int(m.group(1)), int(m.group(2))
    return 480 <= base + 40*inc < 1500

def band_of(elo):
    for name,lo,hi in BANDS:
        if lo<=elo<hi: return name
    return None

def parse_eval(c):
    m = re.search(r'\[%eval\s+([#\-\d.]+)\]', c or '')
    if not m: return None
    s=m.group(1)
    if '#' in s: return (-10000 if '-' in s else 10000)
    try: return int(round(float(s)*100))
    except: return None

def blunders_banded(mt, we, be):
    """Yield (band, blunder) for each >=200cp blunder, banded by the MOVER's Elo."""
    if '%eval' not in mt: return
    g = chess.pgn.read_game(io.StringIO(f'[Event "?"]\n[Result "*"]\n\n{mt}'))
    if not g: return
    board=g.board(); prev=None; ply=0
    for node in g.mainline():
        cur=parse_eval(node.comment)
        if prev is not None and cur is not None:
            white=(ply%2==0); mover_elo = we if white else be
            band = band_of(mover_elo)
            if band:
                loss=(prev-cur) if white else (cur-prev)
                if loss>=MIN_LOSS:
                    yield band, {'fen':board.fen(),'blunder_uci':node.move.uci(),'cp_loss':loss,
                                 'eval_before':prev,'eval_after':cur,'ply':ply,'is_white':white,
                                 'white_elo':we,'black_elo':be,'band':band}
        board.push(node.move); prev=cur; ply+=1

files=[f for f in list_repo_files(REPO,repo_type="dataset")
       if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$', f)]
files.sort(reverse=True)
print(f'{len(files)} shards', flush=True)

pos={b:[] for b,_,_ in BANDS}; band_moves={b:0 for b,_,_ in BANDS}  # rapid in-band mover-moves seen
t0=time.time(); ng=0
for fi,f in enumerate(files):
    if all(len(pos[b])>=TARGET for b,_,_ in BANDS): break
    try:
        path=hf_hub_download(REPO,f,repo_type="dataset")
        t=pq.read_table(path,columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e:
        print(f'  shard {f} FAILED: {e}',flush=True); continue
    we_,be_,tc_,mt_=(t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext'])
    for we,be,tc,mt in zip(we_,be_,tc_,mt_):
        ng+=1
        if not we or not be or not is_rapid(tc): continue
        # only parse games where at least one player is in a target band (cheap pre-filter)
        if band_of(we) is None and band_of(be) is None: continue
        for band,bl in blunders_banded(mt or '', we, be):
            if len(pos[band])<TARGET: pos[band].append(bl)
    try: os.remove(path)
    except: pass
    print(f'shard {fi+1}/{len(files)} | {ng} games {time.time()-t0:.0f}s | ' +
          ' '.join(f'{b}:{len(pos[b])}' for b,_,_ in BANDS), flush=True)

print('=== DONE ===',ng,'games',f'{time.time()-t0:.0f}s',flush=True)
for b,_,_ in BANDS: print(f'  {b}: {len(pos[b])}')
json.dump(pos,open('proof_3band.json','w'))
print('wrote proof_3band.json')

