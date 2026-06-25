"""Unified pull (fixes the num/denom sample-mismatch bug): per band, scan rapid eval-annotated games
(mover-banded) until MOVE_TARGET mover-moves, counting BOTH total moves (+ endgame moves) AND keeping
ALL >=200cp blunders found in the SAME moves. rate = blunders_in_scan / moves_in_scan — one scan.
Numerator and denominator now share the exact same game population. Outputs:
  fifa_blunders_all.json {band:[blunder...]}   band_denominators.json {moves, endmoves}."""
import os, re, io, json, time
assert os.environ.get("HF_TOKEN")
import chess, chess.pgn
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
REPO="Lichess/standard-chess-games"
BANDS=[('600-800',600,800),('800-1000',800,1000),('1000-1200',1000,1200),('1200-1400',1200,1400),
       ('1400-1600',1400,1600),('1600-1800',1600,1800),('1800-2000',1800,2000),('2000-2200',2000,2200),
       ('2200-2400',2200,2400),('2400-2600',2400,2600),('2600-2800',2600,2800)]
MOVE_TARGET=60000; MIN_LOSS=200; MAX_SHARDS=260
def is_rapid(tc):
    m=re.match(r'(\d+)\+(\d+)',str(tc or ''))
    if not m: return False
    b,i=int(m.group(1)),int(m.group(2)); return 480<=b+40*i<1500
def band_of(e):
    for n,lo,hi in BANDS:
        if lo<=e<hi: return n
    return None
def is_endgame(board):
    pm=board.piece_map()
    if len(pm)<=12: return True
    return sum(1 for p in pm.values() if p.piece_type not in (chess.PAWN,chess.KING))<=4
def pe(c):
    m=re.search(r'\[%eval\s+([#\-\d.]+)\]',c or '')
    if not m: return None
    s=m.group(1)
    if '#' in s: return -10000 if '-' in s else 10000
    try: return int(round(float(s)*100))
    except: return None

files=[f for f in list_repo_files(REPO,repo_type="dataset") if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$',f)]
files.sort(reverse=True)
moves={b:0 for b,_,_ in BANDS}; endmoves={b:0 for b,_,_ in BANDS}; blunders={b:[] for b,_,_ in BANDS}
t0=time.time(); ng=0
def done(): return all(moves[b]>=MOVE_TARGET for b,_,_ in BANDS)

for fi,f in enumerate(files):
    if fi>=MAX_SHARDS or done(): break
    try:
        p=hf_hub_download(REPO,f,repo_type="dataset")
        t=pq.read_table(p,columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e: print('skip',str(e)[:60],flush=True); continue
    for we,be,tc,mt in zip(*[t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext']]):
        ng+=1
        if not we or not be or not is_rapid(tc) or '%eval' not in mt: continue
        wb,bb=band_of(we),band_of(be)
        if (wb is None or moves[wb]>=MOVE_TARGET) and (bb is None or moves[bb]>=MOVE_TARGET): continue
        try: g=chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{mt}'))
        except Exception: continue
        if not g: continue
        board=g.board(); prev_eval=None; ply=0
        for node in g.mainline():
            mover_white=(board.turn==chess.WHITE); me=we if mover_white else be; bn=band_of(me)
            cur_eval=pe(node.comment)   # eval AFTER this move (from the move's own comment)
            if bn and moves[bn]<MOVE_TARGET:
                moves[bn]+=1
                if is_endgame(board): endmoves[bn]+=1
                if prev_eval is not None and cur_eval is not None:
                    loss=(prev_eval-cur_eval) if mover_white else (cur_eval-prev_eval)
                    if loss>=MIN_LOSS:
                        blunders[bn].append({'fen':board.fen(),'blunder_uci':node.move.uci(),'cp_loss':loss,
                            'eval_before':prev_eval,'eval_after':cur_eval,'ply':ply,'is_white':mover_white,
                            'white_elo':we,'black_elo':be,'band':bn})
            prev_eval=cur_eval; board.push(node.move); ply+=1
    try: os.remove(p)
    except: pass
    if fi%5==0:
        print(f'shard {fi+1} | {ng}g {time.time()-t0:.0f}s | '+' '.join(f'{b.split("-")[0]}:{moves[b]}m/{len(blunders[b])}b' for b,_,_ in BANDS),flush=True)
        json.dump({b:blunders[b] for b,_,_ in BANDS},open('fifa_blunders_all.json','w'))
        json.dump({'moves':moves,'endmoves':endmoves},open('band_denominators.json','w'))
json.dump({b:blunders[b] for b,_,_ in BANDS},open('fifa_blunders_all.json','w'))
json.dump({'moves':moves,'endmoves':endmoves},open('band_denominators.json','w'))
print('=== DONE ===',f'{(time.time()-t0)/60:.1f}min',flush=True)
for b,_,_ in BANDS:
    bl=len(blunders[b]); print(f'  {b}: {moves[b]} moves, {bl} blunders ({100*bl/max(moves[b],1):.1f}%), {endmoves[b]} endmoves')
