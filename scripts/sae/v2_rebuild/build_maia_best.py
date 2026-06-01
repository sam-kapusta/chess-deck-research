"""Build Maia-best@2600 best_uci for all 200k v2 positions.
Batched ONNX (CPU). Resumable. Output: maia_best_200k.json {fen|blunder_uci: best_uci}.

Construction note: best = Maia3 policy argmax at elo_self=elo_oppo=2600, masked to legal moves,
using the maia3 package's own tokenization (mirrors board for black) + 4352-move vocab +
mirror_move on the chosen move for black-to-move. Validated: start->e2e4, hanging Q->capture,
M1->mate; 50% agreement with Stockfish on the 19k overlap (the divergence is Maia's
human-best vs engine-best, which is intended)."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import onnxruntime as ort, numpy as np, json, chess, time, os
from maia3.dataset import tokenize_board, get_legal_moves_mask
from maia3.utils import get_all_possible_moves, mirror_move

BASE='/home/ec2-user/SageMaker/chess-stage-a'
OUT='/home/ec2-user/SageMaker/maia_best_200k.json'
ELO=2600.; BS=256

so=ort.SessionOptions(); so.intra_op_num_threads=0  # use all cores
sess=ort.InferenceSession('/home/ec2-user/SageMaker/maia3_with_probe.onnx',so,providers=['CPUExecutionProvider'])
ALL=get_all_possible_moves(); ALLD={m:i for i,m in enumerate(ALL)}
assert len(ALL)==4352

src=json.load(open(BASE+'/cache/real_game_blunder_positions.json'))
print(f'{len(src)} positions',flush=True)

best={}
if os.path.exists(OUT):
    best=json.load(open(OUT)); print(f'resumed {len(best)}',flush=True)

def key(p): return p['fen']+'|'+p['blunder_uci']
todo=[p for p in src if key(p) not in best]
print(f'todo {len(todo)}',flush=True)

t0=time.time()
for i in range(0,len(todo),BS):
    chunk=todo[i:i+BS]
    toks=[]; boards=[]
    for p in chunk:
        b=chess.Board(p['fen']); boards.append(b)
        toks.append(tokenize_board(b).numpy().astype(np.float32))
    T=np.stack(toks)
    elos=np.full(len(chunk),ELO,np.float32)
    out=sess.run(['logits_move'],{'tokens':T,'elo_self':elos,'elo_oppo':elos})[0]
    for j,(p,b) in enumerate(zip(chunk,boards)):
        mask=get_legal_moves_mask(b,ALLD).numpy().astype(bool)
        lg=out[j].copy(); lg[~mask]=-1e9
        mv=ALL[int(lg.argmax())]
        if b.turn==chess.BLACK: mv=mirror_move(mv)
        best[key(p)]=mv
    done=len(best)
    if (i//BS)%20==0 or i+BS>=len(todo):
        el=time.time()-t0; rate=(i+len(chunk))/max(el,1); eta=(len(todo)-i-len(chunk))/max(rate,1)
        print(f'{done}/{len(src)} ({rate:.0f}/s, ETA {eta/60:.0f}min)',flush=True)
        json.dump(best,open(OUT,'w'))

json.dump(best,open(OUT,'w'))
# stats
ndegen=sum(1 for p in src if best.get(key(p))==p['blunder_uci'])
print(f'DONE {len(best)} | degenerate(==blunder): {ndegen} ({100*ndegen/len(best):.1f}%)',flush=True)
