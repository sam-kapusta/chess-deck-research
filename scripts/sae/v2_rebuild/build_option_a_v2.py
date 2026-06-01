"""V2 Option-A diff cache: h[best_to_sq] - h[blunder_to_sq] on before-board.
Source: real_game_blunder_positions.json (v2, corrected) + maia_best_200k.json (Maia@2600 best).
elo from white_elo/black_elo/is_white. ONNX layer-7 probe, 512-dim."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import onnxruntime as ort, numpy as np, torch, chess, time, json, os
BASE='/home/ec2-user/SageMaker/chess-stage-a'
OUT=BASE+'/cache/maia3_option_a_diff_v2.pt'
PROBE='/model/transformer/layers.7/Add_2_output_0'
PIECES=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]

so=ort.SessionOptions(); so.intra_op_num_threads=0
sess=ort.InferenceSession('/home/ec2-user/SageMaker/maia3_with_probe.onnx',so,providers=['CPUExecutionProvider'])
src=json.load(open(BASE+'/cache/real_game_blunder_positions.json'))
best=json.load(open('/home/ec2-user/SageMaker/maia_best_200k.json'))
print(f'{len(src)} positions, {len(best)} best moves',flush=True)

def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t
def elos(p):
    es=float(p['white_elo'] if p['is_white'] else p['black_elo'])
    eo=float(p['black_elo'] if p['is_white'] else p['white_elo'])
    return es,eo

BS=256; diffs=[]; metas=[]; nerr=0; ndegen=0; t0=time.time()
# valid = has best move that differs from blunder
valid=[]
for p in src:
    k=p['fen']+'|'+p['blunder_uci']; bu=best.get(k)
    if bu is None: continue
    if bu==p['blunder_uci']: ndegen+=1; continue  # drop degenerate
    valid.append((p,bu))
print(f'valid (non-degenerate): {len(valid)}, dropped degenerate: {ndegen}',flush=True)

for i in range(0,len(valid),BS):
    chunk=valid[i:i+BS]; toks=[]; es_l=[]; eo_l=[]; info=[]
    for p,bu in chunk:
        try:
            b=chess.Board(p['fen'])
            bl=chess.Move.from_uci(p['blunder_uci']); bs=chess.Move.from_uci(bu)
            toks.append(tok(b)); e=elos(p); es_l.append(e[0]); eo_l.append(e[1])
            info.append((p,bu,bl.to_square,bs.to_square))
        except: nerr+=1
    if not toks: continue
    H=sess.run([PROBE],{'tokens':np.stack(toks),'elo_self':np.array(es_l,np.float32),'elo_oppo':np.array(eo_l,np.float32)})[0]
    for j,(p,bu,bl_to,bs_to) in enumerate(info):
        d=H[j,bs_to]-H[j,bl_to]
        if np.linalg.norm(d)>1e-6:
            diffs.append(d.astype(np.float32))
            metas.append({'fen':p['fen'],'blunder_uci':p['blunder_uci'],'best_uci':bu,'cp_loss':p['cp_loss'],'is_white':p['is_white']})
    if (i//BS)%20==0 or i+BS>=len(valid):
        el=time.time()-t0; r=(i+len(chunk))/max(el,1)
        print(f'{i+len(chunk)}/{len(valid)} ({r:.0f}/s, ETA {(len(valid)-i)/max(r,1)/60:.0f}min)',flush=True)

A=np.stack(diffs).astype(np.float32)
torch.save({'activations':torch.tensor(A),'mean':A.mean(0),'std':A.std(0),'metadata':metas,
    'config':{'construction':'h[best_to]-h[blunder_to] before-board','source':'v2+maia_best@2600','n':len(diffs)}},OUT)
print(f'SAVED {OUT} ({os.path.getsize(OUT)//1024//1024}MB) n={len(diffs)} err={nerr}',flush=True)
