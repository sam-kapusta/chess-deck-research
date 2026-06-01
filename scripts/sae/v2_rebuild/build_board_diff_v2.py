"""V2 board_diff cache: mean64(h_after_best - h_after_blunder). ONNX L7, 512-dim."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import onnxruntime as ort, numpy as np, torch, chess, time, json, os
BASE='/home/ec2-user/SageMaker/chess-stage-a'
OUT=BASE+'/cache/maia3_board_diff_v2.pt'
PROBE='/model/transformer/layers.7/Add_2_output_0'
PIECES=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
so=ort.SessionOptions(); so.intra_op_num_threads=0
sess=ort.InferenceSession('/home/ec2-user/SageMaker/maia3_with_probe.onnx',so,providers=['CPUExecutionProvider'])
src=json.load(open(BASE+'/cache/real_game_blunder_positions.json'))
best=json.load(open('/home/ec2-user/SageMaker/maia_best_200k.json'))
def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t
def elos(p):
    es=float(p['white_elo'] if p['is_white'] else p['black_elo']); eo=float(p['black_elo'] if p['is_white'] else p['white_elo']); return es,eo
def enc(toks,es,eo): return sess.run([PROBE],{'tokens':np.stack(toks),'elo_self':np.array(es,np.float32),'elo_oppo':np.array(eo,np.float32)})[0]
valid=[]
for p in src:
    k=p['fen']+'|'+p['blunder_uci']; bu=best.get(k)
    if bu and bu!=p['blunder_uci']: valid.append((p,bu))
print(f'valid {len(valid)}',flush=True)
BS=128; diffs=[]; metas=[]; nerr=0; t0=time.time()
for i in range(0,len(valid),BS):
    chunk=valid[i:i+BS]; bl_t=[]; bs_t=[]; es=[]; eo=[]; info=[]
    for p,bu in chunk:
        try:
            b=chess.Board(p['fen']); bb=b.copy(); bb.push(chess.Move.from_uci(p['blunder_uci']))
            sb=b.copy(); sb.push(chess.Move.from_uci(bu))
            bl_t.append(tok(bb)); bs_t.append(tok(sb)); e=elos(p); es.append(e[0]); eo.append(e[1]); info.append((p,bu))
        except: nerr+=1
    if not bl_t: continue
    Hbl=enc(bl_t,es,eo); Hbs=enc(bs_t,es,eo)
    for j,(p,bu) in enumerate(info):
        d=(Hbs[j]-Hbl[j]).mean(0)
        if np.linalg.norm(d)>1e-6:
            diffs.append(d.astype(np.float32)); metas.append({'fen':p['fen'],'blunder_uci':p['blunder_uci'],'best_uci':bu,'cp_loss':p['cp_loss'],'is_white':p['is_white']})
    if (i//BS)%20==0 or i+BS>=len(valid):
        el=time.time()-t0; r=(i+len(chunk))/max(el,1); print(f'{i+len(chunk)}/{len(valid)} ({r:.0f}/s ETA {(len(valid)-i)/max(r,1)/60:.0f}min)',flush=True)
A=np.stack(diffs).astype(np.float32)
torch.save({'activations':torch.tensor(A),'mean':A.mean(0),'std':A.std(0),'metadata':metas,
    'config':{'construction':'mean64(h_after_best-h_after_blunder)','source':'v2+maia_best@2600','n':len(diffs)}},OUT)
print(f'SAVED {OUT} ({os.path.getsize(OUT)//1024//1024}MB) n={len(diffs)} err={nerr}',flush=True)
