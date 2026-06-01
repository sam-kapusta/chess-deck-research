"""V2 L2+L7 concat cache: concat(mean64(L2 best-blunder), mean64(L7 best-blunder)), 2048-dim.
79M PyTorch Maia3 on GPU, batched. l7only = second 1024 half (sliced at train time)."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import torch, numpy as np, chess, time, json, os
from types import SimpleNamespace
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board, get_historical_tokens

BASE='/home/ec2-user/SageMaker/chess-stage-a'
OUT=BASE+'/cache/maia3_l2l7_concat_v2.pt'
dev='cuda' if torch.cuda.is_available() else 'cpu'
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,
    num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,
    use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',
    dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,gab_per_square_dim=32,gab_intermediate_dim=128)
ckpt=torch.load('/home/ec2-user/SageMaker/maia3_79m_fixed.pt',map_location='cpu',weights_only=False)
model=MAIA3Model(cfg); model.load_state_dict(ckpt); model.eval().to(dev)
tcfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)
print('79M loaded',dev,flush=True)

src=json.load(open(BASE+'/cache/real_game_blunder_positions.json'))
best=json.load(open('/home/ec2-user/SageMaker/maia_best_200k.json'))
def elos(p):
    es=float(p['white_elo'] if p['is_white'] else p['black_elo']); eo=float(p['black_elo'] if p['is_white'] else p['white_elo']); return es,eo
def histtok(b):
    t=tokenize_board(b)
    return get_historical_tokens([t]*8,tcfg,0,0,0,0)  # [64, ...]
valid=[]
for p in src:
    k=p['fen']+'|'+p['blunder_uci']; bu=best.get(k)
    if bu and bu!=p['blunder_uci']: valid.append((p,bu))
print(f'valid {len(valid)}',flush=True)

# hooks on layers[1] (L2) and layers[6] (L7)
buf={}
def mk(name):
    def h(m,i,o): buf[name]=o.detach()
    return h
h2=model.transformer.layers[1].register_forward_hook(mk('L2'))
h7=model.transformer.layers[6].register_forward_hook(mk('L7'))

def encode(boards,es,eo):
    T=torch.stack([histtok(b) for b in boards]).to(dev)
    esT=torch.tensor(es,dtype=torch.float32,device=dev); eoT=torch.tensor(eo,dtype=torch.float32,device=dev)
    with torch.no_grad(): model(T,esT,eoT)
    return buf['L2'].float().cpu().numpy(), buf['L7'].float().cpu().numpy()  # [B,64,1024] each

BS=64; diffs=[]; metas=[]; nerr=0; t0=time.time()
for i in range(0,len(valid),BS):
    chunk=valid[i:i+BS]; blb=[]; bsb=[]; es=[]; eo=[]; info=[]
    for p,bu in chunk:
        try:
            b=chess.Board(p['fen']); bb=b.copy(); bb.push(chess.Move.from_uci(p['blunder_uci']))
            sb=b.copy(); sb.push(chess.Move.from_uci(bu)); blb.append(bb); bsb.append(sb)
            e=elos(p); es.append(e[0]); eo.append(e[1]); info.append((p,bu))
        except: nerr+=1
    if not blb: continue
    L2bl,L7bl=encode(blb,es,eo); L2bs,L7bs=encode(bsb,es,eo)
    for j,(p,bu) in enumerate(info):
        d2=(L2bs[j]-L2bl[j]).mean(0); d7=(L7bs[j]-L7bl[j]).mean(0)
        d=np.concatenate([d2,d7])
        if np.linalg.norm(d)>1e-6:
            diffs.append(d.astype(np.float32)); metas.append({'fen':p['fen'],'blunder_uci':p['blunder_uci'],'best_uci':bu,'cp_loss':p['cp_loss'],'is_white':p['is_white']})
    if (i//BS)%20==0 or i+BS>=len(valid):
        el=time.time()-t0; r=(i+len(chunk))/max(el,1); print(f'{i+len(chunk)}/{len(valid)} ({r:.0f}/s ETA {(len(valid)-i)/max(r,1)/60:.0f}min)',flush=True)
h2.remove(); h7.remove()
A=np.stack(diffs).astype(np.float32)
torch.save({'activations':torch.tensor(A),'mean':A.mean(0),'std':A.std(0),'metadata':metas,
    'config':{'construction':'concat(L2_mean64_diff,L7_mean64_diff)','dims':'1024+1024','source':'v2+maia_best@2600','n':len(diffs)}},OUT)
print(f'SAVED {OUT} ({os.path.getsize(OUT)//1024//1024}MB) n={len(diffs)} err={nerr}',flush=True)
