"""Run the 10 known test positions through BOTH k4 and k6, show top firing features + names.
Diff repr = (L7[after best] - L7[after blunder]).mean(0), z-score only, elo 1500 (corpus-matched).
Threshold = calibrated corpus threshold per model (mean k-th largest). Reports per position:
the mistake type + which features fire (sorted by activation) with their integrated-label chips.
Run on chess-poc: python test_positions_dual.py
"""
import json,numpy as np,torch,torch.nn.functional as F,chess
from types import SimpleNamespace
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
from maia3.models import MAIA3Model
import maia3.dataset as ds
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
tests=json.load(open(B+'/chess-deck-research/output/test_positions.json'))
# 79M encoder
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,num_blocks=8,mlp_ratio=2.0,
    dropout=0.0,use_gab=True,use_relative_bias=False,use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,
    activation='gelu',dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,gab_per_square_dim=32,gab_intermediate_dim=128)
ck=torch.load(B+'/maia3_79m_fixed.pt',map_location='cpu',weights_only=False)
m79=MAIA3Model(cfg);m79.load_state_dict(ck);m79.eval()
tok_cfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)
def L7(b,elo=1500.):
    t=ds.tokenize_board(b);tokens=ds.get_historical_tokens([t]*8,tok_cfg,0,0,0,0).unsqueeze(0)
    et=torch.tensor([float(elo)],dtype=torch.float32);H={}
    h=m79.transformer.layers[6].register_forward_hook(lambda mm,i,o:H.__setitem__('h',o.detach()))
    with torch.no_grad(): m79(tokens,et,et)
    h.remove();return H['h'][0].numpy()
def diff(fen,blunder,best):
    b=chess.Board(fen)
    bb=b.copy();bb.push(chess.Move.from_uci(blunder))
    bs=b.copy();bs.push(chess.Move.from_uci(best))
    return (L7(bs)-L7(bb)).mean(0)
# corpus z-score params + per-model threshold
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
craw=c['activations'].float();zmean=craw.mean(0).numpy();zstd=craw.std(0).clamp(min=1e-6).numpy()
xcorp=(craw-craw.mean(0))/craw.std(0).clamp(min=1e-6)
def load_model(tag,dct,kk,labelfile):
    sd=torch.load(BASE+f'/output/maia3_sae/{tag}.pt',map_location='cpu',weights_only=False)['state_dict']
    kth=[]
    for i in range(0,40000,8192):
        z=F.relu((xcorp[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']);kth.append(torch.topk(z,kk,1).values[:,-1].numpy())
    th=float(np.concatenate(kth).mean())
    lab=json.load(open(B+'/'+labelfile))
    return sd,th,lab
models={'k4':load_model('btk_1024_k4_nol2',1024,4,'feature_labels_integrated_d1024_k4.json'),
        'k6':load_model('btk_2048_k6_nol2',2048,6,'feature_labels_integrated_d2048_k6.json')}
def fire(dvec,sd,th):
    xn=(dvec-zmean)/zstd
    z=F.relu((torch.tensor(xn,dtype=torch.float32)-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
    z=z*(z>th)
    idx=np.where(z>0)[0]
    return sorted([(int(i),float(z[i])) for i in idx],key=lambda t:-t[1])
out={}
for tc in tests:
    bu=tc.get('best_uci')
    d=diff(tc['fen'],tc['blunder_uci'],bu)
    rec={'mistake':tc['mistake_type'],'move':tc['move']}
    print(f"\n{'='*70}\n{tc['id']}  ({tc['move']}, {tc['mistake_type']})")
    for mk,(sd,th,lab) in models.items():
        feats=fire(d,sd,th)
        rec[mk]=[{'f':i,'act':round(a,2),'chip':(lab.get(str(i)) or {}).get('chip','?')} for i,a in feats[:8]]
        print(f"  --- {mk}: {len(feats)} fire ---")
        for i,a in feats[:8]:
            chip=(lab.get(str(i)) or {}).get('chip','?')
            print(f"     f{i} ({a:.1f})  {chip}")
    out[tc['id']]=rec
json.dump(out,open(B+'/chess-deck-research/output/test_positions_dual_k4k6.json','w'),indent=1)
print("\nwrote test_positions_dual_k4k6.json")
