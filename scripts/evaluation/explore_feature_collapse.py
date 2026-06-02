"""Explore monosemanticity: do the 10 test positions collapse onto a few broad features?
1. Fire matrix: which features fire on multiple test positions
2. Breadth: corpus fire-rate + piece-hanging diversity of the dominant features
3. Specific-feature search: for each position, is there a SPECIFIC feature that fired (even if not #1)?
"""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import torch,numpy as np,chess,json
import torch.nn as nn,torch.nn.functional as F
from types import SimpleNamespace
from collections import Counter, defaultdict
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board,get_historical_tokens

B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a';dev='cuda';ELO=1800
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,gab_per_square_dim=32,gab_intermediate_dim=128)
m=MAIA3Model(cfg);m.load_state_dict(torch.load(B+'/maia3_79m_fixed.pt',map_location='cpu',weights_only=False));m.eval().to(dev)
tcfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128);buf={}
m.transformer.layers[6].register_forward_hook(lambda a,i,o:buf.__setitem__('L7',o.detach()))
def ht(b):return get_historical_tokens([tokenize_board(b)]*8,tcfg,0,0,0,0)
def enc(bs,es,eo):
    T=torch.stack([ht(x) for x in bs]).to(dev)
    with torch.no_grad():m(T,torch.tensor(es,dtype=torch.float32,device=dev),torch.tensor(eo,dtype=torch.float32,device=dev))
    return buf['L7'].float().cpu().numpy()
class SAE(nn.Module):
    def __init__(s,d,h):
        super().__init__();s.W_enc=nn.Parameter(torch.empty(d,h));s.W_dec=nn.Parameter(torch.empty(h,d));s.b_enc=nn.Parameter(torch.zeros(h));s.b_dec=nn.Parameter(torch.zeros(d));s.register_buffer('num_batches_not_active',torch.zeros(h))
    def e(s,x,t):z=F.relu((x-s.b_dec)@s.W_enc+s.b_enc);return z*(z>t)
wd=torch.load(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights.pt',map_location='cpu',weights_only=False)
ns=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights_stats.json'))
mean=torch.tensor(ns['mean']);std=torch.tensor(ns['std']).clamp(min=1e-6)
th=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_calibration.json'))['global_threshold']
sae=SAE(wd['config']['d_input'],wd['config']['dict_size']);sae.load_state_dict(wd['state_dict'],strict=False);sae.eval()
labels=json.load(open(BASE+'/output/feature_labels_btk_2048_k16_v2.json'))
stats=json.load(open(BASE+'/output/feature_stats_btk_2048_k16_v2.json'))
def chip(f):
    v=labels.get(str(f),{});a=v.get('analysis',v) if 'error' not in v else {};return a.get('chip','(unlabeled)')
test=json.load(open(B+'/chess-deck-research/output/test_positions.json'))

# Encode all 10, get full activation vectors
pos_acts={}
for tc in test:
    b=chess.Board(tc['fen'])
    bb=b.copy();bb.push(chess.Move.from_uci(tc['blunder_uci']))
    sb=b.copy();sb.push(chess.Move.from_uci(tc['best_uci']))
    L7=enc([bb,sb],[ELO,ELO],[ELO,ELO]);diff=L7[1].mean(0)-L7[0].mean(0)
    x=torch.tensor(diff,dtype=torch.float32);x=(x-mean)/std;x=x/x.norm().clamp(min=1e-8)
    pos_acts[tc['id']]=sae.e(x.unsqueeze(0),th)[0].detach().numpy()

# === 1. Fire matrix: which features fire across multiple positions ===
feat_hits=defaultdict(list)
for tid,acts in pos_acts.items():
    for f in np.where(acts>0)[0]:
        feat_hits[int(f)].append((tid,float(acts[f])))
print("=== FEATURES FIRING ON MULTIPLE TEST POSITIONS (collapse check) ===")
multi=sorted([(f,h) for f,h in feat_hits.items() if len(h)>=2],key=lambda x:-len(x[1]))
for f,hits in multi[:12]:
    fr=stats.get(str(f),{}).get('n_activating',0)
    frpct=100*fr/168132
    print(f"  f{f} fires on {len(hits)}/10 positions | corpus fire {frpct:.2f}% | {chip(f)}")
    print(f"      on: {', '.join(t for t,_ in hits)}")
print(f"\ntotal distinct features fired across 10 positions: {len(feat_hits)}")
print(f"features firing on 1 position only: {sum(1 for h in feat_hits.values() if len(h)==1)}")
print(f"features firing on >=3 positions: {sum(1 for h in feat_hits.values() if len(h)>=3)}")

# === 2. Per-position: how many DISTINCT features, and is the top one broad or specific? ===
print("\n=== PER-POSITION: feature specificity ===")
for tc in test:
    acts=pos_acts[tc['id']];fired=np.where(acts>0)[0]
    top=int(np.argmax(acts))
    fr=100*stats.get(str(top),{}).get('n_activating',0)/168132
    print(f"  {tc['id']:22s} {len(fired)} feats fired | top f{top} fires {fr:.2f}% corpus = {chip(top)}")
