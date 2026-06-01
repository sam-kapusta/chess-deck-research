"""Eval 4 v2 SAEs on 10 test positions -> HTML.
For each SAE: encode its 200k cache (profiles: top-12 examples/feature + fire rates),
encode 10 test diffs, rank fired features, render HTML with chess.com FEN links +
each feature's top profile examples. No LLM labels yet (features shown by id + examples)."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import torch, torch.nn.functional as F, numpy as np, json, os, time, chess
import onnxruntime as ort
from maia3.dataset import tokenize_board, get_historical_tokens
from types import SimpleNamespace

BASE='/home/ec2-user/SageMaker/chess-stage-a'
SAE=BASE+'/output/maia3_sae'; CACHE=BASE+'/cache'
TEST=json.load(open('/home/ec2-user/SageMaker/chess-deck-research/output/test_positions.json'))
best200k=json.load(open('/home/ec2-user/SageMaker/maia_best_200k.json'))
PIECES=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
PROBE='/model/transformer/layers.7/Add_2_output_0'

# SAE registry: name -> (weights, cache, construction)
SAES=[
 ('option_a', SAE+'/maia3_option_a_v2_2048_k16.pt', CACHE+'/maia3_option_a_diff_v2.pt', 'option_a'),
 ('board_diff',SAE+'/maia3_board_diff_v2_2048_k16.pt',CACHE+'/maia3_board_diff_v2.pt','board_diff'),
 ('l2l7',     SAE+'/maia3_l2l7_v2_2048_k16.pt',     CACHE+'/maia3_l2l7_concat_v2.pt','l2l7'),
 ('l7only',   SAE+'/maia3_l7only_v2_2048_k16.pt',   CACHE+'/maia3_l7only_v2.pt','l7only'),
]

# --- ONNX + 79M for test-position diffs ---
so=ort.SessionOptions(); so.intra_op_num_threads=0
sess=ort.InferenceSession('/home/ec2-user/SageMaker/maia3_with_probe.onnx',so,providers=['CPUExecutionProvider'])
def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t
def onnx_h(b,es,eo): return sess.run([PROBE],{'tokens':tok(b)[None],'elo_self':np.array([es],np.float32),'elo_oppo':np.array([eo],np.float32)})[0][0]

_m79=[None]
def m79():
    if _m79[0] is None:
        from maia3.models import MAIA3Model
        cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,gab_per_square_dim=32,gab_intermediate_dim=128)
        ck=torch.load('/home/ec2-user/SageMaker/maia3_79m_fixed.pt',map_location='cpu',weights_only=False)
        m=MAIA3Model(cfg); m.load_state_dict(ck); m.eval()
        _m79[0]=m
    return _m79[0]
def l2l7_h(b,es,eo):
    m=m79(); tc=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)
    buf={}
    h2=m.transformer.layers[1].register_forward_hook(lambda md,i,o:buf.__setitem__('L2',o.detach()))
    h7=m.transformer.layers[6].register_forward_hook(lambda md,i,o:buf.__setitem__('L7',o.detach()))
    T=get_historical_tokens([tokenize_board(b)]*8,tc,0,0,0,0).unsqueeze(0)
    with torch.no_grad(): m(T,torch.tensor([es]),torch.tensor([eo]))
    h2.remove(); h7.remove()
    return buf['L2'][0].numpy(), buf['L7'][0].numpy()  # [64,1024] each

def elos_test(tc):
    # test positions don't carry elo; use 2600 self (top) for eval-time conditioning consistency
    return 2600.,2600.
def maia_best_for(b):
    # reuse corrected extraction
    from maia3.dataset import get_legal_moves_mask
    from maia3.utils import get_all_possible_moves, mirror_move
    if not hasattr(maia_best_for,'ALL'):
        maia_best_for.ALL=get_all_possible_moves(); maia_best_for.D={m:i for i,m in enumerate(maia_best_for.ALL)}
    out=sess.run(['logits_move'],{'tokens':tok(b)[None],'elo_self':np.array([2600.],np.float32),'elo_oppo':np.array([2600.],np.float32)})[0][0]
    mask=get_legal_moves_mask(b,maia_best_for.D).numpy().astype(bool); lg=out.copy(); lg[~mask]=-1e9
    mv=maia_best_for.ALL[int(lg.argmax())]
    if b.turn==chess.BLACK: mv=mirror_move(mv)
    return mv

def test_diff(tc, construction):
    b=chess.Board(tc['fen']); es,eo=elos_test(tc)
    bu=maia_best_for(b)
    if bu==tc['blunder_uci']: bu=tc.get('best_uci',bu)  # fallback to curated if maia agrees w/ blunder
    bl=chess.Move.from_uci(tc['blunder_uci']); bs=chess.Move.from_uci(bu)
    if construction=='option_a':
        H=onnx_h(b,es,eo); return H[bs.to_square]-H[bl.to_square], bu
    if construction=='board_diff':
        bb=b.copy(); bb.push(bl); sb=b.copy(); sb.push(bs)
        return onnx_h(sb,es,eo).mean(0)-onnx_h(bb,es,eo).mean(0), bu
    # l2l7 / l7only need after-boards through 79M
    bb=b.copy(); bb.push(bl); sb=b.copy(); sb.push(bs)
    L2bl,L7bl=l2l7_h(bb,es,eo); L2bs,L7bs=l2l7_h(sb,es,eo)
    d2=(L2bs-L2bl).mean(0); d7=(L7bs-L7bl).mean(0)
    if construction=='l7only': return d7, bu
    return np.concatenate([d2,d7]), bu

# --- SAE forward ---
def load_sae(path):
    d=torch.load(path,map_location='cpu',weights_only=False); sd=d['state_dict']
    mean=torch.tensor(np.array(d['norm']['mean'])).float(); std=torch.tensor(np.array(d['norm']['std'])).float().clamp(min=1e-6)
    return sd,mean,std,d['config']['k']
def encode(diff,sd,mean,std):
    x=(torch.tensor(np.asarray(diff,np.float32))-mean)/std; x=x/x.norm().clamp(min=1e-8)
    with torch.no_grad(): z=F.relu((x-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    return z.numpy()
def threshold(cache_acts,sd,mean,std,k,n=4000):
    raw=cache_acts[:n].float(); x=(raw-mean)/std; x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    with torch.no_grad(): z=F.relu((x-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    v=z.flatten().numpy(); v=v[v>0]; return float(np.sort(v)[::-1][min(n*k,len(v)-1)])

def cl(fen): return 'https://www.chess.com/analysis?fen='+fen.replace(' ','%20')

results={}
for name,wpath,cpath,constr in SAES:
    if not os.path.exists(wpath): print('MISSING',wpath,flush=True); continue
    print(f'=== {name} ===',flush=True)
    sd,mean,std,k=load_sae(wpath)
    cache=torch.load(cpath,map_location='cpu',weights_only=False)
    acts_raw=cache['activations']; meta=cache['metadata']
    thr=threshold(acts_raw,sd,mean,std,k)
    # profiles: encode all 200k, top-12 examples per feature
    D=acts_raw.shape[1]; allz=np.zeros((len(acts_raw),2048),np.float32)
    x=(acts_raw.float()-mean)/std; x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    BS=8192
    for i in range(0,len(x),BS):
        with torch.no_grad(): z=F.relu((x[i:i+BS]-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
        za=z.numpy(); za[za<thr]=0; allz[i:i+BS]=za
    fire=(allz>0).mean(0)
    prof={}
    for fi in range(2048):
        if fire[fi]<=0: continue
        top=np.argsort(-allz[:,fi])[:12]; top=[t for t in top if allz[t,fi]>0]
        prof[fi]=[(meta[t]['fen'],meta[t]['blunder_uci'],meta[t].get('best_uci',''),float(allz[t,fi])) for t in top]
    # test positions
    perpos={}
    for tc in TEST:
        diff,bu=test_diff(tc,constr)
        z=encode(diff,sd,mean,std); z[z<thr]=0
        fired=[(int(fi),float(z[fi])) for fi in np.where(z>0)[0]]; fired.sort(key=lambda x:-x[1])
        perpos[tc['id']]={'fired':fired,'maia_best':bu}
    results[name]={'profiles':prof,'perpos':perpos,'fire':fire.tolist(),'thr':thr,'k':int(k),'live':int((fire>0.001).sum())}
    print(f'  live={results[name]["live"]} thr={thr:.3f}',flush=True)

json.dump({n:{'perpos':r['perpos'],'thr':r['thr'],'k':r['k'],'live':r['live']} for n,r in results.items()},
          open('/home/ec2-user/SageMaker/eval_v2_results.json','w'))

# --- HTML ---
H=['<!DOCTYPE html><html><head><meta charset=utf-8><title>v2 SAE eval</title><style>',
'body{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:0;padding:24px}',
'h1{font-size:20px;border-bottom:1px solid #30363d;padding-bottom:12px}h2{color:#58a6ff;margin-top:32px}',
'.pos{border:1px solid #30363d;border-radius:8px;margin:16px 0;overflow:hidden}',
'.ph{background:#161b22;padding:12px 16px}.pt{font-weight:600}.pm{color:#f0883e;margin-left:8px}',
'.tag{background:#21262d;color:#8b949e;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px}',
'.co{color:#7ee787;font-size:13px;margin-top:6px;font-style:italic}.mb{color:#d2a8ff;font-size:12px;margin-top:4px}',
'.feat{padding:8px 16px;border-top:1px solid #21262d}.fid{font-weight:600;color:#58a6ff}.act{color:#8b949e;font-size:11px}',
'.ex{font-size:11px;margin-left:16px}.ex a{color:#6e9fff;text-decoration:none}.ex a:hover{text-decoration:underline}',
'a{color:#58a6ff}</style></head><body>']
H.append('<h1>v2 SAE eval — 4 constructions, 10 real blunders (Maia-best@2600, k=16)</h1>')
for name,r in results.items():
    H.append(f'<h2>{name} — {r["live"]} live features, k={r["k"]}, thr={r["thr"]:.3f}</h2>')
    for tc in TEST:
        pp=r['perpos'][tc['id']]; fired=pp['fired'][:8]
        H.append(f'<div class=pos><div class=ph><span class=pt><a href="{cl(tc["fen"])}" target=_blank>{tc["id"]}</a></span>'
                 f'<span class=pm>{tc["move"]}</span><span class=tag>{tc["mistake_type"]}</span>'
                 f'<div class=co>{tc["coach_would_say"]}</div><div class=mb>maia_best@2600: {pp["maia_best"]} | curated: {tc.get("best_uci","?")}</div></div>')
        if not fired: H.append('<div class=feat>(no features fired)</div>')
        for fi,act in fired:
            exs=r['profiles'].get(fi,[])[:4]
            exhtml=''.join(f'<div class=ex><a href="{cl(f)}" target=_blank>{f.split()[0][:24]} {bl}→{bu or "?"}</a> ({a:.2f})</div>' for f,bl,bu,a in exs)
            H.append(f'<div class=feat><span class=fid>f{fi}</span> <span class=act>act={act:.3f} fire={r["fire"][fi]*100:.2f}%</span>{exhtml}</div>')
        H.append('</div>')
H.append('</body></html>')
open('/home/ec2-user/SageMaker/eval_v2.html','w').write('\n'.join(H))
print('WROTE eval_v2.html + eval_v2_results.json',flush=True)
