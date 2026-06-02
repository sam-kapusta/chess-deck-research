"""Test-position diagnostic instrument — rebuilt for legibility.

Layout: each test position is a CASE card. Left: the pinned test board + known mistake.
Right: fired features as horizontal 'readings' — activation bar, chip, confidence,
and a tight filmstrip of example boards that expands on hover. Designed so match/mismatch
is legible at a glance.
"""
import sys; sys.path.insert(0, '/home/ec2-user/SageMaker/maia3')
import torch, numpy as np, chess, chess.svg, json, html, urllib.parse
import torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board, get_historical_tokens

B='/home/ec2-user/SageMaker'; BASE=B+'/chess-stage-a'
dev='cuda' if torch.cuda.is_available() else 'cpu'; ELO=1800

cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,
    num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,
    use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',
    dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,gab_per_square_dim=32,gab_intermediate_dim=128)
ck=torch.load(B+'/maia3_79m_fixed.pt',map_location='cpu',weights_only=False)
model=MAIA3Model(cfg); model.load_state_dict(ck); model.eval().to(dev)
tcfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)
buf={}
model.transformer.layers[6].register_forward_hook(lambda m,i,o: buf.__setitem__('L7',o.detach()))
def histtok(b): return get_historical_tokens([tokenize_board(b)]*8,tcfg,0,0,0,0)
def encode_l7(boards,es,eo):
    T=torch.stack([histtok(b) for b in boards]).to(dev)
    with torch.no_grad(): model(T,torch.tensor(es,dtype=torch.float32,device=dev),torch.tensor(eo,dtype=torch.float32,device=dev))
    return buf['L7'].float().cpu().numpy()

class SAE(nn.Module):
    def __init__(s,d,h):
        super().__init__()
        s.W_enc=nn.Parameter(torch.empty(d,h)); s.W_dec=nn.Parameter(torch.empty(h,d))
        s.b_enc=nn.Parameter(torch.zeros(h)); s.b_dec=nn.Parameter(torch.zeros(d))
        s.register_buffer('num_batches_not_active',torch.zeros(h))
    def enc(s,x,th):
        z=F.relu((x-s.b_dec)@s.W_enc+s.b_enc); return z*(z>th)

wd=torch.load(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights.pt',map_location='cpu',weights_only=False)
ns=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights_stats.json'))
mean=torch.tensor(ns['mean']); std=torch.tensor(ns['std']).clamp(min=1e-6)
theta=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_calibration.json'))['global_threshold']
sae=SAE(wd['config']['d_input'],wd['config']['dict_size']); sae.load_state_dict(wd['state_dict'],strict=False); sae.eval()
labels=json.load(open(BASE+'/output/feature_labels_btk_2048_k16_v2.json'))
profiles=json.load(open(BASE+'/output/btk_profiles_btk_2048_k16_v2.json'))
stats=json.load(open(BASE+'/output/feature_stats_btk_2048_k16_v2.json'))
FIRE_CEILING=0.10  # hide features firing on >10% of corpus (too generic for a specific lesson)
def frate(fid): return stats.get(str(fid),{}).get('n_activating',0)/168132
def lab(fid):
    v=labels.get(str(fid),{})
    if 'error' in v or not v: return {'chip':'unlabeled','conf':0,'why':'','label':''}
    a=v.get('analysis',v); return {'chip':a.get('chip','?'),'conf':a.get('confidence',0) or 0,'why':a.get('why_bad',''),'label':a.get('label','')}

def svg(fen,uci,size=128):
    try:
        b=chess.Board(fen); arr=[]
        if uci and len(uci)>=4: arr=[chess.svg.Arrow(chess.parse_square(uci[:2]),chess.parse_square(uci[2:4]),color='#ff5e3a')]
        return chess.svg.board(b,size=size,arrows=arr,coordinates=False)
    except: return ''
def cc(fen): return 'https://www.chess.com/analysis?fen='+urllib.parse.quote(fen,safe='')

test=json.load(open(B+'/chess-deck-research/output/test_positions.json'))

CSS='''
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e0e;--panel:#101716;--panel2:#16201e;--line:#243230;--ink:#e9e4d6;--dim:#8b9a93;--faint:#5a6a64;
--gold:#e0a838;--gold-hi:#ffc759;--green:#6cc08a;--red:#e0664f;--steel:#6f9bb5;--mono:'JetBrains Mono',monospace}
body{background:var(--bg);color:var(--ink);font-family:'Hanken Grotesk',sans-serif;font-size:14px;line-height:1.5;
background-image:radial-gradient(900px 500px at 85% -5%,rgba(224,168,56,.05),transparent 60%);padding:0 0 80px}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:99;opacity:.03;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
header{padding:34px 44px 26px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--panel),transparent)}
header h1{font-family:'Fraunces',serif;font-weight:900;font-size:30px;letter-spacing:-.3px}
header h1 em{font-style:italic;color:var(--gold)}
header .sub{color:var(--dim);font-size:13px;margin-top:8px;max-width:760px;line-height:1.55}
header .legend{margin-top:16px;display:flex;gap:20px;font-size:11.5px;color:var(--dim);font-family:var(--mono)}
header .legend i{font-style:normal;display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}
.wrap{padding:34px 44px;display:flex;flex-direction:column;gap:30px}
.case{display:grid;grid-template-columns:230px 1fr;gap:26px;background:var(--panel);border:1px solid var(--line);
border-radius:16px;padding:22px;animation:rise .5s both}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.pin{position:sticky;top:18px;align-self:start}
.pin .tag{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin-bottom:9px}
.pin h2{font-family:'Fraunces',serif;font-size:19px;font-weight:600;margin:11px 0 4px;line-height:1.15}
.pin .mt{font-family:var(--mono);font-size:11px;color:var(--steel);margin-bottom:8px}
.pin .coach{font-size:12px;color:var(--dim);line-height:1.5;border-left:2px solid var(--line);padding-left:10px;margin-top:10px}
.pin .bd{border-radius:8px;overflow:hidden;border:1px solid var(--line);box-shadow:0 10px 30px -12px #000}
.pin .mv{font-family:var(--mono);font-size:11px;margin-top:8px;color:var(--dim)}
.pin .mv b{color:var(--red)} .pin .mv .g{color:var(--green)}
.reads{display:flex;flex-direction:column;gap:11px}
.read{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px 15px;transition:.18s;position:relative;overflow:hidden}
.read::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--strength,var(--faint))}
.read.hi{border-color:rgba(224,168,56,.35)}
.read:hover{border-color:var(--gold);transform:translateX(2px)}
.rh{display:flex;align-items:center;gap:12px;margin-bottom:3px}
.rank{font-family:var(--mono);font-size:11px;color:var(--faint);width:18px}
.actbar{flex:0 0 92px;height:7px;background:var(--line);border-radius:4px;overflow:hidden}
.actbar i{display:block;height:100%;background:linear-gradient(90deg,var(--gold),var(--gold-hi));border-radius:4px}
.actval{font-family:var(--mono);font-size:11px;color:var(--gold-hi);width:42px}
.fid{font-family:var(--mono);font-size:11px;color:var(--faint)}
.chip{font-family:'Fraunces',serif;font-size:16px;font-weight:600;flex:1;line-height:1.2}
.conf{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px}
.conf.h{background:rgba(108,192,138,.13);color:var(--green)}
.conf.m{background:rgba(224,168,56,.12);color:var(--gold)}
.conf.l{background:rgba(224,102,79,.12);color:var(--red)}
.why{font-size:12px;color:var(--dim);margin:5px 0 0 30px;line-height:1.45}
.film{display:flex;gap:10px;margin:13px 0 2px 30px;flex-wrap:wrap}
.film .ex{position:relative;border-radius:7px;overflow:hidden;border:1px solid var(--line);transition:.15s;width:150px}
.film .ex svg{width:150px!important;height:150px!important;display:block}
.film .ex:hover{border-color:var(--gold);box-shadow:0 10px 26px -10px #000;transform:translateY(-2px)}
.film .ex .cap{background:rgba(0,0,0,.55);font-family:var(--mono);font-size:11px;color:var(--gold-hi);
text-align:center;padding:3px 0;letter-spacing:.04em}
.unl{opacity:.5}
.hidden-row{margin-top:6px;padding:9px 13px;font-size:11.5px;color:var(--faint);background:rgba(255,255,255,.015);
border:1px dashed var(--line);border-radius:9px;line-height:1.5}
'''
FONTS='<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,900;1,9..144,500&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">'
parts=['<!DOCTYPE html><html><head><meta charset="utf-8">',FONTS,'<style>',CSS,'</style></head><body>',
 '<header><h1>k=16 SAE · <em>does it fire the right mistake?</em></h1>',
 '<div class="sub">Ten positions with a known mistake. For each, the SAE\'s top fired features — '
 'after hiding features that fire on &gt;10% of all blunders (too generic to be a specific lesson; shown muted at the bottom of each case). '
 'The question: does the strongest specific feature describe <em>this</em> mistake, and do its examples cohere?</div>',
 '<div class="legend"><i><span class="dot" style="background:var(--green)"></span>conf ≥80</i>'
 '<i><span class="dot" style="background:var(--gold)"></span>60–79</i>'
 '<i><span class="dot" style="background:var(--red)"></span>&lt;60</i>'
 '<i><span class="dot" style="background:var(--gold-hi)"></span>activation strength</i></div></header>',
 '<div class="wrap">']

for tc in test:
    fen,bu,best_uci=tc['fen'],tc['blunder_uci'],tc.get('best_uci')
    try:
        b=chess.Board(fen); bb=b.copy(); bb.push(chess.Move.from_uci(bu))
        sb=b.copy(); sb.push(chess.Move.from_uci(best_uci))
        L7=encode_l7([bb,sb],[ELO,ELO],[ELO,ELO]); diff=L7[1].mean(0)-L7[0].mean(0)
    except Exception as e:
        parts.append(f'<div class="case">ERROR {tc["id"]}: {e}</div>'); continue
    x=torch.tensor(diff,dtype=torch.float32); x=(x-mean)/std; x=x/x.norm().clamp(min=1e-8)
    with torch.no_grad(): acts=sae.enc(x.unsqueeze(0),theta)[0].detach().numpy()
    allfired=[(int(f),float(acts[f])) for f in np.where(acts>0)[0]]
    allfired.sort(key=lambda z:-z[1])
    fired=[(f,a) for f,a in allfired if frate(f)<=FIRE_CEILING][:5]   # specific features only
    hidden=[(f,a) for f,a in allfired if frate(f)>FIRE_CEILING][:5]   # broad blobs, shown muted
    maxact=fired[0][1] if fired else 1
    parts.append('<div class="case"><div class="pin">'
                 f'<div class="tag">test case</div>'
                 f'<div class="bd"><a href="{cc(fen)}" target="_blank">{svg(fen,bu,196)}</a></div>'
                 f'<div class="mv"><b>{html.escape(bu)}</b> played · <span class="g">{html.escape(best_uci)}</span> best</div>'
                 f'<h2>{tc["id"].replace("_"," ")}</h2>'
                 f'<div class="mt">{tc["mistake_type"]}</div>'
                 f'<div class="coach">{html.escape(tc.get("coach_would_say",""))}</div></div>'
                 '<div class="reads">')
    for rank,(fid,act) in enumerate(fired):
        L=lab(fid); conf=L['conf']
        cc_cls='h' if conf>=80 else ('m' if conf>=60 else 'l')
        unl=' unl' if L['chip']=='unlabeled' else ''
        hi=' hi' if rank==0 else ''
        pct=int(100*act/maxact)
        parts.append(f'<div class="read{hi}{unl}" style="--strength:{"var(--gold)" if rank==0 else "var(--faint)"}">'
                     f'<div class="rh"><span class="rank">#{rank+1}</span>'
                     f'<span class="actbar"><i style="width:{pct}%"></i></span><span class="actval">{act:.2f}</span>'
                     f'<span class="fid">f{fid}</span>'
                     f'<span class="chip">{html.escape(str(L["chip"]))}</span>'
                     f'<span class="conf {cc_cls}">{conf}</span></div>')
        if L['why']: parts.append(f'<div class="why">{html.escape(L["why"][:150])}</div>')
        parts.append('<div class="film">')
        for ex in profiles.get(str(fid),{}).get('examples',[])[:8]:
            parts.append(f'<a class="ex" href="{cc(ex["fen"])}" target="_blank">{svg(ex["fen"],ex["uci"],150)}'
                         f'<span class="cap">{html.escape(ex["uci"])}</span></a>')
        parts.append('</div></div>')
    if hidden:
        chips=', '.join(f'{html.escape(str(lab(f)["chip"]))} <span style="color:var(--faint)">({frate(f)*100:.0f}%)</span>' for f,a in hidden)
        parts.append(f'<div class="hidden-row">hidden — fire on &gt;{int(FIRE_CEILING*100)}% of corpus (too generic): {chips}</div>')
    parts.append('</div></div>')

parts.append('</div></body></html>')
open(B+'/test_positions_k16_v2.html','w').write('\n'.join(parts))
print('WROTE test_positions_k16_v2.html')
