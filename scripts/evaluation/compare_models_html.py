"""k8 vs k16 side-by-side, v2 (constrained) labels, BOTH moves on every board:
orange arrow = blunder, green arrow = Maia best. On test boards AND example boards."""
import sys; sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
import torch,numpy as np,chess,chess.svg,json,html,urllib.parse,torch.nn as nn,torch.nn.functional as F
from types import SimpleNamespace
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board,get_historical_tokens
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a';dev='cuda';ELO=2600
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];craw=c['activations'].float();zmean=craw.mean(0);zstd=craw.std(0).clamp(min=1e-6)
test=json.load(open(B+'/chess-deck-research/output/test_positions.json'))
labs={'k8':json.load(open(B+'/test_labels_v2_k8.json')),'k16':json.load(open(B+'/test_labels_v2_k16.json'))}
fired={'k8':json.load(open(B+'/test_fired_k8.json')),'k16':json.load(open(B+'/test_fired_k16.json'))}
def svg(fen,blunder,best,sz=104):
    try:
        b=chess.Board(fen);arr=[]
        if blunder and len(blunder)>=4:arr.append(chess.svg.Arrow(chess.parse_square(blunder[:2]),chess.parse_square(blunder[2:4]),color='#ff5e3a'))
        if best and len(best)>=4:arr.append(chess.svg.Arrow(chess.parse_square(best[:2]),chess.parse_square(best[2:4]),color='#5ec27a'))
        return chess.svg.board(b,size=sz,arrows=arr,coordinates=False)
    except:return ''
def cc(fen):return 'https://www.chess.com/analysis?fen='+urllib.parse.quote(fen,safe='')
def feats_for(tag,posid):
    rows=[]
    for f,d in fired[tag].items():
        for h in d.get('test_hits',[]):
            if h['id']==posid:rows.append((h['act'],f,d))
    rows.sort(reverse=True);return rows[:3]
# per-feature top example boards WITH best_uci
sd_cache={}
def acts_for(tag):
    if tag in sd_cache:return sd_cache[tag]
    sd=torch.load(BASE+f'/output/maia3_sae/btk_2048_{tag}_nol2.pt',map_location='cpu',weights_only=False)['state_dict']
    x=(craw-zmean)/zstd;N=len(x);allz=np.zeros((N,2048),np.float32)
    for i in range(0,N,8192):
        allz[i:i+8192]=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
    sd_cache[tag]=allz;return allz
def ex_boards(tag,f):
    col=acts_for(tag)[:,int(f)];top=np.argsort(-col)[:6]
    return [(meta[int(t)]['fen'],meta[int(t)]['blunder_uci'],meta[int(t)].get('best_uci','')) for t in top if col[t]>0]
CSS='''
body{background:#0d1313;color:#ece3d0;font:13px -apple-system,sans-serif;margin:0;padding:22px}
h1{font-size:20px}.leg{color:#8aa;font-size:12px;margin:4px 0 16px}.leg b{color:#ff5e3a}.leg i{color:#5ec27a;font-style:normal}
.pos{margin:26px 0;border-top:1px solid #2c3c38;padding-top:16px}
.posh{font-size:16px;color:#f0bd5a;font-weight:600}.posh .mt{font-family:monospace;font-size:12px;color:#6f9bb5;margin-left:8px}
.pcoach{color:#8aa;font-size:12px;margin:5px 0 12px;max-width:900px}
.cols{display:grid;grid-template-columns:180px 1fr 1fr;gap:16px;align-items:start}
.testb{border:2px solid #ffcf6b;border-radius:7px;padding:4px}
.colh{font-family:monospace;font-size:13px;color:#f0bd5a;font-weight:700;margin-bottom:8px}
.feat{background:#141d1c;border-radius:8px;border-left:3px solid #d9a441;padding:9px 11px;margin-bottom:9px}
.feat.blob{border-left-color:#c8604c}
.chip{font-family:'Fraunces',serif;font-weight:600;font-size:14px}
.fmeta{font-family:monospace;font-size:10px;color:#8aa;margin:2px 0 6px}
.pm{font-size:11px;color:#c4bca9;margin:3px 0}.pm b{color:#5ec27a}.pm i{color:#ff5e3a;font-style:normal}
.facts{font-family:monospace;font-size:9.5px;color:#6f9bb5;margin:4px 0;line-height:1.4}
.bds{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}
'''
P=['<!DOCTYPE html><html><head><meta charset=utf-8>',
'<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600&display=swap" rel=stylesheet>',
f'<style>{CSS}</style></head><body>',
'<h1>k=8 vs k=16 — v2 labels (concentration-thresholded)</h1>',
'<div class=leg>Arrows: <b>orange = blunder played</b>, <i>green = Maia best move</i>. On every board incl examples. '
'Red-bordered feature = blob (fires >15%). Facts = only axes concentrated &ge;70%.</div>']
for tc in test:
    P.append(f'<div class=pos><div class=posh>{tc["id"]}<span class=mt>{tc["mistake_type"]} · blunder {tc["blunder_uci"]} · best {tc["best_uci"]}</span></div>')
    P.append(f'<div class=pcoach>{html.escape(tc["coach_would_say"])}</div><div class=cols>')
    P.append(f'<div class=testb><a href="{cc(tc["fen"])}" target=_blank>{svg(tc["fen"],tc["blunder_uci"],tc["best_uci"],170)}</a></div>')
    for tag in ['k8','k16']:
        P.append(f'<div><div class=colh>{tag}</div>')
        for act,f,d in feats_for(tag,tc['id']):
            lab=labs[tag].get(f,{});chip=lab.get('chip','?');conf=lab.get('confidence','?')
            blob=' blob' if d['fire_rate']>0.15 else ''
            facts=lab.get('facts',{})
            factstr=' · '.join(f"{k}={v['value']}({int(v['pct']*100)}%)" for k,v in facts.items() if v.get('value'))
            P.append(f'<div class="feat{blob}"><div class=chip>{html.escape(str(chip))}</div>')
            P.append(f'<div class=fmeta>f{f} · act {act:.1f} · fires {d["fire_rate"]*100:.0f}% · conf {conf}</div>')
            if lab.get('played'):P.append(f'<div class=pm>played: <i>{html.escape(str(lab["played"])[:95])}</i></div>')
            if lab.get('missed'):P.append(f'<div class=pm>missed: <b>{html.escape(str(lab["missed"])[:95])}</b></div>')
            if factstr:P.append(f'<div class=facts>{html.escape(factstr)}</div>')
            P.append('<div class=bds>')
            for fen,bl,bs in ex_boards(tag,f):
                P.append(f'<a href="{cc(fen)}" target=_blank>{svg(fen,bl,bs,100)}</a>')
            P.append('</div></div>')
        P.append('</div>')
    P.append('</div></div>')
P.append('</body></html>')
open(B+'/test_compare_v2.html','w').write('\n'.join(P))
print('WROTE test_compare_v2.html')
