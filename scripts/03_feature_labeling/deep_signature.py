"""Compute DEEP multi-granularity, multi-axis signature for a set of features.
For each feature, sample up to 1000 of its top-firing positions, compute concentration on:
  - hang granularities: exact piece / major-minor class / any-hangs
  - blunder piece, best-move piece (+ major/minor), best-move type (cap/chk/quiet)
  - is_capture, phase, eval trajectory
Output per feature: the concentrated facts (>=threshold) ready for a constrained labeler.
Usage: python3 deep_signature.py --model <k8|k16> --feats <comma ids or 'fired'> --out sig.json"""
import torch,numpy as np,json,chess,argparse,torch.nn.functional as F
from collections import Counter
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--feats',required=True);ap.add_argument('--out',required=True);ap.add_argument('--depth',type=int,default=1000);a=ap.parse_args()
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];craw=c['activations'].float();zmean=craw.mean(0);zstd=craw.std(0).clamp(min=1e-6)
enr=json.load(open(B+'/position_enrichment_cache.json'))
wp=BASE+f'/output/maia3_sae/btk_2048_{a.model}_nol2.pt'
sd=torch.load(wp,map_location='cpu',weights_only=False)['state_dict']
x=(craw-zmean)/zstd;N=len(x)
VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:100}
PIECE={chess.KNIGHT:'knight',chess.BISHOP:'bishop',chess.ROOK:'rook',chess.QUEEN:'queen',chess.PAWN:'pawn',chess.KING:'king'}
def see(bd,t,stm):
    aa=bd.attackers(stm,t)
    if not aa:return 0
    lva=min(aa,key=lambda s:VAL.get(bd.piece_type_at(s),99));cv=VAL.get(bd.piece_type_at(t),0)
    b2=bd.copy();b2.remove_piece_at(t);b2.set_piece_at(t,bd.piece_at(lva));b2.remove_piece_at(lva)
    return max(0,cv-see(b2,t,not stm))
def evn(s):
    if not s:return 0
    s=str(s).strip()
    if s.startswith('#'):v=int(s[1:].replace('−','-'));return (10000-abs(v)*10)*(1 if v>=0 else -1)
    try:return int(float(s)*100)
    except:return 0
def cls(p):return 'major piece' if p in('queen','rook') else 'minor piece' if p in('bishop','knight') else p
feats=a.feats.split(',') if a.feats!='fired' else list(json.load(open(f'{B}/test_fired_{a.model}.json')).keys())
out={}
for fid in feats:
    f=int(fid);w=sd['W_enc'][:,f];be=sd['b_enc'][f]
    acts=np.zeros(N,np.float32)
    for i in range(0,N,8192):
        acts[i:i+8192]=F.relu((x[i:i+8192]-sd['b_dec'])@w+be).numpy()
    nfire=int((acts>0).sum())
    order=np.argsort(-acts)
    d=min(a.depth,nfire); seg=order[:d]
    samp=seg if d<=600 else seg[np.linspace(0,d-1,600).astype(int)]
    AX={'blunder_piece':Counter(),'best_piece':Counter(),'best_class':Counter(),'best_type':Counter(),
        'hang_exact':Counter(),'hang_class':Counter(),'anyhang':Counter(),'iscap':Counter(),'phase':Counter(),'traj':Counter(),'examples':[]}
    n=0
    for idx in samp:
        m=meta[int(idx)];key=m['fen']+'|'+m['blunder_uci']
        try:
            b=chess.Board(m['fen']);mover=b.turn;bm=chess.Move.from_uci(m['blunder_uci']);pc=b.piece_at(bm.from_square)
            AX['blunder_piece'][PIECE.get(pc.piece_type,'?') if pc else '?']+=1
            AX['iscap']['capture' if b.is_capture(bm) else 'noncapture']+=1
            bu=m.get('best_uci','')
            if bu and len(bu)>=4:
                bmv=chess.Move.from_uci(bu);bpc=b.piece_at(bmv.from_square);bpn=PIECE.get(bpc.piece_type,'?') if bpc else '?'
                AX['best_piece'][bpn]+=1;AX['best_class'][cls(bpn)]+=1
                AX['best_type']['capture' if b.is_capture(bmv) else 'check' if b.gives_check(bmv) else 'quiet']+=1
            b2=b.copy();b2.push(bm);opp=not mover;w2=0;wp2='none'
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq):
                    l=see(b2,sq,opp)
                    if l>w2:w2=l;wp2=PIECE.get(p.piece_type,'?')
            AX['hang_exact'][wp2]+=1;AX['hang_class'][cls(wp2) if wp2!='none' else 'none']+=1
            AX['anyhang']['hangs' if wp2!='none' else 'safe']+=1
            npc=len(b.piece_map());AX['phase']['endgame' if npc<=14 else 'opening' if b.fullmove_number<=12 else 'middlegame']+=1
            e=enr.get(key,{})
            if e and 'error' not in e:
                iw=m.get('is_white',True);eb=evn(e.get('eval_before',0))*(1 if iw else -1);ea=evn(e.get('eval_after',0))*(1 if iw else -1)
                AX['traj']['already_losing_made_worse' if(eb<-150 and ea<eb-100) else 'threw_winning' if(eb>150 and ea<50) else 'even_to_losing' if(abs(eb)<=150 and ea<-150) else 'mixed']+=1
                if len(AX['examples'])<10:
                    AX['examples'].append(f"played {e.get('played_san','?')} best {e.get('best_san','?')} ({e.get('eval_before','?')}->{e.get('eval_after','?')})")
            n+=1
        except:pass
    # concentrated facts (>=0.7) with the value+pct
    facts={}
    for ax,cnt in AX.items():
        if ax=='examples' or not cnt:continue
        v,k=cnt.most_common(1)[0];pct=k/n
        facts[ax]={'value':v,'pct':round(pct,2)} if pct>=0.7 else {'value':None,'pct':round(pct,2),'top':v}
    out[fid]={'fire_rate':round(nfire/N,4),'n_sampled':n,'facts':facts,'examples':AX['examples'][:8]}
json.dump(out,open(a.out,'w'),indent=1)
print(f"{a.model}: deep signatures for {len(out)} features (depth {a.depth})")
