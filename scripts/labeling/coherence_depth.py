"""Measure each feature's COHERENCE DEPTH: does its dominant signature hold as you descend
its activation distribution (peak -> 0.8max -> 0.7max)? Objective, no LLM.

For each feature, in 3 activation bands (>=0.9max 'peak', >=0.8max, >=0.7max), compute the
dominant value of its single most-concentrated axis (the feature's "identity" axis). Coherence
depth = how well the peak's dominant value persists at 0.8 and 0.7. A coherent feature keeps its
identity to the 70-80th pct of activation; an incoherent one decays toward corpus base rate.

Identity axis = the axis with the highest peak concentration among:
  material_kind, moved_piece, captured_piece, best_captured_piece, best_piece, own_hang_piece,
  played_check(bool), best_is_check(bool), phase.
Reports per feature: identity axis+value, peak%, %@0.8, %@0.7, and a 'holds_to' verdict.

Run on chess-poc: python coherence_depth.py --model k6 --dict 2048 --out coherence_depth_d2048_k6.json
"""
import torch, numpy as np, json, chess, argparse, torch.nn.functional as F
from collections import Counter
from multiprocessing import Pool
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--dict',type=int,default=2048);ap.add_argument('--out',required=True)
a=ap.parse_args();KK=int(''.join(c for c in a.model if c.isdigit()))
VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:100}
PIECE={chess.KNIGHT:'knight',chess.BISHOP:'bishop',chess.ROOK:'rook',chess.QUEEN:'queen',chess.PAWN:'pawn',chess.KING:'king'}
def cls(p):return 'major' if p in('queen','rook') else 'minor' if p in('bishop','knight') else (p or 'none')
def see(bd,t,stm):
    aa=bd.attackers(stm,t)
    if not aa:return 0
    lva=min(aa,key=lambda s:VAL.get(bd.piece_type_at(s),99));cv=VAL.get(bd.piece_type_at(t),0)
    b2=bd.copy();b2.remove_piece_at(t);b2.set_piece_at(t,bd.piece_at(lva));b2.remove_piece_at(lva)
    return max(0,cv-see(b2,t,not stm))
def worst_hang(board,owner):
    opp=not owner;worst=0;wp=None
    for sq in chess.SQUARES:
        p=board.piece_at(sq)
        if p and p.color==owner and board.is_attacked_by(opp,sq):
            l=see(board,sq,opp)
            if l>worst:worst=l;wp=p.piece_type
    return worst,(PIECE.get(wp) if wp else None)
def feat_axes(args):
    fen,bl,bu=args
    try:
        b=chess.Board(fen);mover=b.turn;bm=chess.Move.from_uci(bl)
        r={}
        mpc=b.piece_at(bm.from_square);r['moved']=PIECE.get(mpc.piece_type) if mpc else 'none'
        r['captured']=PIECE.get(b.piece_at(bm.to_square).piece_type) if (b.is_capture(bm) and b.piece_at(bm.to_square)) else ('pawn' if b.is_capture(bm) else 'none')
        r['played_check']='check' if b.gives_check(bm) else 'no'
        npc=len(b.piece_map());r['phase']='endgame' if npc<=12 else 'opening' if b.fullmove_number<=12 else 'middlegame'
        bb=b.copy();bb.push(bm);w,wp=worst_hang(bb,mover)
        gain=(VAL.get(b.piece_at(bm.to_square).piece_type,1) if (b.is_capture(bm) and b.piece_at(bm.to_square)) else (1 if b.is_capture(bm) else 0))
        net=gain-w
        r['material']=('safe' if w==0 else 'hangs' if gain==0 else 'trade' if net>=-1 else 'loses' if net<=-3 else 'down')
        r['own_piece']=cls(wp) if wp else 'none'
        if bu and len(bu)>=4:
            mv=chess.Move.from_uci(bu);r['best_check']='check' if b.gives_check(mv) else 'no'
            r['best_captured']=PIECE.get(b.piece_at(mv.to_square).piece_type) if (b.is_capture(mv) and b.piece_at(mv.to_square)) else ('pawn' if b.is_capture(mv) else 'none')
        return r
    except Exception:return None
AXES=['material','moved','captured','best_captured','best_check','played_check','own_piece','phase']
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];craw=c['activations'].float();zmean=craw.mean(0);zstd=craw.std(0).clamp(min=1e-6);x=(craw-zmean)/zstd;N=len(x)
sd=torch.load(BASE+f'/output/maia3_sae/btk_{a.dict}_{a.model}_nol2.pt',map_location='cpu',weights_only=False)['state_dict']
kth=[]
for i in range(0,40000,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']);kth.append(torch.topk(z,KK,1).values[:,-1].numpy())
th=float(np.concatenate(kth).mean())
D=sd['W_enc'].shape[1];ACT=np.zeros((N,D),np.float32)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy();ACT[i:i+z.shape[0]]=z*(z>th)
fire=(ACT>0).mean(0);live=np.where(fire>0)[0]
print(f"{a.model}: {len(live)} live; computing axes for union of band positions",flush=True)
# collect all positions in any feature's >=0.7max band (cap per feature), dedup, SEE once
need={}
feat_bands={}
for f in live:
    col=ACT[:,f];mx=col.max()
    p9=np.where(col>=0.9*mx)[0];p8=np.where(col>=0.8*mx)[0];p7=np.where(col>=0.7*mx)[0]
    # cap each band to 400 strongest for speed
    def cap(idxs):
        return idxs[np.argsort(-col[idxs])[:400]] if len(idxs)>400 else idxs
    p9,p8,p7=cap(p9),cap(p8),cap(p7)
    feat_bands[int(f)]=(p9,p8,p7)
    for idxs in (p7,):  # p7 is superset of p8,p9
        for i in idxs: need[int(i)]=None
keys=list(need.keys())
print(f"unique positions to SEE: {len(keys)}",flush=True)
args=[(meta[i]['fen'],meta[i]['blunder_uci'],meta[i].get('best_uci','')) for i in keys]
with Pool(16) as p: res=p.map(feat_axes,args,chunksize=256)
pos={keys[j]:res[j] for j in range(len(keys)) if res[j] is not None}
def dom(idxs):
    """dominant value + pct of the most-concentrated axis over idxs"""
    best=(None,None,0.0)
    recs=[pos[int(i)] for i in idxs if int(i) in pos]
    if not recs:return best
    for ax in AXES:
        cnt=Counter(r[ax] for r in recs if ax in r and r[ax]!='none')
        if not cnt:continue
        v,k=cnt.most_common(1)[0];pct=k/len(recs)
        if pct>best[2]:best=(ax,v,pct)
    return best
out={}
for f,(p9,p8,p7) in feat_bands.items():
    ax,val,peak=dom(p9)
    if ax is None:continue
    # measure SAME axis+value at 0.8 and 0.7
    def hold(idxs):
        recs=[pos[int(i)] for i in idxs if int(i) in pos]
        if not recs:return 0.0
        return sum(1 for r in recs if r.get(ax)==val)/len(recs)
    h8=hold(p8);h7=hold(p7)
    verdict='holds_to_0.7' if h7>=0.6 else 'holds_to_0.8' if h8>=0.6 else 'peak_only'
    out[f'f{f}']={'identity':f'{ax}={val}','peak_pct':round(peak,2),'pct_0.8':round(h8,2),'pct_0.7':round(h7,2),
                  'n9':len(p9),'n8':len(p8),'n7':len(p7),'verdict':verdict}
json.dump(out,open(a.out,'w'),indent=1)
from collections import Counter as C
vc=C(v['verdict'] for v in out.values())
print(f"coherence-depth verdicts ({len(out)} feats): {dict(vc)}",flush=True)
