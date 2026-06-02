"""Coherence measured on BOTH moves the diff is built from:
  blunder move (what was played) AND best move (maia top-1, the other half of the diff).
For each feature's top-30 positions, compute concentration on each axis:
  - blunder: piece moved, is_capture, what-hangs-after (real SEE)
  - best:    piece moved, is_capture, is_check (does best move defend/counter)
A feature is coherent if it concentrates on EITHER axis (or jointly). Report which axis explains each.
"""
import torch,numpy as np,json,chess,torch.nn.functional as F
from collections import Counter
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];craw=c['activations'].float();zmean=craw.mean(0);zstd=craw.std(0).clamp(min=1e-6)
wd=torch.load(BASE+'/output/maia3_sae/btk_2048_k16_nol2.pt',map_location='cpu',weights_only=False);sd=wd['state_dict']
xc=(craw-zmean)/zstd;N=len(xc)
allz=np.zeros((N,2048),np.float32)
for i in range(0,N,8192):
    allz[i:i+8192]=F.relu((xc[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:100}
PIECE={chess.KNIGHT:'N',chess.BISHOP:'B',chess.ROOK:'R',chess.QUEEN:'Q',chess.PAWN:'P',chess.KING:'K'}
def see(bd,target,stm):
    atks=bd.attackers(stm,target)
    if not atks:return 0
    lva=min(atks,key=lambda s:VAL.get(bd.piece_type_at(s),99))
    cv=VAL.get(bd.piece_type_at(target),0)
    b2=bd.copy();b2.remove_piece_at(target);b2.set_piece_at(target,bd.piece_at(lva));b2.remove_piece_at(lva)
    return max(0,cv-see(b2,target,not stm))
def move_sig(fen,uci,tag):
    """Return signature tokens for a move: piece, capture, check, and (for blunder) worst-hang."""
    try:
        b=chess.Board(fen);mv=chess.Move.from_uci(uci);mover=b.turn
        pc=b.piece_at(mv.from_square);pt=PIECE.get(pc.piece_type,'?') if pc else '?'
        cap='x' if b.is_capture(mv) else '-'
        chk='+' if b.gives_check(mv) else '-'
        toks={f'{tag}_piece':pt, f'{tag}_cap':cap, f'{tag}_chk':chk}
        if tag=='bl':  # what hangs after blunder
            b2=b.copy();b2.push(mv);opp=not mover;worst=0;wp='none'
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq):
                    l=see(b2,sq,opp)
                    if l>worst:worst=l;wp=PIECE.get(p.piece_type,'?')
            toks['bl_hang']=wp if worst>=2 else 'none'
        return toks
    except: return {}
# precompute signatures for all positions that appear in any top-30 (lazy: only top features)
fire=(allz>2.29).mean(0)
cand=[f for f in range(2048) if 0.002<=fire[f]<=0.15]
sigcache={}
def get_sig(idx):
    if idx in sigcache:return sigcache[idx]
    m=meta[int(idx)]
    s=move_sig(m['fen'],m['blunder_uci'],'bl')
    s.update(move_sig(m['fen'],m.get('best_uci',''),'bs'))
    sigcache[idx]=s;return s
rows=[]
for f in cand:
    top=np.argsort(-allz[:,f])[:30]
    top=[t for t in top if allz[t,f]>0]
    if len(top)<15:continue
    keys=['bl_piece','bl_hang','bl_cap','bs_piece','bs_cap','bs_chk']
    conc={}
    for k in keys:
        cnt=Counter(get_sig(t).get(k,'?') for t in top)
        top1=cnt.most_common(1)[0]
        conc[k]=(top1[0],top1[1]/len(top))
    # best single explanatory axis (exclude trivial cap/chk '-' dominance by requiring informative value)
    blunder_axis=max([conc['bl_hang'],conc['bl_piece']],key=lambda x:x[1])
    best_axis=conc['bs_piece']
    rows.append((f,fire[f]*100,conc,blunder_axis,best_axis))
# how many coherent on blunder-axis only, best-axis only, both
def coh(v,thr=0.6): return v[1]>=thr and v[0] not in ('none','?','-')
bl_only=sum(1 for r in rows if coh(r[3]) and not coh(r[4]))
bs_only=sum(1 for r in rows if coh(r[4]) and not coh(r[3]))
both=sum(1 for r in rows if coh(r[3]) and coh(r[4]))
neither=sum(1 for r in rows if not coh(r[3]) and not coh(r[4]))
print(f"{len(rows)} candidate features (fire 0.2-15%). Coherence at >=60% concentration:")
print(f"  blunder-axis only:  {bl_only}")
print(f"  best-move-axis only: {bs_only}   <-- features my old probe MISSED")
print(f"  both axes:           {both}")
print(f"  neither:             {neither}")
print(f"\nExamples of BEST-MOVE-axis-only (old probe blind to these):")
shown=0
for f,fr,conc,ba,bsa in rows:
    if coh(bsa) and not coh(ba):
        print(f"  f{f} fire{fr:.1f}%: best-move {conc['bs_piece'][0]}({conc['bs_piece'][1]:.0%}) cap{conc['bs_cap'][0]}({conc['bs_cap'][1]:.0%}) chk{conc['bs_chk'][0]}({conc['bs_chk'][1]:.0%}) | blunder-hang {conc['bl_hang'][0]}({conc['bl_hang'][1]:.0%})")
        shown+=1
        if shown>=10:break
