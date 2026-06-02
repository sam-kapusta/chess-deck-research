"""Honest coherence: real Static Exchange Evaluation for 'does this move lose material',
and report hang-concentration SEPARATELY from moved-piece-concentration (the latter is weak/inflated).
SEE = simulate the full capture sequence on the destination/hanging square with least-valuable-attacker."""
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
def see(board, sq, side):
    """Static exchange eval on square sq, captures initiated by `side`. Returns material gain for side."""
    occ_val=VAL.get(board.piece_type_at(sq),0) if board.piece_at(sq) else 0
    def recur(bd, target, stm):
        atks=bd.attackers(stm, target)
        if not atks: return 0
        # least valuable attacker
        lva=min(atks, key=lambda s: VAL.get(bd.piece_type_at(s),99))
        cap_val=VAL.get(bd.piece_type_at(target),0)
        bd2=bd.copy(); bd2.remove_piece_at(target); bd2.set_piece_at(target,bd.piece_at(lva)); bd2.remove_piece_at(lva)
        return max(0, cap_val - recur(bd2, target, not stm))
    return recur(board, sq, side)
def worst_hang(fen,uci):
    """After the move, what's the most valuable mover-piece that LOSES material by SEE (real)."""
    try:
        b=chess.Board(fen);mv=chess.Move.from_uci(uci);mover=b.turn
        b2=b.copy();b2.push(mv);opp=not mover;worst=0;wp=None
        for sq in chess.SQUARES:
            p=b2.piece_at(sq)
            if p and p.color==mover and b2.is_attacked_by(opp,sq):
                loss=see(b2,sq,opp)  # opp captures on sq
                if loss>worst: worst=loss; wp=PIECE.get(p.piece_type,'?')
        return wp, worst
    except: return None,0
fire=(allz>2.29).mean(0)
res=[]
for f in range(2048):
    if fire[f]<0.002 or fire[f]>0.15: continue
    top=np.argsort(-allz[:,f])[:30]
    hangs=Counter();n=0;lossvals=[]
    for idx in top:
        if allz[idx,f]<=0:break
        wp,loss=worst_hang(meta[int(idx)]['fen'],meta[int(idx)]['blunder_uci'])
        if loss>=2:  # genuinely loses a minor+ by SEE
            hangs[wp]+=1
        else:
            hangs['none']+=1
        lossvals.append(loss);n+=1
    if n<15:continue
    hc=hangs.most_common(1)[0]
    real_hang_frac=1-hangs.get('none',0)/n  # fraction that actually lose material
    res.append((f,fire[f]*100,hc[0],hc[1]/n,real_hang_frac,np.median(lossvals)))
# rank by: real material loss AND concentrated on one piece
res.sort(key=lambda r:-(r[3] if r[2]!='none' else 0))
print("Features by REAL-SEE hang concentration (excludes 'nothing actually hangs'):\n")
print(f"{'feat':5s} {'fire%':>5s} {'hangpiece':>9s} {'conc':>5s} {'realhang%':>9s} {'medLoss':>7s}")
clean=0
for f,fr,hp,conc,rhf,ml in res[:30]:
    if hp!='none' and conc>=0.6: clean+=1
    print(f"f{f:4d} {fr:5.1f} {hp:>9s} {conc:4.0%} {rhf:8.0%} {ml:6.1f}")
nclean=sum(1 for r in res if r[2]!='none' and r[3]>=0.6)
print(f"\n{nclean} features: >=60% of top-30 hang the SAME piece by REAL SEE (trustworthy-coherent)")
print(f"{sum(1 for r in res if r[4]>=0.6)} features: >=60% of top-30 actually lose material (any piece)")
