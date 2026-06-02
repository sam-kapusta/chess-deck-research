"""2x2: {k16,k32} x {L2, z-score} — identical dual-axis probe for clean comparison.
Tests Sam's hypothesis: L2 normalization is worse than z-score-only."""
import torch,numpy as np,json,chess,torch.nn.functional as F
from collections import Counter
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];raw=c['activations'].float()
zmean=raw.mean(0);zstd=raw.std(0).clamp(min=1e-6)
VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:100}
PIECE={chess.KNIGHT:'N',chess.BISHOP:'B',chess.ROOK:'R',chess.QUEEN:'Q',chess.PAWN:'P',chess.KING:'K'}
def see(bd,t,stm):
    a=bd.attackers(stm,t)
    if not a:return 0
    lva=min(a,key=lambda s:VAL.get(bd.piece_type_at(s),99));cv=VAL.get(bd.piece_type_at(t),0)
    b2=bd.copy();b2.remove_piece_at(t);b2.set_piece_at(t,bd.piece_at(lva));b2.remove_piece_at(lva)
    return max(0,cv-see(b2,t,not stm))
sigc={}
def get_sig(idx):
    if idx in sigc:return sigc[idx]
    m=meta[int(idx)];out={}
    try:
        b=chess.Board(m['fen']);mover=b.turn
        for uci,tag in [(m['blunder_uci'],'bl'),(m.get('best_uci',''),'bs')]:
            if not uci:continue
            mv=chess.Move.from_uci(uci);pc=b.piece_at(mv.from_square)
            out[tag+'_piece']=PIECE.get(pc.piece_type,'?') if pc else '?'
        b2=b.copy();b2.push(chess.Move.from_uci(m['blunder_uci']));opp=not mover;w=0;wp='none'
        for sq in chess.SQUARES:
            p=b2.piece_at(sq)
            if p and p.color==mover and b2.is_attacked_by(opp,sq):
                l=see(b2,sq,opp)
                if l>w:w=l;wp=PIECE.get(p.piece_type,'?')
        out['bl_hang']=wp if w>=2 else 'none'
    except:pass
    sigc[idx]=out;return out
def analyze(tag,wp,k,norm):
    wd=torch.load(wp,map_location='cpu',weights_only=False);sd=wd['state_dict']
    if norm=='l2':
        ns=json.load(open(wp.replace('.pt','_stats.json')));mean=torch.tensor(ns['mean']);std=torch.tensor(ns['std']).clamp(min=1e-6)
        x=(raw-mean)/std;x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    else:
        x=(raw-zmean)/zstd
    N=len(x);allz=np.zeros((N,2048),np.float32)
    for i in range(0,N,8192):
        allz[i:i+8192]=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
    kth=[]
    for i in range(0,40000,8192):
        z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']);kth.append(torch.topk(z,k,1).values[:,-1].numpy())
    th=float(np.concatenate(kth).mean());fire=(allz>th).mean(0)
    cand=[f for f in range(2048) if 0.002<=fire[f]<=0.15]
    def coh(top,key):
        cnt=Counter(get_sig(t).get(key,'?') for t in top);t1=cnt.most_common(1)[0];return t1[0],t1[1]/len(top)
    blo=bso=both=nei=0
    for f in cand:
        top=[t for t in np.argsort(-allz[:,f])[:30] if allz[t,f]>0]
        if len(top)<15:continue
        ba=max([coh(top,'bl_hang'),coh(top,'bl_piece')],key=lambda z:z[1])
        bsa=coh(top,'bs_piece')
        blc=ba[1]>=0.6 and ba[0] not in('none','?');bsc=bsa[1]>=0.6 and bsa[0] not in('none','?')
        if blc and bsc:both+=1
        elif blc:blo+=1
        elif bsc:bso+=1
        else:nei+=1
    tot=blo+bso+both+nei;cohn=blo+bso+both
    print(f"{tag:12s}: cand {tot:4d} | coherent {cohn:3d} ({100*cohn//tot}%) | bl-only {blo} best-only {bso} both {both}")
analyze("k16 L2",BASE+'/output/maia3_sae/btk_2048_k16_v2_weights.pt',16,'l2')
analyze("k16 zscore",BASE+'/output/maia3_sae/btk_2048_k16_nol2.pt',16,'zscore')
analyze("k32 L2",BASE+'/output/maia3_sae/btk_2048_k32_v2_weights.pt',32,'l2')
