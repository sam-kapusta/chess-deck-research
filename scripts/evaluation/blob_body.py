"""Characterize f101 across its FULL activation range, not just top-60.
A 33%-fire feature activates on ~55k positions. The top-60 are the extreme tip.
Bucket by activation percentile and measure hang-rate in each bucket."""
import torch,numpy as np,json,chess,torch.nn.functional as F
from collections import Counter
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
sd=torch.load(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights.pt',map_location='cpu',weights_only=False)['state_dict']
ns=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights_stats.json'))
mean=torch.tensor(ns['mean']);std=torch.tensor(ns['std']).clamp(min=1e-6)
th=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_calibration.json'))['global_threshold']
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];raw=c['activations'].float()
x=(raw-mean)/std;x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8);N=len(x)
PIECE={chess.PAWN:'pawn',chess.KNIGHT:'knight',chess.BISHOP:'bishop',chess.ROOK:'rook',chess.QUEEN:'queen',chess.KING:'king'}
vals={chess.QUEEN:9,chess.ROOK:5,chess.BISHOP:3,chess.KNIGHT:3,chess.PAWN:1}
F101=101
w=sd['W_enc'][:,F101];be=sd['b_enc'][F101]
acts=np.zeros(N,np.float32)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@w+be).numpy();acts[i:i+8192]=z*(z>th)
fire=(acts>0)
print(f"f101 fires on {fire.sum()} / {N} = {100*fire.mean():.0f}%")
# bucket firing positions by activation level
firing_idx=np.where(fire)[0]
firing_act=acts[firing_idx]
order=firing_idx[np.argsort(-firing_act)]  # high to low
def hangrate(idxs):
    hangs=Counter();n=0
    for idx in idxs:
        m=meta[int(idx)]
        try:
            b=chess.Board(m['fen']);mv=chess.Move.from_uci(m['blunder_uci'])
            b2=b.copy();b2.push(mv);mover=b.turn;opp=not mover;bv=0;bh=None
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq) and len(b2.attackers(opp,sq))>len(b2.attackers(mover,sq)):
                    if vals.get(p.piece_type,0)>bv:bv=vals.get(p.piece_type,0);bh=PIECE[p.piece_type]
            hangs[bh or 'nothing']+=1;n+=1
        except:pass
    return n,hangs
# top 60, then percentile bands of the firing distribution
bands=[('top-60',order[:60]),
       ('p0-5% (tip)',order[:len(order)//20]),
       ('p25-30%',order[len(order)//4:len(order)//4+300]),
       ('p50-55% (median)',order[len(order)//2:len(order)//2+300]),
       ('p90-95% (weak)',order[int(len(order)*0.9):int(len(order)*0.9)+300])]
for name,idxs in bands:
    n,h=hangrate(idxs)
    qr=100*(h.get('queen',0)+h.get('rook',0))//max(n,1)
    print(f"  {name:18s} n={n:4d}  Q+R hang {qr:3d}%  | {dict(h.most_common(4))}")
