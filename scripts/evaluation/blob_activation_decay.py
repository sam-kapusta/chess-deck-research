"""Clean measurement: hang-rate (or motif-rate) as a function of activation percentile,
for several blobs. Properly bucketed — no top-N sampling confusion. This is the real signal:
do blobs have a coherent high-activation core + noisy tail?"""
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
vals={chess.QUEEN:9,chess.ROOK:5,chess.BISHOP:3,chess.KNIGHT:3,chess.PAWN:1}
def hangmajor(idxs):  # fraction where a Q or R is left hanging
    qr=0;n=0
    for idx in idxs:
        m=meta[int(idx)]
        try:
            b=chess.Board(m['fen']);mv=chess.Move.from_uci(m['blunder_uci'])
            b2=b.copy();b2.push(mv);mover=b.turn;opp=not mover;hung=set()
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq) and len(b2.attackers(opp,sq))>len(b2.attackers(mover,sq)):
                    hung.add(p.piece_type)
            if chess.QUEEN in hung or chess.ROOK in hung: qr+=1
            n+=1
        except:pass
    return 100*qr/max(n,1)
for f in [101,1487,952]:
    w=sd['W_enc'][:,f];be=sd['b_enc'][f]
    acts=np.zeros(N,np.float32)
    for i in range(0,N,8192):
        z=F.relu((x[i:i+8192]-sd['b_dec'])@w+be).numpy();acts[i:i+8192]=z*(z>th)
    fidx=np.where(acts>0)[0];fa=acts[fidx];order=fidx[np.argsort(-fa)];M=len(order)
    print(f"\nf{f}: fires {M} positions. Q/R-hang rate by activation band:")
    for lo,hi,lbl in [(0,.02,'top 2%'),(.02,.10,'2-10%'),(.10,.30,'10-30%'),(.30,.60,'30-60%'),(.60,1.0,'60-100%')]:
        seg=order[int(M*lo):int(M*hi)]
        samp=seg[np.linspace(0,len(seg)-1,min(300,len(seg))).astype(int)] if len(seg) else seg
        print(f"   {lbl:9s} (act {fa[np.argsort(-fa)][min(int(M*hi),M-1)]:.2f}+): Q/R hang {hangmajor(samp):.0f}%")
