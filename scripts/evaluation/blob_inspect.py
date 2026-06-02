"""Inspect what the blob features REALLY represent. For the top blobs (highest fire rate),
pull a large sample (60) of their top-activating positions + compute objective stats:
piece type, capture/check, piece-left-hanging, eval trajectory, phase, opus motif.
Coherence test: does a 33%-corpus feature share a real pattern, or is it a grab-bag?"""
import torch,numpy as np,json,chess,torch.nn.functional as F
from collections import Counter
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
sd=torch.load(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights.pt',map_location='cpu',weights_only=False)['state_dict']
ns=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_weights_stats.json'))
mean=torch.tensor(ns['mean']);std=torch.tensor(ns['std']).clamp(min=1e-6)
th=json.load(open(BASE+'/output/maia3_sae/btk_2048_k16_v2_calibration.json'))['global_threshold']
labels=json.load(open(BASE+'/output/feature_labels_btk_2048_k16_v2.json'))
opus=json.load(open(B+'/all_positions_labeled_opus.json'))
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
meta=c['metadata'];raw=c['activations'].float()
keys=[m['fen']+'|'+m['blunder_uci'] for m in meta]
x=(raw-mean)/std;x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
N=len(x)
PIECE={chess.PAWN:'pawn',chess.KNIGHT:'knight',chess.BISHOP:'bishop',chess.ROOK:'rook',chess.QUEEN:'queen',chess.KING:'king'}
vals={chess.QUEEN:9,chess.ROOK:5,chess.BISHOP:3,chess.KNIGHT:3,chess.PAWN:1}
def chip(f):
    v=labels.get(str(f),{});a=v.get('analysis',v) if 'error' not in v else {};return a.get('chip','(unl)')
# the top blobs from the analysis
BLOBS=[1487,101,98,959,952,1398]
# compute activations for just these features over full corpus
W=sd['W_enc'][:,BLOBS]; be=sd['b_enc'][BLOBS]
acts=np.zeros((N,len(BLOBS)),np.float32)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@W+be).numpy()
    acts[i:i+8192]=z*(z>th)
for bi,f in enumerate(BLOBS):
    col=acts[:,bi]; fire=100*(col>0).mean()
    top=np.argsort(-col)[:60]
    pieces=Counter();caps=0;checks=0;hangs=Counter();motifs=Counter();phases=Counter();mcov=0
    trj_lose=0;trj_threw=0
    def evn(s):
        if not s:return 0
        s=str(s).strip()
        if s.startswith('#'):return (10000-abs(int(s[1:].replace('−','-')))*10)*(1 if int(s[1:].replace('−','-'))>=0 else -1)
        try:return int(float(s)*100)
        except:return 0
    for idx in top:
        m=meta[int(idx)];key=keys[int(idx)]
        try:
            b=chess.Board(m['fen']);mv=chess.Move.from_uci(m['blunder_uci'])
            pc=b.piece_at(mv.from_square)
            if pc:pieces[PIECE.get(pc.piece_type,'?')]+=1
            if b.is_capture(mv):caps+=1
            if b.gives_check(mv):checks+=1
            b2=b.copy();b2.push(mv);mover=b.turn;opp=not mover;bv=0;bh=None
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq) and len(b2.attackers(opp,sq))>len(b2.attackers(mover,sq)):
                    if vals.get(p.piece_type,0)>bv:bv=vals.get(p.piece_type,0);bh=PIECE[p.piece_type]
            if bh:hangs[bh]+=1
        except:pass
        a=opus.get(key,{}).get('analysis',{})
        if isinstance(a,dict) and a.get('tactical_motif'):motifs[a['tactical_motif']]+=1;mcov+=1
    n=len(top)
    print(f"\n{'='*64}\nf{f} — fire {fire:.0f}% — chip: {chip(f)}")
    print(f"  piece:    {dict(pieces.most_common(4))}")
    print(f"  capture {100*caps//n}%  check {100*checks//n}%")
    print(f"  hangs:    {dict(hangs.most_common(4))}")
    print(f"  motif({100*mcov//n}% cov): {dict(motifs.most_common(4))}")
