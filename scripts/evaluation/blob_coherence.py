"""Score EVERY blob feature (fire>10%) for coherence, to separate real coarse concepts
from mush. Coherence signals (objective, over top-60 positions):
 - hang_concentration: max single piece-type left-hanging fraction (high = coheres on what hangs)
 - motif_concentration: max single opus motif fraction among labeled (high = coheres on mistake type)
 - motif_coverage: fraction with any opus label
 - capture_rate, piece_concentration
Output ranked table: which blobs are real vs mush."""
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
x=(raw-mean)/std;x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8);N=len(x)
PIECE={chess.PAWN:'pawn',chess.KNIGHT:'knight',chess.BISHOP:'bishop',chess.ROOK:'rook',chess.QUEEN:'queen',chess.KING:'king'}
vals={chess.QUEEN:9,chess.ROOK:5,chess.BISHOP:3,chess.KNIGHT:3,chess.PAWN:1}
def chip(f):
    v=labels.get(str(f),{});a=v.get('analysis',v) if 'error' not in v else {};return a.get('chip','(unl)')
# find all blobs: full fire rates
allz_fire=np.zeros(2048)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
    allz_fire+=((z*(z>th))>0).sum(0)
allz_fire/=N
blobs=[int(f) for f in np.where(allz_fire>0.10)[0]]
print(f"{len(blobs)} blob features (>10% fire). Scoring coherence over top-60 each...\n")
# activations for blobs
W=sd['W_enc'][:,blobs];be=sd['b_enc'][blobs]
acts=np.zeros((N,len(blobs)),np.float32)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@W+be).numpy();acts[i:i+8192]=z*(z>th)
rows=[]
for bi,f in enumerate(blobs):
    top=np.argsort(-acts[:,bi])[:60];hangs=Counter();motifs=Counter();mcov=0;caps=0
    for idx in top:
        m=meta[int(idx)];key=keys[int(idx)]
        try:
            b=chess.Board(m['fen']);mv=chess.Move.from_uci(m['blunder_uci'])
            if b.is_capture(mv):caps+=1
            b2=b.copy();b2.push(mv);mover=b.turn;opp=not mover;bv=0;bh=None
            for sq in chess.SQUARES:
                p=b2.piece_at(sq)
                if p and p.color==mover and b2.is_attacked_by(opp,sq) and len(b2.attackers(opp,sq))>len(b2.attackers(mover,sq)):
                    if vals.get(p.piece_type,0)>bv:bv=vals.get(p.piece_type,0);bh=PIECE[p.piece_type]
            if bh:hangs[bh]+=1
        except:pass
        a=opus.get(key,{}).get('analysis',{})
        if isinstance(a,dict) and a.get('tactical_motif'):motifs[a['tactical_motif']]+=1;mcov+=1
    hang_conc=hangs.most_common(1)[0][1]/60 if hangs else 0
    motif_conc=motifs.most_common(1)[0][1]/mcov if mcov else 0
    rows.append((f,allz_fire[f]*100,hang_conc,motif_conc,mcov/60,caps/60,chip(f),
                 motifs.most_common(1)[0][0] if motifs else '-'))
# coherence score = max(hang_conc, motif_conc weighted by coverage)
rows.sort(key=lambda r:-max(r[2],r[3]*r[4]))
print(f"{'feat':5s} {'fire%':>5s} {'hangConc':>8s} {'motifConc':>9s} {'mcov':>5s} {'cap%':>5s}  topMotif / chip")
for f,fire,hc,mc,cov,cap,ch,tm in rows:
    coh='REAL' if max(hc,mc*cov)>0.35 else ('mush' if max(hc,mc*cov)<0.2 else '~')
    print(f"f{f:4d} {fire:5.0f} {hc:8.2f} {mc:9.2f} {cov:5.2f} {cap*100:4.0f}  [{coh}] {tm} / {ch[:30]}")
