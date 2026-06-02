"""Subsample the dedup l7only cache to fixed sizes for the corpus-size experiment.
Random subsample (seeded), preserving the {activations, metadata, mean, std} structure."""
import torch, numpy as np, json
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
acts=c['activations']; meta=c['metadata']; N=len(acts)
print(f"full cache: {N}")
rng=np.random.default_rng(42)
for size in [42000,84000,126000]:
    idx=np.sort(rng.choice(N,size=size,replace=False))
    sub={'activations':acts[idx], 'metadata':[meta[i] for i in idx],
         'mean':c.get('mean'),'std':c.get('std')}
    out=f"{BASE}/cache/maia3_l7only_v2_dedup_{size//1000}k.pt"
    torch.save(sub,out); print(f"wrote {out}: {size}")
