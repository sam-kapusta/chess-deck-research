#!/usr/bin/env python3
"""Print the per-feature fire-rate distribution of a BatchTopK SAE — the 'is the head/tail healthy?'
view. Encodes the full l7only cache through the SAE (z-score, no L2; BatchTopK at the model's k),
computes each feature's fire rate, and buckets them so dictionaries sit side by side.

Run on chess-poc:
  cd ~/SageMaker && python3 fire_distribution.py --weights chess-stage-a/output/maia3_sae/btk_128_k1_nol2.pt
"""
import argparse, torch, numpy as np, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--cache", default="/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
ap.add_argument("--k", type=int, default=0, help="0 = read from weights config")
a = ap.parse_args()

ck = torch.load(a.weights, map_location="cpu", weights_only=False)
sd = ck["state_dict"]; K = a.k or int((ck.get("config") or {}).get("k", 6))
c = torch.load(a.cache, map_location="cpu", weights_only=False)
raw = c["activations"].float(); x = (raw - raw.mean(0)) / raw.std(0).clamp(min=1e-6)
We, be, bd = sd["W_enc"], sd["b_enc"], sd["b_dec"]; N, D = x.shape[0], We.shape[1]

fire = np.zeros(D)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - bd) @ We + be)
    tv, ti = z.topk(K, dim=1)
    m = torch.zeros_like(z).scatter_(1, ti, 1.0) * (z > 0)
    fire += m.sum(0).numpy()
fire = fire / N * 100  # percent
live = int((fire > 0).sum()); dead = D - live
frs = np.sort(fire[fire > 0])[::-1]; tot = frs.sum()

print(f"{a.weights.split('/')[-1]} | dict={D} k={K} | live={live} dead={dead} | total fire mass={tot:.0f}%")
print("-" * 56)
for name, lo, hi in [(">5%",5,1e9),("2-5%",2,5),("1-2%",1,2),("0.5-1%",0.5,1),
                     ("0.2-0.5%",0.2,0.5),("0.1-0.2%",0.1,0.2),("<0.1%",0,0.1)]:
    sel = frs[(frs >= lo) & (frs < hi)]
    print(f"  {name:>9}: {len(sel):>4} feats  ({100*len(sel)/live:>4.0f}% of live, {sel.sum()/tot*100:>4.0f}% of fire)")
for t in (5, 10, 20, 50):
    print(f"  top {t:>2} feats carry {frs[:t].sum()/tot*100:>3.0f}% of all firing")
# Gini of the fire distribution (0 = perfectly even, 1 = all in one feature)
v = np.sort(frs); n = len(v); cum = np.cumsum(v)
gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
print(f"  Gini(fire over live feats) = {gini:.3f}   (lower = more even/healthy)")
