#!/usr/bin/env python3
"""Corpus-wide firing-set overlap for a set of SAE features — the definitive redundancy test.

Top-10-board overlap (cheap) and decoder cosine (cheap) both suggest distinct-vs-redundant, but
the ground truth is: across ALL 168K corpus positions, do these features fire on the SAME boards
or DIFFERENT ones? Two features that fire on disjoint position sets are behaviorally distinct even
if they share a coaching label.

Encodes the corpus through the SAE (z = ReLU((x - b_dec) @ W_enc + b_enc)), then for each feature
takes its firing set = positions where its activation >= frac * (that feature's max activation),
and reports pairwise Jaccard. frac matches the cohort convention used elsewhere.

Run on chess-poc (cache lives there):
  python3 firing_overlap.py --cache ~/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
    --weights ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k6_nol2.pt \
    --feats qh_feats.json --frac 0.5 --out firing_overlap_qh.json
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--frac", type=float, default=0.5, help="a position 'fires' a feature if act >= frac*max_act for that feature")
ap.add_argument("--out", default="")
a = ap.parse_args()

feats = [int(x) for x in json.load(open(a.feats))]
sd = torch.load(a.weights, map_location="cpu", weights_only=False)["state_dict"]
c = torch.load(a.cache, map_location="cpu", weights_only=False)
# z-score normalize exactly as compute_feature_see_stats.py does (SAE trained on z-scored input)
craw = c["activations"].float()
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd
N = x.shape[0]
print(f"corpus {N} x {x.shape[1]} | z-scored | encoding through SAE ...", flush=True)

# encode in chunks; keep only the columns we need (the feature subset) to save memory
cols = torch.tensor(feats)
ACT = np.zeros((N, len(feats)), np.float32)
b_dec, W_enc, b_enc = sd["b_dec"], sd["W_enc"], sd["b_enc"]
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - b_dec) @ W_enc + b_enc)
    ACT[i:i+8192] = z[:, cols].numpy()
print("encoded.", flush=True)

# firing set per feature: act >= frac * max_act(feature)
maxes = ACT.max(0)
fire_sets = []
for j in range(len(feats)):
    thr = a.frac * maxes[j]
    fire_sets.append(set(np.nonzero(ACT[:, j] >= thr)[0].tolist()) if maxes[j] > 0 else set())
sizes = [len(s) for s in fire_sets]
print(f"firing-set sizes @ frac={a.frac}: min {min(sizes)} median {int(np.median(sizes))} max {max(sizes)}\n")

# pairwise Jaccard + overlap coefficient (intersection / smaller set — catches subset relationships)
from itertools import combinations
pairs = list(combinations(range(len(feats)), 2))
js = []; high = []
for i, j in pairs:
    inter = len(fire_sets[i] & fire_sets[j])
    uni = len(fire_sets[i] | fire_sets[j]) or 1
    smaller = min(len(fire_sets[i]), len(fire_sets[j])) or 1
    jac = inter / uni; ovl = inter / smaller
    js.append(jac)
    if jac >= 0.3 or ovl >= 0.5:
        high.append((feats[i], feats[j], len(fire_sets[i]), len(fire_sets[j]), inter, jac, ovl))
js = np.array(js)
print(f"=== corpus-wide firing overlap, {len(feats)} features, {len(pairs)} pairs ===")
print(f"  Jaccard: mean {js.mean():.4f}  median {np.median(js):.4f}  p95 {np.percentile(js,95):.3f}  max {js.max():.3f}")
print(f"  pairs Jaccard>=0.3: {int((js>=0.3).sum())}  >=0.5: {int((js>=0.5).sum())}  >=0.7: {int((js>=0.7).sum())}")
print(f"\n  high-overlap pairs (Jaccard>=0.3 OR overlap-coef>=0.5): {len(high)}")
for fi, fj, ni, nj, inter, jac, ovl in sorted(high, key=lambda r: -r[5])[:30]:
    print(f"    f{fi} ({ni}) ~ f{fj} ({nj}): inter {inter}  Jaccard {jac:.2f}  overlap {ovl:.2f}")

if a.out:
    json.dump({"frac": a.frac, "feats": feats, "sizes": sizes,
               "jaccard_mean": float(js.mean()), "jaccard_max": float(js.max()),
               "high_pairs": [[int(r[0]), int(r[1]), float(r[5]), float(r[6])] for r in high]},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
