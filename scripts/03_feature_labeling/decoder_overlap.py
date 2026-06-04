#!/usr/bin/env python3
"""Decoder cosine-similarity overlap for a set of SAE features — the real redundancy test.

Two features that fire on the same CONCEPT may still be distinct features if their decoder
vectors point different directions (they reconstruct different things → real sub-types). If their
decoders are near-parallel (cosine high), the SAE split ONE feature into several — true redundancy
that merging/higher-k would collapse. Names and top-board overlap can't tell these apart; decoder
geometry can.

W_dec is (dict_size, d_input); row f is feature f's decoder vector.

Usage:
  python3 decoder_overlap.py --weights /tmp/btk_2048_k6_nol2.pt --feats /tmp/qh_feats.json \
    --labels output/relabel_v2_neutral_d2048_k6.json --stats output/see_stats_d2048_k6.json \
    [--thresh 0.5]   # cosine >= thresh => flagged as a near-duplicate pair
"""
import argparse, json
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--feats", required=True, help="JSON list of feature ids (strings)")
ap.add_argument("--labels", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--thresh", type=float, default=0.5)
ap.add_argument("--out", default="")
a = ap.parse_args()

sd = torch.load(a.weights, map_location="cpu", weights_only=False)["state_dict"]
W = sd["W_dec"].float().numpy()                       # (dict, d_input)
W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
feats = [str(x) for x in json.load(open(a.feats))]
lab = json.load(open(a.labels))
st = json.load(open(a.stats))
def S(f): return st.get("f" + f) or st.get(f) or {}
def fr(f): return S(f).get("fire_rate", 0)

idx = [int(f) for f in feats]
M = W[idx]                                            # (n, d)
C = M @ M.T                                           # cosine (rows already unit norm)
n = len(feats)

# baseline: what's the typical cosine between RANDOM feature pairs? (calibrates "high")
rng = np.random.default_rng(0)
ri = rng.choice(W.shape[0], size=min(2000, W.shape[0]), replace=False)
RC = W[ri] @ W[ri].T
off = RC[~np.eye(len(ri), dtype=bool)]
print(f"random-pair cosine baseline: mean {off.mean():.3f}  p95 {np.percentile(off,95):.3f}  p99 {np.percentile(off,99):.3f}")
print(f"  -> a cosine well above ~{np.percentile(off,99):.2f} means genuinely aligned, not chance\n")

# within-set distribution
iu = np.triu_indices(n, 1)
vals = C[iu]
print(f"within-set ({n} features, {len(vals)} pairs): mean {vals.mean():.3f}  median {np.median(vals):.3f}  max {vals.max():.3f}")
print(f"  pairs >= {a.thresh}: {int((vals>=a.thresh).sum())}  |  >=0.7: {int((vals>=0.7).sum())}  |  >=0.9: {int((vals>=0.9).sum())}\n")

# greedy clustering at thresh: union-find on edges >= thresh
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y): parent[find(x)] = find(y)
for i in range(n):
    for j in range(i + 1, n):
        if C[i, j] >= a.thresh: union(i, j)
from collections import defaultdict
clusters = defaultdict(list)
for i in range(n): clusters[find(i)].append(i)
clusters = sorted(clusters.values(), key=lambda c: -sum(fr(feats[i]) for i in c))

merged = sum(1 for c in clusters if len(c) > 1)
singles = sum(1 for c in clusters if len(c) == 1)
print(f"=== at cosine>={a.thresh}: {len(clusters)} groups ({merged} multi-feature, {singles} singletons) ===")
print(f"    {n} features collapse to {len(clusters)} distinct directions\n")
for c in clusters:
    if len(c) == 1: continue
    c = sorted(c, key=lambda i: -fr(feats[i]))
    tot = sum(fr(feats[i]) for i in c)
    print(f"  GROUP ({len(c)} feats, {tot*100:.1f}% combined fire):")
    for i in c:
        f = feats[i]
        print(f"      f{f:>5} fire {fr(f)*100:>5.2f}%  cos-to-lead {C[c[0],i]:.2f}  | {lab[f]['chip']}")
sings = [feats[c[0]] for c in clusters if len(c) == 1]
if sings:
    print(f"\n  SINGLETONS (distinct directions, no near-duplicate): {len(sings)}")
    print("   ", ", ".join("f" + s for s in sorted(sings, key=lambda f: -fr(f))[:30]))

if a.out:
    json.dump({"thresh": a.thresh, "feats": feats,
               "clusters": [[feats[i] for i in c] for c in clusters]}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
