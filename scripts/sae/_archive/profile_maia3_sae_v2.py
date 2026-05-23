#!/usr/bin/env python3
"""Profile Maia 3 SAE v2 (z-score only) — show top features with high activations."""
import torch
import torch.nn.functional as F
import json

SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_v2.pt"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"

print("Loading...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
act_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)

state = sae_data["state_dict"]
W_enc = state["W_enc"]
W_dec = state["W_dec"]
b_enc = state["b_enc"]
b_dec = state["b_dec"]
k = sae_data["config"]["k"]

# Normalize same as training: z-score only
x = act_data["activations"].float()
mean = x.mean(dim=0)
std = x.std(dim=0).clamp(min=1e-6)
x = (x - mean) / std

# Forward pass (per-sample top-k for profiling)
print("Running SAE forward pass...")
z = (x - b_dec) @ W_enc + b_enc
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"Activation stats:")
print(f"  Max: {acts.max():.2f}")
print(f"  Mean (nonzero): {acts[acts>0].mean():.2f}")
print(f"  Median (nonzero): {acts[acts>0].median():.2f}")
print(f"  >5.0: {(acts.max(dim=0).values > 5.0).sum()} features")
print(f"  >10.0: {(acts.max(dim=0).values > 10.0).sum()} features")

# Profile all features
metadata = act_data["metadata"]
n_features = acts.shape[1]

# Get per-feature stats
feat_stats = []
for feat_id in range(n_features):
    feat_acts = acts[:, feat_id]
    fire_mask = feat_acts > 0
    if not fire_mask.any():
        continue
    max_act = feat_acts.max().item()
    fire_count = fire_mask.sum().item()
    fire_rate = fire_count / len(feat_acts)
    mean_act = feat_acts[fire_mask].mean().item()
    feat_stats.append({
        "id": feat_id,
        "max": max_act,
        "mean": mean_act,
        "fire_rate": fire_rate,
        "fire_count": fire_count,
    })

feat_stats.sort(key=lambda x: -x["max"])

print(f"\n{'='*70}")
print(f"TOP 30 FEATURES BY MAX ACTIVATION")
print(f"{'='*70}")

for rank, fs in enumerate(feat_stats[:30]):
    feat_id = fs["id"]
    feat_acts = acts[:, feat_id]
    top_idx = torch.argsort(feat_acts, descending=True)[:5]

    print(f"\nFeature {feat_id}: max={fs['max']:.2f}, mean={fs['mean']:.2f}, "
          f"fire_rate={fs['fire_rate']:.3f} ({fs['fire_count']} positions)")
    for idx in top_idx:
        idx = idx.item()
        m = metadata[idx]
        strength = feat_acts[idx].item()
        fen_short = m["fen"][:55]
        uci = m.get("blunder_uci", "")
        cp = m.get("cp_loss", 0)
        print(f"  [{strength:5.2f}] {fen_short}  {uci}  cp={cp}")

# Distribution of max activations across all features
maxes = torch.tensor([fs["max"] for fs in feat_stats])
print(f"\n{'='*70}")
print(f"MAX ACTIVATION DISTRIBUTION ACROSS ALL {len(feat_stats)} FEATURES")
print(f"{'='*70}")
print(f"  Percentiles: 25%={maxes.quantile(0.25):.2f}, 50%={maxes.quantile(0.5):.2f}, "
      f"75%={maxes.quantile(0.75):.2f}, 95%={maxes.quantile(0.95):.2f}")
print(f"  >2.0: {(maxes > 2.0).sum()} features")
print(f"  >5.0: {(maxes > 5.0).sum()} features")
print(f"  >10.0: {(maxes > 10.0).sum()} features")
print(f"  >15.0: {(maxes > 15.0).sum()} features")
