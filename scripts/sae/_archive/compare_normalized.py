#!/usr/bin/env python3
"""Fair comparison of SAE versions using per-feature normalized activations.

Normalizes each feature to [0, 1] (divide by feature's max on training data),
then compares coverage on Gemini positions at the same relative thresholds.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from maia3_activations import extract_activations, pool_activations

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"

VERSIONS = {
    "L2 200ep": "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_l2_200ep.pt",
    "Z-score only": "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_v2.pt",
    "Raw (no norm)": "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_raw.pt",
}

# Load Gemini positions
print("Loading Gemini positions...")
with open(GEMINI_PATH) as f:
    gemini_data = json.load(f)

positions = []
seen = set()
for fid, feat in gemini_data.items():
    for ex in feat.get("examples", []):
        fen = ex.get("fen", "")
        uci = ex.get("uci", "")
        if fen and uci and fen not in seen:
            seen.add(fen)
            positions.append({"fen": fen, "uci": uci})
print(f"  {len(positions)} unique Gemini positions")

# Extract Maia 3 activations for Gemini positions
fens = [p["fen"] for p in positions]
ucis = [p["uci"] for p in positions]
rng = np.random.default_rng(42)
elos = rng.integers(600, 2601, size=len(fens)).tolist()

print("Extracting Maia 3 activations...")
raw_acts, mirrored = extract_activations(fens, elo_self=elos, elo_oppo=elos)
pooled = pool_activations(raw_acts, "diff", ucis, mirrored)
x_gemini = torch.from_numpy(pooled).float()
print(f"  Shape: {x_gemini.shape}")

# Load training data for computing feature maxes
print("Loading training activations for normalization stats...")
train_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)
x_train = train_data["activations"].float()

# For each version, compute feature activations on both training data and Gemini
print("\n" + "=" * 70)
print("FAIR COMPARISON: Per-feature normalized to [0, 1]")
print("=" * 70)

for name, sae_path in VERSIONS.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    sae = torch.load(sae_path, map_location="cpu", weights_only=False)
    state = sae["state_dict"]
    k = sae["config"]["k"]

    # Prepare inputs based on normalization
    if "L2" in name:
        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0).clamp(min=1e-6)
        x_t = (x_train - mean) / std
        norms = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_t = x_t / norms
        x_g = (x_gemini - mean) / std
        norms_g = x_g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_g = x_g / norms_g
    elif "Z-score" in name:
        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0).clamp(min=1e-6)
        x_t = (x_train - mean) / std
        x_g = (x_gemini - mean) / std
    else:  # Raw
        x_t = x_train
        x_g = x_gemini

    # Get feature activations on training data (for max computation)
    # Use subset for speed
    z_t = (x_t[:50000] - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
    z_relu_t = F.relu(z_t)
    topk_vals_t, topk_idx_t = torch.topk(z_relu_t, k=k, dim=-1)
    acts_t = torch.zeros_like(z_relu_t)
    acts_t.scatter_(-1, topk_idx_t, topk_vals_t)

    # Per-feature max (95th percentile for robustness)
    feat_maxes = []
    for feat_id in range(acts_t.shape[1]):
        feat_vals = acts_t[:, feat_id]
        nonzero = feat_vals[feat_vals > 0]
        if len(nonzero) > 10:
            feat_maxes.append(nonzero.quantile(0.95).item())
        elif len(nonzero) > 0:
            feat_maxes.append(nonzero.max().item())
        else:
            feat_maxes.append(1.0)
    feat_maxes = torch.tensor(feat_maxes).clamp(min=1e-8)

    # Get feature activations on Gemini positions
    z_g = (x_g - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
    z_relu_g = F.relu(z_g)
    topk_vals_g, topk_idx_g = torch.topk(z_relu_g, k=k, dim=-1)
    acts_g = torch.zeros_like(z_relu_g)
    acts_g.scatter_(-1, topk_idx_g, topk_vals_g)

    # Normalize activations per-feature to [0, 1]
    acts_g_normed = acts_g / feat_maxes.unsqueeze(0)

    # Coverage at relative thresholds
    print(f"\n  Per-feature normalized coverage (threshold = fraction of feature's 95th pctl):")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_features = 0
        for feat_id in range(acts_g_normed.shape[1]):
            n_strong = (acts_g_normed[:, feat_id] > thresh).sum().item()
            if n_strong >= 5:
                n_features += 1
        print(f"    Features with 5+ positions > {thresh:.0%} of max: {n_features}")

    # Also report raw stats for context
    print(f"\n  Raw activation stats:")
    print(f"    Max: {acts_g.max():.3f}")
    print(f"    Mean (nonzero): {acts_g[acts_g > 0].mean():.3f}")
    print(f"    Feature max (95pctl) range: [{feat_maxes.min():.3f}, {feat_maxes.max():.3f}]")
