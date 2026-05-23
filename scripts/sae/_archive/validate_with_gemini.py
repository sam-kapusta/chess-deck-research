#!/usr/bin/env python3
"""Validate Maia 3 SAE features against Gemini-analyzed positions.

Takes positions that already have Gemini tactical analysis, runs them through
Maia 3 + SAE, and checks if SAE features align with Gemini's tactical labels.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from maia3_activations import extract_activations, pool_activations

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32.pt"
ACT_CACHE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"

# Load Gemini analyses and extract unique positions with UCIs
print("Loading Gemini position analyses...")
with open(GEMINI_PATH) as f:
    gemini_data = json.load(f)

positions = []
seen_fens = set()
for fid, feat in gemini_data.items():
    for ex in feat.get("examples", []):
        fen = ex.get("fen", "")
        uci = ex.get("uci", "")
        intent = ex.get("gemini_intent", "")
        blunder = ex.get("gemini_blunder", "")
        failure = ex.get("gemini_failure", "")
        if fen and uci and (intent or blunder) and fen not in seen_fens:
            seen_fens.add(fen)
            positions.append({
                "fen": fen,
                "uci": uci,
                "gemini_intent": intent,
                "gemini_blunder": blunder,
                "gemini_failure": failure,
                "full_text": f"{intent} {blunder} {failure}".strip(),
            })

print(f"  {len(positions)} unique positions with Gemini analysis + UCI")

# Extract Maia 3 activations for these positions
fens = [p["fen"] for p in positions]
ucis = [p["uci"] for p in positions]

print(f"\nExtracting Maia 3 activations for {len(fens)} positions...")
rng = np.random.default_rng(42)
elos = rng.integers(600, 2601, size=len(fens)).tolist()
raw, mirrored = extract_activations(fens, elo_self=elos, elo_oppo=elos)
pooled = pool_activations(raw, "diff", ucis, mirrored)
print(f"  Pooled shape: {pooled.shape}")
print(f"  NaN: {np.isnan(pooled).any()}")

# Load SAE and run forward pass
print("\nLoading SAE...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
state = sae_data["state_dict"]
W_enc = state["W_enc"]
W_dec = state["W_dec"]
b_enc = state["b_enc"]
b_dec = state["b_dec"]
k = sae_data["config"]["k"]

# Need normalization stats from training data
print("Loading training stats for normalization...")
train_data = torch.load(ACT_CACHE_PATH, map_location="cpu", weights_only=False)
train_acts = train_data["activations"].float()
train_mean = train_acts.mean(dim=0)
train_std = train_acts.std(dim=0).clamp(min=1e-6)
del train_acts

# Normalize Gemini positions using training stats
x = torch.from_numpy(pooled).float()
x = (x - train_mean) / train_std
norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
x = x / norms

# SAE forward
z = (x - b_dec) @ W_enc + b_enc
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"  Feature activations shape: {acts.shape}")
print(f"  L0: {(acts > 0).sum(-1).float().mean():.1f}")

# For each feature, find which Gemini-analyzed positions it fires on
print("\n" + "=" * 70)
print("TOP FEATURES BY FIRE COUNT ON GEMINI POSITIONS")
print("=" * 70)

feature_fires = {}
for feat_id in range(acts.shape[1]):
    firing_mask = acts[:, feat_id] > 0
    if firing_mask.sum() > 0:
        firing_idx = torch.where(firing_mask)[0].tolist()
        feature_fires[feat_id] = firing_idx

# Sort by fire count
sorted_features = sorted(feature_fires.items(), key=lambda x: len(x[1]), reverse=True)

# Show top 20 features with their Gemini analyses
for feat_id, firing_idx in sorted_features[:20]:
    fire_rate = len(firing_idx) / len(positions)
    print(f"\nFeature {feat_id}: fires on {len(firing_idx)}/{len(positions)} positions ({fire_rate:.1%})")
    # Show 3 example positions with their Gemini text
    for idx in firing_idx[:3]:
        p = positions[idx]
        strength = acts[idx, feat_id].item()
        # Truncate gemini text
        text = p["gemini_intent"][:120] if p["gemini_intent"] else p["gemini_blunder"][:120]
        print(f"  [{strength:.2f}] {p['fen'][:45]}... {p['uci']}")
        print(f"         {text}")

# Save full results for further analysis
results = {
    "n_positions": len(positions),
    "n_features_active": len(feature_fires),
    "features": {},
}
for feat_id, firing_idx in sorted_features[:100]:
    results["features"][str(feat_id)] = {
        "fire_count": len(firing_idx),
        "fire_rate": len(firing_idx) / len(positions),
        "examples": [
            {
                "fen": positions[i]["fen"],
                "uci": positions[i]["uci"],
                "strength": round(acts[i, feat_id].item(), 3),
                "gemini_intent": positions[i]["gemini_intent"][:200],
                "gemini_blunder": positions[i]["gemini_blunder"][:200],
            }
            for i in firing_idx[:10]
        ],
    }

output_path = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_validation.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to {output_path}")
