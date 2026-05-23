#!/usr/bin/env python3
"""Profile Maia 3 SAE features — find top positions per feature."""
import torch
import torch.nn.functional as F
import json
import sys

SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32.pt"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"
OUTPUT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/feature_profiles.json"

print("Loading SAE and activations...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
act_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)

config = sae_data["config"]
print(f"  SAE: dict_size={config['dict_size']}, k={config['k']}")
print(f"  Activations: {act_data['activations'].shape}")

state = sae_data["state_dict"]
W_enc = state["W_enc"]
W_dec = state["W_dec"]
b_enc = state["b_enc"]
b_dec = state["b_dec"]
k = config["k"]

# Normalize same as training
x = act_data["activations"].float()
mean = x.mean(dim=0)
std = x.std(dim=0).clamp(min=1e-6)
x = (x - mean) / std
norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
x = x / norms

# Forward pass — per-sample top-k for profiling
print("Running SAE forward pass...")
z = (x - b_dec) @ W_enc + b_enc
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"  L0: {(acts > 0).sum(-1).float().mean():.1f}")

# Profile each feature
metadata = act_data["metadata"]
n_features = acts.shape[1]
print(f"Profiling {n_features} features...")

profiles = {}
for feat_id in range(n_features):
    feat_acts = acts[:, feat_id]
    if feat_acts.max() == 0:
        continue
    top_idx = torch.argsort(feat_acts, descending=True)[:20]
    examples = []
    for idx in top_idx:
        idx = idx.item()
        if feat_acts[idx] > 0:
            m = metadata[idx]
            examples.append({
                "idx": idx,
                "fen": m["fen"],
                "uci": m.get("blunder_uci", ""),
                "strength": round(feat_acts[idx].item(), 4),
                "cp_loss": m.get("cp_loss", 0),
            })
    profiles[str(feat_id)] = {
        "fire_count": int((feat_acts > 0).sum()),
        "fire_rate": round(float((feat_acts > 0).float().mean()), 5),
        "mean_strength": round(float(feat_acts[feat_acts > 0].mean()), 4),
        "top_examples": examples,
    }

print(f"  Active features: {len(profiles)}")

# Show sample features
sample_ids = [0, 100, 500, 1000, 1500, 2000]
for feat_id in sample_ids:
    key = str(feat_id)
    if key in profiles:
        p = profiles[key]
        print(f"\nFeature {feat_id}: fire_rate={p['fire_rate']:.4f} ({p['fire_count']} positions)")
        for ex in p["top_examples"][:5]:
            print(f"  {ex['fen'][:55]}  uci={ex['uci']}  cp={ex['cp_loss']}")

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(profiles, f)
print(f"\nSaved {len(profiles)} feature profiles to {OUTPUT_PATH}")
