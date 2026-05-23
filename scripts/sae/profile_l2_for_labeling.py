#!/usr/bin/env python3
"""Profile L2 SAE and prep positions for Gemini labeling.

Uses the L2 200-epoch SAE. For each feature, saves top-20 positions with
relevance scores (activation / feature_max), matching Sandstone's metadata format.
"""
import torch
import torch.nn.functional as F
import json
import numpy as np

SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_l2_200ep.pt"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"
PROFILE_OUTPUT = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json"
LABELING_OUTPUT = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_positions_for_gemini.json"

print("Loading...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
act_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)

state = sae_data["state_dict"]
k = sae_data["config"]["k"]

# Apply same normalization as training: Z-score + L2
x = act_data["activations"].float()
mean = x.mean(dim=0)
std = x.std(dim=0).clamp(min=1e-6)
x = (x - mean) / std
norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
x = x / norms

# Forward pass
print("Running SAE forward pass on 200K positions...")
z = (x - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"  Max activation: {acts.max():.4f}")
print(f"  Mean (nonzero): {acts[acts > 0].mean():.4f}")

# Compute per-feature max (for relevance normalization)
feat_maxes = acts.max(dim=0).values
print(f"  Feature max range: [{feat_maxes[feat_maxes > 0].min():.4f}, {feat_maxes.max():.4f}]")

# Profile each feature
metadata = act_data["metadata"]
n_features = acts.shape[1]

profiles = {}
all_positions_needed = {}  # fen -> {uci, cp_loss, feature_ids}

n_labelable = 0
for feat_id in range(n_features):
    feat_acts = acts[:, feat_id]
    if feat_acts.max() == 0:
        continue

    fire_count = (feat_acts > 0).sum().item()
    fire_rate = fire_count / len(feat_acts)
    feat_max = feat_maxes[feat_id].item()

    # Top 20 positions
    top_idx = torch.argsort(feat_acts, descending=True)[:20]
    examples = []
    n_above_07 = 0
    for idx in top_idx:
        idx = idx.item()
        if feat_acts[idx] <= 0:
            break
        m = metadata[idx]
        relevance = feat_acts[idx].item() / feat_max
        examples.append({
            "idx": idx,
            "fen": m["fen"],
            "uci": m.get("blunder_uci", ""),
            "activation": round(feat_acts[idx].item(), 5),
            "relevance": round(relevance, 4),
            "cp_loss": m.get("cp_loss", 0),
        })
        if relevance >= 0.7:
            n_above_07 += 1
            # Track for Gemini
            fen = m["fen"]
            if fen not in all_positions_needed:
                all_positions_needed[fen] = {
                    "fen": fen,
                    "uci": m.get("blunder_uci", ""),
                    "cp_loss": m.get("cp_loss", 0),
                    "feature_ids": [],
                }
            all_positions_needed[fen]["feature_ids"].append(feat_id)

    profiles[str(feat_id)] = {
        "fire_count": fire_count,
        "fire_rate": round(fire_rate, 5),
        "feat_max": round(feat_max, 5),
        "n_above_07_relevance": n_above_07,
        "top_examples": examples,
    }

    if n_above_07 >= 5:
        n_labelable += 1

print(f"\nProfiled {len(profiles)} features")
print(f"Labelable (5+ positions above 0.7 relevance): {n_labelable}")
print(f"Unique positions for Gemini: {len(all_positions_needed)}")

# Save profiles
with open(PROFILE_OUTPUT, "w") as f:
    json.dump(profiles, f)
print(f"Saved profiles to {PROFILE_OUTPUT}")

# Save positions for Gemini
positions_list = list(all_positions_needed.values())
with open(LABELING_OUTPUT, "w") as f:
    json.dump(positions_list, f)
print(f"Saved {len(positions_list)} positions for Gemini to {LABELING_OUTPUT}")

# Summary
fire_rates = np.array([p["fire_rate"] for p in profiles.values()])
print(f"\nFire rate distribution:")
print(f"  Hubs (>10%): {(fire_rates > 0.10).sum()}")
print(f"  5-10%: {((fire_rates > 0.05) & (fire_rates <= 0.10)).sum()}")
print(f"  0.5-3%: {((fire_rates >= 0.005) & (fire_rates <= 0.03)).sum()}")
print(f"  <0.5%: {(fire_rates < 0.005).sum()}")

# Show sample features
print("\n" + "=" * 60)
for feat_id in ["0", "100", "500", "1000"]:
    if feat_id in profiles:
        p = profiles[feat_id]
        print(f"\nFeature {feat_id}: fire_rate={p['fire_rate']:.4f}, n_above_0.7={p['n_above_07_relevance']}")
        for ex in p["top_examples"][:3]:
            print(f"  [rel={ex['relevance']:.2f}] {ex['fen'][:50]}  {ex['uci']}  cp={ex['cp_loss']}")
