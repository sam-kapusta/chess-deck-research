#!/usr/bin/env python3
"""Profile Maia 3 SAE (raw normalization) and prep positions for Gemini labeling."""
import torch
import torch.nn.functional as F
import json
import numpy as np

SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_raw.pt"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"
PROFILE_OUTPUT = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/raw_feature_profiles.json"
LABELING_OUTPUT = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/positions_for_gemini.json"

print("Loading SAE and activations...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
act_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)

state = sae_data["state_dict"]
W_enc = state["W_enc"]
W_dec = state["W_dec"]
b_enc = state["b_enc"]
b_dec = state["b_dec"]
k = sae_data["config"]["k"]

# Raw activations (no normalization — matching training)
x = act_data["activations"].float()
print(f"  Input: {x.shape}, norm mean={x.norm(dim=-1).mean():.1f}")

# Forward pass (per-sample top-k for profiling)
print("Running SAE forward pass...")
z = (x - b_dec) @ W_enc + b_enc
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"  Max activation: {acts.max():.2f}")
print(f"  Mean (nonzero): {acts[acts > 0].mean():.2f}")

# Profile all features
metadata = act_data["metadata"]
n_features = acts.shape[1]

profiles = {}
positions_for_labeling = []
seen_positions = set()

for feat_id in range(n_features):
    feat_acts = acts[:, feat_id]
    fire_mask = feat_acts > 0
    if not fire_mask.any():
        continue

    fire_count = fire_mask.sum().item()
    fire_rate = fire_count / len(feat_acts)
    mean_strength = feat_acts[fire_mask].mean().item()

    # Top 20 positions
    top_idx = torch.argsort(feat_acts, descending=True)[:20]
    examples = []
    for idx in top_idx:
        idx = idx.item()
        if feat_acts[idx] <= 0:
            break
        m = metadata[idx]
        examples.append({
            "idx": idx,
            "fen": m["fen"],
            "uci": m.get("blunder_uci", ""),
            "strength": round(feat_acts[idx].item(), 4),
            "cp_loss": m.get("cp_loss", 0),
        })
        # Collect unique positions for Gemini analysis
        if m["fen"] not in seen_positions:
            seen_positions.add(m["fen"])
            positions_for_labeling.append({
                "fen": m["fen"],
                "uci": m.get("blunder_uci", ""),
                "cp_loss": m.get("cp_loss", 0),
                "feature_ids": [feat_id],
            })
        else:
            # Add this feature to existing position entry
            for p in positions_for_labeling:
                if p["fen"] == m["fen"]:
                    p["feature_ids"].append(feat_id)
                    break

    profiles[str(feat_id)] = {
        "fire_count": fire_count,
        "fire_rate": round(fire_rate, 5),
        "mean_strength": round(mean_strength, 4),
        "top_examples": examples,
    }

print(f"\nProfiled {len(profiles)} active features")
print(f"Unique positions for Gemini labeling: {len(positions_for_labeling)}")

# Save profiles
with open(PROFILE_OUTPUT, "w") as f:
    json.dump(profiles, f)
print(f"Saved profiles to {PROFILE_OUTPUT}")

# Save positions for Gemini (deduplicated)
with open(LABELING_OUTPUT, "w") as f:
    json.dump(positions_for_labeling, f)
print(f"Saved {len(positions_for_labeling)} positions for Gemini to {LABELING_OUTPUT}")

# Summary stats
fire_rates = np.array([p["fire_rate"] for p in profiles.values()])
max_acts = np.array([p["top_examples"][0]["strength"] if p["top_examples"] else 0 for p in profiles.values()])
print(f"\nFire rate distribution:")
print(f"  >10%: {(fire_rates > 0.10).sum()} (hubs)")
print(f"  5-10%: {((fire_rates > 0.05) & (fire_rates <= 0.10)).sum()}")
print(f"  0.5-3%: {((fire_rates >= 0.005) & (fire_rates <= 0.03)).sum()}")
print(f"\nMax activation distribution:")
print(f"  Median: {np.median(max_acts):.2f}")
print(f"  >5.0: {(max_acts > 5).sum()}")
print(f"  >10.0: {(max_acts > 10).sum()}")

# Show a few sample features
print("\n" + "=" * 60)
print("SAMPLE FEATURES")
print("=" * 60)
for feat_id in ["0", "100", "500", "1000", "1500"]:
    if feat_id in profiles:
        p = profiles[feat_id]
        print(f"\nFeature {feat_id}: fire_rate={p['fire_rate']:.4f}, top examples:")
        for ex in p["top_examples"][:3]:
            print(f"  [{ex['strength']:.2f}] {ex['fen'][:50]}  {ex['uci']}  cp={ex['cp_loss']}")
