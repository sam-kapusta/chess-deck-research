#!/usr/bin/env python3
"""Run existing Gemini-labeled positions through the raw SAE.

Takes the 5,568 positions that already have Gemini tactical analysis,
extracts Maia 3 diff activations, runs through raw SAE, and checks
which features fire strongly on which labeled positions.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from maia3_activations import extract_activations, pool_activations

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_raw.pt"

# Load Gemini analyses
print("Loading Gemini position analyses...")
with open(GEMINI_PATH) as f:
    gemini_data = json.load(f)

# Extract unique positions with UCI and Gemini text
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
            })

print(f"  {len(positions)} unique positions with Gemini analysis + UCI")

# Extract Maia 3 activations
fens = [p["fen"] for p in positions]
ucis = [p["uci"] for p in positions]

print(f"Extracting Maia 3 activations...")
rng = np.random.default_rng(42)
elos = rng.integers(600, 2601, size=len(fens)).tolist()
raw, mirrored = extract_activations(fens, elo_self=elos, elo_oppo=elos)
pooled = pool_activations(raw, "diff", ucis, mirrored)
print(f"  Pooled shape: {pooled.shape}, NaN: {np.isnan(pooled).any()}")

# Load raw SAE (no normalization)
print("Loading SAE...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
state = sae_data["state_dict"]
k = sae_data["config"]["k"]

# Forward pass — raw input, no normalization (matching training)
x = torch.from_numpy(pooled).float()
z = (x - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

print(f"  Max activation: {acts.max():.2f}")
print(f"  Mean (nonzero): {acts[acts > 0].mean():.2f}")
print(f"  >1.0: {(acts.max(dim=0).values > 1.0).sum()} features")
print(f"  >3.0: {(acts.max(dim=0).values > 3.0).sum()} features")
print(f"  >5.0: {(acts.max(dim=0).values > 5.0).sum()} features")

# For each feature, find its top Gemini-labeled positions
print("\n" + "=" * 70)
print("TOP FEATURES WITH STRONGEST ACTIVATIONS ON GEMINI POSITIONS")
print("=" * 70)

# Find features sorted by their max activation on these positions
feat_maxes = acts.max(dim=0).values
top_features = torch.argsort(feat_maxes, descending=True)[:30]

for feat_id in top_features:
    feat_id = feat_id.item()
    feat_acts = acts[:, feat_id]
    max_val = feat_acts.max().item()
    fire_count = (feat_acts > 0).sum().item()

    if max_val < 1.0:
        break

    # Get top 3 positions for this feature
    top_pos_idx = torch.argsort(feat_acts, descending=True)[:3]
    print(f"\nFeature {feat_id}: max={max_val:.2f}, fires on {fire_count}/{len(positions)}")
    for idx in top_pos_idx:
        idx = idx.item()
        if feat_acts[idx] <= 0:
            break
        p = positions[idx]
        strength = feat_acts[idx].item()
        intent = p["gemini_intent"][:120] if p["gemini_intent"] else ""
        blunder = p["gemini_blunder"][:120] if p["gemini_blunder"] else ""
        print(f"  [{strength:.2f}] {p['fen'][:50]}  {p['uci']}")
        if intent:
            print(f"         Intent: {intent}")
        if blunder:
            print(f"         Blunder: {blunder}")

# Summary: how many features have at least 5 positions with activation > 2.0?
print("\n" + "=" * 70)
print("COVERAGE SUMMARY")
print("=" * 70)
for thresh in [1.0, 2.0, 3.0, 5.0]:
    n_features = 0
    for feat_id in range(acts.shape[1]):
        n_strong = (acts[:, feat_id] > thresh).sum().item()
        if n_strong >= 5:
            n_features += 1
    print(f"  Features with 5+ positions > {thresh}: {n_features}")

# Save results
output_path = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_raw_validation.json"
results = {"n_positions": len(positions), "n_features_with_5_above_2": 0}
feat_results = {}
for feat_id in range(acts.shape[1]):
    feat_acts = acts[:, feat_id]
    strong_idx = (feat_acts > 1.0).nonzero(as_tuple=True)[0]
    if len(strong_idx) >= 3:
        examples = []
        for idx in strong_idx[:10]:
            idx = idx.item()
            p = positions[idx]
            examples.append({
                "strength": round(feat_acts[idx].item(), 3),
                "fen": p["fen"],
                "uci": p["uci"],
                "gemini_intent": p["gemini_intent"][:200],
                "gemini_blunder": p["gemini_blunder"][:200],
            })
        feat_results[str(feat_id)] = {
            "max": round(feat_acts.max().item(), 3),
            "n_strong": len(strong_idx),
            "examples": sorted(examples, key=lambda x: -x["strength"]),
        }

results["features"] = feat_results
results["n_features_with_3_above_1"] = len(feat_results)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {output_path}")
