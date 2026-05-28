#!/usr/bin/env python3
"""Label Matryoshka SAE features using existing Opus position analyses.

Pipeline:
1. Load trained Matryoshka SAE + 200K activations
2. Run forward pass, normalize each feature to [0, 1] by per-feature max
3. For each feature, find positions with normalized activation > 0.7
4. Cross-reference with existing 19K Opus analyses (matched by position idx)
5. If >= 20 analyzed positions available, synthesize feature label

Usage (on chess-poc):
    python3 scripts/sae/label_matryoshka_features.py --model <path_to_matryoshka.pt>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
ANALYSES_PATH = "/home/ec2-user/SageMaker/all_positions_labeled_opus.json"
PROFILES_PATH = "/home/ec2-user/SageMaker/l2_feature_profiles_v2.json"
ACTIVATIONS_PATH = BASE + "/cache/maia3_blunder_diff.pt"

MIN_POSITIONS = 20
ACTIVATION_THRESHOLD = 0.7


def load_analyses_by_idx():
    """Load Opus analyses and build idx->analysis lookup from profiles."""
    # The profiles file has idx for each position
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    # Collect all unique idx -> fen|uci mappings
    idx_to_key = {}
    for feat_id, feat_data in profiles.items():
        examples = feat_data.get("examples", feat_data if isinstance(feat_data, list) else [])
        for ex in examples:
            if "idx" in ex and "fen" in ex and "uci" in ex:
                idx_to_key[ex["idx"]] = f"{ex['fen']}|{ex['uci']}"

    print(f"  Profiles: {len(idx_to_key)} unique positions with idx mapping")

    # Load analyses
    with open(ANALYSES_PATH) as f:
        analyses_raw = json.load(f)

    # Build idx -> analysis lookup
    # The analysis keys have full FENs with move counters
    # We need to match via the profiles which give us both idx and full FEN
    idx_to_analysis = {}
    for idx, key in idx_to_key.items():
        if key in analyses_raw:
            idx_to_analysis[idx] = analyses_raw[key]

    # Also try matching by stripping move counters from analysis keys
    # and matching to profiles
    print(f"  Direct key matches: {len(idx_to_analysis)}")

    return idx_to_analysis


def run_matryoshka_forward(model_path):
    """Load model and run all 200K positions through it."""
    print(f"Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sd = ckpt["state_dict"]

    We = sd.get("W_enc", sd.get("We"))
    Wd = sd.get("W_dec", sd.get("Wd"))
    be = sd.get("b_enc", sd.get("be"))
    bd = sd.get("b_dec", sd.get("bd"))
    dict_size = We.shape[1]

    prefixes = config.get("prefixes", [dict_size])
    k_per_level = config.get("k_per_level", None)
    k = config.get("k", config.get("total_k", 16))

    print(f"  Dict: {dict_size}, Prefixes: {prefixes}")
    if k_per_level:
        print(f"  k_per_level: {k_per_level}")
    else:
        print(f"  k (global): {k}")

    # Load activations
    print("Loading activations...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    raw_acts = data["activations"].float()
    fens = data["fens"]
    meta = data["metadata"]

    # Normalize
    mean = raw_acts.mean(dim=0)
    std = raw_acts.std(dim=0).clamp(min=1e-6)
    x = (raw_acts - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw_acts

    # Forward pass
    print("Running forward pass...")
    z = (x - bd) @ We + be
    z_relu = torch.relu(z)

    # Apply topk (global or per-level)
    n = x.shape[0]
    if k_per_level:
        acts = torch.zeros_like(z_relu)
        for i, (prefix_size, level_k) in enumerate(zip(prefixes, k_per_level)):
            start = 0 if i == 0 else prefixes[i - 1]
            end = prefix_size
            group_z = z_relu[:, start:end]
            flat_group = group_z.reshape(-1)
            group_k = min(level_k * n, flat_group.numel())
            topk_vals, topk_idx = torch.topk(flat_group, k=group_k)
            group_acts = torch.zeros_like(flat_group)
            group_acts[topk_idx] = topk_vals
            acts[:, start:end] = group_acts.reshape(n, end - start)
    else:
        flat = z_relu.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat, k=min(n * k, flat.numel()))
        acts = torch.zeros_like(flat)
        acts[topk_idx] = topk_vals
        acts = acts.reshape(z_relu.shape)

    # Normalize to [0, 1] per feature using per-feature max
    feature_max = acts.max(dim=0).values.clamp(min=1e-8)
    acts_normalized = acts / feature_max.unsqueeze(0)

    print(f"  Activations shape: {acts_normalized.shape}")
    print(f"  Features with any activation: {(feature_max > 1e-6).sum()}")

    return acts_normalized.numpy(), fens, meta, config


def find_labelable_features(acts_normalized, idx_to_analysis):
    """Find features with >= MIN_POSITIONS analyzed positions above threshold."""
    n_positions, n_features = acts_normalized.shape
    analyzed_indices = set(idx_to_analysis.keys())

    results = []
    for feat_idx in range(n_features):
        # Find positions above threshold
        above_threshold = np.where(acts_normalized[:, feat_idx] > ACTIVATION_THRESHOLD)[0]

        # Which of those have analyses?
        analyzed_above = [idx for idx in above_threshold if idx in analyzed_indices]

        results.append({
            "feature_idx": feat_idx,
            "n_above_threshold": len(above_threshold),
            "n_with_analysis": len(analyzed_above),
            "labelable": len(analyzed_above) >= MIN_POSITIONS,
            "analyzed_indices": analyzed_above[:20],  # Keep top 20 for labeling
        })

    n_labelable = sum(1 for r in results if r["labelable"])
    print(f"\n  Features above {ACTIVATION_THRESHOLD} threshold:")
    print(f"    Total with any: {sum(1 for r in results if r['n_above_threshold'] > 0)}")
    print(f"    With >= {MIN_POSITIONS} analyzed positions: {n_labelable}/{n_features}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to Matryoshka .pt file")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-positions", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    global ACTIVATION_THRESHOLD, MIN_POSITIONS
    ACTIVATION_THRESHOLD = args.threshold
    MIN_POSITIONS = args.min_positions

    # Load analyses
    print("Loading Opus analyses...")
    idx_to_analysis = load_analyses_by_idx()
    print(f"  Total indexed analyses: {len(idx_to_analysis)}")

    # Run model
    acts_normalized, fens, meta, config = run_matryoshka_forward(args.model)

    # Find labelable features
    feature_results = find_labelable_features(acts_normalized, idx_to_analysis)

    # Summary by prefix level
    prefixes = config.get("prefixes", [acts_normalized.shape[1]])
    print("\n  Per-level labelable features:")
    for i, ps in enumerate(prefixes):
        start = 0 if i == 0 else prefixes[i - 1]
        level_results = [r for r in feature_results if start <= r["feature_idx"] < ps]
        n_labelable = sum(1 for r in level_results if r["labelable"])
        print(f"    Prefix {ps} [{start}:{ps}]: {n_labelable}/{ps - start} labelable")

    # Save results
    output_path = args.output or args.model.replace(".pt", "_labeling_coverage.json")
    save_data = {
        "config": config,
        "threshold": ACTIVATION_THRESHOLD,
        "min_positions": MIN_POSITIONS,
        "n_analyses_available": len(idx_to_analysis),
        "features": feature_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
