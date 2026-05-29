#!/usr/bin/env python3
"""Extract top-activating positions with verified Opus analyses for labeling.

Builds a verified idx->analysis lookup by confirming FEN matches between
the 200K activation file and the Opus analysis keys. Then extracts top
positions per feature for synthesis labeling.

Usage (on chess-poc):
    python3 scripts/sae/extract_for_labeling.py \
        --model chess-stage-a/output/maia3_sae/maia3_matryoshka_perlevel_2336_p32_288_2336_k3_8_16.pt \
        --n-features 32 \
        --output matryoshka_top32_verified.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
ACTIVATIONS_PATH = BASE + "/cache/maia3_blunder_diff.pt"
OPUS_PATH = "/home/ec2-user/SageMaker/all_positions_labeled_opus.json"
PROFILES_PATH = "/home/ec2-user/SageMaker/l2_feature_profiles_v2.json"
VERIFIED_CACHE = "/home/ec2-user/SageMaker/verified_idx_to_analysis.pkl"


def strip_fen(fen):
    parts = fen.split(" ")
    return " ".join(parts[:4]) if len(parts) >= 4 else fen


def build_verified_lookup(fens, meta, opus, profiles):
    """Build idx->analysis with FEN verification."""
    if os.path.exists(VERIFIED_CACHE):
        print(f"  Loading cached lookup from {VERIFIED_CACHE}")
        with open(VERIFIED_CACHE, "rb") as f:
            return pickle.load(f)

    print("  Building verified idx->analysis lookup...")
    idx_to_analysis = {}
    mismatches = 0
    total = 0

    for feat_data in profiles.values():
        for ex in feat_data.get("examples", []):
            idx = ex.get("idx")
            if idx is None or idx >= len(fens):
                continue
            total += 1
            prof_fen = ex.get("fen", "")
            prof_uci = ex.get("uci", "")
            our_fen = fens[idx]
            our_uci = meta[idx]["blunder_uci"]

            # Verify: stripped profile FEN must match 200K FEN
            if strip_fen(prof_fen) != our_fen:
                mismatches += 1
                continue

            # Verify: UCI must match
            if prof_uci != our_uci:
                mismatches += 1
                continue

            # Look up in opus
            opus_key = prof_fen + "|" + prof_uci
            if opus_key in opus:
                idx_to_analysis[idx] = opus[opus_key]

    print(f"  Verified: {len(idx_to_analysis)} positions "
          f"(checked {total}, mismatches {mismatches})")

    with open(VERIFIED_CACHE, "wb") as f:
        pickle.dump(idx_to_analysis, f)
    print(f"  Cached to {VERIFIED_CACHE}")

    return idx_to_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-features", type=int, default=32,
                        help="Number of top-level features to extract")
    parser.add_argument("--examples-per-feature", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="extracted_for_labeling.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    if args.rebuild_cache and os.path.exists(VERIFIED_CACHE):
        os.remove(VERIFIED_CACHE)

    # Load activations + metadata
    print("Loading activations...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    raw_acts = data["activations"].float()
    fens = data["fens"]
    meta = data["metadata"]

    mean = raw_acts.mean(dim=0)
    std = raw_acts.std(dim=0).clamp(min=1e-6)
    x = (raw_acts - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw_acts

    # Load opus + profiles
    print("Loading analyses...")
    with open(OPUS_PATH) as f:
        opus = json.load(f)
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    # Build verified lookup
    idx_to_analysis = build_verified_lookup(fens, meta, opus, profiles)

    # Load model
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    We = sd.get("W_enc", sd.get("We"))
    Wd = sd.get("W_dec", sd.get("Wd"))
    be = sd.get("b_enc", sd.get("be"))
    bd = sd.get("b_dec", sd.get("bd"))
    config = ckpt["config"]

    prefixes = config["prefixes"]
    k_per_level = config.get("k_per_level", [config.get("k", 16)] * len(prefixes))
    print(f"  Prefixes: {prefixes}, k_per_level: {k_per_level}")

    # Forward pass with per-level topk
    print("Running forward pass...")
    z = F.relu((x - bd) @ We + be)
    n = x.shape[0]
    acts = torch.zeros_like(z)
    for i, (ps, k) in enumerate(zip(prefixes, k_per_level)):
        s = 0 if i == 0 else prefixes[i - 1]
        g = z[:, s:ps]
        fg = g.reshape(-1)
        gk = min(k * n, fg.numel())
        tv, ti = torch.topk(fg, k=gk)
        ga = torch.zeros_like(fg)
        ga[ti] = tv
        acts[:, s:ps] = ga.reshape(n, ps - s)

    # Normalize to [0, 1]
    feat_max = acts.max(dim=0).values.clamp(min=1e-8)
    acts_norm = acts / feat_max.unsqueeze(0)

    # Extract for labeling
    print(f"Extracting top {args.examples_per_feature} examples for {args.n_features} features...")
    output = {}
    n_with_enough = 0

    for fi in range(args.n_features):
        feat_acts = acts_norm[:, fi]
        above = torch.where(feat_acts > args.threshold)[0]
        sorted_above = above[torch.argsort(feat_acts[above], descending=True)]

        examples = []
        for idx in sorted_above.tolist():
            if idx in idx_to_analysis and len(examples) < args.examples_per_feature:
                analysis_data = idx_to_analysis[idx]
                # Flatten nested analysis structure
                if "analysis" in analysis_data and isinstance(analysis_data["analysis"], dict):
                    analysis = analysis_data["analysis"]
                else:
                    analysis = analysis_data

                examples.append({
                    "idx": int(idx),
                    "activation": round(float(feat_acts[idx]), 4),
                    "fen": fens[idx],
                    "uci": meta[idx]["blunder_uci"],
                    "cp_loss": int(meta[idx].get("cp_loss", 0)),
                    "analysis": analysis,
                })

        has_enough = len(examples) >= args.examples_per_feature
        if has_enough:
            n_with_enough += 1

        output[str(fi)] = {
            "n_above_threshold": int(len(above)),
            "n_examples": len(examples),
            "has_enough": has_enough,
            "examples": examples,
        }

        if fi % 8 == 0:
            print(f"  F{fi:3d}: {len(above):5d} above {args.threshold}, "
                  f"{len(examples)} verified examples")

    print(f"\n  Features with >= {args.examples_per_feature} examples: "
          f"{n_with_enough}/{args.n_features}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
