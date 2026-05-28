#!/usr/bin/env python3
"""Test Elo discrimination across all Matryoshka configs.

Runs 56K rating-band positions (6 bands, ~10K each) through each trained
Matryoshka SAE and measures per-prefix:
  - % features that vary significantly by rating band
  - Mean Cohen's d between lowest and highest band
  - Which prefix level captures the most Elo signal

Usage (on chess-poc):
    cd ~/SageMaker && python3 scripts/sae/matryoshka_elo_test.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
OUTPUT = BASE + "/output/maia3_sae"

MODELS = [
    # Matryoshka configs
    ("A", f"{OUTPUT}/maia3_matryoshka_2048_k16_p64_256_2048.pt", [64, 256, 2048], 16),
    ("B", f"{OUTPUT}/maia3_matryoshka_2048_k16_p32_128_512_2048.pt", [32, 128, 512, 2048], 16),
    ("E", f"{OUTPUT}/maia3_matryoshka_2048_k16_p32_96_224_480_992_2048.pt", [32, 96, 224, 480, 992, 2048], 16),
    ("C2", f"{OUTPUT}/maia3_matryoshka_2720_k22_p32_160_672_2720.pt", [32, 160, 672, 2720], 22),
    ("C3", f"{OUTPUT}/maia3_matryoshka_2720_k24_p32_160_672_2720.pt", [32, 160, 672, 2720], 24),
    ("F", f"{OUTPUT}/maia3_matryoshka_2336_k20_p32_288_2336.pt", [32, 288, 2336], 20),
    # Baselines (standard BatchTopK, no Matryoshka)
    ("OG_k32", "s3_maia3", [2048], 32),  # original production SAE
    ("STD_k16", f"{OUTPUT}/sweep_k16_d2048.pt", [2048], 16),  # k-sweep winner
]

BANDS = ["1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200"]


def load_rating_positions():
    """Load the 56K rating-stratified positions used in previous validation."""
    path = BASE + "/cache/sae_rating_validation.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data

    # Alternative: use sweep_blunders_2000.json
    sweep_path = "/home/ec2-user/SageMaker/sweep_blunders_2000.json"
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            return json.load(f)

    print("ERROR: No rating validation data found")
    sys.exit(1)


def get_activations_for_positions(model_path, positions_by_band, k, prefixes):
    """Run positions through Maia 3 -> SAE and get activations per band.

    Since we can't run Maia 3 here easily, we'll use the approach of
    loading the cached diff activations and filtering by the rating bands
    from the original validation run.
    """
    # The rating validation was already done on the 200K positions
    # Each position has a rating band. We need the activation indices.
    #
    # Actually - the 56K sweep positions are a SUBSET of the 200K training data
    # (they were drawn from the same Lichess blunder pool with rating labels).
    # The cached activations at maia3_blunder_diff.pt don't have rating labels.
    #
    # We need the pre-computed per-band activations from the previous session.
    # Those live in sae_rating_validation.json (fire rates only, not raw acts).
    #
    # Better approach: load the full 200K activations + metadata to get ratings.
    pass


def run_elo_test_on_cached_acts(model_path, k, prefixes, tag):
    """Use the 200K cached activations with rating metadata."""

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    We = sd.get("W_enc", sd.get("We"))
    Wd = sd.get("W_dec", sd.get("Wd"))
    be = sd.get("b_enc", sd.get("be"))
    bd = sd.get("b_dec", sd.get("bd"))
    dict_size = We.shape[1]

    # Load activations
    data = torch.load(BASE + "/cache/maia3_blunder_diff.pt", map_location="cpu", weights_only=False)
    raw_acts = data["activations"].float()

    # Normalize
    mean = raw_acts.mean(dim=0)
    std = raw_acts.std(dim=0).clamp(min=1e-6)
    x = (raw_acts - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw_acts

    # Ratings are in elo_self field
    ratings = data.get("elo_self")
    if ratings is None:
        print("  ERROR: No elo_self in activation file")
        return None

    # Assign bands
    ratings = np.array(ratings[:len(x)])
    band_masks = {}
    for band in BANDS:
        lo, hi = band.split("-")
        lo, hi = int(lo), int(hi)
        mask = (ratings >= lo) & (ratings < hi)
        band_masks[band] = mask

    print(f"  Positions per band: {', '.join(f'{b}={m.sum()}' for b, m in band_masks.items())}")
    sys.stdout.flush()

    # Forward pass
    z = (x - bd) @ We + be
    z_relu = torch.relu(z)
    flat = z_relu.reshape(-1)
    topk_vals, topk_idx = torch.topk(flat, k=min(x.shape[0] * k, flat.numel()))
    acts = torch.zeros_like(flat)
    acts[topk_idx] = topk_vals
    acts = acts.reshape(z_relu.shape)
    acts_binary = (acts > 0).numpy()

    # Per-prefix Elo discrimination
    results = {"tag": tag, "prefixes": prefixes, "levels": {}}

    for prefix_size in prefixes:
        prefix_acts = acts_binary[:, :prefix_size]

        # Per-feature fire rate by band
        band_rates = {}
        for band, mask in band_masks.items():
            if mask.sum() > 0:
                band_rates[band] = prefix_acts[mask].mean(axis=0)

        # Metrics per feature
        n_varying = 0
        cohens_ds = []

        for feat_idx in range(prefix_size):
            rates = [band_rates[b][feat_idx] for b in BANDS if b in band_rates]
            if len(rates) < 2:
                continue

            lo_rate = (band_rates.get("1000-1200", np.zeros(1))[feat_idx] +
                      band_rates.get("1200-1400", np.zeros(1))[feat_idx]) / 2
            hi_rate = (band_rates.get("1800-2000", np.zeros(1))[feat_idx] +
                      band_rates.get("2000-2200", np.zeros(1))[feat_idx]) / 2

            # Feature varies if max/min ratio > 1.5 or difference > 0.01
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate > 0.001:
                ratio = max_rate / max(min_rate, 0.0001)
                if ratio > 1.5 or (max_rate - min_rate) > 0.01:
                    n_varying += 1

            # Cohen's d between low and high bands
            if lo_rate > 0 or hi_rate > 0:
                pooled_rate = (lo_rate + hi_rate) / 2
                pooled_std = max(np.sqrt(pooled_rate * (1 - pooled_rate)), 0.001)
                d = abs(hi_rate - lo_rate) / pooled_std
                cohens_ds.append(d)

        pct_varying = n_varying / prefix_size * 100 if prefix_size > 0 else 0
        mean_d = float(np.mean(cohens_ds)) if cohens_ds else 0

        level_result = {
            "n_varying": n_varying,
            "pct_varying": pct_varying,
            "mean_cohens_d": mean_d,
            "n_features": prefix_size,
        }
        results["levels"][prefix_size] = level_result

        print(f"  Prefix {prefix_size:>4}: {pct_varying:.0f}% vary by Elo "
              f"({n_varying}/{prefix_size}) | mean Cohen's d={mean_d:.3f}")

    sys.stdout.flush()
    return results


def main():
    print("=" * 60)
    print("MATRYOSHKA ELO DISCRIMINATION TEST")
    print("=" * 60)
    sys.stdout.flush()

    # First check what data we have
    data = torch.load(BASE + "/cache/maia3_blunder_diff.pt", map_location="cpu", weights_only=False)
    print(f"Activation file keys: {list(data.keys()) if isinstance(data, dict) else 'tensor'}")
    if isinstance(data, dict):
        for k_name in data.keys():
            v = data[k_name]
            if isinstance(v, torch.Tensor):
                print(f"  {k_name}: tensor {v.shape}")
            elif isinstance(v, (list, dict)):
                print(f"  {k_name}: {type(v).__name__} len={len(v)}")
            else:
                print(f"  {k_name}: {type(v).__name__} = {str(v)[:100]}")
    sys.stdout.flush()

    all_results = {}
    for tag, path, prefixes, k in MODELS:
        if not os.path.exists(path):
            print(f"\n--- {tag}: FILE NOT FOUND ({path}) ---")
            continue
        print(f"\n--- {tag}: prefixes={prefixes}, k={k} ---")
        sys.stdout.flush()
        result = run_elo_test_on_cached_acts(path, k, prefixes, tag)
        if result:
            all_results[tag] = result

    # Summary comparison
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY: Elo Discrimination by Config")
        print("=" * 60)
        print(f"{'Config':<6} | {'Prefix':<8} | {'% Varying':<10} | {'Cohen d':<8}")
        print("-" * 45)
        for tag, res in all_results.items():
            for psize, lv in res["levels"].items():
                print(f"{tag:<6} | {psize:<8} | {lv['pct_varying']:>7.0f}%   | {lv['mean_cohens_d']:.3f}")
            print("-" * 45)

    # Save
    out_path = f"{OUTPUT}/matryoshka_elo_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
