#!/usr/bin/env python3
"""Elo sensitivity test for Maia 3 SAE.

Takes a trained SAE and a set of positions, runs them through Maia 3 at
multiple Elo levels, and compares which features fire at each Elo.

Expected results:
- Tactical features (hanging piece, fork) fire MORE at low Elo
- Some features should be Elo-invariant (positional structure)
- If features DON'T change with Elo → SAE learned position structure, not blind spots

Usage (on chess-poc GPU):
    python scripts/sae/elo_sensitivity.py \
      --sae ~/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_from-square_2048_k32.pt \
      --positions ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
      --elos 1200,1500,1800 --limit 100
"""
import argparse
import json
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maia3_activations import extract_activations, pool_activations


class BatchTopKSAE(torch.nn.Module):
    """Minimal inference-only BatchTopK SAE."""

    def __init__(self, d_input, d_hidden, k):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.empty(d_input, d_hidden))
        self.W_dec = torch.nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden
        self.k = k
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        batch_size = x.shape[0]
        total_k = batch_size * self.k
        z_relu = F.relu(z)
        flat_z = z_relu.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat_z, k=min(int(total_k), flat_z.numel()))
        acts = torch.zeros_like(flat_z)
        acts[topk_idx] = topk_vals
        acts = acts.reshape(z.shape)
        return acts


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def main():
    parser = argparse.ArgumentParser(description="Elo sensitivity test for Maia 3 SAE")
    parser.add_argument("--sae", type=str, required=True, help="Path to trained SAE .pt")
    parser.add_argument("--positions", type=str, required=True,
                        help="Path to blunder cache .pt (has fens + UCIs)")
    parser.add_argument("--elos", type=str, default="1200,1500,1800",
                        help="Comma-separated Elo levels to test")
    parser.add_argument("--limit", type=int, default=100, help="Number of positions to test")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    elos = [int(e) for e in args.elos.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAE
    print(f"Loading SAE from {args.sae}")
    sae_data = torch.load(args.sae, map_location="cpu", weights_only=False)
    sae_config = sae_data["config"]
    pool_mode = sae_config["pool_mode"]

    model = BatchTopKSAE(
        d_input=sae_config["d_input"],
        d_hidden=sae_config["dict_size"],
        k=sae_config["k"],
    )
    model.load_state_dict(sae_data["state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"  Loaded: d_input={sae_config['d_input']}, dict_size={sae_config['dict_size']}, k={sae_config['k']}")

    # Load positions
    print(f"Loading positions from {args.positions}")
    cache = torch.load(args.positions, map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    fens = [m["fen"] for m in metadata[:args.limit]]
    ucis = [m.get("blunder_uci", m.get("uci", "")) for m in metadata[:args.limit]]
    print(f"  Using {len(fens)} positions")

    # Run at each Elo
    results_by_elo = {}
    for elo in elos:
        print(f"\nExtracting at Elo {elo}...")
        raw = extract_activations(fens, elo_self=elo, elo_oppo=elo)
        pooled = pool_activations(raw, pool_mode, ucis if pool_mode == "from-square" else None)
        pooled_norm = l2_normalize(pooled)

        # Run through SAE
        x = torch.tensor(pooled_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            acts = model(x).cpu().numpy()

        # Per-feature fire rate
        fire_rate = (acts > 0).mean(axis=0)
        results_by_elo[elo] = {
            "fire_rates": fire_rate,
            "mean_activation": acts.mean(axis=0),
            "l0": float((acts > 0).sum(axis=1).mean()),
        }
        print(f"  L0: {results_by_elo[elo]['l0']:.1f}")
        print(f"  Features with >1% fire rate: {int((fire_rate > 0.01).sum())}")

    # Compare across Elos
    print("\n" + "=" * 60)
    print("ELO SENSITIVITY ANALYSIS")
    print("=" * 60)

    elo_low, elo_high = min(elos), max(elos)
    fr_low = results_by_elo[elo_low]["fire_rates"]
    fr_high = results_by_elo[elo_high]["fire_rates"]

    # Features that fire more at low Elo (tactical — player misses them)
    diff = fr_low - fr_high
    tactical_candidates = np.where(diff > 0.05)[0]
    print(f"\nFeatures firing MORE at low Elo ({elo_low}) vs high ({elo_high}):")
    print(f"  (diff > 5%): {len(tactical_candidates)} features")
    for idx in tactical_candidates[:20]:
        print(f"    Feature {idx:4d}: {fr_low[idx]:.3f} ({elo_low}) → {fr_high[idx]:.3f} ({elo_high}) "
              f"  Δ={diff[idx]:+.3f}")

    # Features that fire more at high Elo
    positional_candidates = np.where(diff < -0.05)[0]
    print(f"\nFeatures firing MORE at high Elo ({elo_high}):")
    print(f"  (diff < -5%): {len(positional_candidates)} features")
    for idx in positional_candidates[:20]:
        print(f"    Feature {idx:4d}: {fr_low[idx]:.3f} ({elo_low}) → {fr_high[idx]:.3f} ({elo_high}) "
              f"  Δ={diff[idx]:+.3f}")

    # Elo-invariant features
    invariant = np.where(np.abs(diff) < 0.01)[0]
    invariant_active = invariant[(fr_low[invariant] > 0.01) | (fr_high[invariant] > 0.01)]
    print(f"\nElo-invariant features (|diff| < 1%, active > 1%):")
    print(f"  {len(invariant_active)} features")

    # Overall sensitivity score
    active_mask = (fr_low > 0.005) | (fr_high > 0.005)
    if active_mask.sum() > 0:
        mean_abs_diff = np.abs(diff[active_mask]).mean()
        print(f"\nOverall sensitivity (mean |Δfire_rate| across active features): {mean_abs_diff:.4f}")
        if mean_abs_diff < 0.01:
            print("  ⚠ LOW SENSITIVITY: SAE may be learning position structure, not Elo-dependent blind spots")
        elif mean_abs_diff > 0.03:
            print("  ✓ GOOD SENSITIVITY: Features change meaningfully with Elo")
        else:
            print("  ~ MODERATE SENSITIVITY: Some Elo-dependent features present")

    # Save results
    output_data = {
        "elos": elos,
        "n_positions": len(fens),
        "pool_mode": pool_mode,
        "sae_path": args.sae,
        "per_elo": {
            str(elo): {
                "l0": results_by_elo[elo]["l0"],
                "fire_rates": results_by_elo[elo]["fire_rates"].tolist(),
                "mean_activation": results_by_elo[elo]["mean_activation"].tolist(),
            }
            for elo in elos
        },
        "sensitivity": {
            "mean_abs_diff": float(mean_abs_diff) if active_mask.sum() > 0 else 0,
            "n_tactical_candidates": len(tactical_candidates),
            "n_positional_candidates": len(positional_candidates),
            "n_invariant_active": len(invariant_active),
            "tactical_feature_ids": tactical_candidates.tolist(),
            "positional_feature_ids": positional_candidates.tolist(),
        },
    }

    if args.output:
        out_path = args.output
    else:
        out_path = args.sae.replace(".pt", "_elo_sensitivity.json")

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
