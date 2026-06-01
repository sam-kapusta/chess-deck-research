#!/usr/bin/env python3
"""Full T1 structural eval for BatchTopK SAE — matches SandstonePersonas compute_t1.py exactly.

Usage (on chess-poc):
    python scripts/evaluation/eval_t1_btk.py \
      --weights ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
      --cache ~/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
      --output ~/SageMaker/chess-stage-a/output/t1_btk_2048_k32.json
"""
import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

GATE = {"max_dead": 50, "max_fvu": 0.15, "min_l0": 20, "max_l0": 50,
        "max_redundant_pairs": 500}


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden
        self.k = k
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def encode_threshold(self, x, threshold):
        """Deterministic threshold inference — no batch dependency."""
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        z_relu = F.relu(z)
        return z_relu * (z_relu > threshold)

    def forward_threshold(self, x, threshold):
        acts = self.encode_threshold(x, threshold)
        x_hat = acts @ self.W_dec + self.b_dec
        l2_loss = (x_hat.float() - x.float()).pow(2).mean()
        return torch.tensor(0.0), x_hat, acts, l2_loss, torch.tensor(0.0)


def normalize(raw, mean, std):
    x = (raw - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load weights + config
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt["config"]
    d_input, dict_size, k = cfg["d_input"], cfg["dict_size"], cfg["k"]

    # Load normalization stats (saved alongside weights)
    stats_path = args.weights.replace(".pt", "_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std  = torch.tensor(stats["std"],  dtype=torch.float32)

    # Load + normalize corpus
    print("Loading corpus...")
    raw = torch.load(args.cache, map_location="cpu",
                     weights_only=False)["activations"].float()
    print(f"  {raw.shape[0]:,} positions, {raw.shape[1]}d")
    x_norm = normalize(raw, mean, std)
    del raw

    # Load model
    model = BatchTopKSAE(d_input, dict_size, k)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # Load calibrated threshold (deterministic inference, no batch dependency)
    cal_path = args.weights.replace("_weights.pt", "_calibration.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Calibration file not found: {cal_path}\nRun calibrate_threshold_v2.py first.")
    threshold = json.load(open(cal_path))["global_threshold"]
    print(f"Using calibrated threshold θ={threshold:.6f}")

    # Forward pass — collect acts, x, xhat
    loader = DataLoader(TensorDataset(x_norm), batch_size=4096, shuffle=False)
    all_acts, all_x, all_xhat = [], [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, xhat, acts, _, _ = model.forward_threshold(batch, threshold)
            all_acts.append(acts.cpu().numpy())
            all_x.append(batch.cpu().numpy())
            all_xhat.append(xhat.cpu().numpy())

    acts = np.concatenate(all_acts)
    x    = np.concatenate(all_x)
    xhat = np.concatenate(all_xhat)

    # --- Feature frequency ---
    freq = (acts > 0).mean(axis=0)
    dead       = int((freq == 0).sum())
    near_dead  = int(((freq > 0) & (freq < 0.001)).sum())
    useful     = int((freq >= 0.001).sum())
    very_active = int((freq >= 0.05).sum())

    # --- L0 ---
    l0 = float((acts > 0).sum(axis=1).mean())

    # --- Reconstruction ---
    mse   = float(np.mean((x - xhat) ** 2))
    var_x = float(np.var(x))         # population variance, flattened
    fvu   = mse / var_x if var_x > 0 else float("inf")

    # --- Decoder cosine (alive features only) ---
    W_dec = model.W_dec.detach().cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
    alive_idx = np.where(freq > 0)[0]
    mean_cos = max_cos = pct_high = 0.0
    redundant_pairs = 0
    if len(alive_idx) > 1:
        W_alive = W_dec_norm[alive_idx]                    # dict_size<=2048 -> full matrix
        cos_matrix = W_alive @ W_alive.T
        triu = np.triu_indices(len(alive_idx), k=1)
        cs = cos_matrix[triu]
        redundant_pairs = int(np.sum(np.abs(cs) > 0.5))
        mean_cos = float(np.mean(cs))
        max_cos  = float(np.max(np.abs(cs)))
        pct_high = float(np.sum(np.abs(cs) > 0.5) / len(cs) * 100)

    # --- Collapsed norms ---
    dec_norms = np.linalg.norm(W_dec, axis=1)
    collapsed = int((dec_norms < 0.01).sum())

    # --- Bimodality (CV of nonzero acts per feature) ---
    bimod = []
    for i in range(dict_size):
        nz = acts[:, i][acts[:, i] > 0]
        if len(nz) > 10:
            bimod.append(float(np.std(nz) / (np.mean(nz) + 1e-8)))
    mean_bimod = float(np.mean(bimod)) if bimod else 0.0

    metrics = {
        "tier": "T1", "n_positions": len(x),
        "fvu": fvu, "mse": mse, "l0": l0,
        "dead": dead, "near_dead": near_dead,
        "useful": useful, "very_active": very_active,
        "alive": dict_size - dead,
        "mean_decoder_cosine": mean_cos, "max_decoder_cosine": max_cos,
        "pct_high_sim": pct_high, "redundant_pairs": redundant_pairs,
        "collapsed_norms": collapsed, "mean_decoder_norm": float(np.mean(dec_norms)),
        "mean_bimodality": mean_bimod,
    }

    print("\n=== T1 Structural Metrics ===")
    print(f"  FVU={fvu:.4f}  L0={l0:.1f}  Dead={dead}  Near-dead={near_dead}")
    print(f"  Useful={useful}  Very-active={very_active}")
    print(f"  DecCos: mean={mean_cos:.4f} max={max_cos:.4f} redundant={redundant_pairs}")
    print(f"  Bimodality={mean_bimod:.3f}  Collapsed={collapsed}")

    # --- Gate check ---
    fails = []
    if dead > GATE["max_dead"]:       fails.append(f"dead={dead} > {GATE['max_dead']}")
    if fvu  > GATE["max_fvu"]:        fails.append(f"FVU={fvu:.4f} > {GATE['max_fvu']}")
    if l0 < GATE["min_l0"] or l0 > GATE["max_l0"]:
        fails.append(f"L0={l0:.1f} outside [{GATE['min_l0']},{GATE['max_l0']}]")
    if redundant_pairs > GATE["max_redundant_pairs"]:
        fails.append(f"redundant_pairs={redundant_pairs} > {GATE['max_redundant_pairs']}")

    if fails:
        print("\n!!! GATE FAILED - investigate before labeling !!!")
        for f in fails:
            print(f"  FAIL: {f}")
        metrics["gate_passed"] = False
        metrics["gate_failures"] = fails
    else:
        print("\n Gate passed - proceed to feature stats + labeling")
        metrics["gate_passed"] = True

    import os; os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {args.output}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
