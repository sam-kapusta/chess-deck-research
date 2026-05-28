#!/usr/bin/env python3
"""Sweep k values on BatchTopK SAE — measure decoder orthogonality + feature quality.

Goal: find optimal k for Matryoshka retraining.
Metrics per k:
  - avg_max_cosine: mean of max cosine similarity per decoder vector (lower = more orthogonal)
  - n_interpretable: features with fire rate in 0.1-1% range (Jonathan's "useful" range)
  - n_dead: features that never fire
  - fvu: fraction of variance unexplained (reconstruction quality)
  - l0: actual average active features per sample

Usage (on chess-poc):
    python scripts/sae/sweep_k_orthogonality.py

Expects activations at:
    ~/SageMaker/chess-stage-a/cache/maia3_blunder_activations_diff.pt
    (200K blunders, diff pooling, 512-dim)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
# Try known activation file paths
ACT_PATHS = [
    BASE + "/cache/maia3_blunder_activations_diff.pt",
    BASE + "/cache/maia3_blunder_from_sq.pt",
    BASE + "/cache/maia3_blunder_diff.pt",
]
OUTPUT = BASE + "/output/maia3_sae"
os.makedirs(OUTPUT, exist_ok=True)

DICT_SIZE = 2048
K_VALUES = [8, 12, 16, 20, 24, 32, 48]
N_EPOCHS = 50
BATCH_SIZE = 4096
LR = 3e-4
WARMUP_STEPS = 500
K_AUX = 256
AUX_ALPHA = 1 / 32
SEED = 42
VAL_SPLIT = 0.1


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k, k_aux=256, aux_alpha=1/32,
                 n_batches_to_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        self.d_hidden = d_hidden
        self.k = k
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.n_batches_to_dead = n_batches_to_dead

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

        if self.training:
            feature_active = (acts > 0).any(dim=0)
            self.num_batches_not_active[feature_active] = 0
            self.num_batches_not_active[~feature_active] += 1

        x_hat = acts @ self.W_dec + self.b_dec
        l2_loss = (x_hat - x).pow(2).mean()

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.k_aux > 0:
            dead_features = self.num_batches_not_active >= self.n_batches_to_dead
            if dead_features.sum() > 0:
                error = (x - x_hat).detach()
                dead_pre_acts = F.relu(z[:, dead_features])
                k_aux_actual = min(self.k_aux, int(dead_features.sum()))
                topk_aux = torch.topk(dead_pre_acts, k=k_aux_actual, dim=-1)
                dead_acts = torch.zeros_like(dead_pre_acts).scatter(
                    -1, topk_aux.indices, topk_aux.values)
                error_hat = dead_acts @ self.W_dec[dead_features]
                aux_loss = self.aux_alpha * (error_hat - error).pow(2).mean()

        loss = l2_loss + aux_loss
        return loss, x_hat, acts, l2_loss, aux_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


def normalize_activations(x: torch.Tensor) -> torch.Tensor:
    """Z-score + L2 normalize (canonical for Maia 3 SAE)."""
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-6)
    x_zscore = (x - mean) / std
    norms = x_zscore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x_zscore / norms


def compute_decoder_orthogonality(model):
    """Compute avg max cosine similarity between decoder vectors.

    For each decoder vector, find its maximum cosine similarity with any other
    decoder vector. Return the mean of these maxima. Lower = more orthogonal.
    """
    W_dec = model.W_dec.data  # [d_hidden, d_input]
    W_dec_normed = W_dec / W_dec.norm(dim=-1, keepdim=True)

    # Compute pairwise cosine similarity
    # Do in chunks to avoid OOM on large dicts
    chunk_size = 512
    n = W_dec_normed.shape[0]
    max_cosines = torch.zeros(n, device=W_dec.device)

    for i in range(0, n, chunk_size):
        chunk = W_dec_normed[i:i+chunk_size]  # [chunk, d_input]
        sim = chunk @ W_dec_normed.T  # [chunk, n]
        # Zero out self-similarity
        for j in range(chunk.shape[0]):
            sim[j, i + j] = -1.0
        max_cosines[i:i+chunk_size] = sim.max(dim=-1).values

    return {
        "avg_max_cosine": float(max_cosines.mean()),
        "median_max_cosine": float(max_cosines.median()),
        "p90_max_cosine": float(max_cosines.quantile(0.9)),
        "p99_max_cosine": float(max_cosines.quantile(0.99)),
    }


def train_and_evaluate(k, train_data, val_data, d_input, device):
    """Train one SAE at given k, return metrics."""
    print(f"\n{'='*60}")
    print(f"Training k={k}, dict={DICT_SIZE}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)

    model = BatchTopKSAE(
        d_input=d_input,
        d_hidden=DICT_SIZE,
        k=k,
        k_aux=K_AUX,
        aux_alpha=AUX_ALPHA,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    target_k = k
    n_steps = 0

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            n_steps += 1

            if n_steps <= WARMUP_STEPS:
                model.k = max(1, int(1 + (target_k - 1) * n_steps / WARMUP_STEPS))
            else:
                model.k = target_k

            with torch.autocast("cuda", dtype=torch.float16):
                loss, x_hat, acts, l2_loss, aux_loss = model(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.make_decoder_weights_and_grad_unit_norm()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.6f} | k_eff: {model.k}")

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s")

    # === EVALUATE ===
    model.eval()
    model.k = target_k

    all_acts = []
    all_x = []
    all_xhat = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                _, x_hat, acts, _, _ = model(batch)
            all_acts.append(acts.cpu())
            all_x.append(batch.cpu())
            all_xhat.append(x_hat.cpu())

    acts_cat = torch.cat(all_acts).numpy()
    x_cat = torch.cat(all_x).numpy()
    xhat_cat = torch.cat(all_xhat).numpy()

    # Fire rate per feature
    freq = (acts_cat > 0).mean(axis=0)

    # Core metrics
    n_dead = int((freq == 0).sum())
    n_interpretable = int(((freq >= 0.001) & (freq <= 0.01)).sum())  # 0.1-1%
    n_useful_broad = int(((freq >= 0.001) & (freq <= 0.03)).sum())   # 0.1-3%
    n_active = int((freq > 0).sum())
    l0 = float((acts_cat > 0).sum(axis=1).mean())

    mse = float(np.mean((x_cat - xhat_cat) ** 2))
    var_x = float(np.var(x_cat))
    fvu = mse / var_x if var_x > 0 else float("inf")

    # Decoder orthogonality
    ortho = compute_decoder_orthogonality(model)

    metrics = {
        "k": k,
        "dict_size": DICT_SIZE,
        "n_epochs": N_EPOCHS,
        "train_time_s": elapsed,
        "l0_actual": l0,
        "fvu": fvu,
        "mse": mse,
        "n_dead": n_dead,
        "n_active": n_active,
        "n_interpretable_0.1_1pct": n_interpretable,
        "n_useful_0.1_3pct": n_useful_broad,
        **ortho,
    }

    print(f"\n  Results for k={k}:")
    print(f"    L0 (actual): {l0:.1f}")
    print(f"    FVU: {fvu:.4f}")
    print(f"    Dead features: {n_dead}/{DICT_SIZE}")
    print(f"    Interpretable (0.1-1%): {n_interpretable}/{DICT_SIZE}")
    print(f"    Useful (0.1-3%): {n_useful_broad}/{DICT_SIZE}")
    print(f"    Avg max cosine: {ortho['avg_max_cosine']:.4f}")
    print(f"    P90 max cosine: {ortho['p90_max_cosine']:.4f}")

    # Save weights
    save_path = f"{OUTPUT}/sweep_k{k}_d{DICT_SIZE}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"k": k, "dict_size": DICT_SIZE, "d_input": d_input},
        "metrics": metrics,
    }, save_path)
    print(f"    Saved: {save_path}")

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        print("WARNING: Running on CPU — will be very slow!")

    # Find and load activations
    act_path = None
    for p in ACT_PATHS:
        if os.path.exists(p):
            act_path = p
            break

    if act_path is None:
        print("ERROR: No activation file found. Tried:")
        for p in ACT_PATHS:
            print(f"  {p}")
        print("\nPlease provide the correct path to Maia 3 diff-pooled activations.")
        sys.exit(1)

    print(f"Loading activations from {act_path}")
    data = torch.load(act_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "activations" in data:
        raw_acts = data["activations"]
    elif isinstance(data, torch.Tensor):
        raw_acts = data
    else:
        print(f"Unknown format. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        sys.exit(1)

    if not isinstance(raw_acts, torch.Tensor):
        raw_acts = torch.tensor(raw_acts, dtype=torch.float32)
    else:
        raw_acts = raw_acts.float()

    print(f"  Shape: {raw_acts.shape}")
    d_input = raw_acts.shape[-1]

    # Normalize (Z-score + L2, same as current SAE)
    acts_norm = normalize_activations(raw_acts)
    del raw_acts
    print(f"  Normalized (Z-score + L2). Range: [{acts_norm.min():.2f}, {acts_norm.max():.2f}]")

    # Train/val split (deterministic)
    torch.manual_seed(SEED)
    n = acts_norm.shape[0]
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val
    perm = torch.randperm(n)
    train_data = acts_norm[perm[:n_train]]
    val_data = acts_norm[perm[n_train:]]
    print(f"  Train: {n_train}, Val: {n_val}")
    del acts_norm

    # Run sweep
    results = []
    for k in K_VALUES:
        metrics = train_and_evaluate(k, train_data, val_data, d_input, device)
        results.append(metrics)

    # Summary table
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'k':>4} | {'L0':>5} | {'FVU':>6} | {'Dead':>5} | {'Interp':>6} | {'Useful':>6} | {'AvgMaxCos':>9} | {'P90Cos':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['k']:>4} | {r['l0_actual']:>5.1f} | {r['fvu']:>6.4f} | "
              f"{r['n_dead']:>5} | {r['n_interpretable_0.1_1pct']:>6} | "
              f"{r['n_useful_0.1_3pct']:>6} | {r['avg_max_cosine']:>9.4f} | "
              f"{r['p90_max_cosine']:>7.4f}")

    # Save summary
    summary_path = f"{OUTPUT}/k_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"sweep_config": {
            "dict_size": DICT_SIZE,
            "k_values": K_VALUES,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "normalization": "zscore_l2",
        }, "results": results}, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Recommendation
    print("\n--- RECOMMENDATION ---")
    # Find the "shoulder" — where avg_max_cosine is relatively flat
    # and bias toward fewest active features + most interpretable-range features
    best = None
    best_score = -1
    for r in results:
        # Score: maximize interpretable features, penalize high cosine, penalize dead features
        score = (r["n_interpretable_0.1_1pct"] / DICT_SIZE) - r["avg_max_cosine"] - (r["n_dead"] / DICT_SIZE)
        if score > best_score:
            best_score = score
            best = r

    if best:
        print(f"  Best k = {best['k']} (score={best_score:.4f})")
        print(f"  L0={best['l0_actual']:.1f}, FVU={best['fvu']:.4f}, "
              f"Interpretable={best['n_interpretable_0.1_1pct']}, "
              f"AvgMaxCos={best['avg_max_cosine']:.4f}")

    print("\nJonathan's guidance: find the flat shoulder of avg_max_cosine,")
    print("then within that shoulder pick lowest L0 + highest interpretable count.")


if __name__ == "__main__":
    main()
