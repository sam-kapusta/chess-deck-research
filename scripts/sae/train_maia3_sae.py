#!/usr/bin/env python3
"""Train BatchTopK SAE on Maia 3 activations.

Port of Sandstone's canonical BatchTopK with:
- Batch-level top-k (train AND eval)
- Gradient projection for decoder unit norm
- k warmup (1 → target_k over warmup_steps)
- AMP (fp16 autocast + GradScaler)
- L2 normalization on input
- AuxK loss for persistently dead features

Usage (on chess-poc GPU):
    python scripts/sae/train_maia3_sae.py \
      --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_from_sq.pt \
      --dict-size 2048 --k 32

    python scripts/sae/train_maia3_sae.py \
      --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_mean.pt \
      --dict-size 2048 --k 32
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
OUTPUT = BASE + "/output/maia3_sae"


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE (arXiv:2412.06410).

    Batch-level top-k activation. Selects top n*k activations across the
    entire batch at both train and eval. AuxK loss targets persistently dead
    features.
    """

    def __init__(self, d_input, d_hidden, k, k_aux=256, aux_alpha=1/32,
                 n_batches_to_dead=5, dtype=torch.float32):
        super().__init__()
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(d_input, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_input, dtype=dtype))

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
        l2_loss = (x_hat.float() - x.float()).pow(2).mean()

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
                aux_loss = self.aux_alpha * (error_hat.float() - error.float()).pow(2).mean()

        loss = l2_loss + aux_loss
        return loss, x_hat, acts, l2_loss, aux_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


def normalize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize activations matching Sandstone's canonical pipeline.

    1. Z-score per dimension (mean=0, std=1)
    2. L2 normalize each sample to unit sphere
    3. Return pre-L2 norms as sample weights

    Z-score ensures all dimensions contribute equally.
    L2 ensures endgame positions don't get drowned by middlegame.
    Weights preserve original magnitude info for weighted training.
    """
    # Z-score per dimension
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-6)
    x_zscore = (x - mean) / std

    # L2 normalize to unit sphere, keep norms as weights
    norms = x_zscore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normed = x_zscore / norms

    return x_normed, norms.squeeze(-1)


def train(model, train_loader, val_loader, config, device):
    target_k = model.k
    warmup_steps = config["sparsity_warmup_steps"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.99),
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_steps = 0
    best_val_loss = float("inf")
    log = []

    for epoch in range(config["n_epochs"]):
        model.train()
        epoch_losses = []

        for batch, weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}"):
            batch = batch.to(device, non_blocking=True)
            n_steps += 1

            if n_steps <= warmup_steps:
                model.k = max(1, int(1 + (target_k - 1) * n_steps / warmup_steps))
            else:
                model.k = target_k

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss, x_hat, acts, l2_loss, aux_loss = model(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.make_decoder_weights_and_grad_unit_norm()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

            if n_steps % 500 == 0:
                l0 = (acts != 0).sum(-1).float().mean().item()
                print(f"  Step {n_steps:5d} | Loss: {loss.item():.6f} | "
                      f"L2: {l2_loss.item():.6f} | Aux: {aux_loss.item():.6f} | "
                      f"L0: {l0:.1f} | k: {model.k}")

        # End-of-epoch validation
        model.eval()
        model.k = target_k
        val_losses = []
        all_acts = []
        with torch.no_grad():
            for batch, weights in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss, x_hat, acts, l2_loss, aux_loss = model(batch)
                val_losses.append(loss.item())
                all_acts.append((acts > 0).float().cpu())

        val_loss = np.mean(val_losses)
        train_loss = np.mean(epoch_losses)

        # Feature stats
        act_binary = torch.cat(all_acts, dim=0)
        freq = act_binary.mean(dim=0).numpy()
        n_dead = int((freq == 0).sum())
        n_active = int((freq > 0.005).sum())
        l0_val = act_binary.sum(-1).mean().item()

        entry = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "l0": float(l0_val),
            "n_dead": n_dead,
            "n_active_gt_0.5pct": n_active,
            "steps": n_steps,
        }
        log.append(entry)
        print(f"  Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"L0: {l0_val:.1f} | Dead: {n_dead} | Active>0.5%: {n_active}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    model.k = target_k
    return log


def main():
    parser = argparse.ArgumentParser(description="Train BatchTopK SAE on Maia 3 activations")
    parser.add_argument("--activations", type=str, required=True,
                        help="Path to .pt file from maia3_activations.py")
    parser.add_argument("--dict-size", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--k-aux", type=int, default=256)
    parser.add_argument("--aux-alpha", type=float, default=1/32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load activations
    print(f"Loading activations from {args.activations}")
    data = torch.load(args.activations, map_location="cpu", weights_only=False)
    raw_acts = data["activations"]
    if not isinstance(raw_acts, torch.Tensor):
        raw_acts = torch.tensor(raw_acts, dtype=torch.float32)
    else:
        raw_acts = raw_acts.float()

    print(f"  Raw shape: {raw_acts.shape}")
    d_input = raw_acts.shape[-1]

    # Normalize: Z-score per dim → L2 to unit sphere (matching Sandstone)
    acts_norm, sample_weights = normalize_activations(raw_acts)
    print(f"  Normalized: Z-score → L2 (Sandstone canonical)")
    print(f"  Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    del raw_acts

    # Train/val split
    n = acts_norm.shape[0]
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = torch.randperm(n)
    train_data = acts_norm[perm[:n_train]]
    train_weights = sample_weights[perm[:n_train]]
    val_data = acts_norm[perm[n_train:]]
    val_weights = sample_weights[perm[n_train:]]
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        TensorDataset(train_data, train_weights),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_weights),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    model = BatchTopKSAE(
        d_input=d_input,
        d_hidden=args.dict_size,
        k=args.k,
        k_aux=args.k_aux,
        aux_alpha=args.aux_alpha,
    ).to(device)

    print(f"\nModel: BatchTopKSAE(d_input={d_input}, d_hidden={args.dict_size}, k={args.k})")
    print(f"  k_aux={args.k_aux}, aux_alpha={args.aux_alpha}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    config = {
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "sparsity_warmup_steps": args.warmup_steps,
    }

    # Train
    t0 = time.time()
    log = train(model, train_loader, val_loader, config, device)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    pool_mode = data.get("config", {}).get("pool", "unknown")
    if args.output:
        out_path = args.output
    else:
        os.makedirs(OUTPUT, exist_ok=True)
        out_path = f"{OUTPUT}/maia3_sae_{pool_mode}_{args.dict_size}_k{args.k}.pt"

    save_payload = {
        "state_dict": model.state_dict(),
        "config": {
            "model": "maia3",
            "architecture": "BatchTopKSAE",
            "d_input": d_input,
            "dict_size": args.dict_size,
            "k": args.k,
            "k_aux": args.k_aux,
            "aux_alpha": args.aux_alpha,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "warmup_steps": args.warmup_steps,
            "pool_mode": pool_mode,
            "n_train": n_train,
            "n_val": n_val,
            "l2_normalized": True,
            "seed": args.seed,
        },
        "training_log": log,
        "source_activations": args.activations,
    }

    torch.save(save_payload, out_path)
    print(f"Saved to {out_path}")

    # Final T1 structural metrics
    print("\n--- T1 Structural Metrics ---")
    model.eval()
    all_acts_list = []
    all_x = []
    all_xhat = []
    with torch.no_grad():
        for batch, weights in val_loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                _, x_hat, acts, _, _ = model(batch)
            all_acts_list.append(acts.cpu().numpy())
            all_x.append(batch.cpu().numpy())
            all_xhat.append(x_hat.cpu().numpy())

    acts_np = np.concatenate(all_acts_list)
    x_np = np.concatenate(all_x)
    xhat_np = np.concatenate(all_xhat)

    freq = (acts_np > 0).mean(axis=0)
    n_dead = int((freq == 0).sum())
    n_useful = int((freq >= 0.005).sum())
    l0 = float((acts_np > 0).sum(axis=1).mean())
    mse = float(np.mean((x_np - xhat_np) ** 2))
    var_x = float(np.var(x_np))
    fvu = mse / var_x if var_x > 0 else float("inf")

    print(f"  Dead features: {n_dead}/{args.dict_size}")
    print(f"  Useful (>0.5% fire rate): {n_useful}/{args.dict_size}")
    print(f"  L0 (avg active per sample): {l0:.1f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  FVU: {fvu:.4f}")
    print(f"  Target fire rate range: 0.5-3%")

    fire_in_range = int(((freq >= 0.005) & (freq <= 0.03)).sum())
    print(f"  Features in 0.5-3% range: {fire_in_range}/{args.dict_size}")


if __name__ == "__main__":
    main()
