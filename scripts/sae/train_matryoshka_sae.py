#!/usr/bin/env python3
"""Train Matryoshka BatchTopK SAE on Maia 3 activations.

Matryoshka SAE (Bussmann et al., ICML 2025): trains nested sub-SAEs simultaneously.
Each prefix of latents must independently reconstruct the input, forcing early
latents to learn general concepts and later latents to specialize.

Architecture:
  - Same encoder/decoder as standard BatchTopK
  - Multiple reconstruction losses at prefix boundaries
  - Each prefix must reconstruct alone (no leaning on later latents)
  - BatchTopK sparsity applied once to full latent vector

Usage (on chess-poc):
    python scripts/sae/train_matryoshka_sae.py \
      --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt \
      --dict-size 2048 --k 24 --prefixes 16,64,256,2048

Default prefixes: [16, 64, 256, 2048] — dashboard / deep-dive / drill / atoms.
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
ACT_PATHS = [
    BASE + "/cache/maia3_blunder_diff.pt",
    BASE + "/cache/maia3_blunder_activations_diff.pt",
    BASE + "/cache/maia3_blunder_from_sq.pt",
]


class MatryoshkaBatchTopKSAE(nn.Module):
    """Matryoshka BatchTopK SAE.

    Single encoder produces full latent vector. BatchTopK applied once.
    Reconstruction loss computed at each prefix size independently.
    Early latents forced to be general; later latents specialize.
    """

    def __init__(self, d_input, d_hidden, k, prefixes, k_aux=256,
                 aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        assert prefixes[-1] == d_hidden, "Last prefix must equal dict size"
        assert all(prefixes[i] < prefixes[i+1] for i in range(len(prefixes)-1)), \
            "Prefixes must be strictly increasing"

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k
        self.prefixes = prefixes
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.n_batches_to_dead = n_batches_to_dead

        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def forward(self, x):
        # Encode — same as standard BatchTopK
        z = (x - self.b_dec) @ self.W_enc + self.b_enc

        # BatchTopK sparsity on FULL latent vector
        batch_size = x.shape[0]
        total_k = batch_size * self.k
        z_relu = F.relu(z)
        flat_z = z_relu.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat_z, k=min(int(total_k), flat_z.numel()))
        acts = torch.zeros_like(flat_z)
        acts[topk_idx] = topk_vals
        acts = acts.reshape(z.shape)

        # Track dead features
        if self.training:
            feature_active = (acts > 0).any(dim=0)
            self.num_batches_not_active[feature_active] = 0
            self.num_batches_not_active[~feature_active] += 1

        # Matryoshka reconstruction: compute loss at each prefix
        # Each prefix must independently reconstruct the input
        prefix_losses = []
        current_output = self.b_dec.clone().unsqueeze(0).expand(batch_size, -1)

        for i, prefix_size in enumerate(self.prefixes):
            start_idx = 0 if i == 0 else self.prefixes[i - 1]
            end_idx = prefix_size

            # Add contribution from this group of features
            group_acts = acts[:, start_idx:end_idx]
            group_decoder = self.W_dec[start_idx:end_idx]
            current_output = current_output + group_acts @ group_decoder

            # Reconstruction loss at this prefix level
            prefix_loss = (current_output - x).pow(2).mean()
            prefix_losses.append(prefix_loss)

        # Total reconstruction loss = sum of all prefix losses (equal weighting)
        l2_loss = sum(prefix_losses)

        # Full reconstruction (from last prefix) for aux loss and output
        x_hat = current_output

        # Aux loss for dead features (same as standard)
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

        return loss, x_hat, acts, l2_loss, aux_loss, prefix_losses

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def get_prefix_reconstruction(self, x, prefix_idx):
        """Get reconstruction using only the first N latents (for eval)."""
        z = (x - self.b_dec) @ self.W_enc + self.b_enc

        batch_size = x.shape[0]
        total_k = batch_size * self.k
        z_relu = F.relu(z)
        flat_z = z_relu.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat_z, k=min(int(total_k), flat_z.numel()))
        acts = torch.zeros_like(flat_z)
        acts[topk_idx] = topk_vals
        acts = acts.reshape(z.shape)

        prefix_size = self.prefixes[prefix_idx]
        prefix_acts = acts[:, :prefix_size]
        prefix_decoder = self.W_dec[:prefix_size]
        x_hat = prefix_acts @ prefix_decoder + self.b_dec

        return x_hat, prefix_acts


def normalize_activations(x: torch.Tensor) -> torch.Tensor:
    """Z-score + L2 normalize (canonical for Maia 3 SAE)."""
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-6)
    x_zscore = (x - mean) / std
    norms = x_zscore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x_zscore / norms


def train(model, train_loader, val_loader, config, device):
    target_k = model.k
    warmup_steps = config["sparsity_warmup_steps"]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.99))

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    n_steps = 0
    best_val_loss = float("inf")
    log = []

    for epoch in range(config["n_epochs"]):
        model.train()
        epoch_losses = []
        epoch_prefix_losses = [[] for _ in model.prefixes]

        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}"):
            batch = batch.to(device, non_blocking=True)
            n_steps += 1

            if n_steps <= warmup_steps:
                model.k = max(1, int(1 + (target_k - 1) * n_steps / warmup_steps))
            else:
                model.k = target_k

            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss, x_hat, acts, l2_loss, aux_loss, prefix_losses = model(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.make_decoder_weights_and_grad_unit_norm()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            for pi, pl in enumerate(prefix_losses):
                epoch_prefix_losses[pi].append(pl.item())

            if n_steps % 500 == 0:
                l0 = (acts != 0).sum(-1).float().mean().item()
                prefix_str = " | ".join(
                    [f"P{model.prefixes[i]}={np.mean(epoch_prefix_losses[i][-50:]):.4f}"
                     for i in range(len(model.prefixes))])
                print(f"  Step {n_steps:5d} | Loss: {loss.item():.4f} | "
                      f"L0: {l0:.1f} | k: {model.k} | {prefix_str}")

        # End-of-epoch validation
        model.eval()
        model.k = target_k
        val_losses = []
        all_acts = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    loss, x_hat, acts, l2_loss, aux_loss, prefix_losses = model(batch)
                val_losses.append(loss.item())
                all_acts.append((acts > 0).float().cpu())

        val_loss = np.mean(val_losses)
        train_loss = np.mean(epoch_losses)

        # Feature stats
        act_binary = torch.cat(all_acts, dim=0)
        freq = act_binary.mean(dim=0).numpy()
        n_dead = int((freq == 0).sum())
        l0_val = act_binary.sum(-1).mean().item()

        # Per-prefix feature stats
        prefix_stats = []
        for i, psize in enumerate(model.prefixes):
            start = 0 if i == 0 else model.prefixes[i-1]
            end = psize
            prefix_freq = freq[start:end]
            prefix_stats.append({
                "prefix": psize,
                "range": f"[{start}:{end}]",
                "n_dead": int((prefix_freq == 0).sum()),
                "n_active": int((prefix_freq > 0).sum()),
                "avg_fire_rate": float(prefix_freq.mean()),
            })

        entry = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "l0": float(l0_val),
            "n_dead": n_dead,
            "steps": n_steps,
            "prefix_stats": prefix_stats,
        }
        log.append(entry)

        print(f"  Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"L0: {l0_val:.1f} | Dead: {n_dead}")
        for ps in prefix_stats:
            print(f"    {ps['range']} dead={ps['n_dead']} avg_fire={ps['avg_fire_rate']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    model.k = target_k
    return log


def evaluate_per_prefix(model, val_loader, device):
    """Evaluate reconstruction quality at each prefix level."""
    model.eval()
    results = {}

    for pi, prefix_size in enumerate(model.prefixes):
        all_x = []
        all_xhat = []
        all_acts = []

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                x_hat, prefix_acts = model.get_prefix_reconstruction(batch, pi)
                all_x.append(batch.cpu().numpy())
                all_xhat.append(x_hat.cpu().numpy())
                all_acts.append((prefix_acts > 0).cpu().numpy())

        x_np = np.concatenate(all_x)
        xhat_np = np.concatenate(all_xhat)
        acts_np = np.concatenate(all_acts)

        mse = float(np.mean((x_np - xhat_np) ** 2))
        var_x = float(np.var(x_np))
        fvu = mse / var_x if var_x > 0 else float("inf")

        freq = acts_np.mean(axis=0)
        l0 = float(acts_np.sum(axis=1).mean())
        n_dead = int((freq == 0).sum())
        n_interp = int(((freq >= 0.001) & (freq <= 0.01)).sum())

        results[prefix_size] = {
            "fvu": fvu,
            "mse": mse,
            "l0": l0,
            "n_dead": n_dead,
            "n_features": prefix_size,
            "n_interpretable_0.1_1pct": n_interp,
            "avg_fire_rate": float(freq.mean()),
        }

        print(f"  Prefix {prefix_size:>5}: FVU={fvu:.4f} | L0={l0:.1f} | "
              f"Dead={n_dead}/{prefix_size} | Interp(0.1-1%)={n_interp}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Matryoshka BatchTopK SAE")
    parser.add_argument("--activations", type=str, default=None,
                        help="Path to .pt file (auto-detected if not set)")
    parser.add_argument("--dict-size", type=int, default=2048)
    parser.add_argument("--k", type=int, default=24,
                        help="BatchTopK k (avg active features per sample)")
    parser.add_argument("--prefixes", type=str, default="16,64,256,2048",
                        help="Comma-separated prefix sizes (must end with dict-size)")
    parser.add_argument("--k-aux", type=int, default=256)
    parser.add_argument("--aux-alpha", type=float, default=1/32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prefixes = [int(x) for x in args.prefixes.split(",")]
    if prefixes[-1] != args.dict_size:
        print(f"ERROR: last prefix ({prefixes[-1]}) must equal dict-size ({args.dict_size})")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find activations
    act_path = args.activations
    if act_path is None:
        for p in ACT_PATHS:
            if os.path.exists(p):
                act_path = p
                break
    if act_path is None or not os.path.exists(act_path):
        print("ERROR: No activation file found.")
        sys.exit(1)

    print(f"Loading activations from {act_path}")
    data = torch.load(act_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "activations" in data:
        raw_acts = data["activations"]
    elif isinstance(data, torch.Tensor):
        raw_acts = data
    else:
        print(f"Unknown format: {type(data)}")
        sys.exit(1)

    if not isinstance(raw_acts, torch.Tensor):
        raw_acts = torch.tensor(raw_acts, dtype=torch.float32)
    else:
        raw_acts = raw_acts.float()

    print(f"  Shape: {raw_acts.shape}")
    d_input = raw_acts.shape[-1]

    # Normalize
    acts_norm = normalize_activations(raw_acts)
    del raw_acts
    print(f"  Normalized (Z-score + L2)")

    # Train/val split
    n = acts_norm.shape[0]
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = torch.randperm(n)
    train_data = acts_norm[perm[:n_train]]
    val_data = acts_norm[perm[n_train:]]
    del acts_norm
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    # Create model
    model = MatryoshkaBatchTopKSAE(
        d_input=d_input,
        d_hidden=args.dict_size,
        k=args.k,
        prefixes=prefixes,
        k_aux=args.k_aux,
        aux_alpha=args.aux_alpha,
    ).to(device)

    print(f"\nModel: MatryoshkaBatchTopKSAE")
    print(f"  d_input={d_input}, d_hidden={args.dict_size}, k={args.k}")
    print(f"  prefixes={prefixes}")
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

    # Per-prefix evaluation
    print("\n--- Per-Prefix Evaluation ---")
    prefix_eval = evaluate_per_prefix(model, val_loader, device)

    # Save
    os.makedirs(OUTPUT, exist_ok=True)
    prefix_str = "_".join(str(p) for p in prefixes)
    if args.output:
        out_path = args.output
    else:
        out_path = f"{OUTPUT}/maia3_matryoshka_{args.dict_size}_k{args.k}_p{prefix_str}.pt"

    save_payload = {
        "state_dict": model.state_dict(),
        "config": {
            "model": "maia3",
            "architecture": "MatryoshkaBatchTopKSAE",
            "d_input": d_input,
            "dict_size": args.dict_size,
            "k": args.k,
            "prefixes": prefixes,
            "k_aux": args.k_aux,
            "aux_alpha": args.aux_alpha,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "warmup_steps": args.warmup_steps,
            "normalization": "zscore_l2",
            "n_train": n_train,
            "n_val": n_val,
            "seed": args.seed,
        },
        "training_log": log,
        "prefix_eval": prefix_eval,
        "source_activations": act_path,
    }

    torch.save(save_payload, out_path)
    print(f"\nSaved to {out_path}")

    # Final summary
    print("\n--- FINAL SUMMARY ---")
    print(f"  Architecture: Matryoshka BatchTopK SAE")
    print(f"  Dict: {args.dict_size}, k: {args.k}, Prefixes: {prefixes}")
    print(f"  Training: {args.n_epochs} epochs, {elapsed:.0f}s")
    for psize, peval in prefix_eval.items():
        print(f"  Prefix {psize}: FVU={peval['fvu']:.4f}, L0={peval['l0']:.1f}, "
              f"Dead={peval['n_dead']}/{psize}")


if __name__ == "__main__":
    main()
