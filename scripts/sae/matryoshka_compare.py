#!/usr/bin/env python3
"""Compare two Matryoshka prefix configurations.

Config A: [64, 256, 2048]
Config B: [32, 128, 512, 2048]

Both at k=16, dict=2048, 200 epochs on Maia 3 blunder diff vectors.

Usage (on chess-poc):
    cd ~/SageMaker && python3 scripts/sae/matryoshka_compare.py
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
OUTPUT = BASE + "/output/maia3_sae"
os.makedirs(OUTPUT, exist_ok=True)


class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k, prefixes, k_aux=256,
                 aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
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

        prefix_losses = []
        current_output = self.b_dec.clone().unsqueeze(0).expand(batch_size, -1)
        for i, prefix_size in enumerate(self.prefixes):
            start_idx = 0 if i == 0 else self.prefixes[i - 1]
            group_acts = acts[:, start_idx:prefix_size]
            group_decoder = self.W_dec[start_idx:prefix_size]
            current_output = current_output + group_acts @ group_decoder
            prefix_losses.append((current_output - x).pow(2).mean())

        l2_loss = sum(prefix_losses)
        x_hat = current_output

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

        return l2_loss + aux_loss, x_hat, acts, l2_loss, aux_loss, prefix_losses

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def get_prefix_acts(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        batch_size = x.shape[0]
        total_k = batch_size * self.k
        z_relu = F.relu(z)
        flat_z = z_relu.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat_z, k=min(int(total_k), flat_z.numel()))
        acts = torch.zeros_like(flat_z)
        acts[topk_idx] = topk_vals
        return acts.reshape(z.shape)


def train_and_eval(prefixes, train_data, val_data, device, tag):
    dict_size = prefixes[-1]
    print(f"\n{'='*60}")
    print(f"TRAINING: {tag} prefixes={prefixes}, k=16, dict={dict_size}")
    print(f"{'='*60}")
    sys.stdout.flush()

    torch.manual_seed(42)
    model = MatryoshkaBatchTopKSAE(
        d_input=512, d_hidden=dict_size, k=16, prefixes=prefixes
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=4096,
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=4096,
        shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")
    target_k = 16
    n_steps = 0
    t0 = time.time()

    for epoch in range(200):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            n_steps += 1
            if n_steps <= 500:
                model.k = max(1, int(1 + (target_k - 1) * n_steps / 500))
            else:
                model.k = target_k
            with torch.autocast("cuda", dtype=torch.float16):
                loss, x_hat, acts, l2_loss, aux_loss, pl = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.make_decoder_weights_and_grad_unit_norm()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200 | Loss={loss.item():.4f}")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s")
    sys.stdout.flush()

    # Evaluate per-prefix
    model.eval()
    model.k = target_k
    all_acts = []
    all_x = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=True)
            acts = model.get_prefix_acts(batch)
            all_acts.append(acts.cpu())
            all_x.append(batch.cpu())

    acts_np = torch.cat(all_acts).numpy()
    x_np = torch.cat(all_x).numpy()
    var_x = float(np.var(x_np))

    print(f"\n  --- Results: {tag} ---")
    results = {"tag": tag, "prefixes": prefixes, "levels": {}}

    for pi, psize in enumerate(prefixes):
        prefix_acts = acts_np[:, :psize]
        freq = (prefix_acts > 0).mean(axis=0)
        n_dead = int((freq == 0).sum())
        l0 = float((prefix_acts > 0).sum(axis=1).mean())
        avg_fire = float(freq.mean())
        max_fire = float(freq.max())

        # FVU at this prefix
        prefix_acts_t = torch.tensor(prefix_acts, dtype=torch.float32)
        W_dec_prefix = model.W_dec[:psize].detach().cpu()
        xhat_prefix = (prefix_acts_t @ W_dec_prefix + model.b_dec.detach().cpu()).numpy()
        mse = float(np.mean((x_np - xhat_prefix) ** 2))
        fvu = mse / var_x

        # Orthogonality
        Wd = model.W_dec[:psize].data
        Wn = Wd / Wd.norm(dim=-1, keepdim=True)
        sim = Wn @ Wn.T
        sim.fill_diagonal_(-1)
        avg_max_cos = sim.max(dim=-1).values.mean().item()

        # Fire rate buckets
        in_01_1 = int(((freq >= 0.001) & (freq <= 0.01)).sum())
        in_1_10 = int(((freq >= 0.01) & (freq <= 0.10)).sum())
        in_10_30 = int(((freq >= 0.10) & (freq <= 0.30)).sum())
        above_30 = int((freq > 0.30).sum())

        level = {
            "fvu": fvu, "l0": l0, "n_dead": n_dead,
            "avg_fire": avg_fire, "max_fire": max_fire,
            "avg_max_cos": avg_max_cos,
            "in_01_1pct": in_01_1, "in_1_10pct": in_1_10,
            "in_10_30pct": in_10_30, "above_30pct": above_30,
        }
        results["levels"][psize] = level

        print(f"  Prefix {psize:>4}: FVU={fvu:.3f} L0={l0:.1f} dead={n_dead}/{psize} | "
              f"fire={avg_fire*100:.1f}% max={max_fire*100:.0f}% | "
              f"cos={avg_max_cos:.3f} | <1%={in_01_1} 1-10%={in_1_10} "
              f"10-30%={in_10_30} >30%={above_30}")
    sys.stdout.flush()

    # Save weights
    prefix_str = "_".join(str(p) for p in prefixes)
    save_path = f"{OUTPUT}/maia3_matryoshka_2048_k16_p{prefix_str}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"d_input": 512, "dict_size": 2048, "k": 16, "prefixes": prefixes},
        "eval_results": results,
    }, save_path)
    print(f"  Saved: {save_path}")
    sys.stdout.flush()

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading activations...")
    sys.stdout.flush()
    data = torch.load(
        BASE + "/cache/maia3_blunder_diff.pt",
        map_location="cpu", weights_only=False)
    raw_acts = data["activations"].float()

    mean = raw_acts.mean(dim=0)
    std = raw_acts.std(dim=0).clamp(min=1e-6)
    x_zscore = (raw_acts - mean) / std
    norms = x_zscore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    acts_norm = x_zscore / norms
    del raw_acts

    torch.manual_seed(42)
    n = acts_norm.shape[0]
    n_val = int(n * 0.1)
    perm = torch.randperm(n)
    train_data = acts_norm[perm[:n - n_val]]
    val_data = acts_norm[perm[n - n_val:]]
    del acts_norm
    print(f"Train: {train_data.shape[0]}, Val: {val_data.shape[0]}")
    sys.stdout.flush()

    all_configs = [
        ("A", [64, 256, 2048]),
        ("B", [32, 128, 512, 2048]),
        ("C", [32, 160, 672, 2720]),
        ("D", [64, 320, 1344, 5440]),
        ("E", [32, 96, 224, 480, 992, 2048]),
    ]

    all_results = {}
    for tag, prefixes in all_configs:
        all_results[tag] = train_and_eval(prefixes, train_data, val_data, device, tag)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    configs = [(f"{tag}: {res['prefixes']}", res) for tag, res in all_results.items()]
    for label, res in configs:
        print(f"\nConfig {label}")
        for psize, lv in res["levels"].items():
            print(f"  {psize:>4}: FVU={lv['fvu']:.3f} L0={lv['l0']:.1f} "
                  f"fire={lv['avg_fire']*100:.1f}% max={lv['max_fire']*100:.0f}% "
                  f"cos={lv['avg_max_cos']:.3f} 1-10%={lv['in_1_10pct']}")

    with open(f"{OUTPUT}/matryoshka_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT}/matryoshka_comparison.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
