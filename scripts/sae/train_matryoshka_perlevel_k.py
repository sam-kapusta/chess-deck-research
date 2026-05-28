#!/usr/bin/env python3
"""Matryoshka SAE with per-level k enforcement.

Instead of global BatchTopK, each prefix level gets its own k budget:
  - Level 1 (top): k=1 (one category per blunder)
  - Level 2 (mid): k=5 (subcategories)
  - Level 3 (bot): k=16 (atoms)

This prevents parents from hogging the sparsity budget and ensures
the top level acts as a pure classifier.

Usage (on chess-poc):
    cd ~/SageMaker && python3 scripts/sae/train_matryoshka_perlevel_k.py
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


class MatryoshkaPerLevelKSAE(nn.Module):
    """Matryoshka SAE with per-level topk enforcement.

    Each level gets its own k budget. TopK is applied independently
    within each group of latents rather than globally.
    """

    def __init__(self, d_input, d_hidden, prefixes, k_per_level,
                 k_aux=256, aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        assert len(prefixes) == len(k_per_level)
        assert prefixes[-1] == d_hidden

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.prefixes = prefixes
        self.k_per_level = k_per_level
        self.total_k = sum(k_per_level)
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.n_batches_to_dead = n_batches_to_dead

        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def _apply_per_level_topk(self, z_relu, batch_size):
        """Apply topk independently within each level's latent group."""
        acts = torch.zeros_like(z_relu)

        for i, (prefix_size, k) in enumerate(zip(self.prefixes, self.k_per_level)):
            start = 0 if i == 0 else self.prefixes[i - 1]
            end = prefix_size
            group_size = end - start

            group_z = z_relu[:, start:end]
            group_k = min(k * batch_size, group_z.numel())

            flat_group = group_z.reshape(-1)
            topk_vals, topk_idx = torch.topk(flat_group, k=group_k)

            group_acts = torch.zeros_like(flat_group)
            group_acts[topk_idx] = topk_vals
            acts[:, start:end] = group_acts.reshape(batch_size, group_size)

        return acts

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        batch_size = x.shape[0]
        z_relu = F.relu(z)

        acts = self._apply_per_level_topk(z_relu, batch_size)

        if self.training:
            feature_active = (acts > 0).any(dim=0)
            self.num_batches_not_active[feature_active] = 0
            self.num_batches_not_active[~feature_active] += 1

        # Matryoshka reconstruction losses at each prefix
        prefix_losses = []
        current_output = self.b_dec.clone().unsqueeze(0).expand(batch_size, -1)
        for i, prefix_size in enumerate(self.prefixes):
            start = 0 if i == 0 else self.prefixes[i - 1]
            group_acts = acts[:, start:prefix_size]
            group_decoder = self.W_dec[start:prefix_size]
            current_output = current_output + group_acts @ group_decoder
            prefix_losses.append((current_output - x).pow(2).mean())

        l2_loss = sum(prefix_losses)
        x_hat = current_output

        # Aux loss for dead features
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

        return l2_loss + aux_loss, x_hat, acts, prefix_losses

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


def train_and_eval(prefixes, k_per_level, train_data, val_data, device, tag):
    dict_size = prefixes[-1]
    total_k = sum(k_per_level)
    print(f"\n{'='*60}")
    print(f"Config {tag}: prefixes={prefixes}, k_per_level={k_per_level}")
    print(f"  dict={dict_size}, total_k={total_k}")
    print(f"{'='*60}")
    sys.stdout.flush()

    torch.manual_seed(42)
    model = MatryoshkaPerLevelKSAE(
        d_input=512, d_hidden=dict_size,
        prefixes=prefixes, k_per_level=k_per_level,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=4096,
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=4096,
        shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for epoch in range(200):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, x_hat, acts, prefix_losses = model(batch)
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

    # Evaluate
    model.eval()
    all_acts = []
    all_x = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                _, _, acts, _ = model(batch)
            all_acts.append(acts.cpu())
            all_x.append(batch.cpu())

    acts_np = torch.cat(all_acts).numpy()
    x_np = torch.cat(all_x).numpy()
    var_x = float(np.var(x_np))

    print(f"\n  --- Results: {tag} ---")
    results = {"tag": tag, "prefixes": prefixes, "k_per_level": k_per_level, "levels": {}}

    for i, prefix_size in enumerate(prefixes):
        start = 0 if i == 0 else prefixes[i - 1]
        group_acts = acts_np[:, start:prefix_size]
        group_size = prefix_size - start

        freq = (group_acts > 0).mean(axis=0)
        n_dead = int((freq == 0).sum())
        l0 = float((group_acts > 0).sum(axis=1).mean())
        avg_fire = float(freq.mean())
        max_fire = float(freq.max())

        # FVU at this prefix (cumulative)
        prefix_acts = acts_np[:, :prefix_size]
        Wd_prefix = model.W_dec[:prefix_size].detach().cpu()
        prefix_acts_t = torch.tensor(prefix_acts, dtype=torch.float32)
        recon = (prefix_acts_t @ Wd_prefix + model.b_dec.detach().cpu()).numpy()
        mse = float(np.mean((x_np - recon) ** 2))
        fvu = mse / var_x

        # Orthogonality for this group
        Wd_group = model.W_dec[start:prefix_size].detach()
        Wn = Wd_group / Wd_group.norm(dim=-1, keepdim=True)
        sim = Wn @ Wn.T
        sim.fill_diagonal_(-1)
        avg_max_cos = sim.max(dim=-1).values.mean().item()

        level = {
            "group": f"[{start}:{prefix_size}]",
            "group_size": group_size,
            "k": k_per_level[i],
            "fvu": fvu,
            "l0_group": l0,
            "n_dead": n_dead,
            "avg_fire": avg_fire,
            "max_fire": max_fire,
            "avg_max_cos": avg_max_cos,
        }
        results["levels"][prefix_size] = level

        print(f"  Group [{start}:{prefix_size}] (k={k_per_level[i]}): "
              f"FVU={fvu:.3f} L0={l0:.1f} dead={n_dead}/{group_size} | "
              f"fire={avg_fire*100:.1f}% max={max_fire*100:.0f}% | "
              f"cos={avg_max_cos:.3f}")
    sys.stdout.flush()

    # Save
    prefix_str = "_".join(str(p) for p in prefixes)
    k_str = "_".join(str(k) for k in k_per_level)
    save_path = f"{OUTPUT}/maia3_matryoshka_perlevel_{dict_size}_p{prefix_str}_k{k_str}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "d_input": 512, "dict_size": dict_size,
            "prefixes": prefixes, "k_per_level": k_per_level,
            "total_k": total_k, "architecture": "MatryoshkaPerLevelK",
        },
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
    data = torch.load(BASE + "/cache/maia3_blunder_diff.pt", map_location="cpu", weights_only=False)
    raw_acts = data["activations"].float()
    mean = raw_acts.mean(dim=0)
    std = raw_acts.std(dim=0).clamp(min=1e-6)
    x = (raw_acts - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw_acts

    torch.manual_seed(42)
    n = x.shape[0]
    n_val = int(n * 0.1)
    perm = torch.randperm(n)
    train_data = x[perm[:n - n_val]]
    val_data = x[perm[n - n_val:]]
    del x
    print(f"Train: {train_data.shape[0]}, Val: {val_data.shape[0]}")
    sys.stdout.flush()

    # Groups: 32, 256, 2048. Dict = 2336. Prefixes = [32, 288, 2336]
    # Per-level k from sweep: dict=32→k=3, dict=256→k=8, dict=2048→k=16
    configs = [
        ("H1", [32, 288, 2336], [3, 8, 16]),
        ("H2", [32, 288, 2336], [2, 6, 16]),
        ("H3", [32, 288, 2336], [4, 8, 16]),
        ("H4", [32, 288, 2336], [3, 6, 12]),
    ]

    all_results = {}
    for tag, prefixes, k_per_level in configs:
        result = train_and_eval(prefixes, k_per_level, train_data, val_data, device, tag)
        all_results[tag] = result

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for tag, res in all_results.items():
        print(f"\n{tag}: k_per_level={res['k_per_level']}")
        for psize, lv in res["levels"].items():
            print(f"  {lv['group']} k={lv['k']}: FVU={lv['fvu']:.3f} "
                  f"L0={lv['l0_group']:.1f} fire={lv['avg_fire']*100:.1f}% "
                  f"max={lv['max_fire']*100:.0f}% cos={lv['avg_max_cos']:.3f} "
                  f"dead={lv['n_dead']}")

    with open(f"{OUTPUT}/matryoshka_perlevel_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT}/matryoshka_perlevel_comparison.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
