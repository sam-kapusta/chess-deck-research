#!/usr/bin/env python3
"""Full SAE hyperparameter sweep on v2 (corrected) data.

Replicates all key experiments from the v1 session on the correct dataset:
1. k-sweep at dict=2048 (find optimal k)
2. Small-dict sweep (dict=32/64/128/256 at various k)
3. Per-level-k Matryoshka training (H1 and L3 configs)

Usage (on chess-poc):
    cd ~/SageMaker && python3 scripts/sae/full_sweep_v2.py
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
OUTPUT = BASE + "/output/maia3_sae_v2"
os.makedirs(OUTPUT, exist_ok=True)

# Use V2 data
ACTIVATIONS_PATH = BASE + "/cache/maia3_blunder_diff_v2.pt"


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k, k_aux=256, aux_alpha=1/32, n_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden
        self.k = k
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.n_dead_threshold = n_dead
        self.register_buffer("dead_cnt", torch.zeros(d_hidden))

    def forward(self, x):
        z = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        flat = z.reshape(-1)
        total_k = min(x.shape[0] * self.k, flat.numel())
        tv, ti = torch.topk(flat, k=total_k)
        acts = torch.zeros_like(flat)
        acts[ti] = tv
        acts = acts.reshape(z.shape)

        if self.training:
            active = (acts > 0).any(dim=0)
            self.dead_cnt[active] = 0
            self.dead_cnt[~active] += 1

        x_hat = acts @ self.W_dec + self.b_dec
        loss = (x_hat - x).pow(2).mean()

        if self.training and self.k_aux > 0:
            dead = self.dead_cnt >= self.n_dead_threshold
            if dead.sum() > 0:
                err = (x - x_hat).detach()
                dp = F.relu(((x - self.b_dec) @ self.W_enc + self.b_enc)[:, dead])
                ka = min(self.k_aux, int(dead.sum()))
                tkv = torch.topk(dp, k=ka, dim=-1)
                da = torch.zeros_like(dp).scatter(-1, tkv.indices, tkv.values)
                loss = loss + self.aux_alpha * (da @ self.W_dec[dead] - err).pow(2).mean()

        return loss, x_hat, acts

    @torch.no_grad()
    def norm_decoder(self):
        n = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.grad -= (self.W_dec.grad * n).sum(-1, keepdim=True) * n
        self.W_dec.data = n


class MatryoshkaPerLevelK(nn.Module):
    def __init__(self, d_input, d_hidden, prefixes, k_per_level, k_aux=256, aux_alpha=1/32):
        super().__init__()
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.prefixes = prefixes
        self.k_per_level = k_per_level
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.register_buffer("dead_cnt", torch.zeros(d_hidden))

    def forward(self, x):
        z = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        bs = x.shape[0]
        acts = torch.zeros_like(z)

        for i, (ps, k) in enumerate(zip(self.prefixes, self.k_per_level)):
            s = 0 if i == 0 else self.prefixes[i - 1]
            g = z[:, s:ps]
            fg = g.reshape(-1)
            gk = min(k * bs, fg.numel())
            tv, ti = torch.topk(fg, k=gk)
            ga = torch.zeros_like(fg)
            ga[ti] = tv
            acts[:, s:ps] = ga.reshape(bs, ps - s)

        if self.training:
            active = (acts > 0).any(dim=0)
            self.dead_cnt[active] = 0
            self.dead_cnt[~active] += 1

        # Matryoshka losses
        prefix_losses = []
        cur = self.b_dec.unsqueeze(0).expand(bs, -1)
        for i, ps in enumerate(self.prefixes):
            s = 0 if i == 0 else self.prefixes[i - 1]
            cur = cur + acts[:, s:ps] @ self.W_dec[s:ps]
            prefix_losses.append((cur - x).pow(2).mean())
        l2_loss = sum(prefix_losses)
        x_hat = cur

        # Aux loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.k_aux > 0:
            dead = self.dead_cnt >= 5
            if dead.sum() > 0:
                err = (x - x_hat).detach()
                dp = F.relu(((x - self.b_dec) @ self.W_enc + self.b_enc)[:, dead])
                ka = min(256, int(dead.sum()))
                tkv = torch.topk(dp, k=ka, dim=-1)
                da = torch.zeros_like(dp).scatter(-1, tkv.indices, tkv.values)
                aux_loss = self.aux_alpha * (da @ self.W_dec[dead] - err).pow(2).mean()

        return l2_loss + aux_loss, x_hat, acts

    @torch.no_grad()
    def norm_decoder(self):
        n = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.grad -= (self.W_dec.grad * n).sum(-1, keepdim=True) * n
        self.W_dec.data = n


def load_v2_data():
    print(f"Loading v2 activations from {ACTIVATIONS_PATH}")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    raw = data["activations"].float()

    # Normalize (z-score + L2)
    mean = raw.mean(dim=0)
    std = raw.std(dim=0).clamp(min=1e-6)
    x = (raw - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw

    # Train/val split
    torch.manual_seed(42)
    n = x.shape[0]
    n_val = int(n * 0.1)
    perm = torch.randperm(n)
    train_data = x[perm[:n - n_val]]
    val_data = x[perm[n - n_val:]]
    del x

    print(f"  Shape: {train_data.shape[1]}-dim, Train: {train_data.shape[0]}, Val: {val_data.shape[0]}")
    return train_data, val_data, data


def train_sae(model, train_loader, n_epochs=100, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(n_epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _, _ = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.norm_decoder()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    return model


def evaluate(model, val_loader, device="cuda"):
    model.eval()
    all_acts = []
    all_x = []
    all_xhat = []
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                _, xh, a = model(batch)
            all_acts.append((a > 0).cpu())
            all_x.append(batch.cpu())
            all_xhat.append(xh.cpu())
    acts = torch.cat(all_acts).float().numpy()
    x_np = torch.cat(all_x).numpy()
    xh_np = torch.cat(all_xhat).numpy()

    freq = acts.mean(axis=0)
    dead = int((freq == 0).sum())
    fvu = float(np.mean((x_np - xh_np)**2)) / float(np.var(x_np))

    Wd = model.W_dec.data
    Wn = Wd / Wd.norm(dim=-1, keepdim=True)
    sim = Wn @ Wn.T
    sim.fill_diagonal_(-1)
    cos = sim.max(dim=-1).values.mean().item()

    interp = int(((freq >= 0.001) & (freq <= 0.01)).sum())
    max_fire = float(freq.max())

    return {
        "dead": dead, "fvu": fvu, "cos": cos,
        "interp": interp, "max_fire": max_fire,
        "avg_fire": float(freq.mean()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sys.stdout.flush()

    train_data, val_data, raw_data = load_v2_data()
    train_loader = DataLoader(TensorDataset(train_data), batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=4096, shuffle=False, num_workers=2, pin_memory=True)

    results = {}

    # === PHASE 1: k-sweep at dict=2048 ===
    print("\n" + "=" * 60)
    print("PHASE 1: k-sweep at dict=2048")
    print("=" * 60)
    sys.stdout.flush()

    for k in [8, 12, 16, 20, 24, 32]:
        torch.manual_seed(42)
        model = BatchTopKSAE(512, 2048, k).to(device)
        t0 = time.time()
        model = train_sae(model, train_loader, n_epochs=50, device=device)
        elapsed = time.time() - t0
        metrics = evaluate(model, val_loader, device)
        tag = f"d2048_k{k}"
        results[tag] = metrics
        print(f"  k={k:2d}: dead={metrics['dead']:4d} interp={metrics['interp']:4d} "
              f"cos={metrics['cos']:.3f} fvu={metrics['fvu']:.3f} max={metrics['max_fire']*100:.0f}% [{elapsed:.0f}s]")
        sys.stdout.flush()

        # Save the k=16 model
        if k == 16:
            torch.save({"state_dict": model.state_dict(), "config": {"dict_size": 2048, "k": 16}},
                       f"{OUTPUT}/sweep_v2_k16_d2048.pt")

    # === PHASE 2: Small-dict sweep ===
    print("\n" + "=" * 60)
    print("PHASE 2: Small-dict sweep")
    print("=" * 60)
    sys.stdout.flush()

    small_configs = [
        (32, 3), (32, 6), (64, 6), (64, 8), (128, 8), (128, 12), (256, 8), (256, 12), (512, 10), (512, 12)
    ]
    for dd, k in small_configs:
        torch.manual_seed(42)
        model = BatchTopKSAE(512, dd, k).to(device)
        model = train_sae(model, train_loader, n_epochs=100, device=device)
        metrics = evaluate(model, val_loader, device)
        tag = f"d{dd}_k{k}"
        results[tag] = metrics
        print(f"  d={dd:3d} k={k:2d}: dead={metrics['dead']:3d} cos={metrics['cos']:.3f} "
              f"fvu={metrics['fvu']:.3f} max={metrics['max_fire']*100:.0f}% interp={metrics['interp']}")
        sys.stdout.flush()

    # === PHASE 3: Per-level-k Matryoshka ===
    print("\n" + "=" * 60)
    print("PHASE 3: Per-level-k Matryoshka")
    print("=" * 60)
    sys.stdout.flush()

    matryoshka_configs = [
        ("H1", [32, 288, 2336], [3, 8, 16]),
        ("L3", [128, 640, 2688], [8, 12, 16]),
    ]

    for tag, prefixes, k_per_level in matryoshka_configs:
        dd = prefixes[-1]
        print(f"\n  {tag}: prefixes={prefixes}, k={k_per_level}, dict={dd}")
        sys.stdout.flush()

        torch.manual_seed(42)
        model = MatryoshkaPerLevelK(512, dd, prefixes, k_per_level).to(device)
        t0 = time.time()
        model = train_sae(model, train_loader, n_epochs=200, device=device)
        elapsed = time.time() - t0

        # Per-level eval
        model.eval()
        all_acts = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    _, _, a = model(batch)
                all_acts.append(a.cpu())
        acts = torch.cat(all_acts).numpy()

        level_results = {}
        for i, ps in enumerate(prefixes):
            s = 0 if i == 0 else prefixes[i - 1]
            g_acts = acts[:, s:ps]
            freq = (g_acts > 0).mean(axis=0)
            nd = int((freq == 0).sum())
            l0 = float((g_acts > 0).sum(axis=1).mean())
            mf = float(freq.max())
            Wn = model.W_dec[s:ps].detach()
            Wn = Wn / Wn.norm(dim=-1, keepdim=True)
            sim = Wn @ Wn.T
            sim.fill_diagonal_(-1)
            cos = sim.max(dim=-1).values.mean().item()
            level_results[ps] = {"dead": nd, "l0": l0, "max_fire": mf, "cos": cos}
            print(f"    [{s}:{ps}] k={k_per_level[i]}: dead={nd}/{ps-s} "
                  f"fire={freq.mean()*100:.1f}% max={mf*100:.0f}% cos={cos:.3f}")

        results[tag] = {"levels": level_results, "elapsed": elapsed}
        sys.stdout.flush()

        # Save model
        save_path = f"{OUTPUT}/matryoshka_v2_{tag}_p{'_'.join(str(p) for p in prefixes)}_k{'_'.join(str(k) for k in k_per_level)}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"prefixes": prefixes, "k_per_level": k_per_level, "dict_size": dd},
        }, save_path)
        print(f"    Saved: {save_path}")
        sys.stdout.flush()

    # Save all results
    with open(f"{OUTPUT}/full_sweep_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved: {OUTPUT}/full_sweep_v2_results.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
