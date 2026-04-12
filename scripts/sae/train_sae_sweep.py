#!/usr/bin/env python3
"""SAE hyperparameter sweep with structural metrics.

Trains SAEs at multiple dict sizes on Maia activations.
Reports: MSE, explained variance, dead features, decoder cosine similarity,
feature density, bimodality.

Run on SAIS notebook:
    pip install maia2 einops
    python3 train_sae_sweep.py --n-positions 20000
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


class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.dict_size = dict_size
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=-1)

    def forward(self, x):
        x_centered = x - self.pre_bias
        z = self.encoder(x_centered)
        if self.training:
            batch_size = x.shape[0]
            z_relu = F.relu(z)
            flat_z = z_relu.reshape(-1)
            topk_vals, topk_idx = torch.topk(flat_z, k=min(batch_size * self.k, flat_z.numel()))
            acts = torch.zeros_like(flat_z)
            acts[topk_idx] = topk_vals
            acts = acts.reshape(z.shape)
        else:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
            acts = torch.zeros_like(z)
            acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        x_hat = self.decoder(acts) + self.pre_bias
        return x_hat, acts


def structural_metrics(sae, acts_norm):
    """Compute T1 structural metrics (from Sandstone persona pipeline)."""
    sae.eval()
    with torch.no_grad():
        x_hat, z = sae(acts_norm)
        mse = F.mse_loss(x_hat, acts_norm).item()
        var_data = acts_norm.var(dim=0).mean().item()
        explained = 1 - mse / var_data

        # Per-feature stats
        active_mask = (z > 0).any(dim=0)  # which features ever fire
        active_count = active_mask.sum().item()
        dead_count = sae.dict_size - active_count

        # L0: average number of active features per input
        l0 = (z > 0).float().sum(dim=-1).mean().item()

        # Feature density: how often each feature fires
        fire_rate = (z > 0).float().mean(dim=0)  # [dict_size]
        avg_fire_rate = fire_rate[active_mask].mean().item() if active_count > 0 else 0

        # Decoder cosine similarity: how similar are decoder columns?
        # High similarity = features are redundant
        dec_w = sae.decoder.weight.data  # [input_dim, dict_size]
        dec_normed = F.normalize(dec_w, dim=0)
        cos_sim = (dec_normed.T @ dec_normed)  # [dict_size, dict_size]
        # Mean off-diagonal cosine similarity
        mask = ~torch.eye(sae.dict_size, dtype=torch.bool)
        mean_cos_sim = cos_sim[mask].mean().item()
        max_cos_sim = cos_sim[mask].max().item()

        # Bimodality: for each active feature, is the activation distribution bimodal?
        # Simple proxy: fraction of inputs where feature is exactly 0 vs > 0
        # A good feature has clear on/off states
        if active_count > 0:
            active_features = z[:, active_mask]  # [N, n_active]
            sparsity = (active_features == 0).float().mean(dim=0)
            avg_sparsity = sparsity.mean().item()
        else:
            avg_sparsity = 1.0

    return {
        'mse': mse,
        'explained_variance': explained,
        'active': active_count,
        'dead': dead_count,
        'l0': l0,
        'avg_fire_rate': avg_fire_rate,
        'mean_decoder_cosine': mean_cos_sim,
        'max_decoder_cosine': max_cos_sim,
        'avg_feature_sparsity': avg_sparsity,
    }


def extract_maia_activations(n, data_path):
    """Extract Maia hidden states. Works on any machine with maia2 installed."""
    from maia2 import model as maia_model, inference as maia_inference
    import chess

    print(f"Loading Maia-2...", flush=True)
    model = maia_model.from_pretrained(type="rapid", device="cpu")
    prepared = maia_inference.prepare()

    # Hook to capture hidden state
    hiddens = []
    def hook_fn(module, input, output):
        hiddens.append(output.detach())
    model.last_ln.register_forward_hook(hook_fn)

    print(f"Extracting {n} activations...", flush=True)
    acts = []
    all_moves_dict, elo_dict, _ = prepared

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            fen = json.loads(line.strip()).get('fen', '')
            if not fen:
                continue
            try:
                bi, es, eo, _ = maia_inference.preprocessing(fen, 1800, 1800, elo_dict, all_moves_dict)
                hiddens.clear()
                with torch.no_grad():
                    model(bi.unsqueeze(0), torch.tensor([es]), torch.tensor([eo]))
                if hiddens:
                    acts.append(hiddens[0].squeeze(0).unsqueeze(0))
            except:
                continue
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{n}", flush=True)

    all_acts = torch.cat(acts, dim=0).float()
    print(f"Extracted {all_acts.shape[0]} activations (dim={all_acts.shape[1]})", flush=True)
    return all_acts


def train_sae(acts_norm, dict_size, k, epochs=5, batch_size=256, lr=3e-4):
    sae = BatchTopKSAE(acts_norm.shape[1], dict_size, k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(acts_norm), batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        sae.train()
        for (batch,) in loader:
            x_hat, z = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=-1)

    return sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/samtkap/workspace/chess-coach/research/data/multitask_moments.jsonl')
    parser.add_argument('--n-positions', type=int, default=20000)
    parser.add_argument('--output-dir', default='/Users/samtkap/workspace/chess-coach/research/sae')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    acts = extract_maia_activations(args.n_positions, args.data)
    mean = acts.mean(dim=0)
    std = acts.std(dim=0).clamp(min=1e-6)
    acts_norm = (acts - mean) / std

    configs = [
        (512, 32),
        (1024, 64),
        (2048, 128),
        (4096, 256),
    ]

    all_results = []
    for dict_size, k in configs:
        print(f"\n=== dict_size={dict_size}, k={k} ===", flush=True)
        t0 = time.time()
        sae = train_sae(acts_norm, dict_size, k, epochs=args.epochs)
        metrics = structural_metrics(sae, acts_norm)
        elapsed = time.time() - t0
        metrics['dict_size'] = dict_size
        metrics['k'] = k
        metrics['train_time'] = elapsed
        all_results.append(metrics)

        # Print
        print(f"  MSE: {metrics['mse']:.4f} | Explained var: {metrics['explained_variance']:.1%}")
        print(f"  Active: {metrics['active']}/{dict_size} | Dead: {metrics['dead']} | L0: {metrics['l0']:.0f}")
        print(f"  Decoder cos sim: mean={metrics['mean_decoder_cosine']:.3f} max={metrics['max_decoder_cosine']:.3f}")
        print(f"  Feature sparsity: {metrics['avg_feature_sparsity']:.3f} | Fire rate: {metrics['avg_fire_rate']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Save checkpoint
        out = os.path.join(args.output_dir, f'maia_sae_{dict_size}.pt')
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': {'input_dim': acts.shape[1], 'dict_size': dict_size, 'k': k},
            'normalization': {'mean': mean.numpy().tolist(), 'std': std.numpy().tolist()},
            'metrics': metrics,
            'n_positions': acts.shape[0],
        }, out)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Size':>6s} {'k':>4s} {'MSE':>8s} {'ExplVar':>8s} {'Active':>7s} {'Dead':>5s} {'L0':>5s} {'DecCos':>7s} {'Sparse':>7s} {'Time':>6s}")
    for r in all_results:
        print(f"{r['dict_size']:>6d} {r['k']:>4d} {r['mse']:>8.4f} {r['explained_variance']:>7.1%} "
              f"{r['active']:>7d} {r['dead']:>5d} {r['l0']:>5.0f} {r['mean_decoder_cosine']:>7.3f} "
              f"{r['avg_feature_sparsity']:>7.3f} {r['train_time']:>5.1f}s")

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
