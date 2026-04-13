#!/usr/bin/env python3
"""Train BTK SAE on blunder activation cache.

Loads cached encoder activations for blunder moves (bad moves that lost ≥200cp)
and trains a BatchTopK SAE with auxiliary dead-feature loss.

Usage (on chess-research notebook):
    python3 train_blunder_sae.py
    python3 train_blunder_sae.py --dict-size 4096 --k 32
    python3 train_blunder_sae.py --use-best  # train on best moves instead
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

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
CACHE = BASE + '/cache/blunder_acts_200k.pt'
OUTPUT = BASE + '/output/blunder_sae'


class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k
        self.dd = dd

    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros_like(z)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


def train_sae(flat, dict_size, k, epochs=5, batch_size=2048, lr=1e-3):
    n = flat.shape[0]
    print(f'Training SAE: dict={dict_size}, k={k}, {n} activations, {epochs} epochs')

    sae = SAE(1024, dict_size, k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    steps_since_fired = torch.zeros(dict_size, device='cuda')

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_mse = 0
        total_aux = 0
        nb = 0
        for i in range(0, n, batch_size):
            batch = flat[perm[i:i + batch_size]].cuda()
            recon, acts = sae(batch)
            mse = F.mse_loss(recon, batch)

            fired = (acts > 0).any(dim=0)
            steps_since_fired[fired] = 0
            steps_since_fired[~fired] += 1
            dead = steps_since_fired > 50

            aux = torch.tensor(0.0, device='cuda')
            if dead.sum() > 0:
                res = (batch - recon).detach()
                de = sae.encoder.weight[dead] @ res.T
                da = F.relu(de).T
                dr = da @ sae.decoder.weight[:, dead].T
                aux = F.mse_loss(dr, res)

            loss = mse + (1 / 32) * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_mse += mse.item()
            total_aux += aux.item()
            nb += 1

        nd = (steps_since_fired > 50).sum().item()
        print(f'  ep{ep} mse={total_mse / nb:.6f} aux={total_aux / nb:.6f} dead={nd}/{dict_size}')
        sys.stdout.flush()

    # Final eval
    with torch.no_grad():
        sample = flat[:min(50000, n)].cuda()
        _, sa = sae(sample)
        dead_n = ((sa > 0).sum(dim=0) == 0).sum().item()
        l0 = (sa > 0).float().sum(dim=-1).mean().item()
        recon, _ = sae(sample)
        fvu = F.mse_loss(recon, sample).item() / sample.var().item()
        dec_w = sae.decoder.weight.data
        dec_n = F.normalize(dec_w, dim=0)
        cos = dec_n.T @ dec_n
        mask = ~torch.eye(dict_size, dtype=torch.bool, device=cos.device)
        c_dec = cos[mask].abs().mean().item()

    elapsed = time.time() - t0
    alive = dict_size - dead_n
    print(f'\nResults: dead={dead_n} alive={alive} L0={l0:.1f} FVU={fvu:.4f} c_dec={c_dec:.4f} ({elapsed:.0f}s)')
    return sae, {'dead': dead_n, 'alive': alive, 'L0': l0, 'FVU': fvu, 'c_dec': c_dec}


def main():
    parser = argparse.ArgumentParser(description='Train BTK SAE on blunder activations')
    parser.add_argument('--dict-size', type=int, default=2048)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--cache', default=CACHE, help='Path to blunder activation cache')
    parser.add_argument('--use-best', action='store_true', help='Train on best moves instead of blunders')
    parser.add_argument('--move-token-only', action='store_true', help='Train on move token (hidden[77]) only, not all 77 tokens')
    args = parser.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    print(f'Loading cache from {args.cache}...')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)

    key = 'best_hidden' if args.use_best else 'blunder_hidden'
    n_positions = cache['n_blunders']
    print(f'Positions: {n_positions}, min_loss: {cache["min_loss"]}cp')

    hidden = cache[key][:n_positions]  # [N, 77, 1024] float16
    print(f'Hidden shape: {hidden.shape}')

    if args.move_token_only:
        # Use only the move token (index 76) — matches production pipeline
        hidden = hidden[:, 76:77, :]  # [N, 1, 1024]
        print(f'Move-token-only: {hidden.shape}')

    # Compute normalization from the activations
    print('Computing normalization...')
    flat = hidden.float().reshape(-1, 1024)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0) + 1e-8
    flat = (flat - mean) / std
    print(f'Flat shape: {flat.shape}')

    # Train
    sae, metrics = train_sae(flat, args.dict_size, args.k, epochs=args.epochs)

    # Save
    tag = 'best' if args.use_best else 'blunder'
    if args.move_token_only:
        tag += '_mt'
    out_path = f'{OUTPUT}/sae_btk_{tag}_{args.dict_size}_k{args.k}_aux.pt'
    torch.save({
        'encoder_weight': sae.encoder.weight.data.cpu(),
        'encoder_bias': sae.encoder.bias.data.cpu(),
        'decoder_weight': sae.decoder.weight.data.cpu(),
        'pre_bias': sae.pre_bias.data.cpu(),
        'k': args.k,
        'dict_size': args.dict_size,
        'mean': mean.numpy(),
        'std': std.numpy(),
        'config': {'dict_size': args.dict_size, 'k': args.k, 'input_dim': 1024},
        'normalization': {'mean': mean.numpy(), 'std': std.numpy()},
        'model_state_dict': sae.state_dict(),
        'metrics': metrics,
        'training_data': f'lichess_blunders_{tag}',
        'min_loss_cp': cache['min_loss'],
        'n_positions': n_positions,
    }, out_path)
    print(f'Saved: {out_path}')

    # Also save metrics JSON
    metrics_path = f'{OUTPUT}/metrics_{tag}_{args.dict_size}_k{args.k}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics: {metrics_path}')

    # Distribution of cp losses
    metadata = cache.get('metadata', [])
    if metadata:
        losses = [m['cp_loss'] for m in metadata[:n_positions]]
        print(f'\nCP loss distribution ({len(losses)} positions):')
        for threshold in [200, 300, 500, 1000, 2000]:
            n = sum(1 for l in losses if l >= threshold)
            print(f'  >= {threshold}cp: {n} ({100 * n / len(losses):.1f}%)')

    print('\nDone.')


if __name__ == '__main__':
    main()
