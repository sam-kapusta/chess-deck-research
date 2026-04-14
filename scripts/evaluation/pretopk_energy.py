#!/usr/bin/env python3
"""Analyze pre-topk activation energy distribution.

Shows how many features naturally activate per position before the
top-k selection, and what fraction of energy different k values capture.

Usage:
    python3 pretopk_energy.py --checkpoint sae.pt --cache cache.pt --n-positions 10000
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--n-positions', type=int, default=10000)
    args = parser.parse_args()

    # Load cache
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if 'blunder_mt' in cache:
        data = cache['blunder_mt'][:args.n_positions].float()
    else:
        data = cache['blunder_hidden'][:args.n_positions, 76, :].float()

    mean = torch.tensor(cache.get('mean', np.zeros(1024)), dtype=torch.float32)
    std = torch.tensor(cache.get('std', np.ones(1024)), dtype=torch.float32) + 1e-8
    data = (data - mean) / std

    # Load SAE weights
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    enc_w = ckpt['encoder_weight']
    enc_b = ckpt['encoder_bias']
    pre_bias = ckpt['pre_bias']

    # Compute pre-topk activations
    z = F.linear(data - pre_bias, enc_w, enc_b)
    z_relu = F.relu(z)

    # Count active per position
    active_per_pos = (z_relu > 0).float().sum(dim=-1)
    print('Pre-topk activations > 0 per position:')
    print('  Mean: ' + str(round(active_per_pos.mean().item(), 1)))
    print('  Median: ' + str(round(active_per_pos.median().item(), 1)))
    print('  Std: ' + str(round(active_per_pos.std().item(), 1)))
    print('  Min: ' + str(int(active_per_pos.min().item())))
    print('  Max: ' + str(int(active_per_pos.max().item())))
    print()

    for thresh in [10, 20, 30, 50, 64, 100, 200, 500, 1000]:
        pct = (active_per_pos >= thresh).float().mean().item() * 100
        print('  >= ' + str(thresh) + ': ' + str(round(pct, 1)) + '%')

    # Energy captured by top-k
    sorted_acts, _ = z_relu.sort(dim=-1, descending=True)
    total_energy = sorted_acts.sum(dim=-1)

    print()
    print('Energy captured by top-k:')
    for k in [8, 16, 32, 64, 128, 256]:
        topk_energy = sorted_acts[:, :k].sum(dim=-1)
        ratio = (topk_energy / total_energy).mean().item() * 100
        print('  Top-' + str(k) + ': ' + str(round(ratio, 1)) + '%')


if __name__ == '__main__':
    main()
