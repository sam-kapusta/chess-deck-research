#!/usr/bin/env python3
"""Compare SAE variants side-by-side: per-variant label stats + Jaccard dedup.

Loads multiple SAE checkpoints, runs them on shared blunder positions,
computes pairwise Jaccard overlap to find duplicate features across SAEs.

Usage:
    python3 compare_saes.py --checkpoints ckpt1.pt ckpt2.pt --positions blunder_positions.json
    python3 compare_saes.py --checkpoints ckpt1.pt ckpt2.pt --cache blunder_move_token_200k.pt --n-positions 10000
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


def load_sae(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    dd = cfg.get('dict_size', ckpt.get('dict_size'))
    k = cfg.get('k', ckpt.get('k'))
    sae = SAE(1024, dd, k)
    if 'model_state_dict' in ckpt:
        sae.load_state_dict(ckpt['model_state_dict'])
    else:
        sae.encoder.weight.data = ckpt['encoder_weight']
        sae.encoder.bias.data = ckpt['encoder_bias']
        sae.decoder.weight.data = ckpt['decoder_weight']
        sae.pre_bias.data = ckpt['pre_bias']

    mean = torch.tensor(ckpt.get('mean', ckpt.get('normalization', {}).get('mean', np.zeros(1024))), dtype=torch.float32)
    std = torch.tensor(ckpt.get('std', ckpt.get('normalization', {}).get('std', np.ones(1024))), dtype=torch.float32) + 1e-8

    name = os.path.basename(path).replace('sae_btk_', '').replace('_aux.pt', '')
    return sae, mean, std, name, dd, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True, help='SAE checkpoint paths')
    parser.add_argument('--cache', required=True, help='Move-token cache path')
    parser.add_argument('--n-positions', type=int, default=10000, help='Number of positions for Jaccard')
    parser.add_argument('--jaccard-threshold', type=float, default=0.3, help='Min Jaccard to flag as duplicate')
    parser.add_argument('--output', default=None, help='Save comparison JSON')
    args = parser.parse_args()

    # Load move-token cache
    print('Loading cache...')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if 'blunder_mt' in cache:
        data = cache['blunder_mt'][:args.n_positions].float()
    else:
        data = cache['blunder_hidden'][:args.n_positions, 76, :].float()
    print('Positions:', data.shape[0])

    # Load all SAEs
    saes = []
    for path in args.checkpoints:
        sae, mean, std, name, dd, k = load_sae(path)
        saes.append({'sae': sae, 'mean': mean, 'std': std, 'name': name, 'dd': dd, 'k': k})
        print('Loaded:', name, 'dict=' + str(dd), 'k=' + str(k))

    # Run each SAE on the data, collect binary fire patterns
    print('\nComputing activations...')
    fire_patterns = {}  # name -> [N, dd] binary
    for s in saes:
        normed = (data - s['mean']) / s['std']
        with torch.no_grad():
            _, acts = s['sae'](normed)
        fires = (acts > 0).numpy()  # [N, dd]
        fire_patterns[s['name']] = fires

        # Per-variant stats
        fire_rate = fires.mean(axis=0)  # [dd]
        alive = (fire_rate > 0).sum()
        print(s['name'] + ':')
        print('  Alive:', alive, '/', s['dd'])
        print('  Fire rate: mean=' + str(round(fire_rate[fire_rate > 0].mean() * 100, 2)) + '% median=' + str(round(np.median(fire_rate[fire_rate > 0]) * 100, 2)) + '%')
        print('  Features > 5%:', (fire_rate > 0.05).sum())
        print('  Features > 10%:', (fire_rate > 0.1).sum())

    # Pairwise Jaccard between all SAE pairs — full matrix multiply, no sampling
    print('\n=== Pairwise Jaccard Overlap ===')
    names = list(fire_patterns.keys())
    results = {'variants': {}, 'pairwise': []}

    for s in saes:
        fp = fire_patterns[s['name']]
        fire_rate = fp.mean(axis=0)
        alive = int((fire_rate > 0).sum())
        results['variants'][s['name']] = {
            'dict_size': s['dd'],
            'k': s['k'],
            'alive': alive,
            'fire_rate_mean': round(float(fire_rate[fire_rate > 0].mean()) * 100, 3),
            'fire_rate_median': round(float(np.median(fire_rate[fire_rate > 0])) * 100, 3),
        }

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            fp1 = fire_patterns[n1].astype(np.float32)  # [N, dd1]
            fp2 = fire_patterns[n2].astype(np.float32)  # [N, dd2]

            print('\n' + n1 + ' vs ' + n2 + ':')

            # Full Jaccard matrix via matrix multiply
            intersection = fp1.T @ fp2                          # [dd1, dd2]
            sum1 = fp1.sum(axis=0)[:, None]                     # [dd1, 1]
            sum2 = fp2.sum(axis=0)[None, :]                     # [1, dd2]
            union = sum1 + sum2 - intersection                  # [dd1, dd2]
            union = np.maximum(union, 1)                        # avoid div by zero
            jaccard = intersection / union                      # [dd1, dd2]

            # Filter to alive features only
            alive1 = sum1.squeeze() > 0
            alive2 = sum2.squeeze() > 0
            n_alive1 = int(alive1.sum())
            n_alive2 = int(alive2.sum())

            # Best match per feature in SAE1
            jacc_alive = jaccard[alive1][:, alive2]             # [alive1, alive2]
            best_per_f1 = jacc_alive.max(axis=1)               # [alive1]
            best_per_f2 = jacc_alive.max(axis=0)               # [alive2]

            print('  Features: ' + str(n_alive1) + ' vs ' + str(n_alive2) + ' (all, no sampling)')
            print('  Mean best Jaccard (SAE1→SAE2): ' + str(round(float(best_per_f1.mean()), 4)))
            print('  Mean best Jaccard (SAE2→SAE1): ' + str(round(float(best_per_f2.mean()), 4)))
            for thresh in [0.3, 0.5, 0.7, 0.9]:
                n_dup1 = int((best_per_f1 >= thresh).sum())
                n_dup2 = int((best_per_f2 >= thresh).sum())
                print('  Jaccard >= ' + str(thresh) + ': ' + str(n_dup1) + '/' + str(n_alive1) + ' (' + str(round(n_dup1/max(n_alive1,1)*100,1)) + '%) | ' + str(n_dup2) + '/' + str(n_alive2) + ' (' + str(round(n_dup2/max(n_alive2,1)*100,1)) + '%)')

            results['pairwise'].append({
                'sae1': n1, 'sae2': n2,
                'alive1': n_alive1, 'alive2': n_alive2,
                'mean_best_jaccard_1to2': round(float(best_per_f1.mean()), 4),
                'mean_best_jaccard_2to1': round(float(best_per_f2.mean()), 4),
                'dup_030': int((best_per_f1 >= 0.3).sum()),
                'dup_050': int((best_per_f1 >= 0.5).sum()),
                'dup_070': int((best_per_f1 >= 0.7).sum()),
                'dup_090': int((best_per_f1 >= 0.9).sum()),
            })

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print('\nSaved to', args.output)

    print('\nDone.')


if __name__ == '__main__':
    main()
