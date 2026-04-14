#!/usr/bin/env python3
"""Compute within-category Jaccard overlap to find redundant features.

For each category, checks how many features fire on the same positions
and how many unique vs duplicate labels exist.

Usage:
    python3 within_category_jaccard.py --checkpoint sae.pt --labels labels.json --cache cache.pt --n-positions 10000
"""
import argparse
import json

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

    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--n-positions', type=int, default=10000)
    parser.add_argument('--categories', nargs='+', default=['hanging_pieces', 'endgame_technique', 'passed_pawn', 'deflection', 'king_attack', 'forcing_moves'])
    args = parser.parse_args()

    # Load cache
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if 'blunder_mt' in cache:
        data = cache['blunder_mt'][:args.n_positions].float()
    else:
        data = cache['blunder_hidden'][:args.n_positions, 76, :].float()

    # Load SAE
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dd = ckpt['config']['dict_size']
    k = ckpt['config']['k']
    sae = SAE(1024, dd, k)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    # Get fire patterns
    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)

    # Load labels
    with open(args.labels) as f:
        labels = json.load(f)

    # Group by category
    by_cat = {}
    for fid_str, v in labels.items():
        fid = int(fid_str)
        if fid >= dd:
            continue
        cat = v.get('category', 'unknown')
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append((fid, v.get('label', ''), v.get('confidence', 'low')))

    for cat in args.categories:
        feats = by_cat.get(cat, [])
        fids = [f[0] for f in feats]
        if len(fids) < 10:
            continue

        fp = fires[:, fids]
        intersection = fp.T @ fp
        sums = fp.sum(axis=0)
        union = sums[:, None] + sums[None, :] - intersection
        union = np.maximum(union, 1)
        jaccard = intersection / union
        np.fill_diagonal(jaccard, 0)
        best = jaccard.max(axis=1)

        unique_labels = set(f[1] for f in feats)
        label_counts = {}
        for f in feats:
            label_counts[f[1]] = label_counts.get(f[1], 0) + 1

        print(cat + ' (' + str(len(fids)) + ' features):')
        print('  Mean best Jaccard: ' + str(round(float(best.mean()), 4)))
        print('  Jaccard >= 0.3: ' + str(int((best >= 0.3).sum())) + ' (' + str(round((best >= 0.3).mean() * 100, 1)) + '%)')
        print('  Jaccard >= 0.5: ' + str(int((best >= 0.5).sum())) + ' (' + str(round((best >= 0.5).mean() * 100, 1)) + '%)')
        print('  Unique labels: ' + str(len(unique_labels)) + ' / ' + str(len(fids)) + ' (' + str(round(len(unique_labels) / len(fids) * 100)) + '%)')
        print('  Most common:')
        for lbl, n in sorted(label_counts.items(), key=lambda x: -x[1])[:5]:
            print('    ' + str(n) + 'x  ' + lbl)
        print()


if __name__ == '__main__':
    main()
