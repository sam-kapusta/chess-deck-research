#!/usr/bin/env python3
"""Greedy set cover: pick the minimum set of features that covers all blunder positions.

Each iteration picks the feature that covers the most uncovered positions.
Output is a ranked list of representative features with cumulative coverage.

Usage:
    python3 greedy_feature_selection.py --checkpoint sae.pt --cache cache.pt --n-positions 50000
    python3 greedy_feature_selection.py --checkpoint sae.pt --cache cache.pt --labels labels.json --target-coverage 0.95
"""
import argparse
import json
import sys

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
    parser.add_argument('--cache', required=True)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--n-positions', type=int, default=50000)
    parser.add_argument('--target-coverage', type=float, default=0.95)
    parser.add_argument('--max-features', type=int, default=500)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # Load data
    print('Loading...')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if 'blunder_mt' in cache:
        data = cache['blunder_mt'][:args.n_positions].float()
    else:
        data = cache['blunder_hidden'][:args.n_positions, 76, :].float()
    N = data.shape[0]

    # Load SAE
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dd = ckpt['config']['dict_size']
    k = ckpt['config']['k']
    sae = SAE(1024, dd, k)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    # Get fire patterns
    print('Computing activations on ' + str(N) + ' positions...')
    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy()  # [N, dd] bool

    # Load labels
    labels = {}
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)

    # Greedy set cover
    print('Running greedy set cover...')
    covered = np.zeros(N, dtype=bool)
    selected = []
    remaining = set(range(dd))

    # Pre-filter to alive features
    alive = set(int(i) for i in range(dd) if fires[:, i].sum() > 0)
    remaining = remaining & alive
    print('Alive features: ' + str(len(alive)))

    while len(selected) < args.max_features and remaining:
        # Find feature that covers most uncovered positions
        best_fid = -1
        best_new = 0

        for fid in remaining:
            new_coverage = (fires[:, fid] & ~covered).sum()
            if new_coverage > best_new:
                best_new = new_coverage
                best_fid = fid

        if best_new == 0:
            break

        # Select this feature
        covered = covered | fires[:, best_fid]
        selected.append(best_fid)
        remaining.discard(best_fid)

        total_covered = covered.sum()
        coverage_pct = total_covered / N * 100

        # Get label info
        lbl_info = labels.get(str(best_fid), {})
        label_text = lbl_info.get('label', '?')
        category = lbl_info.get('category', '?')
        confidence = lbl_info.get('confidence', '?')
        fr = fires[:, best_fid].sum() / N * 100

        if len(selected) <= 50 or len(selected) % 25 == 0:
            print('  #' + str(len(selected)) + ': F' + str(best_fid) +
                  ' +' + str(best_new) + ' new (' + str(round(coverage_pct, 1)) + '% total)' +
                  ' FR=' + str(round(fr, 2)) + '%' +
                  ' [' + category + '] ' + label_text[:50])

        if coverage_pct / 100 >= args.target_coverage:
            print('\nReached ' + str(round(coverage_pct, 1)) + '% coverage with ' + str(len(selected)) + ' features')
            break

    # Summary
    print()
    print('=== Coverage Summary ===')
    total_covered = covered.sum()
    print('Selected: ' + str(len(selected)) + ' features')
    print('Coverage: ' + str(total_covered) + '/' + str(N) + ' (' + str(round(total_covered / N * 100, 1)) + '%)')
    print()

    # Coverage milestones
    recov = np.zeros(N, dtype=bool)
    milestones = [25, 50, 75, 90, 95, 99]
    mi = 0
    print('Coverage milestones:')
    for i, fid in enumerate(selected):
        recov = recov | fires[:, fid]
        pct = recov.sum() / N * 100
        while mi < len(milestones) and pct >= milestones[mi]:
            print('  ' + str(milestones[mi]) + '%: ' + str(i + 1) + ' features')
            mi += 1

    # Category breakdown of selected features
    print()
    print('Categories in selected set:')
    cats = {}
    for fid in selected:
        cat = labels.get(str(fid), {}).get('category', 'unknown')
        cats[cat] = cats.get(cat, 0) + 1
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print('  ' + cat + ': ' + str(n))

    # Save
    if args.output:
        results = {
            'selected_features': selected,
            'n_selected': len(selected),
            'coverage': round(total_covered / N * 100, 2),
            'n_positions': N,
            'feature_details': [],
        }
        recov = np.zeros(N, dtype=bool)
        for fid in selected:
            new = (fires[:, fid] & ~recov).sum()
            recov = recov | fires[:, fid]
            cum = recov.sum()
            lbl = labels.get(str(fid), {})
            results['feature_details'].append({
                'feature_id': int(fid),
                'new_coverage': int(new),
                'cumulative_coverage': int(cum),
                'coverage_pct': round(cum / N * 100, 2),
                'fire_rate_pct': round(fires[:, fid].sum() / N * 100, 3),
                'label': lbl.get('label', ''),
                'short_label': lbl.get('chip', ''),
                'category': lbl.get('category', ''),
                'confidence': lbl.get('confidence', ''),
            })
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print('\nSaved to ' + args.output)


if __name__ == '__main__':
    main()
