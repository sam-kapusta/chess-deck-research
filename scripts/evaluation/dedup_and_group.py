#!/usr/bin/env python3
"""Dedup features at Jaccard threshold, then group similar ones.

Step 1: Remove near-duplicates (Jaccard >= dedup_threshold). Keep highest confidence.
Step 2: Group remaining features where ALL pairs have Jaccard >= group_threshold (clique-based).
Step 3: Report groups with their labels for relabeling.

Usage:
    python3 dedup_and_group.py --checkpoint sae.pt --cache cache.pt --labels labels.json
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
    parser.add_argument('--cache', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--n-positions', type=int, default=10000)
    parser.add_argument('--dedup-threshold', type=float, default=0.8)
    parser.add_argument('--group-threshold', type=float, default=0.3)
    parser.add_argument('--max-fire-rate', type=float, default=5.0)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # Load
    print('Loading...')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:args.n_positions].float()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dd = ckpt['config']['dict_size']
    sae = SAE(1024, dd, ckpt['config']['k'])
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)
    N = fires.shape[0]

    with open(args.labels) as f:
        labels = json.load(f)

    # Filter to quality features
    fire_rates = fires.mean(axis=0) * 100
    quality_fids = []
    for fid in range(dd):
        if fire_rates[fid] == 0 or fire_rates[fid] > args.max_fire_rate:
            continue
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') in ['high', 'medium'] and not lbl.get('polysemantic', False):
            quality_fids.append(fid)
    print('Quality features (conf + mono + FR<' + str(args.max_fire_rate) + '%): ' + str(len(quality_fids)))

    # Build Jaccard matrix for quality features
    print('Computing Jaccard matrix...')
    fp = fires[:, quality_fids]
    intersection = fp.T @ fp
    sums = fp.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection
    union = np.maximum(union, 1)
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)

    # Step 1: Dedup at threshold
    print('\n=== Step 1: Dedup at Jaccard >= ' + str(args.dedup_threshold) + ' ===')
    # Greedy: remove feature with most duplicates, repeat
    alive = set(range(len(quality_fids)))
    removed = set()

    while True:
        # Find feature with most neighbors above threshold
        worst = -1
        worst_count = 0
        for i in alive:
            n_dups = sum(1 for j in alive if j != i and jaccard[i, j] >= args.dedup_threshold)
            if n_dups > worst_count:
                worst_count = n_dups
                worst = i
        if worst_count == 0:
            break
        # Remove the one with lower confidence (or arbitrary if same)
        # Actually remove the one with MORE duplicates — it's the redundant one
        # But keep the highest-confidence version
        neighbors = [j for j in alive if j != worst and jaccard[worst, j] >= args.dedup_threshold]
        # Keep the one with highest fire count (proxy for most representative)
        group = [worst] + neighbors
        fire_counts = [(idx, sums[idx]) for idx in group]
        keep_idx = max(fire_counts, key=lambda x: x[1])[0]
        for idx in group:
            if idx != keep_idx:
                alive.discard(idx)
                removed.add(idx)

    deduped_fids = [quality_fids[i] for i in sorted(alive)]
    print('After dedup: ' + str(len(deduped_fids)) + ' features (removed ' + str(len(removed)) + ')')

    # Rebuild Jaccard for deduped features
    fp2 = fires[:, deduped_fids]
    inter2 = fp2.T @ fp2
    sums2 = fp2.sum(axis=0)
    union2 = sums2[:, None] + sums2[None, :] - inter2
    union2 = np.maximum(union2, 1)
    jacc2 = inter2 / union2
    np.fill_diagonal(jacc2, 0)

    # Step 2: Clique-based grouping at threshold
    print('\n=== Step 2: Clique grouping at Jaccard >= ' + str(args.group_threshold) + ' ===')
    # Greedy clique finding: pick seed, expand with features that have >= threshold to ALL current members
    ungrouped = set(range(len(deduped_fids)))
    groups = []

    while ungrouped:
        # Pick seed: feature with most neighbors above threshold
        best_seed = -1
        best_n = -1
        for i in ungrouped:
            n = sum(1 for j in ungrouped if j != i and jacc2[i, j] >= args.group_threshold)
            if n > best_n:
                best_n = n
                best_seed = i

        # Expand clique from seed
        clique = {best_seed}
        candidates = [j for j in ungrouped if j != best_seed and jacc2[best_seed, j] >= args.group_threshold]

        for c in candidates:
            # Check if c has >= threshold Jaccard with ALL current clique members
            if all(jacc2[c, m] >= args.group_threshold for m in clique):
                clique.add(c)

        groups.append(sorted(clique))
        for c in clique:
            ungrouped.discard(c)

    # Separate singletons from real groups
    real_groups = [g for g in groups if len(g) > 1]
    singletons = [g[0] for g in groups if len(g) == 1]

    print('Groups with 2+ features: ' + str(len(real_groups)))
    print('Singletons: ' + str(len(singletons)))
    print('Total coaching concepts: ' + str(len(real_groups) + len(singletons)))

    # Print groups with labels
    print('\n=== Groups (2+ features) ===')
    group_summaries = []
    for gi, group in enumerate(sorted(real_groups, key=lambda g: -len(g))):
        fids = [deduped_fids[i] for i in group]
        group_labels = []
        group_cats = {}
        for fid in fids:
            lbl = labels.get(str(fid), {})
            group_labels.append(lbl.get('label', '?'))
            cat = lbl.get('category', '?')
            group_cats[cat] = group_cats.get(cat, 0) + 1

        top_cat = max(group_cats.items(), key=lambda x: x[1])[0]
        unique_labels = list(set(group_labels))

        if gi < 30:
            print('Group ' + str(gi + 1) + ' (' + str(len(fids)) + ' features, ' + top_cat + '):')
            for lbl in unique_labels[:5]:
                print('  - ' + lbl)
            if len(unique_labels) > 5:
                print('  ... and ' + str(len(unique_labels) - 5) + ' more')
            print()

        group_summaries.append({
            'feature_ids': fids,
            'n_features': len(fids),
            'dominant_category': top_cat,
            'unique_labels': unique_labels,
            'mean_jaccard': round(float(np.mean([jacc2[i, j] for i in group for j in group if i != j])), 4) if len(group) > 1 else 1.0,
        })

    # Category breakdown
    print('\n=== Category breakdown ===')
    all_cats = {}
    for fid in deduped_fids:
        cat = labels.get(str(fid), {}).get('category', '?')
        all_cats[cat] = all_cats.get(cat, 0) + 1
    for cat, n in sorted(all_cats.items(), key=lambda x: -x[1]):
        print('  ' + cat + ': ' + str(n))

    if args.output:
        result = {
            'dedup_threshold': args.dedup_threshold,
            'group_threshold': args.group_threshold,
            'quality_features': len(quality_fids),
            'after_dedup': len(deduped_fids),
            'n_groups': len(real_groups),
            'n_singletons': len(singletons),
            'total_concepts': len(real_groups) + len(singletons),
            'groups': group_summaries,
            'singleton_fids': [deduped_fids[s] for s in singletons],
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print('\nSaved to ' + args.output)


if __name__ == '__main__':
    main()
