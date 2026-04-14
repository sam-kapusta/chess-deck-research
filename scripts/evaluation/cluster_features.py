#!/usr/bin/env python3
"""Cluster SAE features by fire patterns into coaching categories.

Groups features that fire on similar positions, regardless of label text.
Uses cosine similarity on binary fire vectors + hierarchical clustering.

Usage:
    python3 cluster_features.py --checkpoint sae.pt --cache cache.pt --n-positions 10000 --n-clusters 25
    python3 cluster_features.py --checkpoint sae.pt --cache cache.pt --labels labels.json --n-clusters 25
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


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
    parser.add_argument('--labels', default=None, help='Label JSON for annotation')
    parser.add_argument('--n-positions', type=int, default=10000)
    parser.add_argument('--n-clusters', type=int, default=25)
    parser.add_argument('--output', default=None, help='Save cluster assignments JSON')
    args = parser.parse_args()

    # Load cache
    print('Loading data...')
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
    print('Computing activations...')
    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)  # [N, dd]

    # Filter to alive features
    fire_rates = fires.mean(axis=0)
    alive_mask = fire_rates > 0
    alive_ids = np.where(alive_mask)[0]
    fires_alive = fires[:, alive_mask]  # [N, n_alive]
    print('Alive features:', len(alive_ids))

    # Compute pairwise cosine distance between features
    # Each feature is a vector of length N (which positions it fires on)
    print('Computing cosine distances...')
    # Normalize for cosine
    norms = np.linalg.norm(fires_alive, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    fires_normed = fires_alive / norms  # [N, n_alive]

    # Cosine similarity matrix
    cos_sim = fires_normed.T @ fires_normed  # [n_alive, n_alive]
    # Convert to distance
    cos_dist = 1 - cos_sim
    np.fill_diagonal(cos_dist, 0)
    cos_dist = np.maximum(cos_dist, 0)  # numerical stability

    # Hierarchical clustering
    print('Clustering...')
    # condensed distance matrix for scipy
    condensed = pdist(fires_alive.T, metric='cosine')
    condensed = np.nan_to_num(condensed, nan=1.0)
    Z = linkage(condensed, method='ward')
    cluster_ids = fcluster(Z, t=args.n_clusters, criterion='maxclust')

    # Map back to feature IDs
    clusters = {}
    for i, cid in enumerate(cluster_ids):
        cid = int(cid)
        fid = int(alive_ids[i])
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(fid)

    # Load labels if available
    labels = {}
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)

    # Print cluster summary
    print()
    print('=== ' + str(args.n_clusters) + ' Clusters ===')
    print()

    cluster_summaries = {}
    for cid in sorted(clusters.keys()):
        fids = clusters[cid]
        # Get fire rates for this cluster
        cluster_frs = [fire_rates[f] * 100 for f in fids]
        mean_fr = np.mean(cluster_frs)

        # Get labels if available
        cluster_labels = []
        cluster_cats = {}
        for fid in fids:
            if str(fid) in labels:
                lbl = labels[str(fid)]
                cluster_labels.append(lbl.get('label', ''))
                cat = lbl.get('category', 'unknown')
                cluster_cats[cat] = cluster_cats.get(cat, 0) + 1

        # Most common category
        top_cat = max(cluster_cats.items(), key=lambda x: x[1])[0] if cluster_cats else 'unknown'
        cat_pct = round(cluster_cats.get(top_cat, 0) / max(len(fids), 1) * 100)

        # Unique labels
        unique_labels = list(set(cluster_labels))

        # Within-cluster Jaccard (how tight is this cluster?)
        if len(fids) > 1:
            fp = fires[:, fids].astype(np.float32)
            inter = fp.T @ fp
            sums = fp.sum(axis=0)
            union = sums[:, None] + sums[None, :] - inter
            union = np.maximum(union, 1)
            jacc = inter / union
            np.fill_diagonal(jacc, 0)
            mean_jacc = jacc.mean()
        else:
            mean_jacc = 1.0

        print('Cluster ' + str(cid) + ': ' + str(len(fids)) + ' features, FR=' + str(round(mean_fr, 2)) + '%, Jaccard=' + str(round(mean_jacc, 4)))
        print('  Dominant category: ' + top_cat + ' (' + str(cat_pct) + '%)')
        if unique_labels:
            print('  Sample labels:')
            for lbl in unique_labels[:5]:
                print('    - ' + lbl)
        print()

        cluster_summaries[cid] = {
            'n_features': len(fids),
            'feature_ids': fids,
            'mean_fire_rate': round(float(mean_fr), 3),
            'mean_within_jaccard': round(float(mean_jacc), 4),
            'dominant_category': top_cat,
            'category_pct': cat_pct,
            'sample_labels': unique_labels[:10],
        }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        print('Saved to ' + args.output)

    # Summary stats
    sizes = [len(c) for c in clusters.values()]
    print('Cluster sizes: min=' + str(min(sizes)) + ' max=' + str(max(sizes)) + ' median=' + str(int(np.median(sizes))))


if __name__ == '__main__':
    main()
