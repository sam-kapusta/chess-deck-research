#!/usr/bin/env python3
"""Group SAE features into coaching categories using community detection.

Uses Louvain community detection on the Jaccard similarity graph.
Features that fire on overlapping positions end up in the same community.
No threshold needed — communities form naturally.

Usage:
    python3 community_grouping.py --checkpoint sae.pt --cache cache.pt --labels labels.json
    python3 community_grouping.py --checkpoint sae.pt --cache cache.pt --labels labels.json --resolution 1.5
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import community as community_louvain
    import networkx as nx
except ImportError:
    print('pip install python-louvain networkx')
    exit(1)


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
    parser.add_argument('--max-fire-rate', type=float, default=5.0)
    parser.add_argument('--min-edge-weight', type=float, default=0.05, help='Min Jaccard to create an edge')
    parser.add_argument('--resolution', type=float, default=1.0, help='Louvain resolution (higher = more communities)')
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
    print('Quality features: ' + str(len(quality_fids)))

    # Build Jaccard matrix
    print('Computing Jaccard...')
    fp = fires[:, quality_fids]
    intersection = fp.T @ fp
    sums = fp.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection
    union = np.maximum(union, 1)
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)

    # Build weighted graph
    print('Building graph (min edge weight=' + str(args.min_edge_weight) + ')...')
    G = nx.Graph()
    n = len(quality_fids)
    for i in range(n):
        G.add_node(i)
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = jaccard[i, j]
            if w >= args.min_edge_weight:
                G.add_edge(i, j, weight=float(w))
                edges += 1
    print('Nodes: ' + str(n) + ', Edges: ' + str(edges))
    isolated = sum(1 for node in G.nodes() if G.degree(node) == 0)
    print('Isolated nodes (no edges): ' + str(isolated))

    # Louvain community detection
    print('Running Louvain (resolution=' + str(args.resolution) + ')...')
    partition = community_louvain.best_partition(G, weight='weight', resolution=args.resolution)

    # Group features by community
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)

    # Sort by size
    sorted_comms = sorted(communities.items(), key=lambda x: -len(x[1]))
    print('\nFound ' + str(len(sorted_comms)) + ' communities')

    # Size distribution
    sizes = [len(c) for _, c in sorted_comms]
    print('Sizes: min=' + str(min(sizes)) + ' max=' + str(max(sizes)) + ' median=' + str(int(np.median(sizes))) + ' mean=' + str(round(np.mean(sizes), 1)))
    print()

    # Print communities
    results = []
    for rank, (comm_id, members) in enumerate(sorted_comms):
        fids = [quality_fids[m] for m in members]

        # Category breakdown
        cats = {}
        member_labels = []
        for fid in fids:
            lbl = labels.get(str(fid), {})
            cat = lbl.get('category', '?')
            cats[cat] = cats.get(cat, 0) + 1
            member_labels.append(lbl.get('label', '?'))

        top_cat = max(cats.items(), key=lambda x: x[1])[0]
        top_pct = round(cats[top_cat] / len(fids) * 100)
        unique_labels = list(set(member_labels))

        # Within-community Jaccard
        if len(members) > 1:
            j_sub = jaccard[np.ix_(members, members)]
            np.fill_diagonal(j_sub, 0)
            mean_jacc = j_sub.mean()
        else:
            mean_jacc = 0

        if rank < 40:
            print('Community ' + str(rank + 1) + ': ' + str(len(fids)) + ' features [' + top_cat + ' ' + str(top_pct) + '%] Jaccard=' + str(round(mean_jacc, 4)))
            for lbl in unique_labels[:4]:
                print('  - ' + lbl)
            if len(unique_labels) > 4:
                print('  ... +' + str(len(unique_labels) - 4) + ' more')
            print()

        results.append({
            'community_id': int(comm_id),
            'rank': rank + 1,
            'n_features': len(fids),
            'feature_ids': fids,
            'dominant_category': top_cat,
            'category_pct': top_pct,
            'categories': {k: int(v) for k, v in cats.items()},
            'mean_jaccard': round(float(mean_jacc), 4),
            'unique_labels': unique_labels[:20],
        })

    if args.output:
        out = {
            'n_communities': len(sorted_comms),
            'resolution': args.resolution,
            'min_edge_weight': args.min_edge_weight,
            'n_quality_features': len(quality_fids),
            'communities': results,
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print('Saved to ' + args.output)


if __name__ == '__main__':
    main()
