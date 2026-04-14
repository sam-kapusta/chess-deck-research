#!/usr/bin/env python3
"""Hypothesis: Phase-specific features cluster cleanly, phase-neutral don't.

Splits features by phase specificity, clusters each group separately,
compares within-cluster Jaccard.

Usage:
    python3 phase_cluster_test.py --checkpoint sae.pt --cache cache.pt --labels labels.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

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


def get_phase(fen):
    try:
        board = chess.Board(fen)
        n = len(board.piece_map())
    except:
        return 'middlegame'
    if n > 24: return 'opening'
    if n > 12: return 'middlegame'
    return 'endgame'


def cluster_and_report(name, fids, fires, labels, n_positions):
    """Cluster a subset of features and report quality."""
    if len(fids) < 5:
        print(name + ': too few features (' + str(len(fids)) + ')')
        return

    fp = fires[:, fids].astype(np.float32)
    intersection = fp.T @ fp
    sums = fp.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection
    union = np.maximum(union, 1)
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)

    # Build graph
    G = nx.Graph()
    n = len(fids)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard[i, j] >= 0.05:
                G.add_edge(i, j, weight=float(jaccard[i, j]))

    isolated = sum(1 for node in G.nodes() if G.degree(node) == 0)

    if G.number_of_edges() == 0:
        print(name + ': ' + str(len(fids)) + ' features, NO edges (all isolated)')
        return

    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)

    sorted_comms = sorted(communities.items(), key=lambda x: -len(x[1]))
    sizes = [len(c) for _, c in sorted_comms]
    multi = [c for c in sizes if c > 1]

    # Compute mean within-community Jaccard
    within_jaccards = []
    for _, members in sorted_comms:
        if len(members) < 2:
            continue
        j_sub = jaccard[np.ix_(members, members)]
        np.fill_diagonal(j_sub, 0)
        within_jaccards.append(j_sub.mean())

    mean_within = np.mean(within_jaccards) if within_jaccards else 0

    print(name + ': ' + str(len(fids)) + ' features → ' + str(len(sorted_comms)) + ' communities')
    print('  Multi-feature communities: ' + str(len(multi)))
    print('  Singletons: ' + str(len(sizes) - len(multi)))
    print('  Isolated (no edges): ' + str(isolated))
    print('  Mean within-community Jaccard: ' + str(round(mean_within, 4)))
    print('  Largest community: ' + str(max(sizes)))
    print()

    # Show top communities with labels
    for rank, (comm_id, members) in enumerate(sorted_comms[:8]):
        if len(members) < 2:
            break
        member_fids = [fids[m] for m in members]
        member_labels = list(set(
            labels.get(str(f), {}).get('label', '?')[:50]
            for f in member_fids
        ))
        j_sub = jaccard[np.ix_(members, members)]
        np.fill_diagonal(j_sub, 0)
        mj = j_sub.mean()

        print('  Community ' + str(rank + 1) + ' (' + str(len(members)) + ' features, Jaccard=' + str(round(mj, 4)) + '):')
        for lbl in member_labels[:4]:
            print('    - ' + lbl)
        if len(member_labels) > 4:
            print('    ... +' + str(len(member_labels) - 4) + ' more')
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--n-positions', type=int, default=5000)
    args = parser.parse_args()

    # Load
    print('Loading...')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:args.n_positions].float()
    metadata = cache['metadata'][:args.n_positions]
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dd = ckpt['config']['dict_size']
    sae = SAE(1024, dd, ckpt['config']['k'])
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy()

    with open(args.labels) as f:
        labels = json.load(f)

    # Compute phase per position
    phases = [get_phase(md['fen']) for md in metadata]
    phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
    for i, p in enumerate(phases):
        phase_idx[p].append(i)

    # Classify each feature by phase specificity
    endgame_specific = []
    opening_specific = []
    middlegame_specific = []
    phase_neutral = []

    for fid in range(dd):
        total = fires[:, fid].sum()
        if total < 10:
            continue
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') not in ['high', 'medium']:
            continue

        ratios = {}
        for phase in ['opening', 'middlegame', 'endgame']:
            idx = phase_idx[phase]
            ratios[phase] = fires[idx, fid].sum() / total if total > 0 else 0

        if ratios['endgame'] > 0.8:
            endgame_specific.append(fid)
        elif ratios['opening'] > 0.8:
            opening_specific.append(fid)
        elif ratios['middlegame'] > 0.8:
            middlegame_specific.append(fid)
        elif all(0.15 < r < 0.55 for r in ratios.values()):
            phase_neutral.append(fid)

    print('Phase classification:')
    print('  Endgame-specific (>80%): ' + str(len(endgame_specific)))
    print('  Opening-specific (>80%): ' + str(len(opening_specific)))
    print('  Middlegame-specific (>80%): ' + str(len(middlegame_specific)))
    print('  Phase-neutral (15-55% each): ' + str(len(phase_neutral)))
    print()

    # Cluster each group
    print('=' * 60)
    cluster_and_report('ENDGAME-SPECIFIC', endgame_specific, fires, labels, args.n_positions)
    print('=' * 60)
    cluster_and_report('OPENING-SPECIFIC', opening_specific, fires, labels, args.n_positions)
    print('=' * 60)
    cluster_and_report('MIDDLEGAME-SPECIFIC', middlegame_specific, fires, labels, args.n_positions)
    print('=' * 60)
    cluster_and_report('PHASE-NEUTRAL', phase_neutral, fires, labels, args.n_positions)


if __name__ == '__main__':
    main()
