#!/usr/bin/env python3
"""Experiment 16: Do decoder weight vectors cluster features better than fire patterns?

Hypothesis: Decoder weights encode conceptual similarity that fire patterns miss.
Prediction: Decoder-based clusters have >2x higher category purity than fire-pattern clusters
            for tactical features (which Exp 7-8 showed don't cluster by fire pattern).
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from collections import Counter

try:
    import community as community_louvain
    import networkx as nx
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
except ImportError:
    print('pip install python-louvain networkx scipy')
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


def category_purity(cluster_labels, feature_categories, min_cluster_size=3):
    """For each cluster, what fraction shares the majority category?"""
    clusters = {}
    for fid, cl in cluster_labels.items():
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl].append(fid)

    purities = []
    for cl, fids in clusters.items():
        if len(fids) < min_cluster_size:
            continue
        cats = [feature_categories.get(str(f), 'unknown') for f in fids]
        most_common = Counter(cats).most_common(1)[0][1]
        purities.append(most_common / len(cats))

    return np.mean(purities) if purities else 0, len(purities)


def main():
    print('Experiment 16: Decoder weight clustering vs fire-pattern clustering')
    print('Hypothesis: Decoder weights cluster tactical features better than fire patterns')
    print('Prediction: >2x category purity for tactical features')
    print()

    # Load SAE
    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 2048, 32)
    sae.load_state_dict(ckpt['model_state_dict'])

    # Load labels
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Load activations for fire-pattern comparison
    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()
    metadata = cache['metadata'][:10000]
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)

    # Filter to quality features
    quality_fids = []
    for fid in range(2048):
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') in ['high', 'medium'] and fires[:, fid].sum() >= 10:
            quality_fids.append(fid)

    print('Quality features: ' + str(len(quality_fids)))

    # Get categories
    feature_cats = {str(f): labels.get(str(f), {}).get('category', 'unknown') for f in quality_fids}
    cat_counts = Counter(feature_cats.values())
    print('Categories: ' + str(dict(cat_counts.most_common(10))))

    # Classify phases for tactical vs endgame split
    phases = []
    for md in metadata:
        try:
            n = len(chess.Board(md['fen']).piece_map())
        except:
            n = 20
        phases.append('opening' if n > 24 else ('middlegame' if n > 12 else 'endgame'))

    phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
    for i, p in enumerate(phases):
        phase_idx[p].append(i)

    # Split features into tactical (middlegame/opening) vs endgame
    tactical_fids = []
    endgame_fids = []
    for fid in quality_fids:
        total = fires[:, fid].sum()
        if total < 10:
            continue
        eg_ratio = fires[phase_idx['endgame'], fid].sum() / total
        if eg_ratio > 0.6:
            endgame_fids.append(fid)
        elif eg_ratio < 0.4:
            tactical_fids.append(fid)

    print('Tactical features (endgame<40%): ' + str(len(tactical_fids)))
    print('Endgame features (endgame>60%): ' + str(len(endgame_fids)))
    print()

    # === METHOD 1: Decoder weight clustering ===
    decoder_weights = sae.decoder.weight.data.numpy().T  # shape: (2048, 1024)
    # Normalize decoder weights
    norms = np.linalg.norm(decoder_weights, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    decoder_normed = decoder_weights / norms

    for name, fids in [('TACTICAL', tactical_fids), ('ENDGAME', endgame_fids)]:
        if len(fids) < 10:
            print(name + ': too few features')
            continue

        print('=' * 60)
        print(name + ' (' + str(len(fids)) + ' features)')
        print()

        # Decoder cosine similarity
        dec_sub = decoder_normed[fids]
        dec_sim = dec_sub @ dec_sub.T  # cosine similarity matrix
        np.fill_diagonal(dec_sim, 0)

        # Fire-pattern Jaccard
        fp = fires[:, fids]
        intersection = fp.T @ fp
        sums = fp.sum(axis=0)
        union = sums[:, None] + sums[None, :] - intersection
        union = np.maximum(union, 1)
        jaccard = intersection / union
        np.fill_diagonal(jaccard, 0)

        print('Similarity stats:')
        print('  Decoder cosine: mean=' + str(round(dec_sim.mean(), 4)) +
              ' max=' + str(round(dec_sim.max(), 4)) +
              ' >0.5: ' + str((dec_sim > 0.5).sum() // 2))
        print('  Fire Jaccard:   mean=' + str(round(jaccard.mean(), 4)) +
              ' max=' + str(round(jaccard.max(), 4)) +
              ' >0.1: ' + str((jaccard > 0.1).sum() // 2))
        print()

        # === Cluster with decoder weights (hierarchical) ===
        dec_dist = 1 - dec_sim
        np.fill_diagonal(dec_dist, 0)
        condensed = pdist(dec_sub, metric='cosine')
        Z = linkage(condensed, method='ward')

        # Try multiple cluster counts
        for n_clusters in [15, 25, 50]:
            cl = fcluster(Z, n_clusters, criterion='maxclust')
            dec_cluster_labels = {fids[i]: int(cl[i]) for i in range(len(fids))}
            dec_purity, dec_n = category_purity(dec_cluster_labels, feature_cats)

            # Fire-pattern clustering (Louvain)
            G = nx.Graph()
            n = len(fids)
            for i in range(n):
                G.add_node(i)
            for i in range(n):
                for j in range(i + 1, n):
                    if jaccard[i, j] >= 0.05:
                        G.add_edge(i, j, weight=float(jaccard[i, j]))

            if G.number_of_edges() > 0:
                partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
                fire_cluster_labels = {fids[i]: partition[i] for i in range(n)}
                fire_purity, fire_n = category_purity(fire_cluster_labels, feature_cats)
            else:
                fire_purity, fire_n = 0, 0

            print('  k=' + str(n_clusters) + ': Decoder purity=' + str(round(dec_purity, 3)) +
                  ' (' + str(dec_n) + ' clusters), Fire purity=' + str(round(fire_purity, 3)) +
                  ' (' + str(fire_n) + ' clusters)')

        # Show top decoder clusters
        cl = fcluster(Z, 25, criterion='maxclust')
        clusters = {}
        for i, fid in enumerate(fids):
            c = int(cl[i])
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(fid)

        print()
        print('Top decoder clusters (k=25):')
        for c, members in sorted(clusters.items(), key=lambda x: -len(x[1]))[:8]:
            if len(members) < 3:
                break
            member_labels = [labels.get(str(f), {}).get('label', '?')[:40] for f in members]
            member_cats = [labels.get(str(f), {}).get('category', '?') for f in members]
            majority_cat = Counter(member_cats).most_common(1)[0]
            purity = majority_cat[1] / len(members)

            # Within-cluster decoder cosine
            idx = [fids.index(f) for f in members]
            sub_sim = dec_sim[np.ix_(idx, idx)]
            np.fill_diagonal(sub_sim, 0)
            mean_sim = sub_sim.mean()

            print('  Cluster ' + str(c) + ' (' + str(len(members)) + ' features, cos=' +
                  str(round(mean_sim, 3)) + ', purity=' + str(round(purity * 100)) +
                  '% ' + majority_cat[0] + '):')
            for lbl in member_labels[:3]:
                print('    - ' + lbl)
            if len(member_labels) > 3:
                print('    ... +' + str(len(member_labels) - 3) + ' more')
        print()

    # Verdict
    print('=== Verdict ===')
    print('Compare decoder vs fire-pattern purity for tactical features at k=25 above')
    print('If decoder purity > 2x fire purity → CONFIRMED')
    print('Otherwise → FAILED')


if __name__ == '__main__':
    main()
