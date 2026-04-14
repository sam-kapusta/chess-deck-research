#!/usr/bin/env python3
"""Experiment 28: Dedup check (Jaccard >0.8) + sub-cluster puzzle Forcing Moves.

Part A: How many blunder SAE feature pairs have Jaccard >0.8? (true duplicates)
Part B: Sub-cluster the 611-feature "Forcing Moves" puzzle mega-cluster.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print('pip install scikit-learn')
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


def part_a():
    """Check blunder SAE for true duplicates (Jaccard >0.8)."""
    print('=== PART A: Blunder SAE dedup check (Jaccard >0.8) ===')
    print()

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()

    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 2048, 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Quality features only
    quality_fids = [int(f) for f, info in labels.items()
                    if info.get('confidence') in ['high', 'medium']
                    and int(f) < fires.shape[1]
                    and fires[:, int(f)].sum() >= 10]
    print('Quality features: ' + str(len(quality_fids)))

    # Matmul Jaccard
    fp = fires[:, quality_fids]
    intersection = fp.T @ fp
    sums = fp.sum(axis=0)
    union = sums[:, None] + sums[None, :] - intersection
    union = np.maximum(union, 1)
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)

    # Count pairs at different thresholds
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        n_pairs = (jaccard > thresh).sum() // 2
        print(f'  Jaccard > {thresh}: {n_pairs} pairs')

    # Show any >0.8 pairs
    high_pairs = []
    n = len(quality_fids)
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard[i, j] > 0.8:
                fi, fj = quality_fids[i], quality_fids[j]
                li = labels.get(str(fi), {}).get('label', '?')[:50]
                lj = labels.get(str(fj), {}).get('label', '?')[:50]
                high_pairs.append((fi, fj, jaccard[i, j], li, lj))

    if high_pairs:
        print()
        print('Pairs with Jaccard > 0.8:')
        for fi, fj, j, li, lj in sorted(high_pairs, key=lambda x: -x[2])[:20]:
            print(f'  F{fi} <-> F{fj} (J={j:.3f})')
            print(f'    {li}')
            print(f'    {lj}')
    else:
        print()
        print('NO pairs with Jaccard > 0.8 — all features are unique')

    # Distribution stats
    upper = jaccard[np.triu_indices(n, k=1)]
    print()
    print('Jaccard distribution:')
    print(f'  Mean: {upper.mean():.4f}')
    print(f'  Median: {np.median(upper):.4f}')
    print(f'  Max: {upper.max():.4f}')
    print(f'  99th percentile: {np.percentile(upper, 99):.4f}')

    return jaccard, quality_fids


def part_b():
    """Sub-cluster puzzle SAE Forcing Moves (611 features)."""
    print()
    print('=== PART B: Sub-cluster puzzle "Forcing Moves" (611 features) ===')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/k64_baseline/labels_sonnet_think.json') as f:
        labels = json.load(f)

    # Load puzzle taxonomy to identify Forcing Moves features
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/puzzle_taxonomy_v1.json') as f:
        ptax = json.load(f)

    # Forcing Moves = clusters 1, 4, 8, 11 (from exp 26 mapping)
    forcing_clusters = {1, 4, 8, 11}
    forcing_fids = [f for f, cid in ptax['assignments'].items()
                    if cid in forcing_clusters and labels.get(f, {}).get('confidence') in ['high', 'medium']]
    print('Forcing Moves features: ' + str(len(forcing_fids)))

    # Sonnet categories within Forcing Moves
    sonnet_cats = Counter(labels[f].get('category', '?') for f in forcing_fids)
    print('Sonnet categories:')
    for cat, count in sonnet_cats.most_common():
        print(f'  {cat}: {count}')
    print()

    # TF-IDF sub-cluster
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in forcing_fids]
    vec = TfidfVectorizer(max_features=300, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()

    for k in [3, 5, 8]:
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        cl_labels = cl.fit_predict(embeddings)
        sizes = sorted(Counter(cl_labels).values(), reverse=True)
        print(f'  k={k}: sizes={sizes}')

    # Detailed at k=5
    K = 5
    cl = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
    cl_labels = cl.fit_predict(embeddings)

    cluster_members = defaultdict(list)
    for i, c in enumerate(cl_labels):
        cluster_members[c].append(i)

    print()
    print(f'=== Forcing Moves sub-clusters (k={K}) ===')
    for c in sorted(cluster_members.keys()):
        members = cluster_members[c]
        member_fids = [forcing_fids[m] for m in members]
        member_cats = [labels[f].get('category', '?') for f in member_fids]
        member_labels = [labels[f].get('label', '')[:55] for f in member_fids]

        top_cat = Counter(member_cats).most_common(1)[0]
        purity = top_cat[1] / len(members)

        all_text = ' '.join(labels[f].get('label', '') for f in member_fids).lower()
        words = all_text.split()
        stopwords = {'and', 'the', 'in', 'of', 'with', 'for', 'or', 'a', 'to', 'on',
                     'that', 'is', 'are', 'an', 'by', 'from', 'at', 'as', 'be', 'its'}
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 2)
        top_words = [w for w, _ in word_counts.most_common(5)]

        print(f'\nSub-cluster {c} ({len(members)} features, {round(purity*100)}% {top_cat[0]}):')
        print(f'  Words: {", ".join(top_words)}')
        for lbl in member_labels[:4]:
            print(f'    - {lbl}')
        if len(members) > 4:
            print(f'    ... +{len(members)-4} more')


def main():
    part_a()
    part_b()


if __name__ == '__main__':
    main()
