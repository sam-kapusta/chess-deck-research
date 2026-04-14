#!/usr/bin/env python3
"""Experiment 18: Cluster features by label text similarity, not fire pattern.

Hypothesis: Label-text embeddings produce cleaner tactical categories than fire-pattern clustering.
Prediction: Label-text clusters have >50% category purity for tactical features (vs ~30% from fire patterns).

Method: Embed each feature's label+explanation via sentence transformer, cluster in embedding space.
"""
import json
import numpy as np
from collections import Counter

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print('pip install scikit-learn')
    exit(1)


def get_embeddings(texts):
    """Get text embeddings. Try sentence-transformers first, fall back to TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts, show_progress_bar=True)
    except ImportError:
        print('sentence-transformers not available, using TF-IDF')
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=500, stop_words='english')
        return vec.fit_transform(texts).toarray()


def category_purity(cluster_labels, categories, min_size=3):
    """Mean purity of clusters with >= min_size members."""
    clusters = {}
    for fid, cl in cluster_labels.items():
        clusters.setdefault(cl, []).append(fid)

    purities = []
    for members in clusters.values():
        if len(members) < min_size:
            continue
        cats = [categories[f] for f in members]
        majority = Counter(cats).most_common(1)[0][1]
        purities.append(majority / len(cats))
    return np.mean(purities) if purities else 0, len(purities)


def main():
    print('Experiment 18: Label-text clustering for tactical features')
    print('Hypothesis: Text embeddings cluster tactical features better than fire patterns')
    print('Prediction: >50% category purity (vs ~30% from fire patterns)')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Filter to quality tactical features
    # Use fire pattern data to identify tactical vs endgame
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import chess

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

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()
    metadata = cache['metadata'][:10000]

    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 2048, 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)

    # Classify phases
    phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
    for i, md in enumerate(metadata):
        try:
            n = len(chess.Board(md['fen']).piece_map())
        except:
            n = 20
        p = 'opening' if n > 24 else ('middlegame' if n > 12 else 'endgame')
        phase_idx[p].append(i)

    # Get tactical features (endgame < 40%)
    tactical_fids = []
    for fid in range(2048):
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') not in ['high', 'medium']:
            continue
        total = fires[:, fid].sum()
        if total < 10:
            continue
        eg_ratio = fires[phase_idx['endgame'], fid].sum() / total
        if eg_ratio < 0.4:
            tactical_fids.append(fid)

    print('Tactical features: ' + str(len(tactical_fids)))

    # Build text for each feature: label + explanation
    texts = []
    fid_to_idx = {}
    categories = {}
    for i, fid in enumerate(tactical_fids):
        lbl = labels[str(fid)]
        text = lbl.get('label', '') + '. ' + lbl.get('explanation', '')[:200]
        texts.append(text)
        fid_to_idx[fid] = i
        categories[fid] = lbl.get('category', 'unknown')

    cat_counts = Counter(categories.values())
    print('Sonnet categories: ' + str(dict(cat_counts.most_common(8))))
    print()

    # Embed
    print('Computing text embeddings...')
    embeddings = get_embeddings(texts)
    print('Embedding shape: ' + str(embeddings.shape))

    # Compute similarity
    sim = cosine_similarity(embeddings)
    print('Mean text cosine sim: ' + str(round(sim.mean(), 4)))
    print()

    # === Text-based clustering ===
    for n_clusters in [10, 15, 20, 30]:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        cl = clustering.fit_predict(embeddings)
        text_labels = {tactical_fids[i]: int(cl[i]) for i in range(len(tactical_fids))}
        text_purity, text_n = category_purity(text_labels, categories)

        print('Text k=' + str(n_clusters) + ': purity=' + str(round(text_purity, 3)) +
              ' (' + str(text_n) + ' clusters >=3)')

    # === Compare with fire-pattern clustering ===
    # Louvain from Exp 8
    try:
        import community as community_louvain
        import networkx as nx

        fp = fires[:, tactical_fids]
        intersection = fp.T @ fp
        sums = fp.sum(axis=0)
        union = sums[:, None] + sums[None, :] - intersection
        union = np.maximum(union, 1)
        jaccard = intersection / union
        np.fill_diagonal(jaccard, 0)

        G = nx.Graph()
        n = len(tactical_fids)
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if jaccard[i, j] >= 0.05:
                    G.add_edge(i, j, weight=float(jaccard[i, j]))

        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G, weight='weight')
            fire_labels = {tactical_fids[i]: partition[i] for i in range(n)}
            fire_purity, fire_n = category_purity(fire_labels, categories)
            print()
            print('Fire-pattern Louvain: purity=' + str(round(fire_purity, 3)) +
                  ' (' + str(fire_n) + ' clusters >=3)')
    except ImportError:
        print('(skipping fire-pattern comparison, need python-louvain)')

    # === Show best text clusters ===
    print()
    print('=== Top text clusters (k=20) ===')
    clustering = AgglomerativeClustering(n_clusters=20, metric='cosine', linkage='average')
    cl = clustering.fit_predict(embeddings)

    cluster_members = {}
    for i, fid in enumerate(tactical_fids):
        c = int(cl[i])
        cluster_members.setdefault(c, []).append(fid)

    for c, members in sorted(cluster_members.items(), key=lambda x: -len(x[1]))[:12]:
        if len(members) < 3:
            break
        member_labels = [labels[str(f)].get('label', '?')[:50] for f in members]
        member_cats = [categories[f] for f in members]
        majority = Counter(member_cats).most_common(1)[0]
        purity = majority[1] / len(members)

        print('  Cluster ' + str(c) + ' (' + str(len(members)) + ' features, purity=' +
              str(round(purity * 100)) + '% ' + majority[0] + '):')
        for lbl in member_labels[:4]:
            print('    - ' + lbl)
        if len(member_labels) > 4:
            print('    ... +' + str(len(member_labels) - 4) + ' more')
        print()

    # Verdict
    text_labels_20 = {tactical_fids[i]: int(cl[i]) for i in range(len(tactical_fids))}
    text_p, _ = category_purity(text_labels_20, categories)
    print('=== Verdict ===')
    print('Text clustering purity at k=20: ' + str(round(text_p * 100, 1)) + '%')
    print('Prediction was >50%: ' + ('CONFIRMED' if text_p > 0.5 else 'FAILED'))


if __name__ == '__main__':
    main()
