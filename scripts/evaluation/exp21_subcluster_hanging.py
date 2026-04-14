#!/usr/bin/env python3
"""Experiment 21: Sub-cluster the "hanging material" mega-cluster.

Hypothesis: The 1439-feature "hanging material" cluster contains 5-8 distinct coaching sub-topics.
Prediction: Sub-clustering at k=8 produces clusters with >60% internal label coherence
            and distinct coaching themes.

Method: Take features from exp 20's largest cluster, re-cluster by TF-IDF at higher granularity.
"""
import json
import numpy as np
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print('pip install scikit-learn')
    exit(1)


def main():
    print('Experiment 21: Sub-cluster the "hanging material" mega-cluster')
    print('Hypothesis: Hanging cluster contains 5-8 distinct coaching sub-topics')
    print('Prediction: Sub-clusters have >60% label coherence with distinct themes')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Get all quality features
    quality_fids = []
    for fid_str, info in labels.items():
        if info.get('confidence') in ['high', 'medium']:
            quality_fids.append(fid_str)

    # Build TF-IDF for all features (same as exp 20)
    texts_all = []
    for fid_str in quality_fids:
        info = labels[fid_str]
        texts_all.append(info.get('label', '') + '. ' + info.get('explanation', '')[:200])

    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings_all = vec.fit_transform(texts_all).toarray()

    # Cluster at k=15 (same as exp 20) to find the big cluster
    clustering = AgglomerativeClustering(n_clusters=15, metric='cosine', linkage='average')
    cl = clustering.fit_predict(embeddings_all)

    # Find the biggest cluster
    cluster_sizes = Counter(cl)
    biggest_cl = cluster_sizes.most_common(1)[0][0]
    biggest_size = cluster_sizes.most_common(1)[0][1]
    print('Biggest cluster: ' + str(biggest_cl) + ' with ' + str(biggest_size) + ' features')

    # Extract those features
    hanging_idx = [i for i, c in enumerate(cl) if c == biggest_cl]
    hanging_fids = [quality_fids[i] for i in hanging_idx]
    hanging_embeddings = embeddings_all[hanging_idx]

    # Check what Sonnet categories are in here
    sonnet_cats = [labels[f].get('category', '') for f in hanging_fids]
    print('Sonnet categories in mega-cluster:')
    for cat, count in Counter(sonnet_cats).most_common(10):
        print('  ' + cat + ': ' + str(count) + ' (' + str(round(count / len(hanging_fids) * 100, 1)) + '%)')
    print()

    # Sub-cluster at multiple k values
    print('=== Sub-clustering sweep ===')
    for k in [5, 8, 10, 12, 15]:
        sub_cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        sub_labels = sub_cl.fit_predict(hanging_embeddings)

        cluster_members = defaultdict(list)
        for i, c in enumerate(sub_labels):
            cluster_members[c].append(i)

        # Coherence: within each sub-cluster, what fraction share the most common Sonnet category?
        purities = []
        for members in cluster_members.values():
            if len(members) < 5:
                continue
            cats = [sonnet_cats[m] for m in members]
            majority = Counter(cats).most_common(1)[0][1]
            purities.append(majority / len(cats))

        sizes = sorted([len(m) for m in cluster_members.values()], reverse=True)
        mean_purity = np.mean(purities) if purities else 0

        print('  k=' + str(k) + ': purity=' + str(round(mean_purity, 3)) +
              ', sizes=' + str(sizes[:6]))

    # Detailed analysis at k=8
    K = 8
    print()
    print('=== Detailed sub-clusters (k=' + str(K) + ') ===')
    sub_cl = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
    sub_labels = sub_cl.fit_predict(hanging_embeddings)

    cluster_members = defaultdict(list)
    for i, c in enumerate(sub_labels):
        cluster_members[c].append(i)

    sub_cluster_info = []
    for c in sorted(cluster_members.keys()):
        members = cluster_members[c]
        member_fids = [hanging_fids[m] for m in members]
        member_labels_text = [labels[f].get('label', '')[:60] for f in member_fids]
        member_cats = [labels[f].get('category', '') for f in member_fids]

        cat_counter = Counter(member_cats)
        top_cat = cat_counter.most_common(1)[0]
        purity = top_cat[1] / len(members)

        # Top words
        all_text = ' '.join(labels[f].get('label', '') for f in member_fids).lower()
        words = all_text.split()
        stopwords = {'and', 'the', 'in', 'of', 'with', 'for', 'or', 'a', 'to', 'on',
                     'that', 'is', 'are', 'an', 'by', 'from', 'at', 'as', 'be', 'its'}
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 2)
        top_words = [w for w, _ in word_counts.most_common(5)]

        # Suggest coaching name
        coaching_name = '_'.join(top_words[:2])

        print('Sub-cluster ' + str(c) + ' (' + str(len(members)) + ' features):')
        print('  Top category: ' + top_cat[0] + ' (' + str(round(purity * 100)) + '%)')
        print('  Top words: ' + ', '.join(top_words))
        print('  Coaching name: ' + coaching_name)
        print('  Labels:')
        for lbl in member_labels_text[:5]:
            print('    - ' + lbl)
        if len(members) > 5:
            print('    ... +' + str(len(members) - 5) + ' more')
        print()

        sub_cluster_info.append({
            'id': c, 'size': len(members), 'purity': purity,
            'top_category': top_cat[0], 'coaching_name': coaching_name,
            'top_words': top_words,
        })

    # Verdict
    purities = [s['purity'] for s in sub_cluster_info if s['size'] >= 5]
    mean_purity = np.mean(purities) if purities else 0
    n_distinct = sum(1 for s in sub_cluster_info if s['size'] >= 20)

    print('=== Verdict ===')
    print('Sub-clusters with >=20 features: ' + str(n_distinct))
    print('Mean purity (clusters >=5): ' + str(round(mean_purity * 100, 1)) + '%')
    print('Prediction was >60% purity with 5-8 distinct themes:')
    print('  ' + ('CONFIRMED' if mean_purity > 0.6 and 5 <= n_distinct <= 8 else 'PARTIALLY CONFIRMED' if mean_purity > 0.5 else 'FAILED'))

    # Propose final taxonomy
    print()
    print('=== PROPOSED TAXONOMY (14 top-level + hanging sub-clusters) ===')
    print('Non-hanging categories (from exp 20):')
    print('  1. Passed Pawns (509)')
    print('  2. Rook Endgames (366)')
    print('  3. King & Pawn Endgames (510)')
    print('  4. Forcing Moves (233)')
    print('  5. Discovered Attacks (160)')
    print('  6. Back Rank (107)')
    print('  7. Piece Activity (83)')
    print('  8. Opening Development (35)')
    print()
    print('Hanging material sub-categories:')
    for s in sorted(sub_cluster_info, key=lambda x: -x['size']):
        if s['size'] >= 10:
            print('  ' + str(s['size']).rjust(4) + ': ' + s['coaching_name'] +
                  ' (' + str(round(s['purity'] * 100)) + '% ' + s['top_category'] + ')')


if __name__ == '__main__':
    main()
