#!/usr/bin/env python3
"""Experiment 26: Build puzzle SAE taxonomy (11 categories including Checkmate).

Hypothesis: Puzzle SAE needs an 11-category taxonomy (10 blunder + Checkmate).
Prediction: Checkmate forms a clean cluster (>80% purity), Forcing Moves sub-clusters into
            2-3 distinct coaching themes.

Method: TF-IDF cluster puzzle SAE labels at k=12, identify natural categories,
compare against blunder taxonomy.
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
    print('Experiment 26: Build puzzle SAE taxonomy')
    print('Hypothesis: Puzzle SAE needs Checkmate category + Forcing Moves sub-clustering')
    print('Prediction: Checkmate >80% purity, Forcing splits into 2-3 themes')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/k64_baseline/labels_sonnet_think.json') as f:
        labels = json.load(f)

    quality_fids = [f for f, info in labels.items() if info.get('confidence') in ['high', 'medium']]
    print('Quality puzzle features: ' + str(len(quality_fids)))

    # Sonnet category distribution
    sonnet_cats = Counter(labels[f].get('category', '?') for f in quality_fids)
    print('Sonnet categories:')
    for cat, count in sonnet_cats.most_common():
        print('  ' + cat + ': ' + str(count))
    print()

    # TF-IDF embed
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in quality_fids]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()

    # Sweep k
    print('=== Cluster sweep ===')
    for k in [8, 10, 12, 15, 20]:
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        cl_labels = cl.fit_predict(embeddings)
        sizes = sorted(Counter(cl_labels).values(), reverse=True)
        print(f'  k={k}: sizes={sizes[:8]}')

    # Use k=12 (exp 25 showed 10 isn't enough, need Checkmate + Forcing splits)
    K = 12
    print(f'\nUsing k={K}')
    cl = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
    cl_labels = cl.fit_predict(embeddings)

    cluster_members = defaultdict(list)
    for i, c in enumerate(cl_labels):
        cluster_members[c].append(i)

    # Analyze each cluster
    print()
    print('=== Puzzle taxonomy clusters ===')
    cluster_info = []
    for c in sorted(cluster_members.keys()):
        members = cluster_members[c]
        member_fids = [quality_fids[m] for m in members]
        member_cats = [labels[f].get('category', '?') for f in member_fids]
        member_labels = [labels[f].get('label', '')[:50] for f in member_fids]

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

        info = {
            'id': c, 'size': len(members), 'purity': purity,
            'top_category': top_cat[0], 'top_words': top_words,
        }
        cluster_info.append(info)

        print(f'Cluster {c} ({len(members)} features, {round(purity*100)}% {top_cat[0]}):')
        print('  Words: ' + ', '.join(top_words))
        for lbl in member_labels[:3]:
            print('    - ' + lbl)
        if len(members) > 3:
            print(f'    ... +{len(members)-3} more')
        print()

    # Propose human-readable names
    print('=== PROPOSED PUZZLE TAXONOMY ===')
    for info in sorted(cluster_info, key=lambda x: -x['size']):
        print(f"  {info['size']:>4}: {info['top_category']} ({round(info['purity']*100)}%) — {', '.join(info['top_words'][:3])}")

    # Check: is Checkmate a distinct cluster?
    checkmate_clusters = [info for info in cluster_info
                          if info['top_category'] == 'checkmate' and info['purity'] > 0.5]
    print()
    print('Checkmate clusters: ' + str(len(checkmate_clusters)))
    for info in checkmate_clusters:
        print(f'  Cluster {info["id"]}: {info["size"]} features, {round(info["purity"]*100)}% checkmate')

    # Save assignments
    assignments = {}
    for i, fid in enumerate(quality_fids):
        c = int(cl_labels[i])
        assignments[fid] = c

    output_path = '/home/ec2-user/SageMaker/chess-deck-research/output/puzzle_taxonomy_v1.json'
    taxonomy = {
        'version': 'puzzle_v1_12cat',
        'sae': 'puzzle_2048_k64',
        'n_features': len(quality_fids),
        'n_categories': K,
        'clusters': {str(info['id']): {
            'size': info['size'],
            'purity': round(info['purity'], 3),
            'top_sonnet_category': info['top_category'],
            'top_words': info['top_words'],
        } for info in cluster_info},
        'assignments': assignments,
    }
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    print(f'\nSaved to {output_path}')

    # Verdict
    print()
    print('=== Verdict ===')
    has_checkmate = any(info['top_category'] == 'checkmate' and info['purity'] > 0.8
                        for info in cluster_info)
    print('Checkmate cluster >80% purity: ' + ('CONFIRMED' if has_checkmate else 'FAILED'))


if __name__ == '__main__':
    main()
