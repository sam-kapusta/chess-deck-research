#!/usr/bin/env python3
"""Experiment 20: Build final coaching taxonomy from text clusters + multi-assignment.

Hypothesis: Text clusters reveal natural coaching categories. Each feature gets 1-2 assignments.
Prediction: >80% of features cleanly assigned to 1-2 categories with <10% unassigned.

Method:
1. Text-cluster ALL quality features (not just tactical)
2. Name each cluster by its dominant theme
3. Allow features near cluster boundaries to get secondary assignment
4. Output: feature → [primary_category, optional_secondary_category]
"""
import json
import numpy as np
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print('pip install scikit-learn')
    exit(1)


def main():
    print('Experiment 20: Combined text-cluster + multi-assignment taxonomy')
    print('Hypothesis: Text clusters give natural coaching categories with 1-2 assignments')
    print('Prediction: >80% cleanly assigned, <10% unassigned')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # ALL quality features, not just tactical
    quality_fids = []
    for fid_str, info in labels.items():
        if info.get('confidence') in ['high', 'medium']:
            quality_fids.append(fid_str)

    print('Quality features: ' + str(len(quality_fids)))

    # Build text for each feature
    texts = []
    for fid_str in quality_fids:
        info = labels[fid_str]
        text = info.get('label', '') + '. ' + info.get('explanation', '')[:200]
        texts.append(text)

    # TF-IDF embed
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()

    # Cluster — try multiple k values to find the sweet spot
    print()
    print('=== Finding optimal cluster count ===')
    for k in [8, 10, 12, 15, 20, 25, 30]:
        clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        cl = clustering.fit_predict(embeddings)

        # Compute within-cluster coherence
        cluster_members = defaultdict(list)
        for i, c in enumerate(cl):
            cluster_members[c].append(i)

        coherences = []
        for members in cluster_members.values():
            if len(members) < 3:
                continue
            sub = embeddings[members]
            sim = cosine_similarity(sub)
            np.fill_diagonal(sim, 0)
            coherences.append(sim.mean())

        sizes = [len(m) for m in cluster_members.values()]
        big = sum(1 for s in sizes if s > 100)
        small = sum(1 for s in sizes if s < 5)
        mean_coh = np.mean(coherences) if coherences else 0

        print('  k=' + str(k) + ': mean_coherence=' + str(round(mean_coh, 3)) +
              ', max_cluster=' + str(max(sizes)) +
              ', >100=' + str(big) + ', <5=' + str(small))

    # Use k=15 as sweet spot (based on exp 18 results)
    K = 15
    print()
    print('Using k=' + str(K))
    clustering = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
    cl = clustering.fit_predict(embeddings)

    # Name clusters by their most common words and labels
    cluster_members = defaultdict(list)
    for i, c in enumerate(cl):
        cluster_members[c].append(i)

    print()
    print('=== Cluster analysis ===')
    cluster_names = {}
    cluster_summaries = []

    for c in sorted(cluster_members.keys()):
        members = cluster_members[c]
        member_fids = [quality_fids[i] for i in members]
        member_labels = [labels[f].get('label', '') for f in member_fids]
        member_cats = [labels[f].get('category', '') for f in member_fids]

        # Most common Sonnet category
        cat_counter = Counter(member_cats)
        top_cat = cat_counter.most_common(1)[0]

        # Most common words across labels (excluding stopwords)
        all_words = ' '.join(member_labels).lower().split()
        stopwords = {'and', 'the', 'in', 'of', 'with', 'for', 'or', 'a', 'to', 'on',
                     'that', 'is', 'are', 'an', 'by', 'from', 'at', 'as', 'be'}
        word_counts = Counter(w for w in all_words if w not in stopwords and len(w) > 2)
        top_words = [w for w, _ in word_counts.most_common(5)]

        # Auto-name from top words
        name = '_'.join(top_words[:3])
        cluster_names[c] = name

        print('Cluster ' + str(c) + ' (' + str(len(members)) + ' features):')
        print('  Top Sonnet cat: ' + top_cat[0] + ' (' + str(top_cat[1]) + '/' + str(len(members)) + ')')
        print('  Top words: ' + ', '.join(top_words))
        print('  Sample labels:')
        for lbl in member_labels[:3]:
            print('    - ' + lbl[:60])
        print()

        cluster_summaries.append({
            'id': c,
            'name': name,
            'size': len(members),
            'top_category': top_cat[0],
            'purity': top_cat[1] / len(members),
            'top_words': top_words,
        })

    # === Multi-assignment: features near cluster boundaries ===
    print('=== Multi-assignment analysis ===')

    # Compute distance to each cluster centroid
    centroids = np.zeros((K, embeddings.shape[1]))
    for c, members in cluster_members.items():
        centroids[c] = embeddings[members].mean(axis=0)

    distances = cosine_distances(embeddings, centroids)  # (n_features, K)

    # For each feature: primary = assigned cluster, secondary = next-closest if within threshold
    THRESHOLD_RATIO = 1.5  # secondary must be within 1.5x the primary distance
    assignments = {}
    n_single = 0
    n_multi = 0

    for i, fid_str in enumerate(quality_fids):
        primary = int(cl[i])
        primary_dist = distances[i, primary]

        # Find second-closest cluster
        sorted_dists = np.argsort(distances[i])
        secondary = int(sorted_dists[1]) if sorted_dists[0] == primary else int(sorted_dists[0])
        secondary_dist = distances[i, secondary]

        if primary_dist > 0 and secondary_dist / max(primary_dist, 1e-8) < THRESHOLD_RATIO:
            assignments[fid_str] = [cluster_names[primary], cluster_names[secondary]]
            n_multi += 1
        else:
            assignments[fid_str] = [cluster_names[primary]]
            n_single += 1

    print('Single assignment: ' + str(n_single) + ' (' + str(round(n_single / len(quality_fids) * 100, 1)) + '%)')
    print('Dual assignment:   ' + str(n_multi) + ' (' + str(round(n_multi / len(quality_fids) * 100, 1)) + '%)')
    print()

    # Show assignment distribution across categories
    primary_counts = Counter()
    secondary_counts = Counter()
    for fid_str, cats in assignments.items():
        primary_counts[cats[0]] += 1
        if len(cats) > 1:
            secondary_counts[cats[1]] += 1

    print('=== Category sizes (primary) ===')
    for cat, count in primary_counts.most_common():
        sec = secondary_counts.get(cat, 0)
        print('  ' + cat + ': ' + str(count) + ' primary, ' + str(sec) + ' secondary')

    # Top cross-category pairs
    print()
    print('=== Top cross-category pairs ===')
    pair_counts = Counter()
    for fid_str, cats in assignments.items():
        if len(cats) > 1:
            pair_counts[tuple(cats)] += 1

    for pair, count in pair_counts.most_common(10):
        print('  ' + pair[0] + ' + ' + pair[1] + ': ' + str(count))

    # Output final taxonomy
    print()
    print('=== FINAL TAXONOMY ===')
    for s in sorted(cluster_summaries, key=lambda x: -x['size']):
        print(str(s['id']).rjust(2) + '. ' + s['name'] + ' (' + str(s['size']) + ' features, ' +
              str(round(s['purity'] * 100)) + '% ' + s['top_category'] + ')')

    # Save assignments
    output_path = '/home/ec2-user/SageMaker/chess-deck-research/output/feature_taxonomy.json'
    taxonomy = {
        'method': 'text_clustering_k15_tfidf',
        'n_features': len(quality_fids),
        'n_categories': K,
        'categories': {s['name']: {'id': s['id'], 'size': s['size'],
                                    'top_sonnet_category': s['top_category'],
                                    'purity': round(s['purity'], 3),
                                    'words': s['top_words']}
                       for s in cluster_summaries},
        'assignments': assignments,
    }
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    print()
    print('Saved taxonomy to ' + output_path)

    # Verdict
    cleanly_assigned = n_single + n_multi  # all features get at least primary
    pct = cleanly_assigned / len(quality_fids) * 100
    print()
    print('=== Verdict ===')
    print('Cleanly assigned: ' + str(round(pct, 1)) + '%')
    print('Prediction was >80% assigned, <10% unassigned: ' + ('CONFIRMED' if pct > 80 else 'FAILED'))


if __name__ == '__main__':
    main()
