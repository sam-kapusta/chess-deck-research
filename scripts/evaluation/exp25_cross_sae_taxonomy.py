#!/usr/bin/env python3
"""Experiment 25: Does the blunder SAE taxonomy transfer to the puzzle SAE?

Hypothesis: The 10-category coaching taxonomy works for puzzle SAE features too.
Prediction: >80% of puzzle SAE features map to one of the 10 categories via text clustering.

Method: Load puzzle SAE labels (Sonnet+thinking), embed with TF-IDF, assign to nearest
blunder taxonomy centroid. Check if assignments make sense.
"""
import json
import numpy as np
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    print('pip install scikit-learn')
    exit(1)


def main():
    print('Experiment 25: Cross-SAE taxonomy transfer')
    print('Hypothesis: Blunder taxonomy works for puzzle SAE features')
    print('Prediction: >80% of puzzle features map cleanly to 10 categories')
    print()

    # Load blunder labels + taxonomy
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        blunder_labels = json.load(f)

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/feature_taxonomy_v2.json') as f:
        taxonomy = json.load(f)

    # Load puzzle labels
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/k64_baseline/labels_sonnet_think.json') as f:
        puzzle_labels = json.load(f)

    print('Blunder features: ' + str(len(blunder_labels)))
    print('Puzzle features: ' + str(len(puzzle_labels)))
    print()

    # Build TF-IDF on COMBINED corpus (blunder + puzzle)
    blunder_fids = [f for f, info in blunder_labels.items() if info.get('confidence') in ['high', 'medium']]
    puzzle_fids = [f for f, info in puzzle_labels.items() if info.get('confidence') in ['high', 'medium']]

    blunder_texts = [blunder_labels[f].get('label', '') + '. ' + blunder_labels[f].get('explanation', '')[:200]
                     for f in blunder_fids]
    puzzle_texts = [puzzle_labels[f].get('label', '') + '. ' + puzzle_labels[f].get('explanation', '')[:200]
                    for f in puzzle_fids]

    # Fit TF-IDF on combined, transform separately
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    all_texts = blunder_texts + puzzle_texts
    all_embeddings = vec.fit_transform(all_texts).toarray()

    blunder_emb = all_embeddings[:len(blunder_texts)]
    puzzle_emb = all_embeddings[len(blunder_texts):]

    # Compute blunder taxonomy centroids
    centroids = {}
    for cat_name in taxonomy['categories']:
        cat_fids = [f for f, c in taxonomy['assignments'].items() if c == cat_name]
        cat_idx = [blunder_fids.index(f) for f in cat_fids if f in blunder_fids]
        if cat_idx:
            centroids[cat_name] = blunder_emb[cat_idx].mean(axis=0)

    print('Taxonomy centroids computed for ' + str(len(centroids)) + ' categories')

    # Assign each puzzle feature to nearest centroid
    centroid_names = list(centroids.keys())
    centroid_matrix = np.array([centroids[c] for c in centroid_names])

    distances = cosine_distances(puzzle_emb, centroid_matrix)
    nearest = distances.argmin(axis=1)
    nearest_dist = distances.min(axis=1)

    puzzle_assignments = {}
    for i, fid in enumerate(puzzle_fids):
        puzzle_assignments[fid] = {
            'category': centroid_names[nearest[i]],
            'distance': round(float(nearest_dist[i]), 4),
        }

    # Category distribution
    cat_counts = Counter(a['category'] for a in puzzle_assignments.values())
    print()
    print('=== Puzzle SAE category distribution ===')
    for cat, count in cat_counts.most_common():
        blunder_count = taxonomy['categories'].get(cat, {}).get('size', 0)
        print(f'  {cat}: {count} puzzle ({round(count/len(puzzle_fids)*100, 1)}%) vs {blunder_count} blunder')

    # Compare distributions
    print()
    print('=== Distribution comparison (blunder vs puzzle %) ===')
    for cat in sorted(taxonomy['categories'].keys()):
        b_pct = taxonomy['categories'][cat]['size'] / len(blunder_fids) * 100
        p_pct = cat_counts.get(cat, 0) / len(puzzle_fids) * 100
        delta = p_pct - b_pct
        marker = '>>>' if abs(delta) > 5 else ''
        print(f'  {cat:<25} blunder={b_pct:5.1f}%  puzzle={p_pct:5.1f}%  delta={delta:+5.1f}% {marker}')

    # Check assignment quality: distance distribution
    dists = [a['distance'] for a in puzzle_assignments.values()]
    print()
    print('=== Assignment distances (lower = more confident) ===')
    print('  Mean: ' + str(round(np.mean(dists), 3)))
    print('  Median: ' + str(round(np.median(dists), 3)))
    print('  <0.5 (confident): ' + str(sum(1 for d in dists if d < 0.5)) +
          ' (' + str(round(sum(1 for d in dists if d < 0.5) / len(dists) * 100, 1)) + '%)')
    print('  <0.7 (reasonable): ' + str(sum(1 for d in dists if d < 0.7)) +
          ' (' + str(round(sum(1 for d in dists if d < 0.7) / len(dists) * 100, 1)) + '%)')
    print('  >0.9 (poor fit): ' + str(sum(1 for d in dists if d > 0.9)) +
          ' (' + str(round(sum(1 for d in dists if d > 0.9) / len(dists) * 100, 1)) + '%)')

    # Show confident examples per category
    print()
    print('=== Most confident puzzle assignments per category ===')
    for cat in sorted(taxonomy['categories'].keys()):
        cat_features = [(f, a) for f, a in puzzle_assignments.items() if a['category'] == cat]
        cat_features.sort(key=lambda x: x[1]['distance'])
        if not cat_features:
            continue
        print(f'\n  {cat} ({len(cat_features)} features):')
        for fid, a in cat_features[:3]:
            lbl = puzzle_labels[fid].get('label', '')[:50]
            print(f'    F{fid} (d={a["distance"]:.3f}): {lbl}')

    # Show worst fits (might be categories not in the blunder taxonomy)
    print()
    print('=== Worst fits (potential new categories) ===')
    worst = sorted(puzzle_assignments.items(), key=lambda x: -x[1]['distance'])[:15]
    for fid, a in worst:
        lbl = puzzle_labels[fid].get('label', '')[:50]
        cat = puzzle_labels[fid].get('category', '?')
        print(f'  F{fid} (d={a["distance"]:.3f}, assigned={a["category"]}, sonnet_cat={cat}): {lbl}')

    # Verdict
    confident = sum(1 for d in dists if d < 0.7)
    pct = confident / len(dists) * 100
    print()
    print('=== Verdict ===')
    print(f'Puzzle features with reasonable assignment (<0.7 distance): {confident}/{len(dists)} ({pct:.1f}%)')
    print('Prediction was >80%: ' + ('CONFIRMED' if pct > 80 else 'FAILED'))


if __name__ == '__main__':
    main()
