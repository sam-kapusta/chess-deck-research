#!/usr/bin/env python3
"""Experiment 22: Validate the 10-category taxonomy with spot checks.

Hypothesis: The proposed taxonomy correctly categorizes >85% of features.
Prediction: Random 5-feature samples from each category have >4/5 correct assignments.

Method: Build the full taxonomy (exp 20 + exp 21 sub-clustering), sample 5 features
per category, print their labels+examples for human review. Also compute:
- Within-category label similarity (are labels in same category actually about same thing?)
- Cross-category distinctness (are categories actually different from each other?)
"""
import json
import random
import numpy as np
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print('pip install scikit-learn')
    exit(1)


# Human-readable category names (from exp 20+21 analysis)
CATEGORY_NAMES = {
    0: 'Piece Activity',
    1: 'Passed Pawns',
    2: 'Discovered Attacks',
    3: 'Forcing Moves',
    4: 'King Attacks',
    5: 'Opening Play',
    6: 'Rook Endgames',
    7: 'Back Rank',
    8: 'King & Pawn Endgames',
    9: 'Diagonal Play',
    10: 'Material Captures',
    11: 'Rook Activity',
    12: 'Engine Moves',
    13: 'Pins & Skewers',
    # 14 gets sub-clustered
    '14_a': 'Hanging Pieces',
    '14_b': 'Overloaded Defenders',
    '14_mixed': 'Mixed Tactical',
}


def main():
    print('Experiment 22: Taxonomy validation via spot-check + similarity')
    print('Hypothesis: >85% of features correctly categorized')
    print('Prediction: >4/5 correct per category sample')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    quality_fids = [fid_str for fid_str, info in labels.items()
                    if info.get('confidence') in ['high', 'medium']]
    print('Quality features: ' + str(len(quality_fids)))

    # Build TF-IDF
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in quality_fids]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()

    # Level 1: k=15 clustering
    cl1 = AgglomerativeClustering(n_clusters=15, metric='cosine', linkage='average')
    level1 = cl1.fit_predict(embeddings)

    # Find biggest cluster (should be ~14)
    biggest = Counter(level1).most_common(1)[0][0]

    # Level 2: sub-cluster the biggest at k=8
    big_idx = [i for i, c in enumerate(level1) if c == biggest]
    big_emb = embeddings[big_idx]
    cl2 = AgglomerativeClustering(n_clusters=8, metric='cosine', linkage='average')
    sub_labels = cl2.fit_predict(big_emb)

    # Identify the two main sub-clusters (biggest two)
    sub_sizes = Counter(sub_labels)
    top2 = sub_sizes.most_common(2)
    hanging_cl = top2[0][0]  # larger = hanging
    overloaded_cl = top2[1][0]  # second = overloaded

    # Build final assignments
    final_categories = {}
    for i, fid_str in enumerate(quality_fids):
        c = level1[i]
        if c != biggest:
            cat_name = CATEGORY_NAMES.get(c, f'Cluster_{c}')
        else:
            # Sub-clustered
            sub_i = big_idx.index(i)
            sub_c = sub_labels[sub_i]
            if sub_c == hanging_cl:
                cat_name = 'Hanging Pieces'
            elif sub_c == overloaded_cl:
                cat_name = 'Overloaded Defenders'
            else:
                cat_name = 'Mixed Tactical'
        final_categories[fid_str] = cat_name

    # Category sizes
    cat_sizes = Counter(final_categories.values())
    print()
    print('=== Final category sizes ===')
    for cat, size in cat_sizes.most_common():
        print('  ' + cat + ': ' + str(size))

    # Group features by category
    cat_members = defaultdict(list)
    for fid_str, cat in final_categories.items():
        cat_members[cat].append(fid_str)

    # === SPOT CHECK: 5 random features per category ===
    print()
    print('=' * 70)
    print('SPOT CHECK: 5 random features per category')
    print('For each: does the label match the coaching theme?')
    print('=' * 70)

    random.seed(42)
    for cat in sorted(cat_members.keys()):
        members = cat_members[cat]
        if len(members) < 5:
            sample = members
        else:
            sample = random.sample(members, 5)

        print()
        print('--- ' + cat + ' (' + str(len(members)) + ' features) ---')
        for fid_str in sample:
            info = labels[fid_str]
            print('  F' + fid_str + ': ' + info.get('label', '')[:60])
            # Show 2 example FENs
            examples = info.get('examples', [])[:2]
            if examples:
                for ex in examples:
                    if isinstance(ex, dict):
                        print('    FEN: ' + ex.get('fen', '')[:50] + '  move: ' + ex.get('move', ''))
                    elif isinstance(ex, str):
                        print('    FEN: ' + ex[:50])

    # === SIMILARITY CHECK ===
    print()
    print('=' * 70)
    print('SIMILARITY: Within-category vs cross-category')
    print('=' * 70)

    # Build category-level embeddings (mean of member embeddings)
    fid_to_idx = {f: i for i, f in enumerate(quality_fids)}
    cat_centroids = {}
    cat_within_sim = {}

    for cat, members in cat_members.items():
        if len(members) < 3:
            continue
        idx = [fid_to_idx[f] for f in members]
        sub_emb = embeddings[idx]
        centroid = sub_emb.mean(axis=0)
        cat_centroids[cat] = centroid

        # Within-category similarity
        sim = cosine_similarity(sub_emb)
        np.fill_diagonal(sim, 0)
        cat_within_sim[cat] = sim.mean()

    print()
    print('Within-category similarity (higher = more coherent):')
    for cat, sim in sorted(cat_within_sim.items(), key=lambda x: -x[1]):
        print('  ' + cat + ': ' + str(round(sim, 3)))

    # Cross-category similarity
    cats = sorted(cat_centroids.keys())
    centroid_matrix = np.array([cat_centroids[c] for c in cats])
    cross_sim = cosine_similarity(centroid_matrix)
    np.fill_diagonal(cross_sim, 0)

    print()
    print('Cross-category similarity (lower = more distinct):')
    print('  Mean: ' + str(round(cross_sim.mean(), 3)))
    print('  Max: ' + str(round(cross_sim.max(), 3)))

    # Most similar pairs
    print()
    print('Most similar category pairs:')
    pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs.append((cats[i], cats[j], cross_sim[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for c1, c2, sim in pairs[:5]:
        print('  ' + c1 + ' <-> ' + c2 + ': ' + str(round(sim, 3)))

    # Most distinct pairs
    print()
    print('Most distinct category pairs:')
    for c1, c2, sim in pairs[-5:]:
        print('  ' + c1 + ' <-> ' + c2 + ': ' + str(round(sim, 3)))

    # Verdict
    within_mean = np.mean(list(cat_within_sim.values()))
    cross_mean = cross_sim.mean()
    ratio = within_mean / max(cross_mean, 1e-8)
    print()
    print('=== Verdict ===')
    print('Mean within-category similarity: ' + str(round(within_mean, 3)))
    print('Mean cross-category similarity: ' + str(round(cross_mean, 3)))
    print('Ratio (within/cross): ' + str(round(ratio, 2)) + 'x')
    print('Good taxonomy if ratio > 2x: ' + ('YES' if ratio > 2 else 'NO'))


if __name__ == '__main__':
    main()
