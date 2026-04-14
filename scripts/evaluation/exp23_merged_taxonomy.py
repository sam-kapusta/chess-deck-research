#!/usr/bin/env python3
"""Experiment 23: Merge small clusters into 10-category coaching taxonomy.

Hypothesis: Merging 17 clusters into 10 improves within/cross similarity ratio to >2x.
Prediction: Ratio goes from 1.33x to >2.0x while keeping spot-check quality.

Method:
1. Start from exp 22's 17 clusters
2. Merge tiny clusters into nearest major category
3. Merge "Mixed Tactical" into Hanging/Overloaded by nearest centroid
4. Re-compute similarity metrics
5. Output final feature_taxonomy_v2.json
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


# Merge map: source → target
MERGE_MAP = {
    'Diagonal Play': 'Rook Endgames',       # 7 features, endgame-related
    'Engine Moves': 'Forcing Moves',          # 8 features, "best moves" = finding forcing moves
    'Material Captures': 'Hanging Pieces',    # 16 features, capturing material
    'King Attacks': 'Back Rank',              # 13 features, king safety related
    'Pins & Skewers': 'Overloaded Defenders', # 13 features, piece exploitation
    'Rook Activity': 'Rook Endgames',         # 30 features, rook technique
    'Mixed Tactical': None,                   # 77 features — split by nearest centroid
}

# Final 10 categories
FINAL_CATEGORIES = [
    'Hanging Pieces',
    'Overloaded Defenders',
    'Passed Pawns',
    'King & Pawn Endgames',
    'Rook Endgames',
    'Forcing Moves',
    'Discovered Attacks',
    'Back Rank',
    'Piece Activity',
    'Opening Play',
]


def main():
    print('Experiment 23: Merged 10-category taxonomy')
    print('Hypothesis: Merging to 10 categories improves ratio to >2x')
    print('Prediction: Within/cross ratio > 2.0x')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    quality_fids = [f for f, info in labels.items()
                    if info.get('confidence') in ['high', 'medium']]
    print('Quality features: ' + str(len(quality_fids)))

    # Build TF-IDF
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in quality_fids]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()
    fid_to_idx = {f: i for i, f in enumerate(quality_fids)}

    # Level 1: k=15
    cl1 = AgglomerativeClustering(n_clusters=15, metric='cosine', linkage='average')
    level1 = cl1.fit_predict(embeddings)
    biggest = Counter(level1).most_common(1)[0][0]

    # Level 2: sub-cluster biggest
    big_idx = [i for i, c in enumerate(level1) if c == biggest]
    big_emb = embeddings[big_idx]
    cl2 = AgglomerativeClustering(n_clusters=8, metric='cosine', linkage='average')
    sub_labels = cl2.fit_predict(big_emb)
    sub_sizes = Counter(sub_labels)
    top2 = sub_sizes.most_common(2)
    hanging_cl = top2[0][0]
    overloaded_cl = top2[1][0]

    # Map from exp 22 cluster IDs to names (same as exp 22)
    CATEGORY_NAMES_BY_CL = {
        0: 'Piece Activity', 1: 'Passed Pawns', 2: 'Discovered Attacks',
        3: 'Forcing Moves', 4: 'King Attacks', 5: 'Opening Play',
        6: 'Rook Endgames', 7: 'Back Rank', 8: 'King & Pawn Endgames',
        9: 'Diagonal Play', 10: 'Material Captures', 11: 'Rook Activity',
        12: 'Engine Moves', 13: 'Pins & Skewers',
    }

    # Build initial 17-category assignments
    initial_cats = {}
    for i, fid_str in enumerate(quality_fids):
        c = level1[i]
        if c != biggest:
            initial_cats[fid_str] = CATEGORY_NAMES_BY_CL.get(c, 'Unknown')
        else:
            sub_i = big_idx.index(i)
            sub_c = sub_labels[sub_i]
            if sub_c == hanging_cl:
                initial_cats[fid_str] = 'Hanging Pieces'
            elif sub_c == overloaded_cl:
                initial_cats[fid_str] = 'Overloaded Defenders'
            else:
                initial_cats[fid_str] = 'Mixed Tactical'

    # Apply merge map
    # First compute centroids for final categories (before merging Mixed)
    cat_members_initial = defaultdict(list)
    for f, cat in initial_cats.items():
        cat_members_initial[cat].append(f)

    final_centroids = {}
    for cat in FINAL_CATEGORIES:
        members = cat_members_initial.get(cat, [])
        if members:
            idx = [fid_to_idx[f] for f in members]
            final_centroids[cat] = embeddings[idx].mean(axis=0)

    # Now merge
    final_cats = {}
    for fid_str, cat in initial_cats.items():
        if cat in FINAL_CATEGORIES:
            final_cats[fid_str] = cat
        elif cat in MERGE_MAP:
            target = MERGE_MAP[cat]
            if target is not None:
                final_cats[fid_str] = target
            else:
                # Mixed Tactical: assign to nearest centroid among Hanging/Overloaded
                idx = fid_to_idx[fid_str]
                emb = embeddings[idx]
                dist_h = 1 - cosine_similarity(emb.reshape(1, -1), final_centroids['Hanging Pieces'].reshape(1, -1))[0, 0]
                dist_o = 1 - cosine_similarity(emb.reshape(1, -1), final_centroids['Overloaded Defenders'].reshape(1, -1))[0, 0]
                final_cats[fid_str] = 'Hanging Pieces' if dist_h < dist_o else 'Overloaded Defenders'
        else:
            final_cats[fid_str] = cat  # keep as-is (shouldn't happen)

    # Final category sizes
    cat_sizes = Counter(final_cats.values())
    print()
    print('=== Final 10-category taxonomy ===')
    for cat in FINAL_CATEGORIES:
        print('  ' + cat + ': ' + str(cat_sizes.get(cat, 0)))

    total_assigned = sum(cat_sizes.values())
    print()
    print('Total assigned: ' + str(total_assigned) + '/' + str(len(quality_fids)))

    # Compute within/cross similarity
    cat_members_final = defaultdict(list)
    for f, cat in final_cats.items():
        cat_members_final[cat].append(f)

    cat_within_sim = {}
    cat_centroids_final = {}
    for cat in FINAL_CATEGORIES:
        members = cat_members_final[cat]
        if len(members) < 3:
            continue
        idx = [fid_to_idx[f] for f in members]
        sub_emb = embeddings[idx]
        cat_centroids_final[cat] = sub_emb.mean(axis=0)
        sim = cosine_similarity(sub_emb)
        np.fill_diagonal(sim, 0)
        cat_within_sim[cat] = sim.mean()

    print()
    print('Within-category similarity:')
    for cat, sim in sorted(cat_within_sim.items(), key=lambda x: -x[1]):
        print('  ' + cat + ': ' + str(round(sim, 3)))

    # Cross-category
    cats = sorted(cat_centroids_final.keys())
    centroid_matrix = np.array([cat_centroids_final[c] for c in cats])
    cross_sim = cosine_similarity(centroid_matrix)
    np.fill_diagonal(cross_sim, 0)

    print()
    print('Cross-category:')
    print('  Mean: ' + str(round(cross_sim.mean(), 3)))
    print('  Max: ' + str(round(cross_sim.max(), 3)))

    print()
    print('Most similar pairs:')
    pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs.append((cats[i], cats[j], cross_sim[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for c1, c2, sim in pairs[:5]:
        print('  ' + c1 + ' <-> ' + c2 + ': ' + str(round(sim, 3)))

    within_mean = np.mean(list(cat_within_sim.values()))
    cross_mean = cross_sim.mean()
    ratio = within_mean / max(cross_mean, 1e-8)

    print()
    print('=== Verdict ===')
    print('Mean within: ' + str(round(within_mean, 3)))
    print('Mean cross: ' + str(round(cross_mean, 3)))
    print('Ratio: ' + str(round(ratio, 2)) + 'x')
    print('Prediction was >2.0x: ' + ('CONFIRMED' if ratio > 2.0 else 'FAILED'))

    # Save final taxonomy
    output_path = '/home/ec2-user/SageMaker/chess-deck-research/output/feature_taxonomy_v2.json'
    taxonomy = {
        'version': 'v2_merged_10cat',
        'method': 'tfidf_hierarchical_k15_subclustered_merged',
        'n_features': len(quality_fids),
        'n_categories': 10,
        'categories': {cat: {'size': int(cat_sizes.get(cat, 0)),
                             'within_similarity': round(float(cat_within_sim.get(cat, 0)), 3)}
                       for cat in FINAL_CATEGORIES},
        'within_cross_ratio': round(float(ratio), 2),
        'assignments': final_cats,
    }
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    print('Saved to ' + output_path)


if __name__ == '__main__':
    main()
