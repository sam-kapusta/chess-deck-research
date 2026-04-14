#!/usr/bin/env python3
"""Experiment 30: Do Lichess tactical themes emerge as separate clusters?

Hypothesis: Forks, pins, skewers, trapped pieces, quiet moves each form distinct
            clusters in the blunder SAE — supporting Lichess-level granularity.
Prediction: At least 5 of these 8 Lichess themes form clusters with >60% purity:
            fork, pin, skewer, discovered attack, hanging piece, deflection/overloaded,
            back rank, trapped piece.

Method: Search blunder labels for Lichess theme keywords, check if features with
matching keywords ended up in the same text clusters (from Exp 20/23), and test
whether finer-grained clustering separates them.
"""
import json
import re
import numpy as np
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print('pip install scikit-learn')
    exit(1)


# Lichess tactical themes mapped to keyword patterns
LICHESS_THEMES = {
    'fork': ['fork', 'double attack', 'two targets', 'simultaneous.*attack', 'attacks.*two'],
    'pin': ['pin', 'pinned', 'pinning', 'absolute pin', 'relative pin'],
    'skewer': ['skewer', 'x-ray', 'xray'],
    'discovered_attack': ['discovered', 'discovery', 'uncovered', 'battery'],
    'hanging_piece': ['hanging', 'undefended', 'unprotected', 'en prise', 'free.*capture',
                      'inadequately defended', 'loose piece'],
    'deflection': ['overloaded', 'overworked', 'deflect', 'removing.*guard',
                   'removing.*defender', 'capture.*defender'],
    'back_rank': ['back rank', 'back-rank', 'backrank', '8th rank', '1st rank'],
    'trapped_piece': ['trapped', 'no escape', 'boxed in', 'restricted'],
    'quiet_move': ['quiet', 'prophyla', 'zwischenzug', 'intermezzo', 'in-between',
                   'intermediate'],
    'sacrifice': ['sacrifice', 'sac ', 'gambit'],
    'attraction': ['attract', 'lure', 'decoy'],
    'interference': ['interfer', 'block.*line', 'interpos'],
    'exposed_king': ['exposed king', 'king.*safe', 'weak.*king', 'king.*danger',
                     'king.*vulnerable', 'pawn.*shelter', 'pawn.*shield'],
    'promotion': ['promot', 'queening', 'advanced pawn.*eighth', 'pawn.*eighth'],
    'zugzwang': ['zugzwang', 'compulsion'],
}


def main():
    print('Experiment 30: Lichess theme granularity in blunder SAE')
    print('Hypothesis: Lichess tactical themes form distinct clusters')
    print('Prediction: ≥5 of 8 core themes form >60% pure clusters')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    quality_fids = [f for f, info in labels.items()
                    if info.get('confidence') in ['high', 'medium']]
    print('Quality features: ' + str(len(quality_fids)))

    # === PART A: How many features match each Lichess theme? ===
    print()
    print('=== PART A: Feature counts per Lichess theme ===')

    theme_features = defaultdict(list)
    for fid in quality_fids:
        info = labels[fid]
        text = (info.get('label', '') + ' ' + info.get('explanation', '')).lower()
        for theme, keywords in LICHESS_THEMES.items():
            if any(re.search(kw, text) for kw in keywords):
                theme_features[theme].append(fid)

    for theme in sorted(LICHESS_THEMES.keys()):
        count = len(theme_features[theme])
        print(f'  {theme:<20}: {count} features')

    # Multi-theme features
    fid_themes = defaultdict(list)
    for theme, fids in theme_features.items():
        for fid in fids:
            fid_themes[fid].append(theme)

    multi = sum(1 for themes in fid_themes.values() if len(themes) >= 2)
    unmatched = len(quality_fids) - len(fid_themes)
    print(f'\n  Multi-theme features: {multi}')
    print(f'  Unmatched (no Lichess theme): {unmatched} ({round(unmatched/len(quality_fids)*100, 1)}%)')

    # === PART B: Do theme-matched features cluster together? ===
    print()
    print('=== PART B: Do Lichess themes form distinct text clusters? ===')

    # Build TF-IDF on all quality features
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in quality_fids]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()
    fid_to_idx = {f: i for i, f in enumerate(quality_fids)}

    # For each Lichess theme with ≥20 features, compute within-theme similarity
    print()
    print('Within-theme text similarity (higher = more coherent):')
    theme_coherences = {}
    for theme in sorted(LICHESS_THEMES.keys()):
        fids = theme_features[theme]
        if len(fids) < 10:
            continue
        idx = [fid_to_idx[f] for f in fids if f in fid_to_idx]
        if len(idx) < 10:
            continue
        sub_emb = embeddings[idx]
        sim = cosine_similarity(sub_emb)
        np.fill_diagonal(sim, 0)
        coherence = sim.mean()
        theme_coherences[theme] = coherence
        print(f'  {theme:<20}: {coherence:.3f} ({len(idx)} features)')

    # === PART C: Cluster at Lichess granularity and check purity ===
    print()
    print('=== PART C: Fine-grained clustering (k=20,25,30) ===')

    for K in [20, 25, 30]:
        cl = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
        cl_labels = cl.fit_predict(embeddings)

        # For each cluster, what Lichess theme dominates?
        cluster_members = defaultdict(list)
        for i, c in enumerate(cl_labels):
            cluster_members[c].append(i)

        theme_purities = defaultdict(list)
        cluster_themes = {}

        for c, members in cluster_members.items():
            if len(members) < 5:
                continue
            member_fids = [quality_fids[m] for m in members]

            # Count Lichess themes in this cluster
            theme_counts = Counter()
            for fid in member_fids:
                for theme in fid_themes.get(fid, ['none']):
                    theme_counts[theme] += 1

            if theme_counts:
                top_theme, top_count = theme_counts.most_common(1)[0]
                purity = top_count / len(members)
                cluster_themes[c] = (top_theme, purity, len(members))
                if top_theme != 'none':
                    theme_purities[top_theme].append(purity)

        # Report per-theme best cluster purity
        print(f'\n  k={K}: Best cluster purity per Lichess theme:')
        themes_above_60 = 0
        for theme in sorted(LICHESS_THEMES.keys()):
            purities = theme_purities.get(theme, [])
            if purities:
                best = max(purities)
                if best > 0.6:
                    themes_above_60 += 1
                marker = ' ✓' if best > 0.6 else ''
                print(f'    {theme:<20}: {best:.0%} purity{marker}')

        print(f'  Themes with >60% purity: {themes_above_60}')

    # === PART D: Show the actual clusters at k=25 ===
    print()
    print('=== PART D: Cluster details at k=25 ===')
    K = 25
    cl = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
    cl_labels = cl.fit_predict(embeddings)

    cluster_members = defaultdict(list)
    for i, c in enumerate(cl_labels):
        cluster_members[c].append(i)

    for c in sorted(cluster_members.keys()):
        members = cluster_members[c]
        if len(members) < 10:
            continue
        member_fids = [quality_fids[m] for m in members]
        member_labels_text = [labels[f].get('label', '')[:50] for f in member_fids]

        # Dominant Lichess theme
        theme_counts = Counter()
        for fid in member_fids:
            for theme in fid_themes.get(fid, ['none']):
                theme_counts[theme] += 1
        top_theme, top_count = theme_counts.most_common(1)[0]
        purity = top_count / len(members)

        # Top words
        all_text = ' '.join(labels[f].get('label', '') for f in member_fids).lower()
        words = all_text.split()
        stopwords = {'and', 'the', 'in', 'of', 'with', 'for', 'or', 'a', 'to', 'on',
                     'that', 'is', 'are', 'an', 'by', 'from', 'at', 'as', 'be', 'its'}
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 2)
        top_words = [w for w, _ in word_counts.most_common(4)]

        print(f'\n  Cluster {c} ({len(members)} features, {round(purity*100)}% {top_theme}):')
        print(f'    Words: {", ".join(top_words)}')
        for lbl in member_labels_text[:3]:
            print(f'      - {lbl}')

    # === VERDICT ===
    print()
    print('=== Verdict ===')
    # Count themes that achieved >60% purity at k=25
    cl = AgglomerativeClustering(n_clusters=25, metric='cosine', linkage='average')
    cl_labels = cl.fit_predict(embeddings)
    cluster_members = defaultdict(list)
    for i, c in enumerate(cl_labels):
        cluster_members[c].append(i)

    theme_best = {}
    for c, members in cluster_members.items():
        if len(members) < 5:
            continue
        member_fids = [quality_fids[m] for m in members]
        theme_counts = Counter()
        for fid in member_fids:
            for theme in fid_themes.get(fid, ['none']):
                theme_counts[theme] += 1
        for theme, count in theme_counts.items():
            if theme == 'none':
                continue
            purity = count / len(members)
            if theme not in theme_best or purity > theme_best[theme]:
                theme_best[theme] = purity

    core_themes = ['fork', 'pin', 'discovered_attack', 'hanging_piece',
                   'deflection', 'back_rank', 'trapped_piece', 'skewer']
    above_60 = sum(1 for t in core_themes if theme_best.get(t, 0) > 0.6)
    print(f'Core Lichess themes with >60% cluster purity: {above_60}/8')
    for t in core_themes:
        p = theme_best.get(t, 0)
        print(f'  {t:<20}: {p:.0%} {"✓" if p > 0.6 else "✗"}')
    print(f'\nPrediction was ≥5: {"CONFIRMED" if above_60 >= 5 else "FAILED"}')


if __name__ == '__main__':
    main()
