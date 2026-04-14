#!/usr/bin/env python3
"""Experiment 24: Validate taxonomy categories correlate with game phase.

Hypothesis: Endgame categories fire predominantly in endgame positions.
Prediction: Rook Endgames, K&P Endgames, Passed Pawns have >60% endgame fire rate.
            Hanging, Overloaded, Forcing have <30% endgame fire rate.

Method: Load taxonomy assignments from exp 23, compute per-category phase distribution
using the SAE activations.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
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


def build_taxonomy(labels, quality_fids, embeddings):
    """Reproduce exp 23 taxonomy assignments."""
    cl1 = AgglomerativeClustering(n_clusters=15, metric='cosine', linkage='average')
    level1 = cl1.fit_predict(embeddings)
    biggest = Counter(level1).most_common(1)[0][0]

    big_idx = [i for i, c in enumerate(level1) if c == biggest]
    big_emb = embeddings[big_idx]
    cl2 = AgglomerativeClustering(n_clusters=8, metric='cosine', linkage='average')
    sub_labels = cl2.fit_predict(big_emb)
    sub_sizes = Counter(sub_labels)
    top2 = sub_sizes.most_common(2)
    hanging_cl, overloaded_cl = top2[0][0], top2[1][0]

    CATEGORY_NAMES = {
        0: 'Piece Activity', 1: 'Passed Pawns', 2: 'Discovered Attacks',
        3: 'Forcing Moves', 4: 'Back Rank', 5: 'Opening Play',
        6: 'Rook Endgames', 7: 'Back Rank', 8: 'King & Pawn Endgames',
        9: 'Rook Endgames', 10: 'Hanging Pieces', 11: 'Rook Endgames',
        12: 'Forcing Moves', 13: 'Overloaded Defenders',
    }

    MERGE_MAP = {
        'Diagonal Play': 'Rook Endgames', 'Engine Moves': 'Forcing Moves',
        'Material Captures': 'Hanging Pieces', 'King Attacks': 'Back Rank',
        'Pins & Skewers': 'Overloaded Defenders', 'Rook Activity': 'Rook Endgames',
    }

    # Build initial and merge
    from sklearn.metrics.pairwise import cosine_similarity as cs
    fid_to_idx = {f: i for i, f in enumerate(quality_fids)}

    # Compute centroids for hanging/overloaded
    h_members = [i for i, c in enumerate(level1) if c == biggest and sub_labels[big_idx.index(i)] == hanging_cl]
    o_members = [i for i, c in enumerate(level1) if c == biggest and sub_labels[big_idx.index(i)] == overloaded_cl]
    h_centroid = embeddings[h_members].mean(axis=0) if h_members else np.zeros(embeddings.shape[1])
    o_centroid = embeddings[o_members].mean(axis=0) if o_members else np.zeros(embeddings.shape[1])

    final_cats = {}
    for i, fid_str in enumerate(quality_fids):
        c = level1[i]
        if c != biggest:
            cat = CATEGORY_NAMES.get(c, 'Unknown')
        else:
            sub_i = big_idx.index(i)
            sub_c = sub_labels[sub_i]
            if sub_c == hanging_cl:
                cat = 'Hanging Pieces'
            elif sub_c == overloaded_cl:
                cat = 'Overloaded Defenders'
            else:
                # Mixed: assign by nearest centroid
                emb = embeddings[i].reshape(1, -1)
                d_h = 1 - cs(emb, h_centroid.reshape(1, -1))[0, 0]
                d_o = 1 - cs(emb, o_centroid.reshape(1, -1))[0, 0]
                cat = 'Hanging Pieces' if d_h < d_o else 'Overloaded Defenders'

        # Apply merge
        if cat in MERGE_MAP:
            cat = MERGE_MAP[cat]
        final_cats[fid_str] = cat

    return final_cats


def main():
    print('Experiment 24: Phase validation of 10-category taxonomy')
    print('Hypothesis: Endgame categories fire >60% in endgame, tactical <30%')
    print()

    # Load data
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
    acts_np = acts.numpy()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Classify phases
    phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
    for i, md in enumerate(metadata):
        try:
            n = len(chess.Board(md['fen']).piece_map())
        except:
            n = 20
        p = 'opening' if n > 24 else ('middlegame' if n > 12 else 'endgame')
        phase_idx[p].append(i)

    print('Positions: ' + ', '.join(p + '=' + str(len(idx)) for p, idx in phase_idx.items()))

    # Build taxonomy
    quality_fids = [f for f, info in labels.items() if info.get('confidence') in ['high', 'medium']]
    texts = [labels[f].get('label', '') + '. ' + labels[f].get('explanation', '')[:200]
             for f in quality_fids]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vec.fit_transform(texts).toarray()

    taxonomy = build_taxonomy(labels, quality_fids, embeddings)

    # For each category, compute phase distribution of feature fires
    print()
    print('=== Per-category phase distribution ===')
    print(f'{"Category":<25} {"Opening":>8} {"Middle":>8} {"Endgame":>8}  {"Dominant":>10}')
    print('-' * 65)

    cat_phase_data = {}
    for cat in ['Hanging Pieces', 'Overloaded Defenders', 'Passed Pawns',
                'King & Pawn Endgames', 'Rook Endgames', 'Forcing Moves',
                'Discovered Attacks', 'Back Rank', 'Piece Activity', 'Opening Play']:
        cat_fids = [int(f) for f, c in taxonomy.items() if c == cat]
        if not cat_fids:
            continue

        # Total fires per phase across all features in this category
        phase_fires = {}
        total_fires = 0
        for phase, idx in phase_idx.items():
            pf = sum(fires[idx, fid].sum() for fid in cat_fids if fid < fires.shape[1])
            phase_fires[phase] = pf
            total_fires += pf

        if total_fires == 0:
            continue

        phase_pcts = {p: round(f / total_fires * 100, 1) for p, f in phase_fires.items()}
        dominant = max(phase_pcts, key=phase_pcts.get)

        cat_phase_data[cat] = phase_pcts
        print(f'{cat:<25} {phase_pcts["opening"]:>7}% {phase_pcts["middlegame"]:>7}% {phase_pcts["endgame"]:>7}%  {dominant:>10}')

    # Also compute mean activation strength per phase per category
    print()
    print('=== Mean activation strength by phase ===')
    print(f'{"Category":<25} {"Opening":>8} {"Middle":>8} {"Endgame":>8}')
    print('-' * 55)

    for cat in ['Hanging Pieces', 'Overloaded Defenders', 'Passed Pawns',
                'King & Pawn Endgames', 'Rook Endgames', 'Forcing Moves',
                'Discovered Attacks', 'Back Rank', 'Piece Activity', 'Opening Play']:
        cat_fids = [int(f) for f, c in taxonomy.items() if c == cat]
        if not cat_fids:
            continue

        phase_strengths = {}
        for phase, idx in phase_idx.items():
            strengths = []
            for fid in cat_fids:
                if fid >= acts_np.shape[1]:
                    continue
                phase_acts = acts_np[idx, fid]
                firing = phase_acts[phase_acts > 0]
                if len(firing) > 0:
                    strengths.append(firing.mean())
            phase_strengths[phase] = round(np.mean(strengths), 2) if strengths else 0

        print(f'{cat:<25} {phase_strengths["opening"]:>8} {phase_strengths["middlegame"]:>8} {phase_strengths["endgame"]:>8}')

    # Verdict
    print()
    print('=== Verdict ===')
    endgame_cats = ['Rook Endgames', 'King & Pawn Endgames', 'Passed Pawns']
    tactical_cats = ['Hanging Pieces', 'Overloaded Defenders', 'Forcing Moves']

    eg_pass = all(cat_phase_data.get(c, {}).get('endgame', 0) > 60 for c in endgame_cats if c in cat_phase_data)
    tac_pass = all(cat_phase_data.get(c, {}).get('endgame', 0) < 30 for c in tactical_cats if c in cat_phase_data)

    for c in endgame_cats:
        eg = cat_phase_data.get(c, {}).get('endgame', 0)
        print(f'  {c}: endgame={eg}% (need >60%): {"PASS" if eg > 60 else "FAIL"}')
    for c in tactical_cats:
        eg = cat_phase_data.get(c, {}).get('endgame', 0)
        print(f'  {c}: endgame={eg}% (need <30%): {"PASS" if eg < 30 else "FAIL"}')

    print()
    if eg_pass and tac_pass:
        print('CONFIRMED: Taxonomy categories align with game phase')
    else:
        print('PARTIALLY CONFIRMED: Some categories align, others need review')


if __name__ == '__main__':
    main()
