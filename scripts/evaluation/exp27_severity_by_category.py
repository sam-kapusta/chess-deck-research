#!/usr/bin/env python3
"""Experiment 27: Does blunder severity correlate with taxonomy category?

Hypothesis: Severe blunders (>700cp) concentrate in endgame categories,
            mild blunders (<300cp) concentrate in tactical categories.
Prediction: >40% of severe blunders are in {Rook Endgames, K&P Endgames, Passed Pawns},
            >50% of mild blunders are in {Hanging Pieces, Overloaded Defenders}.

Extends Exp 4 (severity by Sonnet category) with the validated 10-category taxonomy.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
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


def main():
    print('Experiment 27: Blunder severity by taxonomy category')
    print('Hypothesis: Severe blunders → endgame categories, mild → tactical')
    print()

    # Load
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
    acts_np = acts.numpy()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Load taxonomy
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/feature_taxonomy_v2.json') as f:
        taxonomy = json.load(f)

    # Severity buckets
    severity_idx = {'mild': [], 'medium': [], 'severe': []}
    for i, md in enumerate(metadata):
        cp = md.get('cp_loss', 0)
        if isinstance(cp, str):
            try:
                cp = int(cp)
            except:
                cp = 200
        if cp < 300:
            severity_idx['mild'].append(i)
        elif cp < 700:
            severity_idx['medium'].append(i)
        else:
            severity_idx['severe'].append(i)

    print('Severity distribution:')
    for sev, idx in severity_idx.items():
        print('  ' + sev + ': ' + str(len(idx)))
    print()

    # For each position, find the STRONGEST category
    # (sum activation strength across all features in each category)
    cat_to_fids = defaultdict(list)
    for fid_str, cat in taxonomy['assignments'].items():
        cat_to_fids[cat].append(int(fid_str))

    # Per-position dominant category (by total activation strength)
    CATEGORIES = list(taxonomy['categories'].keys())
    position_cats = []
    for i in range(len(metadata)):
        cat_strengths = {}
        for cat in CATEGORIES:
            fids = cat_to_fids[cat]
            valid_fids = [f for f in fids if f < acts_np.shape[1]]
            if valid_fids:
                cat_strengths[cat] = acts_np[i, valid_fids].sum()
            else:
                cat_strengths[cat] = 0
        position_cats.append(max(cat_strengths, key=cat_strengths.get))

    # Per severity: dominant category distribution
    print('=== Dominant category by severity ===')
    print(f'{"Category":<25} {"Mild":>8} {"Medium":>8} {"Severe":>8}')
    print('-' * 55)

    for cat in CATEGORIES:
        mild_pct = sum(1 for i in severity_idx['mild'] if position_cats[i] == cat) / max(len(severity_idx['mild']), 1) * 100
        med_pct = sum(1 for i in severity_idx['medium'] if position_cats[i] == cat) / max(len(severity_idx['medium']), 1) * 100
        sev_pct = sum(1 for i in severity_idx['severe'] if position_cats[i] == cat) / max(len(severity_idx['severe']), 1) * 100

        # Mark significant shifts
        shift = sev_pct - mild_pct
        marker = ' ↑↑' if shift > 5 else (' ↓↓' if shift < -5 else '')

        print(f'{cat:<25} {mild_pct:>7.1f}% {med_pct:>7.1f}% {sev_pct:>7.1f}%{marker}')

    # Mean activation strength per category per severity
    print()
    print('=== Mean activation strength by severity ===')
    print(f'{"Category":<25} {"Mild":>8} {"Medium":>8} {"Severe":>8} {"Ratio":>8}')
    print('-' * 63)

    for cat in CATEGORIES:
        fids = [f for f in cat_to_fids[cat] if f < acts_np.shape[1]]
        if not fids:
            continue

        mild_str = np.mean([acts_np[i, fids].sum() for i in severity_idx['mild']])
        med_str = np.mean([acts_np[i, fids].sum() for i in severity_idx['medium']])
        sev_str = np.mean([acts_np[i, fids].sum() for i in severity_idx['severe']])
        ratio = sev_str / max(mild_str, 1e-8)

        print(f'{cat:<25} {mild_str:>8.1f} {med_str:>8.1f} {sev_str:>8.1f} {ratio:>7.2f}x')

    # Verdict
    endgame_cats = {'Rook Endgames', 'King & Pawn Endgames', 'Passed Pawns'}
    tactical_cats = {'Hanging Pieces', 'Overloaded Defenders'}

    sev_endgame = sum(1 for i in severity_idx['severe'] if position_cats[i] in endgame_cats)
    sev_total = len(severity_idx['severe'])
    sev_eg_pct = sev_endgame / max(sev_total, 1) * 100

    mild_tactical = sum(1 for i in severity_idx['mild'] if position_cats[i] in tactical_cats)
    mild_total = len(severity_idx['mild'])
    mild_tac_pct = mild_tactical / max(mild_total, 1) * 100

    print()
    print('=== Verdict ===')
    print(f'Severe blunders in endgame categories: {sev_eg_pct:.1f}% (prediction >40%): ' +
          ('CONFIRMED' if sev_eg_pct > 40 else 'FAILED'))
    print(f'Mild blunders in tactical categories: {mild_tac_pct:.1f}% (prediction >50%): ' +
          ('CONFIRMED' if mild_tac_pct > 50 else 'FAILED'))


if __name__ == '__main__':
    main()
