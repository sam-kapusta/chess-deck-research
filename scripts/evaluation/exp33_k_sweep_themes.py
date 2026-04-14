#!/usr/bin/env python3
"""Experiment 33: Do all 16 Lichess themes survive at lower k?

At k=32, we found 16 viable themes. At k=8, do they all still have meaningful
position coverage? Or do rare themes vanish?

Method: For each k in [1,2,4,8,16,32], compute position coverage per theme.
A theme "survives" if it covers >5% of positions at that k.
"""
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k
    def forward(self, x, k_override=None):
        k = k_override or self.k
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, k, dim=-1)
        a = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


LICHESS_THEMES = {
    'fork': ['fork', 'double attack', 'two targets', 'simultaneous.*attack'],
    'pin': ['pin', 'pinned', 'pinning'],
    'skewer': ['skewer', 'x-ray', 'xray'],
    'discovered_attack': ['discovered', 'discovery', 'uncovered', 'battery'],
    'hanging_piece': ['hanging', 'undefended', 'unprotected', 'inadequately defended'],
    'deflection': ['overloaded', 'overworked', 'deflect', 'removing.*defender'],
    'back_rank': ['back rank', 'back-rank', 'backrank'],
    'trapped_piece': ['trapped', 'no escape', 'boxed in'],
    'quiet_move': ['quiet', 'prophyla', 'zwischenzug', 'intermezzo'],
    'sacrifice': ['sacrifice', 'sac '],
    'exposed_king': ['exposed king', 'king.*safe', 'weak.*king', 'king.*vulnerable',
                     'pawn.*shelter', 'pawn.*shield'],
    'passed_pawn': ['passed pawn', 'promot', 'advanced pawn'],
    'rook_endgame': ['rook.*endgame', 'rook.*ending', 'rook.*versus'],
    'pawn_endgame': ['pawn.*endgame', 'pawn.*ending', 'king.*pawn.*endgame'],
    'checkmate': ['checkmate', 'mating', 'mate in', 'forced mate'],
    'other': [],  # catch-all
}


def main():
    print('Experiment 33: Theme survival across k values')
    print()

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()

    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 2048, 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8
    normalized = (data - mean) / std

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Map features to themes
    fid_to_themes = defaultdict(set)
    theme_fids = defaultdict(set)
    for fid_str, info in labels.items():
        if info.get('confidence') not in ['high', 'medium']:
            continue
        fid = int(fid_str)
        if fid >= 2048:
            continue
        text = (info.get('label', '') + ' ' + info.get('explanation', '')).lower()
        matched = False
        for theme, keywords in LICHESS_THEMES.items():
            if theme == 'other':
                continue
            if any(re.search(kw, text) for kw in keywords):
                fid_to_themes[fid].add(theme)
                theme_fids[theme].add(fid)
                matched = True
        if not matched:
            fid_to_themes[fid].add('other')
            theme_fids['other'].add(fid)

    print('Features per theme:')
    for theme in sorted(theme_fids.keys()):
        print(f'  {theme:<20}: {len(theme_fids[theme])}')
    print()

    # For each k, compute coverage per theme
    k_values = [1, 2, 4, 8, 16, 32]

    # Header
    header = f'{"Theme":<20}'
    for k in k_values:
        header += f' {"k="+str(k):>7}'
    print(header)
    print('-' * (20 + 8 * len(k_values)))

    theme_results = {}
    for theme in sorted(LICHESS_THEMES.keys()):
        if theme not in theme_fids:
            continue
        fids = theme_fids[theme]
        row = f'{theme:<20}'
        theme_results[theme] = {}

        for k_val in k_values:
            with torch.no_grad():
                _, acts = sae(normalized, k_override=k_val)
            fires = (acts > 0).numpy()

            # What % of positions have ANY feature from this theme firing?
            theme_fire_mask = np.zeros(10000, dtype=bool)
            for fid in fids:
                theme_fire_mask |= fires[:, fid].astype(bool)
            coverage = theme_fire_mask.sum() / 10000 * 100
            theme_results[theme][k_val] = coverage

            row += f' {coverage:>6.1f}%'

        print(row)

    # Summary: themes surviving at each k (>5% coverage)
    print()
    print('Themes with >5% coverage at each k:')
    for k_val in k_values:
        surviving = [t for t in theme_results if theme_results[t].get(k_val, 0) > 5]
        print(f'  k={k_val:>2}: {len(surviving)}/16 themes survive')
        dead = [t for t in theme_results if theme_results[t].get(k_val, 0) <= 5]
        if dead:
            print(f'        Lost: {", ".join(dead)}')

    # Also show: at k=8, which theme is STRONGEST per position?
    print()
    print('=== At k=8: Primary theme per position (by activation strength) ===')
    with torch.no_grad():
        _, acts_k8 = sae(normalized, k_override=8)
    acts_k8_np = acts_k8.numpy()

    primary_themes = []
    for i in range(10000):
        best_theme = 'none'
        best_strength = 0
        for theme, fids in theme_fids.items():
            strength = sum(acts_k8_np[i, fid] for fid in fids if fid < 2048)
            if strength > best_strength:
                best_strength = strength
                best_theme = theme
        primary_themes.append(best_theme)

    from collections import Counter
    primary_counts = Counter(primary_themes)
    print(f'{"Theme":<20} {"Primary %":>10}')
    for theme, count in primary_counts.most_common():
        print(f'  {theme:<20} {count/100:>9.1f}%')


if __name__ == '__main__':
    main()
