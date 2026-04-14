#!/usr/bin/env python3
"""Experiment 31: Position coverage per Lichess theme (not feature count).

The question: do "small" themes (33-41 features) cover enough positions to be
useful for player profiling? Feature count ≠ position coverage.

Method: For each Lichess theme, count what % of blunder positions have ANY
feature from that theme firing. Also compute mean strength when firing.
"""
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict


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


LICHESS_THEMES = {
    'fork': ['fork', 'double attack', 'two targets', 'simultaneous.*attack', 'attacks.*two'],
    'pin': ['pin', 'pinned', 'pinning'],
    'skewer': ['skewer', 'x-ray', 'xray'],
    'discovered_attack': ['discovered', 'discovery', 'uncovered', 'battery'],
    'hanging_piece': ['hanging', 'undefended', 'unprotected', 'en prise',
                      'inadequately defended', 'loose piece'],
    'deflection': ['overloaded', 'overworked', 'deflect', 'removing.*guard',
                   'removing.*defender', 'capture.*defender'],
    'back_rank': ['back rank', 'back-rank', 'backrank'],
    'trapped_piece': ['trapped', 'no escape', 'boxed in', 'restricted'],
    'quiet_move': ['quiet', 'prophyla', 'zwischenzug', 'intermezzo', 'in-between'],
    'sacrifice': ['sacrifice', 'sac '],
    'exposed_king': ['exposed king', 'king.*safe', 'weak.*king', 'king.*danger',
                     'king.*vulnerable', 'pawn.*shelter', 'pawn.*shield'],
    'promotion': ['promot', 'queening', 'advanced pawn'],
    'passed_pawn': ['passed pawn', 'passed.*pawn'],
    'rook_endgame': ['rook.*endgame', 'rook.*ending', 'rook.*pawn.*versus'],
    'pawn_endgame': ['pawn.*endgame', 'pawn.*ending', 'king.*pawn.*endgame'],
    'checkmate': ['checkmate', 'mating', 'mate in', 'forced mate'],
}


def main():
    print('Experiment 31: Position coverage per Lichess theme')
    print('Question: Do "small" themes cover enough positions for profiling?')
    print()

    # Load SAE + data
    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()

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

    n_positions = fires.shape[0]
    print(f'Positions: {n_positions}')
    print()

    # Map features to themes
    theme_fids = defaultdict(list)
    for fid_str, info in labels.items():
        if info.get('confidence') not in ['high', 'medium']:
            continue
        fid = int(fid_str)
        if fid >= fires.shape[1]:
            continue
        text = (info.get('label', '') + ' ' + info.get('explanation', '')).lower()
        for theme, keywords in LICHESS_THEMES.items():
            if any(re.search(kw, text) for kw in keywords):
                theme_fids[theme].append(fid)

    # Per theme: position coverage + strength
    print(f'{"Theme":<20} {"Features":>8} {"Positions":>10} {"Coverage":>8} {"Mean Str":>9} {"Per Game*":>9}')
    print('-' * 70)

    results = []
    for theme in sorted(LICHESS_THEMES.keys()):
        fids = theme_fids[theme]
        n_features = len(fids)
        if n_features == 0:
            continue

        # Position coverage: any feature in this theme fires on this position
        theme_fires = fires[:, fids].max(axis=1)  # 1 if ANY theme feature fires
        n_positions_covered = (theme_fires > 0).sum()
        coverage = n_positions_covered / n_positions * 100

        # Mean activation strength when firing
        theme_acts = acts_np[:, fids]
        firing_vals = theme_acts[theme_acts > 0]
        mean_strength = firing_vals.mean() if len(firing_vals) > 0 else 0

        # Estimated per-game rate (10K positions ≈ 1667 games at 6 blunders/game)
        est_games = n_positions / 6
        per_game = n_positions_covered / est_games

        results.append((theme, n_features, n_positions_covered, coverage, mean_strength, per_game))
        print(f'{theme:<20} {n_features:>8} {int(n_positions_covered):>10} {coverage:>7.1f}% {mean_strength:>9.2f} {per_game:>8.1f}')

    print()
    print('* Per Game estimated assuming ~6 blunders/game from 10K positions')

    # The key question: are "small" themes still useful?
    print()
    print('=== Small themes (< 50 features) — are they useful for profiling? ===')
    for theme, n_feat, n_pos, cov, strength, pg in results:
        if n_feat < 50:
            useful = 'YES' if cov > 1.0 else 'MAYBE' if cov > 0.5 else 'NO'
            print(f'  {theme:<20}: {n_feat} features, covers {cov:.1f}% of positions, ~{pg:.1f}/game → {useful}')

    # Compare: feature count vs position coverage ranking
    print()
    print('=== Ranking comparison: feature count vs position coverage ===')
    by_features = sorted(results, key=lambda x: -x[1])
    by_coverage = sorted(results, key=lambda x: -x[2])

    print(f'{"Rank":<5} {"By Features":<25} {"By Coverage":<25} {"Same?"}')
    for i in range(min(len(by_features), len(by_coverage))):
        f_theme = by_features[i][0]
        c_theme = by_coverage[i][0]
        same = '✓' if f_theme == c_theme else ''
        print(f'{i+1:<5} {f_theme:<25} {c_theme:<25} {same}')


if __name__ == '__main__':
    main()
