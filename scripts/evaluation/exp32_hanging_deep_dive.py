#!/usr/bin/env python3
"""Experiment 32: Deep dive into "Hanging Pieces" — what's actually going on?

Questions:
1. Per position, how many hanging-piece features fire simultaneously?
2. Does k=32 produce redundant hanging signals? Would k=16 or k=8 be cleaner?
3. What are the actual subtypes within "hanging pieces"?
4. Per game estimate: how often does this theme appear?

Also test: what happens at k=16, k=8, k=4, k=1? Do we lose the hanging piece signal
or does it concentrate into fewer, cleaner features?
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
    def forward(self, x, k_override=None):
        k = k_override or self.k
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, k, dim=-1)
        a = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


HANGING_KW = ['hanging', 'undefended', 'unprotected', 'en prise',
              'inadequately defended', 'loose piece']


def is_hanging(label, explanation):
    text = (label + ' ' + explanation).lower()
    return any(re.search(kw, text) for kw in HANGING_KW)


def main():
    print('Experiment 32: Hanging Pieces deep dive')
    print()

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

    normalized = (data - mean) / std

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Identify hanging-piece feature IDs
    hanging_fids = set()
    for fid_str, info in labels.items():
        if info.get('confidence') not in ['high', 'medium']:
            continue
        if int(fid_str) >= 2048:
            continue
        if is_hanging(info.get('label', ''), info.get('explanation', '')):
            hanging_fids.add(int(fid_str))

    print(f'Hanging-piece features: {len(hanging_fids)}')

    # === Q1: At k=32, how many hanging features fire per position? ===
    print()
    print('=== Q1: Hanging features per position at k=32 ===')
    with torch.no_grad():
        _, acts_k32 = sae(normalized)
    fires_k32 = (acts_k32 > 0).numpy()

    hanging_per_pos = []
    total_per_pos = []
    for i in range(10000):
        active = set(np.where(fires_k32[i])[0])
        n_hanging = len(active & hanging_fids)
        hanging_per_pos.append(n_hanging)
        total_per_pos.append(len(active))

    hanging_per_pos = np.array(hanging_per_pos)
    print(f'  Mean hanging features per position: {hanging_per_pos.mean():.1f} / {np.mean(total_per_pos):.1f} total')
    print(f'  Median: {np.median(hanging_per_pos):.0f}')
    print(f'  Positions with 0 hanging: {(hanging_per_pos == 0).sum()} ({(hanging_per_pos == 0).mean()*100:.1f}%)')
    print(f'  Positions with 1-3 hanging: {((hanging_per_pos >= 1) & (hanging_per_pos <= 3)).sum()}')
    print(f'  Positions with 4-10 hanging: {((hanging_per_pos >= 4) & (hanging_per_pos <= 10)).sum()}')
    print(f'  Positions with 10+ hanging: {(hanging_per_pos > 10).sum()}')

    # Distribution
    print()
    print('  Distribution:')
    for n in range(16):
        count = (hanging_per_pos == n).sum()
        if count > 0:
            bar = '#' * (count // 50)
            print(f'    {n:>2} hanging: {count:>5} positions {bar}')

    # === Q2: What happens at lower k? ===
    print()
    print('=== Q2: Hanging features at different k values ===')
    print(f'  {"k":>4} {"Total active":>12} {"Hanging fire":>12} {"Hanging/pos":>12} {"% positions":>12}')

    for k_val in [1, 2, 4, 8, 16, 32]:
        with torch.no_grad():
            _, acts_k = sae(normalized, k_override=k_val)
        fires_k = (acts_k > 0).numpy()

        # Count hanging per position
        h_counts = []
        for i in range(10000):
            active = set(np.where(fires_k[i])[0])
            h_counts.append(len(active & hanging_fids))

        h_counts = np.array(h_counts)
        n_with_hanging = (h_counts > 0).sum()
        pct = n_with_hanging / 10000 * 100

        print(f'  k={k_val:>2}: {fires_k.sum(axis=1).mean():>10.1f}   {h_counts.sum():>10}   {h_counts.mean():>10.1f}   {pct:>10.1f}%')

    # === Q3: At k=1, what IS the single top feature per position? ===
    print()
    print('=== Q3: What are the top-1 features? (k=1 equivalent) ===')

    # Get the pre-topk activations
    with torch.no_grad():
        z = sae.encoder(normalized - sae.pre_bias)
    z_np = z.numpy()

    # For each position, what's the #1 feature?
    top1_fids = z_np.argmax(axis=1)
    top1_is_hanging = np.array([f in hanging_fids for f in top1_fids])

    print(f'  Positions where #1 feature is "hanging": {top1_is_hanging.sum()} ({top1_is_hanging.mean()*100:.1f}%)')

    # What are the most common top-1 features?
    top1_counts = Counter(top1_fids)
    print()
    print('  Most common top-1 features:')
    for fid, count in top1_counts.most_common(15):
        lbl = labels.get(str(fid), {}).get('label', '?')[:50]
        is_h = '🎯' if fid in hanging_fids else '  '
        print(f'    {is_h} F{fid} ({count} positions): {lbl}')

    # === Q4: Among the 678 hanging features, which ones fire MOST? ===
    print()
    print('=== Q4: Top hanging features by fire rate ===')
    fires_k32_np = fires_k32.astype(np.float32)
    hanging_list = sorted(hanging_fids)
    fire_rates = [(fid, fires_k32_np[:, fid].sum()) for fid in hanging_list]
    fire_rates.sort(key=lambda x: -x[1])

    print(f'  Top 20 (of {len(hanging_list)}):')
    cumulative_coverage = set()
    for fid, n_fires in fire_rates[:20]:
        lbl = labels.get(str(fid), {}).get('label', '?')[:50]
        # Positions this feature covers
        positions = set(np.where(fires_k32_np[:, fid])[0])
        new_positions = positions - cumulative_coverage
        cumulative_coverage |= positions
        print(f'    F{fid}: {int(n_fires)} fires, +{len(new_positions)} new positions, cumul={len(cumulative_coverage)}: {lbl}')

    print(f'\n  Top 20 hanging features cover {len(cumulative_coverage)}/10000 positions ({len(cumulative_coverage)/100:.1f}%)')
    print(f'  Top 50:')
    for fid, n_fires in fire_rates[:50]:
        positions = set(np.where(fires_k32_np[:, fid])[0])
        cumulative_coverage |= positions
    print(f'    Cover {len(cumulative_coverage)}/10000 ({len(cumulative_coverage)/100:.1f}%)')

    # === Q5: Label diversity — what are the ACTUAL subtypes? ===
    print()
    print('=== Q5: Hanging piece subtypes (from top-firing features) ===')

    # Get the top 50 by fire rate, look at their labels more carefully
    subtype_labels = []
    for fid, n_fires in fire_rates[:100]:
        info = labels.get(str(fid), {})
        lbl = info.get('label', '')
        expl = info.get('explanation', '')[:150]
        subtype_labels.append((fid, n_fires, lbl, expl))

    # Try to extract subtypes from explanations
    print('  Scanning explanations for piece-specific or context-specific patterns...')
    piece_specific = defaultdict(list)
    context_specific = defaultdict(list)
    for fid, nf, lbl, expl in subtype_labels:
        text = (lbl + ' ' + expl).lower()
        # Piece types
        for piece in ['queen', 'rook', 'bishop', 'knight', 'pawn']:
            if piece in text:
                piece_specific[piece].append((fid, lbl[:50]))
        # Contexts
        if 'endgame' in text:
            context_specific['endgame'].append((fid, lbl[:50]))
        elif 'opening' in text:
            context_specific['opening'].append((fid, lbl[:50]))
        if 'check' in text:
            context_specific['with_check'].append((fid, lbl[:50]))
        if 'capture' in text and 'recapture' not in text:
            context_specific['capture'].append((fid, lbl[:50]))

    print()
    print('  By piece type:')
    for piece, feats in sorted(piece_specific.items(), key=lambda x: -len(x[1])):
        print(f'    {piece}: {len(feats)} features')
        for fid, lbl in feats[:2]:
            print(f'      F{fid}: {lbl}')

    print()
    print('  By context:')
    for ctx, feats in sorted(context_specific.items(), key=lambda x: -len(x[1])):
        print(f'    {ctx}: {len(feats)} features')
        for fid, lbl in feats[:2]:
            print(f'      F{fid}: {lbl}')


if __name__ == '__main__':
    main()
