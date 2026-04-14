#!/usr/bin/env python3
"""Experiment 14: Do opening-specific features separate by opening type (e4 vs d4)?

Hypothesis: Opening-specific features contain sub-groups for different opening types.
Prediction: >20 features fire >60% on one opening type.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from collections import Counter


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


def get_opening_type(fen):
    """Classify by central pawn structure."""
    try:
        board = chess.Board(fen)
        # Find white pawns
        e_pawn = None
        d_pawn = None
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                if f == 4 and r >= 3:  # e-file, advanced (e4+)
                    e_pawn = r
                if f == 3 and r >= 3:  # d-file, advanced (d4+)
                    d_pawn = r
        if e_pawn and not d_pawn:
            return 'e4'
        if d_pawn and not e_pawn:
            return 'd4'
        if e_pawn and d_pawn:
            return 'e4d4'
        return 'other'
    except:
        return 'other'


def main():
    print('Experiment 14: Opening-type separation')
    print('Hypothesis: Opening-specific features separate by e4 vs d4')
    print('Prediction: >20 features fire >60% on one opening type')
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
    fires = (acts > 0).numpy().astype(np.float32)

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Classify positions by phase and opening type
    opening_idx = []
    opening_types = []
    for i, md in enumerate(metadata):
        try:
            board = chess.Board(md['fen'])
            if len(board.piece_map()) > 24:
                opening_idx.append(i)
                opening_types.append(get_opening_type(md['fen']))
        except:
            pass

    type_counts = Counter(opening_types)
    print('Opening positions: ' + str(len(opening_idx)))
    for t, n in type_counts.most_common():
        print('  ' + t + ': ' + str(n))
    print()

    # Find opening-specific features (>80% fire in opening phase)
    all_phases = []
    for md in metadata:
        try:
            n = len(chess.Board(md['fen']).piece_map())
        except:
            n = 20
        all_phases.append('opening' if n > 24 else ('middlegame' if n > 12 else 'endgame'))

    all_opening_idx = [i for i, p in enumerate(all_phases) if p == 'opening']

    opening_specific = []
    for fid in range(2048):
        total = fires[:, fid].sum()
        if total < 10:
            continue
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') not in ['high', 'medium']:
            continue
        ratio = fires[all_opening_idx, fid].sum() / total
        if ratio > 0.8:
            opening_specific.append(fid)

    print('Opening-specific features: ' + str(len(opening_specific)))
    print()

    # For each opening-specific feature: which opening type does it prefer?
    e4_specific = []
    d4_specific = []
    mixed = []

    type_to_idx = {'e4': [], 'd4': [], 'e4d4': [], 'other': []}
    for pos_idx, ot in zip(opening_idx, opening_types):
        type_to_idx[ot].append(pos_idx)

    for fid in opening_specific:
        type_fires = {}
        total_opening_fires = 0
        for ot in ['e4', 'd4', 'e4d4', 'other']:
            idx = type_to_idx[ot]
            type_fires[ot] = fires[idx, fid].sum() if idx else 0
            total_opening_fires += type_fires[ot]

        if total_opening_fires < 5:
            continue

        e4_ratio = type_fires['e4'] / total_opening_fires
        d4_ratio = type_fires['d4'] / total_opening_fires
        lbl = labels.get(str(fid), {}).get('label', '?')
        cat = labels.get(str(fid), {}).get('category', '?')

        if e4_ratio > 0.6:
            e4_specific.append((fid, e4_ratio, lbl, cat))
        elif d4_ratio > 0.6:
            d4_specific.append((fid, d4_ratio, lbl, cat))
        else:
            mixed.append((fid, e4_ratio, d4_ratio, lbl, cat))

    print('=== Results ===')
    print('e4-specific (>60% e4): ' + str(len(e4_specific)))
    for fid, r, lbl, cat in sorted(e4_specific, key=lambda x: -x[1])[:10]:
        print('  F' + str(fid) + ' (' + str(round(r * 100)) + '% e4) [' + cat + '] ' + lbl[:50])

    print()
    print('d4-specific (>60% d4): ' + str(len(d4_specific)))
    for fid, r, lbl, cat in sorted(d4_specific, key=lambda x: -x[1])[:10]:
        print('  F' + str(fid) + ' (' + str(round(r * 100)) + '% d4) [' + cat + '] ' + lbl[:50])

    print()
    print('Mixed (no dominant type): ' + str(len(mixed)))
    for fid, e4r, d4r, lbl, cat in mixed[:10]:
        print('  F' + str(fid) + ' (e4=' + str(round(e4r * 100)) + '% d4=' + str(round(d4r * 100)) + '%) [' + cat + '] ' + lbl[:50])

    # Verdict
    total_typed = len(e4_specific) + len(d4_specific)
    print()
    print('=== Verdict ===')
    print('Features with >60% opening-type specificity: ' + str(total_typed))
    print('Prediction was >20: ' + ('CONFIRMED' if total_typed > 20 else 'FAILED'))


if __name__ == '__main__':
    main()
