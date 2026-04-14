#!/usr/bin/env python3
"""Experiment 29: Cross-SAE Jaccard between puzzle and blunder SAE.

Do the two SAEs fire on the same positions? If Jaccard is high, they're redundant.
If low, they capture different patterns and are complementary.

Method: Run both SAEs on the same 10K blunder positions. Compute cross-Jaccard.
For each blunder feature, find its best-match puzzle feature (and vice versa).
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'config' in ckpt:
        dd = ckpt['config']['dict_size']
        k = ckpt['config']['k']
    else:
        dd = ckpt['model_state_dict']['encoder.weight'].shape[0]
        k = 32
    sae = SAE(1024, dd, k)
    sae.load_state_dict(ckpt['model_state_dict'])
    if 'normalization' in ckpt:
        mean = torch.tensor(ckpt['normalization']['mean'], dtype=torch.float32)
        std = torch.tensor(ckpt['normalization']['std'], dtype=torch.float32) + 1e-8
    else:
        mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
        std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8
    return sae, mean, std, dd, k


def main():
    print('Experiment 29: Cross-SAE Jaccard (puzzle vs blunder)')
    print('Question: Do the two SAEs capture different patterns?')
    print()

    # Load blunder positions (move token activations)
    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()
    print('Positions: ' + str(data.shape[0]))

    # Load blunder SAE
    blunder_sae, b_mean, b_std, b_dd, b_k = load_sae(
        '/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt')
    print(f'Blunder SAE: {b_dd} features, k={b_k}')

    # Load puzzle SAE
    # Download puzzle SAE if not local
    import os, subprocess
    puzzle_path = '/home/ec2-user/SageMaker/chess-stage-a/sae_btk_2048_k64.pt'
    if not os.path.exists(puzzle_path):
        subprocess.run(['aws', 's3', 'cp', 's3://chess-stage-a-140023406996/sae-weights/sae_btk_2048_k64.pt',
                        puzzle_path], check=True)
    puzzle_sae, p_mean, p_std, p_dd, p_k = load_sae(puzzle_path)
    print(f'Puzzle SAE: {p_dd} features, k={p_k}')

    # Run both SAEs on same data
    with torch.no_grad():
        _, b_acts = blunder_sae((data - b_mean) / b_std)
        _, p_acts = puzzle_sae((data - p_mean) / p_std)

    b_fires = (b_acts > 0).numpy().astype(np.float32)
    p_fires = (p_acts > 0).numpy().astype(np.float32)

    # Filter to features with >=50 fires (Sandstone threshold)
    MIN_FIRES = 50
    b_active = [i for i in range(b_dd) if b_fires[:, i].sum() >= MIN_FIRES]
    p_active = [i for i in range(p_dd) if p_fires[:, i].sum() >= MIN_FIRES]
    print(f'\nActive features (>={MIN_FIRES} fires): blunder={len(b_active)}, puzzle={len(p_active)}')

    b_fp = b_fires[:, b_active]
    p_fp = p_fires[:, p_active]

    # Cross-Jaccard: blunder × puzzle
    print('\nComputing cross-Jaccard...')
    J = b_fp.T @ p_fp  # (n_blunder × n_puzzle) intersection
    b_sums = b_fp.sum(axis=0)  # (n_blunder,)
    p_sums = p_fp.sum(axis=0)  # (n_puzzle,)
    union = b_sums[:, None] + p_sums[None, :] - J
    union = np.maximum(union, 1)
    cross_jaccard = J / union

    # For each blunder feature, best match in puzzle
    best_match_b2p = cross_jaccard.max(axis=1)  # best puzzle match for each blunder feature
    best_match_p2b = cross_jaccard.max(axis=0)  # best blunder match for each puzzle feature

    print(f'\nBlunder→Puzzle best match:')
    print(f'  Mean: {best_match_b2p.mean():.4f}')
    print(f'  Median: {np.median(best_match_b2p):.4f}')
    print(f'  Max: {best_match_b2p.max():.4f}')
    for thresh in [0.3, 0.5, 0.8]:
        n = (best_match_b2p >= thresh).sum()
        print(f'  >={thresh}: {n} ({round(n/len(best_match_b2p)*100, 1)}%)')

    print(f'\nPuzzle→Blunder best match:')
    print(f'  Mean: {best_match_p2b.mean():.4f}')
    print(f'  Median: {np.median(best_match_p2b):.4f}')
    print(f'  Max: {best_match_p2b.max():.4f}')
    for thresh in [0.3, 0.5, 0.8]:
        n = (best_match_p2b >= thresh).sum()
        print(f'  >={thresh}: {n} ({round(n/len(best_match_p2b)*100, 1)}%)')

    # Load labels for the top cross-matches
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        b_labels = json.load(f)
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/k64_baseline/labels_sonnet_think.json') as f:
        p_labels = json.load(f)

    # Show highest cross-matches
    print('\n=== Highest cross-matches (most similar across SAEs) ===')
    top_pairs = []
    for i in range(len(b_active)):
        j = cross_jaccard[i].argmax()
        score = cross_jaccard[i, j]
        if score > 0.3:
            b_fid = b_active[i]
            p_fid = p_active[j]
            b_lbl = b_labels.get(str(b_fid), {}).get('label', '?')[:45]
            p_lbl = p_labels.get(str(p_fid), {}).get('label', '?')[:45]
            top_pairs.append((score, b_fid, p_fid, b_lbl, p_lbl))

    top_pairs.sort(reverse=True)
    for score, bf, pf, bl, pl in top_pairs[:15]:
        print(f'  J={score:.3f}: B.F{bf} "{bl}" <-> P.F{pf} "{pl}"')

    # Show features unique to each SAE (no match >0.3)
    b_unique = (best_match_b2p < 0.3).sum()
    p_unique = (best_match_p2b < 0.3).sum()
    print(f'\n=== Unique features (no cross-match >0.3) ===')
    print(f'  Blunder-only: {b_unique}/{len(b_active)} ({round(b_unique/len(b_active)*100, 1)}%)')
    print(f'  Puzzle-only: {p_unique}/{len(p_active)} ({round(p_unique/len(p_active)*100, 1)}%)')

    # Verdict
    overlap_80 = (best_match_b2p >= 0.8).sum()
    print(f'\n=== Verdict ===')
    print(f'Cross-SAE duplicates (Jaccard >=0.8): {overlap_80}')
    print(f'Mean best match: B→P={best_match_b2p.mean():.3f}, P→B={best_match_p2b.mean():.3f}')
    if best_match_b2p.mean() < 0.3:
        print('SAEs are COMPLEMENTARY — they capture different patterns')
    elif best_match_b2p.mean() < 0.5:
        print('SAEs have MODERATE overlap — some shared, some unique')
    else:
        print('SAEs are REDUNDANT — high overlap, one may suffice')


if __name__ == '__main__':
    main()
