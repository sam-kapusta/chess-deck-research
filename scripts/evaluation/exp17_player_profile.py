#!/usr/bin/env python3
"""Experiment 17: Player-specific blunder patterns from SAE features.

Hypothesis: Sam's blunder features differ from the population baseline.
Prediction: >20 features fire >2x more often in Sam's games vs baseline.

Usage:
    python3 exp17_player_profile.py --games /path/to/games.json --cache /path/to/cache.pt \
        --checkpoint /path/to/sae.pt --labels /path/to/labels.json
"""
import argparse
import json
import sys

import chess
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


def encode_positions(fens, moves, encoder_path, device='cpu'):
    """Encode (FEN, move) pairs through the chess encoder, return hidden[77]."""
    # Load encoder
    import onnxruntime as ort

    with open(encoder_path.replace('.onnx', '').rsplit('/', 1)[0] + '/move_to_action.json') as f:
        move_to_action = json.load(f)

    sess = ort.InferenceSession(encoder_path)

    # Tokenize
    from pathlib import Path
    # Simple FEN tokenizer matching the encoder's expected input
    # The encoder expects token IDs — we need the same tokenizer used for training
    # For now, use the cache approach instead
    raise NotImplementedError("Direct encoding requires matching tokenizer — use cache approach")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, help='Blunder cache .pt file')
    parser.add_argument('--checkpoint', required=True, help='SAE checkpoint')
    parser.add_argument('--labels', required=True, help='Labels JSON')
    parser.add_argument('--player-games', help='Player games JSON (optional — uses metadata filter if available)')
    parser.add_argument('--player-username', default='cabbagelover5566', help='Player username to filter')
    parser.add_argument('--n-positions', type=int, default=50000)
    args = parser.parse_args()

    print('Experiment 17: Player-specific blunder patterns')
    print('Hypothesis: Sam\'s blunder features differ from population baseline')
    print('Prediction: >20 features fire >2x more in Sam\'s games vs baseline')
    print()

    # Load cache and SAE
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    n = min(args.n_positions, cache['blunder_mt'].shape[0])
    data = cache['blunder_mt'][:n].float()
    metadata = cache['metadata'][:n]

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dd = ckpt['config']['dict_size'] if 'config' in ckpt else 2048
    k = ckpt['config']['k'] if 'config' in ckpt else 32
    sae = SAE(1024, dd, k)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    fires = (acts > 0).numpy().astype(np.float32)
    acts_np = acts.numpy()

    with open(args.labels) as f:
        labels = json.load(f)

    print('Total positions: ' + str(n))

    # Try to identify player positions from metadata
    player_idx = []
    other_idx = []
    player_user = args.player_username.lower()

    for i, md in enumerate(metadata):
        # Check if this position is from the player's games
        # Metadata might have 'white', 'black', 'url', or 'player' fields
        is_player = False
        for key in ['white', 'black', 'player', 'username']:
            val = md.get(key, '')
            if isinstance(val, str) and player_user in val.lower():
                is_player = True
                break
        # Also check URL for chess.com username
        url = md.get('url', '')
        if player_user in url.lower():
            is_player = True

        if is_player:
            player_idx.append(i)
        else:
            other_idx.append(i)

    # If no player positions found in cache, try external games file
    if len(player_idx) == 0 and args.player_games:
        print('No player positions in cache. Loading external games...')
        # Would need to encode player games through encoder
        print('ERROR: External game encoding not yet implemented')
        print('The blunder cache contains Lichess positions, not Chess.com games')
        print('To test this, need to build a cache from Sam\'s games')
        sys.exit(1)

    if len(player_idx) == 0:
        print()
        print('No positions found for "' + args.player_username + '" in cache metadata')
        print('The blunder cache is from Lichess eval dataset — Sam plays on Chess.com')
        print()
        print('Falling back to: rating-based cohort analysis instead')
        print('Compare ~1800 rated players vs all players')
        print()

        # Rating-based analysis: group by rating bucket
        rating_buckets = {'<1200': [], '1200-1600': [], '1600-2000': [], '>2000': []}
        for i, md in enumerate(metadata):
            # Try to extract rating
            rating = None
            for key in ['white_elo', 'black_elo', 'rating', 'elo']:
                r = md.get(key)
                if r is not None:
                    try:
                        rating = int(r)
                        break
                    except (ValueError, TypeError):
                        pass

            if rating is None:
                continue

            if rating < 1200:
                rating_buckets['<1200'].append(i)
            elif rating < 1600:
                rating_buckets['1200-1600'].append(i)
            elif rating < 2000:
                rating_buckets['1600-2000'].append(i)
            else:
                rating_buckets['>2000'].append(i)

        print('Rating distribution:')
        total_rated = sum(len(v) for v in rating_buckets.values())
        for bucket, idx in rating_buckets.items():
            print('  ' + bucket + ': ' + str(len(idx)))

        if total_rated == 0:
            print('No rating info in metadata. Checking metadata fields...')
            sample = metadata[0] if len(metadata) > 0 else {}
            print('Sample metadata keys: ' + str(list(sample.keys())[:20]))
            print('Sample metadata: ' + json.dumps({k: str(v)[:50] for k, v in list(sample.items())[:10]}))
            print()
            print('Cannot segment by player or rating. Exiting.')
            sys.exit(1)

        # Compare 1600-2000 (Sam's range) vs all others
        target_idx = rating_buckets['1600-2000']
        baseline_idx = rating_buckets['<1200'] + rating_buckets['1200-1600'] + rating_buckets['>2000']

        if len(target_idx) < 100:
            print('Too few positions in Sam\'s rating range. Combining 1600+')
            target_idx = rating_buckets['1600-2000'] + rating_buckets['>2000']
            baseline_idx = rating_buckets['<1200'] + rating_buckets['1200-1600']

        print()
        print('Target group (1600-2000): ' + str(len(target_idx)) + ' positions')
        print('Baseline group: ' + str(len(baseline_idx)) + ' positions')
        player_idx = target_idx
        other_idx = baseline_idx

    if len(player_idx) < 50:
        print('Not enough player/target positions (' + str(len(player_idx)) + '). Need ≥50.')
        sys.exit(1)

    print('Player/target: ' + str(len(player_idx)) + ', Baseline: ' + str(len(other_idx)))
    print()

    # Compare fire rates
    player_fires = fires[player_idx]
    baseline_fires = fires[other_idx]

    player_rates = player_fires.mean(axis=0)
    baseline_rates = baseline_fires.mean(axis=0)

    # Compute strength differences too
    player_strengths = acts_np[player_idx].mean(axis=0)
    baseline_strengths = acts_np[other_idx].mean(axis=0)

    # Find features that are over/under-represented
    over_rep = []  # features firing more in player
    under_rep = []

    for fid in range(dd):
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') not in ['high', 'medium']:
            continue
        if baseline_rates[fid] < 0.005:  # too rare to compare
            continue

        ratio = player_rates[fid] / max(baseline_rates[fid], 1e-8)
        strength_ratio = player_strengths[fid] / max(baseline_strengths[fid], 1e-8)

        entry = (fid, ratio, strength_ratio, player_rates[fid], baseline_rates[fid],
                 lbl.get('label', '?')[:50], lbl.get('category', '?'))

        if ratio > 1.5:
            over_rep.append(entry)
        elif ratio < 0.67:
            under_rep.append(entry)

    over_rep.sort(key=lambda x: -x[1])
    under_rep.sort(key=lambda x: x[1])

    print('=== Over-represented features (>1.5x baseline) ===')
    print('Count: ' + str(len(over_rep)) + ' (>2x: ' + str(sum(1 for x in over_rep if x[1] > 2.0)) + ')')
    print()
    for fid, ratio, sr, pr, br, lbl, cat in over_rep[:20]:
        print('  F' + str(fid) + ' (' + str(round(ratio, 2)) + 'x fire, ' +
              str(round(sr, 2)) + 'x strength) [' + cat + '] ' + lbl)
        print('    Player=' + str(round(pr * 100, 1)) + '% Baseline=' + str(round(br * 100, 1)) + '%')

    print()
    print('=== Under-represented features (<0.67x baseline) ===')
    print('Count: ' + str(len(under_rep)) + ' (<0.5x: ' + str(sum(1 for x in under_rep if x[1] < 0.5)) + ')')
    print()
    for fid, ratio, sr, pr, br, lbl, cat in under_rep[:20]:
        print('  F' + str(fid) + ' (' + str(round(ratio, 2)) + 'x fire, ' +
              str(round(sr, 2)) + 'x strength) [' + cat + '] ' + lbl)
        print('    Player=' + str(round(pr * 100, 1)) + '% Baseline=' + str(round(br * 100, 1)) + '%')

    # Phase breakdown
    print()
    print('=== Phase breakdown of over-represented features ===')
    from collections import Counter
    phase_map = {}
    for i, md in enumerate(metadata):
        try:
            npieces = len(chess.Board(md['fen']).piece_map())
        except:
            npieces = 20
        phase_map[i] = 'opening' if npieces > 24 else ('middlegame' if npieces > 12 else 'endgame')

    over_phases = []
    for fid, ratio, sr, pr, br, lbl, cat in over_rep:
        total = fires[:, fid].sum()
        if total < 10:
            continue
        phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
        for i in range(n):
            phase_idx[phase_map[i]].append(i)
        ratios = {p: fires[idx, fid].sum() / total for p, idx in phase_idx.items()}
        dominant = max(ratios, key=ratios.get)
        over_phases.append(dominant)

    phase_counts = Counter(over_phases)
    for phase, count in phase_counts.most_common():
        print('  ' + phase + ': ' + str(count) + ' features')

    # Verdict
    count_2x = sum(1 for x in over_rep if x[1] > 2.0)
    print()
    print('=== Verdict ===')
    print('Features >2x over-represented: ' + str(count_2x))
    print('Prediction was >20: ' + ('CONFIRMED' if count_2x > 20 else 'FAILED'))


if __name__ == '__main__':
    main()
