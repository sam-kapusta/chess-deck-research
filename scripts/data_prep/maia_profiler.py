"""Profile Maia SAE features on Lichess positions.

Reuses Maia loading and concept computation from chess_sae_pipeline.py.
Outputs profiles in the same format as lichess_rich_profiler.py for
labeling with label_sae_features.py.

Usage (on chess-research notebook):
    python3 maia_profiler.py --sae output/maia_sae_2048_k32_v2.pt --output output/maia_profiles.json

    # Or with more positions:
    python3 maia_profiler.py --sae output/maia_sae_2048_k32_v2.pt --output output/maia_profiles.json --n-positions 100000
"""
import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Import from chess_sae_pipeline on the notebook
# Falls back to inline definitions if not available
try:
    sys.path.insert(0, '/home/ec2-user/SageMaker/chess-stage-a')
    from chess_sae_pipeline import load_sae, compute_concepts
    print('Loaded load_sae and compute_concepts from chess_sae_pipeline')
except ImportError:
    print('chess_sae_pipeline not found, using inline definitions')

    class BatchTopKSAE(nn.Module):
        def __init__(self, input_dim, dict_size, k):
            super().__init__()
            self.encoder = nn.Linear(input_dim, dict_size)
            self.decoder = nn.Linear(dict_size, input_dim, bias=False)
            self.pre_bias = nn.Parameter(torch.zeros(input_dim))
            self.k = k

        def forward(self, x):
            z = self.encoder(x - self.pre_bias)
            topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
            acts = torch.zeros_like(z)
            acts.scatter_(-1, topk_idx, F.relu(topk_vals))
            return self.decoder(acts) + self.pre_bias, acts

    def load_sae(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        sae = BatchTopKSAE(cfg['input_dim'], cfg['dict_size'], cfg['k'])
        sae.load_state_dict(ckpt['model_state_dict'])
        sae.eval()
        mean = torch.tensor(ckpt['normalization']['mean'])
        std = torch.tensor(ckpt['normalization']['std']).clamp(min=1e-6)
        return sae, mean, std, cfg

    def compute_concepts(fen):
        """Minimal concept extraction."""
        try:
            board = chess.Board(fen)
        except:
            return {}
        pc = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
        return {
            'opening': int(pc > 26),
            'middlegame': int(14 < pc <= 26),
            'endgame': int(pc <= 14),
            'in_check': int(board.is_check()),
            'piece_count': pc,
        }


DATA_PATH = '/home/ec2-user/SageMaker/chess-stage-a/data/lichess_evals_200k.jsonl'


def setup_maia(rating=1900):
    """Load Maia-2 and return (model, hook_store, preprocessing_fn)."""
    from maia2 import model as maia_model, inference as maia_inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading Maia-2 on ' + device + '...')
    model = maia_model.from_pretrained(type='rapid', device=device)
    prepared = maia_inference.prepare()
    all_moves_dict, elo_dict, _ = prepared

    hiddens = []
    def hook_fn(module, input, output):
        hiddens.append(output.detach())
    model.last_ln.register_forward_hook(hook_fn)

    def get_hidden(fen):
        bi, es, eo, _ = maia_inference.preprocessing(fen, rating, rating, elo_dict, all_moves_dict)
        hiddens.clear()
        with torch.no_grad():
            model(bi.unsqueeze(0).to(device), torch.tensor([es]).to(device), torch.tensor([eo]).to(device))
        if hiddens:
            return hiddens[0].squeeze(0).cpu().float()
        return None

    return get_hidden


def get_features(hidden, sae, mean, std):
    """Run SAE on a Maia hidden state, return set of active feature IDs."""
    h_norm = (hidden - mean) / std
    with torch.no_grad():
        _, acts = sae(h_norm.unsqueeze(0))
    return set(int(f) for f in np.where(acts.squeeze(0).cpu().numpy() > 0)[0])


def enrich_position(fen, cp=None):
    """Create a rich example string from a FEN."""
    try:
        board = chess.Board(fen)
        pc = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
        phase = 'opening' if pc > 26 else ('middlegame' if pc > 16 else 'endgame')
        in_check = '+' if board.is_check() else ''
        turn = 'white' if board.turn else 'black'
        eval_str = ', eval=' + str(cp) + 'cp' if cp is not None else ''
        return fen + ' (' + turn + ' to move, ' + phase + in_check + eval_str + ')'
    except:
        return fen


def profile_features(sae_path, output_path, n_positions=50000):
    # Load SAE
    sae, sae_mean, sae_std, cfg = load_sae(sae_path)
    sae = sae.cuda().eval()
    sae_mean = sae_mean.cuda()
    sae_std = sae_std.cuda()
    DICT_SIZE = cfg['dict_size']
    print('SAE loaded: ' + str(DICT_SIZE) + ' features, k=' + str(cfg['k']))

    # Load Maia
    get_hidden = setup_maia()
    print('Maia loaded.')

    # Stream positions and profile
    fire_count = np.zeros(DICT_SIZE)
    feature_examples = defaultdict(list)
    feature_concepts = defaultdict(lambda: defaultdict(float))
    n_processed = 0
    t0 = time.time()

    with open(DATA_PATH) as f:
        for line in f:
            if n_processed >= n_positions:
                break

            item = json.loads(line.strip())
            fen = item.get('fen', '')
            cp = item.get('cp')
            if not fen:
                continue

            try:
                hidden = get_hidden(fen)
                if hidden is None:
                    continue
                feats = get_features(hidden.cuda(), sae, sae_mean, sae_std)
            except:
                continue

            # Record features
            for fid in feats:
                fire_count[fid] += 1
                if len(feature_examples[fid]) < 50:
                    feature_examples[fid].append(enrich_position(fen, cp))

            # Concept enrichment
            concepts = compute_concepts(fen)
            for fid in feats:
                for cname, cval in concepts.items():
                    feature_concepts[fid][cname] += cval

            n_processed += 1
            if n_processed % 5000 == 0:
                elapsed = time.time() - t0
                print('  ' + str(n_processed) + '/' + str(n_positions) +
                      ' ({:.0f}/sec)'.format(n_processed / elapsed))
                sys.stdout.flush()

    alive = int((fire_count > 0).sum())
    profiled = sum(1 for f in fire_count if f >= 20)
    print('Done: ' + str(n_processed) + ' positions, ' + str(alive) + ' alive, ' + str(profiled) + ' profiled')

    # Build profiles (same format as lichess_rich_profiler output)
    profiles = {}
    for fid in range(DICT_SIZE):
        if fire_count[fid] < 20:
            continue

        n = int(fire_count[fid])
        profile = {
            'fire_rate': round(100 * n / n_processed, 2),
            'n_fires': n,
            'examples': feature_examples.get(fid, [])[:10],
        }

        # Add concept stats (only notable ones)
        fc = feature_concepts.get(fid, {})
        base_rate = n / n_processed
        for cname, cval in fc.items():
            feat_rate = cval / n if n > 0 else 0
            if isinstance(feat_rate, (int, float)) and feat_rate > 0:
                # Only include if notably different from uniform
                profile['concept_' + cname] = round(feat_rate, 3)

        profiles[str(fid)] = profile

    with open(output_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(str(len(profiles)) + ' profiles saved to ' + output_path)
    return profiles


def main():
    parser = argparse.ArgumentParser(description='Profile Maia SAE features on Lichess positions')
    parser.add_argument('--sae', required=True, help='Path to Maia SAE checkpoint')
    parser.add_argument('--output', required=True, help='Path to write profiles JSON')
    parser.add_argument('--n-positions', type=int, default=50000, help='Number of positions to process')
    args = parser.parse_args()

    profile_features(args.sae, args.output, args.n_positions)
    print('DONE')


if __name__ == '__main__':
    main()
