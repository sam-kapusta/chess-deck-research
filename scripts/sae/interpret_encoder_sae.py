#!/usr/bin/env python3
"""Interpret DeepMind encoder SAE features via Claude Haiku.

Run on SAIS notebook (needs GPU for encoder):
    pip install chess boto3
    python3 interpret_encoder_sae.py \
        --encoder /tmp/chess_encoder_270m.pt \
        --sae /home/ec2-user/SageMaker/chess-stage-a/output/encoder_sae_4096.pt \
        --data /home/ec2-user/SageMaker/chess-stage-a/data/lichess_evals_200k.jsonl \
        --output /home/ec2-user/SageMaker/encoder_4096_labels.json \
        --n-positions 5000 --n-features 500
"""
import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def board_summary(fen):
    try:
        b = chess.Board(fen)
        pieces = len([sq for sq in chess.SQUARES if b.piece_at(sq)])
        side = "W" if b.turn else "B"
        check = "+CHECK" if b.is_check() else ""
        w_pieces, b_pieces = [], []
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p and p.piece_type != chess.PAWN:
                name = chess.piece_name(p.piece_type).title()
                sq_name = chess.square_name(sq)
                if p.color:
                    w_pieces.append(f"{name[0]}{sq_name}")
                else:
                    b_pieces.append(f"{name[0]}{sq_name}")
        return f"{side}move {pieces}pc {check} W:[{','.join(w_pieces[:6])}] B:[{','.join(b_pieces[:6])}]"
    except:
        return fen[:40]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--sae', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-positions', type=int, default=5000)
    parser.add_argument('--n-features', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load encoder
    print(f"Loading encoder on {device}...", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(args.encoder), '..', 'scripts'))
    try:
        from fen_tokenizer import tokenize as chess_tokenize
        from chess_model import ChessEncoder
    except ImportError:
        # Try current directory
        from fen_tokenizer import tokenize as chess_tokenize
        from chess_model import ChessEncoder

    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).float()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    # Load SAE
    print("Loading SAE...", flush=True)
    sae_ckpt = torch.load(args.sae, map_location='cpu', weights_only=False)
    cfg = sae_ckpt['config']
    sae = BatchTopKSAE(cfg['input_dim'], cfg['dict_size'], cfg['k'])
    sae.load_state_dict(sae_ckpt['model_state_dict'])
    sae.eval()
    sae_mean = torch.tensor(sae_ckpt['normalization']['mean'])
    sae_std = torch.tensor(sae_ckpt['normalization']['std']).clamp(min=1e-6)

    # Extract activations + SAE features
    print(f"Processing {args.n_positions} positions...", flush=True)
    feature_fens = defaultdict(list)
    t0 = time.time()

    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n_positions:
                break
            item = json.loads(line.strip())
            fen = item.get('fen', '')
            if not fen:
                continue
            parts = fen.split()
            if len(parts) == 4:
                fen += ' 0 1'
            elif len(parts) == 5:
                fen += ' 1'

            tokens = torch.tensor(
                chess_tokenize(fen).astype(np.int64), dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                hidden = encoder(tokens)
                pooled = hidden.mean(dim=1).cpu()
                h_norm = (pooled - sae_mean) / sae_std
                _, z = sae(h_norm)

            z = z.squeeze(0)
            for fid in torch.topk(z, 5).indices.tolist():
                s = z[fid].item()
                if len(feature_fens[fid]) < 20:
                    feature_fens[fid].append((s, fen))

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{args.n_positions} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Done. {len(feature_fens)} features active.", flush=True)

    # Free GPU
    del encoder
    torch.cuda.empty_cache()

    # Interpret top features with Claude
    import boto3
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    sorted_features = sorted(feature_fens.items(), key=lambda x: -len(x[1]))[:args.n_features]
    print(f"\nInterpreting {len(sorted_features)} features with Claude...", flush=True)

    labels = {}
    errors = 0
    t0 = time.time()

    for idx, (fid, positions) in enumerate(sorted_features):
        positions.sort(key=lambda x: -x[0])
        summaries = [board_summary(fen) for _, fen in positions[:6]]

        prompt = f"""6 chess positions activate the same neural feature. Reply with ONLY a 3-8 word chess concept label. No explanation.

Examples: "knight outpost on e5", "rook on open d-file", "isolated queen pawn", "king safety compromised", "passed pawn on 6th rank", "back rank weakness"

Positions:
{chr(10).join(summaries)}

Label:"""

        try:
            response = bedrock.converse(
                modelId='us.anthropic.claude-haiku-4-5-20251001-v1:0',
                messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                inferenceConfig={'maxTokens': 20, 'temperature': 0}
            )
            label = response['output']['message']['content'][0]['text'].strip().strip('"').strip("'").strip('*')
            labels[str(fid)] = label
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(sorted_features)}] {time.time()-t0:.0f}s — F{fid}: {label}", flush=True)
        except Exception as e:
            errors += 1
            if errors > 10:
                print(f"Too many errors ({errors}), stopping", flush=True)
                break

    # Save
    out = {
        "source": f"DeepMind 270M Encoder SAE ({cfg['dict_size']} features, k={cfg['k']})",
        "method": "Claude Haiku interpretation of top-6 activating positions",
        "labels": labels,
        "n_positions_profiled": args.n_positions,
        "note": f"{len(labels)} features interpreted."
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{len(labels)} labels saved to {args.output} ({errors} errors, {time.time()-t0:.0f}s)")

    # Print samples
    for fid, label in list(labels.items())[:15]:
        print(f"  F{fid}: {label}")


if __name__ == '__main__':
    main()
