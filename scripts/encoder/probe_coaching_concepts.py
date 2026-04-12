#!/usr/bin/env python3
"""Probe the encoder for coaching-relevant concepts beyond the initial 3.

New concepts to probe:
1. Defensive necessity (is the side to move under pressure?)
2. Simplification window (would trading pieces lock in advantage?)
3. King exposure differential (is one king much safer?)
4. Pawn break available (should a pawn advance to open lines?)
5. Positional squeeze (does one side have a space advantage?)

Labels derived from training data (Stockfish evals, tags, position field).

Usage:
  python probe_coaching_concepts.py --num 1000
"""
import sys
import json
import argparse
import torch
import numpy as np
import chess
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder')
sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder/scripts')

from searchless_chess.src.tokenizer import tokenize as chess_tokenize
from convert_and_validate_v2 import ChessEncoder


def encode_all(encoder, fens, device, batch_size=32):
    all_h = []
    for i in range(0, len(fens), batch_size):
        batch = fens[i:i+batch_size]
        tokens = torch.stack([
            torch.tensor(chess_tokenize(f).astype(np.int64), dtype=torch.long) for f in batch
        ]).to(device)
        with torch.no_grad():
            h = encoder(tokens).mean(dim=1).cpu().float().numpy()
        all_h.append(h)
    return np.concatenate(all_h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--data', default='/home/ec2-user/SageMaker/chess-research/data/coaching_training_data.jsonl')
    parser.add_argument('--save-probes', default='/home/ec2-user/SageMaker/chess-research/checkpoints/coaching_probes.pkl')
    args = parser.parse_args()

    device = torch.device('cuda')
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    items = [json.loads(l) for l in Path(args.data).read_text().strip().split('\n')]
    items = [d for d in items if d.get('eval') is not None][:args.num]
    fens = [d['fen'] for d in items]

    print(f"Encoding {len(items)} positions...")
    X = encode_all(encoder, fens, device)

    probes = {}

    # 1. Defensive necessity: eval is negative and getting worse
    print("\n--- Defensive Necessity ---")
    y_def = []
    for item in items:
        ev = item.get('eval', 0)
        # Side to move is under pressure if eval is bad for them
        board = chess.Board(item['fen'])
        player_eval = ev if board.turn == chess.WHITE else -ev
        y_def.append(1 if player_eval < -1.0 else 0)
    y_def = np.array(y_def)
    if y_def.sum() > 10 and (1-y_def).sum() > 10:
        clf = LogisticRegression(max_iter=1000, C=0.1)
        scores = cross_val_score(clf, X, y_def, cv=5, scoring='accuracy')
        baseline = max(y_def.mean(), 1-y_def.mean())
        print(f"  Acc: {scores.mean():.3f}, Baseline: {baseline:.3f}, Lift: {scores.mean()-baseline:+.3f}")
        probes['defensive'] = LogisticRegression(max_iter=1000, C=0.1).fit(X, y_def)

    # 2. Simplification opportunity: winning + captures available
    print("\n--- Simplification Window ---")
    y_simp = []
    for item in items:
        ev = item.get('eval', 0)
        board = chess.Board(item['fen'])
        player_eval = ev if board.turn == chess.WHITE else -ev
        has_captures = any(board.is_capture(m) for m in board.legal_moves)
        y_simp.append(1 if player_eval > 2.0 and has_captures else 0)
    y_simp = np.array(y_simp)
    if y_simp.sum() > 10:
        clf = LogisticRegression(max_iter=1000, C=0.1)
        scores = cross_val_score(clf, X, y_simp, cv=5, scoring='accuracy')
        baseline = max(y_simp.mean(), 1-y_simp.mean())
        print(f"  Acc: {scores.mean():.3f}, Baseline: {baseline:.3f}, Lift: {scores.mean()-baseline:+.3f}")
        probes['simplification'] = LogisticRegression(max_iter=1000, C=0.1).fit(X, y_simp)

    # 3. King exposure differential
    print("\n--- King Exposure ---")
    y_king = []
    for item in items:
        board = chess.Board(item['fen'])
        w_king = board.king(chess.WHITE)
        b_king = board.king(chess.BLACK)
        if w_king is None or b_king is None:
            y_king.append(0); continue
        w_rank = chess.square_rank(w_king)
        b_rank = chess.square_rank(b_king)
        # Exposed = not on home rank
        w_exposed = 1 if w_rank > 1 else 0
        b_exposed = 1 if b_rank < 6 else 0
        y_king.append(w_exposed + b_exposed)  # 0=both safe, 1=one exposed, 2=both exposed
    y_king = np.array(y_king)
    ridge = Ridge(alpha=10.0)
    scores = cross_val_score(ridge, X, y_king, cv=5, scoring='r2')
    ridge.fit(X, y_king)
    r, _ = pearsonr(ridge.predict(X), y_king)
    print(f"  R²: {scores.mean():.3f}, r: {r:.3f}")
    probes['king_exposure'] = Ridge(alpha=10.0).fit(X, y_king)

    # 4. Pawn break available (tag-based)
    print("\n--- Pawn Break ---")
    y_pb = np.array([1 if 'missed_pawn_break' in item.get('tags', []) else 0 for item in items])
    if y_pb.sum() > 10:
        clf = LogisticRegression(max_iter=1000, C=0.1)
        scores = cross_val_score(clf, X, y_pb, cv=5, scoring='accuracy')
        baseline = max(y_pb.mean(), 1-y_pb.mean())
        print(f"  Acc: {scores.mean():.3f}, Baseline: {baseline:.3f}, Lift: {scores.mean()-baseline:+.3f}")
        probes['pawn_break'] = LogisticRegression(max_iter=1000, C=0.1).fit(X, y_pb)

    # 5. Positional squeeze (few legal moves for opponent)
    print("\n--- Positional Squeeze ---")
    y_squeeze = []
    for item in items:
        board = chess.Board(item['fen'])
        # Flip to opponent's perspective
        board_flipped = board.copy()
        board_flipped.push(chess.Move.null()) if board.is_legal(chess.Move.null()) else None
        n_moves = len(list(board.legal_moves))
        y_squeeze.append(1 if n_moves < 15 else 0)  # Cramped = few moves
    y_squeeze = np.array(y_squeeze)
    if y_squeeze.sum() > 10:
        clf = LogisticRegression(max_iter=1000, C=0.1)
        scores = cross_val_score(clf, X, y_squeeze, cv=5, scoring='accuracy')
        baseline = max(y_squeeze.mean(), 1-y_squeeze.mean())
        print(f"  Acc: {scores.mean():.3f}, Baseline: {baseline:.3f}, Lift: {scores.mean()-baseline:+.3f}")
        probes['squeeze'] = LogisticRegression(max_iter=1000, C=0.1).fit(X, y_squeeze)

    # Save all probes
    import pickle
    Path(args.save_probes).write_bytes(pickle.dumps(probes))
    print(f"\nSaved {len(probes)} probes to {args.save_probes}")


if __name__ == "__main__":
    main()
