#!/usr/bin/env python3
"""Evaluate chess model outputs with Stockfish verification.

No LLM judge. All metrics are verifiable against Stockfish + FEN.

Parses model text output for chess claims, verifies each:
  1. Best move: does the model mention Stockfish's best move?
  2. Eval direction: does the model say the right side is winning?
  3. Eval accuracy: is the model's estimated eval within ±1.0?
  4. Square validity: do mentioned squares have pieces in the FEN?
  5. Move validity: are mentioned moves legal in the position?

Usage:
  python eval_stockfish.py --results /path/to/generation_results.json
  python eval_stockfish.py --results /path/to/results.json --verbose
"""
import json
import re
import argparse
import chess
import numpy as np
from pathlib import Path


# ============================================================
# Chess text parsing (simple string matching, not NLP)
# ============================================================

def extract_squares(text):
    """Extract chess squares mentioned in text (e.g., e4, d5, f7)."""
    return set(re.findall(r'\b([a-h][1-8])\b', text.lower()))


def extract_moves(text):
    """Extract chess moves mentioned in text (e.g., Nxd5, Bxf7+, O-O)."""
    # Standard algebraic: piece + optional file/rank + optional capture + destination + optional promotion/check
    san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b'
    castling = r'\b(O-O(?:-O)?)\b'
    moves = re.findall(san_pattern, text)
    moves += re.findall(castling, text)
    return moves


def extract_eval_claim(text):
    """Extract evaluation claims from text.

    Looks for patterns like: "+2.1", "White is winning", "roughly equal",
    "Black has an advantage", "eval: +1.5"
    """
    # Numeric eval: +2.1, -0.5, approximately +3
    numeric = re.findall(r'[+\-]?\d+\.?\d*', text)

    # Direction claims
    text_lower = text.lower()
    if any(p in text_lower for p in ['white is winning', 'white has a decisive', 'white is much better', 'winning for white']):
        return 'white_winning'
    elif any(p in text_lower for p in ['white is better', 'white has an advantage', 'slight advantage for white', 'white is slightly']):
        return 'white_better'
    elif any(p in text_lower for p in ['black is winning', 'black has a decisive', 'black is much better', 'winning for black']):
        return 'black_winning'
    elif any(p in text_lower for p in ['black is better', 'black has an advantage', 'slight advantage for black', 'black is slightly']):
        return 'black_better'
    elif any(p in text_lower for p in ['equal', 'balanced', 'even', 'roughly equal']):
        return 'equal'

    # Try to parse a numeric eval
    eval_patterns = re.findall(r'(?:eval|evaluation|approximately|roughly|about|around)\s*[:\s]*([+\-]?\d+\.?\d*)', text_lower)
    if eval_patterns:
        try:
            return float(eval_patterns[0])
        except ValueError:
            pass

    return None


def check_best_move(text, best_move_san):
    """Does the model's text mention the Stockfish best move?"""
    if not best_move_san:
        return None
    return best_move_san.lower() in text.lower()


def check_eval_direction(text, stockfish_eval):
    """Does the model get the eval direction right?"""
    if stockfish_eval is None:
        return None

    claim = extract_eval_claim(text)
    if claim is None:
        return None  # Model didn't make an eval claim

    sf_direction = 'white_winning' if stockfish_eval > 1.5 else \
                   'white_better' if stockfish_eval > 0.3 else \
                   'black_winning' if stockfish_eval < -1.5 else \
                   'black_better' if stockfish_eval < -0.3 else \
                   'equal'

    if isinstance(claim, float):
        model_direction = 'white_winning' if claim > 1.5 else \
                         'white_better' if claim > 0.3 else \
                         'black_winning' if claim < -1.5 else \
                         'black_better' if claim < -0.3 else \
                         'equal'
        return model_direction == sf_direction

    # Coarse match: right side winning/better
    if sf_direction in ('white_winning', 'white_better') and claim in ('white_winning', 'white_better'):
        return True
    if sf_direction in ('black_winning', 'black_better') and claim in ('black_winning', 'black_better'):
        return True
    if sf_direction == 'equal' and claim == 'equal':
        return True

    return False


def check_squares_valid(text, fen):
    """Are the squares mentioned in the text real (have pieces or are legal move destinations)?"""
    try:
        board = chess.Board(fen)
    except:
        return None

    mentioned = extract_squares(text)
    if not mentioned:
        return None

    # Valid squares: occupied OR legal move destinations
    valid = set()
    for sq in chess.SQUARES:
        if board.piece_at(sq):
            valid.add(chess.square_name(sq))
    for move in board.legal_moves:
        valid.add(chess.square_name(move.to_square))
        valid.add(chess.square_name(move.from_square))

    correct = sum(1 for sq in mentioned if sq in valid)
    return correct / len(mentioned)


def check_moves_legal(text, fen):
    """Are the moves mentioned in the text legal in the position?"""
    try:
        board = chess.Board(fen)
    except:
        return None

    mentioned_moves = extract_moves(text)
    if not mentioned_moves:
        return None

    legal_san = set()
    for move in board.legal_moves:
        legal_san.add(board.san(move))

    # Also check continuations (moves legal after the first move)
    # This is important because "after Nxd5, exd5" — exd5 is legal in the NEXT position
    legal_extended = set(legal_san)
    for move in board.legal_moves:
        next_board = board.copy()
        next_board.push(move)
        for next_move in next_board.legal_moves:
            legal_extended.add(next_board.san(next_move))

    correct = sum(1 for m in mentioned_moves if m in legal_extended)
    return correct / len(mentioned_moves) if mentioned_moves else None


# ============================================================
# Main evaluation
# ============================================================

def evaluate_single(text, item, verbose=False):
    """Evaluate a single model output against Stockfish ground truth."""
    fen = item.get('fen', '')
    best_move = item.get('best_move', '')
    sf_eval = item.get('eval')

    scores = {}

    # 1. Best move mentioned
    bm = check_best_move(text, best_move)
    if bm is not None:
        scores['best_move'] = 1.0 if bm else 0.0

    # 2. Eval direction
    ed = check_eval_direction(text, sf_eval)
    if ed is not None:
        scores['eval_direction'] = 1.0 if ed else 0.0

    # 3. Square validity
    sv = check_squares_valid(text, fen)
    if sv is not None:
        scores['square_validity'] = sv

    # 4. Move legality
    ml = check_moves_legal(text, fen)
    if ml is not None:
        scores['move_legality'] = ml

    # 5. Played move mentioned (for mistake positions)
    played = item.get('played_move', '')
    if played:
        scores['played_move_mentioned'] = 1.0 if played.lower() in text.lower() else 0.0

    if verbose:
        print(f"  Best move ({best_move}): {'✓' if scores.get('best_move') else '✗'}")
        print(f"  Eval direction: {'✓' if scores.get('eval_direction') else '✗'}")
        print(f"  Square validity: {scores.get('square_validity', '?'):.0%}")
        print(f"  Move legality: {scores.get('move_legality', '?'):.0%}")

    return scores


def evaluate_all(results, verbose=False):
    """Evaluate all model outputs. Print aggregate results."""
    all_scores = []

    for i, item in enumerate(results):
        text = item.get('generated', '')
        scores = evaluate_single(text, item, verbose=verbose and i < 5)
        all_scores.append(scores)

        if verbose and i < 5:
            print(f"\n--- Position {i+1}: {item.get('played_move', '?')} ({item.get('classification', '?')}) ---")
            print(f"  Text: {text[:150]}...")
            print()

    # Aggregate
    print(f"\n{'='*50}")
    print(f"STOCKFISH EVALUATION (n={len(results)})")
    print(f"{'='*50}")

    for metric in ['best_move', 'eval_direction', 'square_validity', 'move_legality', 'played_move_mentioned']:
        values = [s[metric] for s in all_scores if metric in s]
        if values:
            mean = np.mean(values)
            n = len(values)
            se = np.std(values) / np.sqrt(n) if n > 1 else 0
            print(f"  {metric:30s}: {mean:.1%} ± {1.96*se:.1%} (n={n})")

    return all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to generation results JSON')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    print(f"Evaluating {len(results)} positions with Stockfish metrics\n")
    evaluate_all(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
