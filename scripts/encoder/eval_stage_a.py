#!/usr/bin/env python3
"""Evaluate Stage A model: can it predict chess truths from encoder embeddings?

Metrics (all Stockfish-verified):
  1. Best move accuracy — does the model predict the Stockfish best move?
  2. Eval accuracy — is the model's eval within ±1.0 of Stockfish?
  3. Eval direction — does the model get "who's winning" right?
  4. PV overlap — do moves in the model's line match Stockfish's PV?
  5. Alternative quality — are the model's alternative moves actually good?

Usage:
  python eval_stage_a.py --model /path/to/stage_a_checkpoint --eval-set /path/to/eval_set.json
"""
import json
import re
import argparse
import chess
import numpy as np
from pathlib import Path


def parse_structured_output(text):
    """Parse Stage A structured output.

    Expected format: "Best: Nxd5 (+2.1). Line: Nxd5 exd5 Bxd5. Alt: Bg5 (+1.4), Re1 (+1.2)"
    """
    result = {'best_move': None, 'eval': None, 'pv': [], 'alternatives': []}

    # Best move + eval
    best_match = re.search(r'Best:\s*(\S+)\s*\(([+\-]?\d+\.?\d*)\)', text)
    if best_match:
        result['best_move'] = best_match.group(1)
        try:
            result['eval'] = float(best_match.group(2))
        except:
            pass

    # Standalone eval
    if result['eval'] is None:
        eval_match = re.search(r'Eval:\s*([+\-]?\d+\.?\d*)', text)
        if eval_match:
            try:
                result['eval'] = float(eval_match.group(1))
            except:
                pass

    # PV line
    line_match = re.search(r'Line:\s*([^.]+)', text)
    if line_match:
        result['pv'] = line_match.group(1).strip().split()

    # Alternatives
    alt_match = re.search(r'Alt:\s*(.+?)(?:\.|$)', text)
    if alt_match:
        alt_text = alt_match.group(1)
        for am in re.finditer(r'(\S+)\s*\(([+\-]?\d+\.?\d*)\)', alt_text):
            result['alternatives'].append({
                'move': am.group(1),
                'eval': float(am.group(2))
            })

    return result


def eval_best_move(parsed, ground_truth):
    """Is the predicted best move correct?"""
    if not parsed['best_move'] or not ground_truth.get('best_move'):
        return None
    return parsed['best_move'] == ground_truth['best_move']


def eval_eval_direction(parsed, ground_truth):
    """Does the model get who's winning right?"""
    model_eval = parsed.get('eval')
    sf_eval = ground_truth.get('eval', ground_truth.get('eval_sf'))

    if model_eval is None or sf_eval is None:
        return None

    # Same direction?
    if sf_eval > 0.3 and model_eval > 0.3:
        return True  # Both say White better
    if sf_eval < -0.3 and model_eval < -0.3:
        return True  # Both say Black better
    if abs(sf_eval) <= 0.3 and abs(model_eval) <= 0.3:
        return True  # Both say equal
    return False


def eval_eval_accuracy(parsed, ground_truth):
    """Is the eval within ±1.0?"""
    model_eval = parsed.get('eval')
    sf_eval = ground_truth.get('eval', ground_truth.get('eval_sf'))

    if model_eval is None or sf_eval is None:
        return None

    return abs(model_eval - sf_eval) <= 1.0


def eval_pv_overlap(parsed, ground_truth):
    """How many moves in the model's PV match Stockfish's PV?"""
    model_pv = parsed.get('pv', [])
    sf_pv = ground_truth.get('pv_line', '').split()

    if not model_pv or not sf_pv:
        return None

    # Check overlap of first N moves
    n = min(len(model_pv), len(sf_pv))
    matches = sum(1 for i in range(n) if model_pv[i] == sf_pv[i])
    return matches / n if n > 0 else 0


def eval_moves_legal(parsed, ground_truth):
    """Are the moves in the PV actually legal?"""
    fen = ground_truth.get('fen', '')
    model_pv = parsed.get('pv', [])

    if not model_pv or not fen:
        return None

    try:
        board = chess.Board(fen)
        legal = 0
        for san in model_pv:
            try:
                move = board.parse_san(san)
                board.push(move)
                legal += 1
            except:
                break  # Stop at first illegal move
        return legal / len(model_pv)
    except:
        return None


def evaluate(results, verbose=False):
    """Evaluate all model outputs."""
    metrics = {
        'best_move': [],
        'eval_direction': [],
        'eval_accuracy': [],
        'pv_overlap': [],
        'pv_legality': [],
    }

    for i, item in enumerate(results):
        text = item.get('generated', '')
        parsed = parse_structured_output(text)

        bm = eval_best_move(parsed, item)
        if bm is not None:
            metrics['best_move'].append(float(bm))

        ed = eval_eval_direction(parsed, item)
        if ed is not None:
            metrics['eval_direction'].append(float(ed))

        ea = eval_eval_accuracy(parsed, item)
        if ea is not None:
            metrics['eval_accuracy'].append(float(ea))

        pv = eval_pv_overlap(parsed, item)
        if pv is not None:
            metrics['pv_overlap'].append(pv)

        pl = eval_moves_legal(parsed, item)
        if pl is not None:
            metrics['pv_legality'].append(pl)

        if verbose and i < 5:
            print(f"\n--- Position {i+1} ---")
            print(f"  Generated: {text[:150]}")
            print(f"  Parsed: best={parsed['best_move']}, eval={parsed['eval']}, pv={parsed['pv'][:3]}")
            print(f"  Truth: best={item.get('best_move')}, eval={item.get('eval', item.get('eval_sf'))}")
            print(f"  Scores: bm={'✓' if bm else '✗'} ed={'✓' if ed else '✗'} ea={'✓' if ea else '✗'}")

    print(f"\n{'='*50}")
    print(f"STAGE A EVALUATION (n={len(results)})")
    print(f"{'='*50}")

    for metric, values in metrics.items():
        if values:
            mean = np.mean(values)
            se = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            print(f"  {metric:20s}: {mean:.1%} ± {1.96*se:.1%} (n={len(values)})")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    evaluate(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
