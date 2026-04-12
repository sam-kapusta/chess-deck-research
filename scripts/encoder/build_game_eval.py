#!/usr/bin/env python3
"""Build Sam's game-level evaluation benchmark.

For each game, identify the 6 most critical positions (by eval swing).
The model must identify these from encoder tokens alone.

Usage:
    AWS_PROFILE=chess-deck python3 build_game_eval.py --output research/data/game_eval_benchmark.jsonl --n-games 50
"""
import argparse
import json
import sys
from typing import Any

import boto3
from boto3.dynamodb.types import TypeDeserializer

deserializer = TypeDeserializer()


def dynamo_to_python(item: dict) -> dict:
    return {k: deserializer.deserialize(v) for k, v in item.items()}


def extract_critical_moments(game: dict, top_k: int = 6) -> dict | None:
    """Find the top-K most critical positions in a game by eval swing.

    Reconstructs FENs from PGN since allMoves doesn't store them.
    """
    import chess
    import chess.pgn
    import io

    all_moves = game.get('allMoves', [])
    pgn_str = game.get('pgn', '')
    if not all_moves or len(all_moves) < 20 or not pgn_str:
        return None

    # Reconstruct FENs from PGN
    try:
        pgn_game = chess.pgn.read_game(io.StringIO(pgn_str))
        if not pgn_game:
            return None
        board = pgn_game.board()
        fens = [board.fen()]
        for move in pgn_game.mainline_moves():
            board.push(move)
            fens.append(board.fen())
    except Exception:
        return None

    if len(fens) < len(all_moves):
        return None

    # Calculate eval swings
    moments = []
    for i, move_data in enumerate(all_moves):
        eval_val = move_data.get('eval')
        prev_eval = all_moves[i - 1].get('eval') if i > 0 else '0'

        if eval_val is None:
            continue

        try:
            ev = float(eval_val)
            prev = float(prev_eval) if prev_eval is not None else 0
            swing = abs(ev - prev)
        except (ValueError, TypeError):
            continue

        fen = fens[i] if i < len(fens) else ''
        classification = move_data.get('classification', 'normal')
        san = move_data.get('san', '')

        moments.append({
            'ply': i + 1,
            'fen': fen,
            'san': san,
            'eval': ev,
            'prev_eval': prev,
            'swing': swing,
            'classification': classification,
        })

    if len(moments) < top_k * 2:
        return None

    # Top-K by eval swing
    top_moments = sorted(moments, key=lambda m: m['swing'], reverse=True)[:top_k]
    top_moments.sort(key=lambda m: m['ply'])  # chronological order

    # All FENs in game order (for encoder input)
    all_fens = [m['fen'] for m in moments if m['fen']]

    return {
        'gameOriginId': game.get('gameOriginId', ''),
        'white': game.get('whitePlayer', ''),
        'black': game.get('blackPlayer', ''),
        'result': game.get('result', ''),
        'opening': game.get('opening', ''),
        'total_moves': len(moments),
        'all_fens': all_fens,
        'critical_moments': top_moments,
        'critical_plies': [m['ply'] for m in top_moments],
    }


def scan_all_games(table_name: str, region: str = 'us-east-1') -> list[dict]:
    client = boto3.client('dynamodb', region_name=region)
    items = []
    kwargs = {'TableName': table_name}
    while True:
        resp = client.scan(**kwargs)
        for item in resp.get('Items', []):
            items.append(dynamo_to_python(item))
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='research/data/game_eval_benchmark.jsonl')
    parser.add_argument('--table', default='chess-coach-analysis')
    parser.add_argument('--n-games', type=int, default=50)
    parser.add_argument('--top-k', type=int, default=6)
    args = parser.parse_args()

    print(f"Scanning {args.table}...", flush=True)
    games = scan_all_games(args.table)
    print(f"Found {len(games)} games", flush=True)

    benchmarks = []
    for game in games:
        result = extract_critical_moments(game, top_k=args.top_k)
        if result and result['total_moves'] >= 20:
            benchmarks.append(result)

    print(f"Games with enough data: {len(benchmarks)}", flush=True)

    # Select N games with highest total swing (most interesting games)
    benchmarks.sort(key=lambda b: sum(m['swing'] for m in b['critical_moments']), reverse=True)
    selected = benchmarks[:args.n_games]

    # Stats
    avg_swing = sum(m['swing'] for b in selected for m in b['critical_moments']) / (len(selected) * args.top_k)
    avg_moves = sum(b['total_moves'] for b in selected) / len(selected)
    print(f"\nBenchmark: {len(selected)} games", flush=True)
    print(f"  Avg moves per game: {avg_moves:.0f}", flush=True)
    print(f"  Avg swing per critical moment: {avg_swing:.2f}", flush=True)
    print(f"  Avg game total swing: {sum(m['swing'] for m in selected[0]['critical_moments']):.1f} (top game)", flush=True)

    # Random baseline expectation
    avg_total = avg_moves
    expected_random = args.top_k * args.top_k / avg_total
    print(f"  Random baseline: {expected_random:.1f} / {args.top_k} overlap expected", flush=True)

    with open(args.output, 'w') as f:
        for b in selected:
            f.write(json.dumps(b) + '\n')
    print(f"\nWritten to {args.output}", flush=True)

    # Sample
    s = selected[0]
    print(f"\nSample game: {s['white']} vs {s['black']} ({s['opening'][:50]})", flush=True)
    print(f"  Critical plies: {s['critical_plies']}", flush=True)
    for m in s['critical_moments']:
        print(f"    Ply {m['ply']}: {m['san']} ({m['classification']}) swing={m['swing']:.1f}", flush=True)


if __name__ == '__main__':
    main()
