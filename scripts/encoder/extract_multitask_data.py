#!/usr/bin/env python3
"""Extract multi-task training data from DynamoDB analyzed games.

Path D from brainstorm: eval + move + tags + position context per moment.
Generates JSONL for encoder→LLM training.

Usage:
    AWS_PROFILE=chess-deck python3 extract_multitask_data.py --output research/data/multitask_moments.jsonl
"""
import argparse
import json
import sys
from typing import Any

import boto3
from boto3.dynamodb.types import TypeDeserializer

deserializer = TypeDeserializer()


def dynamo_to_python(item: dict) -> dict:
    """Convert DynamoDB item to plain Python dict."""
    return {k: deserializer.deserialize(v) for k, v in item.items()}


def extract_moments_from_game(game: dict) -> list[dict]:
    """Extract training examples from a single analyzed game."""
    examples = []

    for color in ['white', 'black']:
        coaching = game.get(f'{color}Coaching', {})
        if not coaching:
            continue

        moments = coaching.get('moments', [])
        player = game.get(f'{color}Player', 'unknown')

        for m in moments:
            fen = m.get('fen', '')
            if not fen:
                continue

            classification = m.get('classification', '')
            if classification not in ('blunder', 'mistake', 'inaccuracy'):
                continue

            eval_val = m.get('eval')
            best_move = m.get('bestMove', '')  # SAN format
            tags = m.get('tags', [])
            position = m.get('position', '')
            win_pct = m.get('winPercent')
            played_san = m.get('san', '')
            ply = m.get('ply')
            alternatives = m.get('alternatives', [])

            if not best_move or eval_val is None:
                continue

            # Build multi-task target
            parts = []

            # Eval
            try:
                ev = float(eval_val)
                parts.append(f"Eval: {ev:+.1f}")
            except (ValueError, TypeError):
                continue

            # Best move (keep SAN — more human-readable than UCI for coaching)
            parts.append(f"Best: {best_move}")

            # Classification
            parts.append(f"Classification: {classification}")

            # Played move
            if played_san:
                parts.append(f"Played: {played_san}")

            # Tags
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")

            # Position context (from our position analysis)
            if position:
                parts.append(f"Position: {position}")

            # Win percent
            if win_pct is not None:
                try:
                    parts.append(f"WinPct: {float(win_pct):.0f}%")
                except (ValueError, TypeError):
                    pass

            target = ". ".join(parts) + "."

            example = {
                'fen': fen,
                'target': target,
                'eval': ev,
                'best_move': best_move,
                'classification': classification,
                'tags': tags,
                'played': played_san,
                'ply': int(ply) if ply else None,
            }
            examples.append(example)

    return examples


def scan_all_games(table_name: str, region: str = 'us-east-1') -> list[dict]:
    """Scan all games from DynamoDB."""
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
        print(f"  Scanned {len(items)} games...", file=sys.stderr, flush=True)

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='research/data/multitask_moments.jsonl')
    parser.add_argument('--table', default='chess-coach-analysis')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--stats', action='store_true', help='Print stats only')
    args = parser.parse_args()

    print(f"Scanning {args.table}...", flush=True)
    games = scan_all_games(args.table, args.region)
    print(f"Found {len(games)} games", flush=True)

    all_examples = []
    games_with_moments = 0
    tag_counts: dict[str, int] = {}

    for game in games:
        examples = extract_moments_from_game(game)
        if examples:
            games_with_moments += 1
            all_examples.extend(examples)
            for ex in examples:
                for tag in ex.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\nResults:", flush=True)
    print(f"  Games with moments: {games_with_moments}/{len(games)}", flush=True)
    print(f"  Total moments: {len(all_examples)}", flush=True)
    print(f"  Unique tags: {len(tag_counts)}", flush=True)

    # Classification distribution
    cls_counts: dict[str, int] = {}
    for ex in all_examples:
        c = ex['classification']
        cls_counts[c] = cls_counts.get(c, 0) + 1
    print(f"\n  Classification distribution:", flush=True)
    for c, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n} ({n/len(all_examples)*100:.1f}%)", flush=True)

    # Top tags
    print(f"\n  Top 15 tags:", flush=True)
    for tag, n in sorted(tag_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {tag}: {n}", flush=True)

    # Eval distribution
    evals = [ex['eval'] for ex in all_examples]
    pos = sum(1 for e in evals if e > 0)
    neg = sum(1 for e in evals if e < 0)
    zero = sum(1 for e in evals if e == 0)
    print(f"\n  Eval distribution: {pos} positive, {neg} negative, {zero} zero", flush=True)

    if args.stats:
        return

    # Write JSONL
    with open(args.output, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"\nWritten to {args.output}", flush=True)

    # Sample
    print(f"\nSample target:", flush=True)
    print(f"  {all_examples[0]['target'][:200]}", flush=True)


if __name__ == '__main__':
    main()
