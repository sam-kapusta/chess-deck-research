#!/usr/bin/env python3
"""Prepare filtered training data from Lichess studies.

Filters for quality:
- Comment length >= 50 chars (meaningful commentary)
- Balance opening/middlegame/endgame positions
- Deduplicate by FEN prefix (avoid redundant positions)
- Optional: filter by like count

Outputs:
- alignment subset (shorter comments, board descriptions)
- coaching SFT subset (longer comments, instructional)

Usage:
  python prepare_training_data.py
  python prepare_training_data.py --min-likes 100 --min-comment-len 80
"""
import json
import argparse
import random
from pathlib import Path
from collections import Counter

import chess


def load_lichess_data(path):
    return [json.loads(l) for l in Path(path).read_text().strip().split('\n')]


def classify_phase(fen):
    try:
        board = chess.Board(fen)
        n = len(board.piece_map())
        if n > 24: return 'opening'
        elif n > 10: return 'middlegame'
        else: return 'endgame'
    except:
        return 'unknown'


def prepare(args):
    data = load_lichess_data(args.input)
    print(f"Raw: {len(data)} pairs")

    # Filter by comment length
    data = [d for d in data if len(d.get('comment', '')) >= args.min_comment_len]
    print(f"After length filter (>={args.min_comment_len} chars): {len(data)}")

    # Filter by likes if available
    if args.min_likes > 0:
        before = len(data)
        data = [d for d in data if (d.get('likes') or 0) >= args.min_likes]
        print(f"After likes filter (>={args.min_likes}): {len(data)} (removed {before - len(data)})")

    # Add phase
    for d in data:
        d['phase'] = classify_phase(d['fen'])

    phases = Counter(d['phase'] for d in data)
    print(f"Phase distribution: {dict(phases)}")

    # Deduplicate by FEN (keep first occurrence)
    seen_fens = set()
    unique = []
    for d in data:
        fen_key = d['fen'].split(' ')[0]  # Position only, not move counters
        if fen_key not in seen_fens:
            seen_fens.add(fen_key)
            unique.append(d)
    print(f"After FEN dedup: {len(unique)} (removed {len(data) - len(unique)})")
    data = unique

    # Split into alignment (shorter, all phases) and coaching SFT (longer, quality)
    alignment = [d for d in data if len(d['comment']) < 150]
    coaching = [d for d in data if len(d['comment']) >= 80]

    # Balance coaching set phases (oversample minority)
    if args.balance_phases:
        coaching_by_phase = {}
        for d in coaching:
            coaching_by_phase.setdefault(d['phase'], []).append(d)

        if coaching_by_phase:
            max_phase = max(len(v) for v in coaching_by_phase.values())
            target = min(max_phase, args.max_coaching)

            balanced = []
            for phase, items in coaching_by_phase.items():
                if len(items) >= target // 3:
                    random.shuffle(items)
                    balanced.extend(items[:target])
                else:
                    # Oversample minority phases
                    balanced.extend(items * (target // max(len(items), 1) + 1))
            random.shuffle(balanced)
            coaching = balanced[:args.max_coaching]

    # Cap sizes
    random.shuffle(alignment)
    random.shuffle(coaching)
    alignment = alignment[:args.max_alignment]
    coaching = coaching[:args.max_coaching]

    print(f"\nFinal splits:")
    print(f"  Alignment: {len(alignment)} pairs (avg {sum(len(d['comment']) for d in alignment)//max(len(alignment),1)} chars)")
    print(f"  Coaching:  {len(coaching)} pairs (avg {sum(len(d['comment']) for d in coaching)//max(len(coaching),1)} chars)")

    coaching_phases = Counter(d['phase'] for d in coaching)
    print(f"  Coaching phases: {dict(coaching_phases)}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'alignment_lichess.jsonl', 'w') as f:
        for d in alignment:
            f.write(json.dumps({'fen': d['fen'], 'description': d['comment']}) + '\n')

    with open(out_dir / 'coaching_lichess.jsonl', 'w') as f:
        for d in coaching:
            f.write(json.dumps(d) + '\n')

    print(f"\nSaved to {out_dir}/")
    print(f"  alignment_lichess.jsonl ({len(alignment)} pairs)")
    print(f"  coaching_lichess.jsonl ({len(coaching)} pairs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='research/data/lichess_studies.jsonl')
    parser.add_argument('--output-dir', default='research/data/prepared')
    parser.add_argument('--min-comment-len', type=int, default=50)
    parser.add_argument('--min-likes', type=int, default=0)
    parser.add_argument('--max-alignment', type=int, default=30000)
    parser.add_argument('--max-coaching', type=int, default=20000)
    parser.add_argument('--balance-phases', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    prepare(args)


if __name__ == '__main__':
    main()
