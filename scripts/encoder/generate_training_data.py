#!/usr/bin/env python3
"""Generate training data for the chess coaching model.

Extracts moments from the local game store and formats them as
(encoder_input, coaching_text) pairs for training the projection + LLM.

For now, generates the INPUT side only. The OUTPUT (coaching text) will be
generated via Bedrock Claude API in a separate step, or hand-written for
initial validation.

Output format (JSONL):
{
    "fen": "r1bqkbnr/...",
    "played_move": "Qd7",
    "best_move": "Nxd5",
    "classification": "blunder",
    "eval_before": 0.5,
    "eval_after": -1.2,
    "tags": ["missed_fork", "left_piece_hanging"],
    "player_color": "white",
    "position_summary": "...",
    "coaching_text": ""  # To be filled by Claude or hand-written
}
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, '/Users/samtkap/workspace/chess-coach/backend/mcp')
from store import _load_index, load_game

OUTPUT_PATH = Path('/Users/samtkap/workspace/chess-coach/research/data/training_moments.jsonl')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_moments():
    """Extract all tagged moments from the game store."""
    idx = _load_index()
    all_moments = []

    for entry in idx:
        game = load_game(entry['game_id'])
        if not game or 'moments' not in game:
            continue

        player_color = game.get('player', {}).get('color', 'white')
        player_rating = game.get('player', {}).get('rating')
        opponent_rating = game.get('opponent', {}).get('rating')

        for m in game['moments']:
            # Only keep moments with tags and a best move
            if not m.get('bestMove') or not m.get('fen'):
                continue

            moment = {
                'fen': m['fen'],
                'played_move': m.get('san', ''),
                'best_move': m.get('bestMove', ''),
                'classification': m.get('classification', ''),
                'eval': m.get('eval', 0),
                'tags': m.get('tags', []),
                'player_color': player_color,
                'player_rating': player_rating,
                'opponent_rating': opponent_rating,
                'position': m.get('position', ''),
                'best_line': m.get('bestLine', ''),
                'alternatives': m.get('alternatives', []),
                'game_id': entry['game_id'],
                'opening': entry.get('opening', ''),
                'coaching_text': '',  # To be filled later
            }
            all_moments.append(moment)

    return all_moments


def main():
    print("=== Training Data Extraction ===")

    moments = extract_moments()
    print(f"Total moments: {len(moments)}")

    # Stats
    by_class = {}
    by_tag = {}
    for m in moments:
        cls = m['classification']
        by_class[cls] = by_class.get(cls, 0) + 1
        for tag in m['tags']:
            by_tag[tag] = by_tag.get(tag, 0) + 1

    print(f"\nBy classification:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    print(f"\nTop 20 tags:")
    for tag, count in sorted(by_tag.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag}: {count}")

    # Filter: only blunders and mistakes with tags (best for coaching)
    coaching_moments = [m for m in moments if m['classification'] in ('blunder', 'mistake') and m['tags']]
    print(f"\nCoaching-quality moments (blunder/mistake with tags): {len(coaching_moments)}")

    # Save all moments
    with open(OUTPUT_PATH, 'w') as f:
        for m in moments:
            f.write(json.dumps(m) + '\n')
    print(f"\nSaved to {OUTPUT_PATH}")

    # Save coaching subset
    coaching_path = OUTPUT_PATH.parent / 'coaching_moments.jsonl'
    with open(coaching_path, 'w') as f:
        for m in coaching_moments:
            f.write(json.dumps(m) + '\n')
    print(f"Saved coaching subset to {coaching_path}")

    # Print a sample
    if coaching_moments:
        print(f"\nSample coaching moment:")
        m = coaching_moments[0]
        print(f"  FEN: {m['fen'][:50]}...")
        print(f"  Played: {m['played_move']} ({m['classification']})")
        print(f"  Best: {m['best_move']}")
        print(f"  Tags: {m['tags']}")
        print(f"  Rating: {m['player_rating']}")


if __name__ == "__main__":
    main()
