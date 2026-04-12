#!/usr/bin/env python3
"""Generate coaching text for training moments using Bedrock Claude Haiku.

Takes coaching_moments.jsonl (FEN + move + tags) and generates
natural language coaching text using the feigned discovery prompt.

Uses Bedrock Haiku for cost efficiency (~$0.001 per moment).
1,076 moments × $0.001 ≈ $1.08 total.

Usage:
    # Generate for first 10 (test)
    python generate_coaching_text.py --limit 10

    # Generate all
    python generate_coaching_text.py

    # Resume from where you left off
    python generate_coaching_text.py --resume
"""
import json
import argparse
import time
from pathlib import Path

import boto3

INPUT_PATH = Path('/Users/samtkap/workspace/chess-coach/research/data/coaching_moments.jsonl')
OUTPUT_PATH = Path('/Users/samtkap/workspace/chess-coach/research/data/coaching_training_data.jsonl')

# Bedrock client
bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
)
MODEL_ID = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'


TAG_DESCRIPTIONS = {
    'premature_push': 'pawn push when piece development or consolidation was needed',
    'premature_trade': 'captured when keeping tension was better',
    'left_piece_hanging': 'left a piece undefended',
    'conversion_failure': 'big eval drop from a winning position',
    'quiet_when_winning': 'chose a passive move when check/capture was available',
    'undeveloped_pieces': 'minor pieces still on starting squares',
    'missed_capture': 'missed a favorable capture',
    'missed_check': 'missed a check that was strong',
    'missed_pin': 'missed a pin',
    'missed_fork': 'missed a fork',
    'missed_skewer': 'missed a skewer',
    'missed_pawn_break': 'should have advanced a pawn to break through',
    'missed_discovery': 'missed a discovered attack',
    'back_rank_threat': 'back rank was vulnerable',
    'missed_simplification': 'missed simplifying when ahead',
    'missed_deflection': 'missed deflecting a key defender',
    'missed_overloaded_piece': 'missed exploiting an overloaded piece',
    'rushed_move': 'very fast move without time pressure',
    'allowed_pin': 'move allowed opponent to pin',
    'allowed_fork': 'move allowed opponent to fork',
    'allowed_skewer': 'move allowed opponent to skewer',
    'missed_quiet_move': 'best move was subtle/positional, not a forcing move',
    'missed_sacrifice': 'missed a material sacrifice that was winning',
}


def build_prompt(moment):
    """Build the feigned discovery coaching prompt for a moment."""
    fen = moment['fen']
    played = moment['played_move']
    best = moment['best_move']
    classification = moment['classification']
    tags = moment['tags']
    player_color = moment['player_color']
    position = moment.get('position', '')
    best_line = moment.get('best_line', '')

    tag_text = ', '.join(TAG_DESCRIPTIONS.get(t, t) for t in tags)

    return f"""You are a chess coach explaining a single move to an 1800-rated player. Use feigned discovery — reason through the position as if discovering the problem yourself, not justifying a known answer.

PLAYER COLOR: {player_color.upper()}
MOVE: {played} ({classification})
POSITION (FEN): {fen}

WHAT THE POSITION LOOKS LIKE:
{f'- Position: {position}' if position else '- (no position summary)'}

WHAT HAPPENED:
- The player played {played} ({classification})
- The engine preferred {best}
{f'- Best line: {best_line}' if best_line else ''}

DETECTED PATTERNS: {tag_text}

Write 2-3 short paragraphs:
1. What the position demanded — the key features to notice
2. Why the played move misses this and what the better move achieves
3. A concrete "Next time:" tip for spotting this pattern

RULES:
- Start from what the position demands, not from the engine's answer
- Use **bold** for move names and key concepts
- Reference specific squares and pieces
- DO NOT invent move sequences beyond what's given
- Output ONLY coaching text, no headers"""


def call_bedrock(prompt, max_retries=3):
    """Call Bedrock Haiku and return the response text."""
    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                inferenceConfig={'maxTokens': 300, 'temperature': 0.3},
            )
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Max moments to process (0=all)')
    parser.add_argument('--resume', action='store_true', help='Skip already-processed moments')
    args = parser.parse_args()

    # Load input
    moments = [json.loads(line) for line in INPUT_PATH.read_text().strip().split('\n')]
    print(f"Loaded {len(moments)} coaching moments")

    # Resume logic
    processed_fens = set()
    if args.resume and OUTPUT_PATH.exists():
        existing = [json.loads(line) for line in OUTPUT_PATH.read_text().strip().split('\n') if line]
        processed_fens = {m['fen'] for m in existing}
        print(f"Resuming — {len(processed_fens)} already processed")

    # Process
    limit = args.limit if args.limit > 0 else len(moments)
    processed = 0
    errors = 0

    with open(OUTPUT_PATH, 'a') as f:
        for i, moment in enumerate(moments):
            if processed >= limit:
                break
            if moment['fen'] in processed_fens:
                continue

            prompt = build_prompt(moment)
            coaching_text = call_bedrock(prompt)

            if coaching_text.startswith('ERROR:'):
                errors += 1
                print(f"  [{i+1}] ERROR: {coaching_text[:100]}")
                if errors > 5:
                    print("Too many errors, stopping")
                    break
                continue

            moment['coaching_text'] = coaching_text
            f.write(json.dumps(moment) + '\n')
            processed += 1

            if processed % 10 == 0:
                print(f"  [{processed}/{limit}] Generated coaching for {moment['played_move']} ({moment['classification']})")

    print(f"\nDone: {processed} generated, {errors} errors")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
