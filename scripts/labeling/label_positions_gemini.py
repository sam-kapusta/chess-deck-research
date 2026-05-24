#!/usr/bin/env python3
"""Label chess positions via Gemini 3.1 Pro API (live, not batch).

Calls one position at a time, saves incrementally. Handles rate limits with backoff.
Uses the PROVEN format from the successful April batch (FEN + UCI + cp_loss, no SF lines).

Usage:
    python scripts/labeling/label_positions_gemini.py --limit 100   # test
    python scripts/labeling/label_positions_gemini.py               # full run
    python scripts/labeling/label_positions_gemini.py --resume      # continue from checkpoint
"""
import argparse
import json
import time
import sys
from google import genai
from google.genai import types

API_KEY = "AIzaSyAtc6yThiFPg7bqHEtVHFE6cqH51Y1Uzco"
MODEL = "gemini-3.1-pro-preview"
POSITIONS_PATH = "output/maia3_positions_for_labeling.json"
OUTPUT_PATH = "output/maia3_gemini_labels.json"

SYSTEM = """You are a chess grandmaster analyzing blunder positions. For each position, explain:
1. What the player was trying to do (their intent)
2. What goes wrong after this move (the refutation/punishment)
3. The specific point of failure (which square, piece, or tactical motif was missed)

Be concrete and specific. Name squares, pieces, and tactical patterns."""

SCHEMA = {
    'type': 'OBJECT',
    'properties': {
        'intent': {'type': 'STRING', 'description': 'What the player was trying to achieve (1-2 sentences)'},
        'blunder_trace': {'type': 'STRING', 'description': 'The refutation: what the opponent does and why it works (2-3 sentences with specific moves)'},
        'point_of_failure': {'type': 'STRING', 'description': 'The specific tactical/positional element missed (1 sentence, name the motif)'},
        'best_move_rationale': {'type': 'STRING', 'description': 'Why the best move works better (1-2 sentences)'},
        'position_context': {'type': 'STRING', 'enum': ['only_move', 'thematic', 'normal']},
        'tags': {'type': 'ARRAY', 'items': {'type': 'STRING'}}
    },
    'required': ['intent', 'blunder_trace', 'point_of_failure', 'best_move_rationale', 'position_context', 'tags']
}


def label_position(client, pos):
    prompt = f"""Analyze this chess blunder:

Position (FEN): {pos['fen']}
Move played: {pos['uci']}
Centipawn loss: {pos['cp_loss']}

This move was a blunder that lost {pos['cp_loss']} centipawns of evaluation. Explain what happened."""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type='application/json',
            response_schema=SCHEMA,
        )
    )

    return json.loads(response.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', default=POSITIONS_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.positions) as f:
        positions = json.load(f)

    if args.limit:
        positions = positions[:args.limit]

    # Resume from checkpoint
    results = {}
    if args.resume:
        try:
            with open(args.output) as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} existing labels")
        except FileNotFoundError:
            pass

    client = genai.Client(api_key=API_KEY)

    n_done = len(results)
    n_errors = 0
    t0 = time.time()
    backoff = 1

    for i, pos in enumerate(positions):
        key = pos['fen'] + '|' + pos['uci']
        if key in results:
            continue

        try:
            label = label_position(client, pos)
            results[key] = label
            n_done += 1
            backoff = 1  # reset backoff on success

            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed * 60
                eta = (len(positions) - n_done) / max(rate, 0.1)
                print(f"  {n_done}/{len(positions)} ({rate:.1f}/min, ETA {eta:.0f}min, {n_errors} errors)", flush=True)

            # Save checkpoint every 50
            if n_done % 50 == 0:
                with open(args.output, 'w') as f:
                    json.dump(results, f)

        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                print(f"  Rate limited. Backing off {backoff}s...", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue  # retry same position
            else:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  Error at {i}: {err_str[:100]}", flush=True)
                results[key] = {'error': err_str[:200]}

    # Final save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {n_done} labeled, {n_errors} errors, {elapsed/60:.1f}min")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
