#!/usr/bin/env python3
"""Step 2: Generate JSONL batch input for Gemini Pro from Stockfish data.

Reads stockfish_data.json and writes batch_input.jsonl where each line
is one position ready for Gemini 3.1 Pro analysis.

Also writes id_mapping.json to map short IDs back to FEN|UCI keys.

Output format (one JSON per line):
  {"custom_id": "pos_0001", "request": {"contents": [...]}}

Usage:
    python3 build_batch_input.py
"""
import json

STOCKFISH_DATA = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/stockfish_data.json'
OUTPUT_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/batch_input.jsonl'
MAPPING_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/id_mapping.json'

INSTRUCTIONS = (
    "Hypothesize the human intent behind the played move. "
    "Trace the refutation move-by-move explaining what each move attacks or defends. "
    "Identify the point of failure (specific piece/square). "
    "Note if this was an 'only move' situation (top alternatives are all much worse) "
    "or a thematic position (top alternatives share a common idea)."
)

RESPONSE_SCHEMA = {
    'type': 'OBJECT',
    'properties': {
        'intent': {'type': 'STRING'},
        'blunder_trace': {'type': 'STRING'},
        'point_of_failure': {'type': 'STRING'},
        'best_move_rationale': {'type': 'STRING'},
        'position_context': {
            'type': 'STRING',
            'enum': ['only_move', 'thematic', 'normal']
        },
        'tags': {
            'type': 'ARRAY',
            'items': {'type': 'STRING'}
        }
    },
    'required': ['intent', 'blunder_trace', 'point_of_failure',
                 'best_move_rationale', 'position_context', 'tags']
}


def build_prompt(pos):
    """Build the text prompt for one position."""
    played = pos['played_san']
    if pos.get('is_check'):
        played += ' (check)'
    if pos.get('is_capture'):
        played += ' (capture)'

    data = {
        'fen': pos['fen'],
        'played_move': played,
        'best_move': pos['best_san'],
        'eval_delta': f"{pos['eval_before']} -> {pos['eval_after']}",
        'phase': pos['phase'],
        'side_to_move': pos['side_to_move'],
        'top_lines': pos.get('top_lines', []),
        'refutation_lines': pos.get('refutation_lines', []),
    }

    return json.dumps(data) + '\n\nInstructions: ' + INSTRUCTIONS


def main():
    with open(STOCKFISH_DATA) as f:
        sf_data = json.load(f)

    # Filter out error entries
    valid = {k: v for k, v in sf_data.items() if 'error' not in v}
    print(f"Positions: {len(valid)} valid, {len(sf_data) - len(valid)} errors", flush=True)

    id_mapping = {}  # pos_NNNN -> FEN|UCI key
    count = 0

    with open(OUTPUT_PATH, 'w') as out:
        for key, pos in valid.items():
            short_id = f"pos_{count:04d}"
            id_mapping[short_id] = key

            prompt_text = build_prompt(pos)

            line = {
                'custom_id': short_id,
                'request': {
                    'systemInstruction': {
                        'parts': [{'text': 'You are an expert chess coach analyzing amateur games.'}]
                    },
                    'contents': [
                        {
                            'role': 'user',
                            'parts': [{'text': prompt_text}]
                        }
                    ],
                    'generationConfig': {
                        'responseMimeType': 'application/json',
                        'responseSchema': RESPONSE_SCHEMA
                    }
                }
            }
            out.write(json.dumps(line) + '\n')
            count += 1

    # Save ID mapping
    with open(MAPPING_PATH, 'w') as f:
        json.dump(id_mapping, f, indent=2)

    print(f"Wrote {count} requests to {OUTPUT_PATH}", flush=True)
    print(f"ID mapping saved to {MAPPING_PATH}", flush=True)

    # Estimate cost
    avg_input = 250  # tokens per request
    avg_output = 150
    input_cost = count * avg_input / 1_000_000 * 1.00  # Pro batch input
    output_cost = count * avg_output / 1_000_000 * 6.00  # Pro batch output
    print(f"Estimated cost: ${input_cost + output_cost:.2f} (Pro batch)", flush=True)


if __name__ == '__main__':
    main()
