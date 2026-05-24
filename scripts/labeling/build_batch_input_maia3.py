#!/usr/bin/env python3
"""Build Gemini 3.1 Pro batch input from Maia3 Stockfish-enriched positions.

Same format as build_batch_input.py (the proven $3/5K run), with one addition:
includes Lichess deep eval cp_loss alongside our depth-18 SF data so Gemini
knows the position IS a blunder even when shallow SF doesn't see it.

Usage:
    python3 build_batch_input_maia3.py \
        --sf-data ../../output/maia3_stockfish_data.json \
        --profiles /tmp/l2_feature_profiles.json \
        -o ../../output/batch_input_maia3.jsonl
"""
import argparse
import json

INSTRUCTIONS = (
    "Hypothesize the human intent behind the played move. "
    "Trace the refutation move-by-move explaining what each move attacks or defends. "
    "Identify the point of failure (specific piece/square). "
    "Note if this was an 'only move' situation (top alternatives are all much worse) "
    "or a thematic position (top alternatives share a common idea)."
)

DEPTH_DISAGREEMENT_NOTE = (
    "NOTE: The Stockfish lines above were computed at depth 18 and may not fully show "
    "the refutation. The Lichess cloud analysis (depth 40+, with tablebases) confirmed "
    "this is a {lichess_cp}cp blunder. If the depth-18 lines don't show a clear punishment, "
    "analyze the position yourself to find the tactical mechanism."
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


def build_prompt(pos, lichess_cp=None):
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

    if lichess_cp is not None:
        data['lichess_deep_eval_cp_loss'] = lichess_cp

    prompt = json.dumps(data) + '\n\nInstructions: ' + INSTRUCTIONS

    if lichess_cp is not None and pos.get('cp_loss', 0) < 100 and lichess_cp >= 200:
        prompt += '\n\n' + DEPTH_DISAGREEMENT_NOTE.format(lichess_cp=lichess_cp)

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sf-data', required=True, help='Path to stockfish_data.json')
    parser.add_argument('--profiles', required=True, help='Path to l2_feature_profiles.json (for Lichess cp_loss)')
    parser.add_argument('-o', '--output', required=True, help='Output batch_input.jsonl path')
    parser.add_argument('--mapping', default=None, help='Output id_mapping.json (default: next to output)')
    args = parser.parse_args()

    mapping_path = args.mapping or args.output.replace('.jsonl', '_id_mapping.json')

    with open(args.sf_data) as f:
        sf_data = json.load(f)

    with open(args.profiles) as f:
        profiles = json.load(f)

    # Build Lichess cp_loss lookup from profiles
    lichess_cp_lookup = {}
    for fid, prof in profiles.items():
        for ex in prof.get('examples', []):
            key = ex.get('fen', '') + '|' + ex.get('uci', '')
            if key and ex.get('cp_loss') is not None:
                lichess_cp_lookup[key] = ex['cp_loss']

    valid = {k: v for k, v in sf_data.items() if 'error' not in v}
    print(f"Positions: {len(valid)} valid, {len(sf_data) - len(valid)} errors", flush=True)
    print(f"Lichess cp_loss available for: {sum(1 for k in valid if k in lichess_cp_lookup)}/{len(valid)}", flush=True)

    # Stats on depth disagreement
    disagree = sum(1 for k, v in valid.items()
                   if v.get('cp_loss', 0) < 100 and lichess_cp_lookup.get(k, 0) >= 200)
    print(f"Depth disagreement (SF<100 but Lichess>=200): {disagree} ({100*disagree/len(valid):.0f}%)", flush=True)

    id_mapping = {}
    count = 0

    with open(args.output, 'w') as out:
        for key, pos in valid.items():
            short_id = f"pos_{count:04d}"
            id_mapping[short_id] = key

            lichess_cp = lichess_cp_lookup.get(key)
            prompt_text = build_prompt(pos, lichess_cp)

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

    with open(mapping_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)

    print(f"Wrote {count} requests to {args.output}", flush=True)
    print(f"ID mapping saved to {mapping_path}", flush=True)

    # Cost estimate (Gemini 3.1 Pro batch rates)
    avg_input = 250
    avg_output = 150
    input_cost = count * avg_input / 1_000_000 * 1.00
    output_cost = count * avg_output / 1_000_000 * 6.00
    total = input_cost + output_cost
    print(f"Estimated cost: ${total:.2f} (Pro batch: ${input_cost:.2f} input + ${output_cost:.2f} output)", flush=True)
    print(f"  (Original 5K run formula said $5.75 but actual was ~$3 — expect ~60% of estimate)", flush=True)


if __name__ == '__main__':
    main()
