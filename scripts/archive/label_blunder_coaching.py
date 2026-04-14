#!/usr/bin/env python3
"""Label blunder SAE features for coaching product.

Produces structured output per feature:
  - short_label (2-4 words): UI chip
  - coaching_theme (from fixed taxonomy): top-level grouping
  - coaching_subtopic (from fixed taxonomy): drill category
  - coaching_advice (one sentence): what to practice
  - mistake_type: what the player did wrong
  - confidence: high/medium/low
  - polysemantic: true/false

Usage:
    python3 label_blunder_coaching.py --profiles profiles.json --output output.jsonl
    python3 label_blunder_coaching.py --profiles profiles.json --submit  # submit Bedrock Batch
"""
import argparse
import json
import os
import sys
import time
import hashlib

import boto3

# ── Fixed coaching taxonomy ──
# These are chess coaching concepts, not SAE concepts.
# They should be stable across SAE architectures.

COACHING_THEMES = {
    'piece_safety': 'Keeping your pieces defended and avoiding material loss',
    'tactical_awareness': 'Recognizing tactics: forks, pins, discovered attacks, back rank',
    'endgame_play': 'Endgame technique, king activity, pawn promotion',
    'positional_judgment': 'Piece activity, pawn structure, weak squares',
    'attack_defense': 'Attacking the king, defending threats, creating/meeting multiple threats',
    'pawn_play': 'Passed pawns, pawn breaks, pawn structure decisions',
}

COACHING_SUBTOPICS = {
    # piece_safety
    'hanging_piece': 'Left a piece undefended that could be captured for free',
    'overloaded_defender': 'A piece was defending too many things at once',
    'inadequate_trade': 'Made a bad exchange or trade',
    # tactical_awareness
    'missed_fork': 'Missed a double attack opportunity',
    'missed_pin_skewer': 'Missed a pin or skewer',
    'missed_discovered_attack': 'Missed a discovered attack',
    'missed_back_rank': 'Missed a back rank threat or vulnerability',
    'missed_check': 'Missed an important check',
    'missed_deflection': 'Missed a deflection tactic',
    'missed_forcing_sequence': 'Played quiet when there was a forcing sequence',
    # endgame_play
    'king_activity': 'Failed to activate the king in the endgame',
    'rook_endgame': 'Wrong technique in rook endgame',
    'minor_piece_endgame': 'Wrong technique in knight/bishop endgame',
    'queen_endgame': 'Wrong technique in queen endgame',
    'pawn_endgame': 'Wrong technique in king and pawn endgame',
    'theoretical_endgame': 'Missed known theoretical endgame technique',
    # positional_judgment
    'passive_piece': 'Placed a piece on a passive square',
    'wrong_piece': 'Moved the wrong piece',
    'positional_misjudgment': 'Misjudged the position evaluation',
    # attack_defense
    'missed_attack': 'Missed an attack on the king',
    'failed_defense': 'Chose the wrong defensive move',
    'ignored_threat': 'Didn\'t address an immediate threat',
    'premature_attack': 'Attacked without sufficient preparation',
    # pawn_play
    'missed_passed_pawn': 'Missed a passed pawn opportunity',
    'wrong_pawn_move': 'Wrong pawn advance or capture',
    'pawn_structure_damage': 'Weakened pawn structure unnecessarily',
}

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
ENRICHMENT_CACHE = {}


def load_enrichments(profiles):
    """Load FEN enrichment cache if available."""
    cache_path = '/home/ec2-user/SageMaker/chess-deck-research/scripts/output/fen_enrichment_cache.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def extract_fen(example_str):
    return example_str.split(' | ')[0].strip()


def build_prompt(feature_id, examples, stats, enrichments):
    lines = []
    for i, e in enumerate(examples[:15]):
        fen = extract_fen(e)
        annotation = enrichments.get(fen, '')
        if annotation:
            lines.append(f'  {i+1}. {e}\n      Analysis: {annotation}')
        else:
            lines.append(f'  {i+1}. {e}')
    examples_str = '\n'.join(lines)

    stats_lines = []
    for key in ['fire_rate', 'phase_opening', 'phase_middlegame', 'phase_endgame',
                'piece_pawn', 'piece_knight', 'piece_bishop', 'piece_rook',
                'piece_queen', 'piece_king', 'captures', 'checks']:
        if key in stats:
            label = key.replace('piece_', '').replace('phase_', '')
            stats_lines.append(f'{label}: {stats[key]}')

    themes_list = '\n'.join(f'  - {k}: {v}' for k, v in COACHING_THEMES.items())
    subtopics_list = '\n'.join(f'  - {k}: {v}' for k, v in COACHING_SUBTOPICS.items())

    return f"""You are a chess coach analyzing blunder patterns from a neural network.

These positions are where a player made a BAD MOVE (lost ≥200 centipawns). The feature detects a specific type of mistake. Your job is to identify what the player did wrong and how to coach them.

POSITIONS WHERE THIS MISTAKE PATTERN FIRES (ranked by strength, #1 is strongest):
{examples_str}

STATISTICS:
{chr(10).join('  ' + s for s in stats_lines)}

COACHING THEMES (pick one):
{themes_list}

COACHING SUBTOPICS (pick one):
{subtopics_list}

RULES:
1. These are BLUNDER positions — the move played was bad. Identify WHAT WENT WRONG, not what the position contains.
2. Positions #1-5 are strongest activations — weight them most.
3. CHECK FOR POLYSEMANTICITY: Do positions #1-5 share the same mistake type as #10-15? If different mistake types are mixed, set polysemantic=true.
4. SHORT_LABEL: 2-4 words describing the mistake. Examples: "Hanging knight", "Missed back rank", "Passive rook retreat", "Wrong king move"
5. COACHING_ADVICE: One sentence telling the player what to practice. Start with a verb. Example: "Check if your pieces are defended before moving."
6. CONFIDENCE: high (8+/15 share one mistake type), medium (5-7), low (<5).

Respond with ONLY this JSON:
{{"short_label": "2-4 words", "coaching_theme": "from list above", "coaching_subtopic": "from list above", "coaching_advice": "One sentence starting with a verb", "mistake_type": "What the player did wrong in 1 sentence", "confidence": "high/medium/low", "polysemantic": true/false}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', required=True, help='Profiles JSON file')
    parser.add_argument('--output', default='/tmp/blunder_coaching_batch.jsonl')
    parser.add_argument('--submit', action='store_true', help='Submit Bedrock Batch')
    parser.add_argument('--model', default='sonnet', choices=['sonnet', 'haiku'])
    parser.add_argument('--thinking', action='store_true')
    args = parser.parse_args()

    MODELS = {
        'sonnet': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
        'haiku': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    }
    model_id = MODELS[args.model]

    with open(args.profiles) as f:
        profiles = json.load(f)

    enrichments = load_enrichments(profiles)
    print(f'Loaded {len(profiles)} features, {len(enrichments)} enriched FENs')

    # Build prompts
    records = []
    for fid, info in profiles.items():
        prompt = build_prompt(fid, info.get('examples', []), info, enrichments)

        record = {
            'recordId': f'blunder_mt_2048_k32_{fid}',
            'modelInput': {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1024,
                'messages': [{'role': 'user', 'content': prompt}],
            }
        }

        if args.thinking:
            record['modelInput']['thinking'] = {'type': 'enabled', 'budget_tokens': 4096}
            record['modelInput']['max_tokens'] = 5120

        records.append(record)

    # Write JSONL
    with open(args.output, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f'Wrote {len(records)} prompts to {args.output}')

    if not args.submit:
        print('Use --submit to submit Bedrock Batch')
        return

    # Upload to S3
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    s3_input = f's3://chess-stage-a-140023406996/sae-eval/{timestamp}/coaching_label_input.jsonl'
    s3_output = f's3://chess-stage-a-140023406996/sae-eval/{timestamp}/coaching_label_output/'

    s3 = boto3.client('s3')
    bucket = 'chess-stage-a-140023406996'
    key = f'sae-eval/{timestamp}/coaching_label_input.jsonl'
    s3.upload_file(args.output, bucket, key)
    print(f'Uploaded to {s3_input}')

    # Submit batch
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    job_name = f'blunder-coaching-{timestamp}'
    resp = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=model_id,
        roleArn='arn:aws:iam::140023406996:role/BedrockBatchInferenceRole',
        inputDataConfig={'s3InputDataConfig': {'s3Uri': s3_input, 's3InputFormat': 'JSONL'}},
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': s3_output}},
    )
    print(f'Submitted: {resp["jobArn"]}')
    print(f'Job name: {job_name}')

    # Save metadata
    meta = {
        'job_arn': resp['jobArn'],
        'timestamp': timestamp,
        'n_records': len(records),
        'model': model_id,
        'thinking': args.thinking,
        'profiles': args.profiles,
    }
    meta_path = f'/tmp/coaching_label_meta_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata: {meta_path}')


if __name__ == '__main__':
    main()
