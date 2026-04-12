"""Relabel WEAK/FAILED SAE features using contrastive examples.

Shows the model positions where the feature fires AND positions where it
doesn't, forcing it to identify what's SPECIFIC to the positive positions.

Uses the same Bedrock call pattern as label_sae_features.py but with a
contrastive prompt design.

Usage:
    # Relabel WEAK features (BA 0.60-0.75)
    python3 relabel_weak_features.py --labels path/to/labels.json --tier weak

    # Relabel FAILED features (BA < 0.60)
    python3 relabel_weak_features.py --labels path/to/labels.json --tier failed

    # Relabel both
    python3 relabel_weak_features.py --labels path/to/labels.json --tier weak,failed

    # Dry run — print prompt for one feature
    python3 relabel_weak_features.py --labels path/to/labels.json --dry-run 175
"""
import argparse
import copy
import json
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

MODEL = 'us.anthropic.claude-sonnet-4-6'
MAX_EXAMPLES = 15  # Use more examples than original (was 10)
N_NEGATIVE = 10    # Contrastive negative examples
THINKING_BUDGET = 4096
MAX_WORKERS = 4
MAX_RETRIES = 3
AWS_PROFILE = None  # Uses default profile (140023406996 Bedrock account)

# Same categories as label_sae_features.py
CATEGORIES = [
    'fork', 'pin_skewer', 'check', 'discovered_attack', 'sacrifice',
    'deflection', 'hanging_pieces', 'back_rank', 'king_attack', 'checkmate',
    'defense', 'passed_pawn', 'endgame_technique', 'zugzwang', 'zwischenzug',
    'multiple_threats', 'forcing_moves', 'captures', 'piece_activity',
    'evaluation', 'quiet_moves', 'opening',
]

CATEGORY_DESCRIPTIONS = {
    'fork': 'Double attacks hitting two targets simultaneously',
    'pin_skewer': 'Pins (piece stuck defending) or skewers (piece forced to move)',
    'check': 'Checks with secondary purpose — winning material, gaining tempo',
    'discovered_attack': 'Moving one piece reveals an attack from another',
    'sacrifice': 'Giving up material for a concrete advantage',
    'deflection': 'Forcing a defending piece away from its duty',
    'hanging_pieces': 'Undefended pieces that can be captured for free',
    'back_rank': 'Threats exploiting a king trapped on the back rank',
    'king_attack': 'Coordinated piece attacks against the enemy king',
    'checkmate': 'Mating patterns and mating nets',
    'defense': 'Finding the best defensive resource when under pressure',
    'passed_pawn': 'Creating, advancing, or promoting passed pawns',
    'endgame_technique': 'Theoretical endgame knowledge — opposition, triangulation, etc.',
    'zugzwang': 'Positions where the obligation to move is a disadvantage',
    'zwischenzug': 'In-between moves inserted before the expected continuation',
    'multiple_threats': 'Creating two or more simultaneous threats',
    'forcing_moves': 'Sequences of checks, captures, and threats',
    'captures': 'Winning exchanges, recaptures, and material decisions',
    'piece_activity': 'Improving piece placement — centralization, outposts',
    'evaluation': 'Positional assessment (use sparingly)',
    'quiet_moves': 'Non-forcing improvements — prophylaxis, regrouping',
    'opening': 'Opening theory, development principles, early-game structure',
}


def extract_fen(example_str):
    return example_str.split(' | ')[0].strip()


def build_fen_pool(labels):
    pool = {}
    for fid, info in labels.items():
        if not isinstance(info, dict):
            continue
        for ex in info.get('examples', []):
            fen = extract_fen(ex)
            if fen not in pool:
                pool[fen] = set()
            pool[fen].add(fid)
    return pool


def sample_negatives(target_fid, fen_pool, n):
    candidates = [fen for fen, fids in fen_pool.items() if target_fid not in fids]
    if len(candidates) < n:
        return random.choices(candidates, k=n) if candidates else []
    return random.sample(candidates, n)


def build_contrastive_prompt(feature_id, positive_examples, negative_fens, stats, old_label, old_ba):
    """Build a contrastive prompt showing both positive and negative examples."""
    pos_str = '\n'.join(f'  {i+1}. {e}' for i, e in enumerate(positive_examples))
    neg_str = '\n'.join(f'  {i+1}. {fen}' for i, fen in enumerate(negative_fens))

    stats_lines = []
    fr = stats.get('fire_rate', '?')
    stats_lines.append(f'Fire rate: {fr}%')
    for key in ['phase_opening', 'phase_middlegame', 'phase_endgame',
                'piece_pawn', 'piece_knight', 'piece_bishop', 'piece_rook',
                'piece_queen', 'piece_king', 'captures', 'checks']:
        if key in stats:
            label = key.replace('piece_', '').replace('phase_', '')
            stats_lines.append(f'{label}: {stats[key]}')

    cat_list = '\n'.join(f'  - {cid}: {CATEGORY_DESCRIPTIONS[cid]}' for cid in CATEGORIES)

    prompt = f"""You are a chess expert analyzing neural network features from a Sparse Autoencoder trained on a chess engine.

A single SAE feature fires on specific chess positions. Your job is to find what DISTINGUISHES the positive positions from the negative ones.

POSITIONS WHERE FEATURE {feature_id} FIRES (positive):
{pos_str}

POSITIONS WHERE FEATURE {feature_id} DOES NOT FIRE (negative):
{neg_str}

STATISTICS:
{chr(10).join('  ' + s for s in stats_lines)}

PREVIOUS LABEL (scored poorly at detection accuracy = {old_ba:.2f}):
  "{old_label}"

This label failed because a judge model couldn't reliably tell which positions it applied to. Your new label must be MORE SPECIFIC — it must describe something visible in the positive positions that is NOT present in the negative positions.

YOUR TASK:
1. Set up each board mentally. Compare positive vs negative positions.
2. What do the POSITIVE positions have in common that the NEGATIVE ones DON'T?
3. Don't just describe what's on the board — describe the SPECIFIC pattern or concept.

CATEGORIES (pick exactly one):
{cat_list}

RULES:
1. READ THE FENS. Compare positive vs negative carefully. What's different?
2. LABEL must be 3-8 words. Must distinguish positive from negative. If you can't find a distinguishing pattern, say confidence="low".
3. CHIP: 2-3 words for UI.
4. CATEGORY: The skill a player needs to learn.
5. COACHING_USEFUL: Could a student practice this? If the feature fires on >20% of positions, it's probably too broad — mark FALSE.
6. CONFIDENCE: high = clear distinguishing pattern in 8+ positive examples. medium = 5-7. low = fewer than 5.
7. EXPLANATION: 2-3 sentences. Reference specific positions by number. Describe what you SEE that differs between positive and negative.

Respond with ONLY this JSON (no markdown, no backticks):
{{"label": "...", "category": "...", "chip": "...", "coaching_useful": true/false, "confidence": "high/medium/low", "explanation": "..."}}"""

    return prompt


def parse_response(text):
    """Parse Sonnet's JSON response. Same as label_sae_features.py."""
    clean = text.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1] if '\n' in clean else clean[3:]
        if clean.endswith('```'):
            clean = clean[:-3]
        clean = clean.strip()
    if clean.startswith('json'):
        clean = clean[4:].strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find('{')
        end = clean.rfind('}')
        if start >= 0 and end > start:
            try:
                data = json.loads(clean[start:end+1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    label = data.get('label', '').strip()
    category = data.get('category', '').strip()
    chip = data.get('chip', '').strip()
    coaching_useful = data.get('coaching_useful', True)
    confidence = data.get('confidence', 'unknown').strip().lower()
    explanation = data.get('explanation', '').strip()

    if not label:
        return None
    if category not in CATEGORIES:
        category = 'evaluation'
    if confidence not in ('high', 'medium', 'low'):
        confidence = 'medium'

    return {
        'label': label,
        'category': category,
        'chip': chip if chip else label[:25],
        'coaching_useful': bool(coaching_useful),
        'confidence': confidence,
        'explanation': explanation,
    }


def run_relabeling(labels_path, tiers, output_path=None, dry_run_fid=None, seed=42):
    random.seed(seed)

    with open(labels_path) as f:
        labels = json.load(f)

    fen_pool = build_fen_pool(labels)
    print(f'FEN pool: {len(fen_pool)} unique positions')

    # Select features to relabel based on tier
    to_relabel = []
    for fid, info in labels.items():
        if not isinstance(info, dict) or not info.get('examples'):
            continue
        ba = info.get('detection_accuracy')
        if ba is None:
            continue
        if dry_run_fid and fid != dry_run_fid:
            continue

        in_tier = False
        if 'weak' in tiers and 0.60 <= ba <= 0.75:
            in_tier = True
        if 'failed' in tiers and ba < 0.60:
            in_tier = True
        if dry_run_fid:
            in_tier = True

        if in_tier:
            to_relabel.append((fid, info))

    to_relabel.sort(key=lambda x: x[1].get('detection_accuracy', 0))
    print(f'{len(to_relabel)} features to relabel')

    if dry_run_fid:
        for fid, info in to_relabel:
            if fid == dry_run_fid:
                examples = info['examples'][:MAX_EXAMPLES]
                neg_fens = sample_negatives(fid, fen_pool, N_NEGATIVE)
                stats = {k: v for k, v in info.items()
                         if k in ('fire_rate', 'phase_opening', 'phase_middlegame',
                                  'phase_endgame', 'piece_pawn', 'piece_knight',
                                  'piece_bishop', 'piece_rook', 'piece_queen',
                                  'piece_king', 'captures', 'checks')}
                prompt = build_contrastive_prompt(
                    fid, examples, neg_fens, stats,
                    info.get('label', ''), info.get('detection_accuracy', 0)
                )
                print(prompt)
                return
        print(f'Feature {dry_run_fid} not found or not in selected tiers')
        return

    save_path = output_path or labels_path

    # Thread-local Bedrock clients
    _thread_local = threading.local()

    def get_bedrock():
        if not hasattr(_thread_local, 'client'):
            session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
            _thread_local.client = session.client('bedrock-runtime', region_name='us-east-1')
        return _thread_local.client

    def relabel_one(fid, info):
        examples = info['examples'][:MAX_EXAMPLES]
        neg_fens = sample_negatives(fid, fen_pool, N_NEGATIVE)
        stats = {k: v for k, v in info.items()
                 if k in ('fire_rate', 'phase_opening', 'phase_middlegame',
                          'phase_endgame', 'piece_pawn', 'piece_knight',
                          'piece_bishop', 'piece_rook', 'piece_queen',
                          'piece_king', 'captures', 'checks')}
        prompt = build_contrastive_prompt(
            fid, examples, neg_fens, stats,
            info.get('label', ''), info.get('detection_accuracy', 0)
        )

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = get_bedrock().converse(
                    modelId=MODEL,
                    messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                    inferenceConfig={'maxTokens': THINKING_BUDGET + 1000},
                    additionalModelRequestFields={
                        'thinking': {
                            'type': 'enabled',
                            'budget_tokens': THINKING_BUDGET,
                        }
                    },
                )
                text = ''
                for block in resp['output']['message']['content']:
                    if block.get('text'):
                        text = block['text']
                        break
                parsed = parse_response(text)
                if parsed:
                    return (fid, parsed, False)
                else:
                    return (fid, {
                        'label': info.get('label', 'PARSE_ERROR'),
                        'category': info.get('category', 'evaluation'),
                        'chip': info.get('chip', ''), 'coaching_useful': info.get('coaching_useful', True),
                        'confidence': 'low', 'explanation': f'PARSE ERROR: {text[:200]}',
                    }, True)
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)

        return (fid, {
            'label': 'API_ERROR', 'category': 'evaluation', 'chip': '',
            'coaching_useful': True, 'confidence': 'low',
            'explanation': str(last_err)[:200],
        }, True)

    results = {}
    errors = []
    done_count = 0

    print(f'Relabeling with {MAX_WORKERS} parallel workers (Sonnet + {THINKING_BUDGET} thinking tokens)...')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(relabel_one, fid, info): fid for fid, info in to_relabel}

        for future in as_completed(futures):
            fid, result, is_error = future.result()
            results[fid] = result
            if is_error:
                errors.append(fid)
            done_count += 1

            if done_count % 10 == 0 or done_count == len(to_relabel):
                old_label = labels[fid].get('label', '?')
                old_ba = labels[fid].get('detection_accuracy', 0)
                new_label = result['label']
                changed = '→' if old_label != new_label else '='
                print(f'  {done_count}/{len(to_relabel)} | F{fid} (old BA={old_ba:.3f})')
                print(f'    OLD: {old_label}')
                print(f'    NEW: {new_label} [{result["category"]}] {result["confidence"]}')
                sys.stdout.flush()

    # Backup before overwrite
    if save_path == labels_path:
        backup_path = labels_path + '.pre-relabel.bak'
        with open(backup_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f'\nBackup saved to {backup_path}')

    # Merge results — preserve examples, fire_rate, detection_accuracy, etc.
    merged = copy.deepcopy(labels)
    for fid, new_fields in results.items():
        if fid in merged:
            old_label = merged[fid].get('label', '')
            merged[fid]['label'] = new_fields['label']
            merged[fid]['category'] = new_fields['category']
            merged[fid]['chip'] = new_fields['chip']
            merged[fid]['coaching_useful'] = new_fields['coaching_useful']
            merged[fid]['confidence'] = new_fields['confidence']
            merged[fid]['explanation'] = new_fields['explanation']
            merged[fid]['old_label'] = old_label
            merged[fid]['relabel_method'] = 'contrastive-v1'

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(merged, f, indent=2)

    # Summary
    n_changed = sum(1 for fid in results
                    if results[fid]['label'] != labels.get(fid, {}).get('label', ''))
    print(f'\nDone: {len(results)} relabeled ({n_changed} labels changed)')
    if errors:
        print(f'  Errors: {len(errors)}: {errors}')
    print(f'Saved to {save_path}')
    print(f'\nNext: re-run detection scoring to measure improvement:')
    print(f'  python3 detection_scoring.py serial --labels {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Relabel weak SAE features with contrastive prompting')
    parser.add_argument('--labels', required=True, help='Path to labels.json (with detection_accuracy)')
    parser.add_argument('--tier', default='weak', help='Tiers to relabel: weak, failed, or weak,failed')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite labels)')
    parser.add_argument('--dry-run', default=None, help='Print prompt for one feature')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tier.split(',')]
    run_relabeling(args.labels, tiers, output_path=args.output,
                   dry_run_fid=args.dry_run, seed=args.seed)


if __name__ == '__main__':
    main()
