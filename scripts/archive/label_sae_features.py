"""Label SAE features using Sonnet 4.6.

Reads FEN examples from the existing labels.json and re-labels every feature
in one pass, producing all fields: label, category, chip, coaching_useful,
confidence, explanation.

This is the canonical labeling script. All SAE feature labels must come from
this script. If the prompt or parsing changes, re-label everything.

Usage:
    # Re-label all features from existing labels.json (uses stored FEN examples)
    python3 label_sae_features.py --labels backend/lambda/sae_features/versions/puzzle_2048_k32_v1/labels.json

    # Label from a profiles JSON (initial labeling, before examples exist in labels.json)
    python3 label_sae_features.py --profiles output/lichess_rich_profiles.json --output output/labels.json

    # Re-label only low-confidence features
    python3 label_sae_features.py --labels path/to/labels.json --only-regrade medium,low

    # Dry run — print prompt for one feature
    python3 label_sae_features.py --labels path/to/labels.json --dry-run 175
"""
import argparse
import json
import os
import sys
import time
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

MODEL = 'us.anthropic.claude-sonnet-4-6'
MAX_EXAMPLES = 10  # 10 is the sweet spot — enough signal, not so many that thinking gets diluted
THINKING_BUDGET = 4096  # tokens for extended thinking — enough to reason through 10 FENs, not enough to overthink
MAX_WORKERS = 4  # parallel Bedrock calls (Bedrock throttle-safe)
MAX_RETRIES = 3  # retries with exponential backoff
AWS_PROFILE = 'chess-deck'  # AWS profile with Bedrock access

# The fixed set of categories. Must match frontend/src/data/saeCategories.ts.
CATEGORIES = [
    'fork', 'pin_skewer', 'check', 'discovered_attack', 'sacrifice',
    'deflection', 'hanging_pieces', 'back_rank', 'king_attack', 'checkmate',
    'defense', 'passed_pawn', 'endgame_technique', 'zugzwang', 'zwischenzug',
    'multiple_threats', 'forcing_moves', 'captures', 'piece_activity',
    'evaluation', 'quiet_moves', 'opening',
]

CATEGORY_DESCRIPTIONS = {
    'fork': 'Double attacks hitting two targets simultaneously (knight forks, pawn forks, etc.)',
    'pin_skewer': 'Piece alignment tactics — pins (piece stuck defending) or skewers (piece forced to move, exposing another)',
    'check': 'Checks with secondary purpose — winning material, gaining tempo, forcing king to bad square',
    'discovered_attack': 'Moving one piece reveals an attack from another piece behind it',
    'sacrifice': 'Giving up material for a concrete advantage — mating attack, winning back more, decisive initiative',
    'deflection': 'Forcing a defending piece away from its duty, or overloading a piece that defends two things',
    'hanging_pieces': 'Undefended pieces that can be captured for free, or pieces left en prise',
    'back_rank': 'Threats exploiting a king trapped on the back rank with no escape squares',
    'king_attack': 'Coordinated piece attacks against the enemy king — not just checks, but building an assault',
    'checkmate': 'Mating patterns and mating nets — the final blow',
    'defense': 'Finding the best defensive resource when under pressure — blocking, interposing, counter-attacking',
    'passed_pawn': 'Creating, advancing, or promoting passed pawns; blockade and anti-blockade',
    'endgame_technique': 'Theoretical endgame knowledge — opposition, triangulation, rook endgame principles, K+P vs K',
    'zugzwang': 'Positions where the obligation to move is a disadvantage — any move worsens the position',
    'zwischenzug': 'In-between moves — inserting a forcing move before the expected recapture or continuation',
    'multiple_threats': 'Creating two or more simultaneous threats the opponent cannot both address',
    'forcing_moves': 'Sequences of checks, captures, and threats that leave the opponent no choice',
    'captures': 'Winning exchanges, recaptures, and material decisions',
    'piece_activity': 'Improving piece placement — centralization, outposts, activating passive pieces',
    'evaluation': 'Positional assessment — who stands better and why (use sparingly, most features are more specific)',
    'quiet_moves': 'Non-forcing improvements — prophylaxis, regrouping, subtle preparation',
    'opening': 'Opening theory, development principles, and early-game structure',
}


def build_prompt(feature_id, examples, stats, baselines_gc=None):
    """Build the Sonnet prompt for a single feature.

    examples: list of FEN strings with annotations
    stats: dict with fire_rate, phase_*, piece_*, etc.
    baselines_gc: dict of {band: {gc, total_games}} from baselines.json (optional)
    """
    examples_str = '\n'.join(f'  {i+1}. {e}' for i, e in enumerate(examples))

    # Stats section
    stats_lines = []
    fr = stats.get('fire_rate', '?')
    stats_lines.append(f'Fire rate (per position): {fr}%')
    for key in ['phase_opening', 'phase_middlegame', 'phase_endgame',
                'piece_pawn', 'piece_knight', 'piece_bishop', 'piece_rook',
                'piece_queen', 'piece_king',
                'captures', 'checks', 'best_pct', 'alt_pct']:
        if key in stats:
            label = key.replace('piece_', '').replace('phase_', '').replace('_pct', '')
            stats_lines.append(f'{label}: {stats[key]}')

    # Game count from baselines (how many games out of 1000 this feature appears in)
    if baselines_gc:
        gc_lines = []
        for band, info in sorted(baselines_gc.items()):
            gc = info['gc']
            total = info['total_games']
            pct = round(gc / total * 100)
            gc_lines.append(f'{band}: {gc}/{total} games ({pct}%)')
        stats_lines.append(f'Games where this feature fires (per 1000-game corpus):')
        for gl in gc_lines:
            stats_lines.append(f'  {gl}')

    cat_list = '\n'.join(f'  - {cid}: {CATEGORY_DESCRIPTIONS[cid]}' for cid in CATEGORIES)

    prompt = f"""You are a chess expert analyzing neural network features from a Sparse Autoencoder trained on a chess engine.

A single SAE feature fires on specific chess positions. Below are {len(examples)} example positions where this feature activates. Each example shows: FEN | move (best/alt, piece, phase, eval).

POSITIONS WHERE FEATURE {feature_id} FIRES:
{examples_str}

STATISTICS:
{chr(10).join('  ' + s for s in stats_lines)}

YOUR TASK: Analyze the FEN positions carefully. What specific chess concept or pattern connects these positions?

EXAMPLE OF GOOD VS BAD LABELING:
Suppose 8 of 10 positions show a rook on an open file pointing at the enemy king, with the best move being a rook lift or doubling rooks:
  BAD: label="Rook activity in complex positions", category="piece_activity", coaching_useful=false
       (Too vague. "Complex positions" says nothing. Marking it not coaching-useful wastes a real pattern.)
  GOOD: label="Rook invasion on open file toward king", category="king_attack", coaching_useful=true
       (Specific, actionable. A student can learn to look for open files near the king. Category matches the skill.)

CATEGORIES (pick exactly one):
{cat_list}

RULES:
1. READ THE FENS. Set up each board mentally. Note piece placement, pawn structure, king safety, material balance. Don't just rely on move annotations.
2. LABEL must be 3-8 words describing the SPECIFIC pattern. Bad: "Complex tactical position". Good: "Knight fork on king and rook". Bad: "Piece coordination". Good: "Rook battery on open file". Bad: "Multiple threats with counterplay". Good: "Overloaded defender on d-file".
3. CHIP must be 2-3 words for UI display. It's a short version of the label.
4. CATEGORY: Pick the category that best describes the SKILL A PLAYER NEEDS TO LEARN to handle these positions. A knight endgame with a passed pawn = "passed_pawn" (the skill is pawn promotion technique), not "endgame_technique". If truly none fit, use "evaluation".
5. COACHING_USEFUL: Would a chess coach create a lesson around this pattern?
   - TRUE examples: "unprotected back rank" (student can learn to create luft), "knight outpost on d5" (student can learn to fight for outposts), "missed fork" (student can learn to check for double attacks)
   - FALSE examples: "positions with two bishops on the board" (just a board state), "middlegame with equal material" (not actionable), "engine-preferred optimal moves" (too generic — every position has a best move), "multiple reasonable moves available" (describes complexity, not a skill)
   - CRITICAL: Check the game count stats. If this feature fires in >20% of games, it is almost certainly too broad to be a specific coaching pattern. A feature in 50%+ of games is DEFINITELY not coaching-useful — mark it FALSE regardless of what the examples look like. The examples are cherry-picked; the game count tells the truth.
   - Ask: "Could a student practice this and get better at it?" If yes → TRUE. If it just describes what the board looks like → FALSE.
6. CONFIDENCE: high = clear pattern in 8+ of 10 examples. medium = pattern fits 5-7. low = pattern fits fewer than 5.
7. EXPLANATION: 2-3 sentences. Reference at least 2 specific positions by number. Describe what you SEE on the board, not just the move annotations.

Respond with ONLY this JSON (no markdown, no backticks):
{{"label": "...", "category": "...", "chip": "...", "coaching_useful": true/false, "confidence": "high/medium/low", "explanation": "..."}}"""

    return prompt


def parse_response(text):
    """Parse Sonnet's JSON response."""
    # Strip markdown code fences if present
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
        # Fallback: try to extract JSON object from text
        start = clean.find('{')
        end = clean.rfind('}')
        if start >= 0 and end > start:
            try:
                data = json.loads(clean[start:end+1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate required fields
    label = data.get('label', '').strip()
    category = data.get('category', '').strip()
    chip = data.get('chip', '').strip()
    coaching_useful = data.get('coaching_useful', True)
    confidence = data.get('confidence', 'unknown').strip().lower()
    explanation = data.get('explanation', '').strip()

    if not label:
        return None

    # Validate category
    if category not in CATEGORIES:
        print(f'  WARNING: invalid category "{category}" — falling back to "evaluation"', file=sys.stderr)
        category = 'evaluation'

    # Normalize confidence
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


def _load_baselines_gc(baselines_path=None):
    """Load per-feature game counts from baselines.json for the prompt."""
    if not baselines_path:
        # Default: look relative to the repo root
        baselines_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'src', 'data', 'baselines.json')
    if not os.path.exists(baselines_path):
        print(f'  No baselines found at {baselines_path}, skipping gc stats')
        return {}

    with open(baselines_path) as f:
        baselines = json.load(f)

    # Build {feature_id: {band: {gc, total_games}}}
    gc_by_feature = {}
    for band_key, band in baselines.items():
        if band_key.startswith('_') or not isinstance(band, dict):
            continue
        total = band.get('total_games', 1000)
        for fid, f in band.get('features', {}).items():
            if fid not in gc_by_feature:
                gc_by_feature[fid] = {}
            gc_by_feature[fid][band_key] = {'gc': f.get('gc', 0), 'total_games': total}

    print(f'  Loaded baselines gc for {len(gc_by_feature)} features')
    return gc_by_feature


def label_from_labels_json(labels_path, only_regrade=None, dry_run_fid=None, budget_override=None, output_path=None):
    """Re-label features using FEN examples already in labels.json."""
    with open(labels_path) as f:
        labels = json.load(f)

    # Load baselines game counts
    baselines_gc = _load_baselines_gc()

    # Preserve _meta and _version
    meta = labels.get('_meta', {})
    version = labels.get('_version', '')

    # Collect features to label
    to_label = []
    for fid, info in labels.items():
        if fid.startswith('_'):
            continue
        if not isinstance(info, dict):
            continue
        if not info.get('examples'):
            print(f'  SKIP F{fid}: no examples')
            continue
        if only_regrade:
            if info.get('confidence', 'unknown') not in only_regrade:
                continue
        to_label.append((fid, info))

    # Sort by fire rate descending (most common features first)
    to_label.sort(key=lambda x: -x[1].get('fire_rate', 0))
    print(f'{len(to_label)} features to label')

    if dry_run_fid:
        # Print prompt for one feature and exit
        for fid, info in to_label:
            if fid == dry_run_fid:
                stats = {k: v for k, v in info.items()
                         if k in ('fire_rate', 'n_fires', 'phase_opening', 'phase_middlegame',
                                  'phase_endgame', 'piece_pawn', 'piece_knight', 'piece_bishop',
                                  'piece_rook', 'piece_queen', 'piece_king', 'captures', 'checks',
                                  'best_pct', 'alt_pct')}
                prompt = build_prompt(fid, info['examples'][:MAX_EXAMPLES], stats, baselines_gc.get(fid))
                print(prompt)
                return
        print(f'Feature {dry_run_fid} not found')
        return

    budget = budget_override or THINKING_BUDGET
    save_path = output_path or labels_path
    print(f'Thinking budget: {budget} tokens')

    # Thread-local Bedrock clients (one per thread)
    _thread_local = threading.local()

    def get_bedrock():
        if not hasattr(_thread_local, 'client'):
            session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
            _thread_local.client = session.client('bedrock-runtime', region_name='us-east-1')
        return _thread_local.client

    def label_one(fid, info):
        """Label a single feature with retries. Returns (fid, result_dict, is_error)."""
        examples = info['examples'][:MAX_EXAMPLES]
        stats = {k: v for k, v in info.items()
                 if k in ('fire_rate', 'n_fires', 'phase_opening', 'phase_middlegame',
                          'phase_endgame', 'piece_pawn', 'piece_knight', 'piece_bishop',
                          'piece_rook', 'piece_queen', 'piece_king', 'captures', 'checks',
                          'best_pct', 'alt_pct')}
        prompt = build_prompt(fid, examples, stats, baselines_gc.get(fid))

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = get_bedrock().converse(
                    modelId=MODEL,
                    messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                    inferenceConfig={'maxTokens': budget + 1000},
                    additionalModelRequestFields={
                        'thinking': {
                            'type': 'enabled',
                            'budget_tokens': budget,
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
                    # Parse error — retry won't help
                    return (fid, {
                        'label': info.get('label', 'PARSE_ERROR'),
                        'category': info.get('category', 'evaluation'),
                        'chip': info.get('chip', ''),
                        'coaching_useful': info.get('coaching_useful', True),
                        'confidence': 'low',
                        'explanation': f'PARSE ERROR: {text[:200]}',
                    }, True)
            except Exception as e:
                last_err = e
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)

        # All retries exhausted
        return (fid, {
            'label': 'API_ERROR',
            'category': 'evaluation',
            'chip': '',
            'coaching_useful': True,
            'confidence': 'low',
            'explanation': str(last_err)[:200],
        }, True)

    results = {}
    errors = []
    done_count = 0

    print(f'Labeling with {MAX_WORKERS} parallel workers...')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(label_one, fid, info): fid for fid, info in to_label}

        for future in as_completed(futures):
            fid, result, is_error = future.result()
            results[fid] = result
            if is_error:
                errors.append(fid)
            done_count += 1

            if done_count % 20 == 0 or done_count == len(to_label):
                n_high = sum(1 for v in results.values() if v['confidence'] == 'high')
                n_coaching = sum(1 for v in results.values() if v['coaching_useful'])
                cats = {}
                for v in results.values():
                    cats[v['category']] = cats.get(v['category'], 0) + 1
                top_cats = sorted(cats.items(), key=lambda x: -x[1])[:5]
                cat_str = ', '.join(f'{c}={n}' for c, n in top_cats)
                print(f'  {done_count}/{len(to_label)} | high={n_high} coaching={n_coaching} err={len(errors)} | {cat_str}')
                print(f'    F{fid}: {result["label"]} [{result["category"]}] '
                      f'{"coaching" if result["coaching_useful"] else "board_state"}')
                sys.stdout.flush()

    # Back up before overwrite (only if writing to original)
    if save_path == labels_path:
        backup_path = labels_path + '.bak'
        with open(backup_path, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f'\nBackup saved to {backup_path}')

    # Merge results back — preserve examples, max_strength, fire_rate, etc.
    merged = copy.deepcopy(labels)
    for fid, new_fields in results.items():
        if fid in merged:
            merged[fid]['label'] = new_fields['label']
            merged[fid]['category'] = new_fields['category']
            merged[fid]['chip'] = new_fields['chip']
            merged[fid]['coaching_useful'] = new_fields['coaching_useful']
            merged[fid]['confidence'] = new_fields['confidence']
            merged[fid]['explanation'] = new_fields['explanation']
            merged[fid]['quality_note'] = f'relabeled-v2-t{budget}'

    with open(save_path, 'w') as f:
        json.dump(merged, f, indent=2)

    # Summary
    n_high = sum(1 for v in results.values() if v['confidence'] == 'high')
    n_med = sum(1 for v in results.values() if v['confidence'] == 'medium')
    n_low = sum(1 for v in results.values() if v['confidence'] == 'low')
    n_coaching = sum(1 for v in results.values() if v['coaching_useful'])
    n_board = sum(1 for v in results.values() if not v['coaching_useful'])
    print()
    print(f'Done: {len(results)} labeled')
    print(f'  confidence: high={n_high} med={n_med} low={n_low}')
    print(f'  coaching_useful: yes={n_coaching} no={n_board}')
    if errors:
        print(f'  errors: {len(errors)} features: {errors}')

    # Category distribution
    cats = {}
    for v in results.values():
        cats[v['category']] = cats.get(v['category'], 0) + 1
    print('  categories:')
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f'    {cat}: {count}')

    print(f'\nSaved to {save_path}')
    return results


def label_from_profiles(profiles_path, output_path, max_fire_rate=40):
    """Original mode: label from a profiles JSON (initial labeling)."""
    with open(profiles_path) as f:
        profiles = json.load(f)

    to_label = [(fid, p) for fid, p in profiles.items()
                if p.get('fire_rate', 100) < max_fire_rate]
    to_label.sort(key=lambda x: -x[1].get('n_fires', 0))
    print(f'{len(to_label)} features to label (fire_rate < {max_fire_rate}%)')

    _thread_local = threading.local()

    def get_bedrock():
        if not hasattr(_thread_local, 'client'):
            session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
            _thread_local.client = session.client('bedrock-runtime', region_name='us-east-1')
        return _thread_local.client

    def label_one_profile(fid, p):
        examples = p.get('examples', [])[:MAX_EXAMPLES]
        stats = {k: v for k, v in p.items()
                 if k in ('fire_rate', 'n_fires', 'phase_opening', 'phase_middlegame',
                          'phase_endgame', 'piece_pawn', 'piece_knight', 'piece_bishop',
                          'piece_rook', 'piece_queen', 'piece_king', 'captures', 'checks',
                          'best_pct', 'alt_pct')}
        prompt = build_prompt(fid, examples, stats)

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

                result_base = {
                    'examples': p.get('examples', [])[:MAX_EXAMPLES],
                    'fire_rate': p.get('fire_rate'),
                    'n_fires': p.get('n_fires'),
                }
                for k in ('phase_opening', 'phase_middlegame', 'phase_endgame',
                          'piece_pawn', 'piece_knight', 'piece_bishop',
                          'piece_rook', 'piece_queen', 'piece_king'):
                    if k in p:
                        result_base[k] = p[k]

                if parsed:
                    return (fid, {**parsed, **result_base}, False)
                else:
                    return (fid, {
                        'label': 'PARSE_ERROR', 'category': 'evaluation',
                        'chip': '', 'coaching_useful': True,
                        'confidence': 'low', 'explanation': text[:200],
                        **result_base,
                    }, True)
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)

        result_base = {
            'examples': p.get('examples', [])[:MAX_EXAMPLES],
            'fire_rate': p.get('fire_rate'), 'n_fires': p.get('n_fires'),
        }
        return (fid, {
            'label': 'API_ERROR', 'category': 'evaluation',
            'chip': '', 'coaching_useful': True,
            'confidence': 'low', 'explanation': str(last_err)[:200],
            **result_base,
        }, True)

    labels = {}
    done_count = 0

    print(f'Labeling with {MAX_WORKERS} parallel workers...')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(label_one_profile, fid, p): fid for fid, p in to_label}
        for future in as_completed(futures):
            fid, result, is_error = future.result()
            labels[fid] = result
            done_count += 1
            if done_count % 20 == 0:
                print(f'  {done_count}/{len(to_label)}')
                sys.stdout.flush()

    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    n_high = sum(1 for v in labels.values() if v['confidence'] == 'high')
    n_coaching = sum(1 for v in labels.values() if v.get('coaching_useful', True))
    print(f'\nDone: {len(labels)} labeled (high={n_high}, coaching={n_coaching})')
    print(f'Saved to {output_path}')
    return labels


def main():
    parser = argparse.ArgumentParser(description='Label SAE features with Sonnet 4.6')
    parser.add_argument('--labels', default=None,
                        help='Path to existing labels.json (re-label mode — uses stored FEN examples)')
    parser.add_argument('--profiles', default=None,
                        help='Path to profiles JSON (initial labeling mode)')
    parser.add_argument('--output', default=None,
                        help='Output path (required for --profiles mode)')
    parser.add_argument('--only-regrade', default=None,
                        help='Comma-separated confidence levels to re-label (e.g. "medium,low")')
    parser.add_argument('--dry-run', default=None,
                        help='Print prompt for one feature ID and exit')
    parser.add_argument('--budget', type=int, default=None,
                        help='Override thinking budget (default: 4096)')
    parser.add_argument('--max-fire-rate', type=float, default=40,
                        help='Max fire rate to label (default 40%%, only for --profiles mode)')
    args = parser.parse_args()

    if args.labels:
        only = args.only_regrade.split(',') if args.only_regrade else None
        label_from_labels_json(args.labels, only_regrade=only, dry_run_fid=args.dry_run,
                               budget_override=args.budget, output_path=args.output)
    elif args.profiles:
        if not args.output:
            parser.error('--output is required with --profiles')
        label_from_profiles(args.profiles, args.output, args.max_fire_rate)
    else:
        parser.error('Either --labels or --profiles is required')


if __name__ == '__main__':
    main()
