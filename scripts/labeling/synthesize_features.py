#!/usr/bin/env python3
"""Synthesize feature labels from Gemini position analyses using Opus/Sonnet.

For each SAE feature, pulls its example positions, looks up Gemini's analysis
of each position, and asks a strong LLM to identify the common mistake pattern.

Usage:
    python3 synthesize_features.py                    # dry run (show coverage)
    python3 synthesize_features.py --run              # run with Sonnet-thinking
    python3 synthesize_features.py --run --model opus # run with Opus 4.7
    python3 synthesize_features.py --run --resume     # skip already-labeled features
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

BASE = Path('/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output')
FEATURES_PATH = BASE / 'labels_512_k8_realgames_v4.json'
ANALYSES_PATH = BASE / 'position_analyses.json'
MAPPING_PATH = BASE / 'id_mapping.json'
OUTPUT_PATH = BASE / 'feature_synthesis.json'
COVERAGE_PATH = BASE / 'feature_coverage.json'

MODELS = {
    'opus': 'us.anthropic.claude-opus-4-7',
    'sonnet': 'us.anthropic.claude-sonnet-4-6',
}

SYSTEM_PROMPT = """You are a chess pattern analyst. You will be given Gemini Pro's analyses of 10-20 chess positions that all activate the same neural network feature in a Sparse Autoencoder trained on blunder positions.

Your job: identify the SPECIFIC common mistake pattern these positions share.

Rules:
- Be specific. "Tactical oversight" is too vague. "Capturing material while ignoring back rank mate threat" is specific.
- Features can be polysemantic — if positions split into 2-3 distinct sub-patterns, list them.
- Ground your label in what the positions actually show, not what you think the feature "should" detect.
- The label should be a coaching concept a 1500-rated player would understand.

Reply as JSON only:
{
  "label": "2-5 word pattern name",
  "description": "One sentence explaining the mistake pattern",
  "sub_patterns": ["sub-pattern 1", "sub-pattern 2"],
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": "high|medium|low",
  "reasoning": "1-2 sentences on why you chose this label"
}"""


def build_position_summary(analysis):
    """Format one Gemini analysis into a compact summary."""
    if not isinstance(analysis, dict):
        return None
    if 'intent' not in analysis:
        return None
    parts = []
    parts.append(f"Intent: {analysis['intent']}")
    parts.append(f"Blunder: {analysis['blunder_trace']}")
    parts.append(f"Failure point: {analysis['point_of_failure']}")
    if analysis.get('best_move_rationale'):
        parts.append(f"Best move: {analysis['best_move_rationale']}")
    if analysis.get('tags'):
        parts.append(f"Tags: {', '.join(analysis['tags'])}")
    if analysis.get('position_context'):
        parts.append(f"Context: {analysis['position_context']}")
    return ' | '.join(parts)


def synthesize(summaries, model_id, client):
    """Call LLM to synthesize a feature label from position summaries."""
    user_text = "Here are the Gemini Pro analyses of positions that all activate the same SAE feature:\n\n"
    for i, s in enumerate(summaries):
        user_text += f"Position {i+1}: {s}\n\n"
    user_text += "What specific chess mistake pattern do these share?"

    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 500,
        'system': SYSTEM_PROMPT,
        'messages': [{'role': 'user', 'content': user_text}],
    }

    if 'opus-4-7' in model_id:
        body['thinking'] = {'type': 'adaptive'}
        body['output_config'] = {'effort': 'high'}
        del body['system']
        body['messages'] = [{'role': 'user', 'content': SYSTEM_PROMPT + '\n\n' + user_text}]
        body['max_tokens'] = 16000
    elif 'sonnet-4-6' in model_id:
        body['thinking'] = {'type': 'enabled', 'budget_tokens': 2000}
        body['temperature'] = 1
        body['max_tokens'] = 2500
    else:
        body['temperature'] = 0.2

    resp = client.invoke_model(
        modelId=model_id, body=json.dumps(body),
        contentType='application/json', accept='application/json'
    )
    result = json.loads(resp['body'].read())

    # Extract text from response (handle thinking blocks)
    text = ''
    for block in result.get('content', []):
        if block.get('type') == 'text':
            text = block['text'].strip()
            break

    # Parse JSON
    clean = text
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1]
        clean = clean.rsplit('```', 1)[0]
    return json.loads(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Actually call LLM (otherwise dry run)')
    parser.add_argument('--model', default='sonnet', choices=['opus', 'sonnet'])
    parser.add_argument('--resume', action='store_true', help='Skip already-labeled features')
    parser.add_argument('--limit', type=int, default=0, help='Max features to process (0=all)')
    parser.add_argument('--parallel', type=int, default=5, help='Starting concurrency (backs off on throttle)')
    args = parser.parse_args()

    with open(FEATURES_PATH) as f:
        features = json.load(f)
    with open(ANALYSES_PATH) as f:
        raw_analyses = json.load(f)
    with open(MAPPING_PATH) as f:
        id_mapping = json.load(f)

    # Build lookup: FEN|UCI -> Gemini analysis
    reverse_map = {v: k for k, v in id_mapping.items()}
    analyses = {}
    for fen_key, pos_id in reverse_map.items():
        if pos_id in raw_analyses:
            analyses[fen_key] = raw_analyses[pos_id]

    print(f"Features: {len(features)}")
    print(f"Gemini analyses: {len(analyses)}")
    print(f"Model: {args.model}")
    print()

    # Load existing results if resuming
    existing = {}
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already done")

    # Compute coverage for all features
    coverage = {}
    low_coverage = []
    no_coverage = []

    for fid, feat in features.items():
        examples = feat.get('examples', [])
        matched = []
        for ex in examples:
            key = f"{ex['fen']}|{ex['uci']}"
            if key in analyses:
                summary = build_position_summary(analyses[key])
                if summary:
                    matched.append({
                        'summary': summary,
                        'fen': ex['fen'],
                        'uci': ex['uci'],
                        'cp_loss': ex.get('cp_loss', 0),
                        'strength': ex.get('strength', 0),
                    })

        coverage[fid] = {
            'total_examples': len(examples),
            'gemini_analyzed': len(matched),
            'old_label': feat.get('label', '?'),
            'fire_rate': feat.get('fire_rate', 0),
        }

        if len(matched) == 0:
            no_coverage.append(fid)
        elif len(matched) < 10:
            low_coverage.append(fid)

    # Save coverage report
    with open(COVERAGE_PATH, 'w') as f:
        json.dump(coverage, f, indent=2)

    good_count = len(features) - len(low_coverage) - len(no_coverage)
    print(f"Coverage: {good_count} good, {len(low_coverage)} low (<10), {len(no_coverage)} none")
    if low_coverage:
        parts = [f"F{fid}({coverage[fid]['gemini_analyzed']})" for fid in low_coverage]
        print(f"Low coverage: {', '.join(parts)}")
    if no_coverage:
        parts = [f"F{fid}" for fid in no_coverage]
        print(f"No coverage: {', '.join(parts)}")
    print()

    if not args.run:
        print("Dry run. Use --run to call LLM.")
        return

    model_id = MODELS[args.model]
    results = dict(existing)
    processed = 0
    errors = 0
    throttles = 0
    skipped = 0

    # Pre-build summaries for all features
    work_items = []
    sorted_features = sorted(features.items(), key=lambda x: -x[1].get('fire_rate', 0))
    if args.limit:
        sorted_features = sorted_features[:args.limit]

    for fid, feat in sorted_features:
        if fid in results and args.resume:
            skipped += 1
            continue

        examples = feat.get('examples', [])
        summaries = []
        for ex in examples:
            key = f"{ex['fen']}|{ex['uci']}"
            if key in analyses:
                s = build_position_summary(analyses[key])
                if s:
                    summaries.append(s)

        if not summaries:
            results[fid] = {
                'label': '?',
                'description': 'No Gemini analyses available',
                'old_label': feat.get('label', '?'),
                'fire_rate': feat.get('fire_rate', 0),
                'gemini_count': 0,
                'method': 'none',
            }
            continue

        work_items.append((fid, feat, summaries))

    total = len(work_items)
    print(f"To process: {total} (skipped {skipped})")

    # Adaptive concurrency: start at --parallel, back off on throttle
    concurrency = args.parallel
    lock = __import__('threading').Lock()
    save_counter = 0

    def process_one(fid, feat, summaries):
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        label_data = synthesize(summaries, model_id, client)
        return fid, feat, summaries, label_data

    done = 0
    i = 0
    while i < total:
        batch_size = min(concurrency, total - i)
        batch = work_items[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {
                pool.submit(process_one, fid, feat, sums): fid
                for fid, feat, sums in batch
            }

            batch_throttled = False
            for future in as_completed(futures):
                fid = futures[future]
                feat = next(f for fi, f, _ in batch if fi == fid)
                sums = next(s for fi, _, s in batch if fi == fid)
                n_analyzed = len(sums)

                try:
                    _, _, _, label_data = future.result()
                    results[fid] = {
                        'label': label_data.get('label', '?'),
                        'description': label_data.get('description', '?'),
                        'sub_patterns': label_data.get('sub_patterns', []),
                        'tags': label_data.get('tags', []),
                        'confidence': label_data.get('confidence', '?'),
                        'reasoning': label_data.get('reasoning', ''),
                        'old_label': feat.get('label', '?'),
                        'fire_rate': feat.get('fire_rate', 0),
                        'gemini_count': n_analyzed,
                        'method': args.model,
                    }
                    processed += 1
                    done += 1
                    print(f"[{done}/{total}] F{fid} ({n_analyzed} pos): {label_data.get('label', '?')}", flush=True)

                except Exception as e:
                    err_str = str(e)
                    if 'ThrottlingException' in err_str or 'Too many requests' in err_str.lower() or 'rate' in err_str.lower():
                        throttles += 1
                        batch_throttled = True
                        # Re-queue this item
                        work_items.append((fid, feat, sums))
                        total += 1
                        print(f"[{done}/{total}] F{fid} THROTTLED (concurrency={concurrency})", flush=True)
                    else:
                        errors += 1
                        done += 1
                        results[fid] = {
                            'label': '?',
                            'description': f'LLM error: {err_str[:100]}',
                            'old_label': feat.get('label', '?'),
                            'fire_rate': feat.get('fire_rate', 0),
                            'gemini_count': n_analyzed,
                            'method': 'error',
                        }
                        print(f"[{done}/{total}] F{fid} ERROR: {err_str[:80]}", flush=True)

            # Adaptive concurrency
            if batch_throttled and concurrency > 1:
                concurrency = max(1, concurrency - 1)
                print(f"  >>> throttled, reducing concurrency to {concurrency}", flush=True)
                time.sleep(2)

        i += batch_size

        # Save periodically
        save_counter += batch_size
        if save_counter >= 25:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            save_counter = 0
            print(f"  --- saved ({processed} done, {errors} err, {throttles} throttles, c={concurrency}) ---", flush=True)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {processed} synthesized, {errors} errors, {throttles} throttles, {skipped} skipped.")
    print(f"Final concurrency: {concurrency} (started at {args.parallel})")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Coverage: {COVERAGE_PATH}")


if __name__ == '__main__':
    main()
