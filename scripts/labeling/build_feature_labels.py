#!/usr/bin/env python3
"""Step 4: Synthesize feature labels from Gemini position analyses.

Groups position analyses by feature, then uses Sonnet (or any LLM) to
identify the common pattern across positions for each feature.

Can also run without an LLM by just aggregating tags.

Usage:
    python3 build_feature_labels.py [--llm] [--model sonnet]
"""
import argparse
import json
import re
import boto3
from collections import Counter

LABELS_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/labels_512_k8_realgames_v4.json'
ANALYSES_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/position_analyses.json'
MAPPING_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/id_mapping.json'
OUTPUT_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/feature_labels.json'

SONNET = 'us.anthropic.claude-sonnet-4-20250514-v1:0'
HAIKU = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'


def tag_based_label(analyses):
    """Simple: count tags across positions, pick most common."""
    all_tags = []
    for a in analyses:
        if isinstance(a, dict) and 'tags' in a:
            all_tags.extend(a['tags'])

    tag_counts = Counter(all_tags)
    top_tags = [t for t, _ in tag_counts.most_common(5)]
    return {
        'label': ', '.join(top_tags[:3]),
        'description': f"Positions tagged with: {', '.join(top_tags)}",
        'global_tags': top_tags,
        'method': 'tag_count',
    }


def llm_label(analyses, model_id=SONNET):
    """Use Bedrock LLM to synthesize pattern from Gemini's analyses."""
    client = boto3.client('bedrock-runtime', region_name='us-east-1')

    # Build summary of each position's analysis
    summaries = []
    for i, a in enumerate(analyses):
        if isinstance(a, dict) and 'intent' in a:
            s = f"P{i+1}: Intent: {a['intent']}. "
            s += f"Blunder: {a['blunder_trace']}. "
            s += f"Failure: {a['point_of_failure']}. "
            s += f"Tags: {', '.join(a.get('tags', []))}"
            summaries.append(s)
        elif isinstance(a, dict) and 'raw' in a:
            summaries.append(f"P{i+1}: {a['raw'][:200]}")

    if not summaries:
        return tag_based_label(analyses)

    text = '\n'.join(summaries)

    prompt = f"""These are Gemini Pro analyses of chess blunder positions that all activate the same neural network feature. They share a common mistake pattern.

{text}

What SPECIFIC chess mistake pattern do these share? Features can be polysemantic — list sub-patterns if present.

Reply strictly as JSON:
{{ "label": "2-4 word pattern name", "description": "one sentence", "sub_patterns": ["...", "..."], "global_tags": ["...", "..."] }}"""

    body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 300,
        'temperature': 0.3,
        'messages': [{'role': 'user', 'content': prompt}]
    }

    try:
        resp = client.invoke_model(
            modelId=model_id, body=json.dumps(body),
            contentType='application/json', accept='application/json'
        )
        result_text = json.loads(resp['body'].read())['content'][0].get('text', '').strip()

        # Parse JSON from response
        clean = result_text.strip()
        if clean.startswith('```'):
            clean = clean.split('\n', 1)[1]
            clean = clean.rsplit('```', 1)[0]

        result = json.loads(clean)
        result['method'] = 'llm'
        return result
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return tag_based_label(analyses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', action='store_true', help='Use Sonnet for synthesis')
    parser.add_argument('--model', default='sonnet', choices=['sonnet', 'haiku'])
    parser.add_argument('--positions', type=int, default=10, help='Positions per feature')
    args = parser.parse_args()

    model_id = SONNET if args.model == 'sonnet' else HAIKU

    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(ANALYSES_PATH) as f:
        raw_analyses = json.load(f)
    with open(MAPPING_PATH) as f:
        id_mapping = json.load(f)

    # Reverse mapping: FEN|UCI -> analysis
    analyses = {}
    for short_id, fen_key in id_mapping.items():
        if short_id in raw_analyses:
            analyses[fen_key] = raw_analyses[short_id]

    print(f"Features: {len(labels)}", flush=True)
    print(f"Position analyses: {len(analyses)} (mapped from {len(raw_analyses)} raw)", flush=True)
    print(f"Method: {'LLM (' + args.model + ')' if args.llm else 'tag counting'}", flush=True)

    results = {}

    for idx, (fid, feat) in enumerate(sorted(labels.items(), key=lambda x: -x[1].get('fire_rate', 0))):
        # Gather this feature's position analyses
        feature_analyses = []
        position_keys = []
        for ex in feat['examples'][:args.positions]:
            key = f"{ex['fen']}|{ex['uci']}"
            position_keys.append(key)
            if key in analyses:
                feature_analyses.append(analyses[key])

        if not feature_analyses:
            results[fid] = {
                'label': '?',
                'description': 'No position analyses available',
                'fire_rate': feat.get('fire_rate', 0),
                'old_label': feat.get('label', '?'),
                'position_keys': position_keys,
            }
            continue

        # Synthesize label
        if args.llm:
            label_data = llm_label(feature_analyses, model_id)
        else:
            label_data = tag_based_label(feature_analyses)

        results[fid] = {
            'label': label_data.get('label', '?'),
            'description': label_data.get('description', '?'),
            'sub_patterns': label_data.get('sub_patterns', []),
            'global_tags': label_data.get('global_tags', []),
            'fire_rate': feat.get('fire_rate', 0),
            'old_label': feat.get('label', '?'),
            'position_keys': position_keys,
            'method': label_data.get('method', '?'),
        }

        print(f"F{fid} ({feat['fire_rate']*100:.1f}%): {feat.get('label','?'):<28} -> {label_data.get('label','?')}", flush=True)

        # Save every 20
        if (idx + 1) % 20 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [{idx+1}/{len(labels)}] saved", flush=True)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} features labeled.", flush=True)
    print(f"Saved to {OUTPUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
