#!/usr/bin/env python3
"""Batch label + detection score SAE variants via Bedrock Batch.

Pipeline:
  1. Download profiles from S3
  2. Generate labeling JSONL (one prompt per feature per variant)
  3. Submit Bedrock Batch job
  4. Poll for completion
  5. Parse results → labels
  6. Generate detection scoring JSONL
  7. Submit detection scoring batch job
  8. Parse → balanced accuracy
  9. Print comparison table

Usage:
    # Generate + submit labeling batch
    python3 batch_label_and_score.py label --profiles-dir /tmp/profiles/

    # Check status
    python3 batch_label_and_score.py status --job-arn <arn>

    # Parse labeling results + submit detection scoring
    python3 batch_label_and_score.py score --job-arn <arn> --profiles-dir /tmp/profiles/

    # Full auto pipeline
    python3 batch_label_and_score.py auto --profiles-dir /tmp/profiles/
"""
import argparse, gzip, json, os, random, re, sys, time
import boto3

ROLE_ARN = 'arn:aws:iam::140023406996:role/BedrockBatchInferenceRole'
S3_BUCKET = 'chess-stage-a-140023406996'
S3_PREFIX = 'sae-eval'
# Model IDs for Bedrock Batch
MODELS = {
    'haiku': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    'sonnet': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
}
LABEL_MODEL = MODELS['haiku']
SCORE_MODEL = MODELS['haiku']

CATEGORIES = [
    'fork', 'pin_skewer', 'check', 'discovered_attack', 'sacrifice',
    'deflection', 'hanging_pieces', 'back_rank', 'king_attack', 'checkmate',
    'defense', 'passed_pawn', 'endgame_technique', 'zugzwang', 'zwischenzug',
    'multiple_threats', 'forcing_moves', 'captures', 'piece_activity',
    'evaluation', 'quiet_moves', 'opening',
]

CATEGORY_DESCRIPTIONS = {
    'fork': 'Double attacks hitting two targets simultaneously',
    'pin_skewer': 'Pins or skewers',
    'check': 'Checks with secondary purpose',
    'discovered_attack': 'Moving one piece reveals an attack from another',
    'sacrifice': 'Giving up material for a concrete advantage',
    'deflection': 'Forcing a defending piece away from its duty',
    'hanging_pieces': 'Undefended pieces that can be captured for free',
    'back_rank': 'Threats exploiting a king trapped on the back rank',
    'king_attack': 'Coordinated attacks against the enemy king',
    'checkmate': 'Mating patterns and mating nets',
    'defense': 'Finding the best defensive resource',
    'passed_pawn': 'Creating, advancing, or promoting passed pawns',
    'endgame_technique': 'Theoretical endgame knowledge',
    'zugzwang': 'Obligation to move is a disadvantage',
    'zwischenzug': 'In-between moves',
    'multiple_threats': 'Creating two or more simultaneous threats',
    'forcing_moves': 'Sequences of checks, captures, and threats',
    'captures': 'Winning exchanges and material decisions',
    'piece_activity': 'Improving piece placement',
    'evaluation': 'Positional assessment (use sparingly)',
    'quiet_moves': 'Non-forcing improvements',
    'opening': 'Opening theory and development',
}


def build_labeling_prompt(feature_id, examples, stats, enrichments=None):
    """Build a labeling prompt for Bedrock Batch."""
    lines = []
    for i, e in enumerate(examples[:15]):
        fen = extract_fen(e)
        annotation = enrichments.get(fen, '') if enrichments else ''
        if annotation:
            lines.append(f'  {i+1}. {e}\n      Analysis: {annotation}')
        else:
            lines.append(f'  {i+1}. {e}')
    examples_str = '\n'.join(lines)

    stats_lines = []
    fr = stats.get('fire_rate', '?')
    stats_lines.append(f'Fire rate: {fr}%')
    for key in ['phase_opening', 'phase_middlegame', 'phase_endgame',
                'piece_pawn', 'piece_knight', 'piece_bishop', 'piece_rook',
                'piece_queen', 'piece_king', 'captures', 'checks']:
        if key in stats:
            label = key.replace('piece_', '').replace('phase_', '')
            stats_lines.append(f'{label}: {stats[key]}')

    cat_list = '\n'.join(f'  - {c}: {CATEGORY_DESCRIPTIONS[c]}' for c in CATEGORIES)

    return f"""You are a chess expert analyzing neural network features from a Sparse Autoencoder.

A single SAE feature fires on specific chess positions. Below are example positions where this feature activates.

POSITIONS WHERE FEATURE {feature_id} FIRES (ranked by activation strength, #1 is strongest):
{examples_str}

STATISTICS:
{chr(10).join('  ' + s for s in stats_lines)}

TASK: What specific chess concept connects these positions?

CATEGORIES (pick one):
{cat_list}

RULES:
1. READ THE ANALYSIS for each position. Positions #1-5 are the strongest activations — weight them most.
2. CHECK FOR POLYSEMANTICITY: Do positions #1-5 share the same pattern as #10-15? If the top and bottom examples suggest different chess concepts, this feature is POLYSEMANTIC — set coaching_useful=false and note it in the explanation.
3. LABEL: 3-8 words, specific pattern. Bad: "Complex tactical position". Good: "Knight fork on king and rook".
4. CHIP: 2-3 words for UI.
5. CATEGORY: Skill a player needs to learn.
6. COACHING_USEFUL: false if polysemantic, fire rate >20%, or too vague to practice.
7. CONFIDENCE: high (8+/15 match one concept), medium (5-7), low (<5 or mixed).
8. EXPLANATION: 2-3 sentences referencing specific positions by number. If polysemantic, explain what different concepts are mixed.

Respond with ONLY JSON: {{"label": "...", "category": "...", "chip": "...", "coaching_useful": true/false, "confidence": "high/medium/low", "explanation": "..."}}"""


def extract_fen(example_str):
    return example_str.split(' | ')[0].strip()


def build_detection_prompt(label, explanation, fens_with_truth, enrichments=None):
    """Build detection scoring prompt. Returns (prompt, ground_truth, prefill).

    If enrichments dict is provided, each FEN gets annotated with
    Stockfish eval + python-chess tactical analysis.
    """
    shuffled = list(fens_with_truth)
    random.shuffle(shuffled)
    ground_truth = [1 if is_pos else 0 for _, is_pos in shuffled]
    n = len(shuffled)

    lines = []
    for i, (fen, _) in enumerate(shuffled):
        annotation = enrichments.get(fen, '') if enrichments else ''
        if annotation:
            lines.append(f'{i+1}. {fen}\n   {annotation}')
        else:
            lines.append(f'{i+1}. {fen}')
    positions_str = '\n'.join(lines)

    prompt = f"""Chess expert task: classify {n} positions.

LABEL: "{label}"
MEANING: "{explanation}"

POSITIONS (with analysis):
{positions_str}

For each position: 1 if the label is a KEY FEATURE, 0 if not.
Reply with ONLY a JSON array of {n} integers. No explanation. No text. Just the array."""

    prefill = "["
    return prompt, ground_truth, prefill


def load_profiles(profiles_dir):
    """Load all profile JSONs from a directory. Returns {variant_name: {fid: profile_dict}}."""
    variants = {}
    for fname in os.listdir(profiles_dir):
        if fname.startswith('profiles_') and fname.endswith('.json'):
            name = fname.replace('profiles_', '').replace('.json', '')
            with open(os.path.join(profiles_dir, fname)) as f:
                variants[name] = json.load(f)
            print(f'  Loaded {name}: {len(variants[name])} features')
    return variants


def cmd_label(args):
    """Generate labeling JSONL and submit Bedrock Batch."""
    random.seed(42)
    model_id = MODELS.get(args.model, args.model) if hasattr(args, 'model') and args.model else LABEL_MODEL
    use_thinking = hasattr(args, 'thinking') and args.thinking
    thinking_budget = getattr(args, 'thinking_budget', 4096) or 4096
    print(f'Model: {model_id}, Thinking: {use_thinking} (budget={thinking_budget})')
    variants = load_profiles(args.profiles_dir)

    # Collect all FENs for enrichment
    print('Collecting FENs for enrichment...')
    all_label_fens = set()
    for profiles in variants.values():
        for info in profiles.values():
            for ex in info.get('examples', [])[:15]:
                all_label_fens.add(extract_fen(ex))

    enrichments = {}
    try:
        from enrich_fens import enrich_batch
        stockfish = os.environ.get('STOCKFISH_PATH', '/opt/homebrew/bin/stockfish')
        enrichments = enrich_batch(list(all_label_fens), stockfish_path=stockfish)
        print(f'  Enriched {len(enrichments)} FENs')
    except Exception as e:
        print(f'  Warning: enrichment failed ({e}), labeling without annotations')

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    local_jsonl = f'/tmp/sae_label_batch_{timestamp}.jsonl'
    s3_key = f'{S3_PREFIX}/{timestamp}/label_input.jsonl'

    n_records = 0
    with open(local_jsonl, 'w') as f:
        for variant_name, profiles in variants.items():
            for fid, info in profiles.items():
                if not info.get('examples'):
                    continue
                prompt = build_labeling_prompt(fid, info['examples'], info, enrichments)
                max_tok = thinking_budget + 1024 if use_thinking else 1024
                model_input = {
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': max_tok,
                    'messages': [{'role': 'user', 'content': prompt}]
                }
                if use_thinking:
                    model_input['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget}
                record = {
                    'recordId': f'{variant_name}__feature_{fid}',
                    'modelInput': model_input
                }
                f.write(json.dumps(record) + '\n')
                n_records += 1

    print(f'\nGenerated {n_records} labeling prompts → {local_jsonl}')

    # Upload to S3
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.upload_file(local_jsonl, S3_BUCKET, s3_key)
    s3_uri = f's3://{S3_BUCKET}/{s3_key}'
    output_uri = f's3://{S3_BUCKET}/{S3_PREFIX}/{timestamp}/label_output/'
    print(f'Uploaded to {s3_uri}')

    # Submit batch job
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    resp = bedrock.create_model_invocation_job(
        roleArn=ROLE_ARN,
        modelId=model_id,
        jobName=f'chess-sae-label-{timestamp}',
        inputDataConfig={'s3InputDataConfig': {'s3Uri': s3_uri}},
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': output_uri}},
    )
    job_arn = resp['jobArn']
    print(f'\nSubmitted: {job_arn}')
    print(f'Check: python3 batch_label_and_score.py status --job-arn {job_arn}')

    # Save metadata
    meta = {'job_arn': job_arn, 'timestamp': timestamp, 'n_records': n_records,
            'variants': list(variants.keys()), 's3_input': s3_uri, 's3_output': output_uri}
    meta_path = f'/tmp/sae_label_meta_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata: {meta_path}')


def cmd_status(args):
    """Check batch job status."""
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    resp = bedrock.get_model_invocation_job(jobIdentifier=args.job_arn)
    print(f'Status: {resp["status"]}')
    print(f'Created: {resp.get("submitTime", "?")}')
    print(f'Modified: {resp.get("lastModifiedTime", "?")}')
    if resp.get('message'):
        print(f'Message: {resp["message"]}')
    if resp['status'] == 'Completed':
        output = resp['outputDataConfig']['s3OutputDataConfig']['s3Uri']
        print(f'Output: {output}')


def parse_batch_output(job_arn):
    """Download and parse Bedrock Batch output. Returns {record_id: parsed_json}."""
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')

    resp = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
    if resp['status'] != 'Completed':
        print(f'Job not complete: {resp["status"]}')
        return None

    output_uri = resp['outputDataConfig']['s3OutputDataConfig']['s3Uri']
    job_id = job_arn.split('/')[-1]
    parts = output_uri.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    prefix = f'{parts[1]}{job_id}/' if len(parts) > 1 else f'{job_id}/'

    results = {}
    errors = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('/') or obj['Size'] == 0:
                continue
            local = f'/tmp/batch_out_{os.path.basename(obj["Key"])}'
            s3.download_file(bucket, obj['Key'], local)

            opener = gzip.open if local.endswith('.gz') else open
            with opener(local, 'rt') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                    except: continue

                    rid = record.get('recordId', '')
                    if record.get('error'):
                        errors.append(f'{rid}: {record["error"]}')
                        continue

                    model_output = record.get('modelOutput', {})
                    if isinstance(model_output, str):
                        try: model_output = json.loads(model_output)
                        except: continue

                    text = ''
                    for block in model_output.get('content', []):
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '')
                            break
                    results[rid] = text

    if errors:
        print(f'{len(errors)} errors (first 5):')
        for e in errors[:5]: print(f'  {e}')

    return results


def parse_label_json(text):
    """Parse labeling JSON from LLM output."""
    clean = text.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1] if '\n' in clean else clean[3:]
        if clean.endswith('```'): clean = clean[:-3]
        clean = clean.strip()
    if clean.startswith('json'): clean = clean[4:].strip()
    try:
        return json.loads(clean)
    except:
        start = clean.find('{')
        end = clean.rfind('}')
        if start >= 0 and end > start:
            try: return json.loads(clean[start:end+1])
            except: pass
    return None


def cmd_score(args):
    """Parse labeling results, generate detection scoring batch, submit."""
    random.seed(42)

    print('Parsing labeling results...')
    raw_results = parse_batch_output(args.job_arn)
    if not raw_results:
        return

    # Load profiles for FEN examples (needed for detection scoring negatives)
    variants = load_profiles(args.profiles_dir)

    # Build all FEN pool for negative sampling
    all_fens = {}  # fen -> set of (variant, fid)
    for vname, profiles in variants.items():
        for fid, info in profiles.items():
            for ex in info.get('examples', []):
                fen = extract_fen(ex)
                if fen not in all_fens: all_fens[fen] = set()
                all_fens[fen].add((vname, fid))

    # Parse labels
    labels_by_variant = {}
    for rid, text in raw_results.items():
        parts = rid.split('__feature_')
        if len(parts) != 2: continue
        vname, fid = parts
        parsed = parse_label_json(text)
        if not parsed: continue
        if vname not in labels_by_variant: labels_by_variant[vname] = {}
        labels_by_variant[vname][fid] = parsed

    for vname, labels in labels_by_variant.items():
        print(f'  {vname}: {len(labels)} labels parsed')
        # Save labels
        label_path = os.path.join(args.profiles_dir, f'labels_{vname}.json')
        with open(label_path, 'w') as f:
            json.dump(labels, f, indent=2)

    # Collect all FENs for batch enrichment
    print('\nCollecting FENs for enrichment...')
    all_scoring_fens = set()
    scoring_data = []  # (vname, fid, label_info, pos_fens, neg_fens)

    N_POS = 15
    N_NEG = 15

    for vname, labels in labels_by_variant.items():
        profiles = variants.get(vname, {})
        for fid, label_info in labels.items():
            if fid not in profiles: continue
            info = profiles[fid]
            label = label_info.get('label', '')
            if not label: continue
            pos_fens = list(set(extract_fen(ex) for ex in info.get('examples', [])))[:N_POS]
            neg_candidates = [fen for fen, owners in all_fens.items()
                              if (vname, fid) not in owners]
            neg_fens = random.sample(neg_candidates, min(N_NEG, len(neg_candidates)))
            all_scoring_fens.update(pos_fens)
            all_scoring_fens.update(neg_fens)
            scoring_data.append((vname, fid, label_info, pos_fens, neg_fens))

    # Enrich all FENs with Stockfish + python-chess
    enrichments = {}
    try:
        from enrich_fens import enrich_batch
        stockfish = os.environ.get('STOCKFISH_PATH', '/opt/homebrew/bin/stockfish')
        enrichments = enrich_batch(list(all_scoring_fens), stockfish_path=stockfish)
        print(f'  Enriched {len(enrichments)} FENs')
    except Exception as e:
        print(f'  Warning: enrichment failed ({e}), scoring without annotations')

    # Generate detection scoring JSONL
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    local_jsonl = f'/tmp/sae_detect_batch_{timestamp}.jsonl'
    s3_key = f'{S3_PREFIX}/{timestamp}/detect_input.jsonl'
    ground_truths = {}

    n_records = 0
    with open(local_jsonl, 'w') as f:
        for vname, fid, label_info, pos_fens, neg_fens in scoring_data:
            label = label_info.get('label', '')
            explanation = label_info.get('explanation', '')

            fens_with_truth = [(fen, True) for fen in pos_fens] + [(fen, False) for fen in neg_fens]
            prompt, gt, prefill = build_detection_prompt(label, explanation, fens_with_truth, enrichments)

            rid = f'{vname}__detect_{fid}'
            record = {
                'recordId': rid,
                'modelInput': {
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 256,
                    'messages': [
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': prefill},
                    ]
                }
            }
            f.write(json.dumps(record) + '\n')
            ground_truths[rid] = {
                'ground_truth': gt, 'variant': vname, 'fid': fid,
                'label': label, 'category': label_info.get('category', ''),
                'confidence': label_info.get('confidence', ''),
                'n_positive': len(pos_fens), 'n_negative': len(neg_fens),
            }
            n_records += 1

    # Save ground truths (persistent — survives tmp cleanup)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    gt_path = os.path.join(output_dir, f'sae_detect_gt_{timestamp}.json')
    with open(gt_path, 'w') as f:
        json.dump(ground_truths, f, indent=2)

    print(f'\nGenerated {n_records} detection scoring prompts')

    # Upload + submit
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.upload_file(local_jsonl, S3_BUCKET, s3_key)
    s3_uri = f's3://{S3_BUCKET}/{s3_key}'
    output_uri = f's3://{S3_BUCKET}/{S3_PREFIX}/{timestamp}/detect_output/'

    bedrock = boto3.client('bedrock', region_name='us-east-1')
    resp = bedrock.create_model_invocation_job(
        roleArn=ROLE_ARN,
        modelId=SCORE_MODEL,
        jobName=f'chess-sae-detect-{timestamp}',
        inputDataConfig={'s3InputDataConfig': {'s3Uri': s3_uri}},
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': output_uri}},
    )
    detect_arn = resp['jobArn']
    print(f'Submitted detection scoring: {detect_arn}')
    print(f'Ground truths: {gt_path}')
    print(f'\nNext: python3 batch_label_and_score.py results --job-arn {detect_arn} --ground-truth {gt_path}')


def cmd_results(args):
    """Parse detection scoring results and print comparison."""
    with open(args.ground_truth) as f:
        ground_truths = json.load(f)

    raw = parse_batch_output(args.job_arn)
    if not raw: return

    # Compute balanced accuracy per feature
    from collections import Counter, defaultdict
    variant_scores = defaultdict(list)
    errors = 0

    for rid, text in raw.items():
        if rid not in ground_truths:
            continue
        gt_info = ground_truths[rid]
        gt = gt_info['ground_truth']

        # Parse list — try multiple formats
        predictions = None
        # Format 1: Python list [1, 0, 1, 0]
        match = re.search(r'\[[\d,\s]+\]', text)
        if match:
            try:
                predictions = [int(x) for x in eval(match.group())]
                if len(predictions) != len(gt): predictions = None
            except: pass
        # Format 2: Line-by-line YES/NO or YES (1) / NO (0)
        if predictions is None:
            preds = []
            for line in text.split('\n'):
                line_clean = line.strip().upper()
                if re.match(r'^\d+\.?\s', line_clean):
                    # Numbered line like "1. YES" or "1. King on e8 - YES (1)"
                    if 'YES' in line_clean or re.search(r'\b1\b\s*$', line_clean):
                        preds.append(1)
                    elif 'NO' in line_clean or re.search(r'\b0\b\s*$', line_clean):
                        preds.append(0)
            if len(preds) == len(gt):
                predictions = preds
        # Format 3: Just count all YES/NO occurrences in order
        if predictions is None:
            preds = []
            for line in text.split('\n'):
                line_upper = line.strip().upper()
                if line_upper.startswith('YES') or '- YES' in line_upper or 'YES (1)' in line_upper:
                    preds.append(1)
                elif line_upper.startswith('NO') or '- NO' in line_upper or 'NO (0)' in line_upper:
                    preds.append(0)
            if len(preds) == len(gt):
                predictions = preds

        if predictions is None:
            errors += 1
            continue

        tp = sum(1 for p, g in zip(predictions, gt) if p == 1 and g == 1)
        tn = sum(1 for p, g in zip(predictions, gt) if p == 0 and g == 0)
        fn = sum(1 for p, g in zip(predictions, gt) if p == 0 and g == 1)
        fp = sum(1 for p, g in zip(predictions, gt) if p == 1 and g == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        ba = (tpr + tnr) / 2

        variant_scores[gt_info['variant']].append({
            'fid': gt_info['fid'], 'ba': ba, 'label': gt_info['label'],
            'category': gt_info['category'], 'confidence': gt_info['confidence'],
        })

    # Print comparison
    print(f'\n{"="*70}')
    print('SAE VARIANT COMPARISON — T3b Detection Scoring')
    print(f'{"="*70}')
    print(f'Parse errors: {errors}')

    print(f'\n{"Variant":<25} {"Features":>8} {"Mean BA":>8} {"HOLDS":>6} {"WEAK":>6} {"FAIL":>6}')
    print('-' * 65)

    # Add current production baseline
    print(f'{"puzzle_2048_k32 (prod)":<25} {"395":>8} {"0.650":>8} {"89":>6} {"169":>6} {"137":>6}')

    results_summary = {}
    for vname in sorted(variant_scores.keys()):
        scores = variant_scores[vname]
        bas = [s['ba'] for s in scores]
        avg = sum(bas) / len(bas) if bas else 0
        holds = sum(1 for b in bas if b > 0.75)
        weak = sum(1 for b in bas if 0.60 <= b <= 0.75)
        failed = sum(1 for b in bas if b < 0.60)
        marker = ' ★' if avg > 0.650 else ''
        print(f'{vname:<25} {len(scores):>8} {avg:>8.3f} {holds:>6} {weak:>6} {failed:>6}{marker}')
        results_summary[vname] = {'mean_ba': avg, 'n': len(scores), 'holds': holds, 'weak': weak, 'failed': failed}

    # Champion
    if results_summary:
        champion = max(results_summary.items(), key=lambda x: x[1]['mean_ba'])
        print(f'\n★ CHAMPION: {champion[0]} (mean BA={champion[1]["mean_ba"]:.3f})')

    # Save full results
    output = {'variants': {}, 'production_baseline': {'mean_ba': 0.650, 'holds': 89, 'weak': 169, 'failed': 137}}
    for vname, scores in variant_scores.items():
        output['variants'][vname] = {
            'summary': results_summary[vname],
            'features': scores,
        }
    out_path = args.output or '/tmp/sae_champion_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nFull results: {out_path}')


def cmd_auto(args):
    """Full auto pipeline: label → wait → score → wait → results."""
    # Step 1: Label
    print('=== STEP 1: LABELING ===')
    cmd_label(args)
    # Would need to poll and wait... for now print instructions
    print('\nAuto mode not fully implemented. Run steps manually:')
    print('  1. python3 batch_label_and_score.py label --profiles-dir <dir>')
    print('  2. python3 batch_label_and_score.py status --job-arn <arn>')
    print('  3. python3 batch_label_and_score.py score --job-arn <arn> --profiles-dir <dir>')
    print('  4. python3 batch_label_and_score.py results --job-arn <arn> --ground-truth <path>')


def main():
    parser = argparse.ArgumentParser(description='Batch label + detection score SAE variants')
    sub = parser.add_subparsers(dest='command', required=True)

    p_label = sub.add_parser('label')
    p_label.add_argument('--profiles-dir', required=True)
    p_label.add_argument('--model', default='haiku', help='Model: haiku, sonnet, or full ID')
    p_label.add_argument('--thinking', action='store_true', help='Enable extended thinking')
    p_label.add_argument('--thinking-budget', type=int, default=4096, help='Thinking token budget')

    p_status = sub.add_parser('status')
    p_status.add_argument('--job-arn', required=True)

    p_score = sub.add_parser('score')
    p_score.add_argument('--job-arn', required=True, help='Labeling job ARN')
    p_score.add_argument('--profiles-dir', required=True)

    p_results = sub.add_parser('results')
    p_results.add_argument('--job-arn', required=True, help='Detection scoring job ARN')
    p_results.add_argument('--ground-truth', required=True)
    p_results.add_argument('--output', default=None)

    p_auto = sub.add_parser('auto')
    p_auto.add_argument('--profiles-dir', required=True)

    args = parser.parse_args()
    {'label': cmd_label, 'status': cmd_status, 'score': cmd_score,
     'results': cmd_results, 'auto': cmd_auto}[args.command](args)


if __name__ == '__main__':
    main()
