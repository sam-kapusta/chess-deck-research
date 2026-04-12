"""Score SAE feature labels using detection accuracy (T3b).

For each labeled feature, ask a judge model whether the label describes
positions where the feature fires (positives) vs random positions (negatives).
Compute balanced accuracy per feature.

Methodology adapted from Sandstone's T3b evaluation framework.

Subcommands:
    prepare  — Generate JSONL input + ground truth file (deterministic, reusable)
    submit   — Upload JSONL to S3, submit Bedrock Batch job
    status   — Check batch job status
    score    — Download batch results, compute balanced accuracy, print report
    serial   — Run everything inline (for debugging or small runs)

Usage:
    # Full batch pipeline
    python3 detection_scoring.py prepare --labels path/to/labels.json
    python3 detection_scoring.py submit --input output/detection_input.jsonl
    python3 detection_scoring.py status --job-arn <arn>
    python3 detection_scoring.py score --job-arn <arn> --ground-truth output/detection_ground_truth.json

    # Serial (debug a single feature)
    python3 detection_scoring.py serial --labels path/to/labels.json --feature 175

    # Serial (all features, ~30 min)
    python3 detection_scoring.py serial --labels path/to/labels.json

    # Dry run — print prompt for one feature
    python3 detection_scoring.py prepare --labels path/to/labels.json --dry-run 175
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

MODEL = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
MAX_WORKERS = 4
MAX_RETRIES = 3
AWS_PROFILE = None  # Uses default profile (140023406996 Bedrock account)
S3_BUCKET = 'chess-stage-a-140023406996'
S3_PREFIX = 'detection-scoring'

N_POSITIVE = 15
N_NEGATIVE = 15


def extract_fen(example_str):
    """Extract just the FEN from an example string like 'FEN | move info'."""
    return example_str.split(' | ')[0].strip()


def build_fen_pool(labels):
    """Build a pool of all FENs across all features, indexed by feature for exclusion."""
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
    """Sample n FENs that do NOT belong to target_fid."""
    candidates = [fen for fen, fids in fen_pool.items() if target_fid not in fids]
    if len(candidates) < n:
        return random.choices(candidates, k=n) if candidates else []
    return random.sample(candidates, n)


def build_prompt(label, explanation, fens_with_truth):
    """Build a prompt that evaluates multiple positions at once.

    Returns (prompt_text, ground_truth_list).
    """
    shuffled = list(fens_with_truth)
    random.shuffle(shuffled)
    ground_truth = [1 if is_pos else 0 for _, is_pos in shuffled]

    positions_str = '\n'.join(
        f'{i+1}. {fen}' for i, (fen, _) in enumerate(shuffled)
    )

    prompt = f"""You are a chess expert evaluating whether a concept label describes specific chess positions.

LABEL: "{label}"
DESCRIPTION: "{explanation}"

For each position below, determine if the label accurately describes a KEY FEATURE of that position. The label should describe something specific and visible — not just vaguely related.

POSITIONS:
{positions_str}

For each position, answer YES or NO. Return ONLY a Python list of 1s and 0s (1=YES, 0=NO), one per position. Example for 4 positions: [1, 0, 1, 0]

Return ONLY the list, nothing else."""

    return prompt, ground_truth


def parse_list_response(text, expected_len):
    """Parse a list response like [1, 0, 1, 0]."""
    clean = text.strip()

    # Try direct eval
    try:
        result = eval(clean)  # noqa: S307
        if isinstance(result, list) and len(result) == expected_len:
            return [int(x) for x in result]
    except Exception:
        pass

    # Fallback: find list pattern
    match = re.search(r'\[[\d,\s]+\]', clean)
    if match:
        try:
            result = eval(match.group())  # noqa: S307
            if isinstance(result, list) and len(result) == expected_len:
                return [int(x) for x in result]
        except Exception:
            pass

    # Last resort: extract individual YES/NO lines
    lines = clean.split('\n')
    predictions = []
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith('YES') or line_upper == '1':
            predictions.append(1)
        elif line_upper.startswith('NO') or line_upper == '0':
            predictions.append(0)
    if len(predictions) == expected_len:
        return predictions

    return None


def balanced_accuracy(predictions, ground_truth):
    """Compute balanced accuracy = (TPR + TNR) / 2."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return (tpr + tnr) / 2, {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'tpr': tpr, 'tnr': tnr}


def print_report(results):
    """Print summary report from scored results."""
    scored = [r for r in results.values() if 'balanced_accuracy' in r]
    if not scored:
        print(f'\nNo features scored successfully. Errors: {sum(1 for r in results.values() if "error" in r)}')
        return

    bas = [r['balanced_accuracy'] for r in scored]
    avg_ba = sum(bas) / len(bas)
    errors = [r for r in results.values() if 'error' in r]

    holds = [r for r in scored if r['balanced_accuracy'] > 0.75]
    weak = [r for r in scored if 0.60 <= r['balanced_accuracy'] <= 0.75]
    failed = [r for r in scored if r['balanced_accuracy'] < 0.60]

    print(f'\n{"="*60}')
    print(f'DETECTION SCORING RESULTS')
    print(f'{"="*60}')
    print(f'Features scored: {len(scored)}')
    print(f'Errors: {len(errors)}')
    print()
    print(f'Mean balanced accuracy: {avg_ba:.3f}')
    print(f'  HOLDS  (BA > 0.75): {len(holds)} ({len(holds)/len(scored)*100:.0f}%)')
    print(f'  WEAK   (0.60-0.75): {len(weak)} ({len(weak)/len(scored)*100:.0f}%)')
    print(f'  FAILED (BA < 0.60): {len(failed)} ({len(failed)/len(scored)*100:.0f}%)')

    # Histogram
    print(f'\nBA distribution:')
    buckets = Counter()
    for ba in bas:
        bucket = int(ba * 10) / 10
        buckets[bucket] += 1
    for bucket in sorted(buckets.keys()):
        bar = '#' * buckets[bucket]
        print(f'  {bucket:.1f}: {bar} ({buckets[bucket]})')

    # Category breakdown
    print(f'\nBy category:')
    cat_scores = {}
    for r in scored:
        cat = r.get('category', 'unknown')
        if cat not in cat_scores:
            cat_scores[cat] = []
        cat_scores[cat].append(r['balanced_accuracy'])
    for cat, scores in sorted(cat_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(scores) / len(scores)
        print(f'  {cat:20s}: avg={avg:.3f} n={len(scores)}')

    # By confidence level
    print(f'\nBy confidence:')
    conf_scores = {}
    for r in scored:
        conf = r.get('confidence', 'unknown')
        if conf not in conf_scores:
            conf_scores[conf] = []
        conf_scores[conf].append(r['balanced_accuracy'])
    for conf in ['high', 'medium', 'low', 'unknown']:
        if conf in conf_scores:
            scores = conf_scores[conf]
            avg = sum(scores) / len(scores)
            print(f'  {conf:10s}: avg={avg:.3f} n={len(scores)}')

    # Top 10 and bottom 10
    sorted_scored = sorted(scored, key=lambda r: -r['balanced_accuracy'])
    print(f'\nTop 10 labels:')
    for r in sorted_scored[:10]:
        print(f'  F{r["feature_id"]:>5s}: BA={r["balanced_accuracy"]:.3f} | {r["label"]}')
    print(f'\nBottom 10 labels:')
    for r in sorted_scored[-10:]:
        print(f'  F{r["feature_id"]:>5s}: BA={r["balanced_accuracy"]:.3f} | {r["label"]}')


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    """Generate JSONL input for Bedrock Batch + ground truth file."""
    with open(args.labels) as f:
        labels = json.load(f)

    random.seed(args.seed)
    fen_pool = build_fen_pool(labels)
    print(f'FEN pool: {len(fen_pool)} unique positions across {len(labels)} features')

    # Collect features
    features = []
    for fid, info in labels.items():
        if not isinstance(info, dict) or not info.get('examples'):
            continue
        features.append((fid, info))
    print(f'{len(features)} features to score')

    if args.dry_run:
        for fid, info in features:
            if fid == args.dry_run:
                pos_fens = list(set(extract_fen(ex) for ex in info['examples']))[:N_POSITIVE]
                neg_fens = sample_negatives(fid, fen_pool, N_NEGATIVE)
                fens_with_truth = [(f, True) for f in pos_fens] + [(f, False) for f in neg_fens]
                prompt, gt = build_prompt(info['label'], info.get('explanation', ''), fens_with_truth)
                print(prompt)
                print(f'\nGround truth ({len(gt)} positions): {gt}')
                return
        print(f'Feature {args.dry_run} not found')
        return

    output_dir = args.output_dir or 'output'
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, 'detection_input.jsonl')
    gt_path = os.path.join(output_dir, 'detection_ground_truth.json')

    ground_truths = {}
    n_records = 0

    with open(input_path, 'w') as f:
        for fid, info in features:
            pos_fens = list(set(extract_fen(ex) for ex in info['examples']))
            if len(pos_fens) > N_POSITIVE:
                pos_fens = random.sample(pos_fens, N_POSITIVE)
            neg_fens = sample_negatives(fid, fen_pool, N_NEGATIVE)
            if not neg_fens:
                print(f'  SKIP F{fid}: no negatives available')
                continue

            fens_with_truth = [(fen, True) for fen in pos_fens] + [(fen, False) for fen in neg_fens]
            prompt, gt = build_prompt(info['label'], info.get('explanation', ''), fens_with_truth)

            record = {
                'recordId': f'feature_{fid}',
                'modelInput': {
                    'anthropic_version': 'bedrock-2023-06-01',
                    'max_tokens': 512,
                    'messages': [
                        {'role': 'user', 'content': prompt}
                    ]
                }
            }
            f.write(json.dumps(record) + '\n')
            n_records += 1

            ground_truths[fid] = {
                'ground_truth': gt,
                'n_positive': len(pos_fens),
                'n_negative': len(neg_fens),
                'label': info['label'],
                'category': info.get('category', ''),
                'confidence': info.get('confidence', ''),
                'coaching_useful': info.get('coaching_useful', True),
            }

    with open(gt_path, 'w') as f:
        json.dump(ground_truths, f, indent=2)

    print(f'\nPrepared {n_records} records')
    print(f'  Input JSONL: {input_path}')
    print(f'  Ground truth: {gt_path}')
    print(f'\nNext: python3 detection_scoring.py submit --input {input_path}')


# ---------------------------------------------------------------------------
# Subcommand: submit
# ---------------------------------------------------------------------------

def cmd_submit(args):
    """Upload JSONL to S3 and submit Bedrock Batch job."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client('s3', region_name='us-east-1')
    bedrock = session.client('bedrock', region_name='us-east-1')

    # Upload input to S3
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    s3_key = f'{S3_PREFIX}/{timestamp}/input.jsonl'
    s3_uri = f's3://{S3_BUCKET}/{s3_key}'
    output_s3_uri = f's3://{S3_BUCKET}/{S3_PREFIX}/{timestamp}/output/'

    print(f'Uploading {args.input} to {s3_uri}...')
    s3.upload_file(args.input, S3_BUCKET, s3_key)

    # Count records
    with open(args.input) as f:
        n_records = sum(1 for line in f if line.strip())
    print(f'  {n_records} records uploaded')

    # Submit batch job
    # Need a role ARN — check if one exists or use the account's service role
    role_arn = args.role_arn
    if not role_arn:
        # Try to find a suitable role
        iam = session.client('iam')
        try:
            roles = iam.list_roles(PathPrefix='/')['Roles']
            bedrock_roles = [r for r in roles if 'bedrock' in r['RoleName'].lower()
                            or 'BedrockBatch' in r['RoleName']]
            if bedrock_roles:
                role_arn = bedrock_roles[0]['Arn']
                print(f'  Using role: {role_arn}')
            else:
                print('ERROR: No Bedrock batch role found. Create one or pass --role-arn.')
                print('  The role needs: bedrock:InvokeModel, s3:GetObject, s3:PutObject')
                return
        except Exception as e:
            print(f'ERROR: Could not list IAM roles: {e}')
            print('  Pass --role-arn explicitly.')
            return

    print(f'Submitting batch job...')
    print(f'  Model: {MODEL}')
    print(f'  Input: {s3_uri}')
    print(f'  Output: {output_s3_uri}')

    resp = bedrock.create_model_invocation_job(
        roleArn=role_arn,
        modelId=MODEL,
        jobName=f'chess-detection-{timestamp}',
        inputDataConfig={
            's3InputDataConfig': {'s3Uri': s3_uri}
        },
        outputDataConfig={
            's3OutputDataConfig': {'s3Uri': output_s3_uri}
        },
    )

    job_arn = resp['jobArn']
    print(f'\nJob submitted: {job_arn}')
    print(f'\nNext: python3 detection_scoring.py status --job-arn {job_arn}')
    print(f'Then: python3 detection_scoring.py score --job-arn {job_arn} --ground-truth output/detection_ground_truth.json')


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    """Check batch job status."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    bedrock = session.client('bedrock', region_name='us-east-1')

    resp = bedrock.get_model_invocation_job(jobIdentifier=args.job_arn)
    status = resp['status']
    print(f'Status: {status}')
    print(f'Created: {resp.get("submitTime", "?")}')
    print(f'Modified: {resp.get("lastModifiedTime", "?")}')
    if resp.get('message'):
        print(f'Message: {resp["message"]}')

    if status == 'Completed':
        output_uri = resp['outputDataConfig']['s3OutputDataConfig']['s3Uri']
        print(f'\nOutput: {output_uri}')
        print(f'\nNext: python3 detection_scoring.py score --job-arn {args.job_arn} --ground-truth output/detection_ground_truth.json')


# ---------------------------------------------------------------------------
# Subcommand: score
# ---------------------------------------------------------------------------

def cmd_score(args):
    """Download batch results, compute balanced accuracy, print report."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    bedrock = session.client('bedrock', region_name='us-east-1')
    s3 = session.client('s3', region_name='us-east-1')

    # Load ground truth
    with open(args.ground_truth) as f:
        ground_truths = json.load(f)

    # Get job output location
    resp = bedrock.get_model_invocation_job(jobIdentifier=args.job_arn)
    if resp['status'] != 'Completed':
        print(f'Job status: {resp["status"]} — not Completed yet')
        return

    output_uri = resp['outputDataConfig']['s3OutputDataConfig']['s3Uri']
    job_id = args.job_arn.split('/')[-1]

    # Parse S3 path
    parts = output_uri.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    prefix = f'{parts[1]}{job_id}/' if len(parts) > 1 else f'{job_id}/'

    # Download all output files
    tmp_dir = '/tmp/detection_batch_output'
    os.makedirs(tmp_dir, exist_ok=True)

    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('/') or obj['Size'] == 0:
                continue
            fname = os.path.basename(obj['Key'])
            local_path = os.path.join(tmp_dir, fname)
            print(f'  Downloading {obj["Key"]}')
            s3.download_file(bucket, obj['Key'], local_path)
            files.append(local_path)

    if not files:
        print('No output files found')
        return

    # Parse results
    results = {}
    errors = []

    for fpath in files:
        opener = gzip.open if fpath.endswith('.gz') else open
        with opener(fpath, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f'invalid JSON in output')
                    continue

                record_id = record.get('recordId', '')
                fid = record_id.replace('feature_', '')

                if record.get('error'):
                    errors.append(f'F{fid}: {record["error"]}')
                    results[fid] = {'error': record['error'], 'label': ground_truths.get(fid, {}).get('label', '?')}
                    continue

                # Extract text
                model_output = record.get('modelOutput', {})
                if isinstance(model_output, str):
                    try:
                        model_output = json.loads(model_output)
                    except json.JSONDecodeError:
                        errors.append(f'F{fid}: can\'t parse modelOutput')
                        continue

                content = model_output.get('content', [])
                text = ''
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        break

                if fid not in ground_truths:
                    errors.append(f'F{fid}: no ground truth')
                    continue

                gt_info = ground_truths[fid]
                gt = gt_info['ground_truth']
                predictions = parse_list_response(text, len(gt))

                if predictions is None:
                    errors.append(f'F{fid}: parse failed: {text[:100]}')
                    results[fid] = {'error': f'parse_failed', 'label': gt_info['label']}
                    continue

                ba, details = balanced_accuracy(predictions, gt)
                results[fid] = {
                    'feature_id': fid,
                    'label': gt_info['label'],
                    'category': gt_info.get('category', ''),
                    'confidence': gt_info.get('confidence', ''),
                    'coaching_useful': gt_info.get('coaching_useful', True),
                    'balanced_accuracy': round(ba, 4),
                    'n_positive': gt_info['n_positive'],
                    'n_negative': gt_info['n_negative'],
                    'predictions': predictions,
                    'ground_truth': gt,
                    **details,
                }

    # Save results
    output_path = args.output or 'output/detection_scores.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if errors:
        print(f'\n{len(errors)} errors:')
        for e in errors[:10]:
            print(f'  {e}')
        if len(errors) > 10:
            print(f'  ... and {len(errors) - 10} more')

    print_report(results)
    print(f'\nSaved to {output_path}')


# ---------------------------------------------------------------------------
# Subcommand: serial
# ---------------------------------------------------------------------------

def cmd_serial(args):
    """Run detection scoring inline (serial with parallel threads)."""
    with open(args.labels) as f:
        labels = json.load(f)

    random.seed(args.seed)
    output_path = args.output or os.path.join(os.path.dirname(args.labels), 'detection_scores.json')

    fen_pool = build_fen_pool(labels)
    print(f'FEN pool: {len(fen_pool)} unique positions across {len(labels)} features')

    features = []
    for fid, info in labels.items():
        if not isinstance(info, dict) or not info.get('examples'):
            continue
        if args.feature and fid != args.feature:
            continue
        features.append((fid, info))
    print(f'{len(features)} features to score')

    _thread_local = threading.local()

    def get_bedrock():
        if not hasattr(_thread_local, 'client'):
            session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
            _thread_local.client = session.client('bedrock-runtime', region_name='us-east-1')
        return _thread_local.client

    def score_one(fid, info):
        pos_fens = list(set(extract_fen(ex) for ex in info['examples']))
        if len(pos_fens) > N_POSITIVE:
            pos_fens = random.sample(pos_fens, N_POSITIVE)
        neg_fens = sample_negatives(fid, fen_pool, N_NEGATIVE)
        if not neg_fens:
            return (fid, {'error': 'no negatives available'}, True)

        fens_with_truth = [(f, True) for f in pos_fens] + [(f, False) for f in neg_fens]
        prompt, ground_truth = build_prompt(info['label'], info.get('explanation', ''), fens_with_truth)

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = get_bedrock().converse(
                    modelId=MODEL,
                    messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                    inferenceConfig={'maxTokens': 512},
                )
                text = ''
                for block in resp['output']['message']['content']:
                    if block.get('text'):
                        text = block['text']
                        break

                predictions = parse_list_response(text, len(ground_truth))
                if predictions is None:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                        continue
                    return (fid, {'error': f'parse_failed: {text[:200]}', 'label': info['label']}, True)

                ba, details = balanced_accuracy(predictions, ground_truth)
                return (fid, {
                    'feature_id': fid,
                    'label': info['label'],
                    'category': info.get('category', ''),
                    'confidence': info.get('confidence', ''),
                    'coaching_useful': info.get('coaching_useful', True),
                    'balanced_accuracy': round(ba, 4),
                    'n_positive': len(pos_fens),
                    'n_negative': len(neg_fens),
                    'predictions': predictions,
                    'ground_truth': ground_truth,
                    **details,
                }, False)

            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)

        return (fid, {'error': str(last_err)[:200], 'label': info['label']}, True)

    results = {}
    errors = []
    done_count = 0

    print(f'Scoring with {MAX_WORKERS} parallel workers...')
    print(f'  {N_POSITIVE} positive + {N_NEGATIVE} negative examples per feature')
    print()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(score_one, fid, info): fid for fid, info in features}
        for future in as_completed(futures):
            fid, result, is_error = future.result()
            results[fid] = result
            if is_error:
                errors.append(fid)
            done_count += 1

            if done_count % 20 == 0 or done_count == len(features):
                scored = [r for r in results.values() if 'balanced_accuracy' in r]
                if scored:
                    bas = [r['balanced_accuracy'] for r in scored]
                    avg_ba = sum(bas) / len(bas)
                    n_holds = sum(1 for b in bas if b > 0.75)
                    n_weak = sum(1 for b in bas if 0.60 <= b <= 0.75)
                    n_failed = sum(1 for b in bas if b < 0.60)
                    print(f'  {done_count}/{len(features)} | avg BA={avg_ba:.3f} | '
                          f'HOLDS={n_holds} WEAK={n_weak} FAILED={n_failed} | err={len(errors)}')
                else:
                    print(f'  {done_count}/{len(features)} | err={len(errors)}')
                if not is_error:
                    print(f'    F{fid}: {result.get("label", "?")} → BA={result["balanced_accuracy"]:.3f}')
                sys.stdout.flush()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_report(results)
    print(f'\nSaved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='T3b detection scoring for SAE labels')
    sub = parser.add_subparsers(dest='command', required=True)

    # prepare
    p_prepare = sub.add_parser('prepare', help='Generate JSONL input + ground truth')
    p_prepare.add_argument('--labels', required=True)
    p_prepare.add_argument('--output-dir', default='output')
    p_prepare.add_argument('--dry-run', default=None, help='Print prompt for one feature')
    p_prepare.add_argument('--seed', type=int, default=42)

    # submit
    p_submit = sub.add_parser('submit', help='Upload to S3 and submit Bedrock Batch job')
    p_submit.add_argument('--input', required=True, help='Path to detection_input.jsonl')
    p_submit.add_argument('--role-arn', default=None, help='IAM role ARN for Bedrock Batch')

    # status
    p_status = sub.add_parser('status', help='Check batch job status')
    p_status.add_argument('--job-arn', required=True)

    # score
    p_score = sub.add_parser('score', help='Download results and compute scores')
    p_score.add_argument('--job-arn', required=True)
    p_score.add_argument('--ground-truth', required=True)
    p_score.add_argument('--output', default=None)

    # serial
    p_serial = sub.add_parser('serial', help='Run everything inline (debug/small runs)')
    p_serial.add_argument('--labels', required=True)
    p_serial.add_argument('--output', default=None)
    p_serial.add_argument('--feature', default=None, help='Score a single feature')
    p_serial.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'score':
        cmd_score(args)
    elif args.command == 'serial':
        cmd_serial(args)


if __name__ == '__main__':
    main()
