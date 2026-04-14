#!/usr/bin/env python3
"""Parse Bedrock Batch labeling and detection scoring results.

Downloads output from S3, parses JSON responses, computes balanced accuracy,
and produces comparison tables.

Usage:
    # Parse labeling results
    python3 parse_batch_results.py labels --job-id pztzjp2jzh8v

    # Parse detection results and compute BA
    python3 parse_batch_results.py detect --job-id cvrbvrpaykib --gt-file output/k64_baseline/sae_detect_gt.json

    # Compare multiple detection runs
    python3 parse_batch_results.py compare --jobs haiku_raw:jojrkl7x0nl9 haiku_enriched:cvrbvrpaykib
"""
import argparse, json, os, re, sys
import boto3

BUCKET = 'chess-stage-a-140023406996'
REGION = 'us-east-1'


def get_job_output_key(job_id):
    """Find the output JSONL key for a batch job."""
    bedrock = boto3.client('bedrock', region_name=REGION)
    s3 = boto3.client('s3')
    arn = f'arn:aws:bedrock:{REGION}:140023406996:model-invocation-job/{job_id}'
    detail = bedrock.get_model_invocation_job(jobIdentifier=arn)
    output_uri = detail['outputDataConfig']['s3OutputDataConfig']['s3Uri']
    prefix = output_uri.replace(f's3://{BUCKET}/', '')

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    for obj in resp.get('Contents', []):
        if obj['Key'].endswith('.jsonl.out'):
            return obj['Key']
    return None


def download_and_parse(s3_key):
    """Download JSONL from S3 and parse each record."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
    records = []
    for line in obj['Body'].read().decode().strip().split('\n'):
        rec = json.loads(line)
        rid = rec.get('recordId', '')
        output = rec.get('modelOutput', {})
        content = output.get('content', [])
        text = ''
        for block in content:
            if block.get('type') == 'text':
                text = block['text']
                break
        records.append({'id': rid, 'text': text, 'error': rec.get('error')})
    return records


def compute_ba(pred_text, ground_truth):
    """Parse prediction array and compute balanced accuracy."""
    text = pred_text.strip()
    if not text.startswith('['):
        text = '[' + text
    match = re.search(r'\[[\d\s,]+\]', text)
    if not match:
        return None
    try:
        preds = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if len(preds) != len(ground_truth):
        return None
    tp = sum(1 for p, g in zip(preds, ground_truth) if p == 1 and g == 1)
    tn = sum(1 for p, g in zip(preds, ground_truth) if p == 0 and g == 0)
    pos = sum(ground_truth)
    neg = len(ground_truth) - pos
    tpr = tp / pos if pos > 0 else 0
    tnr = tn / neg if neg > 0 else 0
    return (tpr + tnr) / 2


def cmd_labels(args):
    """Parse labeling batch results."""
    s3_key = get_job_output_key(args.job_id)
    if not s3_key:
        print(f"No output found for job {args.job_id}")
        return

    records = download_and_parse(s3_key)
    results = []
    errors = 0
    for rec in records:
        if rec['error'] or not rec['text']:
            errors += 1
            continue
        try:
            parsed = json.loads(rec['text'])
            fid = int(rec['id'].split('feature_')[1])
            results.append({'id': fid, 'data': parsed})
        except (json.JSONDecodeError, IndexError, ValueError):
            errors += 1

    print(f"Total: {len(records)}, Parsed: {len(results)}, Errors: {errors}")

    # Stats
    poly = sum(1 for r in results if r['data'].get('polysemantic'))
    confs = {}
    cats = {}
    for r in results:
        c = r['data'].get('confidence', 'unknown')
        confs[c] = confs.get(c, 0) + 1
        cat = r['data'].get('category', 'unknown')
        cats[cat] = cats.get(cat, 0) + 1

    print(f"Polysemantic: {poly}/{len(results)} ({100*poly/max(1,len(results)):.1f}%)")
    print(f"Confidence: {confs}")
    print(f"\nTop categories:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")

    # Save clean labels
    if args.output:
        labels = {}
        for r in results:
            labels[str(r['id'])] = {
                'label': r['data'].get('label', ''),
                'category': r['data'].get('category', ''),
                'confidence': r['data'].get('confidence', ''),
                'polysemantic': r['data'].get('polysemantic', False),
                'chip': r['data'].get('chip', ''),
                'explanation': r['data'].get('explanation', ''),
            }
        with open(args.output, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"\nSaved {len(labels)} labels to {args.output}")


def cmd_detect(args):
    """Parse detection scoring results and compute BA."""
    s3_key = get_job_output_key(args.job_id)
    if not s3_key:
        print(f"No output found for job {args.job_id}")
        return

    with open(args.gt_file) as f:
        gt_data = json.load(f)

    records = download_and_parse(s3_key)
    bas = []
    failed = 0
    for rec in records:
        if rec['id'] in gt_data and rec['text']:
            gt = gt_data[rec['id']]['ground_truth']
            ba = compute_ba(rec['text'], gt)
            if ba is not None:
                bas.append(ba)
            else:
                failed += 1
        else:
            failed += 1

    if bas:
        mean_ba = sum(bas) / len(bas)
        holds = sum(1 for ba in bas if ba >= 0.7)
        strong = sum(1 for ba in bas if ba >= 0.8)
        weak = sum(1 for ba in bas if 0.5 <= ba < 0.7)
        fail = sum(1 for ba in bas if ba < 0.5)
        top200 = sorted(bas, reverse=True)[:200]
        top200_mean = sum(top200) / len(top200)

        print(f"Parsed: {len(bas)}, Failed: {failed}")
        print(f"Mean BA: {mean_ba:.3f}")
        print(f"Top-200 BA: {top200_mean:.3f}")
        print(f"HOLDS(>=0.7): {holds}, STRONG(>=0.8): {strong}")
        print(f"WEAK(0.5-0.7): {weak}, FAIL(<0.5): {fail}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_labels = sub.add_parser('labels')
    p_labels.add_argument('--job-id', required=True)
    p_labels.add_argument('--output', help='Save clean labels JSON')

    p_detect = sub.add_parser('detect')
    p_detect.add_argument('--job-id', required=True)
    p_detect.add_argument('--gt-file', required=True, help='Ground truth JSON')

    args = parser.parse_args()
    if args.cmd == 'labels':
        cmd_labels(args)
    elif args.cmd == 'detect':
        cmd_detect(args)
    else:
        parser.print_help()
