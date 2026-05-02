#!/usr/bin/env python3
"""Step 3: Submit batch job to Gemini and collect results.

Uploads batch_input.jsonl, submits to Gemini 3.1 Pro batch,
polls until complete, downloads results to position_analyses.json.

Usage:
    python3 submit_batch.py --api-key YOUR_KEY [--submit | --poll JOB_NAME | --download JOB_NAME]
"""
import argparse
import json
import time
import sys

API_KEY = 'AIzaSyBxQ9k-yjJ-zRKleN4pIeVEM8yq3IfDSX0'
MODEL = 'gemini-3.1-pro-preview'
BATCH_INPUT = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/batch_input.jsonl'
OUTPUT_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/position_analyses.json'


def submit(client):
    """Upload JSONL and create batch job."""
    from google.genai import types

    print(f"Uploading {BATCH_INPUT}...", flush=True)
    uploaded = client.files.upload(
        file=BATCH_INPUT,
        config=types.UploadFileConfig(
            display_name='chess-position-analysis-batch',
            mime_type='jsonl'
        )
    )
    print(f"Uploaded: {uploaded.name}", flush=True)

    print(f"Creating batch job on {MODEL}...", flush=True)
    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={'display_name': 'chess-sae-position-analysis'}
    )
    print(f"Job created: {job.name}", flush=True)
    print(f"State: {job.state.name}", flush=True)
    print(f"\nTo check status: python3 submit_batch.py --poll {job.name}", flush=True)
    return job.name


def poll(client, job_name):
    """Check batch job status."""
    completed = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    job = client.batches.get(name=job_name)
    print(f"Job: {job_name}", flush=True)
    print(f"State: {job.state.name}", flush=True)

    if job.state.name in completed:
        print(f"Job finished: {job.state.name}", flush=True)
        if job.state.name == 'JOB_STATE_SUCCEEDED':
            print(f"\nTo download: python3 submit_batch.py --download {job_name}", flush=True)
        return job.state.name

    print("Still running. Polling every 60s...", flush=True)
    while job.state.name not in completed:
        time.sleep(60)
        job = client.batches.get(name=job_name)
        print(f"  {job.state.name}", flush=True)

    print(f"\nJob finished: {job.state.name}", flush=True)
    return job.state.name


def download(client, job_name):
    """Download batch results and convert to position_analyses.json."""
    job = client.batches.get(name=job_name)

    if job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job state is {job.state.name}, cannot download.", flush=True)
        return

    # Try file-based output
    if hasattr(job, 'dest') and job.dest:
        if hasattr(job.dest, 'file_name') and job.dest.file_name:
            print(f"Downloading results from {job.dest.file_name}...", flush=True)
            content = client.files.download(file=job.dest.file_name)
            raw = content.decode('utf-8')
        elif hasattr(job.dest, 'inlined_responses') and job.dest.inlined_responses:
            print("Reading inline responses...", flush=True)
            results = {}
            for resp in job.dest.inlined_responses:
                if resp.response and resp.response.text:
                    # Parse the JSON response
                    try:
                        analysis = json.loads(resp.response.text)
                        results[resp.key] = analysis
                    except json.JSONDecodeError:
                        results[resp.key] = {'raw': resp.response.text}
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} analyses to {OUTPUT_PATH}", flush=True)
            return

    # Parse JSONL output
    results = {}
    errors = 0
    for line in raw.strip().split('\n'):
        try:
            entry = json.loads(line)
            custom_id = entry.get('custom_id', '')
            response_text = ''

            # Extract text from response
            if 'response' in entry:
                resp = entry['response']
                if isinstance(resp, dict):
                    candidates = resp.get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts:
                            response_text = parts[0].get('text', '')

            if response_text:
                # Try to parse as JSON
                try:
                    # Strip markdown code fences if present
                    clean = response_text.strip()
                    if clean.startswith('```'):
                        clean = clean.split('\n', 1)[1]
                        clean = clean.rsplit('```', 1)[0]
                    analysis = json.loads(clean)
                    results[custom_id] = analysis
                except json.JSONDecodeError:
                    results[custom_id] = {'raw': response_text}
            else:
                errors += 1
        except Exception as e:
            errors += 1

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} analyses to {OUTPUT_PATH} ({errors} errors)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=API_KEY)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--poll', type=str, metavar='JOB_NAME')
    parser.add_argument('--download', type=str, metavar='JOB_NAME')
    args = parser.parse_args()

    from google import genai
    client = genai.Client(api_key=args.api_key)

    if args.submit:
        submit(client)
    elif args.poll:
        poll(client, args.poll)
    elif args.download:
        download(client, args.download)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
