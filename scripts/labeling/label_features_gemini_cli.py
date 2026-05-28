#!/usr/bin/env python3
"""Label SAE features using Gemini CLI (free via subscription).

Sends top-10 positions per feature in one request, asks Gemini to identify
the common tactical pattern. Uses -p (non-interactive) mode.

Usage:
    python scripts/labeling/label_features_gemini_cli.py --limit 5    # test
    python scripts/labeling/label_features_gemini_cli.py              # full run
    python scripts/labeling/label_features_gemini_cli.py --resume     # continue
"""
import argparse
import json
import subprocess
import time
import sys
import os

FEATURES_PATH = "/tmp/l2_feature_profiles_v2.json"
OUTPUT_PATH = "output/maia3_feature_labels_gemini_v2.json"
MODEL = "gemini-3.1-pro-preview"

PROMPT_TEMPLATE = """IMPORTANT: Do NOT use any tools. Do NOT run code or call APIs. Just analyze the positions mentally and respond with JSON only.

You are a chess grandmaster analyzing SAE (Sparse Autoencoder) features.
Each feature fires on positions where a player made a specific type of mistake.

Below are the top {n} positions where this feature activates most strongly.
All positions are blunders. Find the SPECIFIC tactical or positional pattern that connects them.

{positions_text}

Respond ONLY with this JSON (no other text):
{{
  "specific_label": "<2-5 word mechanism description>",
  "primary_category": "<one of: hanging_piece, fork, pin, skewer, discovered_attack, back_rank, overloaded_defender, trapped_piece, pawn_endgame, rook_endgame, king_safety, passed_pawn, promotion_error, tempo_loss, positional_mistake, other>",
  "piece_involved": "<pawn/knight/bishop/rook/queen/king/mixed>",
  "game_phase": "<opening/middlegame/endgame/mixed>",
  "explanation": "<1-2 sentences explaining the geometric/tactical pattern>",
  "confidence": "<high/medium/low>"
}}"""


def build_prompt(feature_id, examples):
    positions_text = ""
    for i, ex in enumerate(examples[:10]):
        positions_text += f"\n{i+1}. FEN: {ex['fen']}\n"
        positions_text += f"   Move played: {ex['uci']}\n"
        positions_text += f"   Centipawn loss: {ex.get('cp_loss', '?')}\n"

    return PROMPT_TEMPLATE.format(n=len(examples[:10]), positions_text=positions_text)


def call_gemini_cli(prompt, model=MODEL):
    """Call agy CLI in non-interactive mode, return text output."""
    agy_path = os.path.expanduser("~/.local/bin/agy")
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/share/mise/installs/node/22.14.0/bin") + ":" + env.get("PATH", "")
    result = subprocess.run(
        [agy_path, "--print-timeout", "5m", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=600,
        cwd="/tmp",
        
    )
    output = result.stdout.strip()
    if not output and result.returncode != 0:
        err = result.stderr[:300]
        if 'quota' in err.lower() or 'exhausted' in err.lower() or '429' in err:
            raise RuntimeError(f"QUOTA_EXHAUSTED: {err}")
        raise RuntimeError(f"agy CLI error: {err}")
    if not output:
        raise RuntimeError(f"agy CLI empty response, stderr: {result.stderr[:200]}")
    return output


def parse_response(text):
    """Extract JSON from gemini response."""
    # Try direct parse
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', default=FEATURES_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--shard', type=str, default=None, help='Shard N/M (e.g., 1/3, 2/3, 3/3)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.profiles) as f:
        profiles = json.load(f)

    # Skip hub features (the 41 that were excluded from labeling)
    feature_ids = sorted(profiles.keys(), key=int)

    # Shard support: split features across parallel workers
    if args.shard:
        n, m = map(int, args.shard.split('/'))
        chunk_size = len(feature_ids) // m
        start = (n - 1) * chunk_size
        end = start + chunk_size if n < m else len(feature_ids)
        feature_ids = feature_ids[start:end]
        # Shard-specific output file
        if args.output == OUTPUT_PATH:
            args.output = OUTPUT_PATH.replace('.json', f'_shard{n}.json')

    if args.limit:
        feature_ids = feature_ids[:args.limit]

    # Resume
    results = {}
    if args.resume:
        try:
            with open(args.output) as f:
                results = json.load(f)
            print(f"Resuming from {len(results)} existing labels")
        except FileNotFoundError:
            pass

    n_done = len(results)
    n_errors = 0
    consecutive_errors = 0
    t0 = time.time()

    print(f"Features to label: {len(feature_ids)} (skipping {len(results)} already done)")

    for i, fid in enumerate(feature_ids):
        if fid in results:
            continue

        examples = profiles[fid].get('examples', [])
        if len(examples) < 3:
            results[fid] = {'error': 'insufficient_examples'}
            continue

        prompt = build_prompt(fid, examples)

        try:
            raw = call_gemini_cli(prompt, model=args.model)
            parsed = parse_response(raw)

            if parsed:
                parsed['source'] = 'agy'
                results[fid] = parsed
                n_done += 1
                consecutive_errors = 0
            else:
                results[fid] = {'error': 'parse_failed', 'raw': raw[:500]}
                n_errors += 1
                consecutive_errors += 1

        except subprocess.TimeoutExpired:
            results[fid] = {'error': 'timeout'}
            n_errors += 1
            consecutive_errors += 1
        except Exception as e:
            err = str(e)
            if 'QUOTA_EXHAUSTED' in err or '429' in err or 'quota' in err.lower():
                print(f"  QUOTA HIT at {n_done}. Saving and stopping.", flush=True)
                break
            results[fid] = {'error': err[:200]}
            n_errors += 1
            consecutive_errors += 1

        # Circuit breaker: 3 consecutive errors = stop
        if consecutive_errors >= 3:
            print(f"  CIRCUIT BREAKER: 3 consecutive errors. Stopping.", flush=True)
            break

        # Progress
        if (n_done - len([r for r in results.values() if 'error' not in r])) % 10 == 0 or n_done <= 5:
            elapsed = time.time() - t0
            rate = max(n_done, 1) / max(elapsed, 1) * 60
            remaining = len(feature_ids) - i
            print(f"  F{fid}: done={n_done}, errors={n_errors}, rate={rate:.1f}/min, remaining={remaining}", flush=True)

        # Save after every feature
        with open(args.output, 'w') as f:
            json.dump(results, f)

        # Small delay to avoid hammering
        time.sleep(0.5)

    # Final save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    valid = sum(1 for v in results.values() if 'error' not in v)
    print(f"\nDone. {valid} labeled, {n_errors} errors, {elapsed/60:.1f}min")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
