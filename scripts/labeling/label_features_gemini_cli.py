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

PROFILES_PATH = "output/maia3_positions_for_labeling.json"
FEATURES_PATH = "/tmp/l2_feature_profiles.json"
OUTPUT_PATH = "output/maia3_feature_labels_gemini.json"
MODEL = "gemini-3.1-pro-preview"

PROMPT_TEMPLATE = """You are a chess grandmaster analyzing SAE (Sparse Autoencoder) features.
Each feature fires on positions where a player made a specific type of mistake.

Below are the top {n} positions where this feature activates most strongly.
All positions are blunders. Find the SPECIFIC tactical or positional pattern that connects them.

{positions_text}

Respond in this exact JSON format:
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
    """Call gemini CLI in non-interactive mode, return text output."""
    env = os.environ.copy()
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    result = subprocess.run(
        ["gemini", "-m", model, "-p", prompt, "-o", "text"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd="/tmp",
    )
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI error: {result.stderr[:200]}")
    return result.stdout.strip()


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
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.profiles) as f:
        profiles = json.load(f)

    # Skip hub features (the 41 that were excluded from labeling)
    feature_ids = sorted(profiles.keys(), key=int)
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
                results[fid] = parsed
                n_done += 1
            else:
                results[fid] = {'error': 'parse_failed', 'raw': raw[:500]}
                n_errors += 1

        except subprocess.TimeoutExpired:
            results[fid] = {'error': 'timeout'}
            n_errors += 1
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                print(f"  Rate limited at {n_done}. Saving and exiting.", flush=True)
                break
            results[fid] = {'error': err[:200]}
            n_errors += 1

        # Progress
        if (n_done - len([r for r in results.values() if 'error' not in r])) % 10 == 0 or n_done <= 5:
            elapsed = time.time() - t0
            rate = max(n_done, 1) / max(elapsed, 1) * 60
            remaining = len(feature_ids) - i
            print(f"  F{fid}: done={n_done}, errors={n_errors}, rate={rate:.1f}/min, remaining={remaining}", flush=True)

        # Save every 25
        if n_done % 25 == 0:
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
