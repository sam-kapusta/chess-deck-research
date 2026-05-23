#!/usr/bin/env python3
"""Label Maia 3 SAE features using Sonnet 4.6 with thinking.

Calls Bedrock converse() directly (not batch). Shows top-20 positions per feature
regardless of relevance threshold. LLM provides confidence score.

Usage:
    python scripts/sae/label_maia3_features.py --limit 50
    python scripts/sae/label_maia3_features.py --features 500,830,1536
"""
import argparse
import json
import os
import sys
import time
import re

import boto3

PROFILE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_labels_sonnet.json"

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a chess pattern recognition expert. You analyze SAE (Sparse Autoencoder) features trained on blunder positions from chess games.

Each feature fires on a set of positions where a player made a mistake. Your job is to identify the SPECIFIC tactical pattern that unites these positions.

Think carefully about what connects ALL the positions, not just some of them. The pattern should explain why these moves were mistakes."""

CATEGORIES = [
    "hanging_pieces", "overloaded_defenders", "forks", "pins", "skewers",
    "discovered_attacks", "back_rank", "king_safety", "passed_pawns",
    "rook_endgames", "pawn_endgames", "checkmate_patterns", "quiet_moves",
    "trapped_pieces", "sacrifice", "other_tactics",
]


def build_prompt(feat_id, examples):
    positions_text = ""
    for i, ex in enumerate(examples[:20]):
        rel = ex.get("relevance", 0)
        fen = ex.get("fen", "")
        uci = ex.get("uci", "")
        cp = ex.get("cp_loss", 0)
        strength = ex.get("strength", ex.get("activation", 0))
        positions_text += f"{i+1}. FEN: {fen}\n"
        positions_text += f"   Blunder move: {uci}  |  cp_loss: {cp}  |  relevance: {rel:.2f}\n\n"

    return f"""Analyze SAE feature #{feat_id}. This feature fires on the following chess positions (all are blunder moves that lost evaluation).

The positions are ordered by activation strength (strongest match first). "relevance" is normalized 0-1 where 1.0 = strongest activation this feature produces.

=== POSITIONS ===
{positions_text}

=== TASK ===
1. Identify what SPECIFIC tactical pattern connects these positions
2. Name the pattern in 2-5 words
3. Categorize it
4. Rate your confidence

Respond in this exact JSON format:
{{
  "specific_label": "<2-5 words: the specific pattern>",
  "primary_category": "<one of: {', '.join(CATEGORIES)}>",
  "piece_involved": "<pawn/knight/bishop/rook/queen/king/mixed>",
  "game_phase": "<opening/middlegame/endgame/all_phases>",
  "explanation": "<1-2 sentences: what makes this pattern specific>",
  "confidence": "<high/medium/low>",
  "confidence_score": <0.0-1.0 numeric>
}}"""


def call_sonnet(prompt, client):
    """Call Sonnet 4.6 with thinking enabled."""
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 4096},
        additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": 4000}},
    )
    # Extract text from response (skip thinking blocks)
    text = ""
    for block in response["output"]["message"]["content"]:
        if block.get("text"):
            text = block["text"]
    return text


def parse_response(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default=PROFILE_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Max features to label")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature IDs")
    parser.add_argument("--min-fire-rate", type=float, default=0.001, help="Min fire rate")
    parser.add_argument("--max-fire-rate", type=float, default=0.05, help="Max fire rate (exclude hubs)")
    parser.add_argument("--threads", type=int, default=5, help="Parallel threads for Bedrock calls")
    args = parser.parse_args()

    print(f"Loading profiles from {args.profiles}...")
    with open(args.profiles) as f:
        profiles = json.load(f)

    # Fix format if needed
    for fid, p in profiles.items():
        if "top_examples" in p:
            p["examples"] = p.pop("top_examples")
        for ex in p.get("examples", []):
            if "activation" in ex and "strength" not in ex:
                ex["strength"] = ex["activation"]

    # Select features to label
    if args.features:
        feature_ids = [f.strip() for f in args.features.split(",")]
    else:
        # Filter by fire rate (skip hubs and near-dead)
        feature_ids = [
            fid for fid, p in profiles.items()
            if args.min_fire_rate <= p.get("fire_rate", 0) <= args.max_fire_rate
            and len(p.get("examples", [])) >= 10
        ]
        # Sort by fire rate (most specific first)
        feature_ids.sort(key=lambda fid: profiles[fid]["fire_rate"])

    if args.limit:
        feature_ids = feature_ids[:args.limit]

    print(f"Labeling {len(feature_ids)} features with Sonnet 4.6 + thinking...")

    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Load existing labels if any
    labels = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            labels = json.load(f)
        print(f"  Loaded {len(labels)} existing labels")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_success = 0
    n_fail = 0
    t0 = time.time()

    # Filter out already-labeled
    to_label = [(fid, profiles[fid]) for fid in feature_ids if fid not in labels]
    print(f"  Skipping {len(feature_ids) - len(to_label)} already labeled")
    print(f"  Labeling {len(to_label)} features with {args.threads} threads...")

    def invoke_one(fid, p):
        examples = p.get("examples", [])
        prompt = build_prompt(fid, examples)
        try:
            response_text = call_sonnet(prompt, client)
            parsed = parse_response(response_text)
            if parsed:
                return fid, {
                    **parsed,
                    "feature_id": int(fid),
                    "fire_rate": p.get("fire_rate", 0),
                    "n_examples": len(examples),
                }
            return fid, None
        except Exception as e:
            print(f"  F{fid}: ERROR — {e}")
            return fid, None

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(invoke_one, fid, p): fid for fid, p in to_label}
        for i, future in enumerate(as_completed(futures)):
            fid, result = future.result()
            if result:
                labels[fid] = result
                label = result.get("specific_label", "?")
                conf = result.get("confidence_score", "?")
                print(f"  [{i+1}/{len(to_label)}] F{fid}: {label} (conf={conf})")
                n_success += 1
            else:
                n_fail += 1

            # Save every 20
            if (i + 1) % 20 == 0:
                with open(args.output, "w") as f:
                    json.dump(labels, f, indent=2)

    # Final save
    with open(args.output, "w") as f:
        json.dump(labels, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Success: {n_success}, Failed: {n_fail}")
    print(f"Total labels: {len(labels)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
