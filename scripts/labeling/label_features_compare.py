#!/usr/bin/env python3
"""Compare two labeling approaches on 5 features:
  A) Feature-level only (10 positions → 1 label)
  B) Position-level + feature-level (10 positions → 10 labels + 1 synthesis)

Run: python scripts/labeling/label_features_compare.py
"""
import json
import subprocess
import time

FEATURES_PATH = "/tmp/l2_feature_profiles.json"
OUTPUT_PATH = "output/label_comparison_5.json"
MODEL = "gemini-3.1-pro-preview"

# Pick 5 diverse features (high confidence from Sonnet for comparison)
TEST_FEATURES = ["1769", "1597", "1495", "1802", "841"]

PROMPT_A = """You are a chess grandmaster analyzing SAE features.
Each feature fires on positions where a player made a specific type of mistake.

Below are the top 10 positions where this feature activates most strongly.
All are blunders (suboptimal moves). Find the SPECIFIC pattern connecting them.

{positions_text}

Respond in JSON:
{{
  "specific_label": "<2-5 word mechanism description>",
  "primary_category": "<hanging_piece|fork|pin|skewer|discovered_attack|back_rank|overloaded_defender|trapped_piece|pawn_endgame|rook_endgame|king_safety|passed_pawn|promotion_error|tempo_loss|positional_mistake|other>",
  "piece_involved": "<pawn/knight/bishop/rook/queen/king/mixed>",
  "explanation": "<1-2 sentences>",
  "confidence": "<high/medium/low>"
}}"""

PROMPT_B = """You are a chess grandmaster analyzing SAE features.
Each feature fires on positions where a player made a specific type of mistake.

Below are the top 10 positions where this feature activates most strongly.
All are blunders (suboptimal moves).

For EACH position, explain the blunder. Then synthesize a feature-level label.

{positions_text}

Respond in JSON:
{{
  "positions": [
    {{
      "position_number": 1,
      "intent": "<what player was trying to do>",
      "blunder_explanation": "<what goes wrong, name specific moves/squares>",
      "tactical_motif": "<the specific tactic missed>"
    }}
  ],
  "feature_label": {{
    "specific_label": "<2-5 word mechanism description>",
    "primary_category": "<hanging_piece|fork|pin|skewer|discovered_attack|back_rank|overloaded_defender|trapped_piece|pawn_endgame|rook_endgame|king_safety|passed_pawn|promotion_error|tempo_loss|positional_mistake|other>",
    "piece_involved": "<pawn/knight/bishop/rook/queen/king/mixed>",
    "explanation": "<1-2 sentences synthesizing the common pattern>",
    "confidence": "<high/medium/low>"
  }}
}}"""


def build_positions_text(examples):
    text = ""
    for i, ex in enumerate(examples[:10]):
        text += f"\n{i+1}. FEN: {ex['fen']}\n"
        text += f"   Move played: {ex['uci']}\n"
        text += f"   Centipawn loss: {ex.get('cp_loss', '?')}\n"
    return text


def call_gemini(prompt):
    import os
    env = os.environ.copy()
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    result = subprocess.run(
        ["gemini", "-m", MODEL, "-p", prompt, "-o", "text"],
        capture_output=True, text=True, timeout=180,
        env=env, cwd="/tmp",
    )
    if result.returncode != 0:
        return f"ERROR: {result.stderr[:300]}"
    return result.stdout.strip()


def main():
    with open(FEATURES_PATH) as f:
        profiles = json.load(f)

    results = {}

    for fid in TEST_FEATURES:
        prof = profiles.get(fid, {})
        examples = prof.get('examples', [])[:10]
        positions_text = build_positions_text(examples)

        print(f"\n=== Feature {fid} ===", flush=True)

        # Approach A: feature-level only
        print(f"  Running approach A (feature-only)...", flush=True)
        prompt_a = PROMPT_A.format(positions_text=positions_text)
        raw_a = call_gemini(prompt_a)
        time.sleep(2)

        # Approach B: position-level + feature-level
        print(f"  Running approach B (position + feature)...", flush=True)
        prompt_b = PROMPT_B.format(positions_text=positions_text)
        raw_b = call_gemini(prompt_b)
        time.sleep(2)

        results[fid] = {
            'approach_a_raw': raw_a,
            'approach_b_raw': raw_b,
        }

        print(f"  A: {raw_a[:150]}...", flush=True)
        print(f"  B: {raw_b[:150]}...", flush=True)

        # Save after each feature
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved comparison to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
