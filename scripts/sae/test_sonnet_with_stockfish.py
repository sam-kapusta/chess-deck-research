"""Test: re-label 10 low-confidence features using Sonnet + Stockfish data."""
import json
import time
import re
import boto3

PROFILES_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json"
SF_DATA_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/stockfish_data.json"
LABELS_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_labels_sonnet.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/sonnet_sf_test_10.json"

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a chess grandmaster analyzing SAE features. Each feature fires on positions where a player made a specific type of mistake.

You are given Stockfish analysis for each position: the best move, the refutation after the blunder, eval change, and top lines. Use this to identify the SPECIFIC tactical mechanism that unites these positions.

Focus on the MECHANISM (pin break, interposition removal, file opening) not just the symptom (piece hangs)."""

CATEGORIES = [
    "hanging_pieces", "overloaded_defenders", "forks", "pins", "skewers",
    "discovered_attacks", "back_rank", "king_safety", "passed_pawns",
    "rook_endgames", "pawn_endgames", "checkmate_patterns", "quiet_moves",
    "trapped_pieces", "sacrifice", "interposition_break", "file_opening",
    "diagonal_opening", "removed_defender", "other_tactics",
]


def build_prompt(feat_id, examples_with_sf):
    positions_text = ""
    for i, (ex, sf) in enumerate(examples_with_sf[:10]):
        positions_text += f"\n{i+1}. FEN: {ex['fen']}\n"
        positions_text += f"   Blunder played: {sf.get('played_san', ex.get('uci', '?'))}\n"
        positions_text += f"   Best move was: {sf.get('best_san', '?')}\n"
        positions_text += f"   Eval: {sf.get('eval_before', '?')} → {sf.get('eval_after', '?')} (lost {sf.get('cp_loss', '?')} cp)\n"
        positions_text += f"   Phase: {sf.get('phase', '?')}, Side: {sf.get('side_to_move', '?')}\n"
        threat = sf.get('threat', '')
        if threat:
            positions_text += f"   Immediate threat after blunder: {threat}\n"
        refutation = sf.get('refutation_lines', [])
        if refutation and refutation[0].get('moves'):
            positions_text += f"   Refutation: {' '.join(refutation[0]['moves'][:5])}\n"
        top = sf.get('top_lines', [])
        if top and top[0].get('moves'):
            positions_text += f"   Best continuation was: {' '.join(top[0]['moves'][:5])}\n"

    return f"""Analyze SAE feature #{feat_id}. This feature fires on the following blunder positions.

For each position you have the Stockfish analysis showing exactly what went wrong.

=== POSITIONS WITH STOCKFISH ANALYSIS ===
{positions_text}

=== TASK ===
Identify the SPECIFIC tactical MECHANISM that connects all these positions.
Not "piece hangs" (that's a symptom). What is the geometric/tactical pattern?

Examples of good mechanism labels:
- "Bishop abandons interposition on open file"
- "Knight move self-discovers attack on own queen"
- "Pawn advance removes key defender of back rank"
- "Piece steps onto square controlled by pinned piece"

Respond in JSON:
{{
  "specific_label": "<2-5 words: the mechanism>",
  "primary_category": "<one of: {', '.join(CATEGORIES)}>",
  "piece_involved": "<pawn/knight/bishop/rook/queen/king/mixed>",
  "game_phase": "<opening/middlegame/endgame/all_phases>",
  "explanation": "<1-2 sentences explaining the geometric pattern>",
  "confidence": "<high/medium/low>",
  "confidence_score": <0.0-1.0>
}}"""


def call_sonnet(prompt, client):
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 4096},
        additionalModelRequestFields={"thinking": {"type": "enabled", "budget_tokens": 4000}},
    )
    for block in response["output"]["message"]["content"]:
        if block.get("text"):
            return block["text"]
    return ""


def parse_response(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def main():
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)
    with open(SF_DATA_PATH) as f:
        sf_data = json.load(f)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    # Find 10 low-confidence features with good SF coverage
    candidates = []
    for fid, label in labels.items():
        if label.get("confidence_score", 1) < 0.65:
            prof = profiles.get(fid, {})
            examples = prof.get("examples", [])[:10]
            examples_with_sf = []
            for ex in examples:
                key = ex.get("fen", "") + "|" + ex.get("uci", "")
                if key in sf_data:
                    examples_with_sf.append((ex, sf_data[key]))
            if len(examples_with_sf) >= 8:
                candidates.append((fid, label, examples_with_sf))

    candidates.sort(key=lambda x: x[1].get("confidence_score", 1))
    test_features = candidates[:10]

    print(f"Testing 10 low-confidence features with Stockfish enrichment:")
    for fid, label, _ in test_features:
        print(f"  F{fid}: '{label['specific_label']}' (conf={label['confidence_score']})")

    # Label them
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    results = {}

    for fid, old_label, examples_with_sf in test_features:
        prompt = build_prompt(fid, examples_with_sf)
        try:
            text = call_sonnet(prompt, client)
            parsed = parse_response(text)
            if parsed:
                results[fid] = {
                    "old_label": old_label["specific_label"],
                    "old_confidence": old_label["confidence_score"],
                    "new_label": parsed.get("specific_label", "?"),
                    "new_confidence": parsed.get("confidence_score", 0),
                    "new_category": parsed.get("primary_category", "?"),
                    "explanation": parsed.get("explanation", ""),
                }
                print(f"\n  F{fid}:")
                print(f"    OLD: '{old_label['specific_label']}' (conf={old_label['confidence_score']})")
                print(f"    NEW: '{parsed['specific_label']}' (conf={parsed.get('confidence_score', '?')})")
                print(f"    WHY: {parsed.get('explanation', '')[:100]}")
        except Exception as e:
            print(f"  F{fid}: ERROR — {e}")
        time.sleep(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
