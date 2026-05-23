#!/usr/bin/env python3
"""Build Gemini Bedrock Batch input for all Maia 3 SAE top positions.

Takes top-20 positions per feature, deduplicates, builds JSONL for
Bedrock Batch (Gemini 2.5 Flash). Each position gets tactical analysis:
intent, blunder trace, point of failure.

Usage:
    python scripts/sae/build_gemini_batch_maia3.py
    python scripts/sae/build_gemini_batch_maia3.py --top-n 10  # fewer per feature
"""
import argparse
import json
import time

PROFILES_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json"
OUTPUT_JSONL = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_batch_input.jsonl"
MAPPING_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_id_mapping.json"

SYSTEM_PROMPT = """You are a chess grandmaster analyzing blunder positions. For each position, explain:
1. What the player was trying to do (their intent)
2. What goes wrong after this move (the refutation/punishment)
3. The specific point of failure (which square, piece, or tactical motif was missed)

Be concrete and specific. Name squares, pieces, and tactical patterns."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {
            "type": "STRING",
            "description": "What the player was trying to achieve with this move (1-2 sentences)"
        },
        "blunder_trace": {
            "type": "STRING",
            "description": "The refutation: what the opponent does and why it works (2-3 sentences with specific moves)"
        },
        "point_of_failure": {
            "type": "STRING",
            "description": "The specific tactical/positional element the player missed (1 sentence, name the motif)"
        },
        "tactical_motif": {
            "type": "STRING",
            "enum": [
                "hanging_piece", "fork", "pin", "skewer", "discovered_attack",
                "back_rank_mate", "smothered_mate", "overloaded_defender",
                "deflection", "decoy", "interference", "trapped_piece",
                "pawn_promotion", "zugzwang", "stalemate_blunder",
                "king_exposure", "file_opening", "diagonal_opening",
                "removed_defender", "interposition_break", "pawn_weakness",
                "endgame_technique", "other"
            ]
        },
        "severity": {
            "type": "STRING",
            "enum": ["tactical_loss", "positional_loss", "missed_win", "drawn_to_lost"]
        }
    },
    "required": ["intent", "blunder_trace", "point_of_failure", "tactical_motif", "severity"]
}


def build_prompt(fen, uci, cp_loss):
    return f"""Analyze this chess blunder:

Position (FEN): {fen}
Move played: {uci}
Centipawn loss: {cp_loss}

This move was a blunder that lost {cp_loss} centipawns of evaluation. Explain what happened."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20, help="Positions per feature")
    parser.add_argument("--profiles", default=PROFILES_PATH)
    parser.add_argument("--output", default=OUTPUT_JSONL)
    parser.add_argument("--mapping", default=MAPPING_PATH)
    args = parser.parse_args()

    print(f"Loading profiles from {args.profiles}...")
    with open(args.profiles) as f:
        profiles = json.load(f)

    # Collect unique positions across all features
    positions = {}  # fen -> {uci, cp_loss, feature_ids}
    for fid, prof in profiles.items():
        examples = prof.get("examples", [])[:args.top_n]
        for ex in examples:
            fen = ex.get("fen", "")
            if not fen:
                continue
            if fen not in positions:
                positions[fen] = {
                    "fen": fen,
                    "uci": ex.get("uci", ""),
                    "cp_loss": ex.get("cp_loss", 0),
                    "feature_ids": [],
                }
            positions[fen]["feature_ids"].append(int(fid))

    print(f"  Unique positions: {len(positions)} (from {len(profiles)} features × top-{args.top_n})")

    # Build JSONL records (Gemini batch format — matches submit_batch.py)
    id_mapping = {}
    records = []
    for i, (fen, pos) in enumerate(positions.items()):
        record_id = f"pos_{i:05d}"
        id_mapping[record_id] = {"fen": fen, "uci": pos["uci"], "cp_loss": pos["cp_loss"]}

        prompt = build_prompt(fen, pos["uci"], pos["cp_loss"])
        record = {
            "custom_id": record_id,
            "request": {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": RESPONSE_SCHEMA,
                },
            },
        }
        records.append(record)

    # Write JSONL
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records to {args.output}")

    # Write mapping
    with open(args.mapping, "w") as f:
        json.dump(id_mapping, f)
    print(f"  Wrote ID mapping to {args.mapping}")

    # Stats
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n  File size: {size_mb:.1f} MB")
    print(f"  Estimated cost (Gemini Flash): ~${len(records) * 0.001:.0f}")
    print(f"  Estimated time: ~{len(records) / 1000:.0f} minutes in Bedrock Batch")


if __name__ == "__main__":
    main()
