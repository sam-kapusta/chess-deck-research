"""Label individual positions with Sonnet + Stockfish data.

Per-position tactical analysis (not per-feature). Each position gets:
intent, blunder mechanism, point of failure, tactical motif.

Usage:
    python scripts/sae/label_positions_sonnet.py --limit 10  # test
    python scripts/sae/label_positions_sonnet.py --threads 5  # full run
"""
import argparse
import json
import time
import re
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

SF_DATA_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/stockfish_data.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/position_labels_sonnet.json"

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a chess grandmaster explaining why a specific move was a mistake.
You have the Stockfish analysis showing the best move, the evaluation change, and the refutation.
Your job: explain the tactical MECHANISM in precise geometric terms.

Be specific about squares, files, diagonals, and piece relationships.
Not "piece hangs" — say WHY it hangs (what line opened, what defender left, what fork became possible)."""


def build_prompt(sf):
    played = sf.get("played_san", sf.get("uci", "?"))
    best = sf.get("best_san", "?")
    eval_before = sf.get("eval_before", "?")
    eval_after = sf.get("eval_after", "?")
    phase = sf.get("phase", "?")
    side = sf.get("side_to_move", "?")
    threat = sf.get("threat", "")

    refut = sf.get("refutation_lines", [])
    refut_text = ""
    if refut and refut[0].get("moves"):
        refut_text = " ".join(refut[0]["moves"][:6])

    top = sf.get("top_lines", [])
    best_line = ""
    if top and top[0].get("moves"):
        best_line = " ".join(top[0]["moves"][:6])

    return f"""Position (FEN): {sf.get("fen", "?")}
Side to move: {side} | Phase: {phase}
Move played: {played} (eval: {eval_before} → {eval_after}, lost {sf.get("cp_loss", "?")} cp)
Best move was: {best} (line: {best_line})
After the blunder, opponent's threat: {threat}
Refutation line: {refut_text}

Explain in JSON:
{{
  "intent": "<what the player was trying to do, 1 sentence>",
  "mechanism": "<the geometric/tactical reason it fails, 1-2 sentences, name squares and pieces>",
  "motif": "<one of: hanging_piece, fork, pin, skewer, discovered_attack, back_rank, overloaded_defender, removed_guard, interposition_break, trapped_piece, promotion_error, king_exposure, tempo_loss, pawn_structure, endgame_technique, stalemate, other>"
}}"""


def call_sonnet(prompt, client):
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 1024},
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-data", default=SF_DATA_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.sf_data) as f:
        sf_data = json.load(f)

    # Filter to good entries only
    positions = {k: v for k, v in sf_data.items() if "error" not in v}
    print(f"Positions with Stockfish data: {len(positions)}")

    # Resume
    existing = {}
    if args.resume:
        try:
            with open(args.output) as f:
                existing = json.load(f)
            print(f"Resuming from {len(existing)} existing")
        except:
            pass

    todo = {k: v for k, v in positions.items() if k not in existing}
    if args.limit:
        todo = dict(list(todo.items())[:args.limit])
    print(f"To label: {len(todo)}")

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    results = {**existing}
    n_success = 0
    n_fail = 0
    t0 = time.time()

    def invoke_one(key, sf):
        prompt = build_prompt(sf)
        try:
            text = call_sonnet(prompt, client)
            parsed = parse_response(text)
            if parsed:
                return key, parsed
            return key, None
        except Exception as e:
            return key, None

    items = list(todo.items())
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(invoke_one, k, v): k for k, v in items}
        for i, future in enumerate(as_completed(futures)):
            key, result = future.result()
            if result:
                results[key] = result
                n_success += 1
            else:
                n_fail += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(items)} ({rate:.1f}/s, {n_fail} failures)", flush=True)
                with open(args.output, "w") as f:
                    json.dump(results, f)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {n_success} labeled, {n_fail} failed, {elapsed:.0f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
