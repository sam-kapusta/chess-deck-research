"""Run Sonnet on the same 10 positions Gemini already labeled. Output side-by-side."""
import json
import re
import time
import chess
import chess.engine
import boto3

GEMINI_INPUT = "/home/ec2-user/SageMaker/gemini_vs_sonnet_10.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/side_by_side_10.json"
STOCKFISH = "/home/ec2-user/SageMaker/stockfish_compiled"
MODEL_ID = "global.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a chess coach translating engine analysis into plain English.

CRITICAL RULES:
- You MUST NOT calculate chess moves yourself. You CANNOT see the board.
- You MUST ONLY explain what the Stockfish lines already show.
- First, write the engine_trace field by copying the key facts from the Stockfish data.
- Then explain the mechanism using ONLY information from the engine lines.
- If you don't see a tactic in the engine lines, say "unclear" — do NOT invent one.
- Keep mechanism under 50 words. Only describe the immediate refutation.

You are a TRANSLATOR, not a chess engine."""


def analyze_position(engine, fen, uci):
    """Run Stockfish on one position."""
    board = chess.Board(fen)
    played = chess.Move.from_uci(uci)
    if played not in board.legal_moves:
        return {"error": "illegal move"}

    played_san = board.san(played)

    # Eval before + best
    results_before = engine.analyse(board, chess.engine.Limit(depth=14), multipv=3)
    r1 = results_before[0]
    eval_before = str(r1['score'].white())
    best_move = r1['pv'][0]
    best_san = board.san(best_move)
    top_line = []
    b = board.copy()
    for m in r1['pv'][:6]:
        try:
            top_line.append(b.san(m))
            b.push(m)
        except:
            break

    # Eval after + refutation
    board.push(played)
    results_after = engine.analyse(board, chess.engine.Limit(depth=14), multipv=1)
    r2 = results_after[0]
    eval_after = str(r2['score'].white())
    refut_line = []
    b = board.copy()
    for m in r2['pv'][:6]:
        try:
            refut_line.append(b.san(m))
            b.push(m)
        except:
            break
    threat = refut_line[0] if refut_line else ""
    board.pop()

    s1 = r1['score'].white().score(mate_score=10000)
    s2 = r2['score'].white().score(mate_score=10000)
    cp_loss = abs(s1 - s2) if s1 is not None and s2 is not None else 0

    return {
        "played_san": played_san,
        "best_san": best_san,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "cp_loss": cp_loss,
        "best_line": " ".join(top_line),
        "threat": threat,
        "refutation": " ".join(refut_line),
    }


def build_prompt(sf):
    return f"""STOCKFISH DATA:
- Move played: {sf["played_san"]}
- Best move was: {sf["best_san"]}
- Eval before: {sf["eval_before"]}, Eval after: {sf["eval_after"]} (lost {sf["cp_loss"]} cp)
- Best continuation: {sf["best_line"]}
- Opponent's threat after blunder: {sf["threat"]}
- Refutation line: {sf["refutation"]}

Respond in JSON:
{{
  "engine_trace": "<1 sentence: played X instead of Y, opponent punishes with Z>",
  "intent": "<what the player was trying to do>",
  "mechanism": "<why the refutation works — ONLY use moves from above, under 50 words>",
  "motif": "<hanging_piece|fork|pin|skewer|discovered_attack|back_rank|overloaded_defender|removed_guard|interposition_break|trapped_piece|promotion_error|king_exposure|tempo_loss|pawn_structure|endgame_technique|stalemate|other>"
}}"""


def call_sonnet(prompt, client):
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 512},
    )
    for block in response["output"]["message"]["content"]:
        if block.get("text"):
            return block["text"]
    return ""


def parse_response(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def main():
    with open(GEMINI_INPUT) as f:
        positions = json.load(f)

    print(f"Positions: {len(positions)}")

    # Run Stockfish
    print("Running Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    import os
    sf_dir = os.path.dirname(os.path.abspath(STOCKFISH))
    nnue = [f for f in os.listdir(sf_dir) if f.startswith('nn-') and f.endswith('.nnue')]
    big = [f for f in nnue if 'baff' not in f]
    small = [f for f in nnue if 'baff' in f]
    if big and small:
        engine.configure({"EvalFile": os.path.join(sf_dir, big[0]), "EvalFileSmall": os.path.join(sf_dir, small[0])})

    # Run Sonnet
    print("Running Sonnet...")
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    results = []
    for i, pos in enumerate(positions):
        fen = pos["fen"]
        uci = pos["uci"]

        # Stockfish
        sf = analyze_position(engine, fen, uci)
        if "error" in sf:
            print(f"  {i+1}: Stockfish error — {sf['error']}")
            continue

        # Sonnet
        prompt = build_prompt(sf)
        text = call_sonnet(prompt, bedrock)
        parsed = parse_response(text)

        results.append({
            "fen": fen,
            "uci": uci,
            "stockfish": sf,
            "gemini_original": {
                "intent": pos.get("gemini_intent", ""),
                "blunder": pos.get("gemini_blunder", ""),
                "failure": pos.get("gemini_failure", ""),
            },
            "sonnet_new": parsed or {"error": "parse failed"},
        })
        print(f"  {i+1}/{len(positions)}: done")
        time.sleep(0.5)

    engine.quit()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} side-by-side comparisons to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
