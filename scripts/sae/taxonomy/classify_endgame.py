"""Read each of the 133 endgame features (full chip + description) and assign a
coaching topic BY READING — Sonnet. Topics are endgame-technique lessons a coach
teaches. Explicit escape hatch: if a feature is NOT endgame technique (it's a
tactic / hang / bad trade that merely occurs with few pieces), it returns
reroute=<mistake-type> and leaves the Endgame category.

Usage:
    AWS_PROFILE=default python3 scripts/sae/taxonomy/classify_endgame.py
"""
import json, os, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.config import Config

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
IN = "/tmp/endgame_133.json"
OUT = "/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/taxonomy_v2/endgame_classified.json"
_lock = threading.Lock()

TOPICS = """ENDGAME-TECHNIQUE TOPICS (a coach's endgame curriculum):
- pawn_endgame: King & pawn technique — opposition, key squares, the right king path, pawn races (NO pieces other than K+P).
- rook_endgame: Rook endgame technique — active rook, rook behind passed pawn, cutting off the king, rook vs pawn.
- minor_piece_endgame: Bishop/knight endgame technique — piece + king coordination, good vs bad bishop, knight maneuvering.
- queen_endgame: Queen endgame technique — queen + pawn, perpetual/checking technique, queen vs pawn.
- passed_pawn_promotion: Promotion & passed-pawn handling that spans material — when to push, blockading, supporting/escorting the runner, promotion timing.

NOT-ENDGAME escape (use reroute, leave the Endgame category):
- If the PRIMARY mistake is a tactic the player missed or walked into (fork, pin, mate threat) that isn't really endgame technique -> reroute="missed_a_tactic" or "walked_into_tactic".
- If it's hanging a piece -> reroute="hung_a_piece".
- If it's a bad trade/simplification -> reroute="bad_trade".
- If it's a greedy capture -> reroute="greedy_capture"."""


def prompt(chip, desc):
    return f"""You are a chess coach sorting a blunder pattern into your endgame curriculum. Read it and decide the PRIMARY lesson.

FEATURE:
chip: {chip}
description: {desc}

{TOPICS}

Judge by the ACTUAL mistake in the description, not just the material on the board. A position with few pieces where the player hung something or missed a fork is NOT an endgame-technique lesson — reroute it.

Respond with ONLY JSON, no preamble:
{{"topic": "<one topic id OR null if rerouting>", "reroute": "<mistake-type id, or null if it stays as endgame>", "confidence": 0-100}}"""


def parse(t):
    s, e = t.find("{"), t.rfind("}") + 1
    try:
        o = json.loads(t[s:e])
    except Exception:
        return None
    return {"topic": o.get("topic"), "reroute": o.get("reroute"),
            "confidence": int(o.get("confidence", 0)) if str(o.get("confidence", "")).isdigit() else 0}


def invoke(c, p, n=6):
    for a in range(n):
        try:
            r = c.invoke_model(modelId=MODEL_ID, body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 150,
                "messages": [{"role": "user", "content": p}]}))
            return json.loads(r["body"].read())["content"][0]["text"]
        except Exception as ex:
            if any(x in str(ex) for x in ("Throttl", "503", "500", "imeout", "Too many")) and a < n - 1:
                time.sleep(min(2 ** a + random.random(), 20)); continue
            raise


def main():
    feats = json.load(open(IN))
    c = boto3.client("bedrock-runtime", region_name="us-east-1",
                     config=Config(read_timeout=60, retries={"max_attempts": 0}))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [f for f in feats if f not in res]
    print(f"{len(res)} done, {len(todo)} to read", flush=True)

    def work(f):
        r = parse(invoke(c, prompt(feats[f]["chip"], feats[f]["description"][:400])))
        if r is None:
            r = {"topic": None, "reroute": None, "confidence": 0}
        r["feature_id"] = int(f); r["chip"] = feats[f]["chip"]
        return f, r

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work, f) for f in todo]):
            f, r = fut.result(); res[f] = r; done += 1
            if done % 30 == 0:
                with _lock: json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {done}/{len(todo)}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

    from collections import Counter
    stay = Counter(r["topic"] for r in res.values() if not r.get("reroute"))
    rer = Counter(r["reroute"] for r in res.values() if r.get("reroute"))
    print(f"\n=== STAYS IN ENDGAME ({sum(stay.values())} feats) ===")
    for k, v in stay.most_common(): print(f"  {v:>3}  {k}")
    print(f"\n=== REROUTED OUT ({sum(rer.values())} feats) ===")
    for k, v in rer.most_common(): print(f"  {v:>3}  -> {k}")


if __name__ == "__main__":
    main()
