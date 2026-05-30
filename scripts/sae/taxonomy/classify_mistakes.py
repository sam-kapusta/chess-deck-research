"""Reading-based mistake-type classification — Sonnet via boto3 (NOT Opus agents,
which stalled). Reuses the proven relabel_sonnet pattern: concurrent, incremental
save, throttle retry. One Sonnet call per feature reads the description and assigns
the PRIMARY mistake type from the locked seed vocabulary.

Keyword classification is unusable here (77% of features match 3+ category
keywords). This reads each feature and judges the primary mistake.

Usage:
    AWS_PROFILE=default python3 scripts/sae/taxonomy/classify_mistakes.py \
        --features output/taxonomy_v2/cluster_input.json \
        --out output/taxonomy_v2/mistake_assignments.json --threads 8
"""
import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"
_lock = threading.Lock()

# Locked seed vocabulary (grill-with-docs 2026-05-30). Data may revise after.
CATEGORIES = {
    # commission — your move actively lost
    "hung_a_piece": "Your move leaves a piece (the one you moved OR another) undefended and free to capture.",
    "walked_into_tactic": "Your move lets the opponent fork, pin, skewer, or hit you with a discovered attack.",
    "greedy_capture": "You grabbed material (a pawn or piece) and it backfired — lost more than you gained, or threw away an advantage.",
    "exposed_your_king": "Your move (a pawn push, piece move, or king move) weakened your own king's safety or exposed it to attack.",
    "bad_trade": "You traded or simplified into a worse position, or a recapture/exchange that activates the opponent.",
    "abandoned_defender": "You moved a piece away from a square where it was performing a critical defensive duty.",
    # offensive miss — you had a good move and played a nothing-move instead
    "misplayed_attack": "You had an attack or initiative against the enemy king and failed to convert it — missed a forced mate, OR missed winning material via the attack, OR let the attack fizzle with a slow move, OR attacked unsoundly/too early.",
    "missed_a_capture": "There was free or winning material available to capture and you played something else.",
    "missed_a_tactic": "You had a fork, pin, skewer, or winning combination available and played a quiet/non-forcing move instead.",
    # other
    "endgame_error": "An endgame technique mistake — king activity, pawn race, promotion, opposition, conversion.",
    "missed_defense": "You were under threat and had a defensive resource (a save) available but missed it.",
}


def build_prompt(chip, desc, vocab_block):
    return f"""You are classifying a chess-blunder SAE feature by the PRIMARY type of mistake it represents. Read the description and pick the ONE best-fitting category.

FEATURE:
chip: {chip}
description: {desc}

CATEGORIES (id: definition):
{vocab_block}

KEY DISTINCTIONS (these features are easy to mis-bucket):
- COMMISSION (your move actively loses) vs OFFENSIVE MISS (you had a good move and played a passive/nothing move instead). A "quiet pawn push when a forcing capture was available" is an OFFENSIVE MISS (missed_a_capture or missed_a_tactic), NOT a commission — unless the quiet move itself hangs a piece.
- "missed_a_capture/tactic/misplayed_attack" = you FAILED TO PLAY a good move. "hung_a_piece/walked_into_tactic" = the move you PLAYED created the loss.
- If the move both misses a win AND actively loses, pick the COMMISSION category (the active loss is the primary lesson).
- misplayed_attack is for ATTACKING situations (you had initiative vs their king); missed_defense is for DEFENDING situations (you were under attack).

Respond with ONLY the JSON object, no preamble or explanation: {{"category": "<one id>", "confidence": 0-100}}"""


def parse(text, ids):
    s, e = text.find("{"), text.rfind("}") + 1
    try:
        o = json.loads(text[s:e])
    except Exception:
        return {"category": None, "confidence": 0}
    c = o.get("category")
    if c not in ids:
        c = None
    return {"category": c, "confidence": int(o.get("confidence", 0)) if str(o.get("confidence", "")).isdigit() else 0}


def invoke(client, prompt, retries=6):
    for a in range(retries):
        try:
            r = client.invoke_model(modelId=MODEL_ID, body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]}))
            return json.loads(r["body"].read())["content"][0]["text"]
        except Exception as ex:
            if any(t in str(ex) for t in ("Throttl", "throttl", "ServiceUnavailable", "503", "500", "Timeout", "timeout", "Too many")) and a < retries - 1:
                time.sleep(min(2 ** a + random.random(), 30)); continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    rows = json.load(open(args.features))
    ids = set(CATEGORIES)
    vocab_block = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 0}))

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [f for f in rows if f not in results or results[f].get("category") is None]
    print(f"{len(results)} done, {len(todo)} to do, {args.threads} threads", flush=True)

    def work(f):
        r = rows[f]
        res = parse(invoke(client, build_prompt(r["chip"], r["description"][:320], vocab_block)), ids)
        res["feature_id"] = int(f)
        return f, res

    done = 0; errors = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(work, f): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                _, res = fut.result()
            except Exception as e:
                errors += 1; res = {"category": None, "confidence": 0, "feature_id": int(f), "error": str(e)[:80]}
            results[f] = res; done += 1
            if done % 100 == 0:
                with _lock:
                    json.dump(results, open(args.out, "w"), indent=1)
                print(f"  {done}/{len(todo)} (errors={errors})", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)

    from collections import Counter
    c = Counter(r.get("category") for r in results.values())
    n = len(results)
    print(f"\nDone. {n} features, errors={errors}, unassigned={c.get(None, 0)}")
    print(f"{'category':<22}{'feats':>6}{'%':>5}")
    for k, v in c.most_common():
        print(f"{str(k):<22}{v:>6}{v/n*100:>4.0f}%")


if __name__ == "__main__":
    main()
