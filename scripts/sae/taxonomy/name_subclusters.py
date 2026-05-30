"""Name the 200 bge-m3 sub-clusters — Sonnet via boto3 (proven reliable; Opus
agents stalled). Each sub-cluster: read all member chips, assign (a) a short
specific sub-cluster name, and (b) a mistake-type category — seeded by our 11
exemplars as a STRONG STARTING SET, free to propose a new category if the
cluster genuinely doesn't fit. Bottom-up: categories emerge, then we roll up.

Usage:
    AWS_PROFILE=default python3 scripts/sae/taxonomy/name_subclusters.py \
        --packets output/taxonomy_v2/subcluster_packets.json \
        --out output/taxonomy_v2/subcluster_names.json --threads 8
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"
_lock = threading.Lock()

SEED = """Hung a Piece — your move leaves a piece undefended/free to capture
Walked Into a Tactic — your move lets the opponent fork/pin/skewer you
Greedy Capture — grabbed material, backfired or threw away the advantage
Exposed Your King — weakened your own king safety
Bad Trade or Simplification — traded into a worse position
Abandoned a Defender — moved a piece off its critical defensive duty
Misplayed an Attack — had attack/initiative vs their king, failed to convert (missed mate OR missed winning material OR let it fizzle OR unsound/early)
Missed a Capture — free/winning material available, played something else
Missed a Tactic — had a fork/pin/combination available, played a quiet move
Endgame Error — king activity / pawn race / promotion / opposition / conversion
Missed a Defensive Resource — under threat, had a save, missed it"""


def build_prompt(p):
    chips = "\n".join(f"  - {c}" for c in p["chips"])
    descs = "\n".join(f"  • {d}" for d in p["sample_descs"])
    return f"""You are naming ONE semantic cluster of chess-blunder SAE features and assigning it a mistake-type category.

CLUSTER ({p['size']} features) — member chips:
{chips}

A few full descriptions:
{descs}

SEED CATEGORIES (strong starting set — use one of these if it fits; propose a NEW category only if the cluster genuinely doesn't match any):
{SEED}

Decide the PRIMARY mistake this cluster represents (read the whole cluster, not the first chip).
Key distinction: COMMISSION (the move you played actively loses — hung/walked-into/greedy/exposed) vs OFFENSIVE MISS (you had a good move and played a passive/nothing move — missed capture/tactic, misplayed attack). A "quiet move while a forcing capture was available" is an OFFENSIVE MISS, not a commission.

Respond with ONLY this JSON, no preamble:
{{"subcluster_name": "<3-6 word specific name for this cluster>", "category": "<seed category name, or a NEW category name in the same style>", "is_new_category": true/false, "confidence": 0-100}}"""


def parse(text):
    s, e = text.find("{"), text.rfind("}") + 1
    try:
        o = json.loads(text[s:e])
    except Exception:
        return None
    return {
        "subcluster_name": str(o.get("subcluster_name", "")).strip(),
        "category": str(o.get("category", "")).strip(),
        "is_new_category": bool(o.get("is_new_category", False)),
        "confidence": int(o.get("confidence", 0)) if str(o.get("confidence", "")).isdigit() else 0,
    }


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
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    packets = json.load(open(args.packets))
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 0}))
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [s for s in packets if s not in results or not results[s].get("category")]
    print(f"{len(results)} done, {len(todo)} to do, {args.threads} threads", flush=True)

    def work(s):
        res = parse(invoke(client, build_prompt(packets[s])))
        if res is None:
            res = {"subcluster_name": "", "category": "", "is_new_category": False, "confidence": 0}
        res["fine_id"] = int(s)
        res["size"] = packets[s]["size"]
        res["feature_ids"] = packets[s]["feature_ids"]
        return s, res

    done = 0; err = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                _, res = fut.result()
            except Exception as e:
                err += 1; res = {"category": "", "subcluster_name": "", "fine_id": int(s), "error": str(e)[:80]}
            results[s] = res; done += 1
            if done % 40 == 0:
                with _lock:
                    json.dump(results, open(args.out, "w"), indent=1)
                print(f"  {done}/{len(todo)} (err={err})", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)

    from collections import Counter
    cats = Counter(r.get("category", "?") for r in results.values())
    newcats = Counter(r.get("category") for r in results.values() if r.get("is_new_category"))
    print(f"\nDone. {len(results)} sub-clusters named, err={err}")
    print(f"=== emergent category distribution (by # sub-clusters) ===")
    for k, v in cats.most_common():
        print(f"  {v:>3} subclusters  {k}")
    if newcats:
        print(f"\nNEW categories proposed: {dict(newcats)}")


if __name__ == "__main__":
    main()
