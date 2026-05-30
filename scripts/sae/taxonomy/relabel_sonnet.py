"""Assign each feature to one of 20 coaching categories + write a specific,
category-aware chip. One Sonnet call per feature. Reads evidence.json
(description + structural fingerprint). Resumable, throttle-tolerant.

Usage:
    AWS_PROFILE=default python3 scripts/sae/taxonomy/relabel_sonnet.py \
        --evidence output/taxonomy_v2/evidence.json \
        --vocab output/taxonomy_v2/category_vocab.json \
        --out output/taxonomy_v2/assignments.json --threads 8
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

_GENERIC = re.compile(
    r"(ignor\w*|miss\w*|wast\w*)\b.{0,20}\b(tactic|tactics|tempo|urgency|crisis|threat)", re.I
)

_save_lock = threading.Lock()


def vocab_block(vocab):
    return "\n".join(
        f"{c['id']} | {c['name']} | {c['definition']}" for c in vocab["categories"]
    )


def fp_summary(fp):
    if not fp:
        return "no structural data"
    top = ", ".join(f"{s}:{n}" for s, n in list(fp.get("to_sq_top", {}).items())[:4])
    return (f"dominant_piece={fp['dom_piece']} ({int(fp['dom_frac']*100)}% of moves); "
            f"capture_rate={int(fp['cap_rate']*100)}%; check_rate={int(fp['check_rate']*100)}%; "
            f"hang_rate={int(fp['hang_rate']*100)}%; top_to_squares=[{top}]")


def build_prompt(e, vblock, valid_ids):
    return f"""You are labeling a chess-blunder SAE feature for a coaching product. Assign it to EXACTLY ONE category and write a SPECIFIC chip.

FEATURE DESCRIPTION (accurate; synthesized from 10 analyzed blunder positions):
{e['description']}

STRUCTURAL FACTS (ground truth from the board — trust these for what piece moved / rates):
{fp_summary(e['fingerprint'])}

CONTROLLED CATEGORY VOCABULARY (id | name | definition) — pick ONE id:
{vblock}

RULES:
1. Pick the single best-fitting category id from the list above. Use the description for the mechanism (WHY it's a mistake) and the structural facts for what actually moved. If they conflict on the piece, trust structural facts.
2. Write a chip: 3-6 words, SPECIFIC — name the concrete pattern (piece, square, mechanism). It is shown UNDER the category header, so do NOT repeat the category name.
   BANNED (these are exactly the generic labels we are fixing): "ignores tactics", "misses tactics", "wastes tempo", "ignoring tactical urgency", "missing tactics".
   GOOD: "Bishop hangs on g4", "a-pawn push over center break", "Queen to rim allows fork", "Rook abandons back rank", "Knight to e4 walks into fork".
3. confidence 0-100: how clearly this feature fits the chosen category.

Respond ONLY with JSON:
{{"category": "<one id from the vocabulary>", "chip": "3-6 specific words", "confidence": 0-100}}"""


def parse(text, valid_ids):
    st, en = text.find("{"), text.rfind("}") + 1
    try:
        obj = json.loads(text[st:en])
    except Exception:
        return None
    cat = obj.get("category")
    if cat not in valid_ids:
        cat = None
    chip = str(obj.get("chip", "")).strip()
    return {
        "category": cat,
        "chip": chip,
        "confidence": int(obj.get("confidence", 0)) if str(obj.get("confidence", "")).isdigit() else 0,
        "chip_generic": bool(_GENERIC.search(chip)),
    }


def invoke(client, prompt, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            return json.loads(resp["body"].read())["content"][0]["text"]
        except Exception as ex:
            msg = str(ex)
            transient = any(t in msg for t in (
                "Throttl", "throttl", "ServiceUnavailable", "503", "500",
                "Too many", "rate", "TimeoutError", "Read timeout", "ModelStreamError"))
            if transient and attempt < max_retries - 1:
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--regenerate-generic", action="store_true",
                    help="re-run features whose saved chip is generic")
    args = ap.parse_args()

    evidence = json.load(open(args.evidence))
    vocab = json.load(open(args.vocab))
    valid_ids = {c["id"] for c in vocab["categories"]}
    vblock = vocab_block(vocab)

    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(read_timeout=60, connect_timeout=10,
                                        retries={"max_attempts": 0}))

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    def needs(fid):
        if fid not in results:
            return True
        r = results[fid]
        if r.get("category") is None:
            return True
        if args.regenerate_generic and r.get("chip_generic"):
            return True
        return False

    todo = [fid for fid in evidence if needs(fid)]
    print(f"{len(results)} done, {len(todo)} to do, {args.threads} threads...")

    def work(fid):
        e = evidence[fid]
        prompt = build_prompt(e, vblock, valid_ids)
        # up to 2 chip-regeneration attempts if the chip comes back generic
        last = None
        for _ in range(3):
            text = invoke(client, prompt)
            parsed = parse(text, valid_ids)
            last = parsed
            if parsed and not parsed["chip_generic"] and parsed["category"]:
                break
        if last is None:
            last = {"category": None, "chip": "", "confidence": 0, "chip_generic": True}
        last["feature_id"] = int(fid)
        return fid, last

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(work, fid): fid for fid in todo}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                _, res = fut.result()
            except Exception as exc:
                errors += 1
                res = {"category": None, "chip": "", "confidence": 0,
                       "chip_generic": True, "feature_id": int(fid), "error": str(exc)[:100]}
            results[fid] = res
            done += 1
            if done % 50 == 0:
                with _save_lock:
                    json.dump(results, open(args.out, "w"), indent=2)
                print(f"  {done}/{len(todo)}  (errors={errors})", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    from collections import Counter
    cats = Counter(r.get("category") for r in results.values())
    generic = sum(1 for r in results.values() if r.get("chip_generic"))
    none_cat = sum(1 for r in results.values() if r.get("category") is None)
    print(f"\nDone. {len(results)} features. errors={errors}")
    print(f"unassigned category: {none_cat} | still-generic chips: {generic}")
    print("category distribution:")
    names = {c["id"]: c["name"] for c in vocab["categories"]}
    for cid, n in cats.most_common():
        print(f"  {n:>4}  {names.get(cid, cid)}")


if __name__ == "__main__":
    main()
