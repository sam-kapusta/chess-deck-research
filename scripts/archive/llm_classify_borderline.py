#!/usr/bin/env python3
"""LLM-based quality scoring for borderline coaching comments.

Uses Bedrock Claude Haiku to classify comments that the heuristic scorer
rated at score=1 (one coaching signal, just below threshold of 2).

Cost estimate for 3.5K borderline comments:
  - Input: 3.5K * 250 tokens = 875K tokens * $0.25/1M = $0.22
  - Output: 3.5K * 30 tokens = 105K tokens * $1.25/1M = $0.13
  - Total: ~$0.35

Usage:
  python research/scripts/llm_classify_borderline.py --dry-run
  python research/scripts/llm_classify_borderline.py --max 100
  python research/scripts/llm_classify_borderline.py --input research/data/lichess_studies.jsonl
"""
import json
import sys
import argparse
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

PROMPT_TEMPLATE = """You are classifying chess study comments for a coaching dataset.

Rate this comment as either COACHING or GENERIC.

COACHING = explains WHY a move is good/bad, teaches a concept, discusses alternatives
with reasoning, identifies tactical/strategic themes with explanation.

GENERIC = opening name labels, bare evaluations, move narration without insight,
humor without chess reasoning, non-English, just links/references.

Comment on the move {move} in this position:
"{comment}"

Respond with exactly one line: COACHING or GENERIC, then a pipe, then a 1-sentence reason.
Example: COACHING | Explains why trading bishops weakens the dark squares.
"""


def make_client(profile: str):
    """Create a Bedrock client (one per thread)."""
    session = boto3.Session(profile_name=profile)
    return session.client("bedrock-runtime", region_name="us-east-1")


def classify_one(client, comment: str, move: str, model_id: str) -> tuple[str, str]:
    """Classify a single comment using Bedrock Haiku with retry."""
    import time as _time
    prompt = PROMPT_TEMPLATE.format(move=move, comment=comment[:500])
    for attempt in range(5):
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 60, "temperature": 0.0},
            )
            text = response["output"]["message"]["content"][0]["text"].strip()
            parts = text.split("|", 1)
            label = parts[0].strip().upper()
            reason = parts[1].strip() if len(parts) > 1 else ""
            if "COACHING" in label:
                return "coaching", reason
            return "generic", reason
        except Exception as e:
            if "Throttling" in str(e) and attempt < 4:
                _time.sleep(2 ** attempt)  # 1, 2, 4, 8s backoff
                continue
            return "error", str(e)
    return "error", "max retries"


# Thread-local storage for Bedrock clients
_thread_local = threading.local()


def classify_item(item: dict, profile: str, model_id: str) -> dict:
    """Classify one item, creating a per-thread client."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = make_client(profile)
    label, reason = classify_one(
        _thread_local.client, item["comment"], item["move"], model_id
    )
    return {**item, "llm_label": label, "llm_reason": reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="research/data/lichess_studies.jsonl")
    parser.add_argument("--output", default="research/data/prepared/llm_classified.jsonl")
    parser.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", default="chess-deck")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    # Unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    # Load data
    data = [json.loads(l) for l in Path(args.input).read_text().strip().split("\n")]
    print(f"Loaded {len(data)} pairs")

    # Import heuristic scorer
    sys.path.insert(0, str(Path(__file__).parent))
    from classify_coaching_quality import score_comment, NON_ENGLISH_RE

    # Pre-filter: English, >= 50 chars
    filtered = []
    for d in data:
        comment = d.get("comment", "")
        if len(comment) < 50:
            continue
        if NON_ENGLISH_RE.search(comment):
            alpha = sum(1 for c in comment if c.isalpha())
            non_en = len(NON_ENGLISH_RE.findall(comment))
            if alpha > 0 and non_en / alpha > 0.05:
                continue
        filtered.append(d)
    print(f"After pre-filter: {len(filtered)}")

    # Find borderline (score=1)
    borderline = []
    for d in filtered:
        score, cats = score_comment(d["comment"])
        if score == 1:
            borderline.append({**d, "heuristic_score": score, "heuristic_cats": cats})
    print(f"Borderline (score=1): {len(borderline)}")

    n = args.max if args.max > 0 else len(borderline)
    items = borderline[:n]

    # Cost estimate
    input_cost = n * 250 * 0.25 / 1_000_000
    output_cost = n * 30 * 1.25 / 1_000_000
    print(f"Estimated cost for {n} comments: ${input_cost + output_cost:.2f}")

    if args.dry_run:
        return

    # Sequential classification with incremental saves (avoids throttling)
    import time as _time
    print(f"Classifying sequentially (1 req/s to avoid throttle)...")
    client = make_client(args.profile)

    # Resume from existing output if present
    out_path = Path(args.output)
    results = []
    if out_path.exists():
        existing = [json.loads(l) for l in out_path.read_text().strip().split("\n") if l.strip()]
        existing_fens = {r["fen"] for r in existing}
        results = existing
        items = [it for it in items if it["fen"] not in existing_fens]
        print(f"  Resuming: {len(results)} already done, {len(items)} remaining")

    errors = 0
    for i, d in enumerate(items):
        label, reason = classify_one(client, d["comment"], d["move"], args.model)
        if label == "error":
            errors += 1
            if errors <= 5:
                print(f"  Error [{errors}]: {reason[:80]}")
        d["llm_label"] = label
        d["llm_reason"] = reason
        results.append(d)

        # Incremental save every 50
        if (i + 1) % 50 == 0:
            coaching = sum(1 for r in results if r.get("llm_label") == "coaching")
            ok = sum(1 for r in results if r.get("llm_label") != "error")
            print(f"  {i+1}/{len(items)} done (total {ok} ok, {coaching} coaching, {errors} errors)")
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

        # Rate limit: ~1 req/s
        _time.sleep(0.3)

    # Filter out errors for final save
    results = [r for r in results if r.get("llm_label") != "error"]

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for d in results:
            f.write(json.dumps(d) + "\n")

    # Stats
    labels = Counter(d["llm_label"] for d in results)
    print(f"\nLLM classification: {dict(labels)}")
    print(f"Errors: {errors}")
    print(f"Saved {len(results)} to {args.output}")


if __name__ == "__main__":
    main()
