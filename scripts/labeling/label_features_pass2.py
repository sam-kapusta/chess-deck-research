#!/usr/bin/env python3
"""Pass 2: Synthesize per-position analyses into feature-level labels.

Takes 10 Opus position analyses per feature and produces one label per feature.
Runs Opus 4.6 with thinking capped at 4096, concurrency 20.

Usage:
    AWS_PROFILE=samtkap-dev-admin python3 scripts/labeling/label_features_pass2.py
    AWS_PROFILE=samtkap-dev-admin python3 scripts/labeling/label_features_pass2.py --resume
"""
import argparse
import json
import time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

POSITIONS_PATH = "output/maia3_position_analyses_opus.json"
ENRICHMENT_PATH = "/tmp/position_enrichment_cache.json"
PROFILES_PATH = "/tmp/l2_feature_profiles_v2.json"
OUTPUT_PATH = "output/maia3_feature_labels_opus.json"
MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"
MAX_CONCURRENT = 20
REQUEST_TIMEOUT = 120

client = boto3.client("bedrock-runtime", region_name=REGION, config=Config(
    read_timeout=REQUEST_TIMEOUT,
    connect_timeout=10,
    retries={"max_attempts": 0}
))

stats = {"times": [], "throttles": 0, "errors": 0}


PROMPT_TEMPLATE = """You are an elite chess analyst labeling SAE (Sparse Autoencoder) features.

Each SAE feature fires on positions where a player made a specific type of mistake. Below are the top 10 positions where this feature activates most strongly, along with detailed analysis of each.

Your job: identify the SPECIFIC shared pattern — the geometric, tactical, or strategic thread connecting these positions. Focus on what the MOVES have in common, not just the tactical motifs.

=== FEATURE OVERVIEW ({n_positions} positions) ===
Moves played: {moves_list}
Phases: {phase_dist}
Sides: {side_dist}
Avg cp_loss: {avg_cp_loss:.0f}
Avg good moves available: {avg_good_moves:.1f}

{positions_text}

=== INSTRUCTIONS ===
Look at the MOVES FIRST. What piece, what direction, what type of move connects them?
Then use the analyses to understand WHY these moves are mistakes.

Respond in JSON:
{{
  "chip": "<3-5 word punchy title that immediately conveys the full pattern — this is displayed to the user>",
  "label": "<One sentence summarizing the pattern>",
  "description": "<Full paragraph. Reference evidence counts: 'X/10 positions showed...', 'Only Y/10 had...' Be specific about what's consistent vs. variant.>",
  "move_pattern": "<What the moves share geometrically: piece type, direction, check/capture, square patterns>",
  "why_bad": "<The common reason these moves are mistakes — what they give up or miss>",
  "sub_patterns": ["<variant 1>", "<variant 2>", "..."],
  "categories": ["<broadest category>", "<mid-level>", "<most specific>"],
  "confidence": <0-100 integer — how clearly these positions share a single pattern>
}}"""


def build_feature_prompt(fid, examples, enrichment, analyses):
    positions_text = ""
    moves = []
    phases = []
    sides = []
    cp_losses = []
    good_moves_list = []

    for i, ex in enumerate(examples):
        key = f"{ex['fen']}|{ex['uci']}"
        enriched = enrichment.get(key, {})
        analysis = analyses.get(key, {}).get("analysis", {})

        if not enriched or "error" in enriched:
            continue
        if not analysis:
            continue

        played_san = enriched.get("played_san", ex["uci"])
        moves.append(played_san)
        phases.append(enriched.get("phase", "?"))
        sides.append(enriched.get("side", "?"))
        cp_losses.append(enriched.get("cp_loss", 0))
        good_moves_list.append(enriched.get("n_good_moves", 0))

        # Position block
        features_text = '\n'.join(f'    - {f}' for f in enriched.get("position_features", [])) or '    - (standard position)'
        best_text = '\n'.join(f'    {j+1}. {b["line"]} (eval: {b["eval"]})' for j, b in enumerate(enriched.get("top_3_best", [])))
        refut_text = '\n'.join(f'    {j+1}. {r["line"]} (eval: {r["eval"]})' for j, r in enumerate(enriched.get("top_3_refutations", [])))

        positions_text += f"""
=== POSITION {i+1} ===
Move: {played_san} | Side: {enriched.get('side', '?')} | Phase: {enriched.get('phase', '?')} | cp_loss: {enriched.get('cp_loss', '?')}
Eval: {enriched.get('eval_before', '?')} -> {enriched.get('eval_after', '?')} | Good moves: {enriched.get('n_good_moves', '?')} | Punish: {enriched.get('punish_type', '?')}
Position features:
{features_text}
Top 3 best moves:
{best_text}
Top 3 refutations:
{refut_text}

Analysis:
{json.dumps(analysis, indent=2)}
"""

    if len(moves) < 3:
        return None

    phase_dist = ', '.join(f'{phase}({count})' for phase, count in Counter(phases).most_common())
    side_dist = ', '.join(f'{side}({count})' for side, count in Counter(sides).most_common())
    avg_cp_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0
    avg_good_moves = sum(good_moves_list) / len(good_moves_list) if good_moves_list else 0

    return PROMPT_TEMPLATE.format(
        n_positions=len(moves),
        moves_list=', '.join(moves),
        phase_dist=phase_dist,
        side_dist=side_dist,
        avg_cp_loss=avg_cp_loss,
        avg_good_moves=avg_good_moves,
        positions_text=positions_text,
    )


def invoke_opus(prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32000,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    for block in result["content"]:
        if block.get("type") == "text":
            return block["text"]
    return result["content"][-1].get("text", "")


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def process_one(item):
    fid, prompt = item
    pt0 = time.time()
    retries = 0
    while retries < 3:
        try:
            raw = invoke_opus(prompt)
            elapsed = time.time() - pt0
            parsed = parse_json_response(raw)
            if parsed:
                parsed["feature_id"] = int(fid)
                parsed["time_s"] = round(elapsed, 1)
                return (fid, parsed)
            else:
                return (fid, {"error": "parse_failed", "raw": raw[:500], "time_s": round(elapsed, 1)})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException":
                retries += 1
                stats["throttles"] += 1
                wait = 2 ** retries
                print(f"  THROTTLED F{fid}, retry {retries} in {wait}s", flush=True)
                time.sleep(wait)
            else:
                stats["errors"] += 1
                return (fid, {"error": str(e)[:300]})
        except Exception as e:
            stats["errors"] += 1
            return (fid, {"error": str(e)[:300]})
    return (fid, {"error": "max_retries_throttled"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', default=POSITIONS_PATH)
    parser.add_argument('--enrichment', default=ENRICHMENT_PATH)
    parser.add_argument('--profiles', default=PROFILES_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with open(args.positions) as f:
        analyses = json.load(f)
    with open(args.enrichment) as f:
        enrichment = json.load(f)
    with open(args.profiles) as f:
        profiles = json.load(f)

    # Resume
    results = {}
    if args.resume:
        try:
            with open(args.output) as f:
                results = json.load(f)
            done_count = sum(1 for v in results.values() if "error" not in v)
            print(f"Resuming: {done_count} already labeled", flush=True)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Build prompts for each feature
    work = []
    skipped = 0
    for fid in sorted(profiles.keys(), key=int):
        if fid in results and "error" not in results[fid]:
            continue

        examples = profiles[fid].get("examples", [])[:10]
        prompt = build_feature_prompt(fid, examples, enrichment, analyses)
        if prompt is None:
            results[fid] = {"error": "insufficient_analyzed_positions"}
            skipped += 1
            continue
        work.append((fid, prompt))

    total = len(work)
    print(f"Features to label: {total} (skipped {skipped} with <3 analyzed positions)", flush=True)

    if total == 0:
        print("All features already labeled!", flush=True)
        return

    print(f"Labeling {total} features | Opus 4.6 | concurrency={MAX_CONCURRENT} | thinking=4096", flush=True)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            fid, result = future.result()
            results[fid] = result
            done += 1
            if "time_s" in result:
                stats["times"].append(result["time_s"])
            if done % 10 == 0 or done <= 5 or done == total:
                elapsed = time.time() - t0
                avg_t = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
                eta_h = (total - done) * avg_t / MAX_CONCURRENT / 3600 if avg_t > 0 else 0
                print(f"  {done}/{total} | {elapsed/60:.1f}min | avg {avg_t:.0f}s/pos | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)
            if done % 50 == 0:
                with open(args.output, "w") as f:
                    json.dump(results, f)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    successes = sum(1 for v in results.values() if "error" not in v)
    times = stats["times"]
    print(f"\nDone. {successes}/{len(profiles)} features labeled | {elapsed/60:.1f}min | throttles={stats['throttles']}", flush=True)
    if times:
        print(f"Per-feature: min={min(times):.0f}s avg={sum(times)/len(times):.0f}s max={max(times):.0f}s", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
