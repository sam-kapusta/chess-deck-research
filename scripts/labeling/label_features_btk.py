#!/usr/bin/env python3
"""Pass-2 for BTK SAE: synthesize chip+description per feature from top-20 positions.
Reuses label_features_pass2.py pattern exactly. Opus 4.6, thinking=4096, concurrency=20.

Usage (on chess-poc):
    AWS_PROFILE=default python scripts/labeling/label_features_btk.py \
      --profiles ~/SageMaker/chess-stage-a/output/btk_profiles.json \
      --positions ~/SageMaker/all_positions_labeled_opus.json \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --output ~/SageMaker/chess-stage-a/output/feature_labels_btk_2048_k32.json \
      --resume
"""
import argparse, json, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"
MAX_CONCURRENT = 20
REQUEST_TIMEOUT = 120

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=REQUEST_TIMEOUT, connect_timeout=10,
                  retries={"max_attempts": 0}))
stats = {"times": [], "throttles": 0, "errors": 0}

PROMPT_TEMPLATE = """You are an elite chess analyst labeling SAE (Sparse Autoencoder) features.

Each SAE feature fires on positions where a player made a specific type of mistake. Below are the top positions where this feature activates most strongly, along with detailed analysis.

Your job: identify the SPECIFIC shared pattern - the geometric, tactical, or strategic thread. Focus on what the MOVES have in common.

=== FEATURE OVERVIEW ({n_positions} positions) ===
Moves played: {moves_list}
Phases: {phase_dist}
Sides: {side_dist}
Avg cp_loss: {avg_cp_loss:.0f}
Avg good moves available: {avg_good_moves:.1f}

{positions_text}

=== INSTRUCTIONS ===
Look at the MOVES FIRST. What piece, direction, or move type connects them?
Then use the analyses to understand WHY these moves are mistakes.
Note: displayed examples are TOP activators - most extreme cases. Typical activations are milder.

Respond in JSON:
{{
  "chip": "<3-5 word punchy title>",
  "label": "<one sentence summary>",
  "description": "<full paragraph, reference evidence counts X/N>",
  "move_pattern": "<geometric description: piece type, direction, check/capture>",
  "why_bad": "<common reason these moves fail>",
  "sub_patterns": ["<variant 1>"],
  "categories": ["<broad>", "<mid>", "<specific>"],
  "confidence": <0-100>
}}"""


def build_feature_prompt(fid, examples, enrichment, analyses):
    positions_text = ""
    moves, phases, sides, cp_losses, good_moves_list = [], [], [], [], []

    for i, ex in enumerate(examples):
        key = f"{ex['fen']}|{ex['uci']}"
        enriched = enrichment.get(key, {})
        analysis = analyses.get(key, {}).get("analysis", {})
        if not enriched or "error" in enriched or not analysis:
            continue
        played_san = enriched.get("played_san", ex["uci"])
        moves.append(played_san)
        phases.append(enriched.get("phase", "?"))
        sides.append(enriched.get("side", "?"))
        cp_losses.append(enriched.get("cp_loss", 0) or 0)
        good_moves_list.append(enriched.get("n_good_moves", 0) or 0)

        features_text = "\n".join(f"    - {f}" for f in enriched.get("position_features", [])) \
                        or "    - (standard position)"
        best_text  = "\n".join(f"    {j+1}. {b['line']} (eval: {b['eval']})"
                               for j,b in enumerate(enriched.get("top_3_best", [])))
        refut_text = "\n".join(f"    {j+1}. {r['line']} (eval: {r['eval']})"
                               for j,r in enumerate(enriched.get("top_3_refutations", [])))
        positions_text += f"""
=== POSITION {i+1} ===
Move: {played_san} | Side: {enriched.get('side','?')} | Phase: {enriched.get('phase','?')} | cp_loss: {enriched.get('cp_loss','?')}
Eval: {enriched.get('eval_before','?')} -> {enriched.get('eval_after','?')} | Good moves: {enriched.get('n_good_moves','?')} | Punish: {enriched.get('punish_type','?')}
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

    phase_dist = ", ".join(f"{p}({c})" for p,c in Counter(phases).most_common())
    side_dist  = ", ".join(f"{s}({c})" for s,c in Counter(sides).most_common())
    return PROMPT_TEMPLATE.format(
        n_positions=len(moves), moves_list=", ".join(moves),
        phase_dist=phase_dist, side_dist=side_dist,
        avg_cp_loss=sum(cp_losses)/len(cp_losses),
        avg_good_moves=sum(good_moves_list)/len(good_moves_list),
        positions_text=positions_text)


def parse_json_response(text):
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix): text = text[len(prefix):]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try: return json.loads(text[start:end])
        except: pass
    return None


def invoke_opus(prompt):
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8000,
                       "thinking": {"type": "enabled", "budget_tokens": 4096},
                       "messages": [{"role": "user", "content": prompt}]})
    resp = client.invoke_model(modelId=MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
    result = json.loads(resp["body"].read())
    for block in result["content"]:
        if block.get("type") == "text": return block["text"]
    return result["content"][-1].get("text", "")


def process_one(item):
    fid, prompt = item
    t0 = time.time()
    for attempt in range(3):
        try:
            raw = invoke_opus(prompt)
            parsed = parse_json_response(raw)
            return (fid, {"analysis": parsed, "time_s": round(time.time()-t0,1)}
                    if parsed else {"error": "parse_failed"})
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                stats["throttles"] += 1; time.sleep(2**(attempt+1))
            else:
                return (fid, {"error": str(e)[:200]})
    return (fid, {"error": "max_retries"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",   required=True)
    parser.add_argument("--positions",  required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    profiles   = json.load(open(args.profiles))
    analyses   = json.load(open(args.positions))
    enrichment = json.load(open(args.enrichment))

    results = {}
    if args.resume:
        try:
            results = json.load(open(args.output))
            done_n = sum(1 for v in results.values() if "error" not in v)
            print(f"Resuming: {done_n} already labeled")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    work = []
    skipped = 0
    for fid in sorted(profiles.keys(), key=int):
        if fid in results and "error" not in results[fid]:
            continue
        examples = profiles[fid].get("examples", [])[:20]
        prompt = build_feature_prompt(fid, examples, enrichment, analyses)
        if prompt is None:
            results[fid] = {"error": "insufficient_analyzed_positions"}
            skipped += 1
            continue
        work.append((fid, prompt))

    print(f"Features to label: {len(work)} (skipped {skipped} with <3 analyzed positions)")
    print(f"Labeling | Opus 4.6 | concurrency={MAX_CONCURRENT} | thinking=4096", flush=True)
    t0 = time.time(); done = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            fid, result = future.result()
            results[fid] = result
            done += 1
            if "time_s" in result: stats["times"].append(result["time_s"])
            if done % 50 == 0:
                with open(args.output, "w") as f: json.dump(results, f)
                avg_t = sum(stats["times"])/len(stats["times"]) if stats["times"] else 0
                eta_h = (len(work)-done)*avg_t/MAX_CONCURRENT/3600 if avg_t else 0
                print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | "
                      f"avg {avg_t:.0f}s | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)

    with open(args.output, "w") as f: json.dump(results, f, indent=2)
    ok = sum(1 for v in results.values() if "error" not in v)
    print(f"Done. {ok}/{len(profiles)} features labeled | throttles={stats['throttles']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
