#!/usr/bin/env python3
"""Pass-1 for BTK SAE: Opus-label gap positions (those in btk_profiles but not
already in all_positions_labeled_opus.json). Reuses proven label_all_positions_opus
pattern: Opus 4.6, thinking=4096, max_tokens=8000, concurrency=60, resume-safe.

Usage (on chess-poc):
    AWS_PROFILE=default python scripts/labeling/label_positions_btk.py \
      --profiles ~/SageMaker/chess-stage-a/output/btk_profiles.json \
      --existing ~/SageMaker/all_positions_labeled_opus.json \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --output ~/SageMaker/all_positions_labeled_opus.json
"""
import argparse, json, time, boto3, sys
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION   = "us-east-1"
MAX_CONCURRENT = 60
REQUEST_TIMEOUT = 120

MOTIFS = ("hanging_piece|fork|pin|skewer|discovered_attack|back_rank|"
          "overloaded_defender|trapped_piece|pawn_endgame|rook_endgame|"
          "king_safety|passed_pawn|promotion_error|tempo_loss|positional_mistake|other")

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=REQUEST_TIMEOUT, connect_timeout=10,
                  retries={"max_attempts": 0}))
stats = {"times": [], "throttles": 0, "errors": 0}


def build_prompt(enriched):
    played_san = enriched["played_san"]
    features_text = "\n".join(f"  - {f}" for f in enriched.get("position_features", [])) \
                    or "  - (standard position)"
    best_text  = "\n".join(f"  {i+1}. {b['line']} (eval: {b['eval']})"
                           for i,b in enumerate(enriched.get("top_3_best", [])))
    refut_text = "\n".join(f"  {i+1}. {r['line']} (eval: {r['eval']})"
                           for i,r in enumerate(enriched.get("top_3_refutations", [])))
    return (
        f"You are an elite chess grandmaster and coach. Give a THOROUGH, DETAILED analysis.\n\n"
        f"=== POSITION DATA ===\n"
        f"FEN: {enriched['fen']}\nSide: {enriched['side']}\nPhase: {enriched['phase']}\n"
        f"Move played: {played_san}\nCentipawn loss: {enriched['cp_loss']}cp\n"
        f"Eval shift: {enriched['eval_before']} -> {enriched['eval_after']}\n"
        f"Good moves available: {enriched['n_good_moves']} within 50cp\n"
        f"Punishment type: {enriched['punish_type']}\n\n"
        f"Position features:\n{features_text}\n\n"
        f"=== TOP 3 BEST MOVES ===\n{best_text}\n\n"
        f"=== TOP 3 REFUTATIONS (after {played_san}) ===\n{refut_text}\n\n"
        f"Respond in JSON:\n{{"
        f'"position_description":"<3-4 sentences>",'
        f'"best_moves_analysis":"<4-6 sentences covering all 3>",'
        f'"move_intent":"<1-2 sentences>",'
        f'"refutation_analysis":"<4-6 sentences covering all 3>",'
        f'"blunder_summary":"<2-3 sentences>",'
        f'"tactical_motif":"<{MOTIFS}>",'
        f'"tags":["<tag>"]'
        f"}}"
    )


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
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
    result = json.loads(resp["body"].read())
    for block in result["content"]:
        if block.get("type") == "text": return block["text"]
    return result["content"][-1].get("text", "")


def process_one(item):
    key, enriched = item
    t0 = time.time()
    for attempt in range(3):
        try:
            raw = invoke_opus(build_prompt(enriched))
            parsed = parse_json_response(raw)
            return (key, {"analysis": parsed, "time_s": round(time.time()-t0, 1)}
                    if parsed else {"error": "parse_failed", "raw": raw[:300]})
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                stats["throttles"] += 1
                time.sleep(2 ** (attempt+1))
            else:
                return (key, {"error": str(e)[:200]})
    return (key, {"error": "max_retries"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",   required=True)
    parser.add_argument("--existing",   required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    profiles  = json.load(open(args.profiles))
    results   = json.load(open(args.existing)) if args.existing else {}
    enrichment = json.load(open(args.enrichment))

    needed = set()
    for fid_data in profiles.values():
        for ex in fid_data.get("examples", [])[:15]:
            needed.add(ex["key"])

    gap = [k for k in needed
           if k not in results or "analysis" not in results[k]]
    print(f"Total unique positions in profiles: {len(needed)}")
    print(f"Already labeled: {len(needed) - len(gap)}")
    print(f"Gap to label: {len(gap)}")

    if not gap:
        print("No gap - all positions already labeled.")
        return

    work = [(k, enrichment[k]) for k in gap if k in enrichment and "error" not in enrichment.get(k, {"error":1})]
    no_enrich = len(gap) - len(work)
    if no_enrich:
        print(f"WARNING: {no_enrich} gap positions lack enrichment - will be skipped")

    print(f"Labeling {len(work)} positions | model=Opus 4.6 | concurrency={MAX_CONCURRENT}", flush=True)
    t0 = time.time(); done = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
            done += 1
            if "time_s" in result: stats["times"].append(result["time_s"])
            if done % 50 == 0:
                with open(args.output, "w") as f: json.dump(results, f)
                avg_t = sum(stats["times"])/len(stats["times"]) if stats["times"] else 0
                eta_h = (len(work)-done)*avg_t/MAX_CONCURRENT/3600 if avg_t else 0
                print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | "
                      f"avg {avg_t:.0f}s | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)

    with open(args.output, "w") as f: json.dump(results, f, indent=2)
    print(f"Done. Saved {len(results)} total labeled positions to {args.output}")


if __name__ == "__main__":
    main()
