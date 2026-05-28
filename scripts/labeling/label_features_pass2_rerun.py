#!/usr/bin/env python3
"""Pass 2 re-run: Relabel low-confidence features with enhanced geometric context.

Adds from-file/rank distributions, bimodal detection, and board-state emphasis
to help the model distinguish features the first pass couldn't differentiate.

Only processes features with confidence < 80 from the first pass.
Marks output with "relabeled": true and "relabel_reason".

Usage:
    AWS_PROFILE=samtkap-dev-admin python3 scripts/labeling/label_features_pass2_rerun.py
"""
import argparse
import json
import time
import statistics
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

LABELS_PATH = "output/maia3_feature_labels_opus.json"
POSITIONS_PATH = "/tmp/all_positions_labeled_opus_final.json"
ENRICHMENT_PATH = "/tmp/position_enrichment_cache.json"
PROFILES_PATH = "/tmp/l2_feature_profiles_v2.json"
OUTPUT_PATH = "output/maia3_feature_labels_opus.json"  # overwrites in place
MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"
MAX_CONCURRENT = 20
REQUEST_TIMEOUT = 120
CONFIDENCE_THRESHOLD = 80

client = boto3.client("bedrock-runtime", region_name=REGION, config=Config(
    read_timeout=REQUEST_TIMEOUT,
    connect_timeout=10,
    retries={"max_attempts": 0}
))

stats = {"times": [], "throttles": 0, "errors": 0}


PROMPT_TEMPLATE = """You are an elite chess analyst labeling SAE (Sparse Autoencoder) features.

Each SAE feature fires on positions where a player made a specific type of mistake. Below are the top positions where this feature activates most strongly.

CRITICAL CONTEXT: This SAE uses "diff pooling" (to_square - from_square activation vectors). Features encode MOVE GEOMETRY — the piece type, direction, and distance of the move, PLUS the board context from the residual stream. Two features can both be "pawn pushes" but differ by WHICH FILE or WHICH RANK the pawn is on.

=== FEATURE GEOMETRIC SIGNATURE ===
{geometric_section}

=== FEATURE OVERVIEW ({n_positions} positions) ===
Moves played: {moves_list}
Phases: {phase_dist}
Sides: {side_dist}
Avg cp_loss: {avg_cp_loss:.0f}
Avg good moves available: {avg_good_moves:.1f}

{positions_text}

=== INSTRUCTIONS ===
1. FIRST: Look at the geometric signature. What SPECIFIC move shape does this encode?
   - If from-files cluster: this feature fires on a SPECIFIC FILE's piece/pawn
   - If bimodal ranks (2 and 7): same move for both colors (White rank 2 = Black rank 7)
   - If distances are uniform: specific piece type with specific reach
2. THEN: Look at the board contexts for what positional theme connects them
3. Make the chip SPECIFIC — "h-pawn push from starting square" not "pawn push"
4. Distinguish this from related features: what makes THIS one unique?

Respond in JSON:
{{
  "chip": "<3-5 word punchy title — SPECIFIC to this feature's geometric pattern>",
  "label": "<One sentence: what specific move geometry + board context this encodes>",
  "description": "<Full paragraph with evidence counts. What distinguishes this from other similar features?>",
  "move_pattern": "<SPECIFIC geometric pattern: piece type, from-file/rank, to-file/rank, direction, distance>",
  "why_bad": "<The common reason these specific moves are mistakes in these specific contexts>",
  "sub_patterns": ["<variant 1>", "<variant 2>", "..."],
  "categories": ["<broadest>", "<mid-level>", "<most specific>"],
  "confidence": <0-100>
}}"""


def compute_geometric_signature(examples, enrichment):
    """Compute detailed geometric analysis of the feature's move pattern."""
    ucis = [ex['uci'] for ex in examples]

    from_files = [u[0] for u in ucis]
    from_ranks = [int(u[1]) for u in ucis]
    to_files = [u[2] for u in ucis]
    to_ranks = [int(u[3]) for u in ucis]

    diffs = [(ord(u[2])-ord(u[0]), int(u[3])-int(u[1])) for u in ucis]
    dxs = [d[0] for d in diffs]
    dys = [d[1] for d in diffs]

    dx_std = statistics.stdev(dxs) if len(dxs) > 1 else 0
    dy_std = statistics.stdev(dys) if len(dys) > 1 else 0
    distances = [abs(d[0]) + abs(d[1]) for d in diffs]

    # Piece types from SAN
    pieces = []
    for ex in examples:
        key = ex['fen'] + '|' + ex['uci']
        e = enrichment.get(key, {})
        if e and 'error' not in e:
            san = e.get('played_san', '?')
            if san[0] in 'KQRBN':
                pieces.append(san[0])
            else:
                pieces.append('Pawn')

    # File clustering
    from_file_counter = Counter(from_files)
    dominant_file = from_file_counter.most_common(1)[0] if from_files else ('?', 0)
    file_purity = dominant_file[1] / len(from_files) if from_files else 0

    # Bimodal rank detection (ranks 2+7 or 1+8 = same move both colors)
    rank_counter = Counter(from_ranks)
    has_low = sum(rank_counter.get(r, 0) for r in [1, 2, 3])
    has_high = sum(rank_counter.get(r, 0) for r in [6, 7, 8])
    bimodal = has_low >= 3 and has_high >= 3

    # Build text
    lines = []
    lines.append(f"Piece types: {Counter(pieces).most_common()}")
    lines.append(f"Diff vectors (dx, dy): {diffs}")
    lines.append(f"Diff tightness: dx_std={dx_std:.2f}, dy_std={dy_std:.2f}")
    lines.append(f"Distances (manhattan): {distances} (avg={sum(distances)/len(distances):.1f})")
    lines.append(f"From files: {from_files} → distribution: {from_file_counter.most_common()}")
    lines.append(f"From ranks: {from_ranks} → distribution: {Counter(from_ranks).most_common()}")
    lines.append(f"To files: {to_files}")
    lines.append(f"To ranks: {to_ranks}")

    if file_purity >= 0.6:
        lines.append(f"*** FILE CLUSTER: {dominant_file[1]}/10 moves from {dominant_file[0]}-file ***")

    if bimodal:
        lines.append(f"*** BIMODAL RANKS: {has_low} from ranks 1-3, {has_high} from ranks 6-8 = SAME MOVE BOTH COLORS ***")

    if dx_std < 0.5 and dy_std < 0.5:
        avg_dx = sum(dxs) / len(dxs)
        avg_dy = sum(dys) / len(dys)
        lines.append(f"*** VERY TIGHT VECTOR: all moves are approximately ({avg_dx:.0f}, {avg_dy:.0f}) ***")

    # Diagonal vs straight
    diag = sum(1 for d in diffs if abs(d[0]) == abs(d[1]) and d[0] != 0)
    straight = sum(1 for d in diffs if d[0] == 0 or d[1] == 0)
    knight = sum(1 for d in diffs if sorted([abs(d[0]), abs(d[1])]) == [1, 2])
    if diag >= 7:
        lines.append(f"*** ALL DIAGONAL MOVES (bishop-like) ***")
    elif straight >= 7:
        lines.append(f"*** ALL STRAIGHT MOVES (rook/pawn-like) ***")
    elif knight >= 7:
        lines.append(f"*** ALL KNIGHT MOVES ***")

    return '\n'.join(lines)


def build_feature_prompt(fid, examples, enrichment, analyses):
    """Build enhanced prompt with geometric context."""
    geometric_section = compute_geometric_signature(examples, enrichment)

    positions_text = ""
    moves = []
    phases = []
    sides = []
    cp_losses = []
    good_moves_list = []

    for i, ex in enumerate(examples):
        key = ex['fen'] + '|' + ex['uci']
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

        features_text = '\n'.join(f'    - {f}' for f in enriched.get("position_features", [])) or '    - (standard position)'
        best_text = '\n'.join(f'    {j+1}. {b["line"]} (eval: {b["eval"]})' for j, b in enumerate(enriched.get("top_3_best", [])))
        refut_text = '\n'.join(f'    {j+1}. {r["line"]} (eval: {r["eval"]})' for j, r in enumerate(enriched.get("top_3_refutations", [])))

        positions_text += f"""
=== POSITION {i+1} ===
Move: {played_san} (UCI: {ex['uci']}) | Side: {enriched.get('side', '?')} | Phase: {enriched.get('phase', '?')} | cp_loss: {enriched.get('cp_loss', '?')}
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
        geometric_section=geometric_section,
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
                parsed["relabeled"] = True
                parsed["relabel_reason"] = "confidence_below_80_geometric_rerun"
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
    parser.add_argument('--labels', default=LABELS_PATH)
    parser.add_argument('--positions', default=POSITIONS_PATH)
    parser.add_argument('--enrichment', default=ENRICHMENT_PATH)
    parser.add_argument('--profiles', default=PROFILES_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--threshold', type=int, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    with open(args.labels) as f:
        all_labels = json.load(f)
    with open(args.positions) as f:
        analyses = json.load(f)
    with open(args.enrichment) as f:
        enrichment = json.load(f)
    with open(args.profiles) as f:
        profiles = json.load(f)

    # Find features to relabel
    to_relabel = {fid: v for fid, v in all_labels.items()
                  if 'confidence' in v and v['confidence'] < args.threshold}

    print(f"Features below confidence {args.threshold}: {len(to_relabel)}", flush=True)

    # Build prompts
    work = []
    skipped = 0
    for fid in sorted(to_relabel.keys(), key=int):
        examples = profiles[fid].get("examples", [])[:10]
        prompt = build_feature_prompt(fid, examples, enrichment, analyses)
        if prompt is None:
            skipped += 1
            continue
        work.append((fid, prompt))

    total = len(work)
    print(f"Relabeling {total} features (skipped {skipped}) | concurrency={MAX_CONCURRENT}", flush=True)

    if total == 0:
        return

    t0 = time.time()
    done = 0
    improved = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            fid, result = future.result()

            if "error" not in result:
                old_conf = all_labels[fid].get('confidence', 0)
                new_conf = result.get('confidence', 0)
                if new_conf > old_conf:
                    improved += 1
                # Always take the re-run result (it has geometric context the first didn't)
                result["prev_confidence"] = old_conf
                all_labels[fid] = result

            done += 1
            if "time_s" in result:
                stats["times"].append(result["time_s"])
            if done % 10 == 0 or done <= 5 or done == total:
                elapsed = time.time() - t0
                avg_t = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
                eta_h = (total - done) * avg_t / MAX_CONCURRENT / 3600 if avg_t > 0 else 0
                print(f"  {done}/{total} | {elapsed/60:.1f}min | avg {avg_t:.0f}s | improved={improved} | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)
            if done % 50 == 0:
                with open(args.output, "w") as f:
                    json.dump(all_labels, f)

    with open(args.output, "w") as f:
        json.dump(all_labels, f, indent=2)

    elapsed = time.time() - t0
    relabeled_count = sum(1 for v in all_labels.values() if v.get('relabeled'))
    print(f"\nDone. {relabeled_count} relabeled, {improved} improved confidence | {elapsed/60:.1f}min", flush=True)

    # Stats on improvement
    new_confs = [v['confidence'] for v in all_labels.values() if 'confidence' in v]
    above_80 = sum(1 for c in new_confs if c >= 80)
    print(f"Features >= 80 confidence: {above_80}/{len(new_confs)} ({above_80/len(new_confs)*100:.0f}%)", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
