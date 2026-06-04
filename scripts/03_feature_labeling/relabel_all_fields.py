#!/usr/bin/env python3
"""All-fields feature relabeler — the production version of the validated prototype.

WHY THIS EXISTS (the lesson):
  The previous labeler (label_features_integrated.py) fed Opus only ~2.5 of the 7 per-position
  fields it had: tactical_motif, tags, a truncated blunder_summary. It NEVER showed Opus
  `best_moves_analysis` (what the player should have played) or `move_intent`. That is why
  "missed-X" features were mislabeled as passive/hung — SEE is single-ply and blind to the
  multi-move tactic the player skipped (e.g. f1487, f745, f950). When you can't see the best
  move, you can't tell "hung a piece" from "missed winning a piece".

THE FIX (this script):
  For each feature, show Opus its top-10 positions WITH all the signal:
    - WHAT WENT WRONG   (blunder_summary)
    - BEST MOVE missed  (best_moves_analysis)   <-- the field that was being dropped
    - INTENT            (move_intent)
  plus a ONE-LINE SEE signature as an objective material floor. The SEE caveat is deliberately
  one line: the earlier 6-line "distrust SEE" block over-steered and made labels flip
  missed<->hung between runs (f745). One neutral line -> stable.

SCALING DECISION (single-call + flag low-consistency):
  One Opus call per feature. The model emits a `consistency` 0-100 (how many of the 10 boards
  fit the named mistake). That field IS the dual-label / mixed-feature signal for free — a
  genuinely mixed feature like f745 comes back ~82, not 100, and gets flagged for review.
  3-vote consensus was rejected: the instability it would fix was a prompt bug, already fixed.

Validated on 20 features (corrected 17/20): f1487->"Missed free capture",
f745->"Missed winning capture" (cons 82), f952->"Hangs own major piece" (cons 100).

Run on chess-poc (inputs live there):
  cd ~/SageMaker && python3 relabel_all_fields.py \
    --profiles d2048_k6_profiles.json --positions all_positions_labeled_opus.json \
    --seestats see_stats_d2048_k6.json --prev feature_labels_integrated_d2048_k6.json \
    --output relabel_allfields_d2048_k6.json --resume
"""
import argparse, json, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"   # same model the 20-feature validation used
REGION = "us-east-1"
MAX_CONCURRENT = 12
TIMEOUT = 150
MIN_BOARDS = 3                                  # skip features with <3 Opus-covered boards
LOW_CONSISTENCY = 70                            # features at/below this get flagged for review

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"throttles": 0}


def top(d, k=4):
    return ", ".join(f"{kk} {vv*100:.0f}%" for kk, vv in list(d.items())[:k]) if d else "—"


def seeblock(s):
    """One-line SEE signature. Explicitly flagged single-ply so Opus defers to the per-board
    BEST-MOVE analyses for multi-move tactics SEE cannot see."""
    return ("SEE signature (single-ply approximation; for multi-move tactics defer to the per-board BEST-MOVE analyses):\n"
            f"  player moved: {top(s.get('moved_piece_pct',{}))} | played-capture: {s.get('played_capture_pct',0)*100:.0f}% | played-check: {s.get('played_is_check_pct',0)*100:.0f}%\n"
            f"  player hung own piece: {s.get('blunder_hangs_own_pct',0)*100:.0f}% | material outcome: {top(s.get('material_kind_pct',{}))}\n"
            f"  best-move-wins-material(SEE floor): {s.get('best_wins_material_pct',0)*100:.0f}% | trajectory: {top({k:v for k,v in (s.get('trajectory_pct',{}) or {}).items() if k!='?->?'})} | phase: {top(s.get('phase_pct',{}))}")


def build(fid, prof_f, op, st):
    """Build the prompt for one feature, or None if too few Opus-covered boards."""
    s = st.get("f" + fid) or st.get(str(fid)) or {}
    lines = []
    for i, ex in enumerate(prof_f.get("examples", [])[:10]):
        k = ex["fen"] + "|" + ex["uci"]
        a = op.get(k, {}).get("analysis") if isinstance(op.get(k), dict) else None
        if isinstance(a, dict):
            lines.append(
                f"{i+1}. [{a.get('tactical_motif','?')}] played {ex['uci']}\n"
                f"   WHAT WENT WRONG: {(a.get('blunder_summary') or '')[:170]}\n"
                f"   BEST MOVE (what they should've done): {(a.get('best_moves_analysis') or '')[:170]}\n"
                f"   INTENT: {(a.get('move_intent') or '')[:110]}")
    if len(lines) < MIN_BOARDS:
        return None
    return ("Name this sparse-autoencoder chess feature: the ONE recurring mistake across its top-10 positions.\n"
            "CRITICAL: a feature's identity can be what the player MISSED (see BEST MOVE lines) as much as what they "
            "did wrong. If the best move is consistently a capture/tactic the player skipped, it's a 'Missed X' feature "
            "even if the played move also looks passive. Use the SEE signature for objective material direction "
            "(hung-own vs missed-winning), but trust the per-board BEST MOVE analyses for traps/multi-move tactics SEE can't see.\n\n"
            + seeblock(s) + "\n\nPER-POSITION ANALYSES (top 10):\n" + chr(10).join(lines) +
            '\n\nJSON: {"chip":"<2-4 words>","mistake_type":"<missed_win|hung_own|greedy|trade|positional|endgame>",'
            '"consistency":<0-100 how many of 10 fit>,"label":"<one sentence>"}')


def parse(t):
    t = t.strip()
    for p in ["```json", "```"]:
        if t.startswith(p): t = t[len(p):]
    if t.endswith("```"): t = t[:-3]
    s, e = t.find("{"), t.rfind("}") + 1
    try: return json.loads(t[s:e])
    except Exception: return None


def call(fid, prompt):
    for att in range(3):
        try:
            r = client.invoke_model(modelId=MODEL_ID, body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}]}))
            txt = "".join(b.get("text", "") for b in json.loads(r["body"].read())["content"] if b.get("type") == "text")
            p = parse(txt)
            return (fid, p if p else {"error": "parse"})
        except (ReadTimeoutError, ClientError) as e:
            if isinstance(e, ClientError) and e.response["Error"]["Code"] != "ThrottlingException":
                return (fid, {"error": str(e)[:150]})
            stats["throttles"] += 1; time.sleep(2 ** (att + 1))
    return (fid, {"error": "retries"})


ap = argparse.ArgumentParser()
ap.add_argument("--profiles", required=True)
ap.add_argument("--positions", required=True)
ap.add_argument("--seestats", required=True)
ap.add_argument("--prev", default="", help="previous integrated labels, for a changed-label diff")
ap.add_argument("--output", required=True)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

profiles = json.load(open(a.profiles))
op = json.load(open(a.positions))
st = json.load(open(a.seestats))
prev = json.load(open(a.prev)) if a.prev else {}

results = {}
if a.resume:
    try:
        results = json.load(open(a.output))
        print(f"resume: {sum(1 for v in results.values() if 'error' not in v)} already done", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

work = []
for fid in sorted(profiles.keys(), key=int):
    if fid in results and "error" not in results[fid]:
        continue
    p = build(fid, profiles[fid], op, st)
    if p is None:
        results[fid] = {"error": "insufficient_boards"}; continue
    work.append((fid, p))

print(f"to label: {len(work)} | {MODEL_ID} | conc={MAX_CONCURRENT}", flush=True)
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
    for fu in as_completed([pool.submit(call, f, p) for f, p in work]):
        fid, res = fu.result(); results[fid] = res; done += 1
        if done % 100 == 0:
            json.dump(results, open(a.output, "w"))
            print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | throttles={stats['throttles']}", flush=True)

json.dump(results, open(a.output, "w"), indent=1)
ok = {f: v for f, v in results.items() if "error" not in v}
print(f"\ndone. {len(ok)} labeled -> {a.output} | {(time.time()-t0)/60:.1f}min", flush=True)

# Summary: consistency distribution + low-consistency flags + changed-label count
from collections import Counter
cons = [v.get("consistency", 0) for v in ok.values() if isinstance(v.get("consistency"), (int, float))]
if cons:
    cons.sort()
    print(f"consistency: min {cons[0]} | median {cons[len(cons)//2]} | mean {sum(cons)/len(cons):.0f} | max {cons[-1]}")
    print(f"FLAGGED (consistency <= {LOW_CONSISTENCY}): {sum(1 for c in cons if c <= LOW_CONSISTENCY)} features need review")
mt = Counter(v.get("mistake_type", "?") for v in ok.values())
print("mistake_type spread:", dict(mt.most_common()))
if prev:
    changed = sum(1 for f, v in ok.items()
                  if v.get("chip", "").lower() != ((prev.get(f) or {}).get("chip", "")).lower())
    print(f"chip changed vs previous integrated label: {changed}/{len(ok)}")
