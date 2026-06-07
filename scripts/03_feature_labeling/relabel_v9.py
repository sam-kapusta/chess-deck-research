#!/usr/bin/env python3
"""Feature relabeler — v9: editable system prompt + best-move-aware SEE block.

Two changes from v7/v8:
  1. SYSTEM PROMPT lives in prompts/label_system.txt (you own the wording, edit it freely).
  2. The SEE block now surfaces the BEST-MOVE signals that decide direction and category — and that
     v7/v8 were blind to. The old block showed what the PLAYER did (played_capture, played_check)
     but not what the BEST move was, so the labeler couldn't see "best move is a check 100% of the
     time" => Missed Check. v9 splits the block into "WHAT YOU PLAYED" vs "WHAT WAS BEST", because
     missed-vs-hung and the mistake category live in the best move. Internal SAE diagnostics
     (at_0.8, cohort, max_act, own_hang_median) are dropped as noise.

Feeds peak (top-10) + median (typical-10) boards, same as v7. opus-4-8 xhigh.

  cd ~/SageMaker && EFFORT=xhigh python3 relabel_v9.py \
    --profiles peak_median_profiles_d64_k1.json --positions all_positions_labeled_opus.json \
    --seestats see_stats_d64_k1.json --system prompts/label_system.txt \
    --output relabel_v9_d64_k1.json --resume
"""
import argparse, json, time, os, boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed

REGION = "us-east-1"; MAX_CONCURRENT = 12; TIMEOUT = 360; MIN_BOARDS = 3
EFFORT = os.environ.get("EFFORT", "")
THINK_MODEL = os.environ.get("THINK_MODEL", "us.anthropic.claude-opus-4-8")
MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"throttles": 0}


def top(d, k=3):
    if not isinstance(d, dict) or not d:
        return "—"
    return ", ".join(f"{kk} {vv*100:.0f}%" for kk, vv in sorted(d.items(), key=lambda x: -x[1])[:k])


def seeblock(s):
    """Board-derived signature, split by WHO acted. Direction + category live in the BEST move,
    so those fields lead. All percentages are over the feature's activating boards."""
    fr = s.get("fire_rate", 0) * 100
    traj = top({k: v for k, v in (s.get("trajectory_pct", {}) or {}).items() if k != "?->?"})
    return (
        f"FEATURE STATS (over its activating boards; fires on {fr:.1f}% of all positions):\n"
        f"  WHAT YOU PLAYED: piece moved {top(s.get('moved_piece_pct',{}))} | "
        f"was a capture {s.get('played_capture_pct',0)*100:.0f}% | was a check {s.get('played_is_check_pct',0)*100:.0f}% | "
        f"lost your own material {s.get('blunder_hangs_own_pct',0)*100:.0f}% | outcome {top(s.get('material_kind_pct',{}))}\n"
        f"  WHAT WAS BEST (this decides Missed-vs-Hung): best move was a capture {s.get('best_is_capture_pct',0)*100:.0f}% | "
        f"was a check {s.get('best_is_check_pct',0)*100:.0f}% | wins material {s.get('best_wins_material_pct',0)*100:.0f}% | "
        f"best piece {top(s.get('best_piece_pct',{}))} | best captures {top(s.get('best_captured_piece_pct',{}))}\n"
        f"  CONTEXT: median eval swing {s.get('eval_drop_median',0):.0f}cp | phase {top(s.get('phase_pct',{}))} | trajectory {traj}\n"
        "  (Read: high 'best move was a check/capture' + low 'you lost own material' => an OMISSION "
        "you should name 'Missed X'. High 'lost your own material' => self-inflicted, name 'Hung X' / "
        "'Bad sacrifice' / 'Premature trade'. Name the piece only if it dominates; else stay general.)")


def board_line(i, ex, op):
    k = ex["fen"] + "|" + ex["uci"]
    a = op.get(k, {}).get("analysis") if isinstance(op.get(k), dict) else op.get(k)
    if not isinstance(a, dict):
        a = {}
    return (f"{i}. [{a.get('tactical_motif','?')}] played {ex['uci']} (best {ex.get('best','?')})\n"
            f"   WHAT WENT WRONG: {(a.get('blunder_summary') or '')[:220]}\n"
            f"   HOW IT'S PUNISHED: {(a.get('refutation_analysis') or '')[:240]}\n"
            f"   BEST MOVE: {(a.get('best_moves_analysis') or '')[:140]}")


def build(fid, prof_f, op, st, system):
    s = st.get("f" + fid) or st.get(str(fid)) or {}
    peak = [board_line(i, ex, op) for i, ex in enumerate(prof_f.get("peak", [])[:10], 1)]
    med = [board_line(i, ex, op) for i, ex in enumerate(prof_f.get("median", [])[:10], 1)]
    if len(peak) + len(med) < MIN_BOARDS:
        return None
    body = ""
    if peak: body += "TOP positions (strongest activation):\n" + chr(10).join(peak) + "\n\n"
    if med: body += "MEDIAN positions (typical activation):\n" + chr(10).join(med) + "\n\n"
    return (system.strip() + "\n\n" + seeblock(s) + "\n\n" + body +
            'Return JSON: {"chip":"<the feature title>","direction":"<missed|hung|other>",'
            '"consistency":<0-100: how many of the 20 boards fit this title>,'
            '"label":"<one sentence: the pattern + the top-vs-median difference>"}')


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
            if EFFORT:
                body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16000,
                        "thinking": {"type": "adaptive"}, "output_config": {"effort": EFFORT},
                        "messages": [{"role": "user", "content": prompt}]}
                model = THINK_MODEL
            else:
                body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 800,
                        "messages": [{"role": "user", "content": prompt}]}
                model = MODEL_ID
            r = client.invoke_model(modelId=model, body=json.dumps(body))
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
ap.add_argument("--system", required=True, help="path to the editable system-prompt text file")
ap.add_argument("--output", required=True)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--only", default="", help="comma-separated fids to label (test subset)")
a = ap.parse_args()
ONLY = set(x.strip() for x in a.only.split(",") if x.strip())

system = open(a.system).read()
profiles = json.load(open(a.profiles)); op = json.load(open(a.positions)); st = json.load(open(a.seestats))
print(f"system prompt ({len(system)} chars):\n  {system.strip()[:200]}...\n", flush=True)

results = {}
if a.resume:
    try:
        results = json.load(open(a.output))
        print(f"resume: {sum(1 for v in results.values() if 'error' not in v)} done", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

work = []
for fid in sorted(profiles.keys(), key=int):
    if ONLY and fid not in ONLY:
        continue
    if fid in results and "error" not in results[fid]:
        continue
    p = build(fid, profiles[fid], op, st, system)
    if p is None:
        results[fid] = {"error": "insufficient_boards"}; continue
    work.append((fid, p))

print(f"to label: {len(work)} | EFFORT={EFFORT or 'off'} | conc={MAX_CONCURRENT}", flush=True)
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
    for fu in as_completed([pool.submit(call, f, p) for f, p in work]):
        fid, res = fu.result(); results[fid] = res; done += 1
        if done % 20 == 0:
            json.dump(results, open(a.output, "w"))
            print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | throttles={stats['throttles']}", flush=True)

json.dump(results, open(a.output, "w"), indent=1)
ok = {f: v for f, v in results.items() if "error" not in v}
cons = sorted(v.get("consistency", 0) for v in ok.values() if isinstance(v.get("consistency"), (int, float)))
if cons:
    print(f"\ndone. {len(ok)} labeled | consistency: min {cons[0]} median {cons[len(cons)//2]} "
          f"mean {sum(cons)/len(cons):.0f} max {cons[-1]} | >=80: {sum(1 for c in cons if c>=80)}", flush=True)
from collections import Counter
print("direction:", dict(Counter(v.get("direction", "?") for v in ok.values())))
