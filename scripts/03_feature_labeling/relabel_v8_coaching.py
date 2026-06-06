#!/usr/bin/env python3
"""Feature relabeler — v8: COACHING ALTITUDE + hard direction prior.

Two faults in v7 chips both deflated consistency and they were LABEL faults, not feature faults:
  1. Over-specificity — "Missed free QUEEN capture" when the feature is just "Missed free capture"
     (the queen was peak-noise). Measuring 10 boards against the too-narrow sentence => ~60% fit.
  2. Direction errors — "Hangs piece to pawn advance" for a feature whose best move is a capture in
     20/20 boards => it's a MISSED capture (omission), not a hung piece (self-inflicted). The old
     v1 missed-vs-hung bias, still leaking through the LLM's prose.

v8 fixes both:
  - PROMPT at coaching altitude: name the ONE broad mistake category a coach would write on the
    board, 1-3 words (chapter heading, not mechanism). "Missed capture", "Bad sacrifice",
    "Missed check", "Hangs piece", "Premature trade". Specificity is the enemy here.
  - HARD DIRECTION PRIOR computed from the board, fed as fact (not the LLM's guess): for each
    feature we compute, across its peak+median boards, how often the BEST move is a capture
    (=missed material) vs the PLAYED move hangs own material. If best-is-capture dominates, we TELL
    the model "direction = MISSED (omission)"; if played-hangs-own dominates, "direction = HUNG
    (self-inflicted)". The objective signal decides direction; the LLM only names the category.

Reads the SAME peak_median profiles + opus positions as v7. opus-4-8 xhigh.

  cd ~/SageMaker && EFFORT=xhigh python3 relabel_v8_coaching.py \
    --profiles peak_median_profiles_d64_k1.json --positions all_positions_labeled_opus.json \
    --output relabel_v8_d64_k1.json --resume
"""
import argparse, json, time, os, chess, boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed

REGION = "us-east-1"
MAX_CONCURRENT = 12
TIMEOUT = 360
MIN_BOARDS = 3
EFFORT = os.environ.get("EFFORT", "")
THINK_MODEL = os.environ.get("THINK_MODEL", "us.anthropic.claude-opus-4-8")
MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"throttles": 0}


def direction_prior(prof_f):
    """Objective direction from the board: best-move-capture (missed) vs played-hangs-own (hung).
    Returns (label, detail) where label in {MISSED, HUNG, MIXED}."""
    boards = (prof_f.get("peak", []) + prof_f.get("median", []))[:20]
    nbest_cap = nplayed_cap = n = 0
    for ex in boards:
        fen, uci, best = ex.get("fen"), ex.get("uci", ""), ex.get("best", "")
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        n += 1
        if best and len(best) >= 4:
            try:
                if b.is_capture(chess.Move.from_uci(best)):
                    nbest_cap += 1
            except Exception:
                pass
        if len(uci) >= 4:
            try:
                if b.is_capture(chess.Move.from_uci(uci)):
                    nplayed_cap += 1
            except Exception:
                pass
    if n == 0:
        return "MIXED", "no boards"
    bc, pc = nbest_cap / n, nplayed_cap / n
    detail = f"best-move-is-capture {bc*100:.0f}% of boards, played-move-is-capture {pc*100:.0f}%"
    # best is a capture on most boards AND the player declined it => MISSED (omission)
    if bc >= 0.7 and pc <= 0.4:
        return "MISSED", detail
    if bc < 0.5:
        return "HUNG", detail
    return "MIXED", detail


def board_line(i, ex, op):
    k = ex["fen"] + "|" + ex["uci"]
    a = op.get(k, {}).get("analysis") if isinstance(op.get(k), dict) else op.get(k)
    if not isinstance(a, dict):
        a = {}
    return (f"{i}. played {ex['uci']} (best {ex.get('best','?')})\n"
            f"   WRONG: {(a.get('blunder_summary') or '')[:200]}\n"
            f"   BEST: {(a.get('best_moves_analysis') or '')[:140]}")


def build(fid, prof_f, op, fire_pct):
    peak = [l for i, ex in enumerate(prof_f.get("peak", [])[:10], 1) for l in [board_line(i, ex, op)]]
    med = [l for i, ex in enumerate(prof_f.get("median", [])[:10], 1) for l in [board_line(i, ex, op)]]
    if len(peak) + len(med) < MIN_BOARDS:
        return None
    dlabel, ddetail = direction_prior(prof_f)
    dir_instr = {
        "MISSED": "DIRECTION = MISSED (omission): the best move is a capture/tactic the player SKIPPED. "
                  "Name it as a 'Missed X' — Missed capture, Missed check, Missed fork, Missed mate. "
                  "Do NOT call it 'hangs'.",
        "HUNG": "DIRECTION = HUNG (self-inflicted): the move played loses the player's OWN material. "
                "Name it as 'Hangs X' or the self-inflicted act — Hangs piece, Bad sacrifice, "
                "Premature trade, Pointless check. Do NOT call it 'missed'.",
        "MIXED": "DIRECTION = unclear from material signal; decide from the board evidence.",
    }[dlabel]
    # fire-rate note: a feature firing on many positions MUST be broad — don't let the LLM
    # over-specify a common detector (the blob lesson).
    fire_note = (f"This feature fires on {fire_pct:.1f}% of all positions — "
                 + ("very common, so it MUST be a BROAD category, not a specific tactic.\n"
                    if fire_pct >= 2 else "a moderate/narrow firing rate.\n"))
    body = ""
    if peak: body += "TOP positions (strongest):\n" + chr(10).join(peak) + "\n\n"
    if med: body += "MEDIUM positions (typical):\n" + chr(10).join(med) + "\n\n"
    return (
        "You are naming one SAE feature for a chess coaching atlas. Name the ONE broad mistake "
        "CATEGORY these positions share — the chapter heading a coach would write, NOT the mechanism. "
        "1-3 words. Examples of the right altitude: 'Missed capture', 'Missed check', 'Missed mate', "
        "'Hangs piece', 'Bad sacrifice', 'Premature trade', 'Pointless check', 'King safety', "
        "'Endgame technique'. RESIST specificity: if the boards show free captures of different "
        "pieces, the answer is 'Missed capture' — NOT 'Missed free queen'. The piece, square, and "
        "exact tactic vary board to board; name what is CONSTANT.\n\n"
        f"{dir_instr}\nObjective signal: {ddetail}.\n{fire_note}\n"
        + body +
        'JSON: {"chip":"<1-3 word category>","direction":"<missed|hung|other>",'
        '"consistency":<0-100: how many of the boards fit this category>,'
        '"label":"<one sentence: the category + how it varies across boards>"}')


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
                body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 600,
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
ap.add_argument("--seestats", default="", help="see_stats json (for per-feature fire rate)")
ap.add_argument("--output", required=True)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

profiles = json.load(open(a.profiles))
op = json.load(open(a.positions))
st = json.load(open(a.seestats)) if a.seestats else {}
def fire_of(fid):
    return (st.get("f" + fid) or st.get(fid) or {}).get("fire_rate", 0) * 100
results = {}
if a.resume:
    try:
        results = json.load(open(a.output))
        print(f"resume: {sum(1 for v in results.values() if 'error' not in v)} done", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

work = []
for fid in sorted(profiles.keys(), key=int):
    if fid in results and "error" not in results[fid]:
        continue
    p = build(fid, profiles[fid], op, fire_of(fid))
    if p is None:
        results[fid] = {"error": "insufficient_boards"}; continue
    work.append((fid, p))

print(f"to label: {len(work)} | EFFORT={EFFORT or 'off'} | conc={MAX_CONCURRENT}", flush=True)
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
    futs = [pool.submit(call, f, p) for f, p in work]
    for fu in as_completed(futs):
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
