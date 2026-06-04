#!/usr/bin/env python3
"""Unbiased per-feature categorization — evidence in, category out. No priors.

WHY (the bias we are removing):
  assign_to_buckets.py imposed THREE biases: (1) a hardcoded RULES block mapping SEE signals to
  bucket ids, (2) a fixed 11-bucket list written top-down, (3) it fed the model the prior `chip`
  and `mistake_type`. So it could only ever confirm the taxonomy we'd already decided on.

  This script removes all three. Each feature is shown ONLY evidence:
    - its SEE signature (objective, flagged single-ply), and
    - its top-10 activating boards with the per-position Opus analyses
      (motif, what-went-wrong, best-move, intent) — the same ground truth, minus our read.
  No chip, no mistake_type, no preset category list, no rules. The model names the recurring
  pattern from scratch. Because it never sees the chip, it is also an independent check on the
  relabel.

  Bottom-up by design: emergent free-form categories here, clustered into a taxonomy in a
  SEPARATE pass (cluster_categories.py). We do not hand it the answer.

SEE is single-ply (sees the immediate capture/recapture only; blind to traps & multi-move
tactics). best_wins_material fires only when the best move WINS material — a correct trade or
defensive move scores 0. So a low best_wins does NOT mean "no better move existed". The board
analyses are the arbiter for anything multi-move; SEE is an objective floor, nothing more.

Run on chess-poc (boards live there):
  python3 emergent_categories.py --profiles d2048_k6_profiles.json \
    --positions all_positions_labeled_opus.json --seestats see_stats_d2048_k6.json \
    --output emergent_categories_d2048_k6.json [--only f1,f2,...] [--sample 40] [--resume]
"""
import argparse, json, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"
MAX_CONCURRENT = 12
TIMEOUT = 150
MIN_BOARDS = 3

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"throttles": 0}


def top(d, k=4):
    return ", ".join(f"{kk} {vv*100:.0f}%" for kk, vv in list(d.items())[:k]) if d else "—"


def seeblock(s):
    """Objective signature only. No interpretation, no leading toward an answer."""
    return ("SEE signature (single-ply objective stats; blind to multi-move tactics — see the board analyses for those):\n"
            f"  player's move: {top(s.get('moved_piece_pct',{}))} | was-a-capture: {s.get('played_capture_pct',0)*100:.0f}% | was-a-check: {s.get('played_is_check_pct',0)*100:.0f}%\n"
            f"  player's move loses own material: {s.get('blunder_hangs_own_pct',0)*100:.0f}% | material result: {top(s.get('material_kind_pct',{}))}\n"
            f"  a material-winning move was available (single-ply): {s.get('best_wins_material_pct',0)*100:.0f}% | game phase: {top(s.get('phase_pct',{}))}\n"
            f"  eval trajectory across these positions: {top({k:v for k,v in (s.get('trajectory_pct',{}) or {}).items() if k!='?->?'})}")


def build(fid, prof_f, op, st):
    s = st.get("f" + fid) or st.get(str(fid)) or {}
    lines = []
    for i, ex in enumerate(prof_f.get("examples", [])[:10]):
        k = ex["fen"] + "|" + ex["uci"]
        a = op.get(k, {}).get("analysis") if isinstance(op.get(k), dict) else None
        if isinstance(a, dict):
            lines.append(
                f"{i+1}. motif={a.get('tactical_motif','?')} | player played {ex['uci']}\n"
                f"   what happened: {(a.get('blunder_summary') or '')[:175]}\n"
                f"   stronger move available: {(a.get('best_moves_analysis') or '')[:175]}\n"
                f"   player's apparent intent: {(a.get('move_intent') or '')[:110]}")
    if len(lines) < MIN_BOARDS:
        return None
    # Deliberately NO category list, NO rules, NO prior label, NO steer toward missed-vs-made.
    return ("This is one feature of a sparse autoencoder trained on chess blunders. Below are its top-10 "
            "activating positions (every position is a position where the player blundered) plus objective "
            "stats. Your job: identify the ONE recurring kind of mistake this feature detects, and name it.\n\n"
            "Decide the category purely from the evidence. There is no fixed list to choose from — name what "
            "you actually see. Be specific about WHAT recurs (a specific tactical pattern, a specific kind of "
            "bad move, a specific missed opportunity). If the 10 positions do not share one clear pattern, say so.\n\n"
            + seeblock(s) + "\n\nTOP-10 POSITIONS:\n" + chr(10).join(lines) +
            '\n\nReturn JSON only:\n'
            '{"category":"<2-5 word name of the recurring mistake>",'
            '"description":"<one sentence: what the player repeatedly does or misses here>",'
            '"evidence":"<one sentence citing what in the boards/stats made you decide>",'
            '"consistency":<0-100, how many of the 10 share this exact pattern>}')


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
ap.add_argument("--output", required=True)
ap.add_argument("--only", default="", help="comma-separated fids")
ap.add_argument("--sample", type=int, default=0, help="diverse stratified sample of N features")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
ONLY = set(x.strip().lstrip("f") for x in a.only.split(",") if x.strip())

profiles = json.load(open(a.profiles))
op = json.load(open(a.positions))
st = json.load(open(a.seestats))


def pick_sample(n):
    """Diverse, reproducible (no RNG): stratify by (player-move-loses bucket, phase, fire-rate
    rank) so the sample spans the SEE space rather than cherry-picking. Deterministic ordering."""
    def S(f): return st.get("f" + f) or st.get(f) or {}
    fids = [f for f in profiles if (S(f).get("n") or 0)]
    def strat(f):
        s = S(f)
        hang = s.get("blunder_hangs_own_pct", 0)
        hb = 0 if hang < 0.34 else 1 if hang < 0.67 else 2
        ph = max((s.get("phase_pct") or {"?": 1}), key=(s.get("phase_pct") or {"?": 1}).get)
        return (hb, ph)
    from collections import defaultdict
    groups = defaultdict(list)
    for f in fids:
        groups[strat(f)].append(f)
    for g in groups:
        groups[g].sort(key=lambda f: -(S(f).get("fire_rate", 0)))  # high-fire first within stratum
    out, gi = [], sorted(groups)
    i = 0
    while len(out) < n and any(groups[g] for g in gi):  # round-robin across strata
        g = gi[i % len(gi)]
        if groups[g]:
            out.append(groups[g].pop(0))
        i += 1
    return out


results = {}
if a.resume:
    try:
        results = json.load(open(a.output))
        print(f"resume: {sum(1 for v in results.values() if 'error' not in v)} done", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

if ONLY:
    target = [f for f in profiles if f in ONLY]
elif a.sample:
    target = pick_sample(a.sample)
else:
    target = sorted(profiles.keys(), key=int)

work = []
for fid in target:
    if fid in results and "error" not in results[fid]:
        continue
    p = build(fid, profiles[fid], op, st)
    if p is None:
        results[fid] = {"error": "insufficient_boards"}; continue
    work.append((fid, p))

print(f"to categorize: {len(work)} | {MODEL_ID} | conc={MAX_CONCURRENT}", flush=True)
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
    for fu in as_completed([pool.submit(call, f, p) for f, p in work]):
        fid, res = fu.result(); results[fid] = res; done += 1
        if done % 100 == 0:
            json.dump(results, open(a.output, "w"))
            print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | throttles={stats['throttles']}", flush=True)

json.dump(results, open(a.output, "w"), indent=1)
ok = {f: v for f, v in results.items() if "error" not in v}
print(f"\ndone. {len(ok)} categorized -> {a.output} | {(time.time()-t0)/60:.1f}min", flush=True)
from collections import Counter
cats = Counter(v.get("category", "?") for v in ok.values())
print(f"distinct emergent categories: {len(cats)}")
for c, n in cats.most_common(40):
    print(f"  {n:>3}  {c}")
