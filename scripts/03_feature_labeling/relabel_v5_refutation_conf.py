#!/usr/bin/env python3
"""All-fields feature relabeler — v5: v3 framing + REFUTATION evidence + piece discipline + confidence.

v5 fixes two defects found via f1536 ("Hangs knight in opening" — but only 75% knight even at peak,
and several boards hang via 2-move tactics the summary called 'tempo loss'):
  1. FEED refutation_analysis. v3 fed blunder_summary (truncated) + best_moves + intent but NOT the
     refutation line, so multi-move hangs (e.g. Nb4 6.Qa4+ pin wins it) read as 'loses tempo' and the
     hang was invisible. The hang mechanism lives in refutation_analysis. (NOTE: this is the field v4
     also added — v4 was rejected for its RESULT-FRAMING over-steer, not for refutation. v5 keeps v3's
     neutral framing and only adds the evidence.)
  2. PIECE DISCIPLINE + CONFIDENCE. v3's "name the piece" made it over-commit: f1536 is really "hangs a
     piece in the opening", not knight-specific. v5: only name a specific piece when it's the clear
     dominant piece (>=~85% of boards); otherwise say "a piece". And emit a `confidence` (high|low) +
     `review` flag when the boards don't cleanly share ONE mistake at ONE piece-specificity, so weak/
     fuzzy features are caught for human review instead of shipping a false-specific chip.

--- v3 description below ---
All-fields feature relabeler — v3: NEUTRAL direction + 5-WORD MECHANISM chips.

Same as v2 (neutral direction, all Opus fields) but the chip is allowed 3-5 words and the prompt
asks for the MECHANISM (which enemy piece captures, hangs-while-giving-check, promoted queen,
specific square/file, opening vs endgame). Reason: v2's 2-4 word chips collapsed many DISTINCT
features onto one name — 92 queen-hang features (proven distinct by decoder + corpus-firing
overlap, Jaccard~0) shared "Hangs own queen" 26x. 5-word mechanism chips cut that cluster's worst
duplicate from 26 to 8 (46 distinct chips -> 70 of 89). Readability fix, NOT dedup — same-mechanism
features stay same-named because they genuinely share the mechanism (disjoint boards).

WHY v2 EXISTS (the bias):
  relabel_all_fields.py (v1) carried a thumb on the scale: "If the best move is consistently a
  capture/tactic the player skipped, it's a 'Missed X' feature EVEN IF the played move also looks
  passive." That over-favored the omission direction. Result: 275/2035 features (13%) ended up
  direction-conflicted — chip says "Missed X" but the played move actually hangs its own material
  (player-hung-own >=70%), and 61 of those had NO win to miss at all (best-wins 0%). Both chip AND
  label inherited the bias, so chips can't be fixed from labels — only by re-deciding direction
  from evidence. This variant keeps v1's prompt structure and output format (tight chips) but
  replaces the steer with a NEUTRAL instruction: decide direction from the objective
  hung-own-vs-best-wins signal, name the dominant pattern, lower consistency when genuinely dual.

WHY THE ALL-FIELDS APPROACH (unchanged from v1):
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
                f"   WHAT WENT WRONG: {(a.get('blunder_summary') or '')[:240]}\n"
                f"   HOW IT'S PUNISHED (refutation — read this; multi-move hangs hide here, not in the summary): {(a.get('refutation_analysis') or '')[:300]}\n"
                f"   BEST MOVE (what they should've done): {(a.get('best_moves_analysis') or '')[:150]}\n"
                f"   INTENT: {(a.get('move_intent') or '')[:100]}")
    if len(lines) < MIN_BOARDS:
        return None
    return ("Name this sparse-autoencoder chess feature: the ONE recurring mistake across its top-10 positions.\n"
            "DIRECTION (decide from evidence, no default): a move that loses the player's OWN material (high 'hung own piece %', "
            "OR the refutation shows their piece falling even over 2-3 moves) is SELF-INFLICTED → 'Hangs X'. A materially-safe move "
            "that skips a win (high best-move-wins, low hung-own) is an OMISSION → 'Missed X'. Read the REFUTATION line on every "
            "board — a played move that looks like a 'tempo loss' often actually hangs a piece a couple moves later.\n"
            "PIECE SPECIFICITY — be honest, do NOT over-commit: only name a specific piece as the thing hung/missed if that piece is "
            "the CLEAR majority across the boards (≈8/10+). If the hung piece varies (knight here, bishop there, pawn elsewhere), say "
            "'a piece' — naming one piece would be false. (Lesson: a feature was wrongly called 'Hangs knight' when only ~75% were "
            "knight moves and the real invariant was just 'a piece hangs in the opening, a knight is somewhere in the tactic'.)\n"
            "RECURRING ELEMENT — if some element appears in (nearly) ALL boards but ISN'T the thing hung (e.g. a knight is always the "
            "capturing/forking piece, or it's always the opening, or always involves a check), add it as a parenthetical: "
            "e.g. 'Hangs piece in opening (knight involved)'. This separates the certain core mistake from a true-but-secondary pattern.\n"
            "CONFIDENCE — set confidence='high' only if the 10 boards clearly share ONE mistake at the stated specificity. Set "
            "confidence='low' and review=true when the boards are mixed, the piece is fuzzy, or you had to generalize — so a human "
            "reviews it later. A wrong-but-confident label (like 'Hangs knight' for a non-knight feature) is the failure to avoid.\n\n"
            + seeblock(s) + "\n\nPER-POSITION ANALYSES (top 10):\n" + chr(10).join(lines) +
            '\n\nJSON: {"chip":"<3-6 words; core mistake + optional (element involved)>","mistake_type":"<missed_win|hung_own|greedy|trade|positional|endgame>",'
            '"consistency":<0-100 how many of 10 fit the core>,"confidence":"<high|low>","review":<true|false>,"label":"<one sentence>"}')


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
ap.add_argument("--only", default="", help="comma-separated fids to label (spot-check subset)")
a = ap.parse_args()
ONLY = set(x.strip() for x in a.only.split(",") if x.strip())

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
    if ONLY and fid not in ONLY:
        continue
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
