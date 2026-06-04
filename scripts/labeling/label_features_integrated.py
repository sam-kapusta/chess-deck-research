#!/usr/bin/env python3
"""Integrated feature labeler: Opus names each feature from THREE grounded signals.

Lesson learned (f91): SEE alone mis-reads a TRADE as a material loss and is blind to
positional/trajectory mistakes. The fix is to integrate:
  1. SEE descriptive stats (precomputed, per-feature, >=0.7max cohort): moved/captured piece,
     net-material kind (trade vs loses vs hangs), missed-material, phase, played-check.
  2. Eval TRAJECTORY (winning->drawn, drawn->losing...) — what the mistake cost, player POV.
  3. Opus per-position TACTICAL MOTIF + tags, tallied over the feature's top positions — the
     positional/strategic layer SEE can't see (e.g. queen_trade_error, king_safety).
Plus the top-N boards for the eyeball check.

Reads see_stats (with material_kind_pct, trajectory_pct), all_positions_labeled_opus, profiles.
Usage (on chess-poc):
  AWS_PROFILE=default python label_features_integrated.py --profiles d1024_k4_profiles.json \
    --positions all_positions_labeled_opus.json --seestats see_stats_d1024_k4.json \
    --output feature_labels_integrated_d1024_k4.json --nshow 12 --resume
"""
import argparse, json, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"; REGION = "us-east-1"
MAX_CONCURRENT = 16; TIMEOUT = 120
client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}))
stats = {"throttles": 0}

def top(dist, k=4):
    if not dist: return "—"
    return ", ".join(f"{kk} {vv*100:.0f}%" for kk, vv in list(dist.items())[:k])

def opus_motifs(examples, analyses, ncap=40):
    """Tally tactical_motif + tags over a feature's example positions (uses up to ncap)."""
    mot = Counter(); tags = Counter(); summaries = []
    for ex in examples[:ncap]:
        key = f"{ex['fen']}|{ex['uci']}"
        a = analyses.get(key, {}).get("analysis") if isinstance(analyses.get(key), dict) else None
        if isinstance(a, dict):
            mot[a.get("tactical_motif", "?")] += 1
            for t in (a.get("tags") or [])[:6]: tags[t] += 1
            if len(summaries) < 4 and a.get("blunder_summary"):
                summaries.append(a["blunder_summary"][:180])
    return mot, tags, summaries

PROMPT = """You are naming an SAE feature that fires on ONE recurring type of chess mistake.
Name it from THREE integrated signals (do not rely on any one alone):

=== 1. SEE MATERIAL/POSITION SIGNATURE (objective, over top {ncohort} activating positions) ===
player moved: {moved} | player captured: {captured} | played a check: {played_check:.0f}%
material outcome of the move: {material}  (net material median: {net})
   -> "trade" = captured AND lost ~equal (an EXCHANGE, NOT a blunder of material);
      "loses" = gave up more than captured;  "hangs" = non-capture that drops a piece;  "safe" = nothing hangs.
Maia's move: {bestpiece} | Maia captures: {bestcap} | Maia gives check: {bestcheck:.0f}%
missed winning material: {missed:.0f}% | phase: {phase}

=== 2. EVAL TRAJECTORY (what the mistake cost, from the player's perspective) ===
{traj}   (median eval lost: {evaldrop} centipawns)

=== 3. OPUS PER-POSITION TACTICAL ANALYSIS (the positional/strategic layer) ===
dominant motifs: {motifs}
common tags: {tags}
example summaries:
{summaries}

=== TOP {nshow} BOARDS (eyeball the concrete pattern) ===
{boards}

CRITICAL GUIDANCE:
- If material outcome is mostly "trade", this is an EXCHANGE decision, NOT a material blunder.
  Combine with motif (e.g. queen_trade_error) and trajectory: a queen trade that goes
  winning->drawn is a "Premature Queen Trade", not a "Greedy Capture" and not "Hung Queen".
- Let the trajectory sharpen it: "threw a win" (winning->drawn/losing) vs "made a bad position worse".
- The motif/tags carry positional mistakes SEE can't see (attack abandoned, king safety, simplification).

Respond JSON:
{{"chip":"<2-4 word name>","label":"<one sentence>","mistake_type":"<material|tactical|positional|trade|endgame>","description":"<paragraph citing the signals>","confidence":<0-100>}}"""

def build(fid, prof_f, analyses, st):
    s = st.get(f"f{fid}") or st.get(str(fid))
    if not s: return None
    mot, tags, summ = opus_motifs(prof_f.get("examples", []), analyses)
    boards = []
    for ex in prof_f.get("examples", [])[:NSHOW]:
        key = f"{ex['fen']}|{ex['uci']}"
        a = analyses.get(key, {}).get("analysis") if isinstance(analyses.get(key), dict) else {}
        boards.append(f"  FEN {ex['fen']} played {ex['uci']}"
                      + (f" | motif {a.get('tactical_motif')}" if isinstance(a, dict) and a.get('tactical_motif') else ""))
    if len(boards) < 3: return None
    return PROMPT.format(
        ncohort=s.get('n','?'), moved=top(s.get('moved_piece_pct',{})), captured=top(s.get('captured_piece_pct',{})),
        played_check=s.get('played_is_check_pct',0)*100, material=top(s.get('material_kind_pct',{})),
        net=s.get('net_material_median','?'), bestpiece=top(s.get('best_piece_pct',{})),
        bestcap=top(s.get('best_captured_piece_pct',{})), bestcheck=s.get('best_is_check_pct',0)*100,
        missed=s.get('best_wins_material_pct',0)*100, phase=top(s.get('phase_pct',{})),
        traj=top({k:v for k,v in s.get('trajectory_pct',{}).items() if k!='?->?'}), evaldrop=s.get('eval_drop_median','?'),
        motifs=top({k:v/sum(mot.values()) for k,v in mot.most_common()} if mot else {}),
        tags=", ".join(f"{t}({c})" for t,c in tags.most_common(8)) or "—",
        summaries="\n".join(f"  - {x}" for x in summ) or "  (none)",
        nshow=len(boards), boards="\n".join(boards))

def parse(t):
    t = t.strip()
    for p in ["```json", "```"]:
        if t.startswith(p): t = t[len(p):]
    if t.endswith("```"): t = t[:-3]
    s, e = t.find("{"), t.rfind("}")+1
    try: return json.loads(t[s:e])
    except: return None

def call(fid, prompt):
    for att in range(3):
        try:
            r = client.invoke_model(modelId=MODEL_ID, body=json.dumps({
                "anthropic_version":"bedrock-2023-05-31","max_tokens":2000,
                "messages":[{"role":"user","content":prompt}]}))
            txt = "".join(b.get("text","") for b in json.loads(r["body"].read())["content"] if b.get("type")=="text")
            p = parse(txt)
            return (fid, p if p else {"error":"parse"})
        except (ReadTimeoutError, ClientError) as e:
            if isinstance(e, ClientError) and e.response["Error"]["Code"] != "ThrottlingException":
                return (fid, {"error": str(e)[:150]})
            stats["throttles"] += 1; time.sleep(2**(att+1))
    return (fid, {"error":"retries"})

ap = argparse.ArgumentParser()
ap.add_argument("--profiles", required=True); ap.add_argument("--positions", required=True)
ap.add_argument("--seestats", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--nshow", type=int, default=12); ap.add_argument("--resume", action="store_true")
ap.add_argument("--dump-prompts", default="", help="if set, write {fid: prompt} JSON here and exit (no LLM calls)")
a = ap.parse_args(); NSHOW = a.nshow
profiles = json.load(open(a.profiles)); analyses = json.load(open(a.positions)); st = json.load(open(a.seestats))
if a.dump_prompts:
    prompts = {}
    for fid in sorted(profiles.keys(), key=int):
        p = build(fid, profiles[fid], analyses, st)
        if p is not None: prompts[fid] = p
    json.dump(prompts, open(a.dump_prompts, "w"), indent=1)
    print(f"dumped {len(prompts)} prompts -> {a.dump_prompts}"); raise SystemExit
results = {}
if a.resume:
    try: results = json.load(open(a.output)); print(f"resume: {sum(1 for v in results.values() if 'error' not in v)} done")
    except (FileNotFoundError, json.JSONDecodeError): pass
work = []
for fid in sorted(profiles.keys(), key=int):
    if fid in results and "error" not in results[fid]: continue
    p = build(fid, profiles[fid], analyses, st)
    if p is None: results[fid] = {"error":"insufficient"}; continue
    work.append((fid, p))
print(f"to label: {len(work)} | Opus 4.6 | conc={MAX_CONCURRENT}", flush=True)
t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
    for fu in as_completed([pool.submit(call, f, p) for f, p in work]):
        fid, res = fu.result(); results[fid] = res; done += 1
        if done % 50 == 0:
            json.dump(results, open(a.output,"w"))
            print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | throttles={stats['throttles']}", flush=True)
json.dump(results, open(a.output,"w"), indent=1)
print(f"done. {sum(1 for v in results.values() if 'error' not in v)} labeled -> {a.output}", flush=True)
