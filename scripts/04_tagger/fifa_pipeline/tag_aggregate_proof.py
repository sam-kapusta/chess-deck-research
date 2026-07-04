"""Validate the chain: enrichment records -> Mistake -> current tagger -> 10 cats -> 6 groups -> rates.
Mirrors run_rating_bands.build_mistake EXCEPT my re-detect stores eval_before/after as WHITE-POV
(info.score.white()), so I do NOT re-flip for black (build_mistake flips mover-POV->white; mine are
already white). win_drop is computed inside Mistake from white-POV evals + mover (correct)."""
import os, sys, json, chess
sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
from mistake import Mistake
import tagger as T

def line_to_sans(line):
    return [t for t in (line or "").split() if not t.replace('.','').isdigit() and t not in ('1-0','0-1','1/2-1/2','*')]

def eval_to_cp(s):
    if s is None: return None
    s=str(s)
    if '#' in s or 'M' in s.upper(): return (-10000 if '-' in s else 10000)
    try: return int(s)
    except: return None

# 10-category -> 6 group map (the locked scheme). Endgame handled separately (denominator).
def to_group(cat, label):
    l=label.lower()
    if cat in ("Missed Tactic","Missed Mate"): return "Offensive Tactics"
    if cat=="Allowed Tactic": return "Defensive Tactics"
    if cat=="Calculation": return "Calculation"
    if cat in ("Hung Piece","Missed Capture"): return "Piece Safety"
    if cat in ("Position","Trading"): return "Positional"
    if cat=="Endgame": return "Endgame"
    if cat=="King Safety":
        # attacks/mate -> tactics by direction; castling/exposed-king -> Positional
        if "mate" in l or "attack" in l:
            return "Defensive Tactics" if l.startswith("allowed") else "Offensive Tactics"
        return "Positional"
    return None  # Meta/Other

enrich = json.load(open(sys.argv[1]))          # proof_2band_enrich.json
sweep  = json.load(open(sys.argv[2]))          # proof_2band_sweep.json
from collections import Counter, defaultdict
band_moves = Counter()                          # positions per band (denominator proxy = blunders here)
band_group = defaultdict(Counter)               # band -> group -> fires
raw_cat = defaultdict(Counter)                  # band -> 10-cat -> fires (sanity)
seen_labels = Counter()
stale = {"Bad Capture","Wrong Capture","Captured With Wrong Piece","Lost Material to Combination","Wrong Move Order"}
stale_hits = Counter()

for row in sweep:
    band = row["band"]; key = f'{row["fen"]}|{row["uci"]}'
    ce = enrich.get(key)
    if not ce: continue
    band_moves[band]+=1
    fen=row["fen"]; b=chess.Board(fen); mover=b.turn
    best_line_san = line_to_sans(ce.get("top_3_best",[{}])[0].get("line","")) if ce.get("top_3_best") else []
    best_uci=""
    if best_line_san:
        try: best_uci=b.parse_san(best_line_san[0]).uci()
        except: pass
    refut = line_to_sans(ce.get("top_3_refutations",[{}])[0].get("line","")) if ce.get("top_3_refutations") else []
    eb=eval_to_cp(ce.get("eval_before")); ea=eval_to_cp(ce.get("eval_after"))   # already WHITE-POV; do NOT flip
    m=Mistake(fen_before=fen, played_uci=row["uci"], best_uci=best_uci,
              best_line_san=best_line_san, refutation_san=refut, eval_before=eb, eval_after=ea,
              cp_loss=int(ce.get("cp_loss",0) or 0), mover=mover,
              played_san=ce.get("played_san",""), best_san=ce.get("best_san",""))
    res=T.tag_mistake_full(m, with_maia=False)
    groups=set()
    for t in res["tags"]:
        lab=t["label"]; seen_labels[lab]+=1
        if lab in stale: stale_hits[lab]+=1
        cat=T.categorize(lab, t.get("direction"))
        raw_cat[band][cat]+=1
        g=to_group(cat,lab)
        if g: groups.add(g)
    for g in groups: band_group[band][g]+=1

GROUPS=["Offensive Tactics","Defensive Tactics","Calculation","Piece Safety","Positional","Endgame"]
print("=== per-band GROUP fire RATE (fires / blunders-in-band) ===")
print(f"{'band':<12}"+ "".join(f"{g[:10]:>12}" for g in GROUPS))
for band in sorted(band_moves):
    n=band_moves[band]
    print(f"{band:<12}"+ "".join(f"{100*band_group[band][g]/n:>11.1f}%" for g in GROUPS) + f"   (n={n})")
print("\n=== STALE catch-all labels present? (should be ZERO with current tagger) ===")
print(dict(stale_hits) or "NONE — clean")
print("\n=== Greedy Capture fires:", seen_labels.get("Greedy Capture",0))
print("=== top 15 labels ===")
for lab,c in seen_labels.most_common(15): print(f"  {c:>5} {lab}")

