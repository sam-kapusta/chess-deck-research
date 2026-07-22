"""Stage 3/4: turn the re-detected rapid corpus into per-band 6-group RATES + rate-direct anchors,
and write fifaSkillRatings.json (the frontend artifact).

Inputs (on chess-poc ~/SageMaker):
  fifa_enrich.json     {fen|uci -> enrichment record}   (from redetect_positions_d16.py)
  fifa_sweep.json      [{band, fen, uci, cp_loss}]       (band membership per blunder)
  band_denominators.json {moves:{band:n}, endmoves:{band:n}}  (rapid mover-move denominators)
Tagger: ~/SageMaker/tagger_run (current — synced). Out: fifa_skill_ratings.json + console table.
"""
import os, sys, json, chess
sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
from mistake import Mistake
import tagger as T

GROUPS=["Offensive Tactics","Defensive Tactics","Calculation","Piece Safety","Positional","Endgame"]
BAND_ORDER=["600-800","800-1000","1000-1200","1200-1400","1400-1600","1600-1800","1800-2000",
            "2000-2200","2200-2400","2400-2600","2600-2800"]
MID={b:(int(b.split('-')[0])+int(b.split('-')[1]))//2 for b in BAND_ORDER}

def line_to_sans(line):
    return [t for t in (line or "").split() if not t.replace('.','').isdigit() and t not in ('1-0','0-1','1/2-1/2','*')]
def eval_to_cp(s):
    if s is None: return None
    s=str(s)
    if '#' in s or 'M' in s.upper(): return (-10000 if '-' in s else 10000)
    try: return int(s)
    except: return None
def to_group(cat, label):
    l=label.lower()
    if cat in ("Missed Tactic","Missed Mate"): return "Offensive Tactics"
    if cat=="Allowed Tactic": return "Defensive Tactics"
    if cat=="Calculation": return "Calculation"
    if cat in ("Hung Piece","Missed Capture"): return "Piece Safety"
    if cat in ("Position","Trading"): return "Positional"
    if cat=="Endgame": return "Endgame"
    if cat=="King Safety":
        if "mate" in l or "attack" in l: return "Defensive Tactics" if l.startswith("allowed") else "Offensive Tactics"
        return "Positional"
    return None

enrich=json.load(open("fifa_enrich.json")); sweep=json.load(open("fifa_sweep.json"))
denoms=json.load(open("band_denominators.json")); bmoves=denoms["moves"]; bend=denoms["endmoves"]

from collections import defaultdict, Counter
import time as _time
band_group_fires=defaultdict(Counter)   # band -> group -> moments that fired it
band_label_fires=defaultdict(Counter)   # band -> label -> moments that fired it (per-FEATURE)
label_group={}                          # label -> its group
stale={"Bad Capture","Wrong Capture","Captured With Wrong Piece","Lost Material to Combination","Wrong Move Order"}
stale_hits=Counter()
_t0=_time.time(); _n=0; _err=0
def _tag_one(row):
    """Worker: tag one position. Returns (band, groups, labels_here, stale_labels) or None."""
    ce=enrich.get(f'{row["fen"]}|{row["uci"]}')
    if not ce: return None
    try:
        fen=row["fen"]; b=chess.Board(fen); mover=b.turn
        bl=line_to_sans(ce.get("top_3_best",[{}])[0].get("line","")) if ce.get("top_3_best") else []
        bu=""
        if bl:
            try: bu=b.parse_san(bl[0]).uci()
            except: pass
        refut=line_to_sans(ce.get("top_3_refutations",[{}])[0].get("line","")) if ce.get("top_3_refutations") else []
        eb=eval_to_cp(ce.get("eval_before")); ea=eval_to_cp(ce.get("eval_after"))  # white-POV; no flip
        m=Mistake(fen_before=fen,played_uci=row["uci"],best_uci=bu,best_line_san=bl,refutation_san=refut,
                  eval_before=eb,eval_after=ea,cp_loss=int(ce.get("cp_loss",0) or 0),mover=mover,
                  played_san=ce.get("played_san",""),best_san=ce.get("best_san",""))
        groups=set(); labels_here=set(); stale_here=[]
        for t in T.tag_mistake_full(m,with_maia=False)["tags"]:
            lab=t["label"]
            if t.get("direction")=="info": continue   # orient/context tags (incl. endgame-TYPE) are
            # NOT skill mistakes — excluding them fixes the Endgame non-monotonicity (the TYPE tags
            # Rook/Pawn/Knight Endgame leak into the group via categorize's "endgame" substring and are
            # 59% of its numerator). Matches the frontend's direction!=="info" filter. (2026-06-22.)
            if lab in stale: stale_here.append(lab)
            g=to_group(T.categorize(lab,t.get("direction")),lab)
            if g: groups.add(g); labels_here.add((lab,g))
        return (row["band"], groups, labels_here, stale_here)
    except Exception:
        return "ERR"

# Parallel tagging (2026-07-17): the serial loop took ~2.5h at 25 pos/s on ONE of 64 cores; the tagger
# is pure CPU (python-chess) so a Pool gives near-linear speedup (~8-10 min at 40 workers). Aggregation
# semantics identical — workers return per-position sets, parent does all Counter updates.
from multiprocessing import Pool
print(f"tagging {len(sweep)} positions (parallel)...", flush=True)
with Pool(40) as _pool:
    for res in _pool.imap_unordered(_tag_one, sweep, chunksize=200):
        _n+=1
        if _n % 20000 == 0: print(f"  {_n}/{len(sweep)} ({_time.time()-_t0:.0f}s, {_err} errs)", flush=True)
        if res is None: continue
        if res == "ERR": _err+=1; continue
        band, groups, labels_here, stale_here = res
        for lab in stale_here: stale_hits[lab]+=1
        for g in groups: band_group_fires[band][g]+=1
        for (lab,g) in labels_here: band_label_fires[band][lab]+=1; label_group[lab]=g

# per-band per-group RATE: fires / denominator (Endgame -> endmoves, others -> moves)
bands_out={}
for band in BAND_ORDER:
    if band not in bmoves: continue
    mv=bmoves[band]; em=bend.get(band,0); row={}
    for g in GROUPS:
        denom=em if g=="Endgame" else mv
        row[g]=(band_group_fires[band][g]/denom) if denom else None
    bands_out[band]=row

# rate-direct anchors: beginner = lowest band present, master = highest band present, per group
present=[b for b in BAND_ORDER if b in bands_out]
beg_band, mas_band = present[0], present[-1]
anchors={}
for g in GROUPS:
    anchors[g]={"beginner_rate":bands_out[beg_band][g],"master_rate":bands_out[mas_band][g],
                "beginner_band":beg_band,"master_band":mas_band}

# PER-FEATURE (per-label) rates per band, same denominator rule as the group it belongs to.
# A label needs enough fires per band to be stable; flag sparse ones (min band fires < 30).
all_labels=sorted({l for band in present for l in band_label_fires[band]})
features={}
for lab in all_labels:
    g=label_group.get(lab); denom_endgame=(g=="Endgame")
    seq=[]; n_total=0
    for band in present:
        denom=bend.get(band,0) if denom_endgame else bmoves[band]
        f=band_label_fires[band][lab]; n_total+=f
        seq.append((band, f, (f/denom) if denom else None))
    rates=[r for (_,_,r) in seq if r is not None]
    mono=all(rates[i]>=rates[i+1] for i in range(len(rates)-1)) if len(rates)>1 else None
    min_band_fires=min((f for (_,f,_) in seq), default=0)
    features[lab]={"group":g,"total_fires":n_total,"min_band_fires":min_band_fires,
                   "monotonic":mono,"sparse":min_band_fires<30,
                   "by_band":{b:{"fires":f,"rate":(round(r,6) if r is not None else None)} for (b,f,r) in seq}}

out={"_version":"production_2026-06-22","_source":"rapid-only 11-band (600-2800), 60k moves/band one-scan, depth-16 re-detect, current tagger. info/orient tags EXCLUDED from scoring (fixes Endgame). Endgame denom = endgame-moves; others = total moves.",
     "_groups":GROUPS,"_band_n":{b:bmoves.get(b) for b in present},
     "_endmoves":{b:bend.get(b) for b in present},
     "anchors":anchors,"bands":bands_out,"features":features}
json.dump(out,open("fifa_skill_ratings.json","w"),indent=1)

print("STALE catch-alls (should be 0):", dict(stale_hits) or "NONE")
print(f"\n{'band':<12}"+"".join(f"{g[:9]:>11}" for g in GROUPS))
for band in present:
    print(f"{band:<12}"+"".join(f"{(bands_out[band][g]*100):>10.1f}%" if bands_out[band][g] is not None else f"{'-':>11}" for g in GROUPS))
print(f"\nanchors: beginner={beg_band} master={mas_band}")
print("monotonic check (rate should FALL beginner->master per group):")
for g in GROUPS:
    seq=[bands_out[b][g] for b in present if bands_out[b][g] is not None]
    mono = all(seq[i]>=seq[i+1] for i in range(len(seq)-1))
    print(f"  {g:<18} {'OK (falls)' if mono else 'NON-MONOTONIC'}  {[round(x*100,1) for x in seq]}")

print("\n=== PER-FEATURE monotonicity (does each individual tag fall with rating?) ===")
print(f"{'label':<32}{'group':<14}{'tot':>6}{'minbf':>7}  mono?")
for lab in sorted(features, key=lambda l:(features[l]['group'] or '', -features[l]['total_fires'])):
    fe=features[lab]
    mark = "SPARSE" if fe['sparse'] else ("OK" if fe['monotonic'] else "NON-MONO")
    seq=[fe['by_band'][b]['rate'] for b in present if fe['by_band'][b]['rate'] is not None]
    print(f"{lab:<32}{(fe['group'] or '-'):<14}{fe['total_fires']:>6}{fe['min_band_fires']:>7}  {mark:<9} {[round(x*100,2) for x in seq]}")
print("wrote fifa_skill_ratings.json")
