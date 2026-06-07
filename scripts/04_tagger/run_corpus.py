#!/usr/bin/env python3
"""Run the full tagger over a Stockfish-analysis corpus and save per-position tags.

  python3 run_corpus.py --sf /tmp/stockfish_data_v2.json --out output/mistake_tags.json [--maia] [--limit N]

Maia rarity (Layer 3) is OFF by default (40k ONNX calls is slow); pass --maia to include it, or
--maia-sample N to run Maia only on the first N. L1+L2 run on everything (fast, pure python-chess).
"""
import argparse, json, sys, os, time, chess
sys.path.insert(0, os.path.dirname(__file__))
from mistake import from_sf_entry
import cook_adapter as CA
import predicates as PR
from tagger import MOTIF_LABEL, DIR_PREFIX, categorize

ap = argparse.ArgumentParser()
ap.add_argument("--sf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--player-elo", type=int, default=1500)
ap.add_argument("--maia", action="store_true", help="run Maia rarity on all (slow)")
ap.add_argument("--maia-sample", type=int, default=0, help="run Maia only on first N")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

sf = json.load(open(a.sf))
keys = list(sf.keys())
if a.limit:
    keys = keys[:a.limit]

MR = None
if a.maia or a.maia_sample:
    sys.path.insert(0, os.path.dirname(__file__))
    import maia_rarity as MR

t0 = time.time(); out = []; n_with_tags = 0
for i, key in enumerate(keys):
    if "|" not in key:
        continue
    fen, uci = key.rsplit("|", 1)
    m = from_sf_entry(fen, uci, sf[key], player_elo=a.player_elo, oppo_elo=a.player_elo)
    if m is None:
        continue
    fine = []
    for motif, direction, ev in CA.tag_mistake(m):
        label = MOTIF_LABEL.get(motif, motif)
        disp = f"{DIR_PREFIX.get(direction,'')} {label}".strip()
        fine.append((disp, direction, ev, "tactic"))
    for label, direction, ev in PR.tag_predicates(m):
        fine.append((label, direction, ev, "position"))
    seen = set(); tags = []
    for t in fine:
        if t[0] in seen: continue
        seen.add(t[0]); tags.append({"label": t[0], "direction": t[1], "evidence": t[2], "layer": t[3]})
    maia = {}
    if MR and (a.maia or i < a.maia_sample):
        try: maia = MR.rarity(m)
        except Exception: maia = {}
    out.append({
        "fen": fen, "played": uci, "best": m.best_uci,
        "played_san": m.played_san, "best_san": m.best_san,
        "cp_loss": m.cp_loss, "eval_before": m.eval_before,
        "tags": tags, "categories": sorted({categorize(t["label"]) for t in tags}),
        "maia": maia,
    })
    if tags: n_with_tags += 1
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(keys)} | {(time.time()-t0)/60:.1f}min | tagged {n_with_tags}", flush=True)

json.dump(out, open(a.out, "w"))
print(f"\ndone. {len(out)} positions, {n_with_tags} with >=1 tag -> {a.out} | {(time.time()-t0)/60:.1f}min", flush=True)

# tag-frequency summary
from collections import Counter
c = Counter(t["label"] for o in out for t in o["tags"])
print(f"\nTOP TAGS ({len(c)} distinct):")
for lab, n in c.most_common(40):
    print(f"  {n:>6}  {lab}")
