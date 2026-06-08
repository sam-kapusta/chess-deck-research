#!/usr/bin/env python3
"""Run the full tagger over a Stockfish-analysis corpus and save per-position tags.

  python3 run_corpus.py --sf /tmp/stockfish_data_v2.json --out output/mistake_tags.json [--maia] [--limit N]

Maia rarity (Layer 3) is OFF by default (40k ONNX calls is slow); pass --maia to include it, or
--maia-sample N to run Maia only on the first N. L1+L2 run on everything (fast, pure python-chess).

Uses the OWNED tagger (tagger.tag_mistake_full -> motifs.py). No cook_adapter.
"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from mistake import from_sf_entry
from tagger import tag_mistake_full
import maia_rarity as MR

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

t0 = time.time(); out = []; n_with_tags = 0
for i, key in enumerate(keys):
    if "|" not in key:
        continue
    fen, uci = key.rsplit("|", 1)
    m = from_sf_entry(fen, uci, sf[key], player_elo=a.player_elo, oppo_elo=a.player_elo)
    if m is None:
        continue
    use_maia = a.maia or (i < a.maia_sample)
    res = tag_mistake_full(m, with_maia=use_maia)
    out.append({
        "fen": fen, "played": uci, "best": m.best_uci,
        "played_san": m.played_san, "best_san": m.best_san,
        "cp_loss": m.cp_loss, "eval_before": m.eval_before,
        "tags": res["tags"], "categories": res["categories"],
        "maia": res["maia"],
    })
    if res["tags"]:
        n_with_tags += 1
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(keys)} | {(time.time()-t0)/60:.1f}min | tagged {n_with_tags}", flush=True)

json.dump(out, open(a.out, "w"))
print(f"\ndone. {len(out)} positions, {n_with_tags} with >=1 tag -> {a.out} | {(time.time()-t0)/60:.1f}min", flush=True)

# tag-frequency summary
from collections import Counter
c = Counter(t["label"] for o in out for t in o["tags"])
print(f"\nTOP TAGS ({len(c)} distinct):")
for lab, n in c.most_common(60):
    print(f"  {n:>6}  {lab}  ({100*n/max(len(out),1):.1f}%)")
