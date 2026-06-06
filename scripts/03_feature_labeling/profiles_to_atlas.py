#!/usr/bin/env python3
"""Adapt peak_median profiles -> the two inputs render_atlas_v3.py expects.

render_atlas_v3.py reads (a) profiles as {fid: {examples:[{fen,uci}], fire_rate}} and (b) an
optional best_map {f"{fen}|{uci}": best_uci} for the green best-move arrow. The v7 profiler
(build_peak_median_profiles.py) instead emits {fid: {peak:[{fen,uci,best,act}], median:[...]}}.

This adapter flattens peak+median into one examples list (peak first — strongest activation shown
first), pulls fire_rate from see_stats, and extracts the inline `best` into the best_map. Lets the
atlas render from committed files with no /tmp intermediates.

  python3 profiles_to_atlas.py --pm output/peak_median_profiles_d2048_k4.json \
    --stats output/see_stats_d2048_k4.json \
    --out-profiles output/atlas_profiles_d2048_k4.json --out-best output/best_uci_map_d2048_k4.json
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--pm", required=True, help="peak_median profiles")
ap.add_argument("--stats", required=True, help="see_stats (fire_rate)")
ap.add_argument("--out-profiles", required=True)
ap.add_argument("--out-best", required=True)
a = ap.parse_args()

pm = json.load(open(a.pm))
st = json.load(open(a.stats))


def fr(f):
    return (st.get("f" + f) or st.get(f) or {}).get("fire_rate", 0)


profiles, best_map = {}, {}
for f, v in pm.items():
    examples = []
    for ex in (v.get("peak", []) + v.get("median", [])):
        examples.append({"fen": ex["fen"], "uci": ex["uci"]})
        if ex.get("best"):
            best_map[ex["fen"] + "|" + ex["uci"]] = ex["best"]
    profiles[f] = {"examples": examples, "fire_rate": fr(f)}

json.dump(profiles, open(a.out_profiles, "w"))
json.dump(best_map, open(a.out_best, "w"))
print(f"wrote {a.out_profiles} ({len(profiles)} features) + {a.out_best} ({len(best_map)} best moves)")
