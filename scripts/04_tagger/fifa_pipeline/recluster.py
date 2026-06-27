"""Re-cluster the non-endgame/non-opening FIFA groups from a NEW cluster->features spec, recomputing
every per-band rate offline from the shipped per-feature `by_band` fires (no corpus re-pull needed).

The shipped fifaSkillRatings.json already holds, per feature: by_band fires + the band denominators
(_band_n = 200000/band for non-endgame). A cluster's band rate = sum(member fires) / denom. So any
re-grouping of existing labels is a pure offline recompute — this script does it.

Endgame + Openings clusters are produced by their own aggregators (aggregate_endgame_material.py,
aggregate_opening_rates.py) and are PRESERVED untouched.

Usage: python recluster.py SCHEME.json /path/to/fifaSkillRatings.json
SCHEME.json = {"clusters":[{"name","group","features":[...], "scoreable":bool(optional),
                            "spotlight":bool(optional)}]}
Only Offensive Tactics / Defensive Tactics / Calculation / Positional clusters are replaced.
"""
import sys, json

BAND_ORDER = ["600-800","800-1000","1000-1200","1200-1400","1400-1600","1600-1800",
              "1800-2000","2000-2200","2200-2400","2400-2600","2600-2800"]
REBUILT_GROUPS = {"Offensive Tactics","Defensive Tactics","Calculation","Positional"}

def isotonic_min(rates):
    out, run = [], float("inf")
    for r in rates:
        if r is not None:
            run = min(run, r); out.append(round(run, 7))
        else:
            out.append(None)
    return out

def build_cluster(spec, feats, denoms):
    """spec: {name, group, features, scoreable?, spotlight?}. feats: shipped per-feature dict.
    denoms: {band: denominator}. Returns a cluster dict matching the shipped schema."""
    by_band = []
    rates = []
    for b in BAND_ORDER:
        fires = sum(feats.get(f, {}).get("by_band", {}).get(b, {}).get("fires", 0)
                    for f in spec["features"])
        denom = denoms.get(b, 0)
        rate = (fires / denom) if denom else None
        by_band.append({"band": b, "fires": fires, "rate": (round(rate, 7) if rate is not None else None)})
        rates.append(rate)
    smoothed = isotonic_min(rates)
    pres = [r for r in smoothed if r is not None]
    beginner = max(pres) if pres else 0.0
    master = pres[-1] if pres else 0.0
    raw_present = [r for r in rates if r is not None]
    mono_raw = all(raw_present[i] >= raw_present[i+1] for i in range(len(raw_present)-1))
    out = {
        "name": spec["name"], "group": spec["group"], "score": 0,
        "features": spec["features"],
        "anchor": {"beginner_rate": round(beginner, 7), "master_rate": round(master, 7)},
        "smoothed_rates": smoothed,
        "by_band": by_band,
        "denom_key": "moves",
        "monotonic_raw": mono_raw,
    }
    if "scoreable" in spec: out["scoreable"] = spec["scoreable"]
    if "spotlight" in spec: out["spotlight"] = spec["spotlight"]
    return out

def main():
    scheme = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    fifa = json.load(open(out_path))
    feats = fifa["features"]
    denoms = fifa["_band_n"]

    new_clusters = [build_cluster(s, feats, denoms) for s in scheme["clusters"]
                    if s["group"] in REBUILT_GROUPS]
    # preserve Endgame + Openings clusters; replace the four tactical/positional groups
    kept = [c for c in fifa["clusters"] if c.get("group") not in REBUILT_GROUPS]
    fifa["clusters"] = kept + new_clusters
    json.dump(fifa, open(out_path, "w"), indent=1)

    # report
    print(f"Rebuilt {len(new_clusters)} clusters in {sorted(REBUILT_GROUPS)}; kept {len(kept)} (Endgame/Openings).\n")
    for grp in ["Offensive Tactics","Defensive Tactics","Calculation","Positional"]:
        print(f"### {grp}")
        for c in new_clusters:
            if c["group"] != grp: continue
            tot = sum(b["fires"] for b in c["by_band"])
            a = c["anchor"]; spread = (a["beginner_rate"] - a["master_rate"]) * 1000
            sl = "" if c.get("spotlight", True) else "  [score-only]"
            mono = "MONO" if c["monotonic_raw"] else "non-mono(raw; smoothed OK)"
            print(f"  [{tot:>6}] spread={spread:>5.1f}/k  {c['name']:<30}{sl}  {mono}")
        print()

if __name__ == "__main__":
    main()
