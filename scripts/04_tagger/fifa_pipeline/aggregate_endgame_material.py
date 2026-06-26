"""endgame_material_rates.json -> Endgame group clusters BY MATERIAL TYPE.

Each material type (Pawn, Rook, Queen, Minor, RookMinor, Heavy) becomes a SkillCluster-shaped
entry in the "Endgame" group, with per-band smoothed rates + anchors. The concept detectors
(opposition, rook-behind-passer, square rule, ...) are attached as the cluster's `features` so they
surface as drill detail inside the material cluster. Types below MIN_TOTAL_MOVES / not present in
all bands are dropped (or pooled — endgame has few types so we just drop sparse ones; "Other" rare).

Usage: python aggregate_endgame_material.py /tmp/endgame_material_rates.json /path/to/fifaSkillRatings.json
"""
import sys, json

BAND_ORDER = ["600-800","800-1000","1000-1200","1200-1400","1400-1600","1600-1800",
              "1800-2000","2000-2200","2200-2400","2400-2600","2600-2800"]
MIN_TOTAL_MOVES = 3000
MIN_BANDS_PRESENT = 9   # endgame top bands are scarce; allow a couple missing

# Display name + which existing tagger feature-labels are the drill detail for each material type.
TYPE_LABELS = {
    "Pawn": ("Pawn Endgames", [
        "Lost the Opposition", "Wrong Pawn Race", "Missed King Activity", "Missed Passed Pawn",
        "Missed Connected Passers", "Missed Outside Passer", "Missed Protected Passer",
        "Missed Square Rule", "Missed Promotion", "Missed Underpromotion", "Missed Push to Promote",
        "Missed En Passant", "Allowed Promotion", "Allowed Underpromotion", "Allowed En Passant",
    ]),
    "Rook": ("Rook Endgames", [
        "Missed Rook to 7th", "Missed Rook Cut-Off", "Missed Active Rook", "Missed Rook to Open File",
        "Rook Behind Passer", "Missed Doubled Rooks",
    ]),
    "Queen": ("Queen Endgames", []),
    "Minor": ("Minor-Piece Endgames", []),
    "RookMinor": ("Rook + Minor Endgames", []),
    "Heavy": ("Heavy-Piece Endgames", []),
}

def isotonic_min(rates):
    out, run = [], float("inf")
    for r in rates:
        if r is not None:
            run = min(run, r); out.append(round(run, 7))
        else:
            out.append(None)
    return out

def type_rates(bands_dict):
    rates, present, total = [], 0, 0
    for b in BAND_ORDER:
        c = bands_dict.get(b)
        if c and c["moves"] > 0:
            rates.append(c["blunders"] / c["moves"]); present += 1; total += c["moves"]
        else:
            rates.append(None)
    return rates, present, total

def _cluster(name, features, rates):
    smoothed = isotonic_min(rates)
    pres = [r for r in smoothed if r is not None]
    beginner = max(pres) if pres else 0.0
    master = pres[-1] if pres else 0.0
    return {"name": name, "group": "Endgame", "score": 0, "features": features,
            "anchor": {"beginner_rate": round(beginner, 7), "master_rate": round(master, 7), "n": 0},
            "smoothed_rates": smoothed}

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/endgame_material_rates.json"
    out_path = sys.argv[2]
    data = json.load(open(in_path))
    clusters = []
    skipped = []
    for mtype, (name, features) in TYPE_LABELS.items():
        bd = data.get(mtype)
        if not bd:
            skipped.append((mtype, "absent")); continue
        rates, present, total = type_rates(bd)
        if total < MIN_TOTAL_MOVES or present < MIN_BANDS_PRESENT:
            skipped.append((mtype, f"total={total} bands={present}")); continue
        clusters.append(_cluster(name, features, rates))

    fifa = json.load(open(out_path))
    # Replace the old concept-based Endgame clusters with the material clusters.
    fifa["clusters"] = [c for c in fifa["clusters"] if c.get("group") != "Endgame"]
    fifa["clusters"].extend(clusters)
    json.dump(fifa, open(out_path, "w"), indent=1)

    print(f"Endgame material clusters: {len(clusters)}")
    for c in clusters:
        rr = [round(r*1000,1) if r else 0 for r in c["smoothed_rates"]]
        mono = all(c["smoothed_rates"][i] >= c["smoothed_rates"][i+1]
                   for i in range(len(c["smoothed_rates"])-1)
                   if c["smoothed_rates"][i] is not None and c["smoothed_rates"][i+1] is not None)
        print(f"  {c['name']:<22} feat={len(c['features']):>2}  {rr}  {'MONO' if mono else 'NON-MONO'}")
    if skipped:
        print("skipped:", skipped)

if __name__ == "__main__":
    main()
