"""opening_winrates{,_variations}.json -> openingBandRates.json (frontend Openings-page artifact).

Emits the shape the frontend bridge parses (src/pages/stats/openingBandRates.ts):
  { "clusters": [ { name, group: "Openings - White"|"Openings - Black",
                    by_band:      [ { band, win_rate, games, rate: null } ],
                    by_variation: [ { name, by_band: [ ... ] } ]   # NEW level-2 (optional) } ] }
`rate` is null on purpose — win-rate-only table (openings page ignores blunder rate; that still
comes from fifaSkillRatings.json for the Drill skill card, which we do NOT touch).

A family is emitted if it clears MIN_FAMILY_GAMES across its populated bands; thin families pool
into "Other". A VARIATION is emitted under its family if it clears MIN_VARIATION_GAMES (looser —
variations are naturally smaller); thin variations are simply dropped (family row still covers
them). Only bands with >= MIN_BAND_GAMES ship as cells.

Usage: python aggregate_opening_winrates.py opening_winrates.json \
         /path/to/chess-deck-code/frontend/src/data/openingBandRates.json \
         [opening_winrates_variations.json]   # optional; adds by_variation
"""
import sys, json

BAND_ORDER = ["600-800", "800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800",
              "1800-2000", "2000-2200", "2200-2400", "2400-2600", "2600-2800"]

# Emit a family (vs pooling into "Other") only with real volume behind it. #76 targets >=1,000
# games/major-family/band; total across bands is a looser but honest gate on "is this a real
# baseline or noise".
MIN_FAMILY_GAMES = 2000
# A variation needs less than a family (it's a slice of one) but still enough that its per-band rates
# aren't noise.
#
# 200, was 1000. A flat 1000 across all families hit small ones brutally, because it ignores how big the
# parent is: measured at band 1800 it dropped 38.5% of King's Gambit Accepted and 40.8% of Polish
# Opening (which shipped ONE variation) while costing the Sicilian only 3.8%. Page-wide, 16,746 games
# (6.8%) sat in no shipped variation at all, so a family's rows didn't add up to the family. MIN_BAND_GAMES
# still keeps individual thin CELLS out, so lowering this floor doesn't admit noisy per-band points.
MIN_VARIATION_GAMES = 200
# Drop a single band cell thinner than this — better a gap than a noisy point (the frontend already
# renders "-" for a missing cell).
MIN_BAND_GAMES = 50


def win_rate(cell):
    g = cell["games"]
    return (cell["wins"] + 0.5 * cell["draws"]) / g if g > 0 else None


def by_band_rows(bands_dict):
    """bands_dict: {band: {games, wins, draws, losses}} -> (rows[], total_games)."""
    rows, total = [], 0
    for b in BAND_ORDER:
        c = bands_dict.get(b)
        if c and c["games"] >= MIN_BAND_GAMES:
            rows.append({"band": b, "win_rate": round(win_rate(c), 4), "games": c["games"], "rate": None})
            total += c["games"]
        elif c:
            total += c["games"]  # counts toward family gate, just not shipped as a cell
    return rows, total


def build_variations(fam_var_data, family_bands=None):
    """fam_var_data: {variation: {band: cell}} -> [{name, by_band}] for variations clearing the gate.
    Sorted by total volume, with an explicit remainder row so the rows always sum to the family.

    `family_bands` is the parent family's {band: cell}. When given, whatever the gate dropped is emitted
    as a single "Other lines" variation instead of vanishing — previously a family's variations could
    account for 60% of it with no indication the rest existed."""
    out = []
    kept = {}
    for var, bands_dict in (fam_var_data or {}).items():
        if var == "Unknown":
            continue
        rows, total = by_band_rows(bands_dict)
        if total < MIN_VARIATION_GAMES or not rows:
            continue
        kept[var] = bands_dict
        out.append({"name": var, "by_band": rows, "_total": total})
    out.sort(key=lambda v: v.pop("_total"), reverse=True)

    if family_bands and out:
        # Per band: family total minus what the kept variations account for.
        rest = {}
        for b, fcell in family_bands.items():
            used = sum(kv.get(b, {}).get("games", 0) for kv in kept.values())
            uw = sum(kv.get(b, {}).get("wins", 0) for kv in kept.values())
            ud = sum(kv.get(b, {}).get("draws", 0) for kv in kept.values())
            g = fcell["games"] - used
            if g > 0:
                rest[b] = {"games": g, "wins": fcell["wins"] - uw, "draws": fcell["draws"] - ud,
                           "losses": fcell["losses"] - (used - uw - ud)}
        rows, total = by_band_rows(rest)
        if rows and total >= MIN_BAND_GAMES:
            out.append({"name": "Other lines", "by_band": rows})
    return out


def build_group(color_data, group_name, var_data):
    other = {b: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for b in BAND_ORDER}
    clusters = []
    for fam, bands_dict in color_data.items():
        rows, total = by_band_rows(bands_dict)
        if fam == "Unknown" or total < MIN_FAMILY_GAMES:
            for b in BAND_ORDER:
                c = bands_dict.get(b)
                if c:
                    for k in ("games", "wins", "draws", "losses"):
                        other[b][k] += c.get(k, 0)
            continue
        cluster = {"name": fam, "group": group_name, "by_band": rows}
        variations = build_variations(var_data.get(fam), bands_dict) if var_data else []
        if variations:
            cluster["by_variation"] = variations
        clusters.append(cluster)
    o_rows, o_total = by_band_rows(other)
    if o_total >= MIN_FAMILY_GAMES and o_rows:
        clusters.append({"name": "Other", "group": group_name, "by_band": o_rows})
    return clusters


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src))
    vdata = json.load(open(sys.argv[3])) if len(sys.argv) == 4 else {}
    clusters = (build_group(data.get("White", {}), "Openings - White", vdata.get("White", {}))
                + build_group(data.get("Black", {}), "Openings - Black", vdata.get("Black", {})))
    out = {"clusters": clusters}
    json.dump(out, open(dst, "w"), indent=0)
    # Report so the runner can eyeball coverage before shipping.
    print(f"Wrote {len(clusters)} clusters -> {dst}")
    for c in clusters:
        gmin = min((r["games"] for r in c["by_band"]), default=0)
        gmax = max((r["games"] for r in c["by_band"]), default=0)
        nv = len(c.get("by_variation", []))
        print(f"  {c['group'][9:]:6} {c['name']:28} bands={len(c['by_band']):2} games={gmin}-{gmax} vars={nv}")


if __name__ == "__main__":
    main()
