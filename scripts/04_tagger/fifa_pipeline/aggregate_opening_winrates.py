"""opening_winrates.json -> openingBandRates.json (the frontend artifact for the Openings page).

Emits the SAME shape the frontend bridge already parses (src/pages/stats/openingBandRates.ts):
  { "clusters": [ { name, group: "Openings - White"|"Openings - Black",
                    by_band: [ { band, win_rate, games, rate: null } ] } ] }
`rate` is null on purpose — this table is win-rate-only (the openings page ignores blunder rate;
that still comes from fifaSkillRatings.json for the Drill skill card, which we do NOT touch).

A family is emitted per color if it clears MIN_FAMILY_GAMES total across its populated bands (so we
don't ship 3-game "baselines"). Thin families pool into "Other" per color. Only bands with
>= MIN_BAND_GAMES are kept as cells — a 4-game band is dropped rather than shown as a baseline.

Usage: python aggregate_opening_winrates.py opening_winrates.json \
         /path/to/chess-deck-code/frontend/src/data/openingBandRates.json
"""
import sys, json

BAND_ORDER = ["600-800", "800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800",
              "1800-2000", "2000-2200", "2200-2400", "2400-2600", "2600-2800"]

# Emit a family (vs pooling into "Other") only with real volume behind it. #76 targets >=1,000
# games/major-family/band; total across bands is a looser but honest gate on "is this a real
# baseline or noise".
MIN_FAMILY_GAMES = 2000
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


def build_group(color_data, group_name):
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
        clusters.append({"name": fam, "group": group_name, "by_band": rows})
    o_rows, o_total = by_band_rows(other)
    if o_total >= MIN_FAMILY_GAMES and o_rows:
        clusters.append({"name": "Other", "group": group_name, "by_band": o_rows})
    return clusters


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src))
    clusters = (build_group(data.get("White", {}), "Openings - White")
                + build_group(data.get("Black", {}), "Openings - Black"))
    out = {"clusters": clusters}
    json.dump(out, open(dst, "w"), indent=0)
    # Report so the runner can eyeball coverage before shipping.
    print(f"Wrote {len(clusters)} clusters -> {dst}")
    for c in clusters:
        gmin = min((r["games"] for r in c["by_band"]), default=0)
        gmax = max((r["games"] for r in c["by_band"]), default=0)
        print(f"  {c['group'][9:]:6} {c['name']:28} bands={len(c['by_band']):2} games={gmin}-{gmax}")


if __name__ == "__main__":
    main()
