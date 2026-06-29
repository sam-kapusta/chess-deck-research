"""opening_rates.json -> two FIFA group blocks ('Openings - White' / 'Openings - Black').

Each family becomes a SkillCluster-shaped entry:
  {name, group, score:0, features:[], anchor:{beginner_rate, master_rate, n}, smoothed_rates:[11]}
Families with < MIN_BAND_FIRES blunders in their weakest populated band pool into "Other"
(per color). Isotonic smoothing (cumulative min) makes rates non-increasing across bands.

Usage: python aggregate_opening_rates.py /tmp/opening_rates.json /path/to/fifaSkillRatings.json
"""
import sys, json

BAND_ORDER = ["600-800","800-1000","1000-1200","1200-1400","1400-1600","1600-1800",
              "1800-2000","2000-2200","2200-2400","2400-2600","2600-2800"]
# A family is scored directly if it has broad volume: every band populated AND enough total moves
# that the per-band rates (after isotonic smoothing) are trustworthy. A single thin band is fine —
# smoothing absorbs it — so we gate on TOTAL volume + full band coverage, not weakest-band fires.
MIN_TOTAL_MOVES = 3000
MIN_BANDS_PRESENT = 11

def isotonic_min(rates):
    """Cumulative min left->right: forces non-increasing (beginner worst -> master best)."""
    out, run = [], float("inf")
    for r in rates:
        if r is not None:
            run = min(run, r)
            out.append(round(run, 7))
        else:
            out.append(None)
    return out

def family_rates(bands_dict):
    """bands_dict: {band: {moves, blunders}} -> (rates[11], bands_present, total_moves)."""
    rates, bands_present, total_moves = [], 0, 0
    for b in BAND_ORDER:
        cell = bands_dict.get(b)
        if cell and cell["moves"] > 0:
            rates.append(cell["blunders"] / cell["moves"])
            bands_present += 1
            total_moves += cell["moves"]
        else:
            rates.append(None)
    return rates, bands_present, total_moves

def build_group(color_data, group_name):
    """color_data: {family: {band: {moves, blunders, games, wins, draws, losses}}} -> cluster dicts."""
    other = {b: {"moves": 0, "blunders": 0, "games": 0, "wins": 0, "draws": 0, "losses": 0} for b in BAND_ORDER}
    clusters = []
    for fam, bands_dict in color_data.items():
        rates, bands_present, total_moves = family_rates(bands_dict)
        if fam == "Unknown" or total_moves < MIN_TOTAL_MOVES or bands_present < MIN_BANDS_PRESENT:
            for b in BAND_ORDER:
                c = bands_dict.get(b)
                if c:
                    for k in ("moves", "blunders", "games", "wins", "draws", "losses"):
                        other[b][k] += c.get(k, 0)
            continue
        clusters.append(_cluster(fam, group_name, rates, bands_dict))
    # "Other" pooled family (the long tail of rare openings)
    o_rates, o_bands, o_moves = family_rates(other)
    if o_moves >= MIN_TOTAL_MOVES and o_bands >= MIN_BANDS_PRESENT:
        clusters.append(_cluster("Other", group_name, o_rates, other))
    return clusters

def _cluster(name, group, rates, bands_dict):
    smoothed = isotonic_min(rates)
    present = [r for r in smoothed if r is not None]
    beginner = max(present) if present else 0.0
    master = present[-1] if present else 0.0
    # by_band per band: fires = games-in-family volume (frontend volume-sort, issue #37);
    # rate = blunder-rate (blunders/moves); win_rate = (wins+0.5*draws)/games from this color's POV.
    # Two metrics so the Openings page can show either per selected band.
    by_band = []
    for b in BAND_ORDER:
        c = bands_dict.get(b) or {}
        moves = c.get("moves", 0); blun = c.get("blunders", 0)
        games = c.get("games", 0); wins = c.get("wins", 0); draws = c.get("draws", 0)
        rate = (blun / moves) if moves > 0 else None
        win_rate = ((wins + 0.5 * draws) / games) if games > 0 else None
        by_band.append({"band": b, "fires": games or moves,
                        "rate": (round(rate, 7) if rate is not None else None),
                        "win_rate": (round(win_rate, 5) if win_rate is not None else None),
                        "games": games})
    return {
        "name": name, "group": group, "score": 0, "features": [],
        "anchor": {"beginner_rate": round(beginner, 7), "master_rate": round(master, 7), "n": 0},
        "smoothed_rates": smoothed, "by_band": by_band,
    }

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opening_rates.json"
    out_path = sys.argv[2]
    data = json.load(open(in_path))
    white = build_group(data.get("White", {}), "Openings - White")
    black = build_group(data.get("Black", {}), "Openings - Black")
    fifa = json.load(open(out_path))
    # Drop any prior openings clusters, then append the new ones.
    fifa["clusters"] = [c for c in fifa["clusters"]
                        if c.get("group") not in ("Openings - White", "Openings - Black", "Openings")]
    fifa["clusters"].extend(white + black)
    if "_groups" in fifa:
        fifa["_groups"] = [g for g in fifa["_groups"] if g not in ("Openings", "Openings - White", "Openings - Black")]
        fifa["_groups"] += ["Openings - White", "Openings - Black"]
    json.dump(fifa, open(out_path, "w"), indent=1)
    print(f"White families: {len(white)} | Black families: {len(black)}")
    for c in (white + black):
        rr = [round(r*1000, 1) if r else 0 for r in c["smoothed_rates"]]
        print(f"  {c['group']:<18} {c['name'][:30]:<32} {rr}")

if __name__ == "__main__":
    main()
