"""Dedicated per-(family, color, band) opening WIN-RATE table from the Lichess rapid corpus.

Issue #76: the openings-page band baselines currently piggyback on the FIFA blunder scan
(pull_opening_rates.py), which budgets 60k *moves* per band → only ~2,400 games/band → 5-128
games per opening per band. Noise, not baselines.

This scan is HEADERS-ONLY — it reads Opening + Result + WhiteElo + BlackElo and never touches
movetext or engine evals. The openings page consumes only win_rate + games (blunder `rate` is
unused there), so we can skip everything expensive and pull ~100x more games for the same time.

Family = Opening column before ':' (matches pull_opening_rates.family_of and the frontend's
familyOf/rootToFifaFamily). Each game contributes ONE outcome per color, banded by THAT color's
Elo, win% = (wins + 0.5*draws) / games from that color's perspective.

Output: opening_winrates.json =
  {"White": {family: {band: {games, wins, draws, losses}}}, "Black": {...}}
then aggregate_opening_winrates.py turns it into the frontend's openingBandRates.json.

Run on chess-poc:  export HF_TOKEN=...; python -u pull_opening_winrates.py
Tune GAME_TARGET / MAX_SHARDS below. No GPU, no Stockfish — pure header scan.
"""
import os, re, json, time
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from collections import defaultdict

REPO = "Lichess/standard-chess-games"
BANDS = [('600-800', 600, 800), ('800-1000', 800, 1000), ('1000-1200', 1000, 1200),
         ('1200-1400', 1200, 1400), ('1400-1600', 1400, 1600), ('1600-1800', 1600, 1800),
         ('1800-2000', 1800, 2000), ('2000-2200', 2000, 2200), ('2200-2400', 2200, 2400),
         ('2400-2600', 2400, 2600), ('2600-2800', 2600, 2800)]

# GAMES (not moves) counted toward each band, per COLOR. #76 wants >=1,000 games/major-family/band;
# a family is a fraction of all games, so the band total must be well above that. 100k games/band
# puts the big families (Sicilian, Caro-Kann, QP) in the 5k-30k range and most named families over
# 1k. The scarce anchor bands (600-800, 2600-2800) won't hit target — take what exists, log it.
GAME_TARGET = 100_000
MAX_SHARDS = 200  # safety cap; headers-only shards fill bands fast, but low/high bands are rare


def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m:
        return False
    b, i = int(m.group(1)), int(m.group(2))
    return 480 <= b + 40 * i < 1500


def band_of(e):
    for n, lo, hi in BANDS:
        if lo <= e < hi:
            return n
    return None


def family_of(opening):
    if not opening:
        return "Unknown"
    name = opening.split(":")[0].strip()
    return name if name and name != "Unknown Opening" else "Unknown"


def result_pts(res, is_white):
    """Mover-color result -> (win, draw, loss) one-hot from THAT color's perspective."""
    if res == "1-0":
        return (1, 0, 0) if is_white else (0, 0, 1)
    if res == "0-1":
        return (0, 0, 1) if is_white else (1, 0, 0)
    if res == "1/2-1/2":
        return (0, 1, 0)
    return None


files = [f for f in list_repo_files(REPO, repo_type="dataset")
         if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$', f)]
files.sort(reverse=True)

# stats[color][family][band] = [games, wins, draws, losses]
stats = {"White": defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])),
         "Black": defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))}
band_games = {b: 0 for b, _, _ in BANDS}
t0 = time.time()
ng = 0


def done():
    return all(band_games[b] >= GAME_TARGET for b, _, _ in BANDS)


for fi, f in enumerate(files):
    if fi >= MAX_SHARDS or done():
        break
    try:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        # Headers only — no movetext column. This is the whole speed win.
        t = pq.read_table(p, columns=['WhiteElo', 'BlackElo', 'TimeControl', 'Opening', 'Result'])
    except Exception as e:
        print('skip', str(e)[:60], flush=True)
        continue
    cols = [t.column(c).to_pylist() for c in ['WhiteElo', 'BlackElo', 'TimeControl', 'Opening', 'Result']]
    for we, be, tc, ope, res in zip(*cols):
        ng += 1
        if not we or not be or not is_rapid(tc):
            continue
        wb, bb = band_of(we), band_of(be)
        # Skip only if BOTH colors' bands are full (or out of range) — else we'd drop a game that
        # still helps the other color's band.
        if (wb is None or band_games[wb] >= GAME_TARGET) and (bb is None or band_games[bb] >= GAME_TARGET):
            continue
        fam = family_of(ope)
        for is_white, band in ((True, wb), (False, bb)):
            if band is None or band_games[band] >= GAME_TARGET:
                continue
            pts = result_pts(res, is_white)
            if pts is None:
                continue
            band_games[band] += 1
            cell = stats["White" if is_white else "Black"][fam][band]
            cell[0] += 1
            cell[1] += pts[0]
            cell[2] += pts[1]
            cell[3] += pts[2]
    try:
        os.remove(p)
    except OSError:
        pass
    import shutil
    cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--Lichess--standard-chess-games/blobs")
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
    print(f'shard {fi + 1} | {ng}g {time.time() - t0:.0f}s | ' +
          ' '.join(f'{b.split("-")[0]}:{band_games[b]}' for b, _, _ in BANDS), flush=True)

out = {color: {fam: {b: {"games": stats[color][fam][b][0], "wins": stats[color][fam][b][1],
                         "draws": stats[color][fam][b][2], "losses": stats[color][fam][b][3]}
                     for b in stats[color][fam]}
               for fam in stats[color]}
       for color in ("White", "Black")}
json.dump(out, open("opening_winrates.json", "w"))
print(f"=== DONE === {(time.time() - t0) / 60:.1f}min | bands: " +
      ' '.join(f'{b}:{band_games[b]}' for b, _, _ in BANDS), flush=True)
