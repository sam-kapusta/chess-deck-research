"""Dedicated per-(family, color, band) opening WIN-RATE table from the Lichess BLITZ corpus.

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

Time class is BLITZ (was rapid until 2026-07-28) so the whole product shares ONE rating scale —
Lichess blitz, which is also Maia 3's conditioning scale. See is_blitz() for the full rationale.

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
# a family is a fraction of all games, so the band total must be well above that. The scarce anchor
# bands (600-800, 2600-2800) won't hit target — take what exists, log it.
#
# Raised 100k -> 500k (2026-07-29, issue #82). At 100k the openings page was missing nine families a
# coach would name as common (Réti, Catalan, Trompowsky, Torre, Veresov, Grünfeld, Benko, Budapest,
# London-as-a-family): each landed at 200-1,800 games TOTAL, under the aggregator's
# MIN_FAMILY_GAMES=2000, so they pooled into "Other" and their games showed up under Queen's Pawn.
# Lowering that floor instead would have admitted ~50 junk families (Elephant Gambit, Grob, Ware, "?")
# alongside them, so the fix is more sample, not a looser gate — earn the volume at 2000.
#
# Note the interaction with the per-band cap below (`band_games[band] >= GAME_TARGET`): once a band
# fills it stops ACCEPTING games, so at 100k the mid bands capped on shard 1-3 and every later shard
# only fed 2400/2600. Raising the target is what lets the mid bands keep filling.
GAME_TARGET = 500_000
# Blitz is ~3x denser than rapid per shard, so bands fill faster than the old rapid run needed. The
# close-rated filter still drops ~2/3 of games. Measured on the 2026-07-29 run: 40 shards = 56.2M
# games scanned in 6.4min (~1.4M/shard, ~10s/shard), which filled every band to 100k except
# 2600-2800 (50,083 — that band yields only ~1.25k close-rated blitz per shard, so it is
# shard-limited, not target-limited, and will stay short at any target).
# 5x the target needs ~5x the shards for the mid bands; 200 is sized for that with headroom.
MAX_SHARDS = 200

# CLOSE-RATED ONLY. Banding is by the player's OWN Elo, ignoring the opponent's — and the opponent
# pool is badly asymmetric at the extremes: measured on one 2025-09 shard, 2600-2800 players average
# +334 Elo ABOVE their opponents (win 68% in every opening) while 600-800 average -54 BELOW. That
# gradient swamped the opening signal: every family showed the same fake +31pt "improvement" across
# bands. Requiring a near-equal opponent removes the mismatch at the source, so each band's baseline
# sits near 50% and an opening's deviation is real. Costs sample at the thin top bands.
MAX_ELO_GAP = 100


# TIME CLASS: blitz, not rapid (changed 2026-07-28). ONE rating scale across the whole product —
# Maia 3 is conditioned on Lichess BLITZ, so making the corpus blitz too means a single conversion
# target everywhere (frontend `maiaEloConversion` already maps any platform/time-class → Li-Blitz)
# instead of Maia-on-blitz + bands-on-rapid, which is how players ended up benchmarked two bands off.
# Blitz is also far better sampled: on one 2025-09 shard, 667,926 blitz vs 208,782 rapid games, and
# the gap widens exactly where rapid starved — close-rated 2400-2600: 3,951 blitz vs 171 rapid;
# 2600-2800: 674 vs 8. Lichess time-class boundary: blitz = 180s <= est < 480s (est = base + 40*inc).
def is_blitz(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m:
        return False
    b, i = int(m.group(1)), int(m.group(2))
    return 180 <= b + 40 * i < 480


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


# ── Variation-level bucketing ───────────────────────────────────────────────
# FAITHFUL PORT of the frontend's variationName() (openingRepertoireData.ts). The player's tree
# labels variations with that function; the corpus MUST bucket by the identical key or the band
# lookup won't join (STANDARDS.md: "algorithms in both Python and TS stay in sync"). Verified
# empirically against the TS on a corpus sample (verify_variation_keys) before the production run.
FAMILY_PREFIXES = [
    "Caro Kann Defense", "Caro-Kann Defense", "Sicilian Defense", "French Defense", "Scandinavian Defense",
    "Pirc Defense", "Alekhine Defense", "Philidor Defense", "Slav Defense", "Nimzo-Indian Defense",
    "King's Indian Defense", "Queen's Gambit Declined", "Queen's Gambit Accepted", "Queen's Gambit",
    "Queen's Pawn Game", "Vienna Game", "Italian Game", "Scotch Game", "Ruy Lopez", "English Opening",
    "Bishop's Opening", "Petrov's Defense", "Russian Game", "London System", "Dutch Defense", "Bird Opening",
    "Bird's Opening", "Vienna", "Caro-Kann", "Caro Kann",
]
TWO_WORD_HEADS = [
    "main line", "max lange", "two knights", "panov attack", "fried liver",
    "kings indian", "queens indian", "nimzo indian", "kings gambit", "queens gambit",
    "bishops opening", "scotch gambit", "danish gambit", "evans gambit", "smith morra",
]
# step 1: cut from the first move-notation token onward ("3...cxd5", "4.Nf3", "...5.c3", "2.d4")
_MOVE_TAIL_A = re.compile(r'\s+(\.{3})?\s*\d+\s*\.{1,3}.*$')
_MOVE_TAIL_B = re.compile(r'\s+\.{3}.*$')


def variation_of(opening):
    """Port of variationName(): derive the level-2 variation head from a full opening name."""
    if not opening:
        return "Main Line"
    s = opening.strip()
    s = _MOVE_TAIL_B.sub("", _MOVE_TAIL_A.sub("", s))
    # Strip the opening's OWN family prefix first (family_of == everything before the ':'). The
    # hand-maintained FAMILY_PREFIXES list below only covered ~29 names, so any family missing from it
    # had nothing stripped and every line collapsed to the first word: "Indian Defense: Budapest
    # Gambit" -> "Indian", same as all ~50 other Indian lines. Measured on the 500k scan: 59 families
    # were reduced to ONE variation key holding 700k+ games (King's Pawn 108,947 -> "King's",
    # Indian Defense 57,188 -> "Indian"), which is why the Budapest read 0 games and Grünfeld
    # sublines were invisible. Using family_of makes this work for every family, present or future.
    # Only for COLON-form names: family_of() of a colonless name is the whole string, so stripping it
    # would empty `s` and turn every such line into "Main Line". Must match the TS guard exactly.
    fam = family_of(opening) if ":" in opening else None
    if fam and fam != "Unknown" and s.lower().startswith(fam.lower()):
        s = s[len(fam):].strip()
    else:
        low = s.lower()
        for p in FAMILY_PREFIXES:  # longest-first, matches the TS ordering
            if low.startswith(p.lower()):
                s = s[len(p):].strip()
                break
    # drop leftover ": "/", " from colon-form names, the word "Variation", collapse whitespace
    s = re.sub(r'^[\s:,]+', '', s)
    s = re.sub(r'\bVariation\b', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return "Main Line"
    low = s.lower()
    head = next((t for t in TWO_WORD_HEADS if low.startswith(t)), None)
    return " ".join(s.split(" ")[:2]) if head else s.split(" ")[0]


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
# vstats[color][family][variation][band] = [games, wins, draws, losses] — the level-2 tally, keyed
# by variation_of() (== frontend variationName()). Same banding/skip gate as the family tally.
vstats = {"White": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))),
          "Black": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])))}
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
        if not we or not be or not is_blitz(tc):
            continue
        if abs(we - be) > MAX_ELO_GAP:  # close-rated only — see MAX_ELO_GAP
            continue
        wb, bb = band_of(we), band_of(be)
        # Skip only if BOTH colors' bands are full (or out of range) — else we'd drop a game that
        # still helps the other color's band.
        if (wb is None or band_games[wb] >= GAME_TARGET) and (bb is None or band_games[bb] >= GAME_TARGET):
            continue
        fam = family_of(ope)
        var = variation_of(ope)
        for is_white, band in ((True, wb), (False, bb)):
            if band is None or band_games[band] >= GAME_TARGET:
                continue
            pts = result_pts(res, is_white)
            if pts is None:
                continue
            band_games[band] += 1
            color = "White" if is_white else "Black"
            cell = stats[color][fam][band]
            cell[0] += 1
            cell[1] += pts[0]
            cell[2] += pts[1]
            cell[3] += pts[2]
            vcell = vstats[color][fam][var][band]
            vcell[0] += 1
            vcell[1] += pts[0]
            vcell[2] += pts[1]
            vcell[3] += pts[2]
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

def cell_dict(c):
    return {"games": c[0], "wins": c[1], "draws": c[2], "losses": c[3]}


out = {color: {fam: {b: cell_dict(stats[color][fam][b]) for b in stats[color][fam]}
               for fam in stats[color]}
       for color in ("White", "Black")}
# Level-2: {color: {family: {variation: {band: {...}}}}}. Aggregator nests these under their family.
vout = {color: {fam: {var: {b: cell_dict(vstats[color][fam][var][b]) for b in vstats[color][fam][var]}
                      for var in vstats[color][fam]}
                for fam in vstats[color]}
        for color in ("White", "Black")}
json.dump(out, open("opening_winrates.json", "w"))
json.dump(vout, open("opening_winrates_variations.json", "w"))
print(f"=== DONE === {(time.time() - t0) / 60:.1f}min | bands: " +
      ' '.join(f'{b}:{band_games[b]}' for b, _, _ in BANDS), flush=True)
