#!/usr/bin/env python3
"""Depth-16 player-leak inversion: where does cabbage fall on the d16 band curve?

Fixes the recurring denominator bug: the d16 blunders are PER-PLAYER (only the in-band player's
blunders were detected), so the denominator must ALSO be per-player (in-band-player moves only),
NOT both players' moves. We recount in-band-player moves from the cached PGNs, mirroring
redetect_sweep_d16's exact banding (white's moves if WhiteElo in band & present, black's if BlackElo).

Then per band: rate_c = category_blunders_c / in_band_player_moves (per-opportunity rate).
Cabbage (rapid, >=200cp to match the sweep threshold): rate_c = his_blunders_c / his_rapid_moves.
Invert each category's rate onto the band rate->rating curve -> "plays-like" rating per skill,
and the OVERALL blunder rate too (the headline test of whether the bias is fixed).
"""
import json, io
import chess, chess.pgn
from collections import defaultdict, Counter

PGNS   = "/home/ec2-user/SageMaker/sweep_pgns_cache.json"
BLUND  = "/home/ec2-user/SageMaker/sweep_blunders_d16.json"
STATS  = "/home/ec2-user/SageMaker/rating_band_tag_stats_d16.json"
CABG   = "/home/ec2-user/SageMaker/cabbage_tagged.json"
CABGG  = "/home/ec2-user/SageMaker/cabbage_games.json"
OUT    = "/home/ec2-user/SageMaker/cabbage_d16_inversion.json"

BANDS = [("1000-1200",1000,1200),("1200-1400",1200,1400),("1400-1600",1400,1600),
         ("1600-1800",1600,1800),("1800-2000",1800,2000),("2000-2200",2000,2200)]
MID = {b: (lo+hi)//2 for b,lo,hi in BANDS}
DRILL = ["Hung Piece","Missed Capture","Missed Tactic","Missed Mate","Allowed Tactic",
         "Calculation","Trading","Position","King Safety","Endgame"]

def get_band(elo):
    for name, lo, hi in BANDS:
        if lo <= elo < hi: return name
    return None

# ---- 1. per-band IN-BAND-PLAYER move denominators (mirror redetect banding) ----
pgns = json.load(open(PGNS))
blund = json.load(open(BLUND))
present = defaultdict(set)               # game_id -> bands it appears under
for r in blund:
    present[r["game_id"]].add(r["band"])

band_moves = Counter()                   # band -> in-band-player moves
band_endmoves = Counter()                # band -> in-band-player moves in endgame positions
def phase_endgame(board):
    return len(board.piece_map()) <= 14

for gid, bands_here in present.items():
    pgn = pgns.get(gid)
    if not pgn: continue
    g = chess.pgn.read_game(io.StringIO(pgn))
    if g is None: continue
    we = int(g.headers.get("WhiteElo") or 0); be = int(g.headers.get("BlackElo") or 0)
    wb = get_band(we); bb = get_band(be)
    wb = wb if wb in bands_here else None
    bb = bb if bb in bands_here else None
    if wb is None and bb is None: continue
    board = g.board()
    for mv in g.mainline_moves():
        mover_white = board.turn == chess.WHITE
        band = wb if mover_white else bb
        if band is not None:
            band_moves[band] += 1
            if phase_endgame(board): band_endmoves[band] += 1
        board.push(mv)

# ---- 2. per-band per-category blunder counts (from d16 stats) ----
stats = json.load(open(STATS))
band_cat = {}                            # band -> {cat: positions}
band_total_bl = {}                       # band -> total blunders
for b,_,_ in BANDS:
    bd = stats["bands"][b]
    band_cat[b] = {c: bd["categories"][c]["positions"] for c in DRILL}
    band_total_bl[b] = bd["enriched_positions"]

# per-band rates (per in-band-player move) — and endgame rate per endgame move
band_rate = {b: {} for b,_,_ in BANDS}
band_overall = {}
for b,_,_ in BANDS:
    m = band_moves[b]
    band_overall[b] = band_total_bl[b]/m if m else 0
    for c in DRILL:
        denom = band_endmoves[b] if c == "Endgame" else m
        band_rate[b][c] = band_cat[b][c]/denom if denom else 0

# ---- 3. cabbage rates (rapid, >=200cp) ----
cabg = json.load(open(CABG))
cgames = json.load(open(CABGG))
# his rapid moves = moves where it's his turn, rapid games only
his_moves = 0; his_endmoves = 0
for game in cgames:
    if game.get("time_class") != "rapid": continue
    g = chess.pgn.read_game(io.StringIO(game["pgn"]))
    if g is None: continue
    his_white = game.get("color") == "white"
    board = g.board()
    for mv in g.mainline_moves():
        if (board.turn == chess.WHITE) == his_white:
            his_moves += 1
            if phase_endgame(board): his_endmoves += 1
        board.push(mv)

rapid200 = [x for x in cabg if x.get("time_class")=="rapid" and x.get("cp_loss",0)>=200]
cab_cat = Counter()
for x in rapid200:
    cats = set()
    for c in (x.get("categories") or []):
        if c in DRILL: cats.add(c)
    # categories field may already be deduped per blunder; count each category once per blunder
    for c in cats: cab_cat[c]+=1
cab_overall = len(rapid200)/his_moves if his_moves else 0
cab_rate = {}
for c in DRILL:
    denom = his_endmoves if c=="Endgame" else his_moves
    cab_rate[c] = cab_cat[c]/denom if denom else 0

# ---- 4. invert: rate -> rating on the band curve (monotone interp on midpoints) ----
def invert(rate, curve):
    """curve = list of (rating_mid, band_rate) sorted by rating. Higher rating => lower rate
    (for skill categories). Find rating where curve==rate via linear interp; clamp/extrapolate."""
    pts = sorted(curve, key=lambda p: p[0])
    # work in decreasing-rate space; if not monotonic, flag
    rates = [r for _,r in pts]
    mono_dec = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    mono_inc = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    monotonic = mono_dec or mono_inc
    # interpolate rating as function of rate
    # build (rate, rating) and sort by rate
    rr = sorted([(r, rt) for rt,r in pts])
    if rate <= rr[0][0]:
        # lower than lowest rate -> beyond the strong end (if dec) ; extrapolate with end slope
        (r0,t0),(r1,t1) = rr[0], rr[1]
        if r1==r0: return t0, monotonic
        return t0 + (rate-r0)*(t1-t0)/(r1-r0), monotonic
    if rate >= rr[-1][0]:
        (r0,t0),(r1,t1) = rr[-2], rr[-1]
        if r1==r0: return t1, monotonic
        return t1 + (rate-r1)*(t1-t0)/(r1-r0), monotonic
    for i in range(len(rr)-1):
        r0,t0 = rr[i]; r1,t1 = rr[i+1]
        if r0 <= rate <= r1:
            if r1==r0: return t0, monotonic
            return t0 + (rate-r0)*(t1-t0)/(r1-r0), monotonic
    return None, monotonic

# FIFA curve (chess.com rapid anchored, 99=3000) — anchor points rating->score
FIFA = [(800,23),(1200,46),(1600,67),(1950,80),(2000,82),(2200,88),(2400,92),(2600,95),(3000,99)]
def fifa(rating):
    pts = FIFA
    if rating <= pts[0][0]: return pts[0][1]
    if rating >= pts[-1][0]: return pts[-1][1]
    for i in range(len(pts)-1):
        r0,s0=pts[i]; r1,s1=pts[i+1]
        if r0<=rating<=r1: return round(s0+(rating-r0)*(s1-s0)/(r1-r0),1)
    return None

result = {"band_moves": dict(band_moves), "his_moves": his_moves,
          "his_rapid_blunders_ge200": len(rapid200),
          "cab_overall_rate": cab_overall, "band_overall_rate": band_overall,
          "categories": {}}

# overall inversion
ov_curve = [(MID[b], band_overall[b]) for b,_,_ in BANDS]
ov_rating, ov_mono = invert(cab_overall, ov_curve)
result["overall_playslike_rating"] = round(ov_rating) if ov_rating else None
result["overall_playslike_fifa"] = fifa(ov_rating) if ov_rating else None
result["overall_monotonic"] = ov_mono

print(f"in-band moves/band: {dict(band_moves)}")
print(f"cabbage rapid moves: {his_moves} | rapid blunders >=200cp: {len(rapid200)}")
print(f"\nOVERALL: cab rate={cab_overall*100:.2f}%/move  band rates=" +
      " ".join(f"{MID[b]}:{band_overall[b]*100:.2f}%" for b,_,_ in BANDS))
print(f"OVERALL plays-like: {result['overall_playslike_rating']} (FIFA {result['overall_playslike_fifa']}) "
      f"[monotonic={ov_mono}]")

print(f"\n{'category':<16}{'cab%':>7}{'  band rates (1100..2100)':<46}{'plays-like':>11}{'mono':>6}")
for c in DRILL:
    curve = [(MID[b], band_rate[b][c]) for b,_,_ in BANDS]
    rating, mono = invert(cab_rate[c], curve)
    result["categories"][c] = {
        "cab_rate": cab_rate[c], "band_rates": {b: band_rate[b][c] for b,_,_ in BANDS},
        "playslike_rating": round(rating) if rating else None,
        "playslike_fifa": fifa(rating) if rating else None, "monotonic": mono,
    }
    bands_str = " ".join(f"{band_rate[b][c]*100:.1f}" for b,_,_ in BANDS)
    pl = f"{round(rating)}" if rating else "n/a"
    print(f"{c:<16}{cab_rate[c]*100:>6.2f}  {bands_str:<44}{pl:>10}{'Y' if mono else 'N':>6}")

json.dump(result, open(OUT,"w"), indent=2)
print(f"\n-> {OUT}")
