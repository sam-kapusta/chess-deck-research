"""Per (family, color, band) opening BLUNDER rates + WIN rates from the Lichess rapid corpus.

Family = Opening column before ':'. Two independent tallies, one scan:
  - BLUNDER (per move): for every Opening+Middlegame move (piece-count phase rule matching the tagger),
    count moves; if the move lost >= MIN_LOSS cp (from %eval), count a blunder. Banded by MOVER Elo.
  - WIN (per game): each game contributes ONE outcome to its family for EACH color, banded by that
    color's Elo. wins/draws/losses are from that color's perspective (Result 1-0 = White win).
    win% = (wins + 0.5*draws) / games. This is the "how do players at band B score in opening X" metric.

Output: opening_rates.json = {"White": {family: {band: {moves, blunders, games, wins, draws, losses}}},
                              "Black": {...}}
Run on chess-poc:  python -u pull_opening_rates.py
"""
import os, re, io, json, time
import chess, chess.pgn
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from collections import defaultdict

REPO = "Lichess/standard-chess-games"
BANDS = [('600-800',600,800),('800-1000',800,1000),('1000-1200',1000,1200),('1200-1400',1200,1400),
         ('1400-1600',1400,1600),('1600-1800',1600,1800),('1800-2000',1800,2000),('2000-2200',2000,2200),
         ('2200-2400',2200,2400),('2400-2600',2400,2600),('2600-2800',2600,2800)]
MOVE_TARGET = 60000   # Opening+Middlegame moves per band
MIN_LOSS = 200
MAX_SHARDS = 120

def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m: return False
    b, i = int(m.group(1)), int(m.group(2)); return 480 <= b + 40*i < 1500

def band_of(e):
    for n, lo, hi in BANDS:
        if lo <= e < hi: return n
    return None

def family_of(opening):
    if not opening: return "Unknown"
    name = opening.split(":")[0].strip()
    return name if name and name != "Unknown Opening" else "Unknown"

def phase_of(board):
    """Opening / Middlegame / Endgame — matches predicates.phase + worker _move_phase."""
    pm = board.piece_map(); npieces = len(pm)
    nonpawn = sum(1 for p in pm.values() if p.piece_type not in (chess.PAWN, chess.KING))
    if board.fullmove_number <= 12 and npieces >= 24: return "Opening"
    if npieces <= 12 or nonpawn <= 4: return "Endgame"
    return "Middlegame"

def pe(c):
    m = re.search(r'\[%eval\s+([#\-\d.]+)\]', c or '')
    if not m: return None
    s = m.group(1)
    if '#' in s: return -10000 if '-' in s else 10000
    try: return int(round(float(s) * 100))
    except: return None

files = [f for f in list_repo_files(REPO, repo_type="dataset")
         if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$', f)]
files.sort(reverse=True)

# stats[color][family][band] = [moves, blunders, games, wins, draws, losses]
stats = {"White": defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0, 0])),
         "Black": defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0, 0]))}
band_moves = {b: 0 for b, _, _ in BANDS}
t0 = time.time(); ng = 0
def done(): return all(band_moves[b] >= MOVE_TARGET for b, _, _ in BANDS)

def result_pts(res, is_white):
    """Mover-color result -> (win, draw, loss) one-hot from THAT color's perspective."""
    if res == "1-0":   return (1, 0, 0) if is_white else (0, 0, 1)
    if res == "0-1":   return (0, 0, 1) if is_white else (1, 0, 0)
    if res == "1/2-1/2": return (0, 1, 0)
    return None

for fi, f in enumerate(files):
    if fi >= MAX_SHARDS or done(): break
    try:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        t = pq.read_table(p, columns=['WhiteElo','BlackElo','TimeControl','movetext','Opening','Result'])
    except Exception as e:
        print('skip', str(e)[:60], flush=True); continue
    cols = [t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext','Opening','Result']]
    for we, be, tc, mt, ope, res in zip(*cols):
        ng += 1
        if not we or not be or not is_rapid(tc) or '%eval' not in mt: continue
        wb, bb = band_of(we), band_of(be)
        if (wb is None or band_moves[wb] >= MOVE_TARGET) and (bb is None or band_moves[bb] >= MOVE_TARGET):
            continue
        try:
            g = chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{mt}'))
        except Exception:
            continue
        if not g: continue
        fam = family_of(ope)
        # WIN tally (per game): one outcome per color, banded by that color's Elo. Independent of the
        # move-target gate's remaining budget so win% stays well-sampled even after moves fill up.
        for is_white, band in ((True, wb), (False, bb)):
            if band is None: continue
            pts = result_pts(res, is_white)
            if pts is None: continue
            cell = stats["White" if is_white else "Black"][fam][band]
            cell[2] += 1; cell[3] += pts[0]; cell[4] += pts[1]; cell[5] += pts[2]
        board = g.board(); prev_eval = None
        for node in g.mainline():
            mover_white = (board.turn == chess.WHITE)
            me = we if mover_white else be
            bn = band_of(me)
            cur_eval = pe(node.comment)
            if bn and band_moves[bn] < MOVE_TARGET:
                ph = phase_of(board)
                if ph in ("Opening", "Middlegame"):
                    band_moves[bn] += 1
                    color = "White" if mover_white else "Black"
                    cell = stats[color][fam][bn]
                    cell[0] += 1
                    if prev_eval is not None and cur_eval is not None:
                        loss = (prev_eval - cur_eval) if mover_white else (cur_eval - prev_eval)
                        if loss >= MIN_LOSS:
                            cell[1] += 1
            prev_eval = cur_eval; board.push(node.move)
    try: os.remove(p)
    except: pass
    import shutil
    cache = os.path.expanduser("~/.cache/huggingface/hub/datasets--Lichess--standard-chess-games/blobs")
    if os.path.isdir(cache): shutil.rmtree(cache, ignore_errors=True)
    if fi % 5 == 0:
        print(f'shard {fi+1} | {ng}g {time.time()-t0:.0f}s | ' +
              ' '.join(f'{b.split("-")[0]}:{band_moves[b]}' for b, _, _ in BANDS), flush=True)

out = {color: {fam: {b: {"moves": stats[color][fam][b][0], "blunders": stats[color][fam][b][1],
                         "games": stats[color][fam][b][2], "wins": stats[color][fam][b][3],
                         "draws": stats[color][fam][b][4], "losses": stats[color][fam][b][5]}
                     for b in stats[color][fam]}
               for fam in stats[color]}
       for color in ("White", "Black")}
json.dump(out, open("opening_rates.json", "w"))
print(f"=== DONE === {(time.time()-t0)/60:.1f}min", flush=True)
for color in ("White", "Black"):
    fams = sorted(out[color], key=lambda fm: -sum(d["moves"] for d in out[color][fm].values()))
    print(f"{color}: {len(out[color])} families. Top 10 by moves:")
    for fm in fams[:10]:
        tm = sum(d["moves"] for d in out[color][fm].values())
        tb = sum(d["blunders"] for d in out[color][fm].values())
        print(f"  {fm[:36]:<38} {tm:>7} moves  {round(100*tb/max(tm,1),1)}% blunder")
