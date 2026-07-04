"""Per (material-type, band) endgame blunder rates from the Lichess rapid corpus.

Endgame phase (piece-count rule, matches tagger). Classify each endgame position by MATERIAL TYPE
(the axis every endgame book uses — Dvoretsky, de la Villa):
  Pawn        = K+P only
  Rook        = rooks + pawns only (R / RR each side), no other pieces
  Queen       = queens (+ maybe pawns) only, no R/B/N
  Minor       = only bishops/knights (+pawns): B, N, B+N, BB — the minor-piece endings
  RookMinor   = rooks + one minor + pawns (R vs B, R vs N, R+B, ...)
  Heavy       = queen(s) with rook(s) and/or pieces still on (Q+R, Q+pieces)
  Other       = anything else (e.g. queen vs rook)

Banded by the MOVER's Elo. One scan; numerator (>=MIN_LOSS blunder) + denominator (moves) share
the population per type. Output: endgame_material_rates.json = {type: {band: {moves, blunders}}}.
Run on chess-poc:  python -u pull_endgame_material.py
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
MOVE_TARGET = 30000   # endgame moves per band (top band scarce; 30k ample for rate stability)
MIN_LOSS = 200
MAX_SHARDS = 130

def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m: return False
    b, i = int(m.group(1)), int(m.group(2)); return 480 <= b + 40*i < 1500

def band_of(e):
    for n, lo, hi in BANDS:
        if lo <= e < hi: return n
    return None

def is_endgame(board):
    pm = board.piece_map()
    if len(pm) <= 12: return True
    return sum(1 for p in pm.values() if p.piece_type not in (chess.PAWN, chess.KING)) <= 4

def material_type(board):
    """Classify an endgame position by surviving non-pawn material (both colors pooled)."""
    nq = nr = nb = nn = 0
    for p in board.piece_map().values():
        t = p.piece_type
        if t == chess.QUEEN: nq += 1
        elif t == chess.ROOK: nr += 1
        elif t == chess.BISHOP: nb += 1
        elif t == chess.KNIGHT: nn += 1
    minors = nb + nn
    if nq == 0 and nr == 0 and minors == 0:
        return "Pawn"
    if nq == 0 and nr > 0 and minors == 0:
        return "Rook"
    if nq > 0 and nr == 0 and minors == 0:
        return "Queen"
    if nq == 0 and nr == 0 and minors > 0:
        return "Minor"
    if nq == 0 and nr > 0 and minors > 0:
        return "RookMinor"
    if nq > 0 and (nr > 0 or minors > 0):
        return "Heavy"
    return "Other"

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

# stats[type][band] = [moves, blunders]
stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
band_moves = {b: 0 for b, _, _ in BANDS}
t0 = time.time(); ng = 0
def done(): return all(band_moves[b] >= MOVE_TARGET for b, _, _ in BANDS)

for fi, f in enumerate(files):
    if fi >= MAX_SHARDS or done(): break
    try:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        t = pq.read_table(p, columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e:
        print('skip', str(e)[:60], flush=True); continue
    cols = [t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext']]
    for we, be, tc, mt in zip(*cols):
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
        board = g.board(); prev_eval = None
        for node in g.mainline():
            mover_white = (board.turn == chess.WHITE)
            me = we if mover_white else be
            bn = band_of(me)
            cur_eval = pe(node.comment)
            if bn and band_moves[bn] < MOVE_TARGET and is_endgame(board):
                mtype = material_type(board)
                band_moves[bn] += 1
                cell = stats[mtype][bn]
                cell[0] += 1
                if prev_eval is not None and cur_eval is not None:
                    loss = (prev_eval - cur_eval) if mover_white else (cur_eval - prev_eval)
                    if loss >= MIN_LOSS:
                        cell[1] += 1
            prev_eval = cur_eval; board.push(node.move)
    try: os.remove(p)
    except: pass
    if fi % 5 == 0:
        print(f'shard {fi+1} | {ng}g {time.time()-t0:.0f}s | ' +
              ' '.join(f'{b.split("-")[0]}:{band_moves[b]}' for b, _, _ in BANDS), flush=True)

out = {mt: {b: {"moves": stats[mt][b][0], "blunders": stats[mt][b][1]} for b in stats[mt]}
       for mt in stats}
json.dump(out, open("endgame_material_rates.json", "w"))
print(f"=== DONE === {(time.time()-t0)/60:.1f}min", flush=True)
BAND_ORDER = [b for b,_,_ in BANDS]
print(f"{'type':<12} {'total':>8} {'overall%':>8}  per-band blunder% (600..2800)")
for mt in sorted(out, key=lambda m: -sum(d['moves'] for d in out[m].values())):
    tm = sum(d['moves'] for d in out[mt].values())
    tb = sum(d['blunders'] for d in out[mt].values())
    rates = []
    for b in BAND_ORDER:
        c = out[mt].get(b)
        rates.append(round(100*c['blunders']/c['moves'],1) if c and c['moves'] else 0)
    mono = all(rates[i] >= rates[i+1] for i in range(len(rates)-1) if rates[i] and rates[i+1])
    print(f"{mt:<12} {tm:>8} {round(100*tb/max(tm,1),1):>7}%  {rates}  {'MONO' if mono else 'non-mono'}")
