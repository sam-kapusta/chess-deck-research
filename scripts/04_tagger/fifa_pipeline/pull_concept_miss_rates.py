"""Eligible-denominator miss rates for endgame CONCEPT detectors, per band.

The right question for a drill feature: "when the concept is the BEST move (the opportunity exists),
how often does the player MISS it?" — and does that miss-rate fall with rating?

For each rapid game, at every move we run Stockfish? No — we don't have an engine here. Instead we
use the move actually played + the game's %eval to know if THIS move was the best-ish (low loss) or a
blunder, but we need the BEST move to know if the concept was the opportunity. The Lichess movetext
has %eval but not the engine's best move. So we APPROXIMATE the opportunity using the played move when
it was good (cp_loss small => played ≈ best) — same trick as the openings good-moves baseline:

  opportunity (concept available) ≈ a move where the concept PATTERN holds for the played move AND the
    move was good (loss <= 30cp)   → "found it"
  miss ≈ the concept pattern holds for the engine-best alternative but player played something worse.
    We can't see engine-best here, so we instead count, among BLUNDER moves in eligible positions,
    how many were 'concept-shaped best' — but that needs the best move.

Simplest correct approach given our data: count, per band, in MINOR endgames:
  - eligible = moves where the side to move HAS a bishop that COULD gain >=4 mobility by some legal
    move (the opportunity is on the board), regardless of what was played
  - miss    = eligible AND the played move did NOT take it AND the move lost >= MIN_LOSS (a real error)
Then miss_rate = miss / eligible per band. If beginners miss more WHEN THE CHANCE EXISTS, miss_rate
falls with rating.

Generalized to several endgame concepts via CONCEPTS below. Run on chess-poc with HF_HOME=/tmp/hfcache.
Output: concept_miss_rates.json = {concept: {band: {eligible, miss}}}.
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
ELIGIBLE_TARGET = 1500   # per band per concept (rare top bands cap effort)
MIN_LOSS = 200           # cp loss that marks the played move a real error (a "miss")
GOOD_LOSS = 30           # cp loss at/under which the played move counts as "took the chance"
MAX_SHARDS = 8

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

def is_minor_endgame(board):
    has_minor = False
    for p in board.piece_map().values():
        if p.piece_type in (chess.ROOK, chess.QUEEN): return False
        if p.piece_type in (chess.BISHOP, chess.KNIGHT): has_minor = True
    return has_minor

def is_rook_endgame(board):
    has_rook = False
    for p in board.piece_map().values():
        if p.piece_type not in (chess.KING, chess.ROOK, chess.PAWN): return False
        if p.piece_type == chess.ROOK: has_rook = True
    return has_rook

def bishop_activation_available(board, color):
    """Is there a legal bishop move for `color` that gains >=4 squares of mobility? (opportunity)."""
    if board.turn != color:
        return False
    for sq, p in board.piece_map().items():
        if p.piece_type != chess.BISHOP or p.color != color:
            continue
        before = len(board.attacks(sq))
        for mv in board.legal_moves:
            if mv.from_square != sq:
                continue
            t = board.copy(stack=False); t.push(mv)
            if len(t.attacks(mv.to_square)) - before >= 4:
                return True
    return False

def bishop_activation_taken(board, played_move):
    """Did the played move realize a >=4-mobility bishop activation?"""
    if board.piece_type_at(played_move.from_square) != chess.BISHOP:
        return False
    before = len(board.attacks(played_move.from_square))
    t = board.copy(stack=False); t.push(played_move)
    return len(t.attacks(played_move.to_square)) - before >= 4

def rook_activation_available(board, color):
    if board.turn != color:
        return False
    for sq, p in board.piece_map().items():
        if p.piece_type != chess.ROOK or p.color != color:
            continue
        before = len(board.attacks(sq))
        for mv in board.legal_moves:
            if mv.from_square != sq:
                continue
            t = board.copy(stack=False); t.push(mv)
            if len(t.attacks(mv.to_square)) - before >= 4:
                return True
    return False

def rook_activation_taken(board, played_move):
    if board.piece_type_at(played_move.from_square) != chess.ROOK:
        return False
    before = len(board.attacks(played_move.from_square))
    t = board.copy(stack=False); t.push(played_move)
    return len(t.attacks(played_move.to_square)) - before >= 4

def _developed(board, color):
    back = 0 if color == chess.WHITE else 7
    n = 0
    for sq, p in board.piece_map().items():
        if p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(sq) != back:
            n += 1
    return n

def doubled_rooks_available(board, color):
    """A legal rook move exists that puts two same-color rooks on one file (doubling). Opportunity."""
    if board.turn != color:
        return False
    rook_files = [chess.square_file(sq) for sq, p in board.piece_map().items()
                  if p.piece_type == chess.ROOK and p.color == color]
    if len(rook_files) < 2:
        return False
    for mv in board.legal_moves:
        if board.piece_type_at(mv.from_square) != chess.ROOK: continue
        to_f = chess.square_file(mv.to_square)
        # another rook (not the moving one) already on the destination file
        if any(rf == to_f and rsq != mv.from_square for rsq, rf in
               [(sq, chess.square_file(sq)) for sq, p in board.piece_map().items()
                if p.piece_type == chess.ROOK and p.color == color]):
            return True
    return False

def doubled_rooks_taken(board, played_move):
    if board.piece_type_at(played_move.from_square) != chess.ROOK:
        return False
    to_f = chess.square_file(played_move.to_square)
    return any(chess.square_file(sq) == to_f and sq != played_move.from_square
               for sq, p in board.piece_map().items()
               if p.piece_type == chess.ROOK and p.color == board.turn)

def pawngrab_undev_position(board):
    # opening-ish: full-ish board, before move 16
    return board.fullmove_number <= 15

def pawngrab_undev_available(board, color):
    """A pawn capture of an enemy pawn is available while < 4 minors developed. Opportunity to err."""
    if board.turn != color or _developed(board, color) >= 4:
        return False
    for mv in board.legal_moves:
        if board.piece_type_at(mv.from_square) != chess.PAWN: continue
        if not board.is_capture(mv): continue
        victim = board.piece_at(mv.to_square)
        if victim and victim.piece_type == chess.PAWN:
            return True
    return False

def pawngrab_undev_taken(board, played_move):
    if board.piece_type_at(played_move.from_square) != chess.PAWN: return False
    if not board.is_capture(played_move): return False
    victim = board.piece_at(played_move.to_square)
    return bool(victim and victim.piece_type == chess.PAWN and _developed(board, board.turn) < 4)

# concept -> (position_filter(board), available(board,color), taken(board,move))
CONCEPTS = {
    "BishopActivity":  (is_minor_endgame, bishop_activation_available, bishop_activation_taken),
    "RookActivity":    (is_rook_endgame, rook_activation_available, rook_activation_taken),
    "DoubledRooks":    (lambda b: True, doubled_rooks_available, doubled_rooks_taken),
    "PawnGrabUndev":   (pawngrab_undev_position, pawngrab_undev_available, pawngrab_undev_taken),
}

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

# stats[concept][band] = [eligible, miss, took]
stats = {c: defaultdict(lambda: [0, 0, 0, 0]) for c in CONCEPTS}  # [eligible, miss, took_good, took_bad]
elig_count = {c: {b: 0 for b, _, _ in BANDS} for c in CONCEPTS}
t0 = time.time(); ng = 0
def done():
    return all(all(elig_count[c][b] >= ELIGIBLE_TARGET for b, _, _ in BANDS) for c in CONCEPTS)

for fi, f in enumerate(files):
    if fi >= MAX_SHARDS or done(): break
    try:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        t = pq.read_table(p, columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e:
        print('skip', str(e)[:60], flush=True); continue
    for we, be, tc, mt in zip(*[t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext']]):
        ng += 1
        if not we or not be or not is_rapid(tc) or '%eval' not in mt: continue
        if band_of(we) is None and band_of(be) is None: continue
        try:
            g = chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{mt}'))
        except Exception:
            continue
        if not g: continue
        board = g.board(); prev_eval = None
        for node in g.mainline():
            mover = board.turn
            me = we if mover == chess.WHITE else be
            bn = band_of(me)
            cur_eval = pe(node.comment)
            if bn:
                for cname, (posf, availf, takenf) in CONCEPTS.items():
                    if elig_count[cname][bn] >= ELIGIBLE_TARGET:
                        continue
                    try:
                        # posf() handles the phase/material filter per concept (endgame concepts gate
                        # on their material; opening/middlegame concepts like DoubledRooks/PawnGrab don't).
                        if posf(board) and availf(board, mover):
                            elig_count[cname][bn] += 1
                            cell = stats[cname][bn]
                            cell[0] += 1  # eligible
                            took = takenf(board, node.move)
                            loss = None
                            if prev_eval is not None and cur_eval is not None:
                                loss = (prev_eval - cur_eval) if mover == chess.WHITE else (cur_eval - prev_eval)
                            blundered = loss is not None and loss >= MIN_LOSS
                            if took and (loss is None or loss <= GOOD_LOSS):
                                cell[2] += 1  # took the chance well
                            if (not took) and blundered:
                                cell[1] += 1  # missed it AND blundered (missed-X concepts)
                            if took and blundered:
                                cell[3] += 1  # took it AND blundered (played-X-is-bad concepts)
                    except Exception:
                        pass
            prev_eval = cur_eval; board.push(node.move)
    try: os.remove(p)
    except: pass
    if fi % 5 == 0:
        print(f'shard {fi+1} | {ng}g {time.time()-t0:.0f}s | ' +
              ' '.join(f'{c}:{min(elig_count[c].values())}-{max(elig_count[c].values())}' for c in CONCEPTS), flush=True)

out = {c: {b: {"eligible": stats[c][b][0], "miss": stats[c][b][1],
               "took_good": stats[c][b][2], "took_bad": stats[c][b][3]}
           for b in stats[c]} for c in CONCEPTS}
json.dump(out, open("concept_miss_rates.json", "w"))
print(f"=== DONE === {(time.time()-t0)/60:.1f}min", flush=True)
BAND_ORDER = [b for b, _, _ in BANDS]
# missed-X concepts → miss/eligible should FALL with rating. played-X-is-bad concepts (PawnGrabUndev)
# → took_bad/eligible should FALL with rating (beginners err by doing it more when it's available).
PLAYED_BAD = {"PawnGrabUndev"}
for c in CONCEPTS:
    metric = "took_bad" if c in PLAYED_BAD else "miss"
    print(f"\n{c}  — {metric} rate when the chance exists ({metric}/eligible), per band:")
    rates = []
    for b in BAND_ORDER:
        cell = out[c].get(b)
        rates.append(round(100*cell[metric]/cell['eligible'],1) if cell and cell['eligible'] else 0)
    print(f"  eligible: {[out[c].get(b,{}).get('eligible',0) for b in BAND_ORDER]}")
    print(f"  {metric}%: {rates}")
    nz = [r for r in rates if r]
    mono = all(nz[i] >= nz[i+1] for i in range(len(nz)-1))
    print(f"  falls with rating (beginners err more): {mono}")
