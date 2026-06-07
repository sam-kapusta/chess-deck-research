#!/usr/bin/env python3
"""Layer 1 — our deterministic position/material predicates (the tier Lichess's tagger lacks).

Each predicate is a pure function of the Mistake object returning a list of (tag, direction, evidence)
or []. No engine, no torch. direction in {missed, allowed, hung, played, info}.

These are the crisp, high-volume tags: phase, game-state, capture-vs-exchange (by piece), hung
material (from the refutation line, end-of-line delta — the validated metric, NOT one-ply), king
safety, pawn-structure deltas, wrong move-order, only-move, captured-with-wrong-piece.
"""
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE_NAME = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
              chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}


# ---------- helpers ----------
def _material(board, color):
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color)


def _best_move(m):
    b = m.board_before
    for u in (m.best_uci,):
        if u:
            try:
                mv = chess.Move.from_uci(u)
                if mv in b.legal_moves:
                    return mv
            except Exception:
                pass
    # fall back to first move of best_line
    if m.best_line_san:
        try:
            return b.parse_san(m.best_line_san[0])
        except Exception:
            return None
    return None


def _played_move(m):
    try:
        return chess.Move.from_uci(m.played_uci)
    except Exception:
        return None


def _is_defended(board_after_move, sq, by_color):
    return board_after_move.is_attacked_by(by_color, sq)


# ---------- phase / state ----------
def phase(m):
    b = m.board_before
    npieces = len(b.piece_map())
    nonpawn = sum(1 for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    if b.fullmove_number <= 12 and npieces >= 24:
        ph = "Opening"
    elif npieces <= 12 or nonpawn <= 4:
        ph = "Endgame"
    else:
        ph = "Middlegame"
    return [(ph, "info", f"{npieces} pieces, move {b.fullmove_number}")]


def game_state(m):
    if m.eval_before is None:
        return []
    # white-POV cp -> mover-POV
    cp = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    if cp >= 150:
        s = "Blunder While Winning"
    elif cp <= -150:
        s = "Blunder While Losing"
    else:
        s = "Blunder While Equal"
    return [(s, "info", f"{cp:+d}cp before (mover POV)")]


# ---------- material: capture vs exchange, by piece ----------
def capture_or_exchange(m):
    """Best move is a capture: free (undefended) -> Missed Free Capture (Piece);
    defended/even -> Missed Exchange (Piece)."""
    b = m.board_before
    bm = _best_move(m)
    pm = _played_move(m)
    if bm is None or not b.is_capture(bm):
        return []
    # the played move itself being a capture of the same square is handled by captured_wrong_piece
    victim = b.piece_at(bm.to_square)
    if victim is None:  # en passant
        return [("Missed Capture (Pawn)", "missed", "best move = en passant")]
    after = b.copy(); after.push(bm)
    defended = after.is_attacked_by(not m.mover, bm.to_square)
    pname = PIECE_NAME[victim.piece_type]
    if not defended:
        return [(f"Missed Free Capture ({pname})", "missed", f"best {m.best_san} takes undefended {pname.lower()}")]
    # defended: even trade vs winning capture
    attacker = b.piece_at(bm.from_square)
    if attacker and VAL[victim.piece_type] > VAL[attacker.piece_type] + 0.5:
        return [(f"Missed Winning Capture ({pname})", "missed", f"best wins {pname.lower()} for less")]
    return [(f"Missed Exchange ({pname})", "missed", f"best {m.best_san} = even trade of {pname.lower()}")]


def bad_capture(m):
    """The PLAYED move is a capture that backfires (eval crashed)."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or not b.is_capture(pm) or m.cp_loss < 120:
        return []
    victim = b.piece_at(pm.to_square)
    pname = PIECE_NAME[victim.piece_type] if victim else "Pawn"
    return [(f"Bad Capture", "played", f"played {m.played_san} (took {pname.lower()}) lost {m.cp_loss}cp")]


def hung_material(m):
    """Played move loses own material in the refutation line (end-of-line delta — validated metric)."""
    if not m.refutation_san:
        return []
    b = m.board_after  # opponent to move; refutation is from here
    before = _material(b, m.mover)
    bb = chess.Board(b.fen())
    for san in m.refutation_san:
        try:
            bb.push(bb.parse_san(san))
        except Exception:
            break
    lost = before - _material(bb, m.mover)
    if lost >= 2:
        # name by the largest own piece captured in the line
        return [("Hung Material", "hung", f"refutation nets {lost} pts of mover's material")]
    return []


# ---------- king safety ----------
def king_in_center(m):
    b = m.board_before
    ph = phase(m)[0][0]
    if ph == "Endgame":
        return []
    ks = b.king(m.mover)
    if ks is None:
        return []
    f = chess.square_file(ks)
    # king still in the center files (d/e) past the opening, hasn't castled
    if f in (3, 4) and not (b.has_castling_rights(m.mover)):
        return [("King in Center", "info", "uncastled king on d/e file in middlegame")]
    return []


def lost_castling(m):
    b = m.board_before
    pm = _played_move(m)
    if pm is None:
        return []
    had = b.has_castling_rights(m.mover)
    if not had:
        return []
    after = b.copy(); after.push(pm)
    if not after.has_castling_rights(m.mover) and not b.is_castling(pm):
        return [("Lost Castling Rights", "played", "played move forfeited castling")]
    return []


def exposed_king_pawn(m):
    """Played move is a pawn move in front of own king's shelter."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or b.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    ks = b.king(m.mover)
    if ks is None:
        return []
    # pawn move within 1 file of the king and on the king's side of the board
    if abs(chess.square_file(pm.from_square) - chess.square_file(ks)) <= 1 and m.cp_loss >= 80:
        # only if it's a structural advance near the king
        return [("Pawn Move Exposed King", "played", "pawn push near own king")]
    return []


# ---------- pawn structure deltas ----------
def _pawn_files(board, color):
    files = {}
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.PAWN and p.color == color:
            files.setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))
    return files


def pawn_structure(m):
    pm = _played_move(m)
    if pm is None or m.board_before.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    before = m.board_before
    after = before.copy(); after.push(pm)
    out = []
    bf = _pawn_files(before, m.mover); af = _pawn_files(after, m.mover)
    # doubled: a file gained a 2nd+ pawn
    for f, ranks in af.items():
        if len(ranks) >= 2 and len(bf.get(f, [])) < len(ranks):
            out.append(("Created Doubled Pawn", "played", f"file {chr(97+f)} doubled"))
            break
    # isolated: the moved pawn's file has no friendly pawn on adjacent files
    tf = chess.square_file(pm.to_square)
    if tf in af:
        if (tf - 1) not in af and (tf + 1) not in af:
            out.append(("Created Isolated Pawn", "played", f"pawn on {chr(97+tf)} isolated"))
    return out


# ---------- move quality meta ----------
def wrong_move_order(m):
    """Played move IS in Stockfish's best line, just not first — a transposition/timing error."""
    if not m.best_line_san:
        return []
    b = m.board_before
    try:
        played_san = b.san(_played_move(m)) if _played_move(m) else m.played_san
    except Exception:
        played_san = m.played_san
    # is the played move the same as a LATER move in the best line (not move 0)?
    if played_san and played_san in m.best_line_san[1:]:
        return [("Wrong Move Order", "played", f"{played_san} is in the best line, played too early")]
    return []


def captured_wrong_piece(m):
    """Played and best move both capture the SAME square, with different pieces."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None:
        return []
    if b.is_capture(pm) and b.is_capture(bm) and pm.to_square == bm.to_square and pm.from_square != bm.from_square:
        return [("Captured With Wrong Piece", "played", f"played {m.played_san}, best {m.best_san} (same square)")]
    return []


# ---------- registry ----------
ALL_PREDICATES = [
    phase, game_state, capture_or_exchange, bad_capture, hung_material,
    king_in_center, lost_castling, exposed_king_pawn, pawn_structure,
    wrong_move_order, captured_wrong_piece,
]


def tag_predicates(m):
    out = []
    for fn in ALL_PREDICATES:
        try:
            out.extend(fn(m))
        except Exception:
            pass
    return out
