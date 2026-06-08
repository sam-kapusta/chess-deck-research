"""Motif detectors — OUR structure. Two shapes:

  SINGLE-MOVE  is_X(board, move) -> bool   : does playing `move` (by board.turn) create motif X?
  SEQUENCE     is_X(board, line) -> bool   : motif spanning a short line of moves (sac, deflection...)

Logic ported from lichess-puzzler cook.py (validated correct against its own tests) but rewired to be
pov-explicit and line-shape-agnostic — the mover is always `board.turn`, the tactic is always the
move/line we pass. No puzzle mainline[1::2] parity, no auto-pov. This is what lets MISSED (best move),
ALLOWED (refutation), FAILED (played move) all use the SAME detectors with different inputs.
"""
import chess
from chess import KING, PAWN
import chesslib_util as U


# ---------------- single-move motifs ----------------

def is_fork(board: chess.Board, move: chess.Move) -> bool:
    """After `move`, the moved (non-king) piece attacks >=2 enemy pieces that are each either
    higher-value than the mover, or hanging and not defended by the mover's new square."""
    pov = board.turn
    if board.piece_type_at(move.from_square) == KING:
        return False
    b = board.copy(stack=False); b.push(move)
    to = move.to_square
    if U.is_in_bad_spot(b, to):   # the forking piece itself just hangs -> not a real fork
        return False
    mover_pt = b.piece_type_at(to)
    nb = 0
    for piece, sq in U.attacked_opponent_squares(b, to, pov):
        if piece.piece_type == PAWN:
            continue
        if U.king_values[piece.piece_type] > U.king_values[mover_pt] or \
           (U.is_hanging(b, piece, sq) and sq not in b.attackers(not pov, to)):
            nb += 1
    return nb > 1


def is_hanging_piece(board: chess.Board, move: chess.Move) -> bool:
    """`move` captures a hanging (undefended) enemy piece for free — nothing recaptures, so the whole
    piece is won regardless of the capturer's value. (A defended victim is a trade, not a hang.)"""
    if not board.is_capture(move):
        return False
    victim = board.piece_at(move.to_square)
    if victim is None:  # en passant
        return False
    return U.is_hanging(board, victim, move.to_square)


def is_pin(board: chess.Board, move: chess.Move) -> bool:
    """After `move` by pov, an enemy piece is pinned (absolutely or to a more valuable piece) by the
    moved piece's line. Uses python-chess pin detection on the resulting position."""
    pov = board.turn
    b = board.copy(stack=False); b.push(move)
    # any enemy piece now pinned along a ray that the moved piece participates in
    for sq, p in b.piece_map().items():
        if p.color == pov:
            continue
        if b.is_pinned(not pov, sq):
            # only count if the moved piece is an attacker on that square's pin ray
            if move.to_square in b.attackers(pov, sq) or _on_pin_ray(b, move.to_square, sq, not pov):
                return True
    return False


def _on_pin_ray(board, mover_sq, pinned_sq, pinned_color):
    pin_mask = board.pin(pinned_color, pinned_sq)
    return mover_sq in pin_mask


def is_skewer(board: chess.Board, move: chess.Move) -> bool:
    """After `move`, the moved ray-piece attacks a valuable enemy piece with a less-valuable enemy
    piece (or the king) on the same line behind it (skewer)."""
    pov = board.turn
    pt = board.piece_type_at(move.from_square)
    if pt not in U.ray_piece_types:
        return False
    b = board.copy(stack=False); b.push(move)
    to = move.to_square
    for piece, sq in U.attacked_opponent_squares(b, to, pov):
        # look beyond `sq` along the ray from `to` for a lower-value enemy piece
        behind = _square_beyond(to, sq)
        if behind is not None:
            bp = b.piece_at(behind)
            if bp and bp.color != pov and U.values.get(piece.piece_type, 0) > U.values.get(bp.piece_type, 0):
                # `sq` must shield `behind` along the ray (collinear)
                if U.squares_are_collinear(to, sq, behind):
                    return True
    return False


def _square_beyond(frm, mid):
    df = chess.square_file(mid) - chess.square_file(frm)
    dr = chess.square_rank(mid) - chess.square_rank(frm)
    step_f = (df > 0) - (df < 0)
    step_r = (dr > 0) - (dr < 0)
    nf = chess.square_file(mid) + step_f
    nr = chess.square_rank(mid) + step_r
    if 0 <= nf <= 7 and 0 <= nr <= 7 and (step_f or step_r):
        return chess.square(nf, nr)
    return None


def is_discovered_attack(board: chess.Board, move: chess.Move) -> bool:
    """`move` unveils an attack: a friendly ray piece (not the one that moved) now attacks a valuable
    enemy piece that it did NOT attack before the move (because the mover was blocking the line)."""
    pov = board.turn
    before = board
    after = board.copy(stack=False); after.push(move)
    gained = False
    for sq, p in after.piece_map().items():
        if p.color != pov or p.piece_type not in U.ray_piece_types or sq == move.to_square:
            continue
        for tp, tsq in U.attacked_opponent_squares(after, sq, pov):
            if U.values.get(tp.piece_type, 0) >= 3:  # attacks a minor or better
                # was this attack present before the move?
                if tsq not in before.attacks(sq) or before.piece_at(sq) is None:
                    # confirm the moved piece previously sat on the line between sq and tsq
                    if U.squares_are_collinear(sq, move.from_square, tsq):
                        gained = True
    return gained


def exposes_own_king(board: chess.Board, move: chess.Move, threshold: int = 2) -> bool:
    """After `move`, the mover's own king has more enemy attackers on/around it than before."""
    pov = board.turn
    def pressure(b):
        ks = b.king(pov)
        if ks is None:
            return 0
        sqs = [ks] + [s for s in chess.SQUARES if chess.square_distance(s, ks) == 1]
        return sum(1 for s in sqs if b.is_attacked_by(not pov, s))
    after = board.copy(stack=False); after.push(move)
    return pressure(after) - pressure(board) >= threshold


SINGLE_MOVE = {
    "Fork": is_fork, "Hanging Piece": is_hanging_piece, "Pin": is_pin, "Skewer": is_skewer,
    "Discovered Attack": is_discovered_attack,
}
