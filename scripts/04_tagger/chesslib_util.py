"""Board-level chess primitives for the mistake tagger.

Ported from lichess-puzzler/tagger/util.py (commit c188837) — the genuinely-useful, well-tested
board geometry helpers. We OWN this copy: signatures are board-level (no puzzle/ChildNode coupling),
so our motif detectors call them directly with (board, move). Logic validated against cook's tests.
"""
import chess
from chess import square_rank, square_file, Color, Board, Square, Piece
from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
from typing import List, Tuple

values = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9}
king_values = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 99}
ray_piece_types = [QUEEN, ROOK, BISHOP]
PIECE_NAME = {PAWN: "Pawn", KNIGHT: "Knight", BISHOP: "Bishop", ROOK: "Rook", QUEEN: "Queen", KING: "King"}


def material_count(board: Board, side: Color) -> int:
    return sum(len(board.pieces(pt, side)) * v for pt, v in values.items())


def material_diff(board: Board, side: Color) -> int:
    return material_count(board, side) - material_count(board, not side)


def attacked_opponent_squares(board: Board, from_square: Square, pov: Color) -> List[Tuple[Piece, Square]]:
    """Enemy pieces attacked by the piece on from_square."""
    out = []
    for sq in board.attacks(from_square):
        p = board.piece_at(sq)
        if p and p.color != pov:
            out.append((p, sq))
    return out


def is_defended(board: Board, piece: Piece, square: Square) -> bool:
    if board.attackers(piece.color, square):
        return True
    # ray defense: a friendly ray piece defends through the current attacker
    for attacker in board.attackers(not piece.color, square):
        ap = board.piece_at(attacker)
        if ap and ap.piece_type in ray_piece_types:
            bc = board.copy(stack=False)
            bc.remove_piece_at(attacker)
            if bc.attackers(piece.color, square):
                return True
    return False


def is_hanging(board: Board, piece: Piece, square: Square) -> bool:
    return not is_defended(board, piece, square)


def can_be_taken_by_lower_piece(board: Board, piece: Piece, square: Square) -> bool:
    for asq in board.attackers(not piece.color, square):
        a = board.piece_at(asq)
        if a and a.piece_type != KING and values[a.piece_type] < values[piece.piece_type]:
            return True
    return False


def is_in_bad_spot(board: Board, square: Square) -> bool:
    """Piece on `square` is hanging or takeable by a lower-value piece."""
    p = board.piece_at(square)
    if p is None:
        return False
    return bool(board.attackers(not p.color, square)) and \
        (is_hanging(board, p, square) or can_be_taken_by_lower_piece(board, p, square))


def is_trapped(board: Board, square: Square) -> bool:
    """Piece on `square` (not pawn/king) is in a bad spot and can't escape to a safe square."""
    p = board.piece_at(square)
    if p is None or p.piece_type in (PAWN, KING):
        return False
    if board.is_check() or board.is_pinned(p.color, square):
        return False
    if not is_in_bad_spot(board, square):
        return False
    for esc in board.legal_moves:
        if esc.from_square == square:
            cap = board.piece_at(esc.to_square)
            if cap and values[cap.piece_type] >= values[p.piece_type]:
                return False
            board.push(esc)
            bad = is_in_bad_spot(board, esc.to_square)
            board.pop()
            if not bad:
                return False
    return True


def squares_are_collinear(s1: Square, s2: Square, s3: Square) -> bool:
    r1, f1 = square_rank(s1), square_file(s1)
    r2, f2 = square_rank(s2), square_file(s2)
    r3, f3 = square_rank(s3), square_file(s3)
    return (r1 == r2 == r3) or (f1 == f2 == f3) or \
           ((r1 - f1) == (r2 - f2) == (r3 - f3)) or ((r1 + f1) == (r2 + f2) == (r3 + f3))
