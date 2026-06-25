"""Board-level chess primitives for the mistake tagger.

Ported from lichess-puzzler/tagger/util.py (commit c188837) — the genuinely-useful, well-tested
board geometry helpers. We OWN this copy: signatures are board-level (no puzzle/ChildNode coupling),
so our motif detectors call them directly with (board, move). Logic validated against cook's tests.

Two layers here:
  * BOARD primitives  — (board, square) geometry. Single-move detectors use these directly.
  * NODE/LINE helpers — build a chess.pgn line from (board, ucis) and slice it by pov. Sequence
    detectors (sacrifice, deflection, zwischenzug, ...) port cook's ChildNode logic verbatim,
    but with pov EXPLICIT: cook's hardcoded mainline[1::2] (pov/solver moves) becomes
    pov_nodes(nodes, pov), and mainline[::2] (opponent moves) becomes opp_nodes(nodes, pov).
    Empirically (verified): node.turn() is the color to move AFTER node.move, so a pov move's
    node has turn() == not pov. Hence pov_nodes == [n for n in nodes if n.turn() != pov].
"""
import math
import chess
import chess.pgn
from chess import square_rank, square_file, Color, Board, Square, Piece, square_distance
from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN, WHITE, BLACK
from chess.pgn import ChildNode, Game
from typing import List, Tuple, Optional


# ---------------- win% / win-drop (the tagger's mistake-severity currency, issue #29) ----------------
# Lichess logistic (github.com/lichess-org/lila/pull/11148): the SAME currency prod uses to CLASSIFY
# moves (classifyMoves.ts: INACCURACY=10, MISTAKE=20 win-points) and the leak-metrics win%-lost metric.
# Gating predicates on win_drop instead of cp_loss makes one unit across classification, tagging, and
# the drill metric — and is NONLINEAR in eval, so a slip made while already winning costs little win%.

_WINPCT_MULTIPLIER = -0.00368208
_MATE_CLAMP_CP = 1200  # a mate eval is treated as ±1200cp (leak-metrics convention)


def winpct(cp: float) -> float:
    """Side-to-move win probability as a 0-100 percentage for a centipawn eval (mover POV).
    50 at even, monotonic, bounded. Mate should be passed pre-clamped to ±_MATE_CLAMP_CP."""
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(_WINPCT_MULTIPLIER * cp)) - 1.0)


def win_drop(eval_before: Optional[int], eval_after: Optional[int], mover: Color,
             cp_loss: Optional[int] = None) -> float:
    """Win% the mover GAVE UP on this move, mover-POV, clamped >= 0. The taggable-mistake currency.

    Real path (both evals present, e.g. prod + the band corpus that drives tuning): convert the
    white-POV cp evals to mover POV, then winpct(before) - winpct(after). Nonlinear by construction.

    Fallback (either eval is None — mate positions, and the cp-only SAE-feature analysis caches that
    build Mistake with eval_before/after=None): approximate the drop as if from an even position,
    winpct(0) - winpct(-cp_loss). Without evals AND without cp_loss there is nothing to gate on -> 0.
    Returning 0 here (rather than a default-taggable) is deliberate: it never re-introduces the
    played==best noise that #27's gate removed.
    """
    if eval_before is not None and eval_after is not None:
        before = eval_before if mover == WHITE else -eval_before
        after = eval_after if mover == WHITE else -eval_after
        before = max(-_MATE_CLAMP_CP, min(_MATE_CLAMP_CP, before))
        after = max(-_MATE_CLAMP_CP, min(_MATE_CLAMP_CP, after))
        return max(0.0, winpct(before) - winpct(after))
    if cp_loss:
        loss = min(_MATE_CLAMP_CP, abs(int(cp_loss)))
        return max(0.0, winpct(0) - winpct(-loss))
    return 0.0


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


def attacked_opponent_pieces(board: Board, from_square: Square, pov: Color) -> List[Piece]:
    return [p for (p, _) in attacked_opponent_squares(board, from_square, pov)]


def attacker_pieces(board: Board, color: Color, square: Square) -> List[Piece]:
    return [p for p in (board.piece_at(s) for s in board.attackers(color, square)) if p]


def is_outpost(board: Board, square: Square, pov: Color) -> bool:
    """The piece on `square` (a knight or bishop of `pov`) sits on an OUTPOST: in the enemy half,
    defended by a friendly pawn, and NO enemy pawn can ever advance to attack it.

    'No enemy pawn can challenge it' = for each adjacent file, no enemy pawn is positioned BEHIND the
    square (so it could advance toward it) with a clear path of empty squares to the attacking square.
    A blocked pawn (e.g. White c2 stuck behind its own Nc3) cannot challenge -> the square is permanent.
    """
    p = board.piece_at(square)
    if p is None or p.color != pov or p.piece_type not in (KNIGHT, BISHOP):
        return False
    rank, file = square_rank(square), square_file(square)
    # in the enemy half: pov advances toward higher ranks (White) / lower ranks (Black).
    # White outposts live on ranks 4-6 (idx 3-5), Black on ranks 3-5 (idx 2-4). Require crossing center.
    if pov == WHITE and rank < 3:
        return False
    if pov == BLACK and rank > 4:
        return False
    # defended by a friendly pawn (the square is held)
    pawn_def = any(board.piece_type_at(s) == PAWN for s in board.attackers(pov, square))
    if not pawn_def:
        return False
    # An enemy pawn challenges the outpost by reaching an adjacent-file square ONE rank toward pov's
    # side (= rank + adv), from which it attacks `square`. pov's pawns advance by +adv; enemy pawns
    # advance by -adv. So a challenger sits on the far side (rank beyond attack_from, in the -adv
    # direction) and must have a clear path of empty squares to advance up to attack_from_rank.
    enemy = not pov
    adv = 1 if pov == WHITE else -1
    attack_from_rank = rank + adv          # the square an enemy pawn attacks the outpost FROM
    if not (0 <= attack_from_rank <= 7):
        return True                         # outpost on a rank where no pawn could ever attack it
    for df in (-1, 1):
        af = file + df
        if not (0 <= af <= 7):
            continue
        for r in range(8):
            sq = chess.square(af, r)
            pc = board.piece_at(sq)
            if pc is None or pc.piece_type != PAWN or pc.color != enemy:
                continue
            # enemy pawn advances by -adv. To REACH attack_from_rank it must move in the -adv
            # direction, so (attack_from_rank - r) has the same sign as -adv.
            needs_to_advance = (attack_from_rank - r) * (-adv) > 0
            if not needs_to_advance and r != attack_from_rank:
                continue
            # already on the attacking square -> it challenges now
            if r == attack_from_rank:
                return False
            # clear path from r down to attack_from_rank (advancing by -adv), squares exclusive of r,
            # inclusive of the target — any blocker (own or enemy) stops the pawn (e.g. Nc3 blocks c2).
            step = -adv
            rr = r + step
            clear = True
            while True:
                if board.piece_at(chess.square(af, rr)) is not None:
                    clear = False
                    break
                if rr == attack_from_rank:
                    break
                rr += step
            if clear:
                return False
    return True


# ---------------- endgame geometry (king activity / opposition / passed pawns) ----------------

_CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def center_distance(square: Square) -> int:
    """Chebyshev distance from `square` to the nearest of the four central squares (d4/e4/d5/e5).
    Lower = more central. Used to detect a king move that heads toward the center."""
    return min(square_distance(square, c) for c in _CENTER_SQUARES)


def nearest_enemy_pawn_distance(board: Board, square: Square, pov: Color) -> int:
    """Chebyshev distance from `square` to the nearest enemy (not-pov) pawn; 99 if there are none.
    Used to detect a king move that heads toward the opponent's pawns (the other face of king activity)."""
    enemy_pawns = board.pieces(PAWN, not pov)
    return min((square_distance(square, p) for p in enemy_pawns), default=99)


def is_passed_pawn(board: Board, square: Square, color: Color) -> bool:
    """True if a `color` pawn on `square` is passed: no enemy pawn on its file or the two adjacent
    files, on any rank ahead of it (in `color`'s advance direction). Assumes a pawn of `color` is (or
    will be) on `square` — caller pushes the move first when checking the post-move position."""
    f, r = square_file(square), square_rank(square)
    step = 1 if color == WHITE else -1
    for ef in (f - 1, f, f + 1):
        if not (0 <= ef <= 7):
            continue
        er = r + step
        while 0 <= er <= 7:
            pc = board.piece_at(chess.square(ef, er))
            if pc is not None and pc.piece_type == PAWN and pc.color != color:
                return False
            er += step
    return True


def is_pawn_only_endgame(board: Board) -> bool:
    """True if the only non-king pieces on the board are pawns (a king-and-pawn endgame). This is the
    regime where the opposition is the decisive concept."""
    for p in board.piece_map().values():
        if p.piece_type not in (KING, PAWN):
            return False
    return True


# ---------------- ChildNode helpers (for sequence detectors) ----------------
# A "line" is a list[ChildNode] from build_line(); cook's per-node logic ports verbatim onto it.

def moved_piece_type(node: ChildNode) -> int:
    pt = node.board().piece_type_at(node.move.to_square)
    assert pt
    return pt


def is_capture(node: ChildNode) -> bool:
    return node.parent.board().is_capture(node.move)


def is_king_move(node: ChildNode) -> bool:
    return moved_piece_type(node) == KING


def is_castling(node: ChildNode) -> bool:
    return is_king_move(node) and square_distance(node.move.from_square, node.move.to_square) > 1


def is_advanced_pawn_move(node: ChildNode) -> bool:
    if node.move.promotion:
        return True
    if moved_piece_type(node) != PAWN:
        return False
    to_rank = square_rank(node.move.to_square)
    return to_rank < 3 if node.turn() else to_rank > 4


def is_very_advanced_pawn_move(node: ChildNode) -> bool:
    if not is_advanced_pawn_move(node):
        return False
    to_rank = square_rank(node.move.to_square)
    return to_rank < 2 if node.turn() else to_rank > 5


def next_node(node: ChildNode) -> Optional[ChildNode]:
    return node.variations[0] if node.variations else None


def next_next_node(node: ChildNode) -> Optional[ChildNode]:
    nn = next_node(node)
    return next_node(nn) if nn else None


def build_line(start_board: Board, ucis: List[str]) -> List[ChildNode]:
    """Build a pgn line from a starting board + a list of UCI moves. Returns the mainline
    ChildNodes (one per move). Illegal/garbage UCIs truncate the line (caller gets what parsed)."""
    g = Game.from_board(Board(start_board.fen()))
    node = g
    for u in ucis:
        try:
            mv = chess.Move.from_uci(u)
        except Exception:
            break
        if mv not in node.board().legal_moves:
            break
        node = node.add_main_variation(mv)
    return list(g.mainline())


def pov_nodes(nodes: List[ChildNode], pov: Color) -> List[ChildNode]:
    """Nodes whose move was made BY pov (cook's mainline[1::2] when pov is the solver)."""
    return [n for n in nodes if n.turn() != pov]


def opp_nodes(nodes: List[ChildNode], pov: Color) -> List[ChildNode]:
    """Nodes whose move was made by the opponent of pov (cook's mainline[::2])."""
    return [n for n in nodes if n.turn() == pov]
