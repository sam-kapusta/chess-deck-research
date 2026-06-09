"""Motif detectors — OUR structure. Logic ported from lichess-puzzler cook.py (commit c188837,
validated correct against its own tests), rewired to be pov-EXPLICIT and line-shape-agnostic.

Two detector shapes:

  SINGLE-MOVE   is_X(board, move) -> bool
      Does playing `move` (by board.turn) create motif X? Used for the FAILED direction
      (the played move itself is a tactic that backfired) and as the per-node core of line forks.

  LINE          x_line(nodes, pov) -> bool          [nodes = list[ChildNode] from U.build_line]
      Motif spanning a short line. Ported VERBATIM from cook, with the only changes being:
        puzzle.mainline        -> nodes
        puzzle.pov  (hardcoded)-> pov  (parameter)
        mainline[1::2]         -> U.pov_nodes(nodes, pov)   (moves BY pov / the solver)
        mainline[::2]          -> U.opp_nodes(nodes, pov)   (moves by pov's opponent)
        puzzle.game.board()    -> nodes[0].parent.board()  (position before move 0)
        puzzle.game.end()      -> nodes[-1]                 (last node in the line)
        assert ...             -> guard `return False`      (garbage lines don't crash)
      Every slice cook uses ([1:], [:-1]) is kept as-is — cook is proven; deviating reintroduces
      the pov/parity bug we just eliminated.

The tagger drives these in three directions by choosing (board, line, pov):
  MISSED X  : board=fen_before, line=best_line,            pov=mover
  ALLOWED X : board=fen_before, line=[played]+refutation,  pov=opponent  (== cook's puzzle shape)
  FAILED X  : board=fen_before, move=played,               pov=mover     (single-move only)
"""
import chess
from chess import (KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN, WHITE, BLACK,
                   SquareSet, Piece, square_rank, square_file, square_distance)
from chess.pgn import ChildNode
from typing import List, Optional
import chesslib_util as U


# ============================================================================
#  SINGLE-MOVE cores  (board, move) -> bool      [for FAILED direction + fork core]
# ============================================================================

def is_fork(board: chess.Board, move: chess.Move) -> bool:
    """After `move`, the moved (non-king) piece attacks >=2 enemy pieces each either higher-value
    than the mover, or hanging and not defended by the mover's new square. (cook fork() per-node body.)"""
    pov = board.turn
    if board.piece_type_at(move.from_square) == KING:
        return False
    b = board.copy(stack=False); b.push(move)
    to = move.to_square
    if U.is_in_bad_spot(b, to):
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
    """`move` captures a hanging (undefended) enemy piece for free — nothing recaptures, so the
    whole piece is won regardless of the capturer's value. (A defended victim is a trade.)"""
    if not board.is_capture(move):
        return False
    victim = board.piece_at(move.to_square)
    if victim is None:  # en passant
        return False
    return U.is_hanging(board, victim, move.to_square)


def is_pin(board: chess.Board, move: chess.Move) -> bool:
    """After `move` by pov, an enemy piece is pinned by the moved piece's line."""
    pov = board.turn
    b = board.copy(stack=False); b.push(move)
    for sq, p in b.piece_map().items():
        if p.color == pov:
            continue
        if b.is_pinned(not pov, sq):
            if move.to_square in b.attackers(pov, sq) or _on_pin_ray(b, move.to_square, sq, not pov):
                return True
    return False


def _on_pin_ray(board, mover_sq, pinned_sq, pinned_color):
    return mover_sq in board.pin(pinned_color, pinned_sq)


def is_discovered_attack(board: chess.Board, move: chess.Move) -> bool:
    """`move` unveils an attack: a friendly ray piece (not the one that moved) now attacks a
    valuable enemy piece it did NOT attack before (mover was blocking the line)."""
    pov = board.turn
    before = board
    after = board.copy(stack=False); after.push(move)
    for sq, p in after.piece_map().items():
        if p.color != pov or p.piece_type not in U.ray_piece_types or sq == move.to_square:
            continue
        for tp, tsq in U.attacked_opponent_squares(after, sq, pov):
            if U.values.get(tp.piece_type, 0) >= 3:
                if tsq not in before.attacks(sq) or before.piece_at(sq) is None:
                    if U.squares_are_collinear(sq, move.from_square, tsq):
                        return True
    return False


def exposes_own_king(board: chess.Board, move: chess.Move, threshold: int = 2) -> bool:
    """After `move`, the mover's own king has >= threshold more enemy attackers on/around it."""
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
    "Fork": is_fork, "Hanging Piece": is_hanging_piece, "Pin": is_pin,
    "Discovered Attack": is_discovered_attack,
}


# ============================================================================
#  LINE detectors  (nodes, pov) -> bool          [cook ported verbatim, pov-explicit]
# ============================================================================

def _start_board(nodes: List[ChildNode]) -> Optional[chess.Board]:
    return nodes[0].parent.board() if nodes else None


def _first_fire_index(nodes, pov, single_move_fn, skip_king=True):
    """Index among POV's OWN moves (0 = the move pov should play NOW) at which `single_move_fn`
    (a board,move -> bool detector) first fires, or None. This is what lets us split a tag by DEPTH:
    index 0 = the tactic is directly available (e.g. 'Missed Fork'); index >0 = it comes after a
    setup sequence (e.g. 'Missed Combination -> Fork')."""
    for i, node in enumerate(U.pov_nodes(nodes, pov)[:-1]):
        if skip_king and U.moved_piece_type(node) is KING:
            continue
        if single_move_fn(node.parent.board(), node.move):
            return i
    return None


def fork_line(nodes, pov) -> bool:
    return _first_fire_index(nodes, pov, is_fork) is not None


def fork_depth(nodes, pov):
    """Depth (index among pov's moves) of the first fork, or None. 0 = fork is the move to play now."""
    return _first_fire_index(nodes, pov, is_fork)


def hanging_piece_line(nodes, pov) -> bool:
    # cook hanging_piece(): the opponent's setup move (mainline[0]) leaves a non-pawn piece hanging
    # which pov's first move (mainline[1]) captures, and pov keeps the material to mainline[3].
    if len(nodes) < 2:
        return False
    op0 = nodes[0]                       # opponent's (setup) move == the blunder for ALLOWED
    to = nodes[1].move.to_square
    start = op0.parent.board()
    captured = start.piece_at(to)
    if start.is_check() and (not captured or captured.piece_type == PAWN):
        return False
    if captured and captured.piece_type != PAWN:
        if U.is_hanging(start, captured, to):
            op_move = op0.move
            op_capture = _start_board(nodes).piece_at(op_move.to_square)  # board before op0
            # NB cook uses puzzle.game.board() == position before mainline[0]; that's op0.parent.board()
            op_capture = op0.parent.board().piece_at(op_move.to_square)
            if (op_capture and U.values[op_capture.piece_type] >= U.values[captured.piece_type]
                    and op_move.to_square == to):
                return False
            if len(nodes) < 4:
                return True
            if U.material_diff(nodes[3].board(), pov) >= U.material_diff(nodes[1].board(), pov):
                return True
    return False


def sacrifice_line(nodes, pov) -> bool:
    # A sacrifice = pov INVESTS material that is NOT recovered: pov is still down >=2 at the END of
    # the line (the payoff is positional/mating, not material).
    #
    # cook fires on a TRANSIENT dip at any pov ply (right for puzzles, where the dip is the point).
    # For coaching that's wrong: a mid-exchange dip (you capture, opp recaptures, you recapture back)
    # momentarily reads -3 but nets 0 — NOT a sac. 44% of corpus 'Missed Sacrifice' were these
    # transient dips (e.g. Bxe2 Nxe2 ... Nxc6 bxc6, dead equal). Caught while auditing a real game.
    #
    # Fix: require the deficit to PERSIST to the final position (and pov's last move) — measured vs
    # the position BEFORE pov's first move, so pov's own captures are netted in (the equal-trade fix).
    if len(nodes) < 2:
        return False
    initial = U.material_diff(nodes[0].parent.board(), pov)   # before pov's first move
    pov_nodes = U.pov_nodes(nodes, pov)
    if len(pov_nodes) < 2:
        return False
    end_diff = U.material_diff(nodes[-1].board(), pov)
    last_pov_diff = U.material_diff(pov_nodes[-1].board(), pov)
    # invested >=2 that survives to the end of the line AND to pov's final move (not a transient dip)
    if (end_diff - initial <= -2) and (last_pov_diff - initial <= -2):
        # not a promotion line (cook excludes those — they're combinations, not sacs)
        if not any(n.move.promotion for n in pov_nodes):
            return True
    return False


def x_ray_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        if not U.is_capture(node):
            continue
        prev_op = node.parent
        if not isinstance(prev_op, ChildNode):
            continue
        if prev_op.move.to_square != node.move.to_square or U.moved_piece_type(prev_op) == KING:
            continue
        prev_pl = prev_op.parent
        if not isinstance(prev_pl, ChildNode):
            continue
        if prev_pl.move.to_square != prev_op.move.to_square:
            continue
        if prev_op.move.from_square in SquareSet.between(node.move.from_square, node.move.to_square):
            return True
    return False


def discovered_attack_line(nodes, pov) -> bool:
    if _discovered_check(nodes, pov):
        return True
    for node in U.pov_nodes(nodes, pov)[1:]:
        if U.is_capture(node):
            between = SquareSet.between(node.move.from_square, node.move.to_square)
            if not isinstance(node.parent, ChildNode):
                continue
            if node.parent.move.to_square == node.move.to_square:
                return False
            prev = node.parent.parent
            if not isinstance(prev, ChildNode):
                continue
            if (prev.move.from_square in between
                    and node.move.to_square != prev.move.to_square
                    and node.move.from_square != prev.move.to_square
                    and not U.is_castling(prev)):
                return True
    return False


def _discovered_check(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        checkers = node.board().checkers()
        if checkers and node.move.to_square not in checkers:
            return True
    return False


def double_check_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        if len(node.board().checkers()) > 1:
            return True
    return False


def trapped_piece_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        square = node.move.to_square
        captured = node.parent.board().piece_at(square)
        if captured and captured.piece_type != PAWN:
            prev = node.parent
            if not isinstance(prev, ChildNode):
                continue
            if prev.move.to_square == square:
                square = prev.move.from_square
            if isinstance(prev.parent, (ChildNode,)) or prev.parent is not None:
                if U.is_trapped(prev.parent.board(), square):
                    return True
    return False


def attraction_line(nodes, pov) -> bool:
    for node in nodes[1:]:
        if node.turn() == pov:    # cook: skip pov's own moves here
            continue
        first_move_to = node.move.to_square
        opponent_reply = U.next_node(node)
        if opponent_reply and opponent_reply.move.to_square == first_move_to:
            attracted = U.moved_piece_type(opponent_reply)
            if attracted in (KING, QUEEN, ROOK):
                attracted_to = opponent_reply.move.to_square
                nn = U.next_node(opponent_reply)
                if nn:
                    attackers = nn.board().attackers(pov, attracted_to)
                    if nn.move.to_square in attackers:
                        if attracted == KING:
                            return True
                        n3 = U.next_next_node(nn)
                        if n3 and n3.move.to_square == attracted_to:
                            return True
    return False


def deflection_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        captured_piece = node.parent.board().piece_at(node.move.to_square)
        if captured_piece or node.move.promotion:
            capturing = U.moved_piece_type(node)
            if captured_piece and U.king_values[captured_piece.piece_type] > U.king_values[capturing]:
                continue
            square = node.move.to_square
            prev_op = node.parent
            if not isinstance(prev_op, ChildNode):
                continue
            prev_op_move = prev_op.move
            grandpa = prev_op.parent
            if not isinstance(grandpa, ChildNode):
                continue
            prev_player_move = grandpa.move
            if not isinstance(grandpa.parent, (ChildNode, type(grandpa.parent))) or grandpa.parent is None:
                continue
            prev_player_capture = grandpa.parent.board().piece_at(prev_player_move.to_square)
            if ((not prev_player_capture
                 or U.values[prev_player_capture.piece_type] < U.moved_piece_type(grandpa))
                and square != prev_op_move.to_square
                and square != prev_player_move.to_square
                and (prev_op_move.to_square == prev_player_move.to_square or grandpa.board().is_check())
                and (square in grandpa.board().attacks(prev_op_move.from_square)
                     or (node.move.promotion
                         and square_file(node.move.to_square) == square_file(prev_op_move.from_square)
                         and node.move.from_square in grandpa.board().attacks(prev_op_move.from_square)))
                and (square not in node.parent.board().attacks(prev_op_move.to_square))):
                return True
    return False


def overloading_line(nodes, pov) -> bool:
    return False   # cook stubs this to False; kept for parity, never fires


def intermezzo_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        if U.is_capture(node):
            capture_move = node.move
            capture_square = node.move.to_square
            op_node = node.parent
            if not isinstance(op_node, ChildNode):
                continue
            prev_pov = op_node.parent
            if not isinstance(prev_pov, ChildNode):
                continue
            if op_node.move.from_square not in prev_pov.board().attackers(not pov, capture_square):
                if prev_pov.move.to_square != capture_square:
                    prev_op = prev_pov.parent
                    if not isinstance(prev_op, ChildNode):
                        continue
                    return (prev_op.move.to_square == capture_square
                            and U.is_capture(prev_op)
                            and capture_move in prev_op.board().legal_moves)
    return False


def _self_interference(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        if capture and U.is_hanging(prev_board, capture, square):
            grandpa = node.parent.parent
            if not isinstance(grandpa, ChildNode) and grandpa is None:
                continue
            init_board = grandpa.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            defender_piece = init_board.piece_at(defender) if defender is not None else None
            if defender is not None and defender_piece and defender_piece.piece_type in U.ray_piece_types:
                if node.parent.move and node.parent.move.to_square in SquareSet.between(square, defender):
                    return True
    return False


def _interference(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        if not node.parent.move:
            continue
        if capture and square != node.parent.move.to_square and U.is_hanging(prev_board, capture, square):
            p1 = node.parent
            p2 = p1.parent if p1 else None
            p3 = p2.parent if p2 else None
            if not (isinstance(p2, ChildNode) and p3 is not None):
                continue
            init_board = p3.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            defender_piece = init_board.piece_at(defender) if defender is not None else None
            if defender is not None and defender_piece and defender_piece.piece_type in U.ray_piece_types:
                interfering = p2
                if interfering.move and interfering.move.to_square in SquareSet.between(square, defender):
                    return True
    return False


def interference_line(nodes, pov) -> bool:
    return _self_interference(nodes, pov) or _interference(nodes, pov)


def skewer_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        prev = node.parent
        if not isinstance(prev, ChildNode):
            continue
        capture = prev.board().piece_at(node.move.to_square)
        if capture and U.moved_piece_type(node) in U.ray_piece_types and not node.board().is_checkmate():
            between = SquareSet.between(node.move.from_square, node.move.to_square)
            op_move = prev.move
            if op_move.to_square == node.move.to_square or op_move.from_square not in between:
                continue
            if (U.king_values[U.moved_piece_type(prev)] > U.king_values[capture.piece_type]
                    and U.is_in_bad_spot(prev.board(), node.move.to_square)):
                return True
    return False


def pin_line(nodes, pov) -> bool:
    return _pin_prevents_attack(nodes, pov) or _pin_prevents_escape(nodes, pov)


def _pin_prevents_attack(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        board = node.board()
        for square, piece in board.piece_map().items():
            if piece.color == pov:
                continue
            pin_dir = board.pin(piece.color, square)
            if pin_dir == chess.BB_ALL:
                continue
            for attack in board.attacks(square):
                attacked = board.piece_at(attack)
                if (attacked and attacked.color == pov and attack not in pin_dir
                    and (U.values[attacked.piece_type] > U.values[piece.piece_type]
                         or U.is_hanging(board, attacked, attack))):
                    return True
    return False


def _pin_prevents_escape(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        board = node.board()
        for pinned_sq, pinned_piece in board.piece_map().items():
            if pinned_piece.color == pov:
                continue
            pin_dir = board.pin(pinned_piece.color, pinned_sq)
            if pin_dir == chess.BB_ALL:
                continue
            for attacker_sq in board.attackers(pov, pinned_sq):
                if attacker_sq in pin_dir:
                    attacker = board.piece_at(attacker_sq)
                    if not attacker:
                        continue
                    if U.values[pinned_piece.piece_type] > U.values[attacker.piece_type]:
                        return True
                    if (U.is_hanging(board, pinned_piece, pinned_sq)
                        and pinned_sq not in board.attackers(not pov, attacker_sq)
                        and [m for m in board.pseudo_legal_moves
                             if m.from_square == pinned_sq and m.to_square not in pin_dir]):
                        return True
    return False


def capturing_defender_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        board = node.board()
        capture = node.parent.board().piece_at(node.move.to_square)
        if not isinstance(node.parent, ChildNode):
            continue
        if board.is_checkmate() or (
            capture and U.moved_piece_type(node) != KING
            and U.values[capture.piece_type] <= U.values[U.moved_piece_type(node)]
            and U.is_hanging(node.parent.board(), capture, node.move.to_square)
            and node.parent.move.to_square != node.move.to_square
        ):
            prev = node.parent.parent
            if not isinstance(prev, ChildNode):
                continue
            if not prev.board().is_check() and prev.move.to_square != node.move.from_square:
                if prev.parent is None:
                    continue
                init_board = prev.parent.board()
                defender_sq = prev.move.to_square
                defender = init_board.piece_at(defender_sq)
                if (defender and defender_sq in init_board.attackers(defender.color, node.move.to_square)
                        and not init_board.is_check()):
                    return True
    return False


def exposed_king_line(nodes, pov) -> bool:
    start = _start_board(nodes)
    if start is None:
        return False
    if pov:
        board = start
    else:
        board = start.mirror()
        pov_ = not pov
    pov_eff = pov if pov else (not pov)
    king = board.king(not pov_eff)
    if king is None:
        return False
    if square_rank(king) < 5:
        return False
    squares = SquareSet.from_square(king - 8)
    if square_file(king) > 0:
        squares.add(king - 1); squares.add(king - 9)
    if square_file(king) < 7:
        squares.add(king + 1); squares.add(king - 7)
    for square in squares:
        if board.piece_at(square) == Piece(PAWN, not pov_eff):
            return False
    for node in U.pov_nodes(nodes, pov)[1:-1]:
        if node.board().is_check():
            return True
    return False


def attacking_f2_f7_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        square = node.move.to_square
        if node.parent.board().piece_at(square) and square in (chess.F2, chess.F7):
            king = node.board().piece_at(chess.E8 if square == chess.F7 else chess.E1)
            return king is not None and king.piece_type == KING and king.color != pov
    return False


def _side_attack(nodes, pov, corner_file, king_files, nb_pieces) -> bool:
    start = _start_board(nodes)
    if start is None:
        return False
    back_rank = 7 if pov else 0
    king_square = start.king(not pov)
    if (not king_square or square_rank(king_square) != back_rank
            or square_file(king_square) not in king_files
            or len(start.piece_map()) < nb_pieces
            or not any(n.board().is_check() for n in U.pov_nodes(nodes, pov))):
        return False
    score = 0
    corner = chess.square(corner_file, back_rank)
    for node in U.pov_nodes(nodes, pov):
        corner_dist = square_distance(corner, node.move.to_square)
        if node.board().is_check():
            score += 1
        if U.is_capture(node) and corner_dist <= 3:
            score += 1
        elif corner_dist >= 5:
            score -= 1
    return score >= 2


def kingside_attack_line(nodes, pov) -> bool:
    return _side_attack(nodes, pov, 7, [6, 7], 20)


def queenside_attack_line(nodes, pov) -> bool:
    return _side_attack(nodes, pov, 0, [0, 1, 2], 18)


def clearance_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov)[1:]:
        board = node.board()
        if not node.parent.board().piece_at(node.move.to_square):
            piece = board.piece_at(node.move.to_square)
            if piece and piece.piece_type in U.ray_piece_types:
                prev = node.parent.parent
                if not isinstance(prev, ChildNode):
                    continue
                prev_move = prev.move
                if not isinstance(node.parent, ChildNode):
                    continue
                if (not prev_move.promotion
                    and prev_move.to_square != node.move.from_square
                    and prev_move.to_square != node.move.to_square
                    and not node.parent.board().is_check()
                    and (not board.is_check() or U.moved_piece_type(node.parent) != KING)):
                    if (prev_move.from_square == node.move.to_square
                        or prev_move.from_square in SquareSet.between(node.move.from_square, node.move.to_square)):
                        if (prev.parent and not prev.parent.board().piece_at(prev_move.to_square)
                                or U.is_in_bad_spot(prev.board(), prev_move.to_square)):
                            return True
    return False


def advanced_pawn_line(nodes, pov) -> bool:
    return any(U.is_very_advanced_pawn_move(n) for n in U.pov_nodes(nodes, pov))


def en_passant_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        if (U.moved_piece_type(node) == PAWN
                and square_file(node.move.from_square) != square_file(node.move.to_square)
                and not node.parent.board().piece_at(node.move.to_square)):
            return True
    return False


def castling_line(nodes, pov) -> bool:
    return any(U.is_castling(n) for n in U.pov_nodes(nodes, pov))


def promotion_line(nodes, pov) -> bool:
    return any(n.move.promotion for n in U.pov_nodes(nodes, pov))


def under_promotion_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        if node.board().is_checkmate():
            return node.move.promotion == KNIGHT
        elif node.move.promotion and node.move.promotion != QUEEN:
            return True
    return False


# ============================================================================
#  NAMED MATES  (nodes, pov) -> bool / Optional[str]    [cook ported; read end of line]
# ============================================================================

def mate_in_line(nodes, pov) -> Optional[str]:
    if not nodes or not nodes[-1].board().is_checkmate():
        return None
    # cook uses len(mainline)//2, which assumes node 0 is the opponent's setup move (puzzle shape).
    # We can't assume parity, so count pov's OWN moves directly — robust to MISSED/ALLOWED/raw lines.
    n = len(U.pov_nodes(nodes, pov))
    return f"mateIn{min(n, 5)}" if n >= 1 else None


def _mated_king(nodes, pov):
    end = nodes[-1]
    board = end.board()
    king = board.king(not pov)
    return end, board, king


def back_rank_mate_line(nodes, pov) -> bool:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return False
    back_rank = 7 if pov else 0
    if board.is_checkmate() and square_rank(king) == back_rank:
        squares = SquareSet.from_square(king + (-8 if pov else 8))
        if pov:
            if square_file(king) < 7: squares.add(king - 7)
            if square_file(king) > 0: squares.add(king - 9)
        else:
            if square_file(king) < 7: squares.add(king + 9)
            if square_file(king) > 0: squares.add(king + 7)
        for square in squares:
            piece = board.piece_at(square)
            if piece is None or piece.color == pov or board.attackers(pov, square):
                return False
        return any(square_rank(checker) == back_rank for checker in board.checkers())
    return False


def anastasia_mate_line(nodes, pov) -> bool:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return False
    if square_file(king) in (0, 7) and square_rank(king) not in (0, 7):
        if square_file(end.move.to_square) == square_file(king) and U.moved_piece_type(end) in (QUEEN, ROOK):
            if square_file(king) != 0:
                board.apply_transform(chess.flip_horizontal)
            king = board.king(not pov)
            if king is None:
                return False
            blocker = board.piece_at(king + 1)
            if blocker is not None and blocker.color != pov:
                knight = board.piece_at(king + 3)
                if knight is not None and knight.color == pov and knight.piece_type == KNIGHT:
                    return True
    return False


def hook_mate_line(nodes, pov) -> bool:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return False
    if U.moved_piece_type(end) == ROOK and square_distance(end.move.to_square, king) == 1:
        for rook_def in board.attackers(pov, end.move.to_square):
            defender = board.piece_at(rook_def)
            if defender and defender.piece_type == KNIGHT and square_distance(rook_def, king) == 1:
                for knight_def in board.attackers(pov, rook_def):
                    pawn = board.piece_at(knight_def)
                    if pawn and pawn.piece_type == PAWN:
                        return True
    return False


def arabian_mate_line(nodes, pov) -> bool:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return False
    if (square_file(king) in (0, 7) and square_rank(king) in (0, 7)
            and U.moved_piece_type(end) == ROOK and square_distance(end.move.to_square, king) == 1):
        for knight_sq in board.attackers(pov, end.move.to_square):
            knight = board.piece_at(knight_sq)
            if (knight and knight.piece_type == KNIGHT
                    and abs(square_rank(knight_sq) - square_rank(king)) == 2
                    and abs(square_file(knight_sq) - square_file(king)) == 2):
                return True
    return False


def boden_or_double_bishop_line(nodes, pov) -> Optional[str]:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return None
    bishops = list(board.pieces(BISHOP, pov))
    if len(bishops) < 2:
        return None
    for square in [s for s in SquareSet(chess.BB_ALL) if square_distance(s, king) < 2]:
        if not all(p.piece_type == BISHOP for p in U.attacker_pieces(board, pov, square)):
            return None
    if (square_file(bishops[0]) < square_file(king)) == (square_file(bishops[1]) > square_file(king)):
        return "bodenMate"
    return "doubleBishopMate"


def dovetail_mate_line(nodes, pov) -> bool:
    end, board, king = _mated_king(nodes, pov)
    if king is None:
        return False
    if square_file(king) in (0, 7) or square_rank(king) in (0, 7):
        return False
    queen_sq = end.move.to_square
    if (U.moved_piece_type(end) != QUEEN or square_file(queen_sq) == square_file(king)
            or square_rank(queen_sq) == square_rank(king) or square_distance(queen_sq, king) > 1):
        return False
    for square in [s for s in SquareSet(chess.BB_ALL) if square_distance(s, king) == 1]:
        if square == queen_sq:
            continue
        attackers = list(board.attackers(pov, square))
        if attackers == [queen_sq]:
            if board.piece_at(square):
                return False
        elif attackers:
            return False
    return True


def smothered_mate_line(nodes, pov) -> bool:
    end, board, king_square = _mated_king(nodes, pov)
    if king_square is None:
        return False
    for checker_square in board.checkers():
        piece = board.piece_at(checker_square)
        if piece and piece.piece_type == KNIGHT:
            for esc in [s for s in chess.SQUARES if square_distance(s, king_square) == 1]:
                blocker = board.piece_at(esc)
                if not blocker or blocker.color == pov:
                    return False
            return True
    return False


def named_mate_line(nodes, pov) -> Optional[str]:
    """Returns the most specific mate tag for a line ending in checkmate, or None.
    Mirrors cook's priority: smothered > backRank > anastasia > hook > arabian > boden/doubleBishop > dovetail."""
    mate = mate_in_line(nodes, pov)
    if not mate:
        return None
    if smothered_mate_line(nodes, pov):   return "smotheredMate"
    if back_rank_mate_line(nodes, pov):   return "backRankMate"
    if anastasia_mate_line(nodes, pov):   return "anastasiaMate"
    if hook_mate_line(nodes, pov):        return "hookMate"
    if arabian_mate_line(nodes, pov):     return "arabianMate"
    bdb = boden_or_double_bishop_line(nodes, pov)
    if bdb:                               return bdb
    if dovetail_mate_line(nodes, pov):    return "dovetailMate"
    return None   # generic mate (caller still has mate_in)


# ============================================================================
#  Registries + top-level dispatch
# ============================================================================

# motif key -> line detector
LINE_DETECTORS = {
    "fork": fork_line, "hangingPiece": hanging_piece_line, "sacrifice": sacrifice_line,
    "xRayAttack": x_ray_line, "discoveredAttack": discovered_attack_line, "doubleCheck": double_check_line,
    "trappedPiece": trapped_piece_line, "attraction": attraction_line, "deflection": deflection_line,
    "intermezzo": intermezzo_line, "interference": interference_line, "skewer": skewer_line,
    "pin": pin_line, "capturingDefender": capturing_defender_line, "exposedKing": exposed_king_line,
    "attackingF2F7": attacking_f2_f7_line, "kingsideAttack": kingside_attack_line,
    "queensideAttack": queenside_attack_line, "clearance": clearance_line,
    "advancedPawn": advanced_pawn_line, "enPassant": en_passant_line, "castling": castling_line,
    "promotion": promotion_line, "underPromotion": under_promotion_line,
}


def detect_line(start_board: chess.Board, ucis: List[str], pov: bool) -> dict:
    """Run all line detectors + named mates on the line (start_board then ucis), from pov's POV.
    Returns {motif_key: evidence_str}. Empty if the line is too short / nothing fires."""
    nodes = U.build_line(start_board, ucis)
    if not nodes:
        return {}
    found = {}
    for key, fn in LINE_DETECTORS.items():
        try:
            if fn(nodes, pov):
                found[key] = f"line={' '.join(ucis[:6])}"
        except Exception:
            pass
    # depth annotation for fork: index among pov's moves where it fires (0 = available NOW).
    # The tagger uses this to split "Missed Fork" (depth 0) from "Missed Combination -> Fork" (deeper).
    if "fork" in found:
        try:
            depth = fork_depth(nodes, pov)
            if depth is not None:
                found["fork"] = f"depth={depth} {found['fork']}"
        except Exception:
            pass
    # mates (separate: returns a specific key)
    try:
        mi = mate_in_line(nodes, pov)
        if mi:
            found["mate"] = mi
            named = named_mate_line(nodes, pov)
            if named:
                found[named] = "named mate"
    except Exception:
        pass
    return found


def detect_move(board: chess.Board, move: chess.Move) -> dict:
    """Single-move detectors for the FAILED direction (the played move IS a tactic that backfired)."""
    found = {}
    for key, fn in SINGLE_MOVE.items():
        try:
            if fn(board, move):
                found[key] = f"move={move.uci()}"
        except Exception:
            pass
    return found
