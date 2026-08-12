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
    than the mover, or hanging and not defended by the mover's new square. (cook fork() per-node body.)

    MUTUAL-DEFENSE case (Sam, 2026-07-17): a target of EQUAL value that's "defended" counts toward the
    fork if ALL its defenders are themselves forked — the defense is illusory. E.g. Na4 forks Qb6 + Bc5
    where the queen and bishop guard each other: White plays Nxb6 (knight for queen), and the bishop's
    only defender is gone. cook's strict test missed this because the bishop reads as "defended". A
    higher-value target is also present (the queen), so this only rescues equal-value co-forked pieces."""
    pov = board.turn
    if board.piece_type_at(move.from_square) == KING:
        return False
    b = board.copy(stack=False); b.push(move)
    to = move.to_square
    if U.is_in_bad_spot(b, to):
        return False
    mover_pt = b.piece_type_at(to)
    targets = [(piece, sq) for piece, sq in U.attacked_opponent_squares(b, to, pov)
               if piece.piece_type != PAWN]
    target_squares = {sq for _, sq in targets}
    nb = 0
    for piece, sq in targets:
        if U.king_values[piece.piece_type] > U.king_values[mover_pt] or \
           (U.is_hanging(b, piece, sq) and sq not in b.attackers(not pov, to)):
            nb += 1
            continue
        # Mutual-defense: a defended target still falls if every one of its defenders (other than the
        # forking piece) is ALSO forked — capturing the co-forked defender first collapses the defense.
        defenders = set(b.attackers(not pov, sq)) - {to}
        if defenders and defenders.issubset(target_squares):
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


def _pin_target(board: chess.Board, move: chess.Move):
    """After `move` by pov (a ray piece), does it pin an enemy piece against a MORE VALUABLE enemy
    piece (or the king) behind it on the same ray? Returns the piece_type pinned-TO (the valuable one
    behind), or None. Covers BOTH absolute pins (to king) and RELATIVE pins (to queen/rook) — the
    latter are invisible to python-chess's is_pinned (king-only). The pinned piece must be less
    valuable than what it shields, else it's not a pin (you'd just take)."""
    # board is ALREADY pushed, so board.turn is the opponent; the mover (pov) is the other color.
    pov = not board.turn
    pt = board.piece_type_at(move.to_square)
    if pt not in U.ray_piece_types:
        return None
    b = board
    to = move.to_square
    fr, ff = chess.square_rank(to), chess.square_file(to)
    # for each ray direction the moved piece travels, walk outward: first enemy piece = candidate
    # pinned; next piece along the same ray, if a more-valuable enemy, is what it's pinned to.
    DIRS = {ROOK: [(1,0),(-1,0),(0,1),(0,-1)], BISHOP: [(1,1),(1,-1),(-1,1),(-1,-1)],
            QUEEN: [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}[pt]
    for dr, df in DIRS:
        first = None
        r, f = fr + dr, ff + df
        while 0 <= r <= 7 and 0 <= f <= 7:
            sq = chess.square(f, r)
            p = b.piece_at(sq)
            if p is not None:
                if first is None:
                    if p.color == pov:
                        break          # own piece first -> no pin this ray
                    if p.piece_type == PAWN:
                        break          # a pinned PAWN is not a real pin worth naming (it's blocked
                                       # along the file/diagonal anyway, nothing is won) -> skip ray
                    # An undefended front piece is only NOT a pin when the pinning move gives CHECK —
                    # then it's a fork/check and pov just wins the loose piece next move; the heavier
                    # piece behind it is coincidental geometry (ply-23: Qd4+ lines up an undefended
                    # c3-knight in front of the a1-rook → a fork, not a pin). But a QUIET move that
                    # lines an undefended piece in front of a heavier one IS a real pin — the front
                    # piece still can't move without dropping what it shields, and it's the opponent's
                    # move, so pov isn't "just taking" anything (ply-52: Ba6 pins the d3-knight to the
                    # f1-rook; correct pin even though the knight is undefended). So: reject an
                    # undefended front ONLY on a checking move. (Refined with Sam, 2026-07-12.)
                    if b.is_check() and not b.is_attacked_by(b.turn, sq):
                        break          # check + hanging front -> fork/check, not a pin on this ray
                    first = p           # candidate pinned enemy PIECE (knight/bishop/rook/queen)
                else:
                    # second piece along ray = what `first` is pinned AGAINST.
                    if p.color != pov and U.king_values[p.piece_type] > U.king_values[first.piece_type]:
                        # The PINNING piece must be worth <= the back piece (or the back piece is the
                        # KING = absolute pin, always real). Else the pin is worthless: you'd never
                        # capture along the ray (you're the most valuable thing on it), so the front
                        # piece isn't actually stuck — it leaves with tempo and wins. (Sam, 2026-07-12:
                        # Qd3 "pinning" Nd4→Rd8 fired Failed Pin, but a QUEEN pinning knight-to-rook is
                        # not a pin — Ne2+ just unpins with a winning check. A BISHOP pinning N→R is.)
                        # QUEEN-TO-QUEEN gate: if the pinning piece is the SAME VALUE as the back piece
                        # (queen pins to queen), the pin doesn't win material — the front piece CAN move
                        # because trading queens is equal. Only positional pressure, not a tactic. Only
                        # fire when the pinner is STRICTLY less than the back piece, or the back is the
                        # king (absolute pin, always real). (Sam, 2026-07-17 #64: Qg4 pins Nf3→Qd1;
                        # queen-to-queen = no material gain, shouldn't be "Allowed Pin to Queen".)
                        if p.piece_type == chess.KING or U.king_values[pt] < U.king_values[p.piece_type]:
                            return p.piece_type   # pinned `first` against more-valuable `p`
                    break
            r += dr; f += df
    return None


def is_pin(board: chess.Board, move: chess.Move) -> bool:
    """After `move` by pov, an enemy piece is pinned (absolute or relative) by the moved piece."""
    b = board.copy(stack=False); b.push(move)
    return _pin_target(b, move) is not None


def pin_target_piece(board: chess.Board, move: chess.Move):
    """The piece_type the pin is AGAINST (KING/QUEEN/ROOK...), or None. For naming the tag."""
    b = board.copy(stack=False); b.push(move)
    return _pin_target(b, move)


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


def is_outpost_move(board: chess.Board, move: chess.Move) -> bool:
    """`move` lands a knight/bishop on an outpost (enemy half, pawn-defended, unchallengeable by an
    enemy pawn). Positional, not tactical — the mover is board.turn."""
    pov = board.turn
    if board.piece_type_at(move.from_square) not in (KNIGHT, BISHOP):
        return False
    b = board.copy(stack=False); b.push(move)
    return U.is_outpost(b, move.to_square, pov)


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


def fork_piece(nodes, pov):
    """The piece TYPE that delivers the first fork (for 'Knight Fork' vs 'Queen Fork' — #53). The
    forking piece is the one MOVED at the firing node. Returns a PIECE_NAME string or None."""
    for node in U.pov_nodes(nodes, pov)[:-1]:
        if U.moved_piece_type(node) is KING:
            continue
        if is_fork(node.parent.board(), node.move):
            pt = node.parent.board().piece_type_at(node.move.from_square)
            return U.PIECE_NAME.get(pt)
    return None


def outpost_line(nodes, pov) -> bool:
    """pov's FIRST move establishes an outpost, and that move is QUIET (not a capture).

    (2026-07-14 audit: the old `_first_fire_index is not None` fired whenever an outpost appeared ANYWHERE
    in the line — 51% of Missed Outpost fires had the outpost knight land 2-4 plies deep with the best MOVE
    being something else (Bxc4, b5, Re5+); another chunk were outpost-BY-CAPTURE where a material tag is the
    real story. Gating to 'best move IS the outpost move AND it's quiet' keeps the genuine positional lesson
    — 'you missed a strong Ne5/Nd5' — the 49% real residue.)"""
    povn = U.pov_nodes(nodes, pov)
    if not povn:
        return False
    first = povn[0]
    if U.moved_piece_type(first) is KING:
        return False
    # first pov move must be QUIET (a capture landing on an outpost square is a material move)
    if first.parent.board().is_capture(first.move):
        return False
    return U.is_outpost(first.board(), first.move.to_square, pov)


def hanging_piece_line(nodes, pov) -> bool:
    # cook hanging_piece(): the opponent's setup move (mainline[0]) leaves a non-pawn piece hanging
    # which pov's first move (mainline[1]) captures, and pov keeps the material to mainline[3].
    if len(nodes) < 2:
        return False
    op0 = nodes[0]                       # opponent's (setup) move == the blunder for ALLOWED
    to = nodes[1].move.to_square
    # nodes[1] MUST be a capture. The whole premise is "nodes[0] leaves a piece hanging WHICH nodes[1]
    # captures", but `captured` below is read off the board BEFORE nodes[0] — two plies earlier — so a
    # quiet nodes[1] can land on a square that merely HAPPENED to be occupied back then.
    #
    # Real false positive (Sam, 2026-08-08): r6k/pp1qbrpp/2p1Rn1B/... played Bg5, refutation `h6 Bh4 ...`.
    # nodes[1] is the pawn push h6; on the pre-Bg5 board h6 still holds WHITE'S OWN BISHOP (it moves to
    # g5 as nodes[0]), which is undefended there — so is_hanging said True and the tag fired on a piece
    # nobody captured. The bishop just retreats with Bh4 next ply.
    if not nodes[1].parent.board().is_capture(nodes[1].move):
        return False
    start = op0.parent.board()
    captured = start.piece_at(to)
    if start.is_check() and (not captured or captured.piece_type == PAWN):
        return False
    # The hanging piece must be the OPPONENT's — pov captures an enemy piece. Puzzle parity
    # guarantees this by construction, but MISSED-direction best-lines start with pov's own
    # move, making nodes[1] the OPPONENT's capture of POV's piece (e.g. O-O Bxg7 Kxg7 read as
    # "missed hanging piece" when g7 was pov's OWN bishop — Sam's Gauntlet report, 2026-07-20).
    if captured and captured.color == pov:
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
    # TRUNCATED-LINE guard: a PV can simply stop in the middle of a combination, and then the final
    # position is a snapshot rather than a settled result. If pov is on move at the end and can just take
    # material back with a favourable capture, the "investment" was never established — the line ended
    # before the recapture, it didn't end because pov chose to stay down.
    #
    # Real false positive (Sam, 2026-08-09, ply 15 of a live game): played d5, refutation
    # `e6 Ng5 exd5 Nxf7 Bc5 Nxd8 Nd4 Bxg4 Nxg4 cxd5` ends with White's knight sitting on d8 having just
    # taken a rook — TRAPPED beside the black king, with Kxd8/Rxd8 available immediately. Raw material
    # read -6 for Black, so "Black sacrificed" and the moment tagged Allowed Sacrifice. But eval_after
    # was -0.90, i.e. Black is BETTER by a pawn; a 7-point contradiction between material and eval is the
    # tell that the material number is mid-exchange.
    #
    # Same bug class as the hung_material settled-peak fix (2026-08-08): measuring material at a ply
    # where the trade isn't finished. That fix could not cover this one because it lives in a different
    # detector, and this instance is invisible in the research corpus, where 94% of refutations are
    # exactly 6 plies and stop before the grab.
    # The subject of a sacrifice claim is the square pov's investment landed on — i.e. wherever the
    # opponent last captured. If that square is still contested, the exchange is unfinished and the
    # "investment" was never established. Uses the shared U.subject_resolved primitive rather than a
    # local guard so every detector asks this the same way (see its docstring).
    last_capture_sq = None
    for node in nodes:
        parent = node.parent
        if parent is not None and parent.board().is_capture(node.move):
            last_capture_sq = node.move.to_square
    if last_capture_sq is not None and not U.subject_resolved(nodes[-1].board(), last_capture_sq):
        return False
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


def discovered_check_line(nodes, pov) -> bool:
    """The discovered attack IS a check — a piece moves and REVEALS a check from a piece behind it.
    Sharper than a plain discovered attack (the revealed line hits the KING), so it gets its own label.
    (Sam, 2026-07-17: c5+ discovers the Qb3 check on the g8 king.)"""
    return _discovered_check(nodes, pov)


def discovered_attack_line(nodes, pov) -> bool:
    # A discovered CHECK is its own (sharper) motif — don't also fire the generic discovered-attack tag.
    if _discovered_check(nodes, pov):
        return False
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
                # REVEALED-PIECE guard (2026-08-08): cook fires here on "prev vacated a square on the
                # ray this capture travels." In a puzzle that means a discovery; in a coaching PV it
                # also matches a piece merely moving DOWN a just-opened line to grab material — the
                # capturer is then the only attacker and nothing was revealed (ply21 dxe6…Qxb7,
                # move18 Nd5…Bxh4, both tagged wrongly). A real discovery has a STATIONARY second pov
                # piece already bearing on the target. Require it.
                board_from = node.parent.board()  # position the capture is played from
                other_attackers = [s for s in board_from.attackers(pov, node.move.to_square)
                                   if s != node.move.from_square]
                if other_attackers:
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


def trapped_piece_line(nodes, pov):
    """Returns the trapped piece type (int) or False."""
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
                    return captured.piece_type
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
        # Minor-piece floor: a real skewer wins the (less-valuable) BACK piece — a real piece, not a
        # pawn. Without this, any ray capture of a pawn on a square a major piece vacated along the ray
        # reads as a "skewer" (e.g. Qe6+ deflects the queen off f3, Bxg2 grabs a pawn — geometry only).
        # That degenerate pawn-grab is 37% of corpus skewer fires (105/286 reconstructable) and is
        # better named by deflection / discovered-attack. Verified: ALL 105 removed fires capture a
        # pawn; every fire that wins >= a minor (181) is kept. (Sam.)
        if capture and capture.piece_type == PAWN:
            continue
        if capture and U.moved_piece_type(node) in U.ray_piece_types and not node.board().is_checkmate():
            between = SquareSet.between(node.move.from_square, node.move.to_square)
            op_move = prev.move
            if op_move.to_square == node.move.to_square or op_move.from_square not in between:
                continue
            if (U.king_values[U.moved_piece_type(prev)] > U.king_values[capture.piece_type]
                    and U.is_in_bad_spot(prev.board(), node.move.to_square)):
                return True
    return False


def _fresh_pin_index(nodes, pov):
    """Like _first_fire_index(is_pin) but rejects a pin whose PINNING piece is captured on the very next
    ply — that's a trade artifact, not an exploitable pin (e.g. Qxc1 momentarily pins the e1 rook to g1,
    but Rxc1 removes the queen). (Sam, 2026-07-17: move 20 f6 got a phantom Missed Pin off a first-rank
    alignment during a forced queen trade.)"""
    povn = U.pov_nodes(nodes, pov)
    for i, node in enumerate(povn[:-1]):
        if U.moved_piece_type(node) is KING:
            continue
        if is_pin(node.parent.board(), node.move):
            # A pin created BY a materially-winning capture is incidental — the capture is the story,
            # not the pin. (Sam, 2026-07-26, game1 move 20: Rxe7 wins the hung e7-bishop and happens to
            # line up d7-knight → b7-queen; the "pin" never cashes in the line, and tagging it buried
            # the real lesson — Hung Bishop — under "Allowed Pin".) Quiet pin moves (Bg5, Ba6) pass.
            if node.parent.board().is_capture(node.move) and \
                    U.static_exchange_eval(node.parent.board(), node.move) > 0:
                continue
            # node.move.to_square is where the pinning piece now sits. If the opponent's reply captures
            # it, the pin dissolves — skip.
            child = node.variations[0] if node.variations else None
            if child is not None and child.move.to_square == node.move.to_square:
                continue
            return i
    return None


def pin_line(nodes, pov) -> bool:
    """cook's pin-prevents-attack/escape (king-pin tactics) OR a fresh pin available RIGHT NOW.

    DEPTH 0 ONLY on the fresh path. "Missed Pin" claims the player failed to play a pin, so the pin has
    to be the move in front of them — not a pin that shows up several plies into a line, contingent on
    the opponent walking into it. (Sam, 2026-08-11, ply 5 of a Vienna: played f4, best line
    `Nf3 Nc6 d4 d6 Bb5 exd4` got "Missed Pin (to King)" off the Bb5 at pov-depth 2. Bb5 WAS legal on
    move 3 and pinned nothing — c6 was empty and d7 still had a pawn; the pin only exists after Black
    volunteers ...Nc6 AND ...d6. The line never cashes it in either: ...Bd7 neutralizes it. That's the
    Ruy-López developing bishop, not a tactic anyone missed.)

    Two reasons this is a drop rather than a `Combination → Pin` relabel, which is how the fork family
    handles depth (`fork_depth` / `_motif_label`): a pin wins nothing by itself, so a deep one that never
    cashes in teaches nothing; and of the 13 depth>0 fires in the corpus, 12 already carry a sharper
    explain tag (Missed Sacrifice, Hung Material, ...) — dropping the pin loses the lesson in 1 case out
    of 13 while removing a confidently-wrong claim from the other 12.

    Deliberately NOT applied to `_pin_prevents_*`: those require the pin to be EXPLOITED in the line
    (the pinned piece can't defend / can't escape), which is its own soundness check. They are also 94%
    of all pin fires — this gate touches only the 6% fresh-pin path.
    """
    return (_pin_prevents_attack(nodes, pov) or _pin_prevents_escape(nodes, pov)
            or _fresh_pin_index(nodes, pov) == 0)


def pin_target(nodes, pov):
    """The piece_type the pin is AGAINST (KING/QUEEN/ROOK) — for naming. Returns the most valuable
    target among pov's QUALIFYING pinning moves — the same gates as _fresh_pin_index (skip winning
    captures / dissolved pins), else the name can come from a move the detector didn't fire on
    (game1 move 20: the Rxe8 trade recapture named "to King" while the fired pin was "to Queen")."""
    best = None
    povn = U.pov_nodes(nodes, pov)
    for i, node in enumerate(povn[:-1] or povn):
        tgt = _pin_target_after(node)
        if tgt is None:
            continue
        b = node.parent.board()
        if b.is_capture(node.move) and U.static_exchange_eval(b, node.move) > 0:
            continue  # incidental pin on a winning capture — not the fired pin
        child = node.variations[0] if node.variations else None
        if child is not None and child.move.to_square == node.move.to_square:
            continue  # pinning piece immediately captured — dissolved
        if best is None or U.king_values[tgt] > U.king_values[best]:
            best = tgt
    if best is not None:
        return best
    # FALLBACK: pin_line also fires via _pin_prevents_attack, which exploits a pin pov did NOT create —
    # so no pov move has a target to read and the loop above finds nothing. The label then shipped as a
    # bare "Allowed Pin" with no target (GH #102), even when the pin is ABSOLUTE, which is the most
    # nameable case there is. Real example: Bc5 pins the f2 pawn against Kg1, and that pin is what makes
    # ...Qxg3+ safe. Read the target off the board instead: find pov's opponent's pinned pieces and name
    # the most valuable thing each one shields.
    for node in U.pov_nodes(nodes, pov):
        board = node.board()
        for square, piece in board.piece_map().items():
            if piece.color == pov:
                continue
            if board.pin(piece.color, square) == chess.BB_ALL:
                continue           # not pinned
            # python-chess's board.pin() is king-only, so a pinned piece is by definition shielding the
            # king — an absolute pin. Relative pins (to queen/rook) are found by the pov-move loop above.
            tgt = KING
            if best is None or U.king_values[tgt] > U.king_values[best]:
                best = tgt
    return best


def _pin_target_after(node):
    """pin target of node.move played from node.parent.board()."""
    try:
        return _pin_target(node.board(), node.move)
    except Exception:
        return None


def _preexisting_pins(nodes, pov):
    """Enemy squares ALREADY pinned on the line's start board. A "Missed Pin" must be a pin pov's
    line establishes — a pin that was on the board before pov moved is not something pov missed.
    Without this gate, _pin_prevents_* fire on static pre-existing pins (e.g. a knight already
    pinned to its rook by your queen), producing phantom "Missed Pin" tags on quiet moves. (Caught
    by Sam: ply where best=Ng5 tagged "Missed Pin" off Black's g8-knight pinned to h8 at line start.)"""
    start = _start_board(nodes)
    if start is None:
        return set()
    pinned = set()
    for sq, pc in start.piece_map().items():
        if pc.color != pov and start.pin(pc.color, sq) != chess.BB_ALL:
            pinned.add(sq)
    return pinned


def _pinner_captured_next_ply(node, pinned_sq, pov) -> bool:
    """True if the pov piece delivering the pin on `pinned_sq` gets captured on the very next ply.
    Such a "pin" is an artifact of a capture/trade sequence (e.g. Qxc1 momentarily pins the e1 rook to
    g1, but Rxc1 removes the queen next move) — not an exploitable pin. (Sam, 2026-07-17: move 20 f6
    tagged a phantom Missed Pin off a first-rank alignment during a forced queen trade.)"""
    board = node.board()
    pinners = [s for s in board.attackers(pov, pinned_sq)
               if board.piece_at(s) and board.piece_at(s).piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)]
    if not pinners:
        return False
    child = node.variations[0] if node.variations else None
    if child is None:
        return False
    return child.move.to_square in pinners   # the opponent's reply captures the pinning piece


def _pin_prevents_attack(nodes, pov) -> bool:
    preexisting = _preexisting_pins(nodes, pov)
    for node in U.pov_nodes(nodes, pov):
        board = node.board()
        for square, piece in board.piece_map().items():
            if piece.color == pov or square in preexisting:
                continue
            pin_dir = board.pin(piece.color, square)
            if pin_dir == chess.BB_ALL:
                continue
            for attack in board.attacks(square):
                attacked = board.piece_at(attack)
                if (attacked and attacked.color == pov and attack not in pin_dir
                    and (U.values[attacked.piece_type] > U.values[piece.piece_type]
                         or U.is_hanging(board, attacked, attack))):
                    if _pinner_captured_next_ply(node, square, pov):
                        continue   # pin dissolves next move — a trade artifact, not exploitable
                    return True
    return False


def _pin_prevents_escape(nodes, pov) -> bool:
    preexisting = _preexisting_pins(nodes, pov)
    for node in U.pov_nodes(nodes, pov):
        board = node.board()
        for pinned_sq, pinned_piece in board.piece_map().items():
            if pinned_piece.color == pov or pinned_sq in preexisting:
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


# kingside/queenside_attack + _side_attack REMOVED 2026-07-15. The `_side_attack` heuristic (pov's moves
# cluster near the enemy king's corner + a check somewhere in the line) fired on essentially ANY sharp
# attacking line. Audit of all 4 directions (Missed/Allowed × Kingside/Queenside, ~2.6k fires): every one
# sat at 2% sole-explain — 98% co-fired a sharper tag (Allowed Mate, Missed Sacrifice, Hung X, Missed
# Fork), and reading the sole cases showed concrete sacs/forks/mates (Rxh2+ Kxh2 Rh8+, Bxh3 gxh3 Qxh3+)
# those tags already own. "Kingside Attack" is ambient description of a sharp position, not a distinct
# teachable mistake. Same signature as battery/desperado. See tagger_feature_ledger.md.


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
                        # A clearance worth tagging is FORCING — the clearing move is either:
                        #   (a) a SACRIFICE (clearing piece is en prise / in a bad spot), or
                        #   (b) a CHECK (forces the opponent's reply, so opening the line is the point —
                        #       "clearance with tempo", e.g. N-moves-with-check to open a B's mating
                        #       diagonal even though the knight lands safely).
                        # cook's original loophole — "clearing piece moved to an empty square" — fired
                        # on ANY quiet line-opening move (a safe Rf4-f3 that incidentally vacates a
                        # diagonal a bishop later uses). Dropped; require sacrifice-or-check instead.
                        if U.is_in_bad_spot(prev.board(), prev_move.to_square) or prev.board().is_check():
                            # RE-OCCUPATION guard: when the follower merely LANDS ON the square the
                            # clearer vacated (Rxe6 then Rd8-e8 back to e8), that's the natural follow-up
                            # to a sac — not a clearance tactic — UNLESS the lander creates a threat from
                            # there (check / attacks a non-pawn piece). A follower that travels THROUGH
                            # the cleared square (Ne6+ clearing d4 for Ba1-h8) is real clearance geometry
                            # and needs no extra threat. (Sam, 2026-07-17 #67.)
                            if prev_move.from_square == node.move.to_square:
                                if board.is_check():
                                    return True
                                if any((pc := board.piece_at(t)) and pc.color != pov and pc.piece_type != PAWN
                                       for t in board.attacks(node.move.to_square)):
                                    return True
                            else:
                                return True
    return False


def advanced_pawn_line(nodes, pov) -> bool:
    # NOT a scored skill feature — kept only as an info/context motif. Two measurements proved it can't
    # be a corpus skill-signal: (1) raw "any pawn to rank 6+" is a board FEATURE, flat ~2x band curve;
    # (2) gating on PASSED pawn INVERTS it (ratio 0.2 — masters reach passed-pawn pushes far more often,
    # so it measures position-reaching, not mistake-skill). Dropped from the Piece Activity cluster
    # (2026-06-29). Detector stays for evidence/labeling but feeds no scored cluster.
    return any(U.is_very_advanced_pawn_move(n) for n in U.pov_nodes(nodes, pov))


def en_passant_line(nodes, pov) -> bool:
    for node in U.pov_nodes(nodes, pov):
        if (U.moved_piece_type(node) == PAWN
                and square_file(node.move.from_square) != square_file(node.move.to_square)
                and not node.parent.board().piece_at(node.move.to_square)):
            return True
    return False


def castling_line(nodes, pov) -> bool:
    """Fire ONLY when pov's FIRST move in the line is castling — i.e. the move being recommended IS
    to castle. (2026-07-14 audit: the old `any(... pov moves)` fired whenever castling appeared ANYWHERE
    in the continuation — 73% of Missed Castling fires had the best MOVE be something else, e.g.
    `Qxd5 Bf5 O-O`, with O-O a routine follow-up 2-4 plies deep. That's not a 'you should have castled'
    lesson; the real lesson was the first move. Gating to the first pov move keeps only the genuine
    king-safety case.) Note: `castling` is also MISSED-ONLY (see _MISSED_ONLY_MOTIFS in tagger.py) —
    'Allowed Castling' (opponent castled in the refutation) was deleted: letting the opponent castle is
    normal chess, never the teachable mistake — the real error was always something concrete the sharper
    tags name."""
    povn = U.pov_nodes(nodes, pov)
    return bool(povn) and U.is_castling(povn[0])


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
    # Both bishops must actually bear on the king's zone — a TWO-bishop mate, not a single-bishop mate
    # that happens to have a spectator bishop sitting elsewhere. Boden/Double-Bishop is a named tactic
    # defined by two criss-crossing bishops delivering the net; checking bishop COUNT alone fires on e.g.
    # Bh4# (single bishop mates the e1 king) whenever a second, idle bishop is on the board — 2 of 4
    # corpus fires were exactly this. (Same class as the Greek Gift proper-noun fix; check the
    # definition, not the vibe.)
    zone = [king] + [s for s in chess.SQUARES if square_distance(s, king) == 1]
    participating = [b for b in bishops
                     if any(b in board.attackers(pov, sq) for sq in zone)]
    if len(participating) < 2:
        return None
    for square in [s for s in SquareSet(chess.BB_ALL) if square_distance(s, king) < 2]:
        if not all(p.piece_type == BISHOP for p in U.attacker_pieces(board, pov, square)):
            return None
    # Classify by the two PARTICIPATING bishops (a spectator's file must not decide boden vs double).
    b0, b1 = participating[0], participating[1]
    if (square_file(b0) < square_file(king)) == (square_file(b1) > square_file(king)):
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
    "xRayAttack": x_ray_line, "discoveredAttack": discovered_attack_line,
    "discoveredCheck": discovered_check_line, "doubleCheck": double_check_line,
    "trappedPiece": trapped_piece_line, "attraction": attraction_line, "deflection": deflection_line,
    "intermezzo": intermezzo_line, "interference": interference_line, "skewer": skewer_line,
    "pin": pin_line, "capturingDefender": capturing_defender_line, "exposedKing": exposed_king_line,
    "attackingF2F7": attacking_f2_f7_line, "clearance": clearance_line,
    "advancedPawn": advanced_pawn_line, "enPassant": en_passant_line, "castling": castling_line,
    "promotion": promotion_line, "underPromotion": under_promotion_line, "outpost": outpost_line,
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
            result = fn(nodes, pov)
            if result:
                found[key] = f"line={' '.join(ucis[:6])}"
                if key == "trappedPiece" and isinstance(result, int):
                    found[key] = f"piece={U.PIECE_NAME.get(result, 'Piece')} {found[key]}"
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
        # #53: name the forking piece so the label is "Knight Fork" not generic "Fork".
        try:
            fp = fork_piece(nodes, pov)
            if fp:
                found["fork"] = f"forkpiece={fp} {found['fork']}"
        except Exception:
            pass
    # pin target annotation: what the pin is AGAINST (king/queen/rook), for naming "Pin (to Queen)".
    if "pin" in found:
        try:
            tgt = pin_target(nodes, pov)
            if tgt is not None:
                found["pin"] = f"target={U.PIECE_NAME[tgt]} {found['pin']}"
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
