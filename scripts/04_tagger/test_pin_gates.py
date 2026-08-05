#!/usr/bin/env python3
"""Regression tests for the pin gates that were hand-edited into the ECS worker copy and never
made it back to research (found 2026-08-05 as a 3-way tagger divergence: research, worker, and
tag_moments Lambda each held different fixes).

The rule under test: a pin created BY a materially-winning capture is INCIDENTAL. The capture is
the lesson; the pin is a side effect of where the capturing piece landed. Tagging it buried the
real lesson — game1 move 20, Rxe7 wins a hung bishop and happens to line up d7-knight → b7-queen,
which surfaced as "Allowed Pin" instead of "Hung Bishop" (Sam, 2026-07-26).

Both the detector (`_fresh_pin_index`, via `pin_line`) and the namer (`pin_target`) must apply the
SAME gate — otherwise the tag name comes from a move the detector never fired on.

These assertions were mutation-checked: deleting either winning-capture gate flips
pin_line False→True and pin_target None→QUEEN.
"""
import chess
import chess.pgn

import motifs
import chesslib_util as U


def _nodes(fen: str, sans: list) -> list:
    """Build real ChildNodes from a FEN + SAN line (the tagger walks game mainline nodes)."""
    game = chess.pgn.Game()
    game.setup(chess.Board(fen))
    node = game
    for san in sans:
        node = node.add_variation(node.board().parse_san(san))
    return list(game.mainline())


# White's Rxe7 wins an UNDEFENDED bishop (SEE > 0) and incidentally lines up d7-knight → b7-queen.
# The line runs long enough that pov_nodes[:-1] actually includes the Rxe7 node — on a 2-move line
# the slice is empty and the detector never scans it, which would make these tests vacuous.
_WINNING_CAPTURE_FEN = "6k1/pq1nbppp/8/8/8/8/PPP2PPP/4R1K1 w - - 0 1"
_LINE = ["Rxe7", "Kf8", "Re1", "Kg8", "h3", "Kf8"]


class TestWinningCaptureIsNotAPin:
    def test_the_capture_really_is_winning_and_really_does_pin(self):
        # Guards the fixture itself: if either premise breaks, the tests below pass for free.
        board = chess.Board(_WINNING_CAPTURE_FEN)
        move = board.parse_san("Rxe7")
        assert board.is_capture(move)
        assert U.static_exchange_eval(board, move) > 0, "fixture must be a MATERIALLY WINNING capture"
        assert motifs.is_pin(board, move), "fixture must geometrically create a pin"

    def test_pin_line_does_not_fire_on_a_winning_capture(self):
        assert motifs.pin_line(_nodes(_WINNING_CAPTURE_FEN, _LINE), chess.WHITE) is False

    def test_pin_target_is_none_when_the_only_pin_is_incidental(self):
        # Without the gate this returns QUEEN (5) and the mistake gets named "Allowed Pin to Queen".
        assert motifs.pin_target(_nodes(_WINNING_CAPTURE_FEN, _LINE), chess.WHITE) is None


class TestQuietPinsStillFire:
    """The gate must be narrow: it keys on is_capture AND SEE > 0. A quiet pin move is unaffected."""

    def test_quiet_pin_move_still_fires(self):
        # Re1 pins the e7 bishop to the e8 king — nothing on the e-file between them.
        fen = "4k3/4b1pp/8/8/8/8/PPP2PPP/3R2K1 w - - 0 1"
        board = chess.Board(fen)
        move = board.parse_san("Re1")
        assert not board.is_capture(move), "fixture must be a QUIET move"
        assert motifs.is_pin(board, move)
        nodes = _nodes(fen, ["Re1", "Kf8", "h3", "Kg8", "a3", "Kf8"])
        assert motifs.pin_line(nodes, chess.WHITE) is True
        assert motifs.pin_target(nodes, chess.WHITE) is not None
