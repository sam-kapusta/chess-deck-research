#!/usr/bin/env python3
"""Tests for the win%-drop tagger gate (issue #29 step 1-2).

win_drop is the shared "how taggable is this mistake" currency, replacing 8 magic cp_loss gates.
It must be NONLINEAR in eval (the whole point: a 100cp slip costs ~18 win% at even but ~3 win% at
+400, so we correctly down-weight slips made while already winning) and must degrade gracefully when
signed evals are unavailable (mate positions, and the cp-only SAE-feature analysis caches).
"""
import math
import chess
import pytest

from chesslib_util import winpct, win_drop
from mistake import Mistake


# ---------------- winpct: the Lichess logistic, 0-100, side-to-move POV ----------------

def test_winpct_even_is_fifty():
    assert winpct(0) == pytest.approx(50.0)

def test_winpct_monotonic_and_bounded():
    assert winpct(-2000) < winpct(-100) < winpct(0) < winpct(100) < winpct(2000)
    assert 0.0 <= winpct(-5000) <= 100.0
    assert 0.0 <= winpct(5000) <= 100.0

def test_winpct_symmetric_about_even():
    # logistic is odd around 0 -> win% is symmetric about 50
    assert winpct(300) + winpct(-300) == pytest.approx(100.0)

def test_winpct_matches_lichess_formula():
    cp = 250
    expect = 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)
    assert winpct(cp) == pytest.approx(expect)


# ---------------- win_drop: mover-POV win% lost, nonlinear, clamped >= 0 ----------------

def test_win_drop_white_even_position():
    # White to move, +0 -> -100 after the played move. Real drop, mover POV.
    wd = win_drop(eval_before=0, eval_after=-100, mover=chess.WHITE)
    assert wd == pytest.approx(winpct(0) - winpct(-100))
    assert wd > 0

def test_win_drop_black_pov_sign_flip():
    # Black to move. Stored white-POV: before=0, after=+100 (good for white = bad for black).
    # From black's POV that's a real loss; must be positive and equal the white-even-loss case.
    wd_black = win_drop(eval_before=0, eval_after=100, mover=chess.BLACK)
    wd_white = win_drop(eval_before=0, eval_after=-100, mover=chess.WHITE)
    assert wd_black == pytest.approx(wd_white)
    assert wd_black > 0

def test_win_drop_nonlinearity_down_weights_when_winning():
    # The headline property: the SAME 100cp swing costs monotonically less win% the more you're
    # already winning (leak-metric intent: a slip at +900 barely costs win%, the same slip at 0 hurts).
    drop_at_even = win_drop(eval_before=0, eval_after=-100, mover=chess.WHITE)
    drop_ahead = win_drop(eval_before=400, eval_after=300, mover=chess.WHITE)
    drop_clearly_winning = win_drop(eval_before=900, eval_after=800, mover=chess.WHITE)
    assert drop_clearly_winning < drop_ahead < drop_at_even
    assert drop_clearly_winning < drop_at_even / 3  # at +900 the 100cp slip barely registers

def test_win_drop_never_negative():
    # A move that IMPROVES eval (shouldn't happen for a mistake, but be safe) clamps to 0.
    assert win_drop(eval_before=-100, eval_after=0, mover=chess.WHITE) == 0.0

def test_win_drop_fallback_to_cp_loss_when_evals_missing():
    # cp-only analysis caches (jumprelu_untagged_features etc.) pass eval_before/after=None.
    # Must NOT return 0 (that would silently un-tag everything) — fall back to cp_loss from even.
    wd = win_drop(eval_before=None, eval_after=None, mover=chess.WHITE, cp_loss=300)
    assert wd == pytest.approx(winpct(0) - winpct(-300))
    assert wd > 0

def test_win_drop_fallback_needs_cp_loss():
    # No evals AND no cp_loss -> genuinely nothing to gate on -> 0.
    assert win_drop(eval_before=None, eval_after=None, mover=chess.WHITE) == 0.0

def test_win_drop_partial_evals_use_fallback():
    # Missed mate: eval_before is mate (None) but eval_after present. Need BOTH for the real path;
    # otherwise fall back to cp_loss so mate blunders still clear the gate.
    wd = win_drop(eval_before=None, eval_after=300, mover=chess.WHITE, cp_loss=9999)
    assert wd > 40  # sentinel cp_loss -> large drop -> taggable


# ---------------- Mistake.win_drop property wires the dataclass fields in ----------------

def _mk(eb, ea, mover, cp_loss=0):
    return Mistake(
        fen_before=chess.STARTING_FEN, played_uci="e2e4", best_uci="d2d4",
        best_line_san=[], refutation_san=[], eval_before=eb, eval_after=ea,
        cp_loss=cp_loss, mover=mover,
    )

def test_mistake_win_drop_uses_real_evals():
    m = _mk(0, -120, chess.WHITE)
    assert m.win_drop == pytest.approx(win_drop(0, -120, chess.WHITE))

def test_mistake_win_drop_black_mover():
    m = _mk(0, 120, chess.BLACK)
    assert m.win_drop == pytest.approx(win_drop(0, 120, chess.BLACK, cp_loss=0))
    assert m.win_drop > 0

def test_mistake_win_drop_falls_back_to_cp_loss():
    m = _mk(None, None, chess.WHITE, cp_loss=250)
    assert m.win_drop == pytest.approx(win_drop(None, None, chess.WHITE, cp_loss=250))
    assert m.win_drop > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
