"""Regression set for motif detectors — curated, known-answer positions.

Two sources of truth:
  1. Constructed/clean positions (unambiguous textbook motifs).
  2. Positions drawn from Sam's hand-confirmed SAE gold features (f54=fork, f17/f3=capture, ...),
     curated to the INDIVIDUAL position level (NOT whole features — features are polysemantic).

Each detector must pass its cases. Run: python3 scripts/04_tagger/regression.py
Add cases here as each detector is built/tweaked.
"""
import sys, os, chess
sys.path.insert(0, os.path.dirname(__file__))
import motifs as M

# (name, fen, move_uci, detector, expected) — all FENs verified legal + answer hand-checked
SINGLE_MOVE_CASES = [
    # --- fork (high value, fully validated incl. 6/6 of gold f54 best-moves) ---
    ("constructed Nc7+ royal fork", "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", "d5c7", M.is_fork, True),
    ("f54 gold best e4f2 fork", "2rqr1k1/p4pp1/1pn1p1p1/3p4/3Pn3/P3PN1P/1P1B1PP1/2RQR2K b - - 3 18", "e4f2", M.is_fork, True),
    ("f54 gold best d4e2 fork", "2kr3r/4qp2/3p1b1p/p3p3/2PnP3/P1Q4P/1PPN1PP1/2KR3R b - - 4 21", "d4e2", M.is_fork, True),
    ("quiet opening move != fork", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", M.is_fork, False),
    # --- hanging capture ---
    ("Rxe5 free knight", "4k3/8/8/4n3/8/8/4R3/4K3 w - - 0 1", "e2e5", M.is_hanging_piece, True),
    ("Rxe5 defended knight (not hang)", "4k3/4r3/8/4n3/8/8/4R3/4K3 w - - 0 1", "e2e5", M.is_hanging_piece, False),
    # --- pin (d7 empty so the b5-e8 diagonal is clear) ---
    ("Bb5 pins Nc6 to king", "r1bqkbnr/ppp1pppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "f1b5", M.is_pin, True),
    # --- skewer: KNOWN TODO (geometry needs work; rare 0.6% of corpus). No case asserted yet. ---
]


def run():
    ok = 0; fails = []
    for name, fen, uci, fn, exp in SINGLE_MOVE_CASES:
        b = chess.Board(fen)
        try:
            mv = chess.Move.from_uci(uci)
            got = fn(b, mv) if mv in b.legal_moves else f"ILLEGAL({uci})"
        except Exception as e:
            got = f"ERR:{e}"
        passed = (got == exp)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {fn.__name__:20} {name}: got={got} exp={exp}")
    print(f"\n{ok}/{len(SINGLE_MOVE_CASES)} passed" + (f" | FAILS: {fails}" if fails else ""))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
