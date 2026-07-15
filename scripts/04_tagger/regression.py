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
import chesslib_util as U
import tagger as T

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
    # HANGING front piece is NOT a pin — you just capture it; the valuable piece behind is
    # coincidental geometry. Here Black Qd4 lines up c3-knight (UNDEFENDED) in front of the a1-rook;
    # it's a fork (Qd4+ also checks g1), not a Pin to Rook. (Caught by Sam via Gemini, ply 23.)
    ("Qd4 hits hanging knight = not a pin", "2kr1b1r/pppq2p1/4pn2/4p3/1P2P3/2NP3P/P1P3P1/R1BQ1RK1 b - - 0 12", "d7d4", M.is_pin, False),
    # A QUEEN pinning a knight to a ROOK is NOT a real pin — the pinner (9) outvalues the back piece
    # (rook 5), so you'd never capture along the ray; the "pinned" knight leaves with tempo. (Sam,
    # 2026-07-12: Qd3→Nd4→Rd8 fired a phantom Failed Pin. The pinner's value must be ≤ the back piece,
    # or the back piece is the KING = absolute pin.) Even a DEFENDED front doesn't make it a pin here.
    ("Qd4 (queen) pinning knight to rook is NOT a pin", "2kr1b1r/pppq2p1/4pn2/4p3/4P3/2NP4/P2P2P1/R1BQ1RK1 b - - 0 12", "d7d4", M.is_pin, False),
    # QUIET move lining up an UNDEFENDED front piece in front of a heavier one IS a real pin — the
    # front piece can't move without dropping what it shields, and it's the opponent's move (pov isn't
    # "just taking"). Only a CHECK makes an undefended front a fork instead. Real game (cabbage) ply-52:
    # Ba6 pins the undefended d3-knight to the f1-rook. Distinguishes the guard from "undefended = never
    # a pin". (Refined with Sam, 2026-07-12.)
    ("Ba6 pins UNDEFENDED knight to rook (quiet move)", "2r1r2k/pb4pp/1p3p2/n5n1/3P4/P1PN1PB1/B5PP/2R2RK1 b - - 2 26", "b7a6", M.is_pin, True),
    # --- skewer: KNOWN TODO (geometry needs work; rare 0.6% of corpus). No case asserted yet. ---
]


# (name, fen, line_ucis, pov_is_white, motif_key, expected_present) — line detectors + mates.
# pov is WHATEVER side the detector should evaluate from: for an ALLOWED-shape line (blunder then
# refutation) pov = the punisher (opponent of the mover); for a raw winning line pov = the winner.
LINE_CASES = [
    # back-rank mate: white Re8# (single move, pov=white). king g8 boxed by f7/g7/h7.
    ("back-rank Re8#", "6k1/5ppp/8/8/8/8/8/R3R1K1 w - - 0 1", ["e1e8"], True, "backRankMate", True),
    ("back-rank also tags mate", "6k1/5ppp/8/8/8/8/8/R3R1K1 w - - 0 1", ["e1e8"], True, "mate", True),
    # smothered mate: white Nh6-f7# (king h8 boxed by Rg8 + g7/h7 pawns).
    ("smothered Nf7#", "6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1", ["h6f7"], True, "smotheredMate", True),
    # f59 ALLOWED: black g6h5 allows forced mate; punisher=white. Must tag mate (mateIn2), NOT a named mate.
    ("f59 allowed mate", "r2qr1k1/1b2bp1p/pnp1p1pQ/1p1nP2N/2pP4/5N2/PPB2PPP/R1B1R1K1 b - - 3 16",
     ["g6h5", "h6h7", "g8f8", "h7h8"], True, "mate", True),
    # back-rank should NOT fire on a non-mate quiet line
    ("quiet line != backrank", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     ["e2e4", "e7e5"], True, "backRankMate", False),
    # PRE-EXISTING pin must NOT tag "Missed Pin". Best line is Ng5 (attacks queen, gains tempo) — no
    # pin created. Black's g8-knight is already pinned to h8 by White's Qh8 at line start; the
    # _pin_prevents_escape geometry scan used to fire on that static pin. pov=White (the missed line).
    # (Caught by Sam: ply 19, b3 blunder, this exact false-positive "Missed Pin".)
    ("preexisting pin != missed pin", "r1b1k1nQ/ppp2q1p/3p4/2b1p3/4P3/2N2N2/PP1P1PPP/n1BK3R w q - 0 10",
     ["f3g5", "f7h5", "g5f3", "c8e6", "h8g7", "e8c8"], True, "pin", False),
    # ...but a pin pov's line actually ESTABLISHES still fires. Same position, played b3 allowed
    # Black's Bg4 pinning Nf3 to Kd1. pov=Black (the punisher in the allowed line).
    ("established pin still tags", "r1b1k1nQ/ppp2q1p/3p4/2b1p3/4P3/2N2N2/PP1P1PPP/n1BK3R w q - 0 10",
     ["b2b3", "c8g4", "d2d4", "g4f3", "g2f3", "e8c8"], False, "pin", True),
    # --- skewer: minor-piece floor. A skewer wins the (less-valuable) BACK piece — a real piece.
    # NEG: Qf3 allowed Qe6+ Qe2 Bxg2 — the bishop only grabs a PAWN on g2 (the queen was DEFLECTED off
    # f3 by the check, not driven off by the bishop). Pawn-back => NOT a skewer. (Sam, ply 15.)
    ("pawn-back deflection != skewer", "rn2kbnr/pb2pppp/1p6/1Ppq4/8/P1PB4/3P1PPP/RNBQK1NR w KQkq - 0 8",
     ["d1f3", "d5e6", "f3e2", "b7g2", "f2f3", "g8f6"], False, "skewer", False),
    # POS: classic file skewer — Re1+ checks Ke4, king steps to d4, Rxe8 wins the ROOK behind it.
    # Front=king (skewered), back=rook (won, >= minor). pov=White (the winner). Must still fire.
    ("king-front rook-back still skewers", "4r3/8/8/8/4k3/8/8/R5K1 w - - 0 1",
     ["a1e1", "e4d4", "e1e8"], True, "skewer", True),
]


def run():
    ok = 0; fails = []
    print("--- single-move ---")
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

    print("--- line / mate ---")
    for name, fen, ucis, pov_white, key, exp in LINE_CASES:
        try:
            b = chess.Board(fen)
            pov = chess.WHITE if pov_white else chess.BLACK
            res = M.detect_line(b, ucis, pov)
            got = (key in res)
        except Exception as e:
            got = f"ERR:{e}"
        passed = (got == exp)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {key:16} {name}: present={got} exp={exp}")

    print("--- tagger: fork depth split ---")
    # depth=0 -> 'Fork' (available now); depth>0 -> 'Combination → Fork' (after setup)
    split_cases = [
        ("depth0 -> Fork", "fork", "depth=0 line=...", "Fork"),
        ("depth1 -> Combination", "fork", "depth=1 line=...", "Combination → Fork"),
        ("depth2 -> Combination", "fork", "depth=2 line=...", "Combination → Fork"),
        ("no depth prefix -> Fork", "fork", "line=...", "Fork"),
        # #53: name the forking piece. forkpiece= prefix -> "Knight Fork"; with depth -> combination form.
        ("forkpiece Knight depth0 -> Knight Fork", "fork", "forkpiece=Knight depth=0 line=...", "Knight Fork"),
        ("forkpiece Queen depth1 -> Combination Queen Fork", "fork", "depth=1 forkpiece=Queen line=...", "Combination → Queen Fork"),
        ("forkpiece only (no depth) -> Bishop Fork", "fork", "forkpiece=Bishop line=...", "Bishop Fork"),
    ]
    for name, key, ev, want in split_cases:
        got = T._motif_label(key, ev)
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: label={got!r} exp={want!r}")

    import predicates as PR
    from mistake import Mistake

    print("--- predicates: conversion_outcome (result-band transition, descriptive) ---")
    # (name, eval_before, eval_after, mover, expected). White-POV cp evals.
    conv_cases = [
        ("W winning -> losing", 400, -400, chess.WHITE, "Winning → Losing"),
        ("W winning -> drawn", 400, 20, chess.WHITE, "Winning → Drawn"),
        ("B winning -> drawn (POV)", -400, 20, chess.BLACK, "Winning → Drawn"),
        ("no band change -> no fire", 400, 350, chess.WHITE, None),
        ("even -> losing", 50, -400, chess.WHITE, "Even → Losing"),
    ]
    for name, eb, ea, mv, want in conv_cases:
        m = Mistake("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2", "", [], [], eb, ea, 0, mv)
        res = PR.conversion_outcome(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_conv = len(conv_cases)

    print("--- predicates: blunder_severity (sharp vs slow bleed, saturation-guarded) ---")
    sev_cases = [
        ("balanced +20 -> -400 = Sharp", 20, -400, chess.WHITE, "Sharp Blunder"),
        ("balanced +30 -> +10 = Slow Bleed", 30, 10, chess.WHITE, "Slow Bleed"),
        ("saturated +900 -> +600 = neither (not bleed)", 900, 600, chess.WHITE, None),
        ("missed mate 1100 -> 700 = neither (not bleed)", 1100, 700, chess.WHITE, None),
        ("balanced -30 -> -120 = Slow Bleed", -30, -120, chess.WHITE, "Slow Bleed"),
    ]
    for name, eb, ea, mv, want in sev_cases:
        m = Mistake("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2", "", [], [], eb, ea, 0, mv)
        res = PR.blunder_severity(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_sev = len(sev_cases)

    print("--- predicates: move_difficulty (only-move vs careless, from n_good_moves) ---")
    md_cases = [
        ("1 good move = Only Good Move Missed", 1, "Only Good Move Missed"),
        ("2 good = neither", 2, None),
        ("3 good = neither", 3, None),
        ("4 good = Careless Blunder", 4, "Careless Blunder"),
        ("None (no MultiPV) = no fire", None, None),
    ]
    for name, ngm, want in md_cases:
        m = Mistake("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2", "", [], [], 0, 0, 0, chess.WHITE, n_good_moves=ngm)
        res = PR.move_difficulty(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_md = len(md_cases)

    print("--- predicates: hung material (equal trades excluded, immediate vs delayed) ---")
    # (name, fen, played_uci, refutation_san, expected_label_or_None)
    hung_cases = [
        # dead-equal trade Bxc6 bxc6 (bishop-for-knight, 3-for-3) — must NOT fire (the bug Sam caught)
        ("equal trade != hung", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
         "c4c6", ["bxc6"], None),
        # quiet move, opponent grabs a free bishop next move — immediate hang, NAMED by the victim piece
        ("free bishop = Hung Bishop", "3k4/8/8/3b4/8/8/3R4/4K3 b - - 0 1",
         "d8c8", ["Rxd5"], "Hung Bishop"),
        # PEAK-LOSS: queen hung MID-line for partial compensation. move 18 Qd3, Ne2+ Rxe2 Rxd3 (queen
        # gone, -6 at worst) Rxe8+ Kd7 Rae1 (rook back, nets -1). End-of-line saw only -1 (pawn) and
        # stayed silent; peak-loss + end>=1 catches the real Hung Queen. (Sam, 2026-07-12.)
        ("hung queen with partial recovery = Hung Queen",
         "2krr3/ppp2ppp/8/2q5/3n4/2P2B1P/PPQ2PP1/R3R1K1 w - - 1 18",
         "c2d3", ["Ne2+", "Rxe2", "Rxd3", "Rxe8+", "Kd7", "Rae1"], "Hung Queen"),
        # PROMOTION RACE != hung material (Sam, 2026-07-13). Opponent's passer queens in the refutation;
        # the +8 swing is a LOST PAWN RACE (endgame technique), NOT a hung piece. Must NOT fire —
        # the pawn-endgame fragments own it. (Was mislabeled Hung Material / even Hung Queen, swallowing
        # ~17 passed-pawn SAE features.)
        ("promotion race != hung material", "8/1p6/8/8/8/1k6/1p6/1K6 w - - 0 1",
         "b1c1", ["b1q"], None),
        # CONTROL: opponent CAPTURES a real rook (no promotion) -> still Hung Rook (guard must not
        # over-suppress genuine captures).
        ("real rook capture still hangs", "6k1/8/8/8/8/8/r7/R3K3 w - - 0 1",
         "e1f1", ["Rxa1+"], "Hung Rook"),
        # SACRIFICE guard (Sam 2026-07-13): played move is a SEE<0 capture (Bxf7+ bishop-for-pawn), the
        # player CHOSE to shed material = unsound sacrifice, NOT a hang. Greek Gift; refutation Kxf7.
        # Must NOT fire Hung Material (unsound_sacrifice owns it).
        ("SEE<0 played capture = sacrifice, not hung",
         "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
         "c4f7", ["Kxf7"], None),
    ]
    for name, fen, uci, ref, want in hung_cases:
        b = chess.Board(fen)
        # the move-18 case supplies SAN refutation from the AFTER-played board; others too. eval None →
        # win_drop falls back to cp_loss (these test material, not the gate).
        ref_san = ref
        m = Mistake(fen, uci, "", [], ref_san, 0, -300, 300, b.turn)
        res = PR.hung_material(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")

    print("--- predicates: capture_or_exchange (equal-value gate + cp_loss MISS gate) ---")
    # (name, fen, played_uci, best_uci, best_san, cp_loss, expected_label_or_None)
    exch_cases = [
        # MISFIRE Sam caught (ply 50): best Qxe4+ = queen takes a DEFENDED bishop (9 for 3). That sheds
        # material — a sacrifice, not an even trade. Must NOT tag "Missed Bishop Exchange" (sacrifice_line
        # names it). attacker(9) > victim(3)+0.5 & e4 defended by Qd5.
        ("Q-for-defended-B is not an exchange", "r4r2/1b4pk/p1n4p/1p1Q4/4BN2/4q1P1/PP5P/R4R1K b - - 3 25",
         "h7h8", "e3e4", "Qxe4+", 200, None),
        # POSITIVE: a genuine equal trade the player missed (cp_loss 200) — must still tag.
        ("equal N-for-N (real miss, cp200) tags exchange", "3qk3/3p4/3p4/4n3/8/5N2/8/3QK3 w - - 0 1",
         "e1e2", "f3e5", "Nxe5", 200, "Missed Knight Exchange"),
        # GH #29: capture_or_exchange is now a PURE DETECTOR — it fires whenever best is an even trade,
        # regardless of severity. The played==best / low-win-drop suppression (the #27 fix) moved to the
        # ONE entry gate in tag_mistake_full (tested in the "entry gate" block below). So at the direct-
        # predicate level the cp20 case DOES fire; suppression is the gate's job, not the detector's.
        ("equal N-for-N detects regardless of cp (gate suppresses, not the detector)",
         "3qk3/3p4/3p4/4n3/8/5N2/8/3QK3 w - - 0 1", "e1e2", "f3e5", "Nxe5", 20, "Missed Knight Exchange"),
        # GH #28: bishop takes a DEFENDED knight (even minor trade) must be "Bishop-Knight Exchange",
        # NOT "Missed Knight Exchange" (the old victim-only naming mislabeled 64% of minor trades).
        ("B-for-N mixed minor trade -> Bishop-Knight Exchange", "4k3/8/3p4/4n3/8/8/1B6/4K3 w - - 0 1",
         "e1e2", "b2e5", "Bxe5", 200, "Missed Bishop-Knight Exchange"),
        # CONTROL: true B-for-B still names "Bishop Exchange". Bb2xBe5, e5 defended by d6 pawn.
        ("equal B-for-B still tags Bishop Exchange", "4k3/8/3p4/4b3/8/8/1B6/4K3 w - - 0 1",
         "e1e2", "b2e5", "Bxe5", 200, "Missed Bishop Exchange"),
    ]
    for name, fen, uci, best, bsan, cpl, want in exch_cases:
        b = chess.Board(fen)
        # eval_before/after=None -> win_drop uses its cp_loss fallback (these fixtures express
        # severity via cp_loss, not signed evals). cpl=200 -> 17.6 win-pts (tags); cpl=20 -> 1.8 (the
        # #27 played==best suppression). (GH #29 — the gate is win_drop now, not cp_loss.)
        m = Mistake(fen, uci, best, [], [], None, None, cpl, b.turn, best_san=bsan)
        res = PR.capture_or_exchange(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_exch = len(exch_cases)

    print("--- predicates: greedy_capture (played grabs material, best is QUIET) — GH #29 ---")
    # (name, fen, played_uci, best_uci, cpl, expected_label_or_None). win_drop via cp_loss fallback.
    greedy_cases = [
        # POS: Rd2xd5 grabs an UNDEFENDED pawn (SEE +1 — a real grab you keep); best is the quiet Kf1.
        # (Old FEN's Nxd5 grabbed a QUEEN-DEFENDED pawn = SEE -2, i.e. a losing sac, never real greed —
        #  #52 SEE gate correctly stopped firing on it; the test position was mislabeled. Replaced.)
        ("grab UNDEFENDED pawn when quiet move was best", "4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1",
         "d2d5", "e1f1", 200, "Greedy Capture"),
        # NEG: best is ALSO a capture (Bxd5, capturing the same pawn) — that's a missed capture/exchange
        # decision, not greed. Must NOT fire. (Bc4 attacks d5; Bxd5 is the alt capture.)
        ("best is also a capture -> not greedy", "4k3/8/8/3p4/8/8/3R2B1/4K3 w - - 0 1",
         "d2d5", "g2d5", 200, None),
        # GH #29: pure detector — fires on the grab-vs-quiet PATTERN regardless of severity.
        ("greedy grab detects regardless of cp (gate suppresses)", "4k3/8/8/3p4/8/8/3R4/4K3 w - - 0 1",
         "d2d5", "e1f1", 20, "Greedy Capture"),
        # NEG (#52): an UNSOUND SACRIFICE is not greed. Bxf7+ = bishop (3) takes a pawn (1) and is
        # immediately recaptured Kxf7 -> SEE ~-2. Greedy = grabbing material you keep; this SHEDS it.
        # Must NOT fire even though best (quiet Ng5) is non-capture. (The #45/#52 conflation.)
        ("Greek-Gift Bxf7+ sac is NOT greedy", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
         "c4f7", "g1g5", 260, None),
    ]
    for name, fen, uci, best, cpl, want in greedy_cases:
        b = chess.Board(fen)
        bsan = b.san(chess.Move.from_uci(best)); psan = b.san(chess.Move.from_uci(uci))
        m = Mistake(fen, uci, best, [], [], None, None, cpl, b.turn, played_san=psan, best_san=bsan)
        res = PR.greedy_capture(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_greedy = len(greedy_cases)

    print("--- predicates: unsound_sacrifice (played SEE<0 capture near enemy king) — #52b ---")
    # SAE-derived pattern: 97% of the 'unsound sac' cluster = a material-SHEDDING capture (SEE<0)
    # within 2 squares of the enemy king. The win_drop gate supplies 'unsound' (a sound sac isn't a
    # flagged mistake). (name, fen, played_uci, best_uci, expected)
    usac_cases = [
        # POS: Greek-Gift Bxf7+ — bishop for a pawn (SEE -2), lands next to Black's e8 king. Unsound sac.
        ("Greek-Gift Bxf7+ -> Unsound Sacrifice", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
         "c4f7", "g1g5", "Unsound Sacrifice"),
        # NEG: a SEE<0 capture NOT near the enemy king = a plain hung piece / bad trade, not a king sac.
        # Rd2xd5 into a queen-defended pawn far from the e8 king -> must NOT fire unsound_sacrifice.
        ("SEE<0 capture far from king is NOT unsound sac", "3qk3/8/8/3p4/8/8/3R4/4K3 w - - 0 1",
         "d2d5", "e1f1", None),
        # NEG: a real GRAB (SEE>=0) near the king is not a sac (you keep the material). Rxc7 takes an
        # undefended pawn 2 sq from the a8 king; the king can't recapture -> SEE +1, a grab not a sac.
        ("SEE>=0 capture near king is NOT a sac", "k7/2p5/8/8/8/8/2R5/4K3 w - - 0 1",
         "c2c7", "e1e2", None),
    ]
    for name, fen, uci, best, want in usac_cases:
        b = chess.Board(fen)
        psan = b.san(chess.Move.from_uci(uci)); bsan = b.san(chess.Move.from_uci(best))
        m = Mistake(fen, uci, best, [], [], None, None, 300, b.turn, played_san=psan, best_san=bsan)
        res = PR.unsound_sacrifice(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_usac = len(usac_cases)

    print("--- predicates: pointless_check (played = aimless check, best = quiet) — #47 ---")
    # SAE-derived (f4/f85/f121/f129/f499): the PLAYED move is a NON-CAPTURE check, the BEST move is
    # quiet, and it's a mistake. (name, fen, played_uci, best_uci, expected)
    pchk_cases = [
        # POS: Black king g8; White Qd1 -> Qd8+ is a non-capture check; best is the quiet h2-h3.
        ("Qd8+ non-capture check, best quiet -> Pointless Check",
         "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1", "d1d8", "h2h3", "Pointless Check"),
        # NEG: best move is ALSO a check (Re1-e8+) -> a check WAS called for, not a pointless one.
        ("best is also a check -> NOT pointless",
         "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "e1e8", None),
        # NEG: played check is a CAPTURE (Qxd8+ takes the rook) -> material decision, not a hope check.
        ("capturing check -> NOT pointless",
         "3r2k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1", "d1d8", "h2h3", None),
    ]
    for name, fen, uci, best, want in pchk_cases:
        b = chess.Board(fen)
        psan = b.san(chess.Move.from_uci(uci)); bsan = b.san(chess.Move.from_uci(best))
        m = Mistake(fen, uci, best, [], [], None, None, 300, b.turn, played_san=psan, best_san=bsan)
        res = PR.pointless_check(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_pchk = len(pchk_cases)

    print("--- predicates: missed_attacking_check (best is a forcing check you missed) — SAE jr2048 ---")
    # Mirror of pointless_check. (name, fen, played_uci, best_uci, expected)
    mac_cases = [
        # POS: f508 real board — best Qh5+ (non-capture check on the exposed e8 king), played Be2.
        ("Qh5+ attacking check = Missed Attacking Check",
         "r1bqkbnr/pppp2pp/8/4p3/4P3/8/PPP2PPP/RNBQKB1R w KQkq - 0 6", "f1e2", "d1h5", "Missed Attacking Check"),
        # POS: f2000 real board — best Qh4+, played Nf6.
        ("Qh4+ attacking check = Missed Attacking Check",
         "rnbqkbnr/ppp2ppp/3p4/4pP2/4P3/8/PPPP2PP/RNBQKBNR b KQkq - 0 3", "g8f6", "d8h4", "Missed Attacking Check"),
        # NEG: best is a QUIET move (not a check) -> not this tag.
        ("quiet best -> NOT attacking check",
         "6k1/8/8/8/8/8/8/Q5K1 w - - 0 1", "g1f1", "a1a4", None),
        # NEG: best check is MATE -> Missed Mate owns it, not this.
        ("best is mate -> NOT attacking check",
         "6rk/6pp/8/8/8/8/8/6KQ w - - 0 1", "g1f1", "h1h7", None),
    ]
    for name, fen, uci, best, want in mac_cases:
        b = chess.Board(fen)
        psan = b.san(chess.Move.from_uci(uci)); bsan = b.san(chess.Move.from_uci(best))
        m = Mistake(fen, uci, best, [], [], None, None, 300, b.turn, played_san=psan, best_san=bsan)
        res = PR.missed_attacking_check(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_mac = len(mac_cases)

    print("--- predicates: missed_overloading (must WIN material, not just geometry) — #57 masker ---")
    # Tightened from geometry-only (9.96% corpus, masked 11 features) to require the best LINE nets >=2.
    def _ln(fen, ucis):
        b = chess.Board(fen); out = []
        for u in ucis:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves: break
            out.append(b.san(mv)); b.push(mv)
        return out
    ovl_cases = [
        # POS: real corpus overload — Bg4 attacks the sole defender; the 6-ply line wins a piece (net>=2).
        ("Bg4 overload wins material = Missed Overloading",
         "r4rk1/ppqb1ppp/5n2/2Np4/3P4/2PB4/PP4Pb/R2Q1R1K b - - 1 18",
         "h2d6", ["d7g4","d3h7","g8h8","d1g4","f6g4","h7f5"], "Missed Overloading"),
        # NEG: same overload geometry but the line wins NOTHING (truncated, net 0) -> must NOT fire
        # (geometry alone was the 9.96% over-fire class).
        ("overload geometry but no material win -> no fire",
         "r4rk1/ppqb1ppp/5n2/2Np4/3P4/2PB4/PP4Pb/R2Q1R1K b - - 1 18",
         "h2d6", ["d7g4"], None),
    ]
    for name, fen, uci, pv, want in ovl_cases:
        b = chess.Board(fen)
        bl = _ln(fen, pv)
        m = Mistake(fen, uci, pv[0], bl, [], 300, -100, 0, b.turn,
                    played_san=b.san(chess.Move.from_uci(uci)), best_san=(bl[0] if bl else ""))
        res = PR.missed_overloading(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_ovl = len(ovl_cases)

    print("--- predicates: missed_zwischenzug (right capture, wrong order) — SAE jr2048 ---")
    # (name, fen, played_uci, pv_uci_line, expected). Detector reads best_line_san, so we build it.
    def _line(fen, ucis):
        b = chess.Board(fen); out = []
        for u in ucis:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves: break
            out.append(b.san(mv)); b.push(mv)
        return out
    zz_cases = [
        # POS: real corpus board. Played bxa6 immediately; best inserts Qxf2+ (check), Kxf2, THEN bxa6
        # (same a6 target at ply 3) = zwischenzug.
        ("bxa6 rushed; best Qxf2+ first -> Missed Zwischenzug",
         "2k5/1p5r/R3p2p/3p1p2/2pP2rq/2P1P3/5QPP/5RK1 b - - 0 31", "b7a6",
         ["h4f2", "g1f2", "b7a6", "f1a1"], "Missed Zwischenzug"),
        # NEG: best line's first move is NOT a check (quiet), even though target recurs -> not zwischenzug.
        ("no check inserted -> NOT zwischenzug",
         "2k5/1p5r/R3p2p/3p1p2/2pP2rq/2P1P3/5QPP/5RK1 b - - 0 31", "b7a6",
         ["h6h5", "g1h1", "b7a6", "f1a1"], None),
        # NEG: played move is NOT a capture -> detector requires a played capture.
        ("played non-capture -> NOT zwischenzug",
         "2k5/1p5r/R3p2p/3p1p2/2pP2rq/2P1P3/5QPP/5RK1 b - - 0 31", "c8b8",
         ["h4f2", "g1f2", "b7a6", "f1a1"], None),
    ]
    for name, fen, uci, pv, want in zz_cases:
        b = chess.Board(fen)
        bl = _line(fen, pv)
        m = Mistake(fen, uci, pv[0], bl, [], 300, -100, 0, b.turn,
                    played_san=b.san(chess.Move.from_uci(uci)), best_san=(bl[0] if bl else ""))
        res = PR.missed_zwischenzug(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_zz = len(zz_cases)

    print("--- predicates: missed_greek_gift (missed bishop sac on castled king) — SAE jr2048 ---")
    # (name, fen, played_uci, best_uci, expected)
    gg_cases = [
        # POS: f623 real board — Bxh7+ (bishop sac, SEE<0, next to g8 king), played O-O.
        ("Bxh7+ sac = Missed Greek Gift",
         "r1bq1rk1/pp1nnppp/2p1p3/b2pP3/3P1P2/2PB1N2/PP4PP/RNBQK2R w KQ - 5 9", "e1g1", "d3h7", "Missed Greek Gift"),
        # NEG: bishop capture-check that is NOT a sacrifice (takes a hanging piece, SEE>=0) -> not GG.
        ("bishop capture-check but not a sac -> NOT Greek Gift",
         "4k3/8/8/8/8/5n2/8/2B1K3 w - - 0 1", "e1e2", "c1f4", None),
        # NEG: best is a bishop sac-check NOT next to the king -> not GG geometry.
        ("bishop sac-check far from king -> NOT Greek Gift",
         "4k3/8/8/8/1b6/8/1p6/1B2K3 w - - 0 1", "e1e2", "b1e4", None),
    ]
    for name, fen, uci, best, want in gg_cases:
        b = chess.Board(fen)
        try:
            bsan = b.san(chess.Move.from_uci(best))
        except Exception:
            bsan = ""
        m = Mistake(fen, uci, best, [], [], 300, 0, 0, b.turn,
                    played_san=b.san(chess.Move.from_uci(uci)), best_san=bsan)
        res = PR.missed_greek_gift(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_gg = len(gg_cases)

    print("--- predicates: recapture_exposes_king (pawn recapture opens line to own king) — SAE jr2048 ---")
    # (name, fen, played_uci, best_uci, expected)
    rek_cases = [
        # POS: f24 real board — hxg4 (h3 shelter pawn captures g4, opening the h-file to Kg1); best is
        # the quiet d4. Castled king g1.
        ("hxg4 opens h-file to Kg1 = Recapture Exposed King",
         "r1bqkb1r/ppp2pp1/2np4/1B2p2p/4P1n1/2N2N1P/PPPP1PP1/R1BQ1RK1 w kq - 0 7", "h3g4", "d2d4",
         "Recapture Exposed King"),
        # NEG: the best move IS the recapture (then recapturing was correct, not the error).
        ("best is the recapture -> no fire",
         "r1bqkb1r/ppp2pp1/2np4/1B2p2p/4P1n1/2N2N1P/PPPP1PP1/R1BQ1RK1 w kq - 0 7", "h3g4", "h3g4", None),
        # NEG: uncastled king -> no castled shelter to open.
        ("uncastled king -> no fire",
         "r3k2r/ppp2pp1/2np4/4p2p/4P1n1/2N2N1P/PPPP1PP1/R1BQKB1R w KQkq - 0 7", "h3g4", "d2d4", None),
    ]
    for name, fen, uci, best, want in rek_cases:
        b = chess.Board(fen)
        try:
            bsan = b.san(chess.Move.from_uci(best))
        except Exception:
            bsan = ""
        m = Mistake(fen, uci, best, [], [], 300, 0, 0, b.turn,
                    played_san=b.san(chess.Move.from_uci(uci)), best_san=bsan)
        res = PR.recapture_exposes_king(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_rek = len(rek_cases)

    print("--- predicates: exposed_king_pawn (castled king shelter push only) — #50 ---")
    # Tightened from 'any pawn near king' (9.8% corpus) to castled + non-capture + shelter-pawn advance
    # (3.8%). (name, fen, played_uci, expected)
    ekp_cases = [
        # POS: castled Kg1, g2-g4 pushes a shelter pawn forward -> weakens the shelter.
        ("castled Kg1 g2-g4 = Pawn Move Exposed King",
         "6k1/pppp1ppp/8/8/8/8/PPPP1PPP/5RK1 w - - 0 1", "g2g4", "Pawn Move Exposed King"),
        # NEG: uncastled king in the center -> no shelter to expose.
        ("uncastled center king -> no fire",
         "r3k2r/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2R w KQkq - 0 1", "f2f4", None),
        # NEG: capture near the castled king -> material decision, not a shelter push.
        ("capture near castled king -> no fire",
         "6k1/ppp2ppp/8/3p4/4P3/8/PPP2PPP/5RK1 w - - 0 1", "e4d5", None),
        # NEG: pawn far from the king (a-file, king on g1) -> not shelter.
        ("far pawn -> no fire",
         "6k1/pppp1ppp/8/8/8/P7/1PPP1PPP/5RK1 w - - 0 1", "a3a4", None),
    ]
    for name, fen, uci, want in ekp_cases:
        b = chess.Board(fen)
        m = Mistake(fen, uci, "", [], [], None, None, 0, b.turn, played_san=b.san(chess.Move.from_uci(uci)))
        res = PR.exposed_king_pawn(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_ekp = len(ekp_cases)

    print("--- predicates: pin exploitation / unpinning resource (book TOP-TIER, w5zuk548s) ---")
    # missed_pin_exploitation: an enemy piece is pinned + HELD, best quietly piles a NEW attacker on it.
    # missed_unpinning_resource: OUR piece is pinned, best breaks the pin, played sits in it.
    pinx_cases = [
        # POS pile-on: Black Na5 pinned to Ra8 by Ra1, defended by b6 pawn (held). Best b2-b4 adds a
        # pawn attacker on a5; played is a quiet king move. Fires Missed Pin Exploitation.
        ("pile a pawn onto a held pinned knight", "r7/8/1p6/n7/8/8/1P6/R3K1k1 w - - 0 1",
         "e1e2", "b2b4", PR.missed_pin_exploitation, "Missed Pin Exploitation"),
        # NEG: best move IS a capture of the pinned piece — that's Missed Pin/material, not the quiet
        # pile-on prep. Must NOT fire pin-exploitation.
        ("capturing the pinned piece is not pile-on", "r7/8/1p6/n7/8/8/1P6/R3K1k1 w - - 0 1",
         "e1e2", "a1a5", PR.missed_pin_exploitation, None),
        # POS unpin: Black Ne7 pinned to Ke8 by Re1 (absolute). Best Kf8 breaks it; played h6 sits in it.
        ("king-step unpins; played sits in the pin", "4k3/4n2p/8/8/8/8/8/4R1K1 b - - 0 1",
         "h7h6", "e8f8", PR.missed_unpinning_resource, "Missed Unpinning Resource"),
        # NEG: no pin on the board at all -> unpinning can't fire.
        ("no pin -> no unpinning resource", "4k3/4n2p/8/8/8/8/8/6K1 b - - 0 1",
         "h7h6", "e7c6", PR.missed_unpinning_resource, None),
        # POS interposition: Black Ke8 in check from White Bb5 (b5-c6-d7-e8 diag). Best Nb8-c6 blocks;
        # played Kf8 runs. Fires Missed Interposition.
        ("block the check vs running the king", "1n2k3/8/8/1B6/8/8/8/4K3 b - - 0 1",
         "e8f8", "b8c6", PR.missed_interposition, "Missed Interposition"),
        # NEG: knight check can't be blocked -> no interposition (here just: not in check).
        ("not in check -> no interposition", "1n2k3/8/8/8/8/8/8/4K3 b - - 0 1",
         "e8f8", "b8c6", PR.missed_interposition, None),
        # POS remove-the-guard: Black Kg8 castled, Nf6 guards king-ring (g8/h7). White Bg5xf6 (even
        # trade, f6 defended by g7) strips the guard; played a3 doesn't. Fires Missed Remove the Guard.
        ("trade off the defender of the castled king", "r2q1rk1/ppp2ppp/5n2/6B1/8/8/PPP2PPP/R2Q1RK1 w - - 0 1",
         "a2a3", "g5f6", PR.missed_remove_the_guard, "Missed Remove the Guard"),
        # NEG: a capture far from the enemy king strips no king-guard -> must NOT fire.
        ("capture away from king is not remove-the-guard", "r2q1rk1/ppp2ppp/8/3n4/3N4/8/PPP2PPP/R2Q1RK1 w - - 0 1",
         "a2a3", "d4d5", PR.missed_remove_the_guard, None),
    ]
    for name, fen, uci, best, fn, want in pinx_cases:
        b = chess.Board(fen)
        try: bsan = b.san(chess.Move.from_uci(best))
        except Exception: bsan = best
        try: psan = b.san(chess.Move.from_uci(uci))
        except Exception: psan = uci
        m = Mistake(fen, uci, best, [bsan], [], None, None, 200, b.turn, played_san=psan, best_san=bsan)
        res = fn(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_pinx = len(pinx_cases)

    print("--- tagger ENTRY GATE: win%-drop decides mistake-vs-not, ONCE (GH #29) ---")
    # The single mistake gate lives in tag_mistake_full, not in the detectors. Verify: (1) a real miss
    # (high win_drop) surfaces the explain tag; (2) the SAME position played==best (win_drop~0) suppresses
    # ALL explain tags — this is the #27 suppression, now centralized; (3) INFO/orient tags (phase,
    # game-state) survive the gate either way (they classify the position, not assert a mistake).
    import tagger as TG_GATE
    gate_fen = "3qk3/3p4/3p4/4n3/8/5N2/8/3QK3 w - - 0 1"  # best Nxe5 = even knight trade
    b = chess.Board(gate_fen)
    # mover WHITE, eval_before 0; a real miss drops to -200 after the played move -> win_drop ~17.6
    m_miss = Mistake(gate_fen, "e1e2", "f3e5", [], [], 0, -200, 200, b.turn, best_san="Nxe5")
    # played==best equivalent: eval barely moves (0 -> -10) -> win_drop ~0.9, under the 10.0 gate
    m_nonmiss = Mistake(gate_fen, "e1e2", "f3e5", [], [], 0, -10, 10, b.turn, best_san="Nxe5")
    miss_labels = [t["label"] for t in TG_GATE.tag_mistake_full(m_miss, with_maia=False)["tags"]]
    non_labels = [t["label"] for t in TG_GATE.tag_mistake_full(m_nonmiss, with_maia=False)["tags"]]
    gate_cases = [
        ("real miss surfaces Missed Knight Exchange", "Missed Knight Exchange" in miss_labels, True),
        ("played==best suppresses the explain tag", "Missed Knight Exchange" in non_labels, False),
        ("info/orient tag (game-state) survives the gate", "Equal" in non_labels, True),
    ]
    for name, got, want in gate_cases:
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got} exp={want}")
    extra_gate = len(gate_cases)

    print("--- tagger ENTRY GATE: explicit classification overrides win_drop (inaccuracy display) ---")
    # Product need: the frontend surfaces inaccuracies (5-10 win% drop) on review cards and wants them
    # EXPLAINED, but they must NOT count toward stats/drills (filtered downstream by classification).
    # So tag_mistake_full accepts an optional `classification`: when given, it — not win_drop — decides
    # whether explain tags fire (inaccuracy/mistake/blunder = yes; anything else = no). When absent
    # (research corpus, which never passes it), the win_drop>=10 fallback stands = mistake/blunder only.
    import tagger as TG_CLS
    cls_fen = "3qk3/3p4/3p4/4n3/8/5N2/8/3QK3 w - - 0 1"
    bc = chess.Board(cls_fen)
    # An INACCURACY-magnitude drop (0 -> -80 => win_drop ~7, UNDER the 10.0 win_drop gate).
    m_inacc = Mistake(cls_fen, "e1e2", "f3e5", [], [], 0, -80, 80, bc.turn, best_san="Nxe5")
    inacc_default = [t["label"] for t in TG_CLS.tag_mistake_full(m_inacc, with_maia=False)["tags"]]
    inacc_tagged = [t["label"] for t in TG_CLS.tag_mistake_full(m_inacc, with_maia=False, classification="inaccuracy")["tags"]]
    good_ignored = [t["label"] for t in TG_CLS.tag_mistake_full(m_inacc, with_maia=False, classification="good")["tags"]]
    # A mistake-magnitude position passed classification="good" (contrived) must still be gated OUT —
    # explicit classification wins over win_drop when supplied.
    m_realmiss = Mistake(cls_fen, "e1e2", "f3e5", [], [], 0, -200, 200, bc.turn, best_san="Nxe5")
    realmiss_forced_good = [t["label"] for t in TG_CLS.tag_mistake_full(m_realmiss, with_maia=False, classification="good")["tags"]]
    cls_cases = [
        ("inaccuracy: win_drop<10 gives NO explain tag by default", "Missed Knight Exchange" in inacc_default, False),
        ("inaccuracy: classification='inaccuracy' surfaces the explain tag", "Missed Knight Exchange" in inacc_tagged, True),
        ("classification='good' gates out an inaccuracy-magnitude move", "Missed Knight Exchange" in good_ignored, False),
        ("classification='good' overrides a real win_drop (explicit wins)", "Missed Knight Exchange" in realmiss_forced_good, False),
        ("info/orient tag survives regardless of classification", "Equal" in good_ignored, True),
    ]
    for name, got, want in cls_cases:
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got} exp={want}")
    extra_cls = len(cls_cases)

    print("--- tagger: GOOD moves + played==best earn NO explain tags (even in mating positions) ---")
    # Bugs from a real game review (2026-07-11): the tagger emitted mistake tags on GOOD moves.
    #  ply50 Qh2# (played==best==the mate) got "Missed Mate" — you can't miss a mate you just played.
    #  ply48 Re2 (brilliant, keeps a forced mate) got Missed Mate + Allowed Sacrifice — the MATE
    #        EXEMPTION punched through the classification gate for a good move.
    # Rule: a good classification (brilliant/great/excellent/good/opening) OR played==best suppresses
    # ALL explain tags — including the mate exemption. Info/orient tags still fire.
    import tagger as TG_GM
    # A real forced-mate-available position (mover WHITE has mate; eval_before = +sentinel).
    gm_fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    SENT = TG_GM._MATE_SENTINEL
    # explain (mistake) tags = anything NOT direction=="info" (info/orient tags always fire + are
    # stripped from chips downstream; the bug is about EXPLAIN tags on good moves).
    def _explain(res): return [t["label"] for t in res["tags"] if t["direction"] != "info"]
    # (a) played==best: the played move IS the best move -> nothing missed/allowed.
    m_pb = Mistake(gm_fen, "a1a8", "a1a8", [], [], SENT, None, 0, chess.WHITE, best_san="Ra8#")
    pb_explain = _explain(TG_GM.tag_mistake_full(m_pb, with_maia=False, classification="brilliant"))
    # (b) good move (brilliant) in a mating position, played != best -> still no explain tags.
    m_good = Mistake(gm_fen, "g1h1", "a1a8", [], [], SENT, None, 0, chess.WHITE, best_san="Ra8#")
    good_explain = _explain(TG_GM.tag_mistake_full(m_good, with_maia=False, classification="brilliant"))
    # (c) control: an actual BLUNDER that misses the mate -> Missed Mate SHOULD still fire.
    m_miss_mate = Mistake(gm_fen, "g1h1", "a1a8", [], [], SENT, None, 0, chess.WHITE, best_san="Ra8#")
    missmate_labels = [t["label"] for t in TG_GM.tag_mistake_full(m_miss_mate, with_maia=False, classification="blunder")["tags"]]
    gm_cases = [
        ("played==best (brilliant): no Missed Mate", "Missed Mate" in pb_explain, False),
        ("played==best (brilliant): no explain tags at all", len(pb_explain) > 0, False),
        ("good move in mating pos: no explain tags", len(good_explain) > 0, False),
        ("BLUNDER that misses mate: Missed Mate still fires", "Missed Mate" in missmate_labels, True),
    ]
    # missed+allowed twin collapse: a single move must not carry both "Missed X" and "Allowed X".
    # (Fixture uses Doubled Rooks — any Missed/Allowed twin works; Battery was removed 2026-07-14.)
    twin_in = [("Missed Doubled Rooks","missed","e","tactic"), ("Allowed Doubled Rooks","allowed","e","tactic"),
               ("Missed Fork","missed","e","tactic")]
    twin_out = [t[0] for t in TG_GM._collapse_missed_allowed_twins(twin_in)]
    gm_cases += [
        ("twin collapse: Allowed Doubled Rooks dropped when Missed twin present", "Allowed Doubled Rooks" in twin_out, False),
        ("twin collapse: Missed Doubled Rooks kept", "Missed Doubled Rooks" in twin_out, True),
        ("twin collapse: unrelated Missed Fork untouched", "Missed Fork" in twin_out, True),
    ]
    # parent->child suppression (_PARENT_CHILD table): a generic parent tag is dropped when its more
    # specific child also fired. "Allowed Hanging Piece" (parent) dropped when any "Hung X" (child) is
    # present; kept when it's the lone material signal.
    ph_with_hung = [t[0] for t in TG_GM._suppress_parents(
        [("Hung Rook", "hung", "e", "position"), ("Allowed Hanging Piece", "allowed", "e", "tactic"),
         ("Allowed Skewer", "allowed", "e", "tactic")])]
    ph_alone = [t[0] for t in TG_GM._suppress_parents(
        [("Allowed Hanging Piece", "allowed", "e", "tactic"), ("Missed Free Pawn", "missed", "e", "position")])]
    gm_cases += [
        ("parent->child: Allowed Hanging Piece dropped when Hung Rook present", "Allowed Hanging Piece" in ph_with_hung, False),
        ("parent->child: Hung Rook (child) kept", "Hung Rook" in ph_with_hung, True),
        ("parent->child: unrelated Allowed Skewer untouched", "Allowed Skewer" in ph_with_hung, True),
        ("parent->child: parent kept when no child present", "Allowed Hanging Piece" in ph_alone, True),
    ]
    for name, got, want in gm_cases:
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got} exp={want}")
    extra_gm = len(gm_cases)

    print("--- predicates: pawn break / prophylaxis must not fire on material grabs (GH #28-class) ---")
    # (name, fn, fen, played_uci, best_uci, best_san, refutation_san, cp_loss, expected_label_or_None)
    grab_cases = [
        # Pawn Break NEG: best exd5 captures a BISHOP — material grab, not a structural break. Must NOT fire.
        ("pawn break NEG: pawn takes piece", PR.missed_pawn_break,
         "4k3/8/8/3b4/4P3/8/8/4K3 w - - 0 1", "e1e2", "e4d5", "exd5", [], 300, None),
        # Pawn Break POS: best exd5 captures a PAWN — a real break. Must still fire.
        ("pawn break POS: pawn takes pawn", PR.missed_pawn_break,
         "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e1e2", "e4d5", "exd5", [], 120, "Missed Pawn Break"),
        # Prophylaxis NEG: best move is a capture (Bxd5) — winning material, not quiet prevention. Must NOT fire.
        ("prophylaxis NEG: best is a capture", PR.missed_prophylaxis,
         "r3k3/8/8/3n4/8/3B4/8/4K3 w - - 0 1", "e1e2", "d3d5", "Bxd5", ["d5e3"], 200, None),
        # Open File is a QUIET positional lesson (2026-07-14 audit gate). POS: best Rd1 quietly occupies the
        # half-open d-file (played Re1). NEG-capture: best Rxe4 wins material — a tactic, file incidental.
        # NEG-check: best Rd5+ is a forcing check — the lesson is the check, not the file (endgames are
        # full of open files). Corpus FENs.
        ("open file POS: quiet rook to half-open file", PR.missed_open_file,
         "1k6/1p3ppp/p3p3/3pP3/5P1P/8/2r4r/K6R w - - 3 33", "h1e1", "h1d1", "Rd1", [], 150, "Missed Open File"),
        ("open file NEG: best is a rook capture", PR.missed_open_file,
         "5r1k/p3q2p/3p4/3Q2PR/1R2p3/N2P4/2P2P2/4K3 w - - 1 27", "d3e4", "b4e4", "Rxe4", [], 150, None),
        ("open file NEG: best is a rook check", PR.missed_open_file,
         "2r5/p2k2pp/5p2/1PR5/4PK2/7P/8/8 w - - 1 32", "c5c8", "c5d5", "Rd5+", [], 150, None),
        # Missed Battery — rebuilt 2026-07-14 (Sam's Bc5+Qb6-on-f2 example). A battery = two sliders stacked
        # on ONE line, front directly attacks a DEFENDED target, back xrays through it. POS1: best Qb6 stacks
        # behind Bc5 on the b6-f2 diagonal (bears through f2 onto the defended g1 knight). POS2: best Qd5
        # stacks behind Bc4 on the c4-f7 diagonal (defended f7 pawn). NEG-capture: best Rxe4 is a material
        # tactic. NEG-lone-rook: quiet rook to an open file but NO back piece stacked = not a battery.
        ("battery POS1: Qb6 stacks behind Bc5 on f2/g1", PR.missed_battery,
         "rnbqk1nr/pp3ppp/2p5/2bpP3/4P3/3B1P2/PPP3PP/RNBQK1NR b KQkq - 2 5", "a7a6", "d8b6", "Qb6", [], 150, "Missed Battery"),
        ("battery POS2: Qd5 stacks behind Bc4 on f7", PR.missed_battery,
         "r1bqkb1r/pppp1pp1/2n4p/4P3/2B1n3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 6", "a2a3", "d1d5", "Qd5", [], 150, "Missed Battery"),
        ("battery NEG: best move is a capture", PR.missed_battery,
         "5r1k/p3q2p/3p4/3Q2PR/1R2p3/N2P4/2P2P2/4K3 w - - 1 27", "d3e4", "b4e4", "Rxe4", [], 150, None),
        ("battery NEG: lone rook to open file (no stacked back piece)", PR.missed_battery,
         "1k6/1p3ppp/p3p3/3pP3/5P1P/8/2r4r/K6R w - - 3 33", "h1e1", "h1d1", "Rd1", [], 150, None),
        # capture_or_exchange SEE gate (2026-07-14): a DEFENDED capture that WINS material over the
        # recapture sequence must fire "Missed Free X", even when the attacker outvalues the victim.
        # POS: Qxf4 wins a defended bishop, SEE +3 (recapturer is re-won) — the old piece-value gate wrongly
        # called it a sacrifice and stayed silent (95-position gap). NEG: Qxd5 = queen for a defended
        # bishop, SEE -6 = a real sacrifice, must STAY excluded (the case the gate originally protected).
        ("cap SEE POS: defended capture that nets material fires Missed Free", PR.capture_or_exchange,
         "r5k1/p6p/2p2br1/2P5/5BqP/P2P1pP1/1P3P2/R2Q2KR b - - 0 28", "a7a6", "g4f4", "Qxf4", [], 150, "Missed Free Bishop"),
        ("cap SEE NEG: queen for defended bishop (SEE<0) stays silent", PR.capture_or_exchange,
         "4k3/8/2p5/3b4/8/8/8/3QK3 w - - 0 1", "e1e2", "d1d5", "Qxd5", [], 150, None),
        # Wrong Check (2026-07-14): player DID check, but a DIFFERENT check was stronger. Third case of the
        # check family (pointless=checked-should-have-been-quiet; missed_attacking=should-have-checked).
        # POS: real corpus positions where both best and played give check on different squares. NEG-mate:
        # best check IS mate (Missed Mate owns it). NEG-quiet: played move isn't a check (that's
        # missed_attacking_check, not this).
        ("wrong check POS: Re8+ was stronger than played Ra8+", PR.wrong_check,
         "6k1/R4p2/p5p1/1p3bP1/5P2/P2p4/1Pr5/3KR3 w - - 0 33", "a7a8", "e1e8", "Re8+", [], 150, "Wrong Check"),
        ("wrong check POS: Rh7+ was stronger than played Qg7+", PR.wrong_check,
         "8/ppk5/2p4R/3n2p1/3Qp3/3n2P1/4K1P1/1q6 w - - 4 43", "d4g7", "h6h7", "Rh7+", [], 150, "Wrong Check"),
        ("wrong check NEG: best check delivers mate (Missed Mate owns it)", PR.wrong_check,
         "6k1/5ppp/8/8/8/8/7R/R5K1 w - - 0 1", "h2h8", "a1a8", "Ra8#", [], 150, None),
        ("wrong check NEG: played move is not a check", PR.wrong_check,
         "6k1/R4p2/p5p1/1p3bP1/5P2/P2p4/1Pr5/3KR3 w - - 0 33", "d1d2", "e1e8", "Re8+", [], 150, None),
    ]
    for name, fn, fen, uci, best, bsan, ref, cpl, want in grab_cases:
        b = chess.Board(fen)
        # evals None -> win_drop falls back to cp_loss (these test structure, not eval nuance). GH #29.
        m = Mistake(fen, uci, best, [], ref, None, None, cpl, b.turn, best_san=bsan)
        res = fn(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_grab = len(grab_cases)

    # Castling motif gate (2026-07-14): Missed Castling fires ONLY when the best MOVE is castling (first
    # pov move of the best line), not when castling appears anywhere later. Allowed Castling is deleted
    # (no teachable "you let them castle" lesson). Uses the full tagger (motif system), not a predicate.
    print("--- motifs: castling gated to first-move-is-castle; no Allowed Castling ---")
    castle_cases = [
        # POS: best move IS O-O -> Missed Castling.
        ("castling POS: best move is O-O", "r2qr1k1/pp2bppp/5n2/4pb2/2Pp4/PP1P1N2/1BQ1BPPP/R3K2R w KQ - 4 14",
         "e1e2", ["O-O", "Bf8", "b4"], [], "Missed Castling", True),
        # NEG: best is Bxf3, castling only at ply 4 -> NO Missed Castling.
        ("castling NEG: castle only later in line", "rn1qk1nr/p4p1p/1ppp4/3bp1p1/2P3Pb/3P1N2/PP1BPPB1/RN2QRK1 b kq - 1 13",
         "a7a6", ["Bxf3", "exf3", "Ne7", "Nc3", "O-O"], [], "Missed Castling", False),
        # NEG: opponent castles in the refutation -> Allowed Castling must NOT exist.
        ("castling NEG: no Allowed Castling twin", "1k1r3r/pppq1ppp/2nbb3/4p3/P3P3/1P3PP1/1BP1N1BP/R2QK2R b KQ - 0 13",
         "d7h3", ["Bb4+"], ["O-O", "Be6", "Qe1", "h5"], "Allowed Castling", False),
        # Outpost gate (2026-07-14): Missed Outpost fires only when the best MOVE quietly establishes the
        # outpost (first pov move, non-capture), not when an outpost appears later or via a capture.
        # POS: best Ne5 is a quiet outpost move (d4 defends e5, no black pawn can challenge). NEG: best is
        # a capture that lands on an outpost square -> material move, outpost incidental.
        ("outpost POS: best move is a quiet Ne5 outpost", "r1bqk2r/pp1n1ppp/2p1pn2/8/2BP4/2N2N2/PP3PPP/R1BQK2R w KQkq - 0 8",
         "d1e2", ["Ne5"], [], "Missed Outpost", True),
        ("outpost NEG: outpost only later in line (capture first)", "2rqkb1r/3b1ppp/p3pn2/1p6/2nP4/2NBPN2/PP1B1PPP/R2QK2R w KQk - 4 15",
         "a2a3", ["Bxc4", "bxc4", "Ne5"], [], "Missed Outpost", False),
    ]
    for name, fen, uci, bl_san, rf_san, label, should_fire in castle_cases:
        b = chess.Board(fen)
        m = Mistake(fen, uci, None, bl_san, rf_san, None, None, 150, b.turn)
        labels = [t["label"] for t in TG_GM.tag_mistake_full(m, with_maia=False)["tags"]]
        got = label in labels
        passed = (got == should_fire)
        ok += passed
        if not passed:
            fails.append(name)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: '{label}' present={got} exp={should_fire}")
    extra_castle = len(castle_cases)

    # Missed Stalemate (2026-07-14): from a LOSING position, best move forces stalemate = a draw save.
    # Synthetic anchor — board is K+Q vs lone K (Qg6 mechanically stalemates the boxed king); eval_before
    # is set to exercise the losing-side gate (a genuine losing-side-forces-stalemate FEN is hard to
    # hand-build + ~absent from the 60k middlegame corpus, so ~0 real fires — built for correctness).
    print("--- predicates: missed_stalemate (losing + best move forces stalemate) ---")
    sm_fen = "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"
    sm_cases = [
        # POS: mover losing (eval_before -600, mover=White) + Qg6 forces stalemate -> fires.
        ("stalemate POS: losing, Qg6 forces stalemate", -600, "Missed Stalemate"),
        # NEG: mover winning (eval_before +900) -> stalemate is a blunder, not a save -> silent.
        ("stalemate NEG: winning mover, not a save", 900, None),
    ]
    for name, eb, want in sm_cases:
        m = Mistake(sm_fen, "g5h5", "g5g6", [], [], eb, 0, 600, chess.WHITE, best_san="Qg6", played_san="Qh5")
        res = PR.missed_stalemate(m)
        got = res[0][0] if res else None
        passed = (got == want)
        ok += passed
        if not passed:
            fails.append(name)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: got={got!r} exp={want!r}")
    extra_sm = len(sm_cases)

    print("--- predicates: endgame detectors (king activity / opposition / passed pawn / rook-behind) ---")
    # (name, fen, played_uci, best_uci, best_san, predicate_fn, expected_label_or_None)
    # eval_before set so phase()/game_state don't matter; cp_loss=200. mover = side to move in the FEN.
    eg_cases = [
        # --- Missed King Activity: best is a non-check king move toward center/pawns, played wasn't ---
        # corpus: played Ng2, best Kf3 (king toward center). 8 pieces -> endgame.
        ("king activity: best Kf3 not Ng2", "8/8/6p1/5pkp/8/4N3/4K3/8 w - - 0 51",
         "e3g2", "e2f3", "Kf3", "missed_king_activity", "Missed King Activity"),
        # corpus: played Kb5, best Kd4 (toward center).
        ("king activity: best Kd4 not Kb5", "8/8/3k2pp/p2P1p2/2K2P1P/1p3P2/1P6/8 w - - 0 35",
         "c4b5", "c4d4", "Kd4", "missed_king_activity", "Missed King Activity"),
        # NEG: best king move AWAY from center (Kf3->Kg2, toward corner) must NOT fire. Mirror the
        # first case: from a position where the best is the passive king step.
        ("king activity NEG: best heads to corner", "8/8/6p1/5pkp/8/4NK2/8/8 b - - 0 51",
         "g5g4", "g5h4", "Kh4", "missed_king_activity", None),
        # NEG: not an endgame (full board) — king activity should not fire even if best is a king move.
        ("king activity NEG: not endgame", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
         "e7e5", "e8e7", "Ke7", "missed_king_activity", None),

        # --- Lost the Opposition: pawn-only endgame, best king move takes direct opposition ---
        # corpus: played Kf4, best Kd3 takes opposition vs Kd5 (dist 2, same file d). pawn-only.
        ("opposition: best Kd3 takes it", "8/8/8/3k1p1p/3P2pP/4K1P1/8/8 w - - 2 53",
         "e3f4", "e3d3", "Kd3", "lost_opposition", "Lost the Opposition"),
        # NEG: same geometry but NOT a pawn-only endgame (add a knight) -> opposition concept N/A.
        ("opposition NEG: not pawn-only", "8/8/8/3k1p1p/3P2pP/4K1P1/6n1/8 w - - 2 53",
         "e3f4", "e3d3", "Kd3", "lost_opposition", None),
        # DIAGONAL opposition (#50): corpus board, Black Kc4 vs White Kd1. best Kb3 -> b3/d1 = file&rank
        # diff 2 = diagonal opposition; played b5 (a pawn move) doesn't. pawn-only. Old code missed this.
        ("opposition: diagonal Kb3 takes it", "8/8/1p6/8/2k5/8/8/3K4 b - - 3 62",
         "b6b5", "c4b3", "Kb3", "lost_opposition", "Lost the Opposition"),
        # DIAGONAL opposition, second corpus board: White Kc4 vs Black Ke6, best Kd6? no — verified
        # geometry: 8/6p1/2p1k3/2P2p1p/1K3P1P/6P1/8/8 w, best Kc4->? use the Ka5->Kc4 case (Kc4 vs e6).
        ("opposition: diagonal Kc4 takes it", "8/6p1/2p1k3/2P2p1p/1K3P1P/6P1/8/8 w - - 4 38",
         "b4a5", "b4c4", "Kc4", "lost_opposition", "Lost the Opposition"),
        # NEG: played move ALSO takes opposition (both king moves reach it) -> not a "lost" opposition.
        # Direct case, but played Kd3 and best Kf3 BOTH sit dist-2 same-rank from an e5 enemy king.
        ("opposition NEG: played also holds it", "8/8/8/4k3/8/4K3/8/8 w - - 0 1",
         "e3d3", "e3f3", "Kf3", "lost_opposition", None),

        # --- Missed Passed Pawn: best is a pawn move creating/advancing a passer, played wasn't ---
        # corpus: played Kh4, best h4 makes the h-pawn passed. endgame.
        ("passed pawn: best h4 not Kh4", "8/8/4p3/5k1K/p1pP1P2/P6P/1P6/8 w - - 1 38",
         "h5h4", "h3h4", "h4", "missed_passed_pawn", "Missed Passed Pawn"),
        # NEG: best pawn move that is NOT passed (enemy pawn still blocks on adjacent file).
        ("passed pawn NEG: still blocked", "8/2p5/8/2P5/8/5k2/5p2/5K2 w - - 0 1",
         "f1f2", "c5c6", "c6", "missed_passed_pawn", None),

        # --- Rook Behind Passer: best puts a rook behind a passed pawn, played didn't ---
        # corpus: played Rg8, best Rd8 (rook behind the d5 passer). endgame.
        ("rook behind passer: best Rd8", "7R/8/p1r4p/3pp2P/8/4KP2/k7/8 w - - 0 44",
         "h8g8", "h8d8", "Rd8", "rook_behind_passer", "Rook Behind Passer"),
        # NEG: rook move to a file whose pawns are BLOCKED (c4/c5 mutually block -> neither passed),
        # so "behind a passer" geometry is absent. Must not fire.
        ("rook behind NEG: no passer on file", "7R/8/8/2p5/2P5/4KP2/k7/8 w - - 0 44",
         "h8g8", "h8c8", "Rc8", "rook_behind_passer", None),

        # --- Trade to Simplify: best is an EVEN trade (SEE~0), NOT a free/winning grab (#52) ---
        # POS: Rxd5 is R-for-R, Black Kd6 recaptures (SEE 0) -> a real simplifying trade. played Ke2.
        ("trade: even Rxd5 = Missed Trade to Simplify", "8/8/3k4/3r4/8/8/8/3RK3 w - - 0 1",
         "e1e2", "d1d5", "Rxd5", "trade_to_simplify", "Missed Trade to Simplify"),
        # NEG (#52 bug): best Rxd5 grabs an UNDEFENDED rook (SEE +5) -> that's WINNING material
        # (Missed Free Rook), not a trade. trade_to_simplify must NOT fire (capture_or_exchange owns it).
        ("trade NEG: free rook grab is not a trade", "8/8/8/3r4/8/8/8/3RK1k1 w - - 0 1",
         "e1e2", "d1d5", "Rxd5", "trade_to_simplify", None),
    ]
    for name, fen, uci, best, bsan, fn_name, want in eg_cases:
        b = chess.Board(fen)
        m = Mistake(fen, uci, best, [], [], 200, 0, 0, b.turn, best_san=bsan)
        got_list = getattr(PR, fn_name)(m)
        got = got_list[0][0] if got_list else None
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_eg = len(eg_cases)

    print("--- predicates: pawn structure (recapture + best-line guards) ---")
    # (name, fen, played_uci, best_uci, refutation_san, label_must_NOT_appear)
    ps_cases = [
        # 13.gxf3 — a recapture that doubles the f-pawn, but it IS the best move (recapturing the
        # bishop). best==played -> the doubling is in best_af too -> not blunder-caused. Must NOT tag
        # Created Doubled Pawn. (Caught by Sam.)
        ("recapture doesn't create doubled", "r2qr1k1/pp3ppp/2nb1n2/1Bpp4/8/P1NP1b2/1PPQ1PPP/R1B1R1K1 w - - 0 13",
         "g2f3", "g2f3", "Created Doubled Pawn"),
    ]
    for name, fen, uci, best, must_not in ps_cases:
        b = chess.Board(fen)
        m = Mistake(fen, uci, best, [], [], 0, -200, 200, b.turn)
        labels = [t[0] for t in PR.pawn_structure(m)]
        passed = must_not not in labels
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: labels={labels} (must not contain {must_not!r})")

    print("--- tag_game: deep-best + played==best + only-move (deep-analysis aware) ---")
    import tag_game as TG
    # deep-best: best comes from top_lines[0], NOT the stale best_san field. 40.Bd2: best_san='Bd2'
    # (== played) but deep PV top_lines[0]='cxb4'. Must resolve best to cxb4, NOT played.
    e_stale = {"fen": "8/4b3/8/3p1k1K/1ppPp2P/2P1B3/1P6/8 w - - 0 40", "uci": "e3d2", "best_san": "Bd2",
               "top_lines": [{"eval": "-428", "moves": ["cxb4", "Bxb4", "Kh6"]}]}
    bu, _ = TG._real_best(e_stale, chess.Board(e_stale["fen"]))
    deep_ok = (bu == "c3b4")
    not_pb = not TG._played_is_best(e_stale)
    ok += deep_ok; (fails.append("deep-best from top_lines") if not deep_ok else None)
    ok += not_pb; (fails.append("stale best_san not played==best") if not not_pb else None)
    print(f"  [{'PASS' if deep_ok else 'FAIL'}] deep best from top_lines (cxb4 not Bd2): {bu}")
    print(f"  [{'PASS' if not_pb else 'FAIL'}] stale best_san != played==best: {not_pb}")
    # greedy_capture: a passive (non-capture) played move must NOT fire Greedy Capture. Greedy Capture
    # fires only when the PLAYED move grabs material and best was quiet — the inverse of this case.
    # (Replaced the deleted capture_direction "Wrong/Missed Capture" catch-alls — GH #29.)
    m_stale = TG.deep_entry_to_mistake(e_stale, 1800, 1800)
    labs = [t[0] for t in PR.greedy_capture(m_stale)]
    mc_ok = "Greedy Capture" not in labs
    ok += mc_ok; (fails.append("Greedy Capture not fired on passive move") if not mc_ok else None)
    print(f"  [{'PASS' if mc_ok else 'FAIL'}] no 'Greedy Capture' on passive move: {labs}")
    extra_cd = 1
    # played==best: deep verdict cleared the shallow flag
    e_best = {"fen": "8/4b3/8/p2p1k1K/1PpPp2P/2P1B3/1P6/8 b - - 0 39", "uci": "a5b4", "best_san": "axb4",
              "top_lines": [{"eval": "-427", "moves": ["axb4"]}, {"eval": "+155", "moves": ["Bd8"]}]}
    pib = TG._played_is_best(e_best)
    ok += pib; (fails.append("played==best detect") if not pib else None)
    print(f"  [{'PASS' if pib else 'FAIL'}] played==best detected: {pib}")
    # only-move: >=150cp gap to 2nd best
    om = TG._only_move({"top_lines": [{"eval": "-39"}, {"eval": "-485"}]})
    not_om = TG._only_move({"top_lines": [{"eval": "-375"}, {"eval": "-490"}]})  # 115cp gap < 150
    ok += om; (fails.append("only-move detect") if not om else None)
    ok += (not not_om); (fails.append("only-move false-positive") if not_om else None)
    print(f"  [{'PASS' if om else 'FAIL'}] only-move (446cp gap): {om}")
    print(f"  [{'PASS' if not not_om else 'FAIL'}] not only-move (115cp gap): {not not_om}")
    extra_tg = 3

    print("--- motifs: pin target (relative pins + naming) ---")
    # (name, fen, move_uci, expected_target_piece_type or None)
    pin_cases = [
        # Bg4 pins Ne2 to Qd1 — RELATIVE pin (to queen), invisible to python-chess is_pinned (king-only).
        # FEN is the d4 position AFTER White's d4, Black to move; Bc8-g4 is the pin.
        ("Bg4 pins Ne2 to queen", "r1bq1rk1/5pbp/p5p1/1p1p4/2pP1P2/P7/1PPBN1PP/1R1Q1R1K b - - 0 17",
         "c8g4", chess.QUEEN),
        # Bb5 pins Nc6 to king — absolute pin (to king).
        ("Bb5 pins Nc6 to king", "r1bqkbnr/ppp1pppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
         "f1b5", chess.KING),
        # quiet move pins nothing
        ("e4 pins nothing", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", None),
        # Rf3 down an open f-file sees f5-pawn then f8-rook behind: NOT a pin (pinned piece is a PAWN).
        # (Sam's case — was falsely tagging "Pin (to Rook)".)
        ("Rf3 pawn-not-pin", "r4r1k/pbnnq2p/1p2p3/3pPp1B/1P1P1R2/PN6/6PP/R1BQ3K w - - 6 22",
         "f4f3", None),
    ]
    for name, fen, uci, want in pin_cases:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        got = M.pin_target_piece(b, mv) if mv in b.legal_moves else f"ILLEGAL({uci})"
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: target={got} exp={want}")

    print("--- motifs: clearance requires sacrifice OR check (not a quiet line-opening) ---")
    # NEG: Rf3 (safe rook, no check) then Bh6 incidentally uses the vacated f4 — NOT a clearance.
    clr_neg = U.build_line(chess.Board("r4r1k/pbnnq2p/1p2p3/3pPp1B/1P1P1R2/PN6/6PP/R1BQ3K w - - 6 22"),
                           ["f4f3", "f8g8", "c1h6", "b7a6", "d1d2", "a6c4"])
    neg = M.clearance_line(clr_neg, chess.WHITE)
    ok += (not neg)
    if neg:
        fails.append("safe Rf3 != clearance")
    print(f"  [{'PASS' if not neg else 'FAIL'}] safe-rook line-opening is NOT a clearance: fires={neg} (want False)")
    # POS: Ne6+ (check) clears d4 off the a1-h8 diagonal for Ba1-h8 — clearance-with-tempo (Gemini).
    clr_pos = U.build_line(chess.Board("3k4/8/8/8/3N4/8/8/B5K1 w - - 0 1"), ["d4e6", "d8e8", "a1h8"])
    pos = M.clearance_line(clr_pos, chess.WHITE)
    ok += pos
    if not pos:
        fails.append("check-clearance Ne6+/Bh8")
    print(f"  [{'PASS' if pos else 'FAIL'}] check-clearance (Ne6+ clears for Bh8): fires={pos} (want True)")
    extra_clr = 1   # the positive case (negative is counted inline above with the +1 in total)

    print("--- tag_adapter: leaked-played-move refutation parses to the hung piece (Mistake level) ---")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "..",
                                      "chess-deck-code", "backend", "worker"))
    try:
        import tag_adapter as TA
        e = {"fen_before": "r4r1k/pbnnq2p/1p2p3/3pPp1B/1P1P1R2/PN6/6PP/R1BQ3K w - - 6 22",
             "uci": "f4h4", "pv_uci": ["f4f3"], "san": "Rh4", "bestMoveSan": "Rf3", "eval": 2.36,
             "refutation_uci": ["f4h4", "e7h4", "g2g3"]}   # NOTE leading f4h4 = the played move
        # GH #29: the win%-drop ENTRY gate needs eval_after, which prod's eval_to_mistake does NOT yet
        # supply (hardcodes eval_after=None, cp_loss=0 — the open step-6 plumbing). So tag_mistake_full
        # correctly suppresses ALL explain tags here (no signal that it's a mistake). What we CAN verify
        # without step 6: the leaked-played-move stripping still builds a refutation that names the hung
        # rook at the Mistake/predicate level (gate-independent). When step 6 lands, tags_for_eval will
        # surface "Hung Rook" again — add that assertion then.
        m = TA.eval_to_mistake(e)
        import predicates as PR
        hung = [t[0] for t in PR.hung_material(m)]   # predicate is a pure detector now — fires regardless
        hm_ok = "Hung Rook" in hung
        ok += hm_ok
        if not hm_ok:
            fails.append("leaked-played-move refutation -> Hung Rook (Mistake level)")
        print(f"  [{'PASS' if hm_ok else 'FAIL'}] leaked move stripped, refutation names Hung Rook: {hung}")
        extra_adapter = 1
    except Exception as ex:
        print(f"  [SKIP] tag_adapter not importable ({ex})")
        extra_adapter = 0

    print("--- motifs: outpost (enemy half, pawn-defended, unchallengeable) ---")
    import chesslib_util as CU
    # (name, fen, move_uci_or_None, square, pov_white, expected)
    out_cases = [
        ("Nd4 outpost (c2 blocked by Nc3)", "r1bq1rk1/pp2npbp/2n3p1/1Bpp4/5P2/P1NP1N2/1PP3PP/R1BQ1RK1 b - - 1 10",
         "c6d4", chess.D4, False, True),
        ("Nd4 NOT outpost (c-pawn can play c3)", "r1bq1rk1/pp2npbp/6p1/1Bpp4/3n1P2/P2P1N2/1PP3PP/R1BQ1RK1 w - - 0 11",
         None, chess.D4, False, False),
        ("Ne5 outpost (no enemy d/f pawn)", "4k3/pp6/8/4N3/3P4/8/8/4K3 w - - 0 1",
         None, chess.E5, True, True),
        ("Ne5 NOT outpost (black f7 plays f6)", "4k3/pp3p2/8/4N3/3P4/8/8/4K3 w - - 0 1",
         None, chess.E5, True, False),
        ("Nd4 NOT outpost (no pawn defender)", "4k3/8/8/8/3N4/8/8/4K3 w - - 0 1",
         None, chess.D4, True, False),
    ]
    for name, fen, mv, sq, pov_white, want in out_cases:
        b = chess.Board(fen)
        if mv:
            b.push(chess.Move.from_uci(mv))
        got = CU.is_outpost(b, sq, chess.WHITE if pov_white else chess.BLACK)
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: {got} (want {want})")

    print("--- predicates: bishop endgame color split ---")
    be_cases = [
        ("same-color bishops", "4k3/8/8/3b4/4B3/8/4P1P1/4K3 w - - 0 1", "Bishop Endgame (Same Color)"),
        ("opposite-color bishops", "4k3/8/8/2b5/4B3/8/4P1P1/4K3 w - - 0 1", "Bishop Endgame (Opposite Color)"),
    ]
    for name, fen, want in be_cases:
        b = chess.Board(fen)
        m = Mistake(fen, "e1d1", "e1f1", [], [], 10, 5, 5, b.turn)
        labels = [t[0] for t in PR.endgame_type(m)]
        passed = want in labels
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: {labels} exp contains {want!r}")

    print("--- motifs: sacrifice (persistent investment, not transient dip) ---")
    # (name, fen, line_san, pov_white, expected)
    sac_cases = [
        # dead-equal trade mid-line: Bxe2 Nxe2 Bd6 Nxc6 bxc6 O-O — material recovers, NOT a sac
        ("transient dip != sacrifice", "r2qkb1r/pp3ppp/2n1pn2/3pN1Bb/3P4/2N4P/PPP1BPP1/R2QK2R b KQkq - 2 9",
         ["Bxe2", "Nxe2", "Bd6", "Nxc6", "bxc6", "O-O"], False, False),
        # real sac: white gives a bishop on f7 that is never recovered
        ("real unrecovered sac", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
         ["Bxf7+", "Kxf7", "Qh5+", "Kg8"], True, True),
    ]
    for name, fen, line_san, pov_white, want in sac_cases:
        b = chess.Board(fen)
        ucis = []
        for san in line_san:
            try:
                mv = b.parse_san(san); ucis.append(mv.uci()); b.push(mv)
            except Exception:
                break
        nodes = U.build_line(chess.Board(fen), ucis)
        got = M.sacrifice_line(nodes, chess.WHITE if pov_white else chess.BLACK)
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: fires={got} exp={want}")

    print("--- tagger: mate suppression ---")
    # a forced mate in a direction outranks lesser tactical motifs in that SAME direction only
    supp_cases = [
        ("mate suppresses same-dir fork",
         [("Missed Mate", "missed", "m"), ("Missed Fork", "missed", "f")], "Missed Fork", False),
        ("mate keeps other-dir fork",
         [("Missed Mate", "missed", "m"), ("Allowed Fork", "allowed", "f")], "Allowed Fork", True),
        ("mate keeps material tag",
         [("Missed Mate", "missed", "m"), ("Hung Material", "hung", "h")], "Hung Material", True),
        ("no mate -> fork survives",
         [("Missed Fork", "missed", "f")], "Missed Fork", True),
    ]
    for name, tags, label, should_keep in supp_cases:
        kept = [t[0] for t in T._suppress_lesser_under_mate(tags)]
        got = (label in kept)
        passed = (got == should_keep)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: {label} kept={got} exp={should_keep}")

    print("--- predicates: allowed_pawn_capture refutation-line fix (2026-07-11) ---")
    # Real game (cabbage): 17...Rb8 is a quiet mistake whose punishment is 18.Bxd5 grabbing the d5 pawn;
    # best 17...Nxc6 trades off the bishop that takes d5, so Bxd5 is unavailable after best. Material
    # nets back to 0 over the line (so hung_material correctly stays silent) — this is the gap: "you
    # let the opponent grab a pawn" even when it recaptures later. Lines reconstructed from the shipped
    # dump's UCI PVs.
    def _san_line(fen, ucis):
        bb = chess.Board(fen); out = []
        for u in ucis:
            try: mv = chess.Move.from_uci(u); out.append(bb.san(mv)); bb.push(mv)
            except Exception: break
        return out
    apc_fen = "r1bq2k1/p3rppp/1pBn4/n2pN3/Q2P4/P1P1B3/5PPP/2R1K2R b K - 7 17"
    apc_after = chess.Board(apc_fen); apc_after.push(chess.Move.from_uci("a8b8"))
    m_rb8 = Mistake(
        apc_fen, "a8b8", "a5c6",
        best_line_san=_san_line(apc_fen, ["a5c6", "a4c6", "c8b7", "c6a4", "f7f6", "e5d3"]),
        refutation_san=_san_line(apc_after.fen(), ["c6d5", "c8a6", "c3c4", "b6b5", "a4c2", "b5c4"]),
        eval_before=169, eval_after=18, cp_loss=151, mover=chess.BLACK,
        played_san="Rb8", best_san="Nxc6",
    )
    # NEG for allowed_pawn_capture: the played move IS a capture (equal trade) — that's hung_material /
    # greedy_capture's job, not "allowed pawn capture". A simple even bishop trade w/ recapture refutation.
    neg_fen = "r1bqk2r/ppp2ppp/2n2n2/3p4/1b1P4/2N1BN2/PPP1BPPP/R2QK2R b KQkq - 0 1"
    neg_after = chess.Board(neg_fen); neg_after.push(chess.Move.from_uci("b4c3"))  # Bxc3 (a capture)
    m_neg = Mistake(
        neg_fen, "b4c3", "e8g8",
        best_line_san=["O-O"], refutation_san=_san_line(neg_after.fen(), ["b2c3"]),
        eval_before=0, eval_after=-40, cp_loss=40, mover=chess.BLACK,
        played_san="Bxc3", best_san="O-O",
    )
    # POS path B: Nd2 (move 30) loses a pawn by force (Rxb5 at ply 4, net −1). Best Qd3 holds.
    apc_nd2_fen = "1r4k1/2q2pp1/1rP1p2p/1P1p4/4n3/5N1P/2Q2PP1/1RR3K1 w - - 3 30"
    apc_nd2_after = chess.Board(apc_nd2_fen); apc_nd2_after.push(chess.Move.from_uci("f3d2"))
    m_nd2 = Mistake(
        apc_nd2_fen, "f3d2", "c2d3",
        best_line_san=["Qd3", "Nd6", "Rc5", "Ne4", "Rc2", "Nd6"],
        refutation_san=_san_line(apc_nd2_after.fen(), ["e4d6", "d2f3", "b6b5", "b1b5", "b8b5", "f3d4"]),
        eval_before=155, eval_after=-44, cp_loss=199, mover=chess.WHITE,
        played_san="Nd2", best_san="Qd3",
    )
    apc_cases = [
        ("allowed_pawn_capture fires on Rb8 path A (first-reply pawn grab)", PR.allowed_pawn_capture(m_rb8) and PR.allowed_pawn_capture(m_rb8)[0][0], "Allowed Pawn Capture"),
        ("allowed_pawn_capture fires on Nd2 path B (delayed net-1 loss)", PR.allowed_pawn_capture(m_nd2) and PR.allowed_pawn_capture(m_nd2)[0][0], "Allowed Pawn Capture"),
        ("allowed_pawn_capture NEG: played is itself a capture", PR.allowed_pawn_capture(m_neg)[0][0] if PR.allowed_pawn_capture(m_neg) else None, None),
    ]
    for name, got, want in apc_cases:
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: got={got!r} exp={want!r}")
    extra_apc = len(apc_cases)

    total = (len(SINGLE_MOVE_CASES) + len(LINE_CASES) + len(split_cases)
             + len(hung_cases) + extra_exch + extra_greedy + extra_pinx + extra_gate + extra_cls + extra_gm + extra_grab + extra_castle + extra_sm + extra_eg + len(ps_cases) + extra_tg + 2 + extra_cd
             + len(pin_cases) + 1 + extra_clr + extra_adapter + len(out_cases)
             + len(be_cases) + len(sac_cases) + len(supp_cases) + extra_apc + extra_usac + extra_pchk
             + extra_ekp + extra_mac + extra_zz + extra_gg + extra_rek + extra_ovl + extra_conv + extra_sev + extra_md)
    print(f"\n{ok}/{total} passed" + (f" | FAILS: {fails}" if fails else ""))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
