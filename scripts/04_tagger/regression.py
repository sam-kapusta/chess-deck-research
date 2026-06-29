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
    # control: same geometry but the front knight IS defended (d2 pawn guards c3, OFF the pinning
    # diagonal so the ray to a1 stays clear) → a genuine relative pin to the rook. Distinguishes the
    # fix from a blunt "never pin to rook".
    ("Qd4 pins DEFENDED knight to rook", "2kr1b1r/pppq2p1/4pn2/4p3/4P3/2NP4/P2P2P1/R1BQ1RK1 b - - 0 12", "d7d4", M.is_pin, True),
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
    ]
    for name, key, ev, want in split_cases:
        got = T._motif_label(key, ev)
        passed = (got == want)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        if not passed:
            fails.append(name)
        print(f"  [{mark}] {name}: label={got!r} exp={want!r}")

    print("--- predicates: hung material (equal trades excluded, immediate vs delayed) ---")
    import predicates as PR
    from mistake import Mistake
    # (name, fen, played_uci, refutation_san, expected_label_or_None)
    hung_cases = [
        # dead-equal trade Bxc6 bxc6 (bishop-for-knight, 3-for-3) — must NOT fire (the bug Sam caught)
        ("equal trade != hung", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
         "c4c6", ["bxc6"], None),
        # quiet move, opponent grabs a free bishop next move — immediate hang, NAMED by the victim piece
        ("free bishop = Hung Bishop", "3k4/8/8/3b4/8/8/3R4/4K3 b - - 0 1",
         "d8c8", ["Rxd5"], "Hung Bishop"),
    ]
    for name, fen, uci, ref, want in hung_cases:
        b = chess.Board(fen)
        m = Mistake(fen, uci, "", [], ref, 0, -300, 300, b.turn)
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
        # POS: White Nxd5 grabs a pawn; best is the quiet Bh6. cp200 -> 17.6 win-pts (taggable).
        ("grab pawn when quiet Bh6 was best", "r2qk2r/ppp2ppp/2n5/3p4/3P4/2N1B3/PPP2PPP/R2QK2R w KQkq - 0 1",
         "c3d5", "e3h6", 200, "Greedy Capture"),
        # NEG: best is ALSO a capture (Nxd5) — that's a missed capture/exchange, not greed. Must NOT fire.
        ("best is also a capture -> not greedy", "r2qk2r/ppp2ppp/2n5/3p4/3P4/2N1B3/PPP2PPP/R2QK2R w KQkq - 0 1",
         "c3d5", "d4d5", 200, None),
        # GH #29: pure detector — fires on the grab-vs-quiet PATTERN regardless of severity. The low-
        # win-drop suppression is the entry gate's job (tested in the entry-gate block), not here.
        ("greedy grab detects regardless of cp (gate suppresses)", "r2qk2r/ppp2ppp/2n5/3p4/3P4/2N1B3/PPP2PPP/R2QK2R w KQkq - 0 1",
         "c3d5", "e3h6", 20, "Greedy Capture"),
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

    total = (len(SINGLE_MOVE_CASES) + len(LINE_CASES) + len(split_cases)
             + len(hung_cases) + extra_exch + extra_greedy + extra_pinx + extra_gate + extra_grab + extra_eg + len(ps_cases) + extra_tg + 2 + extra_cd
             + len(pin_cases) + 1 + extra_clr + extra_adapter + len(out_cases)
             + len(be_cases) + len(sac_cases) + len(supp_cases))
    print(f"\n{ok}/{total} passed" + (f" | FAILS: {fails}" if fails else ""))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
