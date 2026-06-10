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
        # quiet move, opponent grabs a free bishop next move — immediate hang
        ("free piece = Hung Material", "3k4/8/8/3b4/8/8/3R4/4K3 b - - 0 1",
         "d8c8", ["Rxd5"], "Hung Material"),
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

    print("--- predicates: pawn structure (recapture + best-line guards) ---")
    # (name, fen, played_uci, best_uci, refutation_san, label_must_NOT_appear)
    ps_cases = [
        # 13.gxf3 — a recapture that doubles the f-pawn, but it's a CAPTURE and the doubling is also
        # in the best line. Must NOT tag Created Doubled Pawn. (Caught by Sam.)
        ("recapture doesn't create doubled", "r2qr1k1/pp3ppp/2nb1n2/1Bpp4/8/P1NP1b2/1PPQ1PPP/R1B1R1K1 w - - 0 13",
         "g2f3", "b5c6", "Created Doubled Pawn"),
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
    # capture-direction: passive instead of capture -> Missed Capture
    m_stale = TG.deep_entry_to_mistake(e_stale, 1800, 1800)
    labs = [t[0] for t in PR.capture_direction(m_stale)]
    mc_ok = "Missed Capture" in labs
    ok += mc_ok; (fails.append("Missed Capture (passive)") if not mc_ok else None)
    print(f"  [{'PASS' if mc_ok else 'FAIL'}] Missed Capture (passive instead of cxb4): {labs}")
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

    print("--- tag_adapter: refutation with leaked played move still tags Hung Material ---")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "..",
                                      "chess-deck-code", "backend", "worker"))
    try:
        import tag_adapter as TA
        e = {"fen_before": "r4r1k/pbnnq2p/1p2p3/3pPp1B/1P1P1R2/PN6/6PP/R1BQ3K w - - 6 22",
             "uci": "f4h4", "pv_uci": ["f4f3"], "san": "Rh4", "bestMoveSan": "Rf3", "eval": 2.36,
             "refutation_uci": ["f4h4", "e7h4", "g2g3"]}   # NOTE leading f4h4 = the played move
        labs = [t["label"] for t in TA.tags_for_eval(e)]
        hm_ok = "Hung Material" in labs
        ok += hm_ok
        if not hm_ok:
            fails.append("adapter strips leaked played move")
        print(f"  [{'PASS' if hm_ok else 'FAIL'}] leaked-played-move refutation still yields Hung Material: {labs}")
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
             + len(hung_cases) + len(ps_cases) + extra_tg + 2 + extra_cd
             + len(pin_cases) + 1 + extra_clr + extra_adapter + len(out_cases)
             + len(be_cases) + len(sac_cases) + len(supp_cases))
    print(f"\n{ok}/{total} passed" + (f" | FAILS: {fails}" if fails else ""))
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
