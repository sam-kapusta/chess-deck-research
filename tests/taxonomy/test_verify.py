from verify_descriptions import move_fingerprint, verify_description


def test_fingerprint_pawn_feature():
    # 1.b3 from the start position: a pawn move, no capture, no check.
    board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fp = move_fingerprint([(board_fen, "b2b3")])
    assert fp["dom_piece"] == "pawn"
    assert fp["dom_frac"] == 1.0
    assert fp["cap_rate"] == 0.0
    assert fp["check_rate"] == 0.0


def test_fingerprint_capture_detection():
    # exd5 — a capture.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    fp = move_fingerprint([(fen, "e4d5")])
    assert fp["cap_rate"] == 1.0


def test_verify_supported_piece_claim():
    board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fp = move_fingerprint([(board_fen, "b2b3")])
    out = verify_description("A slow pawn push wasting tempo.", fp)
    # description names 'pawn', dominant piece is pawn -> supported
    assert out["verdict"] == "supported"


def test_verify_contradicted_piece_claim():
    board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fp = move_fingerprint([(board_fen, "b2b3")])  # pawn move
    out = verify_description("The queen blunders into a fork.", fp)  # claims queen
    assert out["verdict"] == "contradicted"


def test_verify_unverifiable_when_no_claim():
    board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fp = move_fingerprint([(board_fen, "b2b3")])
    out = verify_description("A subtly inaccurate continuation.", fp)
    assert out["verdict"] == "unverifiable"
