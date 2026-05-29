"""Deterministic structural fingerprint + description verification.

For each feature, compute the chess fingerprint of its top-N profile
positions, and check whether the feature's `description` headline claim
(piece type, capture, check, hanging) matches what the board actually shows.
"""
import re
import math
from collections import Counter

import chess

FEAT_INPUTS = None  # test fixture hook; real callers pass positions explicitly


def move_fingerprint(positions):
    """positions: list of (fen, uci). Returns structural fingerprint dict."""
    pieces = Counter(); to_sq = Counter(); from_sq = Counter()
    caps = 0; checks = 0; promos = 0; hangs = 0; n = 0
    for fen, uci in positions:
        try:
            b = chess.Board(fen)
            mv = chess.Move.from_uci(uci)
        except Exception:
            continue
        n += 1
        pc = b.piece_at(mv.from_square)
        pieces[chess.piece_name(pc.piece_type) if pc else "?"] += 1
        to_sq[chess.square_name(mv.to_square)] += 1
        from_sq[chess.square_name(mv.from_square)] += 1
        caps += 1 if b.is_capture(mv) else 0
        checks += 1 if b.gives_check(mv) else 0
        promos += 1 if mv.promotion else 0
        # hang: after the move, the moved piece sits on a square attacked by the
        # opponent and not defended by us (a non-pawn left en prise undefended)
        try:
            bb = b.copy(); bb.push(mv)
            them = bb.turn; us = not them
            sq = mv.to_square
            if (pc and pc.piece_type != chess.PAWN
                    and bb.is_attacked_by(them, sq)
                    and not bb.is_attacked_by(us, sq)):
                hangs += 1
        except Exception:
            pass
    if n == 0:
        return None
    dom_piece, dom_n = pieces.most_common(1)[0]

    def entropy(c):
        t = sum(c.values())
        return -sum((v / t) * math.log2(v / t) for v in c.values()) if t else 0.0

    return {
        "n": n,
        "dom_piece": dom_piece,
        "dom_frac": dom_n / n,
        "piece_dist": dict(pieces.most_common()),
        "to_sq_top": dict(to_sq.most_common(5)),
        "to_sq_entropy": round(entropy(to_sq), 2),
        "cap_rate": caps / n,
        "check_rate": checks / n,
        "promo_rate": promos / n,
        "hang_rate": hangs / n,
    }


# Claim keywords that appear in chips/descriptions, mapped to board predicates.
_CLAIM_KW = {
    "check": ["check"],
    "capture": ["captur", "takes", "grab", "greedy", "snatch"],
    "promote": ["promot", "queens", "underpromo"],
    "fork": ["fork"],
}
_SQ_RE = re.compile(r"\b[a-h][1-8]\b")
_PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def verify_description(description, fingerprint):
    """Check the description's structural claims against the fingerprint.

    Returns dict: {verdict, checks: {claim: (claimed, observed_rate, ok)}}.
    verdict in {supported, partial, contradicted, unverifiable}.
    Mechanism/tactical claims (refutations) are NOT checked here — those came
    from Stockfish depth-18 in Pass 1 and are trusted. We only check the
    surface move facts the board can confirm.
    """
    if not description or fingerprint is None:
        return {"verdict": "unverifiable", "checks": {}}
    low = description.lower()
    checks = {}

    # piece claim: does the description name the dominant piece?
    claimed_pieces = [p for p in _PIECES if p in low]
    if claimed_pieces:
        observed = fingerprint["dom_frac"]
        ok = any(
            fingerprint["piece_dist"].get(p, 0) / fingerprint["n"] >= 0.4
            for p in claimed_pieces
        )
        checks["piece"] = (claimed_pieces, round(observed, 2), ok)

    # capture / check / promote rate claims
    rate_map = {
        "capture": fingerprint["cap_rate"],
        "check": fingerprint["check_rate"],
        "promote": fingerprint["promo_rate"],
    }
    thresh = {"capture": 0.4, "check": 0.4, "promote": 0.3}
    for claim, kws in _CLAIM_KW.items():
        if claim not in rate_map:
            continue
        if any(w in low for w in kws):
            r = rate_map[claim]
            checks[claim] = (True, round(r, 2), r >= thresh[claim])

    if not checks:
        return {"verdict": "unverifiable", "checks": {}}
    oks = [v[-1] for v in checks.values()]
    if all(oks):
        verdict = "supported"
    elif not any(oks):
        verdict = "contradicted"
    else:
        verdict = "partial"
    return {"verdict": verdict, "checks": checks}
