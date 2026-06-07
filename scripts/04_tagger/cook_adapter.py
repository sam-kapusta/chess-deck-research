#!/usr/bin/env python3
"""Layer 2 — adapt Lichess's vendored cook.py to tag a MISTAKE (three directions), not a puzzle.

A puzzle is one forced line from a good position (cook reads mainline[1::2] as the pov player's moves,
pov = not game.turn()). A mistake has three things to tag:
  - MISSED   : motif in the BEST line (pov = mover)            -> "Missed X"
  - ALLOWED  : motif in the REFUTATION (pov = opponent)        -> "Allowed X"
  - FAILED   : motif in the PLAYED move itself + eval crashed  -> "Failed X"

POV mechanics: cook's Puzzle sets pov = not game.turn(), expecting mainline[0] to be the OPPONENT's
setup move so the pov side moves at odd indices. To tag a line whose FIRST move is by side S as S's
tactic, we build the game starting from a board where it's (not S) to move, prepend a NULL move, then
play the line — so the line sits at odd indices and pov == S. python-chess supports null moves
(b.push(chess.Move.null())), which is exactly the "pass" we need for the setup ply.

We override pov explicitly after construction rather than trust the auto value (validation bug:
auto-pov tagged forks on the wrong side).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))
import chess, chess.pgn
import cook as _cook
from model import Puzzle

# Tags worth keeping for a mistake. Puzzle-meta tags (advantage/crushing/length/etc.) are dropped.
MOTIF_TAGS = {
    "fork", "pin", "skewer", "discoveredAttack", "deflection", "attraction", "clearance",
    "interference", "intermezzo", "overloading", "xRayAttack", "trappedPiece", "sacrifice",
    "capturingDefender", "hangingPiece", "exposedKing", "kingsideAttack", "queensideAttack",
    "promotion", "underPromotion", "enPassant", "castling", "doubleCheck", "attackingF2F7",
    "advancedPawn",
    # mates
    "mate", "anastasiaMate", "arabianMate", "bodenMate", "doubleBishopMate", "smotheredMate",
    "backRankMate", "hookMate", "dovetailMate",
    # endgame types
    "pawnEndgame", "queenEndgame", "rookEndgame", "bishopEndgame", "knightEndgame", "queenRookEndgame",
    "zugzwang",
}
# Tags we never want (puzzle quality / line length / collinear noise)
DROP_TAGS = {"advantage", "crushing", "equality", "long", "short", "veryLong", "oneMove",
             "quietMove", "defensiveMove", "collinearMove", "mateIn1", "mateIn2", "mateIn3",
             "mateIn4", "mateIn5", "simplification"}


def _build_puzzle(start_fen, line_san, pov, cp=999):
    """Build a cook Puzzle from `line_san` played as-is from start_fen, with pov set explicitly.
    The line's first move is by the side delivering the tactic. We do NOT prepend a null ply
    (it crashes several index-sensitive detectors and breaks check positions). Index-agnostic
    detectors (mate, named mates, exposedKing) work regardless of parity; for parity-sensitive
    ones (fork/pin/skewer) we run BOTH pov values upstream and union the motif tags.
    Returns None if the line is empty/illegal."""
    if not line_san:
        return None
    b = chess.Board(start_fen)
    g = chess.pgn.Game(); g.setup(b)
    node = g; bb = b.copy(); played_any = False
    for san in line_san:
        try:
            mv = bb.parse_san(san)
        except Exception:
            break
        node = node.add_variation(mv); bb.push(mv); played_any = True
    if not played_any:
        return None
    p = Puzzle(id="m", game=g, cp=cp)
    p.pov = pov  # OVERRIDE the auto pov (validation bug fix)
    return p


def _cook_filtered(puzzle):
    """Run cook defensively (some vendored detectors index out of range on non-puzzle lines).
    Returns the motif-filtered tag set, or empty on any failure."""
    if puzzle is None:
        return set()
    try:
        tags = _cook.cook(puzzle)
    except Exception:
        return set()
    return {t for t in tags if t in MOTIF_TAGS}


def _cook_both_pov(start_fen, line_san, primary_pov):
    """Parity workaround: parity-sensitive detectors scan mainline[1::2]. Run cook with both pov
    values and union the motifs, so a tactic by the line's mover is caught regardless of index parity.
    primary_pov is the side actually delivering the tactic (used for non-parity tags)."""
    a = _cook_filtered(_build_puzzle(start_fen, line_san, primary_pov))
    b = _cook_filtered(_build_puzzle(start_fen, line_san, not primary_pov))
    # mates/named-mates are index-agnostic and direction-true only for primary_pov's game-end;
    # union is fine here because both builds share the same final position.
    return a | b


def tag_mistake(m, eval_crash_cp=120):
    """Return a list of (tag, direction, evidence). direction in {missed, allowed, failed}."""
    out = []
    bb = m.board_before

    # MISSED: motifs in the best line, tactic delivered by mover
    missed = _cook_both_pov(m.fen_before, m.best_line_san, m.mover)

    # ALLOWED: motifs in the refutation (from position AFTER played move), tactic by opponent
    fen_after = m.board_after.fen()
    allowed = _cook_both_pov(fen_after, m.refutation_san, not m.mover)

    # FAILED: the played move itself attempted a motif AND the eval crashed for the mover
    failed = set()
    crashed = (m.cp_loss is not None and m.cp_loss >= eval_crash_cp)
    if crashed:
        try:
            psan = bb.san(chess.Move.from_uci(m.played_uci))
            failed = _cook_both_pov(m.fen_before, [psan] + m.refutation_san[:1], m.mover)
        except Exception:
            failed = set()

    for t in sorted(missed):
        out.append((t, "missed", "best-line"))
    for t in sorted(allowed):
        out.append((t, "allowed", "refutation"))
    # only surface FAILED for genuinely attacking motifs the player initiated
    for t in sorted(failed & {"fork", "sacrifice", "skewer", "pin", "attraction", "discoveredAttack",
                              "kingsideAttack", "queensideAttack", "attackingF2F7"}):
        if t not in missed:  # don't double-count
            out.append((t, "failed", "played-move+eval-crash"))
    return out
