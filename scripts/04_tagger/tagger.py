#!/usr/bin/env python3
"""The orchestrator — runs all layers on a Mistake and rolls fine tags into high-level categories.

  from tagger import tag_mistake_full
  result = tag_mistake_full(mistake)   # {tags:[...], categories:[...], maia:{...}}

Layer 2 = OWNED motif detectors (motifs.py), driven in three directions:
  MISSED X  : the BEST line contained tactic X that the player failed to play   (pov = mover)
  ALLOWED X : the played move let the OPPONENT execute tactic X in refutation   (pov = opponent)
  FAILED X  : the played move WAS tactic X but it backfired (single-move only)  (pov = mover)
Layer 1 = position predicates (predicates.py). Layer 3 = Maia rarity (numeric, not tags).

No cook_adapter, no vendor import — we own motifs.py. The motif->label map and category map are
config; Sam prunes/edits freely.
"""
import sys, os
import chess
sys.path.insert(0, os.path.dirname(__file__))
import motifs as MO
import predicates as PR
import maia_rarity as MR

# direction -> display prefix
DIR_PREFIX = {"missed": "Missed", "allowed": "Allowed", "failed": "Failed",
              "hung": "", "played": "", "info": ""}

# motif key -> human label (combined with the direction prefix)
MOTIF_LABEL = {
    "fork": "Fork", "pin": "Pin", "skewer": "Skewer", "discoveredAttack": "Discovered Attack",
    "deflection": "Deflection", "attraction": "Attraction", "clearance": "Clearance",
    "interference": "Interference", "intermezzo": "Zwischenzug", "overloading": "Overload",
    "xRayAttack": "X-Ray", "trappedPiece": "Trapped Piece", "sacrifice": "Sacrifice",
    "capturingDefender": "Capture of Defender", "hangingPiece": "Hanging Piece",
    "exposedKing": "Exposed King", "kingsideAttack": "Kingside Attack",
    "queensideAttack": "Queenside Attack", "promotion": "Promotion", "underPromotion": "Underpromotion",
    "enPassant": "En Passant", "castling": "Castling", "doubleCheck": "Double Check",
    "attackingF2F7": "f2/f7 Attack", "advancedPawn": "Advanced Pawn",
    "mate": "Mate", "anastasiaMate": "Anastasia's Mate", "arabianMate": "Arabian Mate",
    "bodenMate": "Boden's Mate", "doubleBishopMate": "Double Bishop Mate",
    "smotheredMate": "Smothered Mate", "backRankMate": "Back-Rank Mate", "hookMate": "Hook Mate",
    "dovetailMate": "Dovetail Mate", "Fork": "Fork", "Hanging Piece": "Hanging Piece",
    "Pin": "Pin", "Discovered Attack": "Discovered Attack",
}

# Motifs where the FAILED direction (played move itself was the tactic, and it backfired) is sensible.
# Single-move detectors only — sequence motifs need the opponent's cooperation so "failed X" is noise.
# NB "Hanging Piece" is NOT here: capturing a free piece (is_hanging_piece) can't "fail" — you won
# material. It fired "Failed Hanging Piece" on a recapture that was simply sub-optimal (gxf3 took a
# free bishop but Bxc6 first was better). A category error; only real attacking tactics can backfire.
FAILED_OK = {"Fork", "Pin", "Discovered Attack"}

# Mates render with their distance; keep the mate distance string off the label but in evidence.
def _mate_label(key, evidence):
    return MOTIF_LABEL.get(key, key)


def _san_line_to_ucis(start_board: chess.Board, san_line):
    """Convert a SAN move list (from start_board) to UCI strings, stopping at the first unparseable."""
    out = []
    b = start_board.copy()
    for san in san_line:
        try:
            mv = b.parse_san(san)
        except Exception:
            break
        out.append(mv.uci())
        b.push(mv)
    return out


def _best_line_ucis(m):
    """Best line as UCIs from fen_before. Prefer best_uci + best_line_san; ensure move 0 = best_uci."""
    b = m.board_before
    ucis = []
    if m.best_uci:
        try:
            mv = chess.Move.from_uci(m.best_uci)
            if mv in b.legal_moves:
                ucis.append(m.best_uci)
        except Exception:
            pass
    # append the rest of the SAN best line (which usually starts with best move too — dedupe move 0)
    san_ucis = _san_line_to_ucis(b, m.best_line_san)
    if ucis and san_ucis and san_ucis[0] == ucis[0]:
        ucis = san_ucis
    elif not ucis:
        ucis = san_ucis
    else:
        # best_uci leads; continue the line from after best_uci using SAN if it matches
        bb = b.copy(); bb.push(chess.Move.from_uci(ucis[0]))
        ucis += _san_line_to_ucis(bb, m.best_line_san[1:] if m.best_line_san and san_ucis and san_ucis[0]==ucis[0] else m.best_line_san)
    return ucis


def _allowed_line_ucis(m):
    """[played] + refutation, as UCIs from fen_before. Refutation SAN is from fen_after (after played)."""
    b = m.board_before
    try:
        played = chess.Move.from_uci(m.played_uci)
    except Exception:
        return []
    if played not in b.legal_moves:
        return []
    ucis = [m.played_uci]
    after = m.board_after
    ucis += _san_line_to_ucis(after, m.refutation_san)
    return ucis


def _motif_label(key, ev):
    """Map a motif key + evidence to its display label, applying the DEPTH SPLIT for fork:
    depth=0 -> the tactic is the move to play now ('Fork'); depth>0 -> it comes after a setup
    sequence ('Combination → Fork'). Depth is parsed from the 'depth=N ' prefix detect_line adds."""
    if key == "fork" and ev.startswith("depth="):
        try:
            depth = int(ev.split("depth=", 1)[1].split()[0])
        except Exception:
            depth = 0
        return "Fork" if depth == 0 else "Combination → Fork"
    return MOTIF_LABEL.get(key, key)


def _motif_tags(m):
    """Layer 2: owned motif detectors across MISSED / ALLOWED / FAILED."""
    out = []  # (label, direction, evidence)
    b = m.board_before
    mover = m.mover
    opp = not mover

    # MISSED: best line, pov = mover
    best_ucis = _best_line_ucis(m)
    if len(best_ucis) >= 1:
        for key, ev in MO.detect_line(b, best_ucis, mover).items():
            lab = _motif_label(key, ev)
            out.append((f"Missed {lab}".strip(), "missed", ev))

    # ALLOWED: [played]+refutation, pov = opponent (== cook's puzzle shape, the validated one)
    allowed_ucis = _allowed_line_ucis(m)
    if len(allowed_ucis) >= 2:   # need the played move + at least one punishment ply
        for key, ev in MO.detect_line(b, allowed_ucis, opp).items():
            lab = _motif_label(key, ev)
            out.append((f"Allowed {lab}".strip(), "allowed", ev))

    # FAILED: the played move itself was a (single-move) tactic that backfired
    try:
        played = chess.Move.from_uci(m.played_uci)
        if played in b.legal_moves and m.cp_loss >= 100:
            for key, ev in MO.detect_move(b, played).items():
                if key in FAILED_OK:
                    out.append((f"Failed {key}".strip(), "failed", ev))
    except Exception:
        pass

    return _suppress_lesser_under_mate(out)


# Lesser tactical motifs that a forced mate should outrank in the SAME direction. A coach says
# "you missed mate in 3", not "you missed a fork" — the fork is just a step inside the mating net.
# We keep Mate + named mates + Exposed King / attacks (they describe the mating attack), and we never
# touch the OTHER direction or position/material tags.
_MATE_OUTRANKS = {"Fork", "Combination → Fork", "Pin", "Skewer", "Discovered Attack", "Deflection",
                  "Attraction", "Clearance", "Interference", "Zwischenzug", "Overload", "X-Ray",
                  "Capture of Defender", "Hanging Piece", "Sacrifice", "Trapped Piece"}

def _suppress_lesser_under_mate(tags):
    """If a direction produced 'Missed/Allowed Mate', drop the lesser tactical motifs in that
    same direction (config — Sam can disable). Returns the filtered list, order preserved."""
    mate_dirs = {d for (lab, d, ev) in tags if lab in ("Missed Mate", "Allowed Mate")}
    if not mate_dirs:
        return tags
    kept = []
    for (lab, d, ev) in tags:
        if d in mate_dirs:
            # strip the "Missed "/"Allowed "/"Failed " prefix to get the bare motif name
            bare = lab.split(" ", 1)[1] if " " in lab else lab
            if bare in _MATE_OUTRANKS:
                continue
        kept.append((lab, d, ev))
    return kept


def categorize(label):
    # Material FIRST — "Hung Material" contains the substring "mate" (in "MATErial"), so the tactical
    # check must not see it first. Material/hung labels are unambiguous, so they win the tie.
    l = label.lower()
    if any(w in l for w in ["material", "capture", "exchange", "hung", "wrong piece"]):
        return "Material"
    if any(w in l for w in ["mate", "check", "fork", "combination", "pin", "skewer", "discovered",
                            "deflection", "attraction", "clearance", "interference", "zwischenzug",
                            "overload", "x-ray", "sacrifice", "f2/f7", "trapped piece", "defender"]):
        return "Tactical"
    if any(w in l for w in ["king", "castl", "attack"]):
        return "King Safety"
    if "endgame" in l or "zugzwang" in l or "opposition" in l or "promotion" in l:
        return "Endgame"
    if any(w in l for w in ["pawn", "tempo", "development", "advanced"]):
        return "Positional"
    if any(w in l for w in ["blunder while", "only move", "multiple good", "move order"]):
        return "Meta"
    return "Other"


def tag_mistake_full(m, with_maia=True):
    fine = []  # (label, direction, evidence, layer)

    for (label, direction, ev) in _motif_tags(m):
        fine.append((label, direction, ev, "tactic"))

    for (label, direction, ev) in PR.tag_predicates(m):
        fine.append((label, direction, ev, "position"))

    # dedupe by display label, keep first
    seen = set(); tags = []
    for t in fine:
        if t[0] in seen:
            continue
        seen.add(t[0]); tags.append(t)

    cat_set = sorted({categorize(t[0]) for t in tags})

    maia = {}
    if with_maia:
        try:
            maia = MR.rarity(m)
        except Exception:
            maia = {}

    return {
        "tags": [{"label": l, "direction": d, "evidence": e, "layer": ly} for (l, d, e, ly) in tags],
        "categories": cat_set,
        "maia": maia,
    }
