#!/usr/bin/env python3
"""The orchestrator — runs all layers on a Mistake and rolls fine tags into high-level categories.

  from tagger import tag_mistake_full
  result = tag_mistake_full(mistake)   # {tags:[...], categories:[...], maia:{...}}

Layer 2 (cook motifs, prefixed by direction) + Layer 1 (position predicates) + Layer 3 (Maia rarity).
The CATEGORY_MAP groups fine tags into the report's high-level buckets. Both the cook keep-list and
this map are config — Sam prunes/edits freely.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cook_adapter as CA
import predicates as PR
import maia_rarity as MR

# direction prefix for display
DIR_PREFIX = {"missed": "Missed", "allowed": "Allowed", "failed": "Failed",
              "hung": "", "played": "", "info": ""}

# cook motif -> human label (used with the direction prefix)
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
    "dovetailMate": "Dovetail Mate",
    "pawnEndgame": "Pawn Endgame", "queenEndgame": "Queen Endgame", "rookEndgame": "Rook Endgame",
    "bishopEndgame": "Bishop Endgame", "knightEndgame": "Knight Endgame",
    "queenRookEndgame": "Queen+Rook Endgame", "zugzwang": "Zugzwang",
}

# high-level category for every possible tag label (fine label -> category)
def categorize(label):
    l = label.lower()
    if any(w in l for w in ["mate", "check", "fork", "pin", "skewer", "discovered", "deflection",
                            "attraction", "clearance", "interference", "zwischenzug", "overload",
                            "x-ray", "sacrifice", "double check", "f2/f7", "trapped piece",
                            "capture of defender"]):
        return "Tactical"
    if any(w in l for w in ["capture", "exchange", "hung", "bad capture", "material", "wrong piece"]):
        return "Material"
    if any(w in l for w in ["king", "castl", "attack", "mating"]):
        return "King Safety"
    if "endgame" in l or "zugzwang" in l or "opposition" in l or "promotion" in l:
        return "Endgame"
    if any(w in l for w in ["pawn", "tempo", "development", "advanced"]):
        return "Positional"
    if any(w in l for w in ["blunder while", "only move", "multiple good", "move order"]):
        return "Meta"
    return "Other"


def tag_mistake_full(m):
    fine = []  # (label, direction, evidence, layer)

    # Layer 2: cook motifs
    for motif, direction, ev in CA.tag_mistake(m):
        label = MOTIF_LABEL.get(motif, motif)
        prefix = DIR_PREFIX.get(direction, "")
        disp = f"{prefix} {label}".strip()
        fine.append((disp, direction, ev, "tactic"))

    # Layer 1: predicates (labels already human)
    for label, direction, ev in PR.tag_predicates(m):
        fine.append((label, direction, ev, "position"))

    # dedupe by display label, keep first
    seen = set(); tags = []
    for t in fine:
        if t[0] in seen:
            continue
        seen.add(t[0]); tags.append(t)

    cats = sorted({categorize(t[0]) for t in tags if categorize(t[0]) != "Other" or True})
    # exclude pure-info phase/state from the "category" headline set but keep as tags
    cat_set = sorted({categorize(t[0]) for t in tags})

    maia = {}
    try:
        maia = MR.rarity(m)
    except Exception:
        maia = {}

    return {
        "tags": [{"label": l, "direction": d, "evidence": e, "layer": ly} for (l, d, e, ly) in tags],
        "categories": cat_set,
        "maia": maia,
    }
