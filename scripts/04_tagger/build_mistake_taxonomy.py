#!/usr/bin/env python3
"""Build mistakeTaxonomy.json — the flat label→{category,blurb} lookup the product consumes.

Single source of truth for category = tagger.categorize(). Enumerates every concrete label the
tagger can emit (directional motifs, parametrized pins/captures/endgames, named mates, predicates).
Regenerate after any tag rename/addition:  python3 scripts/04_tagger/build_mistake_taxonomy.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from tagger import categorize, FAILED_OK

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "mistakeTaxonomy.json")

CATEGORIES = ["Tactical", "Material", "King Safety", "Positional", "Endgame", "Meta", "Other"]

# Directional motifs (Missed/Allowed). Pin and Fork are parametrized separately below.
DIRECTIONAL = [
    "Skewer", "Discovered Attack", "Deflection", "Attraction", "Clearance", "Zwischenzug",
    "Overload", "X-Ray", "Trapped Piece", "Sacrifice", "Capture of Defender", "Hanging Piece",
    "Exposed King", "Kingside Attack", "Queenside Attack", "Promotion", "Underpromotion",
    "En Passant", "Castling", "Double Check", "f2/f7 Attack", "Advanced Pawn", "Mate", "Outpost",
]
NAMED_MATES = ["Anastasia's Mate", "Arabian Mate", "Boden's Mate", "Double Bishop Mate",
               "Smothered Mate", "Back-Rank Mate", "Hook Mate", "Dovetail Mate"]
PIECES = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]

# One-line coaching blurb per BASE concept.
BLURB = {
    "Fork": "A fork was available you didn't play",
    "Combination → Fork": "A short combination ending in a fork was available",
    "Skewer": "A skewer was available", "Discovered Attack": "A discovered attack was available",
    "Deflection": "A deflection was available", "Attraction": "An attraction was available",
    "Clearance": "A clearance was available", "Zwischenzug": "An in-between move was available",
    "Overload": "An overloaded enemy piece could be exploited", "X-Ray": "An x-ray was available",
    "Trapped Piece": "An enemy piece could be trapped", "Sacrifice": "A sound sacrifice was available",
    "Capture of Defender": "Capturing the defender was available", "Hanging Piece": "A piece was hanging",
    "Exposed King": "The enemy king was exposed", "Kingside Attack": "A kingside attack was available",
    "Queenside Attack": "A queenside attack was available", "Promotion": "A promotion was available",
    "Underpromotion": "An underpromotion was the move", "En Passant": "En passant was the move",
    "Castling": "Castling was the move", "Double Check": "A double check was available",
    "f2/f7 Attack": "An attack on the f2/f7 square was available",
    "Advanced Pawn": "An advanced pawn push was the move",
    "Mate": "Forced mate was available", "Outpost": "A piece could take a permanent outpost",
}
NAMED_MATE_BLURB = {m: f"{m} was available" for m in NAMED_MATES}


def build_taxonomy():
    tags = {}

    def add(label, blurb):
        tags[label] = {"category": categorize(label), "blurb": blurb}

    # directional motifs
    for x in DIRECTIONAL:
        b = BLURB.get(x, x)
        add(f"Missed {x}", f"{b} (you didn't play it)")
        add(f"Allowed {x}", f"Your move let the opponent: {b[0].lower() + b[1:]}")
    # fork + its combination split
    for x in ["Fork", "Combination → Fork"]:
        add(f"Missed {x}", f"{BLURB[x]} (you didn't play it)")
        add(f"Allowed {x}", f"Your move let the opponent execute {x.lower()}")
    # pins, parametrized by target
    for tgt in ["King", "Queen", "Rook"]:
        add(f"Missed Pin (to {tgt})", f"A pin against the {tgt.lower()} was available")
        add(f"Allowed Pin (to {tgt})", f"Your move let the opponent pin a piece to your {tgt.lower()}")
    # named mates
    for m in NAMED_MATES:
        add(f"Missed {m}", f"{NAMED_MATE_BLURB[m]} (you didn't play it)")
        add(f"Allowed {m}", f"Your move allowed {m}")
    # FAILED direction (single-move tactics that backfired)
    for x in sorted(FAILED_OK):
        add(f"Failed {x}", f"You tried a {x.lower()} but it backfired")
    # Layer-1 material predicates
    for p in PIECES:
        add(f"Missed Free Capture ({p})", f"A free {p.lower()} could be captured")
        add(f"Missed Winning Capture ({p})", f"Winning a {p.lower()} for less was available")
        if p != "Pawn":
            add(f"Missed Exchange ({p})", f"An even {p.lower()} trade was the move")
    add("Missed Pawn Trade", "An even pawn trade was the move")
    add("Missed Capture (Pawn)", "An en-passant capture was available")
    add("Missed Capture", "A capture was the best move; you played a quiet move")
    add("Wrong Capture", "You captured the wrong target")
    add("Bad Capture", "Your capture lost material/eval")
    add("Hung Material", "Your move dropped material to a one-move capture")
    add("Lost Material to Combination", "Your move lost material after a short sequence")
    add("Captured With Wrong Piece", "You recaptured with the wrong piece")
    # King safety predicates
    add("King in Center", "Your king stayed in the center too long")
    add("Lost Castling Rights", "Your move forfeited castling")
    add("Pawn Move Exposed King", "A pawn move weakened your king's shelter")
    # Positional predicates
    for s in ["Doubled", "Isolated", "Backward"]:
        add(f"Created {s} Pawn", f"Your move created a {s.lower()} pawn")
    # Endgame types
    for t in ["Pawn", "Rook", "Queen", "Knight"]:
        add(f"{t} Endgame", f"A {t.lower()} endgame")
    add("Queen + Rook Endgame", "A queen-and-rook endgame")
    add("Bishop Endgame", "A bishop endgame")
    add("Bishop Endgame (Same Color)", "A same-color-bishop endgame")
    add("Bishop Endgame (Opposite Color)", "An opposite-color-bishop endgame (often drawish)")
    # Meta + phase + info
    add("Winning", "You were winning before this move")
    add("Losing", "You were losing before this move")
    add("Equal", "The position was equal before this move")
    add("Only Move", "There was only one good move here")
    add("Wrong Move Order", "Right idea, wrong move order")
    add("Best Move (deep analysis)", "Deep analysis confirms your move was best")
    for ph in ["Opening", "Middlegame", "Endgame"]:
        add(ph, f"{ph} phase")

    return {"categories": {c: {} for c in CATEGORIES}, "tags": tags}


def main():
    tax = build_taxonomy()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(tax, f, indent=2)
    print(f"wrote {len(tax['tags'])} tags across {len(tax['categories'])} categories -> {OUT}")


if __name__ == "__main__":
    main()
