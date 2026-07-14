#!/usr/bin/env python3
"""Build mistakeTaxonomy.json — the flat label→{category,blurb} lookup the product consumes.

Single source of truth for category = tagger.categorize(). Enumerates every concrete label the
tagger can emit (directional motifs, parametrized pins/captures/endgames, named mates, predicates).
Regenerate after any tag rename/addition:  python3 scripts/04_tagger/build_mistake_taxonomy.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from tagger import categorize, family_of, FAILED_OK

# One coaching blurb per FAMILY parent (the concept the piece-specific chips roll up to).
_FAMILY_BLURB = {
    "Missed Free Material": "You could have won free or favorable material",
    "Hung Material": "Your move left material to be captured",
    "Missed Exchange": "A favorable or even trade was the move",
    "Fork": "A fork was available",
}

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "mistakeTaxonomy.json")

CATEGORIES = ["Hung Piece", "Missed Capture", "Missed Tactic", "Missed Mate", "Allowed Tactic",
              "Calculation", "Trading", "Position", "King Safety", "Endgame", "Meta", "Other"]

# Tags that should NOT feed the drill queue. These describe a structural STATE your move left behind
# (king/pawn structure) — there's no single "find the best move" puzzle to re-solve, unlike "Missed
# Fork" (find it) or "Hung Rook" (don't hang it). They still render as Review chips and count in the
# skill card; they're just not drillable. (Sam, 2026-06-19. Meta/Other context tags are already
# excluded upstream via direction=="info"; listed here too so the taxonomy is self-describing.)
# These are all `played`-direction tags — the "resulting state" class, vs the "move to find" class.
NON_DRILLABLE = {
    # king-structure state
    "Exposed King", "King in Center", "Lost Castling Rights", "Pawn Move Exposed King",
    # pawn-structure state
    "Created Doubled Pawn", "Created Isolated Pawn", "Created Backward Pawn",
}

# Directional motifs (Missed/Allowed). Pin and Fork are parametrized separately below.
DIRECTIONAL = [
    "Skewer", "Discovered Attack", "Deflection", "Attraction", "Clearance", "Zwischenzug",
    "Overload", "X-Ray", "Trapped Piece", "Sacrifice", "Capture of Defender", "Hanging Piece",
    "Kingside Attack", "Queenside Attack", "Promotion", "Underpromotion",
    "En Passant", "Castling", "Double Check", "f2/f7 Attack", "Advanced Pawn", "Mate", "Outpost",
]
NAMED_MATES = ["Anastasia's Mate", "Arabian Mate", "Boden's Mate", "Double Bishop Mate",
               "Smothered Mate", "Back-Rank Mate", "Hook Mate", "Dovetail Mate"]
PIECES = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]

# One-line coaching blurb per BASE concept.
BLURB = {
    "Fork": "A fork was available",
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
        cat = categorize(label)
        # drillable: has a concrete best move to re-solve. False for structural-state tags (NON_DRILLABLE)
        # and for Meta/Other context tags (phase/game-state — never a puzzle).
        drillable = label not in NON_DRILLABLE and cat not in ("Meta", "Other")
        tags[label] = {"category": cat, "blurb": blurb, "drillable": drillable}

    # directional motifs
    for x in DIRECTIONAL:
        b = BLURB.get(x, x)
        add(f"Missed {x}", f"{b} (you didn't play it)")
        add(f"Allowed {x}", f"Your move let the opponent: {b[0].lower() + b[1:]}")
    # fork + its combination split, and the by-PIECE variants (#53): _motif_label emits "Knight Fork",
    # "Combination → Queen Fork", etc. Enumerate all so they get a category/blurb (else they fall back
    # to Other in the frontend). Generic "Fork" stays for lines where the forking piece isn't resolved.
    fork_forms = ["Fork", "Combination → Fork"]
    for pc in ["Knight", "Bishop", "Rook", "Queen", "Pawn", "King"]:
        fork_forms += [f"{pc} Fork", f"Combination → {pc} Fork"]
    for x in fork_forms:
        base = "a fork" if x.endswith("Fork") and "→" not in x else "a short combination ending in a fork"
        add(f"Missed {x}", f"{base} was available (you didn't play it)")
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
    # Layer-1 material predicates. Label form is "Missed Free <Piece>" / "Missed <Piece> Exchange"
    # (piece name inline, not parenthesized) — must match the strings predicates.capture_or_exchange
    # emits.
    # "Missed Free X" covers BOTH an undefended grab AND a favorable (win-material) capture — the
    # tagger merges them into one tag (the old separate "Missed Winning Capture (X)" is gone).
    for p in PIECES:
        add(f"Missed Free {p}", f"A free or favorable {p.lower()} capture was available")
        if p != "Pawn":
            add(f"Missed {p} Exchange", f"An even {p.lower()} trade was the move")
    # mixed minor trade (bishop for knight / knight for bishop) — its own decision (bishop pair). (GH #28)
    add("Missed Bishop-Knight Exchange", "An even bishop-for-knight trade was the move")
    add("Missed Pawn Trade", "An even pawn trade was the move")
    add("Missed Capture (Pawn)", "An en-passant capture was available")
    add("Greedy Capture", "You grabbed material when a quiet move was stronger")
    add("Unsound Sacrifice", "You sacrificed material at the enemy king with no real compensation")
    add("Pointless Check", "You gave an aimless check when a quiet improving move was stronger")
    add("Missed Attacking Check", "A forcing check on the exposed enemy king was the strong move")
    add("Missed Zwischenzug", "An in-between move (check) should come before your recapture")
    add("Missed Greek Gift", "A bishop sacrifice on the castled king (Greek Gift) was winning")
    add("Hung Material", "Your move dropped material to a one-move capture")
    for p in ["Knight", "Bishop", "Rook", "Queen"]:
        add(f"Hung {p}", f"Your move left your {p.lower()} to be captured next move")
    # Exposed king — explicit labels (no Missed/Allowed prefix; reads as a positional state).
    add("Exposed King", "Your move left your own king exposed")
    add("Enemy King Exposed", "The enemy king was exposed and you didn't press the attack")
    # King safety predicates
    add("King in Center", "Your king stayed in the center too long")
    add("Lost Castling Rights", "Your move forfeited castling")
    add("Pawn Move Exposed King", "A pawn move weakened your king's shelter")
    add("Recapture Exposed King", "Your pawn recapture opened a line onto your own king")
    # Trapped piece — tagger names the piece ("Trapped Bishop") + a generic fallback ("Trapped Piece").
    for p in ["Piece", "Pawn", "Knight", "Bishop", "Rook", "Queen"]:
        add(f"Missed Trapped {p}", f"An enemy {p.lower() if p != 'Piece' else 'piece'} could be trapped (you didn't play it)")
        add(f"Allowed Trapped {p}", f"Your move let the opponent trap your {p.lower() if p != 'Piece' else 'piece'}")
    # Positional predicates
    for s in ["Doubled", "Isolated", "Backward"]:
        add(f"Created {s} Pawn", f"Your move created a {s.lower()} pawn")
    # Plan-execution positional detectors (2026-06-14)
    add("Missed Pawn Break", "A thematic pawn break was available; you played a waiting move")
    add("Missed Tempo Push", "A pawn push attacking an enemy piece (gaining tempo) was available")
    add("Missed Open File", "A rook could occupy an open or half-open file")
    add("Premature Trade", "You exchanged while ahead, relieving tension that favored you")
    add("Missed Prophylaxis", "You let the opponent carry out a plan a quiet move would have stopped")
    add("Missed Piece Activation", "A passive piece could be repositioned to a more active square")
    add("Wrong Pawn Race", "You went the wrong direction in a pawn race and lost a critical tempo")
    add("Allowed Advanced Pawn", "Your move let the opponent advance a pawn dangerously")
    add("Missed Advanced Pawn", "An advanced pawn push was the move")
    # Battery / overloading / doubled-rooks detectors + the allowed-pawn-grab detector (these fire from
    # predicates.py but were missing taxonomy entries → rendered neutral). Missed + Allowed both exist.
    add("Missed Battery", "Lining up two heavy pieces (a battery) against a target was available")
    add("Allowed Battery", "Your move let the opponent build a battery against your position")
    add("Missed Overloading", "An enemy piece was overloaded and could be exploited")
    add("Allowed Overloading", "Your move left one of your pieces overloaded for the opponent to exploit")
    add("Missed Doubled Rooks", "Doubling rooks on a file was the move")
    add("Allowed Doubled Rooks", "Your move let the opponent double rooks on a file")
    add("Allowed Pawn Capture", "Your quiet move let the opponent grab a pawn the best move prevented")
    # Endgame types
    for t in ["Pawn", "Rook", "Queen", "Knight"]:
        add(f"{t} Endgame", f"A {t.lower()} endgame")
    add("Queen + Rook Endgame", "A queen-and-rook endgame")
    add("Bishop Endgame", "A bishop endgame")
    add("Bishop Endgame (Same Color)", "A same-color-bishop endgame")
    add("Bishop Endgame (Opposite Color)", "An opposite-color-bishop endgame (often drawish)")
    # Endgame mistakes (detectors, 2026-06-13)
    add("Missed King Activity", "Your king should have activated toward the center or the pawns")
    add("Lost the Opposition", "You gave up the opposition in a king-and-pawn endgame")
    add("Missed Passed Pawn", "A move that made or advanced a passed pawn was best")
    add("Rook Behind Passer", "The rook belonged behind the passed pawn")
    # Meta + phase + info
    add("Winning", "You were winning before this move")
    add("Losing", "You were losing before this move")
    add("Equal", "The position was equal before this move")
    # Conversion outcome (result-band transition, descriptive info — names what a move DID to the
    # result, esp. squandering a win). Enumerate the meaningful band changes.
    for _b, _a, _blurb in [
        ("Winning", "Losing", "You went from winning to losing in one move"),
        ("Winning", "Drawn", "You let a winning position slip to a draw"),
        ("Even", "Losing", "You went from equal to losing"),
        ("Even", "Winning", "Your move swung an equal position to winning"),
        ("Losing", "Drawn", "You salvaged a draw from a losing position"),
        ("Losing", "Winning", "You turned a losing position into a win"),
    ]:
        add(f"{_b} → {_a}", _blurb)
    # Blunder severity (descriptive: how the eval was lost, not what the mistake was).
    add("Sharp Blunder", "One move decisively swung the result")
    add("Slow Bleed", "You gave up a small edge from a balanced position")
    add("Only Move", "There was only one good move here")
    add("Best Move (deep analysis)", "Deep analysis confirms your move was best")
    for ph in ["Opening", "Middlegame", "Endgame"]:
        add(ph, f"{ph} phase")

    # FAMILY roll-up (grouping only — NOT emitted chips). Piece-specific tags roll up to a concept
    # parent via tagger.family_of(). Consumers that want to GROUP (SAE-matching, analytics, "how often
    # does this player hang material" across all piece types) read `families`; the product still shows
    # the specific chip. Derive the member lists by asking family_of() over the concrete tags we built,
    # so this stays in sync with the tagger's single source of truth automatically. (Sam, 2026-07-12.)
    families = {}
    for label in tags:
        parent = family_of(label)
        if parent != label:                     # label is a MEMBER of a coarser family
            families.setdefault(parent, {"members": [], "blurb": _FAMILY_BLURB.get(parent, parent)})
            families[parent]["members"].append(label)

    return {"categories": {c: {} for c in CATEGORIES}, "tags": tags, "families": families}


def main():
    tax = build_taxonomy()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(tax, f, indent=2)
    print(f"wrote {len(tax['tags'])} tags across {len(tax['categories'])} categories, "
          f"{len(tax['families'])} families -> {OUT}")
    for fam, d in tax["families"].items():
        print(f"    family {fam}: {len(d['members'])} members")


if __name__ == "__main__":
    main()
