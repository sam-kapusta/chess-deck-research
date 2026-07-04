import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import build_mistake_taxonomy as B
from tagger import categorize


def test_taxonomy_shape_and_coverage():
    tax = B.build_taxonomy()
    assert set(tax.keys()) == {"categories", "tags"}
    # 10-category drill scheme + Meta/Other (Sam, 2026-06-14)
    for cat in ["Hung Piece", "Missed Capture", "Missed Tactic", "Missed Mate", "Allowed Tactic",
                "Calculation", "Trading", "Position", "King Safety", "Endgame"]:
        assert cat in tax["categories"], f"missing category {cat}"
    assert len(tax["tags"]) >= 100
    for label, entry in tax["tags"].items():
        assert entry["category"] == categorize(label), f"{label}: {entry['category']} != {categorize(label)}"
        assert entry["blurb"].strip(), f"{label} has empty blurb"
        assert entry["category"] in tax["categories"]
    for known in ["Missed Fork", "Allowed Pin (to Queen)", "Hung Material",
                  "Bishop Endgame (Same Color)", "Missed Free Knight", "Missed Queen Exchange",
                  "Missed Tempo Push", "Missed Prophylaxis"]:
        assert known in tax["tags"], f"missing {known}"


def test_known_categories():
    tax = B.build_taxonomy()
    # Tactics split by direction (find it vs prevent it).
    assert tax["tags"]["Missed Fork"]["category"] == "Missed Tactic"
    assert tax["tags"]["Allowed Fork"]["category"] == "Allowed Tactic"
    # Material splits into the SKILL: hung (you dropped it) vs missed-free (you didn't take it) vs trade.
    assert tax["tags"]["Hung Material"]["category"] == "Hung Piece"
    assert tax["tags"]["Missed Free Pawn"]["category"] == "Missed Capture"
    assert tax["tags"]["Missed Queen Exchange"]["category"] == "Trading"
    assert tax["tags"]["Missed Pawn Trade"]["category"] == "Trading"
    # Mate vision is its own skill when missed; allowing mate is king safety.
    assert tax["tags"]["Missed Mate"]["category"] == "Missed Mate"
    assert tax["tags"]["Allowed Back-Rank Mate"]["category"] == "King Safety"
    # Calculation = saw-it-miscounted. Greedy Capture replaced the 5 removed catch-alls (GH #29).
    assert tax["tags"]["Greedy Capture"]["category"] == "Calculation"
    # Endgame-type context tags.
    assert tax["tags"]["Bishop Endgame (Same Color)"]["category"] == "Endgame"
