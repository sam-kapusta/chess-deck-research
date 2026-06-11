import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import build_mistake_taxonomy as B
from tagger import categorize


def test_taxonomy_shape_and_coverage():
    tax = B.build_taxonomy()
    assert set(tax.keys()) == {"categories", "tags"}
    assert "Tactical" in tax["categories"]
    assert "Material" in tax["categories"]
    assert len(tax["tags"]) >= 100
    for label, entry in tax["tags"].items():
        assert entry["category"] == categorize(label), f"{label}: {entry['category']} != {categorize(label)}"
        assert entry["blurb"].strip(), f"{label} has empty blurb"
        assert entry["category"] in tax["categories"]
    for known in ["Missed Fork", "Allowed Pin (to Queen)", "Hung Material",
                  "Bishop Endgame (Same Color)", "Missed Free Knight", "Missed Queen Exchange"]:
        assert known in tax["tags"], f"missing {known}"
    # the generic 'Missed Capture' was removed (redundant with the piece-specific tags)
    assert "Missed Capture" not in tax["tags"], "generic 'Missed Capture' should be gone"


def test_known_categories():
    tax = B.build_taxonomy()
    assert tax["tags"]["Missed Fork"]["category"] == "Tactical"
    assert tax["tags"]["Hung Material"]["category"] == "Material"
    assert tax["tags"]["Bishop Endgame (Same Color)"]["category"] == "Endgame"
    # renamed capture tags lost the "capture" substring — they must still land in Material, not
    # fall through to Positional via "pawn"/"free". (Guards the categorize() keyword fix.)
    assert tax["tags"]["Missed Free Pawn"]["category"] == "Material"
    assert tax["tags"]["Missed Queen Exchange"]["category"] == "Material"
    assert tax["tags"]["Missed Pawn Trade"]["category"] == "Material"
