"""Tests for consolidate_chips.py — lightweight rule regression coverage.

Run: python3 scripts/labeling/test_consolidate_chips.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consolidate_chips import (  # noqa: E402
    BY_CODE,
    CANONICAL,
    CATEGORY_FALLBACK,
    consolidate,
    match_chip,
)


def _mk_feature(chip="", label="", sub_patterns=None, category=""):
    """Helper for building test feature records."""
    return {
        "feature_id": 1,
        "label": label,
        "chip": chip,
        "category": category,
        "description": "",
        "sub_patterns": sub_patterns or [],
        "confidence": "high",
        "fire_rate": 0.01,
        "max_strength": 3.0,
        "mean_strength": 2.0,
        "examples": [],
    }


def test_canonical_has_26_entries():
    """The canonical taxonomy must stay at 26 subcategories to match production."""
    assert len(CANONICAL) == 26, f"Expected 26 canonical subcategories, got {len(CANONICAL)}"
    assert len(BY_CODE) == 26


def test_all_categories_have_fallback():
    """Every 2048_k64 `category` must map to some canonical subcategory."""
    expected_cats = {
        "endgame_technique", "tactical_oversight", "piece_safety",
        "mate_awareness", "calculation", "king_safety", "pawn_play",
    }
    assert expected_cats.issubset(CATEGORY_FALLBACK.keys()), \
        f"Missing fallbacks for: {expected_cats - CATEGORY_FALLBACK.keys()}"


def test_all_fallback_codes_are_valid():
    """Every fallback code must exist in the canonical taxonomy."""
    for cat, code in CATEGORY_FALLBACK.items():
        assert code in BY_CODE, f"Fallback for '{cat}' points to unknown code {code}"


def test_chip_match_hanging_material():
    """'hanging piece' chip should map to 1.1."""
    assert match_chip("Hanging Piece", "Moving to attacked square", []) == "1.1"
    assert match_chip("en prise", "", []) == "1.1"


def test_chip_match_tunnel_vision():
    """'tunnel_vision' chip should map to 7.1 (cognitive, not tactical)."""
    assert match_chip("Tunnel Vision", "tactical oversight", []) == "7.1"


def test_defender_removal_override():
    """Features with defender-removal label should promote to 1.4 even if chip says 'hanging'."""
    code = match_chip("Hanging Piece", "abandoning key defender", [])
    assert code == "1.4", f"Defender override failed, got {code}"


def test_check_pattern_does_not_match_checkmate():
    """2.3's `\\bcheck\\b` rule must not match 'checkmate' (which belongs to mate_awareness)."""
    code = match_chip("", "missing a checkmate in two", [])
    # Should NOT be 2.3 (Checks & Tempos); no explicit mate rule means it falls through.
    # The negative lookahead `(?!mate)` prevents false 2.3 match.
    assert code != "2.3", f"'checkmate' incorrectly matched 2.3 (Checks & Tempos), got {code}"


def test_missed_mate_goes_to_3_1():
    """'missed_mate' chip should map to 3.1 Immediate Lethality."""
    assert match_chip("Missed Mate", "", []) == "3.1"
    assert match_chip("missed_forced_mate", "", []) == "3.1"


def test_pawn_race_goes_to_4_1():
    assert match_chip("Pawn Race", "", []) == "4.1"


def test_consolidate_produces_production_schema():
    """Output features must have all fields the production Lambda expects."""
    required = {
        "feature_id", "label", "chip", "description",
        "domain_id", "domain", "subcategory_code", "coaching_category",
        "fire_rate", "max_strength", "coaching_useful",
    }
    labels = {"0": _mk_feature(chip="Hanging Piece", label="undefended knight")}
    out, _ = consolidate(labels)
    entry = out["0"]
    missing = required - set(entry.keys())
    assert not missing, f"Output missing required fields: {missing}"
    assert entry["subcategory_code"] == "1.1"
    assert entry["domain"] == "Tactical Blindness & Vision"
    assert entry["coaching_category"] == "Hanging Material"


def test_consolidate_uses_category_fallback():
    """Features with no rule match but known category should use fallback, not stay unmapped."""
    labels = {"0": _mk_feature(chip="Some Unseen Chip", label="Nothing Matches", category="calculation")}
    out, stats = consolidate(labels)
    assert stats["matched_by_fallback"] == 1
    assert stats["unmapped"] == 0
    assert out["0"]["subcategory_code"] == CATEGORY_FALLBACK["calculation"]


def test_consolidate_unknown_category_is_unmapped():
    """Feature with unknown category and no chip match → unmapped (0.0)."""
    labels = {"0": _mk_feature(chip="???", label="???", category="not_a_real_category")}
    out, stats = consolidate(labels)
    assert stats["unmapped"] == 1
    assert "0" not in out or out["0"]["subcategory_code"] == "0.0"


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"  ERROR {name}:\n{traceback.format_exc()}")

    print(f"\n{passed} passed, {len(failed)} failed (of {len(tests)} tests)")
    sys.exit(0 if not failed else 1)
