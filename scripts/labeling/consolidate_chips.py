"""
Consolidate 2048_k64 chip names to canonical 26 subcategories (production taxonomy).

Input:  labels_2048_k64.json        — 2042 features, 645 chips, 7 flat categories, 139 subcategories
Output: labels_2048_k64_canonical.json — same features with production schema:
          domain, subcategory_code (e.g. "1.1"), coaching_category, chip

Mapping is rule-based (chip substring matching). Features that don't match any rule
are flagged with subcategory_code="0.0" for human review.

See plans/chip-consolidation-2048-k64.md for mapping rationale.

Usage:
  python3 consolidate_chips.py --dry-run     # Print distribution, don't write
  python3 consolidate_chips.py                # Write canonical output
  python3 consolidate_chips.py --show-unmapped  # List unmapped chips for review
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output" / "labels_2048_k64.json"
OUTPUT = ROOT / "output" / "labels_2048_k64_canonical.json"


# Canonical 26 subcategories from realgames_512_k8_v1 taxonomy.
# Ordered as (domain_id, subcategory_code, domain, coaching_category).
CANONICAL = [
    (1, "1.1", "Tactical Blindness & Vision",     "Hanging Material"),
    (1, "1.2", "Tactical Blindness & Vision",     "Square/Line Safety"),
    (1, "1.3", "Tactical Blindness & Vision",     "Vision Failures"),
    (1, "1.4", "Tactical Blindness & Vision",     "Defender Removal"),
    (2, "2.1", "Calculation & Forcing Moves",     "Intermediate Moves (Zwischenzug)"),
    (2, "2.2", "Calculation & Forcing Moves",     "Forcing Move Oversight"),
    (2, "2.3", "Calculation & Forcing Moves",     "Checks & Tempos"),
    (2, "2.4", "Calculation & Forcing Moves",     "Failed Calculation"),
    (3, "3.1", "King Safety & Mating Mechanics",  "Immediate Lethality"),
    (3, "3.2", "King Safety & Mating Mechanics",  "Mating Nets & Batteries"),
    (3, "3.3", "King Safety & Mating Mechanics",  "King Exposure"),
    (3, "3.4", "King Safety & Mating Mechanics",  "Back-Rank & Traps"),
    (4, "4.1", "Endgame: Pawn & King",            "Pawn Races"),
    (4, "4.2", "Endgame: Pawn & King",            "The Opposition"),
    (4, "4.3", "Endgame: Pawn & King",            "King Activity"),
    (5, "5.1", "Endgame: Piece Coordination",     "Basic Mating Technique"),
    (5, "5.2", "Endgame: Piece Coordination",     "Piece vs Piece Endgames"),
    (5, "5.3", "Endgame: Piece Coordination",     "Simplification Errors"),
    (6, "6.1", "Game State Management",           "Conversion Inefficiency"),
    (6, "6.2", "Game State Management",           "Squandering the Advantage"),
    (6, "6.3", "Game State Management",           "Resilience & Defense"),
    (6, "6.4", "Game State Management",           "Drawing Resources"),
    (7, "7.1", "Process & Cognitive Errors",      "Tunnel Vision"),
    (7, "7.2", "Process & Cognitive Errors",      "Autopilot / Natural Moves"),
    (7, "7.3", "Process & Cognitive Errors",      "Greed vs Safety"),
]

BY_CODE = {code: (dom_id, domain, coaching) for (dom_id, code, domain, coaching) in CANONICAL}


# Rule patterns: (subcategory_code, regex on lowercased chip OR sub_pattern text).
# First matching rule wins. Order matters — more specific patterns before general.
RULES = [
    # 7.x Process & Cognitive Errors — check FIRST so tactical-sounding chips in these
    # go to the cognitive domain rather than tactical.
    ("7.1", r"\btunnel[_ ]vision\b"),
    ("7.2", r"\bautopilot\b|\bpassive play\b|\bpassive over forcing\b|\bnatural move\b"),
    ("7.3", r"\bgreedy[_ ]capture\b|\bgreed\b|\bbad recapture\b"),

    # 3.x King Safety & Mating Mechanics
    ("3.1", r"\bmissed[_ ](forced[_ ])?mate\b|\bmissed mate-in|\bignoring mate\b|\bmissed_mate\b|\bmating material\b"),
    ("3.2", r"\bbattery[_ ]mate\b|\bmating net\b"),
    ("3.4", r"\bback[_ ]rank\b"),
    ("3.3", r"\bking[_ ]neglect\b|\bking safety\b|\bking[_ ]misplacement\b|\bking misplaced\b|\bking drift\b|\bking exposure\b"),

    # 4.x Endgame: Pawn & King
    ("4.1", r"\bpawn[_ ]race\b|\bsquare[_ ]rule\b|\bpawn[_ ]square\b|\bking too slow\b|\bking distance\b|\bpawn[_ ]tempo\b|\bpassed[_ ]pawn\b"),
    ("4.2", r"\bopposition\b|\bking[_ ]first\b|\bking leads\b"),
    ("4.3", r"\bking[_ ]passivity\b|\bking[_ ]restriction\b|\bking[_ ]confinement\b|\bking[_ ]boxing\b|\bking escape\b|\bpassive king\b|\bking[_ ]passive\b|\bking placement\b|\bking tempo\b|\bendgame activity\b|\bendgame passivity\b|\bendgame king\b|\bking[_ ]misplace"),

    # 5.x Endgame: Piece Coordination
    ("5.1", r"\bbn[_ ]mate\b|\bb\+n mate\b|\bb\+n endgame\b|\bbox technique\b|\bbox release\b|\bk\+p endgame\b|\bk\+p technique\b|\bk\+p tempo\b|\bk\+p misjudge\b|\blost k\+p\b"),
    ("5.2", r"\bq vs r\b|\bq vs n\b|\brook endgame\b|\bkr endgame\b|\brook passivity\b|\brook cutoff\b|\bwrong bishop\b"),
    ("5.3", r"\b(bad|false|wrong)[_ ]simplification\b|\bbad[_ ]trade\b|\bbad trade-off\b|\bwon[→ ]draw\b|\bdraw blunder\b|\bdead draw\b|\bendgame tech(nique)?\b"),

    # 6.x Game State Management
    ("6.1", r"\bendgame[_ ]conversion\b|\bendgame precision\b|\bslow win\b|\bendgame technique\b"),
    ("6.3", r"\bpassive[_ ]retreat\b|\breactive retreat\b|\bresilience\b|\bdefensive\b|\bpassive endgame\b"),
    ("6.4", r"\bdrawing resources\b|\balready lost\b|\blost endgame\b|\blost position\b"),
    ("6.2", r"\bsquander\b|\btempo[_ ]waste\b|\btempo waste\b|\bwaste(d)?\b"),  # weaker — later rule

    # 2.x Calculation & Forcing Moves
    ("2.1", r"\bzwischenzug\b|\bintermediate move\b"),
    ("2.2", r"\bmissed[_ ]forcing\b|\bmissed forcing move\b"),
    ("2.3", r"\bmissed[_ ]capture\b|\bmissed urgency\b|\btempo[_ ]error\b|\btempo loss\b|\bendgame tempo\b|\bcheck(s|ing)?\b(?![_ ]?mate)"),
    ("2.4", r"\bwrong priority\b|\bwrong order\b|\bmove order\b|\bcalculation\b|\bindirect play\b|\bmissed win\b|\bfork blindness\b|\bknight[_ ]fork\b"),

    # 1.x Tactical Blindness & Vision (check after 7.x and 3.x since tactical has overlap)
    # 1.4 before 1.1 since "abandoning defender" features often have "hanging" chip.
    ("1.4", r"\bdefender removal\b|\bremoving defender\b|\babandon(ed|ing)?[_ ]guard\b|\bsole defender\b|\babandon(ing|ed) (defense|key|critical)\b|\bunblocking\b|\bmoving.*defender\b"),
    ("1.1", r"\bhanging[_ ]piece\b|\bhanging center\b|\ben[_ ]prise\b|\bpiece blindness\b|\bfree piece\b|\ben prise miss\b|\ben prise blind\b|\bpiece safety\b"),
    ("1.2", r"\bunsafe[_ ]square\b|\bunsafe landing\b|\bwrong[_ ]square\b|\bbad square\b|\bsquare safety\b"),
    ("1.3", r"\btactical[_ ]blindness\b|\bmissed[_ ]tactic\b|\bthreat[_ ]blindness\b|\bendgame tactics\b|\bpassive retreat\b"),
]

# Category-level fallback when no chip rule matches.
CATEGORY_FALLBACK = {
    "endgame_technique": "4.3",     # default to King Activity
    "tactical_oversight": "1.3",     # default to Vision Failures
    "piece_safety":       "1.1",     # default to Hanging Material
    "mate_awareness":     "3.1",     # default to Immediate Lethality
    "calculation":        "2.4",     # default to Failed Calculation
    "king_safety":        "3.3",     # default to King Exposure
    "pawn_play":          "4.1",     # default to Pawn Races
}


DEFENDER_PATTERN = re.compile(
    r"\bdefender removal\b|\bremoving defender\b|\babandon(ed|ing)?[_ ]guard\b|"
    r"\bsole defender\b|\babandon(ing|ed) (defense|key|critical)\b|\bunblocking\b|\bmoving.*defender\b"
)


def match_chip(chip: str, label: str, sub_patterns: list[str]) -> str | None:
    """
    Return subcategory_code if any rule matches, else None.

    Two-pass: try chip alone first (most specific), then fall back to label.
    Special case: features with "hanging piece" chip but defender-removal labels
    get promoted to 1.4 (Defender Removal) — that's the more specific pattern.
    """
    chip_l = chip.lower()
    label_l = label.lower()

    # Override: defender-removal language in label promotes to 1.4 regardless of chip
    if DEFENDER_PATTERN.search(label_l):
        return "1.4"

    # Pass 1: chip only (most specific signal)
    for code, pattern in RULES:
        if re.search(pattern, chip_l):
            return code

    # Pass 2: label
    for code, pattern in RULES:
        if re.search(pattern, label_l):
            return code

    return None


def consolidate(labels: dict, show_unmapped: bool = False) -> tuple[dict, dict]:
    """
    Returns (canonical_labels, stats).

    Canonical label schema matches production:
      feature_id, label, chip, description, sub_patterns, confidence,
      fire_rate, max_strength, mean_strength, examples,
      domain_id, domain, subcategory_code, coaching_category, coaching_useful
    """
    out = {}
    stats = {
        "total": 0,
        "matched_by_rule": 0,
        "matched_by_fallback": 0,
        "unmapped": 0,
        "subcategory_counts": Counter(),
        "unmapped_chips": Counter(),
    }

    for fid, d in labels.items():
        if not isinstance(d, dict):
            continue
        stats["total"] += 1

        chip = d.get("chip", "")
        label = d.get("label", "")
        sub_patterns = d.get("sub_patterns", []) or []
        category = d.get("category", "")

        code = match_chip(chip, label, sub_patterns)
        if code:
            stats["matched_by_rule"] += 1
        elif category in CATEGORY_FALLBACK:
            code = CATEGORY_FALLBACK[category]
            stats["matched_by_fallback"] += 1
        else:
            stats["unmapped"] += 1
            stats["unmapped_chips"][chip] += 1
            code = "0.0"  # sentinel for review

        if code == "0.0":
            domain_id, domain, coaching = 0, "UNMAPPED", "Needs Review"
        else:
            domain_id, domain, coaching = BY_CODE[code]

        out[fid] = {
            "feature_id": d.get("feature_id", int(fid)),
            "label": label,
            "chip": chip,
            "description": d.get("description", ""),
            "sub_patterns": sub_patterns,
            "confidence": d.get("confidence", "medium"),
            "fire_rate": d.get("fire_rate"),
            "max_strength": d.get("max_strength"),
            "mean_strength": d.get("mean_strength"),
            "examples": d.get("examples", []),
            # Production schema additions:
            "domain_id": domain_id,
            "domain": domain,
            "subcategory_code": code,
            "coaching_category": coaching,
            "coaching_question": "",  # to be filled by human review pass
            "coaching_useful": code != "0.0",
        }

        stats["subcategory_counts"][code] += 1

    if show_unmapped and stats["unmapped"]:
        print("\n=== Unmapped chips (need rule addition or human review) ===")
        for chip, n in stats["unmapped_chips"].most_common(50):
            print(f"  {n:4d}  {chip!r}")

    return out, stats


def print_stats(stats: dict) -> None:
    print(f"\n=== Consolidation stats ===")
    print(f"Total features:     {stats['total']}")
    print(f"Matched by rule:    {stats['matched_by_rule']}  ({stats['matched_by_rule']/stats['total']:.1%})")
    print(f"Matched by fallback:{stats['matched_by_fallback']}  ({stats['matched_by_fallback']/stats['total']:.1%})")
    print(f"Unmapped:           {stats['unmapped']}  ({stats['unmapped']/stats['total']:.1%})")
    print()
    print("=== Subcategory distribution ===")
    for (dom_id, code, domain, coaching) in CANONICAL:
        n = stats["subcategory_counts"].get(code, 0)
        print(f"  {code}  {n:4d}  {domain:32s}  {coaching}")
    n_unmapped = stats["subcategory_counts"].get("0.0", 0)
    if n_unmapped:
        print(f"  0.0  {n_unmapped:4d}  UNMAPPED (needs review)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't write output file")
    parser.add_argument("--show-unmapped", action="store_true", help="List unmapped chips")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    print(f"Reading {args.input}")
    with args.input.open() as f:
        labels = json.load(f)

    canonical, stats = consolidate(labels, show_unmapped=args.show_unmapped)
    print_stats(stats)

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(canonical, f, indent=2)
        print(f"\nWrote {args.output} ({len(canonical)} features)")
    else:
        print("\n(dry-run — no output written)")


if __name__ == "__main__":
    main()
