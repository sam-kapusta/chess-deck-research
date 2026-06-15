#!/usr/bin/env python3
"""Re-map existing mistake_tags.json onto the proposed 10-category scheme and report
the cluster-size distribution. Answers: are the 10 buckets even, or are some too big?

10 categories (Sam, 2026-06-14):
  Hung Piece, Missed Capture, Missed Tactic, Missed Mate, Allowed Tactic,
  Calculation, Trading, Position, King Safety, Endgame
"""
import json, sys, re
from collections import Counter, defaultdict

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "output/mistake_tags.json"

# ── label → category. Direction-aware where a label exists in both directions. ──
# Returns a category given (label, direction). None = drop (info/uncategorizable).
def categorize(label, direction):
    L = label

    # --- Mate (its own skill: mate vision). Word-boundary: "Mate" not "Material". ---
    if re.search(r"\bMate\b", L):
        if direction == "missed":
            return "Missed Mate"
        # Allowed *Mate / Back-Rank Mate / smothered etc = you let your king get mated
        return "King Safety"

    # --- Endgame-specific ---
    if any(k in L for k in ("Promotion", "Underpromotion", "Passed Pawn", "Opposition",
                            "King Activity", "Pawn Race", "Rook Behind Passer", "En Passant")):
        return "Endgame"

    # --- King safety ---
    if any(k in L for k in ("Exposed King", "Kingside Attack", "Queenside Attack",
                            "Castling", "f2/f7 Attack", "Pawn Move Exposed King")):
        return "King Safety"

    # --- Hung material (you dropped a piece) ---
    if L in ("Hung Material",) or L.startswith("Hung "):
        return "Hung Piece"
    if L == "Allowed Hanging Piece":          # you left a piece hanging for them to take
        return "Hung Piece"
    if L == "Failed Hanging Piece":           # you took a "free" piece that wasn't free
        return "Calculation"

    # --- Missed free material (opponent gave you something) ---
    if L.startswith("Missed Free") or L.startswith("Missed Winning Capture") \
       or L == "Missed Hanging Piece" or L == "Missed Capture of Defender" \
       or L == "Missed Capture (Pawn)":
        return "Missed Capture"

    # --- Trading (exchange decisions) ---
    if L.startswith("Missed Exchange") or "Exchange" in L or L == "Missed Pawn Trade" \
       or L == "Premature Trade":
        return "Trading"

    # --- Calculation (you saw it, miscounted / wrong execution) ---
    if L in ("Wrong Move Order", "Captured With Wrong Piece", "Bad Capture", "Wrong Capture",
             "Lost Material to Combination") or L.startswith("Failed "):
        return "Calculation"

    # --- Positional ---
    if any(k in L for k in ("Advanced Pawn", "Isolated Pawn", "Doubled Pawn", "Backward Pawn",
                            "Outpost", "Open File", "Piece Activation", "Prophylaxis", "Pawn Break")):
        return "Position"
    if L == "Allowed Capture of Defender":    # they removed your defender = you allowed it
        return "Allowed Tactic"

    # --- Tactical motifs: split by direction (missed = find it / allowed = prevent it) ---
    TACTIC = ("Fork", "Pin", "Skewer", "Discovered Attack", "Deflection", "Attraction",
              "Clearance", "Interference", "Zwischenzug", "X-Ray", "Trapped",
              "Sacrifice", "Combination", "Double Check")
    if any(t in L for t in TACTIC):
        if direction == "missed":
            return "Missed Tactic"
        return "Allowed Tactic"   # allowed / failed

    return None  # uncategorized — surface it


def main():
    d = json.load(open(CORPUS))
    cat_counts = Counter()           # category -> # of (position, tag) hits
    cat_positions = defaultdict(set) # category -> set of position indices (a position can hit several)
    uncategorized = Counter()
    label_by_cat = defaultdict(Counter)

    for i, r in enumerate(d):
        for t in r.get("tags", []):
            if t.get("direction") == "info":
                continue
            cat = categorize(t["label"], t.get("direction", ""))
            if cat is None:
                uncategorized[t["label"]] += 1
                continue
            cat_counts[cat] += 1
            cat_positions[cat].add(i)
            label_by_cat[cat][t["label"]] += 1

    total_hits = sum(cat_counts.values())
    print(f"Corpus: {len(d)} positions, {total_hits} non-info tag hits\n")
    print(f"{'CATEGORY':<16}{'hits':>8}{'% hits':>8}{'positions':>11}{'% corpus':>10}")
    print("-" * 53)
    for cat, c in cat_counts.most_common():
        npos = len(cat_positions[cat])
        print(f"{cat:<16}{c:>8}{100*c/total_hits:>7.1f}%{npos:>11}{100*npos/len(d):>9.1f}%")

    print("\nTop labels per category:")
    for cat, _ in cat_counts.most_common():
        top = ", ".join(f"{lab}({n})" for lab, n in label_by_cat[cat].most_common(4))
        print(f"  {cat}: {top}")

    if uncategorized:
        print("\n⚠️  UNCATEGORIZED labels:")
        for lab, n in uncategorized.most_common():
            print(f"    {n:6d}  {lab}")


if __name__ == "__main__":
    main()
