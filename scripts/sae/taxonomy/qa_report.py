"""Quality gate: compare rebuilt taxonomy to old labels. No LLM."""
import argparse
import json
import re
from collections import Counter

GENERIC = re.compile(r"(ignor|miss|wast).{0,20}(tactic|tactics|tempo|urgency|crisis)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--old-labels", required=True)
    args = ap.parse_args()
    t = json.load(open(args.taxonomy))
    old = json.load(open(args.old_labels))
    feats = t["features"]

    old_generic = sum(1 for f in feats if f in old and GENERIC.search(old[f].get("chip", "")))
    new_generic = sum(1 for f, v in feats.items() if GENERIC.search(v.get("chip", "")))
    print(f"Generic chips — OLD: {old_generic}  NEW: {new_generic}  (lower is better)")

    cat_counts = Counter(v["category"] for v in feats.values())
    sized = [(c, n) for c, n in cat_counts.items() if c]
    biggest = max(sized, key=lambda x: x[1]) if sized else (None, 0)
    print(f"Largest category: {biggest[0]} = {biggest[1]} features "
          f"({biggest[1] / len(feats) * 100:.0f}%) — should be <25%")
    print(f"Categories used: {len(sized)} / {len(t['categories'])}")
    print(f"Unassigned (None): {cat_counts.get(None, 0)}")
    empty = [c["id"] for c in t["categories"] if cat_counts.get(c["id"], 0) == 0]
    if empty:
        print(f"NOTE empty categories: {empty}")

    print("\nFull distribution:")
    names = {c["id"]: c["name"] for c in t["categories"]}
    for cid, n in sorted(sized, key=lambda x: -x[1]):
        print(f"  {n:>4}  {names.get(cid, cid)}")

    fail = []
    if old_generic > 0 and new_generic > old_generic * 0.2:
        fail.append(f"too many generic chips remain ({new_generic} vs old {old_generic})")
    if biggest[1] / len(feats) > 0.25:
        fail.append(f"junk-drawer category ({biggest[0]} {biggest[1]})")
    if cat_counts.get(None, 0) > len(feats) * 0.05:
        fail.append(f"too many unassigned ({cat_counts.get(None, 0)})")
    if fail:
        print("\nQA FAILED:", "; ".join(fail))
        raise SystemExit(1)
    print("\nQA PASSED")


if __name__ == "__main__":
    main()
