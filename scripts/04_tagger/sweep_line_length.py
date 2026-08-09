#!/usr/bin/env python3
"""How much does tag output depend on how many plies of engine line we hand the tagger?

Committed because the answer drove a design decision (TAGGER_LINE_PLIES = 8, one number for both
directions in all three producers) and quoted numbers with no reproducible script rot immediately.

The finding: tag output is strongly length-sensitive and does NOT plateau by 6 plies. Whole motif
families are invisible on short lines. That is why the old setup — best line capped at 12, refutation
hardcoded to 6 in the worker and uncapped in the frontend — produced different tags for the same position
depending on which direction a detector ran in and who built the payload.

⚠️ This script cannot measure above the corpus's own cap. `fifa_enrich.json` was generated at 6 plies, so
truncating it to 8/10/12 returns output identical to 6. That is not a plateau — it is a blind spot, and
it is the same blind spot that hid a production false positive from a 56,950-position audit. Regenerate
the corpus at the new cap before trusting any number above it.

Usage:
    python3 sweep_line_length.py --enrich /tmp/fifa_enrich.json
    python3 sweep_line_length.py --enrich ... --plies 2,4,6,8 --limit 3000
    python3 sweep_line_length.py --enrich ... --label "Hung Material"   # one label's curve
"""
import argparse
import collections
import json
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake, _eval_to_cp      # noqa: E402
from tagger import tag_mistake_full           # noqa: E402

PHASE_TAGS = {"Opening", "Middlegame", "Endgame"}


def build(fen, uci, entry, plies):
    """A Mistake with BOTH lines truncated to `plies`, mirroring what a producer would send."""
    best = (entry.get("top_3_best") or [{}])[0].get("line", "").split()
    refut = (entry.get("top_3_refutations") or [{}])[0].get("line", "").split()
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=entry.get("best_uci", ""),
        best_line_san=best[:plies], refutation_san=refut[:plies],
        eval_before=_eval_to_cp(entry.get("eval_before")),
        eval_after=_eval_to_cp(entry.get("eval_after")),
        cp_loss=int(entry.get("cp_loss", 0) or 0),
        mover=chess.Board(fen).turn,
        played_san=entry.get("played_san", ""), best_san=entry.get("best_san", ""),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enrich", required=True)
    p.add_argument("--plies", default="2,4,6,8")
    p.add_argument("--limit", type=int, default=6000)
    p.add_argument("--label", default=None, help="print one label's curve instead of the summary")
    args = p.parse_args()

    plies = [int(x) for x in args.plies.split(",")]
    data = json.load(open(args.enrich))

    totals = {n: 0 for n in plies}
    per_label = {n: collections.Counter() for n in plies}
    seen = 0
    corpus_max = 0

    for key in list(data.keys())[:args.limit]:
        if "|" not in key:
            continue
        fen, uci = key.rsplit("|", 1)
        entry = data[key]
        refut = (entry.get("top_3_refutations") or [{}])[0].get("line", "").split()
        corpus_max = max(corpus_max, len(refut))
        if not refut:
            continue
        seen += 1
        for n in plies:
            try:
                tags = tag_mistake_full(build(fen, uci, entry, n), with_maia=False)["tags"]
            except Exception:
                continue
            teach = [t["label"] for t in tags
                     if t["direction"] != "info" and t["label"] not in PHASE_TAGS]
            totals[n] += len(teach)
            for lab in teach:
                per_label[n][lab] += 1

    if args.label:
        print(f"{args.label}:")
        for n in plies:
            print(f"  {n:2d} plies: {per_label[n][args.label]}")
        return

    print(f"positions: {seen}   longest refutation in this corpus: {corpus_max} plies")
    if corpus_max <= max(plies):
        print(f"⚠️  corpus caps at {corpus_max} — any sweep value above that is MEANINGLESS "
              f"(identical output, not a plateau)")
    print()
    print(f"{'plies':<8}{'teachable tags':>16}{'per position':>15}")
    for n in plies:
        print(f"{n:<8}{totals[n]:>16}{totals[n]/max(1, seen):>15.2f}")

    lo, hi = min(plies), max(plies)
    print()
    print(f"labels that GAIN the most from {lo} -> {hi} plies (short lines delete these):")
    gains = [(per_label[hi][l] - per_label[lo][l], l) for l in per_label[hi]]
    for delta, lab in sorted(gains, reverse=True)[:12]:
        if delta > 0:
            print(f"   {lab:<34} {lo}ply={per_label[lo][lab]:<6} {hi}ply={per_label[hi][lab]:<6} +{delta}")


if __name__ == "__main__":
    main()
