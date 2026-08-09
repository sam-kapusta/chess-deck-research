#!/usr/bin/env python3
"""Re-derive every judge disagreement ON THE BOARD, and kill the ones that don't hold.

The judge (judge_tags_agy.py) proposes; this disposes. It is the mandatory second stage, not a
convenience: on the first sweep the judge flagged `Allowed Deflection` with is_mate=true on a line that
demonstrably does not mate (its own prose said "trapping the king"), and separately overturned a real bug
report of mine. Both directions of error are live, so nothing reaches an issue without a board check.

Rules of engagement:
  * Only python-chess decides. The judge's booleans are hypotheses; the board is the fact.
  * Every ply must parse. An unparseable line means the evidence is unusable — report UNVERIFIABLE
    rather than guessing (a silent "no mate found" on a line that never replayed is how a broken fixture
    reads as a clean result).
  * A flag SURVIVES only if the board agrees with the judge AND the tagger really is missing it.

Usage:
    python3 verify_judge_flags.py --judged /tmp/judge_full.json
    python3 verify_judge_flags.py --judged ... --out /tmp/verified.json
"""
import argparse
import collections
import json

import chess


def replay(fen, played_uci, sans, apply_played):
    """Replay a line, returning (board, ok). ok=False if anything failed to parse/was illegal."""
    b = chess.Board(fen)
    if apply_played:
        try:
            mv = chess.Move.from_uci(played_uci)
            if mv not in b.legal_moves:
                return b, False
            b.push(mv)
        except Exception:
            return b, False
    for san in sans:
        try:
            b.push(b.parse_san(san))
        except Exception:
            return b, False
    return b, True


def verify(rec):
    """Return (verdict, note). verdict in {CONFIRMED, REFUTED, UNVERIFIABLE}."""
    direction = rec["direction"]
    # A MISSED tag is a claim about the best line; ALLOWED/HUNG about [played]+refutation.
    if direction == "missed":
        sans = (rec.get("best_line") or "").split()
        board, ok = replay(rec["fen"], None, sans, apply_played=False)
    else:
        sans = (rec.get("refutation") or "").split()
        played = rec["key"].rsplit("|", 1)[1]
        board, ok = replay(rec["fen"], played, sans, apply_played=True)
    if not ok or not sans:
        return "UNVERIFIABLE", "line did not replay cleanly"

    mates = board.is_checkmate()
    claims_mate = any("MATE" in d.upper() for d in rec["disagreements"])
    any_mate_tag = any("Mate" in l for l in rec.get("all_labels") or [])

    if claims_mate:
        if not mates:
            return "REFUTED", "judge said MATE; the line does not end in checkmate on the board"
        if any_mate_tag:
            return "REFUTED", f"line does mate, but a mate tag already fired: {rec['all_labels']}"
        return "CONFIRMED", "line ends in checkmate and NO mate tag fired"

    if any("already lost" in d for d in rec["disagreements"]):
        # Sacrifice-soundness claim: the board can't settle 'lost', but eval can, and that is what
        # #100 uses. Leave it to the sizing script rather than guessing here.
        return "UNVERIFIABLE", "soundness claim — settle with eval, see #100"

    if any("material is not the point" in d for d in rec["disagreements"]):
        return "UNVERIFIABLE", "judgement call on emphasis, not a board fact"

    return "UNVERIFIABLE", "no board-checkable assertion in this disagreement"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judged", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    d = json.load(open(args.judged))
    flagged = [r for r in d["results"] if r.get("disagreements")]
    print(f"judged pairs: {len(d['results'])}   flagged: {len(flagged)}\n")

    out = []
    counts = collections.Counter()
    for r in flagged:
        verdict, note = verify(r)
        counts[verdict] += 1
        r = dict(r, verdict=verdict, verify_note=note)
        out.append(r)
        if verdict == "CONFIRMED":
            print(f"[CONFIRMED] {r['label']}  ({r['direction']})")
            print(f"    fen   {r['fen']}")
            print(f"    played {r['played']}  best {r['best']}")
            print(f"    line  {r['refutation'] if r['direction'] != 'missed' else r['best_line']}")
            print(f"    tags  {r.get('all_labels')}")
            print(f"    judge {r['judge']['primary_theme']}")
            print(f"    -> {note}\n")

    print("verdicts:", dict(counts))
    refuted = [r for r in out if r["verdict"] == "REFUTED"]
    if refuted:
        print(f"\nREFUTED ({len(refuted)}) — judge was wrong, do NOT file:")
        for r in refuted:
            print(f"  {r['label']:<34} {r['verify_note']}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
