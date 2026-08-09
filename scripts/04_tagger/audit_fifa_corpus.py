#!/usr/bin/env python3
"""Audit every tag label against the positions it fires on — the SAE-eval pattern, for the tagger.

Why: Sam finds a false-positive tag roughly every game he plays. The regression suite only covers 7 of
the 24 line detectors (mutation-swept 2026-08-08: silencing the other 17 leaves it green), and the old
19,362-moment `mistake_tags.json` was generated 2026-06-08 — it predates half the tags, so it can't
measure them. This runs the CURRENT tagger over `fifa_blitz/fifa_enrich.json` (56,950 positions with
real 6-ply refutation lines) and reports, per label:

    fires · rate · sole-tag rate · CONTRADICTIONS

The contradiction checks are the point. They are cheap board facts that must hold if the label is
honest — the same "check the definition, not the vibe" discipline that caught Greek Gift (a proper noun
whose real definition is the h7/h2 bishop sac) and hanging_piece (which read a captured square two
plies early, so a quiet pawn push looked like a capture). No LLM, no engine: pure python-chess over
lines we already have.

Structural mirror of the SAE side (scripts/evaluation/detection_scoring.py): there, a FEATURE is scored
by sampling positions where it fires and asking whether its LABEL describes them. Here a TAG is scored
by sampling positions where it fires and asking whether its DEFINITION holds on the board.

Usage (on chess-poc, from ~/SageMaker/tagger_run):
    python3 audit_fifa_corpus.py --enrich ../fifa_blitz/fifa_enrich.json --out ../fifa_tag_audit.json
    python3 audit_fifa_corpus.py --enrich ... --out ... --limit 2000        # smoke test
    python3 audit_fifa_corpus.py --enrich ... --out ... --label "Hung Material" --examples 5
"""
import argparse
import collections
import json
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake, _eval_to_cp          # noqa: E402
from tagger import tag_mistake_full               # noqa: E402

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def from_fifa_entry(fen, uci, e):
    """FIFA enrich schema -> Mistake.

    Differs from mistake.from_sf_entry in two ways, which is why this adapter exists rather than
    reusing it: FIFA stores lines as SPACE-JOINED SAN strings under `top_3_best[].line` /
    `top_3_refutations[].line`, where stockfish_data_v2 uses `top_lines[].moves` LISTS. Passing the
    wrong shape yields empty lines and every motif tag silently vanishes — the failure mode is a clean
    zero, not an error, so it must be asserted on (see --self-check).
    """
    b = chess.Board(fen)
    best = e.get("top_3_best") or []
    refut = e.get("top_3_refutations") or []
    best_line = (best[0].get("line") or "").split() if best else []
    refut_line = (refut[0].get("line") or "").split() if refut else []
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=e.get("best_uci", ""),
        best_line_san=best_line, refutation_san=refut_line,
        eval_before=_eval_to_cp(e.get("eval_before")), eval_after=_eval_to_cp(e.get("eval_after")),
        cp_loss=int(e.get("cp_loss", 0) or 0), mover=b.turn,
        played_san=e.get("played_san", ""), best_san=e.get("best_san", ""),
    )


def line_facts(m):
    """Board facts over the ALLOWED line ([played] + refutation) and the BEST line.

    Everything the contradiction checks need, computed once: did anyone capture, was there a check,
    what did each side lose, is the line long enough to support a multi-ply claim.
    """
    f = {"allowed_plies": 0, "best_plies": 0, "any_capture": False, "any_check": False,
         "mover_lost": 0, "opp_lost": 0, "net": 0, "ends_mate": False, "promo": False,
         "best_any_capture": False, "best_any_check": False, "best_promo": False,
         "best_ends_mate": False}
    b = chess.Board(m.fen_before)
    try:
        played = chess.Move.from_uci(m.played_uci)
        if played not in b.legal_moves:
            return None
    except Exception:
        return None
    b.push(played)
    mover, opp = m.mover, not m.mover
    for san in m.refutation_san:
        try:
            mv = b.parse_san(san)
        except Exception:
            break
        if b.is_capture(mv):
            f["any_capture"] = True
            victim = b.piece_at(mv.to_square)
            vt = victim.piece_type if victim else chess.PAWN     # en passant
            if b.turn == opp:
                f["mover_lost"] += VAL[vt]
            else:
                f["opp_lost"] += VAL[vt]
        if mv.promotion:
            f["promo"] = True
        b.push(mv)
        f["allowed_plies"] += 1
        if b.is_check():
            f["any_check"] = True
    f["ends_mate"] = b.is_checkmate()
    f["net"] = f["mover_lost"] - f["opp_lost"]
    # BEST-line facts, tracked SEPARATELY. A MISSED tag describes the best line, an ALLOWED/HUNG tag
    # describes the refutation — so a check/promotion/mate check must read the side it belongs to.
    # Conflating them was a bug in the first version of this script: "Missed Double Check" and
    # "Missed Promotion" were flagged as contradictions because the REFUTATION had no check/promo,
    # which says nothing about the best line the player actually missed.
    bb = chess.Board(m.fen_before)
    for san in m.best_line_san:
        try:
            mv = bb.parse_san(san)
        except Exception:
            break
        if bb.is_capture(mv):
            f["best_any_capture"] = True
        if mv.promotion:
            f["best_promo"] = True
        bb.push(mv)
        f["best_plies"] += 1
        if bb.is_check():
            f["best_any_check"] = True
    f["best_ends_mate"] = bb.is_checkmate()
    return f


# Each check answers "could this label possibly be true given the board?" A True return = CONTRADICTION.
# Deliberately conservative: only fire when the label's own definition is violated, never on taste.
#
# DIRECTION MATTERS. A "Missed X" tag is a claim about the BEST line (what the player could have played);
# an "Allowed X" / "Hung X" tag is a claim about the REFUTATION (what the opponent does to punish). Each
# check therefore picks its facts off the matching line — reading the wrong one manufactures fake
# contradictions, which is exactly what the first version of this script did.
def _missed(label):
    return label.startswith("Missed")


def _c_material_no_capture(label, f):
    """Any material-loss claim requires someone to actually capture something in the line."""
    return not (f["best_any_capture"] if _missed(label) else f["any_capture"])


def _c_material_no_net(label, f):
    """"You lost material" requires the mover to be down on net over the refutation.

    MISSED-direction labels are exempt: net material over the BEST line says nothing about what the
    player lost by not playing it.
    """
    if _missed(label):
        return False
    return f["net"] <= 0


def _c_mate_not_mate(label, f):
    """A mate claim requires the line to END in checkmate."""
    return not (f["best_ends_mate"] if _missed(label) else f["ends_mate"])


def _c_check_no_check(label, f):
    """A check-based motif (discovered/double check) requires a check in the line."""
    return not (f["best_any_check"] if _missed(label) else f["any_check"])


def _c_promo_no_promo(label, f):
    """A promotion claim requires a promotion move in the line."""
    return not (f["best_promo"] if _missed(label) else f["promo"])


def _c_too_short(label, f):
    """A multi-ply combination cannot be asserted from a 1-ply line."""
    return (f["best_plies"] if _missed(label) else f["allowed_plies"]) < 3


CHECKS = {
    "Hung Material":            [_c_material_no_capture, _c_material_no_net],
    "Hung Queen":               [_c_material_no_capture, _c_material_no_net],
    "Hung Rook":                [_c_material_no_capture, _c_material_no_net],
    "Hung Bishop":              [_c_material_no_capture, _c_material_no_net],
    "Hung Knight":              [_c_material_no_capture, _c_material_no_net],
    "Hung Pawn":                [_c_material_no_capture, _c_material_no_net],
    "Lost Queen in Exchange":   [_c_material_no_capture, _c_material_no_net],
    "Lost Rook in Exchange":    [_c_material_no_capture, _c_material_no_net],
    "Allowed Hanging Piece":    [_c_material_no_capture],
    "Allowed Discovered Check": [_c_check_no_check],
    "Allowed Double Check":     [_c_check_no_check],
    "Missed Discovered Check":  [_c_check_no_check],
    "Missed Double Check":      [_c_check_no_check],
    "Allowed Promotion":        [_c_promo_no_promo],
    "Missed Promotion":         [_c_promo_no_promo],
    "Allowed Underpromotion":   [_c_promo_no_promo],
    "Missed Underpromotion":    [_c_promo_no_promo],
    "Allowed Mate":             [_c_mate_not_mate],
    "Allowed Back-Rank Mate":   [_c_mate_not_mate],
    "Allowed Smothered Mate":   [_c_mate_not_mate],
    "Allowed Combination → Fork": [_c_too_short],
    "Missed Combination → Fork":  [_c_too_short],
}

INFO_NOISE = {"Opening", "Middlegame", "Endgame"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enrich", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--examples", type=int, default=3, help="contradiction examples to keep per label")
    p.add_argument("--label", default=None, help="print examples for one label and exit")
    args = p.parse_args()

    data = json.load(open(args.enrich))
    keys = list(data.keys())
    if args.limit:
        keys = keys[:args.limit]

    fires = collections.Counter()
    sole = collections.Counter()
    contra = collections.Counter()
    contra_why = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    # Denominators must exclude positions that CANNOT produce a tag, or "didn't fire" conflates
    # "detector declined" with "no line to look at". 1,531 FIFA positions have 1-ply refutations and 34
    # have none — those can't yield ALLOWED tags at all.
    eligible = 0
    skipped_short = 0
    unparsed = 0
    direction_of = {}

    for i, k in enumerate(keys):
        if "|" not in k:
            continue
        fen, uci = k.rsplit("|", 1)
        e = data[k]
        try:
            m = from_fifa_entry(fen, uci, e)
            f = line_facts(m)
            if f is None:
                unparsed += 1
                continue
            if f["allowed_plies"] < 2:
                skipped_short += 1
                continue
            eligible += 1
            res = tag_mistake_full(m, with_maia=False, classification="blunder")
            tags = [t for t in res["tags"] if t["label"] not in INFO_NOISE]
        except Exception:
            unparsed += 1
            continue

        teachable = [t for t in tags if t["direction"] != "info"]
        for t in tags:
            lab = t["label"]
            fires[lab] += 1
            direction_of[lab] = t["direction"]
            if len(teachable) == 1 and t["direction"] != "info":
                sole[lab] += 1
            for chk in CHECKS.get(lab, ()):
                if chk(lab, f):
                    contra[lab] += 1
                    contra_why[lab][chk.__name__] += 1
                    if len(examples[lab]) < args.examples:
                        examples[lab].append({
                            "fen": fen, "uci": uci, "played_san": m.played_san,
                            "refutation": " ".join(m.refutation_san),
                            "why": chk.__name__, "facts": f,
                            "evidence": t.get("evidence", ""),
                        })
                    break
        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1}/{len(keys)}", flush=True)

    if args.label:
        print(f"\n=== {args.label} ===")
        print(f"fires={fires[args.label]} contradictions={contra[args.label]}")
        for ex in examples[args.label]:
            print(json.dumps(ex, indent=2))
        return

    report = {
        "corpus": os.path.basename(args.enrich),
        "positions_seen": len(keys),
        "eligible": eligible,
        "skipped_line_too_short": skipped_short,
        "unparsed": unparsed,
        "labels": {},
    }
    for lab, n in fires.most_common():
        report["labels"][lab] = {
            "direction": direction_of.get(lab, "?"),
            "fires": n,
            "rate_pct": round(100.0 * n / max(1, eligible), 2),
            "sole": sole[lab],
            "sole_pct": round(100.0 * sole[lab] / max(1, n), 1),
            "checked": lab in CHECKS,
            "contradictions": contra[lab],
            "contradiction_pct": round(100.0 * contra[lab] / max(1, n), 1),
            "why": dict(contra_why[lab]),
            "examples": examples[lab],
        }
    json.dump(report, open(args.out, "w"), indent=2)

    print(f"\npositions={len(keys)} eligible={eligible} "
          f"skipped(short line)={skipped_short} unparsed={unparsed}")
    print(f"\n{'label':<34}{'dir':<9}{'fires':>7}{'rate%':>8}{'sole%':>7}{'CONTRA':>8}{'contra%':>9}")
    for lab, r in sorted(report["labels"].items(), key=lambda kv: -kv[1]["contradictions"]):
        flag = "  <== " if r["contradictions"] else ""
        print(f"{lab:<34}{r['direction']:<9}{r['fires']:>7}{r['rate_pct']:>8}"
              f"{r['sole_pct']:>7}{r['contradictions']:>8}{r['contradiction_pct']:>9}{flag}")
    unchecked = [l for l, r in report["labels"].items() if not r["checked"]]
    print(f"\nNO CONTRADICTION CHECK for {len(unchecked)} labels (fires but unverified):")
    print("  " + ", ".join(sorted(unchecked)[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
