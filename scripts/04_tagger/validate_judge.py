#!/usr/bin/env python3
"""Can an LLM judge SHOWN a tag spot the ones we already know are wrong?

THE GO/NO-GO GATE for Sam's proposed sweep ("run positions + tags through an LLM, flag the weird ones,
tackle the labels with the most hits"). The prior harness (`judge_tags_agy.py`) asked BLIND — never
showing the tag — and a 483-pair sweep produced 0 confirmed bugs. That result is uninterpretable: we
never established the harness could detect a wrong tag at all, so "found nothing" and "can't see
anything" are indistinguishable. This script removes that ambiguity BEFORE any wide run.

  validate-the-finder-before-proving-absence, applied to the finder itself.

The set below is every tagger false positive fixed in the last three weeks, each with the position it was
reported on and the label that was wrong — recovered from the regression tests that now guard them. Plus
CONTROL cases where the same tag is CORRECT, taken from those tests' positive assertions. A judge that
flags the bad ones and leaves the controls alone is worth pointing at the corpus. One that can't
distinguish them isn't, and we stop.

Differences from judge_tags_agy.py, both deliberate:
  - SHOWN the tag + its taxonomy blurb, so the judge evaluates a specific CLAIM instead of free-writing a
    theme we then have to match with a rule. That matching rule is what broke the last sweep (4 straight
    false positives from comparing labels in isolation).
  - Judges the position's FULL explain-tag set at once, never one label alone. The coach shows tags
    together, so "this tag is wrong" is only meaningful against everything else that fired.

Usage:
    python3 validate_judge.py                      # full validation set
    python3 validate_judge.py --only vienna_pin    # one case
    python3 validate_judge.py --model gemini-3.1-pro --effort high
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake                       # noqa: E402
from tagger import tag_mistake_full               # noqa: E402
import chesslib_util as U                         # noqa: E402

AGY = os.path.expanduser("~/.local/bin/agy")
TAXONOMY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../../chess-deck-code/frontend/src/data/mistakeTaxonomy.json")

# ---------------------------------------------------------------------------------------------------
#  The validation set.
#
#  `inject` = a label that SHIPPED on this position and was wrong (the detector has since been fixed, so
#  the current tagger won't emit it — we add it back to recreate what the user saw).
#  `expect_flag` = the label the judge must flag. None => a control; flagging anything here is a miss.
#  Every `fen`/line is asserted legal at load time (three of my hand-built FENs were illegal once).
# ---------------------------------------------------------------------------------------------------
CASES = [
    # ---- KNOWN BAD -------------------------------------------------------------------------------
    dict(
        id="vienna_pin", real_game=True,
        note="Sam 2026-08-11. Bb5 pin is 4 plies deep and needs ...Nc6+...d6 first; never cashes in.",
        fen="rnbqkbnr/pppp1pp1/8/4p2p/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3",
        played="f2f4", best="g1f3",
        best_line=["Nf3", "Nc6", "d4", "d6", "Bb5", "exd4", "Nxd4", "Bd7"],
        refutation=["exf4", "Bc4", "g5", "h4", "d5", "exd5", "Bd6", "d4"],
        eval_before=97, eval_after=39, cp_loss=58, mover=chess.WHITE,
        played_san="f4", best_san="d4", classification="inaccuracy",
        inject="Missed Pin (to King)", expect_flag="Missed Pin (to King)",
    ),
    dict(
        id="pawn_trade", real_game=True,
        note="Sam ply 10. e6 lets cxd5 but exd5 recaptures — dead even (SEE 0), not a grab.",
        fen="r2qkbnr/pp2pppp/2n5/2pp4/2P3b1/1P2PN2/PB1P1PPP/RN1QKB1R b KQkq - 0 5",
        played="e7e6", best="d5d4",
        best_line=["d4", "exd4", "cxd4", "h3"],
        refutation=["cxd5", "exd5", "Be2", "Nf6"],
        eval_before=63, eval_after=-22, cp_loss=85, mover=chess.BLACK,
        played_san="e6", best_san="d4", classification="inaccuracy",
        inject="Allowed Pawn Capture", expect_flag="Allowed Pawn Capture",
    ),
    dict(
        id="greek_gift_g7", real_game=True,
        note="Bxg7+ is not a Greek Gift — that tactic is defined by the h7/h2 square.",
        fen="r6k/pp1qbrpp/2p1Rn1B/8/2P5/2N3Q1/PP3PPP/4R1K1 w - - 3 18",
        played="e6e7", best="h6g7",
        # Bxf6 is a DISCOVERED check (Qg3 down the g-file once g7 vacates), so Black cannot recapture —
        # my first line assumed ...Bxf6 and was illegal. The walk caught it.
        best_line=["Bxg7+", "Kg8", "Bxf6+", "Rg7"],
        refutation=["Rxe7", "Rxe7", "Qd3"],
        eval_before=250, eval_after=40, cp_loss=210, mover=chess.WHITE,
        played_san="Re7", best_san="Bxg7+", classification="mistake",
        inject="Missed Greek Gift", expect_flag="Missed Greek Gift",
    ),
    dict(
        id="checking_trade", real_game=False,
        note="Rxe8+ is SEE 0 but it's a CHECK — a tactic, not the 'even trade' the label claims.",
        # played h3 rather than Kh1: with the king on h1 and h2 blocked by its own pawn, ...Rxe1 is MATE,
        # which changes what the position is about. h3 gives the luft so the refutation is a real line.
        fen="r3r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        played="h2h3", best="e1e8",
        best_line=["Rxe8+", "Rxe8", "h3"],
        refutation=["Rxe1+", "Kh2"],
        eval_before=0, eval_after=-150, cp_loss=150, mover=chess.WHITE,
        played_san="h3", best_san="Rxe8+", classification="mistake",
        inject="Missed Rook Exchange", expect_flag="Missed Rook Exchange",
    ),
    # DROPPED: a "#100 mover-is-already-lost sacrifice" case. Building one by hand needs a position where
    # the mover is ~6 pawns down AND the best line sheds material, and my two attempts produced illegal
    # lines. A fabricated fixture that passes for the wrong reason is worse than one fewer data point.
    # ---- CONTROLS (tag is CORRECT — flagging these is a false alarm) ------------------------------
    dict(
        id="ctl_free_grab", real_game=False,
        note="Kg1 lets ...Rxb3 win an UNDEFENDED pawn (SEE +1). Allowed Pawn Capture is right here.",
        assert_ok=["Allowed Pawn Capture"],
        fen="1r4k1/5ppp/8/8/8/1P6/5PPP/R5K1 w - - 0 1",
        played="g1h1", best="a1b1",
        best_line=["Rb1", "Kf8"], refutation=["Rxb3"],   # White has no recapture on b3 — that IS the point
        eval_before=0, eval_after=-100, cp_loss=120, mover=chess.WHITE,
        played_san="Kh1", best_san="Rb1", classification="mistake",
        inject=None, expect_flag=None,
    ),
    dict(
        id="ctl_immediate_pin", real_game=False,
        note="Re1 pins the e7 bishop to Ke8 RIGHT NOW. A real missed pin.",
        assert_ok=["Missed Pin (to King)"],
        fen="4k3/4b1pp/8/8/8/8/PPP2PPP/3R2K1 w - - 0 1",
        played="g1h1", best="d1e1",
        best_line=["Re1", "Kf8", "h3", "Kg8"], refutation=["Kf8", "a3"],   # Kh2 is blocked by its own pawn
        eval_before=0, eval_after=-120, cp_loss=140, mover=chess.WHITE,
        played_san="Kh1", best_san="Re1", classification="mistake",
        inject=None, expect_flag=None,
    ),
    dict(
        id="ctl_real_hang", real_game=False,
        note="Kf1 leaves the a1 rook en prise; ...Rxa1 takes it free. Hung Material is right.",
        assert_ok=["Allowed Hanging Piece", "Missed Free Rook"],
        fen="6k1/8/8/8/8/8/r7/R3K3 w - - 0 1",
        played="e1f1", best="a1a2",
        best_line=["Rxa2"], refutation=["Rxa1"],   # a1->a2 IS the capture of the black rook
        eval_before=0, eval_after=-500, cp_loss=500, mover=chess.WHITE,
        played_san="Kf1", best_san="Ra2", classification="blunder",
        inject=None, expect_flag=None,
    ),
    dict(
        id="ctl_greek_gift_h7", real_game=False,
        note="Textbook Bxh7+ Kxh7 Ng5+. Missed Greek Gift is correct here.",
        assert_ok=["Missed Greek Gift", "Missed Sacrifice"],
        fen="r1bq1rk1/pppn1ppp/8/8/8/3B1N2/PPP2PPP/R1BQ1RK1 w - - 0 10",
        played="c2c3", best="d3h7",
        best_line=["Bxh7+", "Kxh7", "Ng5+", "Kg8", "Qh5"],
        refutation=["Nf6", "Be2"],
        eval_before=30, eval_after=-200, cp_loss=230, mover=chess.WHITE,
        played_san="c3", best_san="Bxh7+", classification="mistake",
        inject=None, expect_flag=None,
    ),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["SUPPORTED", "SUSPICIOUS", "WRONG"]},
                    "why": {"type": "string", "description": "Under 25 words, cite squares."},
                },
                "required": ["label", "verdict", "why"],
            },
        },
    },
    "required": ["verdicts"],
}


def _blurbs():
    try:
        with open(TAXONOMY) as fh:
            tx = json.load(fh)
        return {k: v.get("blurb", "") for k, v in (tx.get("tags") or {}).items()}
    except Exception:
        return {}


def board_facts(m):
    """Board facts computed INDEPENDENTLY of any detector — the thing that made the reading dumps work.

    Mirrors tag_adapter's audit block. Reading the detector's own reasoning back would be circular.
    """
    b = m.board_before

    def describe(uci):
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in b.legal_moves:
                return {"uci": uci, "legal": False}
        except Exception:
            return {"uci": uci, "legal": False}
        victim = b.piece_at(mv.to_square)
        return {"san": b.san(mv), "is_capture": b.is_capture(mv),
                "captures": chess.piece_name(victim.piece_type) if victim else None,
                "gives_check": b.gives_check(mv),
                "see": U.static_exchange_eval(b, mv) if b.is_capture(mv) else 0}

    def plies(start_fen, sans):
        out, bb = [], chess.Board(start_fen)
        for s in sans[:10]:
            try:
                mv = bb.parse_san(s)
            except Exception:
                break
            out.append({"san": s, "is_capture": bb.is_capture(mv), "gives_check": bb.gives_check(mv)})
            bb.push(mv)
        return out

    after = m.board_after
    return {
        "played": describe(m.played_uci),
        "best": describe(m.best_uci),
        "best_line_plies": plies(m.fen_before, m.best_line_san),
        "refutation_plies": plies(after.fen(), m.refutation_san),
        "loose_non_pawns_after_played": [
            f"{chess.piece_name(p.piece_type)} {chess.square_name(sq)} "
            f"({'white' if p.color else 'black'})"
            for sq, p in after.piece_map().items()
            if p.piece_type != chess.PAWN and after.attackers(not p.color, sq)
            and not after.attackers(p.color, sq)
        ],
    }


def build_prompt(m, tags, blurbs):
    """Show the position, both lines, independent board facts, and EVERY explain tag with its claim.

    No hint that anything is wrong: the judge is asked to assess each claim, not to find the odd one out.
    A prompt that says "one of these is wrong" gets a wrong one named every time.
    """
    side = "White" if m.mover == chess.WHITE else "Black"
    lines = [
        "You are auditing an automated chess-coach's labels for one moment in a real game.",
        "Judge ONLY from the board and the lines given. Some labels may be right, some wrong, "
        "any number of either.",
        "",
        f"FEN: {m.fen_before}",
        f"{side} to move, and played: {m.played_san}",
        f"Engine's best move: {m.best_san}",
        f"BEST line (what {side} could have played): {' '.join(m.best_line_san) or '(none)'}",
        f"PUNISHMENT line (after {m.played_san}): {' '.join(m.refutation_san) or '(none)'}",
        "",
        "Independently computed board facts (trust these over your own counting):",
        json.dumps(board_facts(m), indent=1),
        "",
        "The labels this moment was given, each with what it CLAIMS:",
    ]
    for t in tags:
        blurb = blurbs.get(t["label"], "")
        lines.append(f"  - {t['label']} [{t['direction']}] — claims: {blurb or '(no definition)'}")
        lines.append(f"      detector evidence: {t['evidence'][:120]}")
    lines += [
        "",
        "For EACH label, decide whether the position and lines actually support its claim.",
        "SUPPORTED = the claim is true here. SUSPICIOUS = it does not clearly hold. "
        "WRONG = the board contradicts it.",
        "A claim about a tactic must be a tactic the side could actually play NOW, or that actually "
        "happens in the given line — not something that would require the opponent to cooperate.",
        "Cite concrete squares in `why`.",
    ]
    return "\n".join(lines)


def ask(prompt, model, effort, timeout=240):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(SCHEMA, fh)
        schema_path = fh.name
    try:
        r = subprocess.run(
            [AGY, "-p", prompt, "--json-schema", schema_path, "--output-format", "json",
             "--model", model, "--effort", effort],
            capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
        if r.returncode != 0:
            return None, (r.stderr or "")[-200:]
        payload = json.loads(r.stdout.strip().splitlines()[-1])
        return payload.get("structured_output"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        os.unlink(schema_path)


def make_mistake(c):
    return Mistake(
        fen_before=c["fen"], played_uci=c["played"], best_uci=c["best"],
        best_line_san=c["best_line"], refutation_san=c["refutation"],
        eval_before=c["eval_before"], eval_after=c["eval_after"], cp_loss=c["cp_loss"],
        mover=c["mover"], played_san=c["played_san"], best_san=c["best_san"])


def validate_case(c):
    """Assert the fixture is real before it can validate anything (rule 3 of the harness rules)."""
    b = chess.Board(c["fen"])
    mv = chess.Move.from_uci(c["played"])
    assert mv in b.legal_moves, f"{c['id']}: played {c['played']} illegal"
    assert chess.Move.from_uci(c["best"]) in b.legal_moves, f"{c['id']}: best {c['best']} illegal"
    bb = chess.Board(c["fen"])
    for s in c["best_line"]:
        try:
            bb.push(bb.parse_san(s))
        except Exception:
            raise AssertionError(f"{c['id']}: best_line breaks at {s}")
    ab = b.copy()
    ab.push(mv)
    for s in c["refutation"]:
        try:
            ab.push(ab.parse_san(s))
        except Exception:
            raise AssertionError(f"{c['id']}: refutation breaks at {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="judge_validation.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    blurbs = _blurbs()
    if not blurbs:
        print("WARN: no taxonomy blurbs loaded — the judge won't know what each tag claims", file=sys.stderr)

    cases = [c for c in CASES if not args.only or c["id"] == args.only]
    for c in cases:
        validate_case(c)
    print(f"fixtures legal: {len(cases)}/{len(cases)}")

    # Build each position's tag set through the PRODUCTION entry point, then re-inject the label that
    # shipped (the detector is fixed now, so it no longer appears on its own).
    jobs = []
    for c in cases:
        m = make_mistake(c)
        res = tag_mistake_full(m, with_maia=False, classification=c["classification"])
        tags = [t for t in res["tags"] if t["direction"] != "info"]
        if c["inject"] and not any(t["label"] == c["inject"] for t in tags):
            tags.append({"label": c["inject"], "direction":
                         "missed" if c["inject"].startswith("Missed") else "allowed",
                         "evidence": "(as shipped)", "category": "", "layer": "tactic"})
        if c["expect_flag"] and not any(t["label"] == c["expect_flag"] for t in tags):
            print(f"  !! {c['id']}: expected label {c['expect_flag']!r} not in tag set — skipping")
            continue
        jobs.append((c, m, tags))

    print(f"judging {len(jobs)} positions with {args.model}/{args.effort} ...\n")
    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask, build_prompt(m, tags, blurbs), args.model, args.effort): (c, tags)
                for c, m, tags in jobs}
        for fut in cf.as_completed(futs):
            c, tags = futs[fut]
            out, err = fut.result()
            results[c["id"]] = {"case": c, "tags": [t["label"] for t in tags],
                               "judge": out, "error": err}
            print(f"  done: {c['id']}" + (f"  ERROR {err}" if err else ""))

    # ---- score --------------------------------------------------------------------------------
    bad = [r for r in results.values() if r["case"]["expect_flag"]]
    ctl = [r for r in results.values() if not r["case"]["expect_flag"]]
    caught, missed, false_alarms = [], [], []

    def norm(lab):
        """Normalize a label for matching.

        REQUIRED, not defensive: the judge echoes labels back exactly as the prompt renders them —
        `"Missed Pin (to King) [missed]"` — so exact-match scoring reports a MISS on a verdict that
        correctly said WRONG. That silently inverted the smoke-test result, and it is the same class of
        bug that produced the previous sweep's uninterpretable zero: the judge read the board fine, the
        comparison rule was broken. Match on the label text alone.
        """
        lab = (lab or "").strip()
        if " [" in lab:
            lab = lab.split(" [", 1)[0]
        return lab.strip().lower()

    def flagged(r):
        v = (r["judge"] or {}).get("verdicts") or []
        shown = {norm(t): t for t in r["tags"]}
        out = {}
        for x in v:
            if x.get("verdict") not in ("SUSPICIOUS", "WRONG"):
                continue
            key = norm(x.get("label"))
            # Map back to the label we actually showed; ignore verdicts on labels we never sent.
            if key in shown:
                out[shown[key]] = x
        return out

    candidates = []
    for r in bad:
        f = flagged(r)
        tgt = r["case"]["expect_flag"]
        (caught if tgt in f else missed).append((r["case"]["id"], tgt, f.get(tgt, {}).get("why", "")))
        for lab, v in f.items():
            if lab != tgt:
                candidates.append((r["case"]["id"], lab, v.get("why", "")))
    for r in ctl:
        f = flagged(r)
        vetted = set(r["case"].get("assert_ok") or [])
        # A false alarm is ONLY a flag on a label we VETTED as correct. Controls also carry tags nobody
        # vetted; a flag on one of those is a candidate finding, not an error by the judge. Scoring them
        # as errors is how the first pass talked itself into NO-GO on a run that found real problems.
        for lab, v in f.items():
            (false_alarms if lab in vetted else candidates).append(
                (r["case"]["id"], lab, v.get("why", "")))

    print("\n" + "=" * 78)
    print(f"KNOWN-BAD caught : {len(caught)}/{len(bad)}")
    for cid, tgt, why in caught:
        print(f"   \u2713 {cid:18s} {tgt}  \u2014 {why[:88]}")
    for cid, tgt, _ in missed:
        print(f"   \u2717 {cid:18s} {tgt}  MISSED")
    print(f"FALSE ALARMS on vetted-correct labels : {len(false_alarms)}")
    for cid, lab, why in false_alarms:
        print(f"   ! {cid:18s} {lab}  \u2014 {why[:80]}")
    print(f"CANDIDATES on unvetted labels (need board review) : {len(candidates)}")
    for cid, lab, why in candidates:
        print(f"   ? {cid:18s} {lab}  \u2014 {why[:80]}")
    print("=" * 78)
    ok_bad = len(caught) >= max(1, int(0.6 * len(bad)))
    verdict = ("GO \u2014 flags known-bad tags and leaves vetted-correct ones alone"
               if ok_bad and not false_alarms else
               "PARTIAL \u2014 see the per-class breakdown; do not run wide without reading it")
    print(verdict)

    with open(args.out, "w") as fh:
        json.dump({"results": results, "caught": caught, "missed": missed,
                   "false_alarms": false_alarms, "candidates": candidates, "verdict": verdict}, fh, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
