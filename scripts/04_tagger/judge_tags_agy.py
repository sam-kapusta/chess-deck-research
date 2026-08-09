#!/usr/bin/env python3
"""Judge every tag against its positions with an independent model, BLIND to the tag.

The SAE-eval pattern (scripts/evaluation/detection_scoring.py) applied to the tagger: there, a FEATURE is
scored by sampling positions where it fires and asking a judge whether its LABEL describes them. Here a
TAG is scored by sampling positions where it fires, asking a judge what the position's mistake actually
IS — without showing it the tag — and then comparing.

WHY BLIND. If you show the label and ask "does this make sense?", you invite confirmation. The judge will
find *something* matching and validate it. Asked blind, it answers from the board. Both modes were tested
on the known-bad #101 position (best move is mate-in-1, tagged "Missed Free Knight"): blind it said
"delivers immediate checkmate"; shown the label it said NO. Blind stays the default because it can't be
led, and because its free-text answer is reusable evidence.

WHAT THIS IS NOT. The judge is ~club-strength at chess and will be confidently wrong sometimes. Its
disagreements are CANDIDATES, never findings. Every one must be re-derived on the board before it is
filed. That step is not optional: on the FIRST run of this harness the judge overturned a bug report I had
already filed (#102 — I blamed an irrelevant pre-existing pin; the judge found the real load-bearing pin
and was right on the board). It breaks the reviewer's confirmation loop as much as the tagger's.

INVOCATION GOTCHA. `agy -p` must come FIRST with the prompt as its value. With flags first, agy answers
questions ABOUT its own flags — four attempts returned docs on --model/--output-format instead of answers.
--json-schema additionally requires --output-format json.

Usage:
    python3 judge_tags_agy.py --enrich fifa_enrich.json --out judged.json --per-tag 3
    python3 judge_tags_agy.py --enrich ... --out ... --only "Hung Material,Allowed Pin" --per-tag 5
    python3 judge_tags_agy.py --enrich ... --out ... --resume judged.json     # skip already-judged
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake, _eval_to_cp          # noqa: E402
from tagger import tag_mistake_full               # noqa: E402

AGY = os.path.expanduser("~/.local/bin/agy")
INFO_NOISE = {"Opening", "Middlegame", "Endgame"}

# The judge answers three things. `primary_theme` is free text so it can say anything; the booleans are
# what we filter on. Deliberately NOT asking "is the tag right" — the tag is never shown.
SCHEMA = {
    "type": "object",
    "properties": {
        "primary_theme": {"type": "string",
                          "description": "The single most important thing to say, under 20 words"},
        "is_mate": {"type": "boolean", "description": "Does the relevant line force checkmate?"},
        "material_decisive": {"type": "boolean",
                              "description": "Is winning/losing material the main point (vs mate/position)?"},
        "side_is_lost": {"type": "boolean",
                         "description": "Is the side to move already strategically lost regardless?"},
    },
    "required": ["primary_theme", "is_mate", "material_decisive", "side_is_lost"],
}


def from_fifa_entry(fen, uci, e):
    b = chess.Board(fen)
    best = e.get("top_3_best") or []
    refut = e.get("top_3_refutations") or []
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=e.get("best_uci", ""),
        best_line_san=((best[0].get("line") or "").split() if best else []),
        refutation_san=((refut[0].get("line") or "").split() if refut else []),
        eval_before=_eval_to_cp(e.get("eval_before")), eval_after=_eval_to_cp(e.get("eval_after")),
        cp_loss=int(e.get("cp_loss", 0) or 0), mover=b.turn,
        played_san=e.get("played_san", ""), best_san=e.get("best_san", ""),
    )


def build_prompt(m, direction):
    """The position and BOTH lines, with NO tag and no leading question.

    Which line is 'relevant' depends on direction, and saying so is necessary context — a MISSED tag is a
    claim about the best line, an ALLOWED/HUNG tag about the refutation. Withholding that would make the
    judge answer about the wrong half.
    """
    side = "White" if m.mover == chess.WHITE else "Black"
    focus = ("the BEST line (what the player could have played instead)" if direction == "missed"
             else "the PUNISHMENT line (what the opponent does after the played move)")
    return (
        f"Chess position analysis. Answer only from the board.\n\n"
        f"FEN: {m.fen_before}\n"
        f"{side} to move and played: {m.played_san}\n"
        f"Engine's best move was: {m.best_san}\n"
        f"BEST line: {' '.join(m.best_line_san) or '(none)'}\n"
        f"PUNISHMENT line after the played move: {' '.join(m.refutation_san) or '(none)'}\n\n"
        f"Focus on {focus}.\n"
        f"What is the single most important thing a coach should say about this moment? "
        f"Be specific and concrete (name the piece and square). Under 20 words."
    )


def ask_judge(prompt, model, effort, timeout=180):
    """One agy call. Returns the structured dict, or None on any failure (never raises)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(SCHEMA, fh)
        schema_path = fh.name
    try:
        # -p FIRST (see module docstring), and --json-schema needs --output-format json.
        r = subprocess.run(
            [AGY, "-p", prompt, "--json-schema", schema_path, "--output-format", "json",
             "--model", model, "--effort", effort],
            capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir(),
        )
        if r.returncode != 0:
            return None
        payload = json.loads(r.stdout.strip().splitlines()[-1])
        out = payload.get("structured_output")
        if isinstance(out, dict):
            out["_usage"] = payload.get("usage", {})
        return out
    except Exception:
        return None
    finally:
        os.unlink(schema_path)


def disagreements(label, direction, judge, all_labels):
    """Mechanical comparison of the tag against the judge's board reading.

    Each rule states a CONFLICT between what the label asserts and what the judge saw. Kept few and
    literal on purpose: this decides what a human looks at, so a noisy rule wastes the scarce resource
    (attention), and a rule that encodes my own theory would re-introduce the bias blindness avoids.

    `all_labels` is the FULL tag set the tagger produced for this position, and it is load-bearing. The
    coach shows a position's tags together, so "the tagger missed the mate" is only true if NO mate tag
    fired anywhere in the set — not if this particular label isn't the mate one. The first version of this
    function compared each label in isolation and produced 4 straight false positives (Allowed Pawn
    Capture, Allowed Double Check, Allowed Discovered Attack, Allowed Combination → Knight Fork) where
    `Allowed Mate` / `Missed Mate` was in fact co-firing. Judge was right about the board every time; the
    RULE was wrong.
    """
    out = []
    labels = set(all_labels or [label])
    any_mate_tag = any("Mate" in l for l in labels)
    material_family = (label.startswith("Hung ") or label.startswith("Missed Free")
                       or label.startswith("Lost ") or "Material" in label)
    if judge.get("is_mate") and material_family and not any_mate_tag:
        out.append("judge says the line is MATE but the tags claim only material")
    if judge.get("is_mate") and not any_mate_tag and direction in ("missed", "allowed"):
        out.append("judge says MATE but NO mate tag fired for this position")
    if label.endswith("Sacrifice") and judge.get("side_is_lost"):
        out.append("tag claims a sacrifice but judge says that side is already lost")
    if material_family and judge.get("material_decisive") is False and not any_mate_tag:
        out.append("tag is material-based but judge says material is not the point")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enrich", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-tag", type=int, default=3)
    p.add_argument("--scan", type=int, default=56950)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--only", default=None)
    p.add_argument("--model", default="gemini-3.1-pro-high")
    p.add_argument("--effort", default="high")
    p.add_argument("--resume", default=None, help="prior --out file; skip pairs already judged")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    done = set()
    results = []
    if args.resume and os.path.exists(args.resume):
        prior = json.load(open(args.resume))
        results = prior.get("results", [])
        done = {(r["label"], r["key"]) for r in results}
        print(f"resuming: {len(results)} already judged")

    data = json.load(open(args.enrich))
    keys = list(data.keys())[:args.scan][::args.stride]

    # Pick which (label, position) pairs to judge, spread across positions so one game can't dominate.
    todo = collections.defaultdict(list)
    for k in keys:
        if "|" not in k:
            continue
        fen, uci = k.rsplit("|", 1)
        try:
            m = from_fifa_entry(fen, uci, data[k])
            tags = [t for t in tag_mistake_full(m, with_maia=False)["tags"]
                    if t["label"] not in INFO_NOISE and t["direction"] != "info"]
        except Exception:
            continue
        for t in tags:
            lab = t["label"]
            if only and lab not in only:
                continue
            if len(todo[lab]) < args.per_tag and (lab, k) not in done:
                todo[lab].append((k, m, t, [x["label"] for x in tags]))

    total = sum(len(v) for v in todo.values())
    print(f"judging {total} (label, position) pairs across {len(todo)} tags "
          f"with {args.model}/{args.effort}", flush=True)

    # Concurrency: each call is ~20s of remote latency, so sequential would be ~3h for a full sweep.
    # Modest pool — the aim is wall-clock, not throughput records, and hammering a subscription endpoint
    # risks throttling that looks like judge failures.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = [(lab, k, m, t, al) for lab in sorted(todo) for (k, m, t, al) in todo[lab]]
    n = 0
    tok = 0
    lock = __import__("threading").Lock()

    def work(job):
        lab, k, m, t, al = job
        return job, ask_judge(build_prompt(m, t["direction"]), args.model, args.effort)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            try:
                (lab, k, m, t, al), j = fut.result()
            except Exception:
                continue
            with lock:
                n += 1
                if j is None:
                    print(f"  [{n}/{total}] JUDGE FAILED", flush=True)
                    continue
                tok += (j.get("_usage") or {}).get("total_tokens", 0)
                dis = disagreements(lab, t["direction"], j, al)
                results.append({
                    "label": lab, "direction": t["direction"], "key": k,
                    "fen": m.fen_before, "played": m.played_san, "best": m.best_san,
                    "best_line": " ".join(m.best_line_san),
                    "refutation": " ".join(m.refutation_san),
                    "evidence": t.get("evidence", ""),
                    "all_labels": al,
                    "judge": {kk: vv for kk, vv in j.items() if kk != "_usage"},
                    "disagreements": dis,
                })
                mark = "  <== " + "; ".join(dis) if dis else ""
                print(f"  [{n}/{total}] {lab}: {j.get('primary_theme','')[:66]}{mark}", flush=True)
                # Write after every call: a long run against an external service, and a crash at pair
                # 900 must not throw away 899 judgements.
                json.dump({"model": args.model, "effort": args.effort, "total_tokens": tok,
                           "results": results}, open(args.out, "w"), indent=2)

    flagged = [r for r in results if r["disagreements"]]
    by_tag = collections.Counter(r["label"] for r in flagged)
    print(f"\njudged {len(results)} pairs, {tok} tokens")
    print(f"DISAGREEMENTS: {len(flagged)} across {len(by_tag)} tags")
    for lab, c in by_tag.most_common():
        tot = sum(1 for r in results if r["label"] == lab)
        print(f"  {lab:<36} {c}/{tot}")
    print(f"\nwrote {args.out}")
    print("NOTE: disagreements are CANDIDATES. Verify each on the board before filing.")


if __name__ == "__main__":
    main()
