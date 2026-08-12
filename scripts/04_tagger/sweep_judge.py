#!/usr/bin/env python3
"""Rate every tag's quality across the corpus, and collect tags the judge thinks are MISSING.

The wide run gated by `validate_judge.py`. That gate showed the judge, shown a tag + its claim + the
board, distinguishes known-bad tags from correct ones (2/4 caught, 0/4 false alarms on vetted labels) and
reads the board rather than the label text. This points it at the 56,950-position FIFA corpus and
aggregates, because per-instance verdicts are noisy but per-LABEL rates are not.

Two things Sam asked for on top of a yes/no verdict:
  - QUALITY (1-5) per tag, so tags get a score and we tackle the worst — finer than a 3-way verdict and
    what the ranking sorts on.
  - MISSING tags, freeform — the judge names a tactic/idea it thinks the position has that NO tag
    captured. This attacks #106 (24% of mistakes get no teachable tag) from the generative side: instead
    of asking "is what fired right?", ask "what should have fired?".

Design decisions carried from the audit doc:
  - The unit judged is a POSITION with its FULL explain-tag set, not one label (a tag is only "wrong"
    relative to everything else that fired). One agy call per position, verdicts for all its tags.
  - Stratify by label (N positions each), then rank by FLAG RATE, never raw hits — raw hits just ranks by
    frequency, so Missed Development (fires everywhere) always "wins".
  - Independently computed board facts go in the prompt; the judge's own counting is unreliable.
  - Material-arithmetic verdicts (SEE-0 trades) are the judge's known blind spot — SEE checks those
    deterministically, so we DON'T rank on them. `is_material_label` marks them for exclusion from the
    worst-list (kept in raw output).
  - Candidates, never findings: every top-ranked label gets board-verified by hand before it's a bug.

Usage:
    python3 sweep_judge.py --enrich /tmp/fifa_enrich.json --per-tag 4 --out sweep.json
    python3 sweep_judge.py --enrich ... --resume sweep.json          # skip already-judged positions
    python3 sweep_judge.py --enrich ... --max-positions 120          # cost/time cap for a first pass
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake, _eval_to_cp          # noqa: E402
from tagger import tag_mistake_full               # noqa: E402
from validate_judge import board_facts, _blurbs   # noqa: E402  (reuse the vetted facts + blurb loader)

AGY = os.path.expanduser("~/.local/bin/agy")

# Labels whose correctness is decided by SEE/material arithmetic — the judge's documented blind spot.
# Kept in the raw output but excluded from the ranked worst-list so we don't chase what code answers.
_MATERIAL_HINTS = ("exchange", "pawn capture", "free ", "hung", "hanging", "greedy",
                   "winning capture", "capture of defender", "pawn trade", "material")


def is_material_label(label):
    l = label.lower()
    return any(h in l for h in _MATERIAL_HINTS)


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
        played_san=e.get("played_san", ""), best_san=e.get("best_san", ""))


SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 5,
                                "description": "5=nails the key point; 3=true but not the main idea; "
                                               "1=wrong or misleading here"},
                    "verdict": {"type": "string", "enum": ["SUPPORTED", "SUSPICIOUS", "WRONG"]},
                    "why": {"type": "string", "description": "Under 25 words, cite squares."},
                },
                "required": ["label", "quality", "verdict", "why"],
            },
        },
        "missing": {
            "type": "array",
            "description": "Tactics or ideas genuinely present that NO label above captured. Empty if none.",
            "items": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "e.g. 'fork on e6', 'back-rank weakness'"},
                    "why": {"type": "string", "description": "Under 20 words, cite squares."},
                },
                "required": ["idea", "why"],
            },
        },
    },
    "required": ["verdicts", "missing"],
}


def build_prompt(m, tags, blurbs):
    side = "White" if m.mover == chess.WHITE else "Black"
    lines = [
        "You are auditing an automated chess-coach's labels for one moment in a real game.",
        "Judge ONLY from the board and the lines. Some labels may be right, some wrong, any number.",
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
        "Labels this moment was given, each with what it CLAIMS:",
    ]
    for t in tags:
        blurb = blurbs.get(t["label"], "")
        lines.append(f"  - {t['label']} [{t['direction']}] — claims: {blurb or '(no definition)'}")
    lines += [
        "",
        "For EACH label: quality 1-5 (5 = it names the single most important thing; 1 = wrong or "
        "misleading here), a verdict (SUPPORTED / SUSPICIOUS / WRONG), and why in <25 words with squares.",
        "A tactic claim must be playable NOW or actually occur in the given line — not require the "
        "opponent to cooperate several moves deep.",
        "Then in `missing`: name any real tactic or key idea in THIS position that none of the labels "
        "captured (the biggest thing a coach would say that's absent above). Empty if the labels cover it.",
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
            return None, (r.stderr or "")[-160:]
        payload = json.loads(r.stdout.strip().splitlines()[-1])
        return payload.get("structured_output"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        os.unlink(schema_path)


def norm(label):
    """The judge echoes labels as 'Name [direction]' or with stray case; match on the bare name."""
    return label.split(" [")[0].strip().lower()


def stratified_positions(enrich, per_tag, max_positions, scan_limit=20000):
    """Build tag sets for the corpus (CPU only), then pick up to `per_tag` positions per LABEL.

    One position covers several labels, so this de-duplicates: a position selected for label A also
    supplies label B's quota. Positions are the unit of a judge call; labels are the unit of the quota.

    Tagging all 56,950 positions takes >5 min, and it's unnecessary: `scan_limit` caps how many corpus
    entries we tag, and we stop early once every label seen has `per_tag`*3 candidate positions (enough
    to fill the quota with room for de-dup). Rare labels are why the multiplier exists — a label that
    appears once in 20k still gets its single position.
    """
    per_label = collections.defaultdict(list)          # label -> [pos_key, ...]
    pos_tags = {}                                      # pos_key -> (mistake, [tag dicts])
    items = list(enrich.items())[:scan_limit]
    target = per_tag * 3
    scanned = 0
    for key, e in items:
        scanned += 1
        try:
            fen, uci = key.split("|", 1)
            m = from_fifa_entry(fen, uci, e)
            res = tag_mistake_full(m, with_maia=False, classification=None)
        except Exception:
            continue
        explain = [t for t in res.get("tags", []) if t["direction"] != "info"]
        if not explain:
            continue
        pos_tags[key] = (m, explain)
        for t in explain:
            per_label[t["label"]].append(key)
        # early-exit: enough candidates for every label we've seen (checked periodically)
        if scanned % 2000 == 0 and per_label and all(len(v) >= target for v in per_label.values()):
            print(f"  quota filled after scanning {scanned}", flush=True)
            break
    print(f"  scanned {scanned} corpus entries, {len(pos_tags)} tagged", flush=True)

    chosen, per_label_count = {}, collections.Counter()
    # Round-robin by label so rare labels get their quota before the cap is hit.
    labels_by_rarity = sorted(per_label, key=lambda l: len(per_label[l]))
    round_i = 0
    while len(chosen) < max_positions:
        progressed = False
        for lab in labels_by_rarity:
            if per_label_count[lab] >= per_tag:
                continue
            pool = per_label[lab]
            if round_i >= len(pool):
                continue
            key = pool[round_i]
            progressed = True
            if key not in chosen:
                chosen[key] = pos_tags[key]
                if len(chosen) >= max_positions:
                    break
            for t in pos_tags[key][1]:
                per_label_count[t["label"]] += 1
        round_i += 1
        if not progressed:
            break
    return chosen, per_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", default="/tmp/fifa_enrich.json")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--per-tag", type=int, default=4)
    ap.add_argument("--max-positions", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="sweep.json")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    blurbs = _blurbs()
    enrich = json.load(open(args.enrich))
    print(f"corpus: {len(enrich)} positions; building tag sets ...", flush=True)
    chosen, per_label = stratified_positions(enrich, args.per_tag, args.max_positions)
    print(f"labels present: {len(per_label)}; positions to judge: {len(chosen)}", flush=True)

    done = {}
    if args.resume and os.path.exists(args.resume):
        done = {r["key"]: r for r in json.load(open(args.resume)).get("raw", [])}
        print(f"resuming: {len(done)} already judged", flush=True)

    raw = list(done.values())
    todo = [(k, m, tags) for k, (m, tags) in chosen.items() if k not in done]

    def one(key, m, tags):
        out, err = ask(build_prompt(m, tags, blurbs), args.model, args.effort)
        return {"key": key, "played": m.played_san, "best": m.best_san,
                "tags": [t["label"] for t in tags], "judge": out, "error": err}

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, k, m, tags): k for k, m, tags in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            raw.append(r)
            if i % 5 == 0 or r["error"]:
                print(f"  {i}/{len(todo)}" + (f"  ERR {r['error']}" if r["error"] else ""), flush=True)
            if i % 10 == 0:                            # checkpoint (long run, resumable)
                json.dump({"raw": raw}, open(args.out, "w"), indent=1, default=str)

    # ---- aggregate per label --------------------------------------------------------------------
    by_label = collections.defaultdict(lambda: {"n": 0, "flagged": 0, "quality": [], "examples": []})
    missing = collections.Counter()
    missing_examples = collections.defaultdict(list)
    for r in raw:
        j = r.get("judge") or {}
        for v in j.get("verdicts") or []:
            name = v.get("label", "")
            # map judge's echoed label back to one of the position's real tags
            match = next((t for t in r["tags"] if norm(t) == norm(name)), name)
            s = by_label[match]
            s["n"] += 1
            q = v.get("quality")
            if isinstance(q, int):
                s["quality"].append(q)
            if v.get("verdict") in ("SUSPICIOUS", "WRONG"):
                s["flagged"] += 1
                if len(s["examples"]) < 4:
                    s["examples"].append({"key": r["key"], "verdict": v["verdict"],
                                          "why": v.get("why", "")})
        for mi in j.get("missing") or []:
            idea = mi.get("idea", "").strip().lower()
            if idea:
                missing[idea] += 1
                if len(missing_examples[idea]) < 3:
                    missing_examples[idea].append({"key": r["key"], "why": mi.get("why", "")})

    ranked = []
    for lab, s in by_label.items():
        if s["n"] < 2:
            continue
        ranked.append({
            "label": lab, "n": s["n"], "flag_rate": round(s["flagged"] / s["n"], 3),
            "mean_quality": round(sum(s["quality"]) / len(s["quality"]), 2) if s["quality"] else None,
            "material_family": is_material_label(lab), "examples": s["examples"]})
    ranked.sort(key=lambda x: (-x["flag_rate"], x["mean_quality"] or 5))

    out = {"raw": raw, "ranked": ranked,
           "missing": [{"idea": k, "count": c, "examples": missing_examples[k]}
                       for k, c in missing.most_common(30)]}
    json.dump(out, open(args.out, "w"), indent=1, default=str)

    print("\n" + "=" * 82)
    print("WORST TAGS by flag rate (n>=2; ★ = material family, judge's blind spot — verify with SEE)")
    print("=" * 82)
    for r in ranked[:20]:
        star = "★" if r["material_family"] else " "
        print(f"{star} {r['flag_rate']*100:5.0f}%  q={r['mean_quality']}  n={r['n']:3d}  {r['label']}")
    print("\nMOST-CITED MISSING IDEAS (freeform → #106 coverage gaps)")
    for mi in out["missing"][:15]:
        print(f"  {mi['count']:3d}  {mi['idea']}")
    print(f"\nwrote {args.out}  ({len(raw)} positions judged)")


if __name__ == "__main__":
    main()
