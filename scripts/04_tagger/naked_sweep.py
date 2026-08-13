#!/usr/bin/env python3
"""What teachable idea do the UNTAGGED mistakes have? (#106 coverage gap, from the generative side.)

25% of tagged mistakes get NO teachable tag (48% in endgames). The general LLM sweep (`sweep_judge.py`)
sampled positions that ALREADY had tags, so its "missing" signal was diluted by label disputes. This
samples ONLY the naked positions — so any tactic/idea Gemini names is a genuine coverage candidate, not a
"you labeled it X, I'd say Y" quibble.

Ask blind (there is no tag to show): FEN + played + best + both lines + independently-computed board
facts, "what is the single teachable idea here?" + a coarse category. Aggregate by idea; the RECURRING
ideas are candidate detectors to build. Every candidate is board-verified before it's real — the LLM
hallucinates tactics and reads lines the engine doesn't play (2026-08-11 lesson).

Usage: python3 naked_sweep.py --n 60 --out naked.json
"""
import argparse, collections, concurrent.futures as cf, json, os, subprocess, sys, tempfile
import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mistake import Mistake, _eval_to_cp          # noqa: E402
from tagger import tag_mistake_full               # noqa: E402
from sweep_judge import from_fifa_entry           # noqa: E402
from validate_judge import board_facts            # noqa: E402

AGY = os.path.expanduser("~/.local/bin/agy")
TEACH = {"missed", "allowed", "hung", "played"}

SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string", "description": "The single teachable concept, 2-5 words "
                 "(e.g. 'trapped rook', 'back-rank weakness', 'wrong bishop', 'missed zwischenzug')"},
        "category": {"type": "string", "enum": ["tactic", "endgame-technique", "positional", "king-safety",
                     "material", "none-its-fine"]},
        "concrete": {"type": "boolean", "description": "Is there a concrete tactic/win, vs a vague "
                     "positional edge?"},
        "why": {"type": "string", "description": "Under 20 words, cite squares."},
    },
    "required": ["idea", "category", "concrete", "why"],
}


def build_prompt(m, phase):
    side = "White" if m.mover == chess.WHITE else "Black"
    return "\n".join([
        "A player made a mistake here that our automated coach could not label. Tell us what the single",
        "most important teachable idea is — what should a coach actually say?",
        "",
        f"FEN: {m.fen_before}",
        f"Phase: {phase}.  {side} to move, and played: {m.played_san}",
        f"Engine's best move: {m.best_san}",
        f"BEST line: {' '.join(m.best_line_san) or '(none)'}",
        f"PUNISHMENT line after {m.played_san}: {' '.join(m.refutation_san) or '(none)'}",
        "",
        "Independently computed board facts (trust over your own counting):",
        json.dumps(board_facts(m), indent=1),
        "",
        "Name the teachable idea in 2-5 words + a category. If the move is basically fine or the point is",
        "just 'a slightly better move', say category none-its-fine. Be concrete; cite squares.",
    ])


def ask(prompt, model, effort, timeout=240):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(SCHEMA, fh); sp = fh.name
    try:
        r = subprocess.run([AGY, "-p", prompt, "--json-schema", sp, "--output-format", "json",
                            "--model", model, "--effort", effort],
                           capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
        if r.returncode != 0:
            return None, (r.stderr or "")[-160:]
        return json.loads(r.stdout.strip().splitlines()[-1]).get("structured_output"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        os.unlink(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", default="/tmp/fifa_enrich.json")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--scan", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="naked.json")
    ap.add_argument("--forced", action="store_true")
    args = ap.parse_args()

    enrich = json.load(open(args.enrich))
    # collect naked mistakes, bucketed by phase
    buckets = {"Opening": [], "Middlegame": [], "Endgame": []}
    scanned = 0
    for k, e in enrich.items():
        scanned += 1
        if scanned > args.scan:
            break
        try:
            fen, uci = k.split("|", 1); m = from_fifa_entry(fen, uci, e)
            tags = tag_mistake_full(m, with_maia=False, classification=None)["tags"]
        except Exception:
            continue
        if any(t["direction"] in TEACH for t in tags):
            continue
        if args.forced:  # true DETECTOR gap: still naked when the game says it IS a mistake
            ft = tag_mistake_full(m, with_maia=False, classification="blunder")["tags"]
            if any(t["direction"] in TEACH for t in ft):
                continue
        phase = next((t["label"] for t in tags if t["label"] in buckets), "Middlegame")
        buckets[phase].append((k, e, m, phase))
    print(f"naked found: " + ", ".join(f"{p}={len(v)}" for p, v in buckets.items()), flush=True)

    # stratified sample: endgame is the worst gap, oversample it; deterministic stride (no RNG)
    quota = {"Opening": args.n // 6, "Middlegame": args.n // 2, "Endgame": args.n - args.n // 6 - args.n // 2}
    picked = []
    for p, q in quota.items():
        pool = buckets[p]
        if not pool:
            continue
        stride = max(1, len(pool) // max(q, 1))
        picked += pool[::stride][:q]
    print(f"judging {len(picked)} naked positions ...", flush=True)

    def one(item):
        k, e, m, phase = item
        out, err = ask(build_prompt(m, phase), args.model, args.effort)
        return {"key": k, "phase": phase, "played": m.played_san, "best": m.best_san,
                "idea": (out or {}).get("idea"), "category": (out or {}).get("category"),
                "concrete": (out or {}).get("concrete"), "why": (out or {}).get("why"), "error": err}

    raw = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, it) for it in picked]
        for i, f in enumerate(cf.as_completed(futs), 1):
            raw.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(picked)}", flush=True)
                json.dump({"raw": raw}, open(args.out, "w"), indent=1)

    # aggregate
    ideas = collections.Counter()
    cats = collections.Counter()
    examples = collections.defaultdict(list)
    for r in raw:
        if r.get("error") or not r.get("idea"):
            continue
        idea = r["idea"].strip().lower()
        ideas[idea] += 1
        cats[r["category"]] += 1
        if len(examples[idea]) < 3:
            examples[idea].append({"key": r["key"][:44], "phase": r["phase"], "why": r["why"]})
    out = {"raw": raw, "top_ideas": [{"idea": k, "count": c, "examples": examples[k]}
                                     for k, c in ideas.most_common(25)],
           "categories": dict(cats.most_common())}
    json.dump(out, open(args.out, "w"), indent=1)

    print("\n" + "=" * 74)
    print("CATEGORY split of naked mistakes:")
    for c, n in cats.most_common():
        print(f"   {n:3d}  {c}")
    print("\nMOST-CITED teachable ideas on UNTAGGED positions (candidate #106 detectors):")
    for mi in out["top_ideas"]:
        print(f"   {mi['count']:2d}  {mi['idea']}")
    print(f"\nwrote {args.out}  ({len(raw)} judged)")


if __name__ == "__main__":
    main()
