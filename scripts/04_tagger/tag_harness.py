#!/usr/bin/env python3
"""
Tag harness — the single gate every mistake-tagger change runs before it ships.

This is the tagger counterpart to the coaching-prompt harness in
chess-deck-code/backend/scripts/prompt_lab/eval/. The tagger is DETERMINISTIC (rule-based, no LLM), so
the gate is different in kind from the prompt harness:

  TIER 1 — LOCAL, always-runnable, deterministic (this script's default):
    * regression.py    — curated known-answer positions (POS + NEG per detector). MUST be green.
    * internal-consistency checks over a tag output — the tagger contradicting ITSELF (evidence net vs
      line net, a material label on a mate line). No board reasoning => can't be wrong about it.
  TIER 2 — SCALE AUDIT to find NEW false-positive classes (delegated, needs a corpus):
    * sweep_judge.py   — blind LLM judge (agy, subscription = free) over a stratified sample, ranked by
      FLAG RATE (never raw hits). Finds unknown FP classes; every top-ranked label is board-verified by
      hand before it's called a bug. Corpus = FIFA (remote chess-poc) or any run_corpus.py output.

DECISION RULE — a tagger change ships only if:
  1. TIER 1 is green (regression + consistency), AND
  2. you added a POS *and* a NEG regression case for the exact behavior you changed
     (a change with no new regression case is unshipped — the mutation sweep showed 17/24 line
     detectors could be deleted and stay green), AND
  3. TIER 2 shows no NEW false-positive cluster on the detectors you touched (or you fixed it), AND
  4. you board-verified any candidate finding (candidate != finding).

The full method + why each layer exists: chess-deck-code/knowledge/2026-08-09-tagger-audit-harness.md
+ HARNESS.md next to this file.

USAGE
  python3 tag_harness.py                      # TIER 1 gate: regression (green/red, exits non-zero on fail)
  python3 tag_harness.py --tags out.json      # + internal-consistency checks over a tag-output JSON
  python3 tag_harness.py --judge-cmd          # print the TIER-2 scale-audit command (needs a corpus)
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent

def run_regression() -> bool:
    """Subprocess regression.py; gate on zero [FAIL]. Returns True if green."""
    r = subprocess.run([sys.executable, str(HERE / "regression.py")], capture_output=True, text=True)
    out = r.stdout + r.stderr
    fails = re.findall(r"\[FAIL\] (.+)", out)
    passes = len(re.findall(r"\[PASS\]", out))
    print(f"TIER 1 · regression.py: {passes} pass, {len(fails)} fail")
    for f in fails[:20]:
        print(f"   FAIL: {f}")
    return not fails

def consistency_checks(tags_path: Path) -> int:
    """The high-yield 'tagger contradicts itself' layer (no board reasoning needed).
    Expects run_corpus.py output: [{fen, tags:[{label,evidence,direction}], ...}, ...] or {positions:[...]}."""
    data = json.loads(tags_path.read_text())
    rows = data.get("positions", data) if isinstance(data, dict) else data
    bad = []
    MATERIAL = re.compile(r"\b(Free|Hung|Won|Lost|Missed Free|Hangs?)\b|Exchange|Fork|Material", re.I)
    for row in rows:
        for t in row.get("tags", []):
            label, ev = t.get("label", ""), (t.get("evidence") or "")
            # (a) a material/tactic label whose evidence announces mate (#) — mate should outrank material
            if MATERIAL.search(label) and "#" in ev and "mate" not in label.lower():
                bad.append(f"material label '{label}' with mate evidence: {ev[:60]}")
            # (b) evidence claims a net material figure that is zero/negative for a 'Free/Won' label
            m = re.search(r"net[^\d-]*(-?\d+)", ev, re.I)
            if m and int(m.group(1)) <= 0 and re.search(r"\b(Free|Won)\b", label):
                bad.append(f"'{label}' claims a win but evidence net={m.group(1)}: {ev[:60]}")
    print(f"TIER 1 · internal-consistency over {len(rows)} positions: {len(bad)} self-contradictions")
    for b in bad[:20]:
        print(f"   {b}")
    return len(bad)

def main():
    ap = argparse.ArgumentParser(description="Tagger gate — see HARNESS.md")
    ap.add_argument("--tags", type=Path, help="run_corpus.py output JSON for internal-consistency checks")
    ap.add_argument("--judge-cmd", action="store_true", help="print the TIER-2 scale-audit command")
    a = ap.parse_args()

    green = run_regression()
    contradictions = consistency_checks(a.tags) if a.tags else 0

    if a.judge_cmd:
        print("\nTIER 2 (scale audit — needs a corpus; ranks labels by FLAG RATE, board-verify top hits):")
        print("  python3 sweep_judge.py --enrich /tmp/fifa_enrich.json --per-tag 4 --max-positions 200 --out sweep.json")
        print("  # then rank by flag rate; the FIFA corpus is remote on chess-poc: ~/SageMaker/fifa_blitz/fifa_enrich.json")
        print("  # (validate_judge.py is the gate that proved the judge reads the board, not the label text)")

    ok = green and contradictions == 0
    print("\n" + ("GATE: GREEN — TIER 1 clean." if ok else "GATE: RED — fix before shipping.")
          + " Remember: add POS+NEG regression cases for what you changed; run TIER 2 on touched detectors.")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
