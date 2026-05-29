# Maia 3 SAE Taxonomy Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Maia 3 SAE (2048 k=32 v2) feature taxonomy so feature titles, categories, and chips are all derived from the *accurate* per-feature descriptions — fixing the lossy chip-first pipeline that produced junk-drawer categories.

**Architecture:** The existing 2,000-feature label file already contains accurate `description` (87% specific, verified against the board) and `label` (91% specific one-liner) fields. Only the 2-4 word `chip` and the free-text `categories` (2,463 distinct values) are broken. We rebuild in the correct order — **title → categorize → chip** — so compression happens last, with category context, instead of first. No re-pulling Opus English, no notebook, no re-synthesis from positions: all source data is local and verified.

**Tech Stack:** Python 3, `python-chess` (board verification), `boto3` Bedrock (Sonnet 4.6 for categorization + chip generation), local JSON. Runs on the research account (`AWS_PROFILE=default`).

---

## Background: why this plan exists (read before starting)

The shipped v2 labeling pipeline ran **chip-first**:
1. Pass 1 (Opus, 19,342 position analyses) → accurate per-position English. ✓
2. Pass 2 (`label_features_pass2.py`) → synthesized per-feature `description` + `label` (accurate) **and** a 2-4 word `chip` whose prompt literally said *"Keep it BROAD... not a specific tactic"*.
3. Categories were assigned from the **lossy chips**, collapsing distinct features into junk drawers ("Missed Tactics" = 372 features spanning every piece type).

**Verified findings (this session):**
- `description` field: 87% carry specific move-counts/squares; spot-checked 4 features against the board — descriptions are *accurate* (feature 0 desc "8/10 a/b-pawn pushes" → board confirms exactly 8 a/b-pawns + Qa5 + f6).
- `label` field: 91% specific, median 146 chars, reads as a clean feature title.
- `chip` field: generic on ~685 features (the bug).
- `categories` field: 2,463 distinct values for 2,000 features — never controlled, unusable for grouping.
- Decoder geometry: v2 features are near-orthogonal (mean pairwise cosine 0.000) — genuinely distinct features, not duplicates.

**Source data locations (all verified present locally):**
- Labels (source of truth for descriptions): `/tmp/maia3_feature_labels_latest.json` (== S3 `sae/labels/maia3_feature_labels_opus.json`, 2,000 features). **Re-fetch into repo, see Task 0.**
- Profile (top-20 positions/feature, FENs+UCIs): `/tmp/l2_feature_profiles_v2.json` (== S3 `sae/maia3/l2_feature_profiles_v2.json`).
- Full Opus English (19,342 analyses, 100% top-10 coverage): `/tmp/opus_full.json` (from chess-poc `all_positions_labeled_opus.json`; **S3 `..._final.json` is TRUNCATED to 10,648 — do not use it**).
- SAE weights (decoder, for geometry checks): `/tmp/maia3_sae_v2.pt` (== S3 `sae/maia3/maia3_sae_diff_2048_k32_v2.pt`).

**Out of scope (explicitly NOT doing):**
- Re-running Pass 1 or Pass 2. Descriptions are good; don't regenerate them.
- Re-training or re-profiling the SAE.
- Deploying to production (`ship_sae_version.py`) — that's a separate downstream step.
- Fixing the truncated S3 `_final.json` (note it in S3_INVENTORY.md, Task 7).

---

## File Structure

All new code lives in `scripts/sae/taxonomy/` (new dir), outputs in `output/taxonomy_v2/` (new dir).

- `scripts/sae/taxonomy/build_evidence.py` — assembles per-feature evidence packet (description + label + structural fingerprint + verification flags). Pure local computation, no LLM.
- `scripts/sae/taxonomy/verify_descriptions.py` — board-verifies each description's claims; flags features whose description doesn't match the board. Reusable predicate module.
- `scripts/sae/taxonomy/categorize.py` — two-stage: (1) propose a controlled category vocabulary from all descriptions, (2) assign each feature to one category. Sonnet.
- `scripts/sae/taxonomy/generate_chips.py` — category-aware chip generation from description. Sonnet.
- `scripts/sae/taxonomy/assemble.py` — merges everything into the final `taxonomy_v2.json`.
- `output/taxonomy_v2/` — `evidence.json`, `verification.json`, `category_vocab.json`, `assignments.json`, `chips.json`, `taxonomy_v2.json`.
- `tests/taxonomy/test_*.py` — unit tests per module.

---

## Task 0: Stage source data into the repo + scaffold

**Files:**
- Create: `scripts/sae/taxonomy/` (dir), `output/taxonomy_v2/` (dir), `tests/taxonomy/` (dir)
- Create: `output/taxonomy_v2/.gitignore` (ignore the large staged inputs, keep outputs)

- [ ] **Step 1: Create directories**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
mkdir -p scripts/sae/taxonomy output/taxonomy_v2 tests/taxonomy
```

- [ ] **Step 2: Stage the three local source files into a stable working location**

The `/tmp` copies are session-scoped and already verified. Copy into a gitignored working dir so the pipeline has stable paths.

```bash
mkdir -p output/taxonomy_v2/_inputs
cp /tmp/maia3_feature_labels_latest.json output/taxonomy_v2/_inputs/feature_labels.json
cp /tmp/l2_feature_profiles_v2.json       output/taxonomy_v2/_inputs/profiles.json
cp /tmp/opus_full.json                    output/taxonomy_v2/_inputs/opus_english.json
```

- [ ] **Step 3: Verify staged files have the expected counts**

```bash
python3 -c "
import json
L=json.load(open('output/taxonomy_v2/_inputs/feature_labels.json'))
P=json.load(open('output/taxonomy_v2/_inputs/profiles.json'))
O=json.load(open('output/taxonomy_v2/_inputs/opus_english.json'))
assert len(L)==2000, f'labels {len(L)}'
assert len(P)==2048, f'profiles {len(P)}'
assert len(O)==19342, f'opus {len(O)}'
print('OK: labels=2000 profiles=2048 opus=19342')
"
```
Expected: `OK: labels=2000 profiles=2048 opus=19342`

- [ ] **Step 4: Write .gitignore for the large inputs**

```bash
cat > output/taxonomy_v2/.gitignore <<'EOF'
_inputs/
EOF
```

- [ ] **Step 5: Commit scaffold**

```bash
git add scripts/sae/taxonomy output/taxonomy_v2/.gitignore docs/superpowers/plans/2026-05-29-relabel-maia3-taxonomy.md
git commit -m "taxonomy: scaffold rebuild pipeline + plan"
```

---

## Task 1: Structural fingerprint + description verifier

This is the deterministic core. Produces, per feature: the chess fingerprint (dominant piece, to-square cluster, phase, capture/check rates) and a verification verdict on whether the `description`'s headline claim matches the board. Used by categorize.py to give the LLM hard facts alongside prose, and to flag features whose description is itself suspect.

**Files:**
- Create: `scripts/sae/taxonomy/verify_descriptions.py`
- Test: `tests/taxonomy/test_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/taxonomy/test_verify.py
import chess
from scripts.sae.taxonomy.verify_descriptions import move_fingerprint, FEAT_INPUTS

def test_fingerprint_pawn_feature():
    # Feature 0 is verified: 8/10 a/b-pawn pushes + Qa5 + f6
    examples = FEAT_INPUTS  # injected fixture below
    fps = [("b3", "8/8/8/8/8/1P6/P1PPPPPP/RNBQKBNR w KQkq - 0 1")]  # placeholder
    # Use a known position: 1.b3 from start
    board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fp = move_fingerprint([(board_fen, "b2b3")])
    assert fp["dom_piece"] == "pawn"
    assert fp["dom_frac"] == 1.0
    assert fp["cap_rate"] == 0.0
    assert fp["check_rate"] == 0.0

def test_fingerprint_capture_detection():
    # A capture move: exd5
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    fp = move_fingerprint([(fen, "e4d5")])
    assert fp["cap_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research && python3 -m pytest tests/taxonomy/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module not yet written).

- [ ] **Step 3: Write the implementation**

```python
# scripts/sae/taxonomy/verify_descriptions.py
"""Deterministic structural fingerprint + description verification.

For each feature, compute the chess fingerprint of its top-N profile
positions, and check whether the feature's `description` headline claim
(piece type, capture, check, hanging) matches what the board actually shows.
"""
import re
import math
from collections import Counter

import chess

FEAT_INPUTS = None  # test fixture hook; real callers pass positions explicitly


def move_fingerprint(positions):
    """positions: list of (fen, uci). Returns structural fingerprint dict."""
    pieces = Counter(); to_sq = Counter(); from_sq = Counter()
    caps = 0; checks = 0; promos = 0; n = 0
    for fen, uci in positions:
        try:
            b = chess.Board(fen)
            mv = chess.Move.from_uci(uci)
        except Exception:
            continue
        n += 1
        pc = b.piece_at(mv.from_square)
        pieces[chess.piece_name(pc.piece_type) if pc else "?"] += 1
        to_sq[chess.square_name(mv.to_square)] += 1
        from_sq[chess.square_name(mv.from_square)] += 1
        caps += 1 if b.is_capture(mv) else 0
        checks += 1 if b.gives_check(mv) else 0
        promos += 1 if mv.promotion else 0
    if n == 0:
        return None
    dom_piece, dom_n = pieces.most_common(1)[0]

    def entropy(c):
        t = sum(c.values())
        return -sum((v / t) * math.log2(v / t) for v in c.values()) if t else 0.0

    return {
        "n": n,
        "dom_piece": dom_piece,
        "dom_frac": dom_n / n,
        "piece_dist": dict(pieces.most_common()),
        "to_sq_top": dict(to_sq.most_common(5)),
        "to_sq_entropy": round(entropy(to_sq), 2),
        "cap_rate": caps / n,
        "check_rate": checks / n,
        "promo_rate": promos / n,
    }


# Claim keywords that appear in chips/descriptions, mapped to board predicates.
_CLAIM_KW = {
    "check": ["check"],
    "capture": ["captur", "takes", "grab", "greedy", "snatch"],
    "promote": ["promot", "queens", "underpromo"],
    "fork": ["fork"],
}
_SQ_RE = re.compile(r"\b[a-h][1-8]\b")
_PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def verify_description(description, fingerprint):
    """Check the description's structural claims against the fingerprint.

    Returns dict: {verdict, checks: {claim: (claimed, observed_rate, ok)}}.
    verdict in {supported, partial, contradicted, unverifiable}.
    Mechanism/tactical claims (refutations) are NOT checked here — those came
    from Stockfish depth-18 in Pass 1 and are trusted. We only check the
    surface move facts the board can confirm.
    """
    if not description or fingerprint is None:
        return {"verdict": "unverifiable", "checks": {}}
    low = description.lower()
    checks = {}

    # piece claim: does the description name the dominant piece?
    claimed_pieces = [p for p in _PIECES if p in low]
    if claimed_pieces:
        observed = fingerprint["dom_frac"]
        ok = any(
            fingerprint["piece_dist"].get(p, 0) / fingerprint["n"] >= 0.4
            for p in claimed_pieces
        )
        checks["piece"] = (claimed_pieces, round(observed, 2), ok)

    # capture / check / promote rate claims
    rate_map = {
        "capture": fingerprint["cap_rate"],
        "check": fingerprint["check_rate"],
        "promote": fingerprint["promo_rate"],
    }
    thresh = {"capture": 0.4, "check": 0.4, "promote": 0.3}
    for claim, kws in _CLAIM_KW.items():
        if claim not in rate_map:
            continue
        if any(w in low for w in kws):
            r = rate_map[claim]
            checks[claim] = (True, round(r, 2), r >= thresh[claim])

    if not checks:
        return {"verdict": "unverifiable", "checks": {}}
    oks = [v[-1] for v in checks.values()]
    if all(oks):
        verdict = "supported"
    elif not any(oks):
        verdict = "contradicted"
    else:
        verdict = "partial"
    return {"verdict": verdict, "checks": checks}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research && python3 -m pytest tests/taxonomy/test_verify.py -v`
Expected: PASS (2 tests). If `scripts` isn't importable, add `conftest.py` (next step).

- [ ] **Step 5: Add conftest for import path**

```python
# tests/taxonomy/conftest.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```
Also create empty `scripts/__init__.py`, `scripts/sae/__init__.py`, `scripts/sae/taxonomy/__init__.py` if they don't exist:
```bash
touch scripts/__init__.py scripts/sae/__init__.py scripts/sae/taxonomy/__init__.py
```
Re-run the test, expect PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/sae/taxonomy/verify_descriptions.py scripts/sae/taxonomy/__init__.py scripts/sae/__init__.py scripts/__init__.py tests/taxonomy/
git commit -m "taxonomy: deterministic fingerprint + description verifier"
```

---

## Task 2: Build the per-feature evidence packet

Assembles everything categorize.py needs per feature, into one JSON. No LLM. Joins labels + profile + fingerprint + verification.

**Files:**
- Create: `scripts/sae/taxonomy/build_evidence.py`
- Test: `tests/taxonomy/test_evidence.py`
- Output: `output/taxonomy_v2/evidence.json`, `output/taxonomy_v2/verification.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/taxonomy/test_evidence.py
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_evidence_built(tmp_path):
    # Run the builder against staged inputs, check structure of one feature.
    out = tmp_path / "evidence.json"
    ver = tmp_path / "verification.json"
    r = subprocess.run([sys.executable,
        os.path.join(ROOT, "scripts/sae/taxonomy/build_evidence.py"),
        "--labels", os.path.join(ROOT, "output/taxonomy_v2/_inputs/feature_labels.json"),
        "--profiles", os.path.join(ROOT, "output/taxonomy_v2/_inputs/profiles.json"),
        "--out-evidence", str(out), "--out-verification", str(ver)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ev = json.load(open(out))
    assert len(ev) >= 1990  # ~2000 features minus INSUFFICIENT/ERROR
    f0 = ev["0"]
    assert "description" in f0 and "label" in f0
    assert "fingerprint" in f0 and f0["fingerprint"]["dom_piece"] == "pawn"
    assert "verification" in f0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/taxonomy/test_evidence.py -v`
Expected: FAIL (builder not written).

- [ ] **Step 3: Write the builder**

```python
# scripts/sae/taxonomy/build_evidence.py
"""Assemble per-feature evidence: description + label + fingerprint + verification.

Pure local computation. Skips features whose chip is INSUFFICIENT/ERROR or
confidence==0 (4 features). Output keyed by feature_id (string).
"""
import argparse
import json

from verify_descriptions import move_fingerprint, verify_description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out-evidence", required=True)
    ap.add_argument("--out-verification", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    profiles = json.load(open(args.profiles))

    evidence = {}
    verification = {}
    for fid, lab in labels.items():
        chip = lab.get("chip", "")
        if chip in ("INSUFFICIENT", "ERROR") or lab.get("confidence", 0) == 0:
            continue
        prof = profiles.get(fid, {})
        positions = [(ex["fen"], ex["uci"]) for ex in prof.get("examples", [])[: args.top_n]]
        fp = move_fingerprint(positions)
        ver = verify_description(lab.get("description", ""), fp)
        evidence[fid] = {
            "feature_id": int(fid),
            "label": lab.get("label", ""),
            "description": lab.get("description", ""),
            "old_chip": chip,
            "fingerprint": fp,
            "verification": ver,
        }
        verification[fid] = ver
    json.dump(evidence, open(args.out_evidence, "w"), indent=2)
    json.dump(verification, open(args.out_verification, "w"), indent=2)
    print(f"Built evidence for {len(evidence)} features.")
    # verdict histogram
    from collections import Counter
    h = Counter(v["verdict"] for v in verification.values())
    print("verdicts:", dict(h))


if __name__ == "__main__":
    main()
```

Note: `build_evidence.py` imports `verify_descriptions` as a sibling — run it with `cwd` set to its own dir, OR add the dir to `sys.path`. To keep the test simple, add at the top of `build_evidence.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
(Place this before `from verify_descriptions import ...`.)

- [ ] **Step 4: Run test, verify PASS**

Run: `python3 -m pytest tests/taxonomy/test_evidence.py -v`
Expected: PASS.

- [ ] **Step 5: Build the real evidence file + eyeball the verdict histogram**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
python3 scripts/sae/taxonomy/build_evidence.py \
  --labels output/taxonomy_v2/_inputs/feature_labels.json \
  --profiles output/taxonomy_v2/_inputs/profiles.json \
  --out-evidence output/taxonomy_v2/evidence.json \
  --out-verification output/taxonomy_v2/verification.json
```
Expected: `Built evidence for ~1996 features.` + a verdicts histogram. **CHECKPOINT — review with human:** if `contradicted` is a large fraction (>15%), the descriptions themselves may be less reliable than we measured; pause and inspect 10 contradicted features before proceeding. (Spot-checks this session suggest most "contradicted" are verifier crudeness, not bad descriptions — see Background.)

- [ ] **Step 6: Commit**

```bash
git add scripts/sae/taxonomy/build_evidence.py tests/taxonomy/test_evidence.py output/taxonomy_v2/evidence.json output/taxonomy_v2/verification.json
git commit -m "taxonomy: per-feature evidence packet + verification"
```

---

## Task 3: Propose a controlled category vocabulary

The current `categories` field has 2,463 distinct values — unusable. We need a *small, controlled* vocabulary (target 12–20 categories) derived from the actual descriptions, not the chips. One LLM call that sees a representative sample of descriptions and proposes the vocabulary; human approves before assignment.

**Files:**
- Create: `scripts/sae/taxonomy/categorize.py` (vocab-propose mode)
- Output: `output/taxonomy_v2/category_vocab.json`

- [ ] **Step 1: Write the vocabulary proposer**

```python
# scripts/sae/taxonomy/categorize.py
"""Two modes:
  propose-vocab : sample descriptions -> Sonnet proposes a controlled category list
  assign        : assign each feature to one category from the approved vocab

Run on research account: AWS_PROFILE=default.
"""
import argparse
import json
import re

import boto3

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"


def _client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _invoke(client, prompt, max_tokens=2000):
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(resp["body"].read())
    text = result["content"][0]["text"]
    st, en = text.find("{"), text.rfind("}") + 1
    return json.loads(text[st:en])


def propose_vocab(evidence, client, sample_size=300, seed_stride=None):
    """Sample descriptions evenly across feature IDs (deterministic), ask for a
    controlled vocabulary of 12-20 coaching categories grounded in chess."""
    fids = sorted(evidence, key=lambda k: int(k))
    stride = max(1, len(fids) // sample_size)
    sample = fids[::stride][:sample_size]
    lines = []
    for fid in sample:
        e = evidence[fid]
        fp = e["fingerprint"] or {}
        lines.append(f"- {e['label']} [piece={fp.get('dom_piece','?')}, phase-mix]")
    blob = "\n".join(lines)
    prompt = f"""You are designing a coaching taxonomy for chess blunder patterns detected by an SAE on Maia (human-move-prediction) model activations. Below are {len(sample)} feature descriptions sampled evenly from ~2000 features.

Design a CONTROLLED category vocabulary of 12-20 categories. Each category is a TYPE OF MISTAKE a coach would name (e.g. "Hanging a piece", "King walks into danger", "Premature pawn push", "Greedy capture"). Categories must be:
- Mutually distinct (a feature belongs to exactly one)
- Grounded in chess mechanism, not vague ("Missed Tactics" is BANNED — too broad; split it into the actual mechanisms)
- Player-facing (a 1500-2000 player should recognize the mistake)

Feature descriptions:
{blob}

Respond ONLY with JSON:
{{"categories": [{{"id": "snake_case_id", "name": "Player-Facing Name", "definition": "one sentence: what mistake this is", "examples": ["short example moves/patterns"]}}]}}"""
    return _invoke(client, prompt, max_tokens=4000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["propose-vocab", "assign"], required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--vocab", help="approved vocab json (for assign mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-size", type=int, default=300)
    args = ap.parse_args()

    evidence = json.load(open(args.evidence))
    client = _client()

    if args.mode == "propose-vocab":
        vocab = propose_vocab(evidence, client, sample_size=args.sample_size)
        json.dump(vocab, open(args.out, "w"), indent=2)
        print(f"Proposed {len(vocab['categories'])} categories -> {args.out}")
        for c in vocab["categories"]:
            print(f"  {c['id']:28s} {c['name']}")
    else:
        raise SystemExit("assign mode implemented in Task 4")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run vocabulary proposal (requires AWS creds)**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
AWS_PROFILE=default python3 scripts/sae/taxonomy/categorize.py \
  --mode propose-vocab \
  --evidence output/taxonomy_v2/evidence.json \
  --out output/taxonomy_v2/category_vocab.json \
  --sample-size 300
```
Expected: 12–20 categories printed, written to `category_vocab.json`. If `ExpiredTokenException`: refresh creds (`ada credentials update --account=140023406996 ...`, see CLAUDE.md) and retry.

- [ ] **Step 3: CHECKPOINT — human reviews the vocabulary**

Print the full vocab with definitions. **Do not proceed to assignment until the human approves the category list.** This is the single most important decision in the rebuild — categories are the ship artifact's backbone. Edit `category_vocab.json` by hand if needed (add/merge/rename categories).

- [ ] **Step 4: Commit the approved vocabulary**

```bash
git add scripts/sae/taxonomy/categorize.py output/taxonomy_v2/category_vocab.json
git commit -m "taxonomy: controlled category vocabulary (human-approved)"
```

---

## Task 4: Assign each feature to a category

For each feature, give Sonnet the description + fingerprint + the approved vocabulary, and ask for exactly one category. Batched concurrently. This is the main fan-out (~2,000 calls).

**Files:**
- Modify: `scripts/sae/taxonomy/categorize.py` (implement `assign` mode)
- Test: `tests/taxonomy/test_assign.py`
- Output: `output/taxonomy_v2/assignments.json`

- [ ] **Step 1: Write the failing test (assignment parsing, mocked)**

```python
# tests/taxonomy/test_assign.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts/sae/taxonomy"))
from categorize import parse_assignment

def test_parse_valid_category():
    vocab_ids = {"hanging_piece", "king_danger"}
    out = parse_assignment('{"category": "hanging_piece", "confidence": 88}', vocab_ids)
    assert out["category"] == "hanging_piece"
    assert out["confidence"] == 88

def test_parse_rejects_unknown_category():
    vocab_ids = {"hanging_piece"}
    out = parse_assignment('{"category": "made_up", "confidence": 99}', vocab_ids)
    assert out["category"] is None  # invalid -> None, flagged for review
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/taxonomy/test_assign.py -v`
Expected: FAIL (`parse_assignment` not defined).

- [ ] **Step 3: Implement `parse_assignment` + `assign` mode**

Add to `scripts/sae/taxonomy/categorize.py`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_assignment(text, vocab_ids):
    st, en = text.find("{"), text.rfind("}") + 1
    try:
        obj = json.loads(text[st:en])
    except Exception:
        return {"category": None, "confidence": 0}
    cat = obj.get("category")
    if cat not in vocab_ids:
        cat = None
    return {"category": cat, "confidence": int(obj.get("confidence", 0))}


def assign_one(client, e, vocab, vocab_block):
    fp = e["fingerprint"] or {}
    prompt = f"""Assign this chess-blunder SAE feature to EXACTLY ONE category from the controlled vocabulary.

FEATURE:
description: {e['description']}
dominant piece: {fp.get('dom_piece','?')} ({round(fp.get('dom_frac',0)*100)}% of moves)
capture rate: {round(fp.get('cap_rate',0)*100)}%  check rate: {round(fp.get('check_rate',0)*100)}%
top squares: {fp.get('to_sq_top',{})}

VOCABULARY:
{vocab_block}

Pick the single best-fitting category id. Respond ONLY with JSON:
{{"category": "<id from vocabulary>", "confidence": 0-100}}"""
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(resp["body"].read())
    return parse_assignment(result["content"][0]["text"], {c["id"] for c in vocab["categories"]})


def run_assign(evidence, vocab, out_path, threads=12):
    client = _client()
    vocab_block = "\n".join(
        f"- {c['id']}: {c['name']} — {c['definition']}" for c in vocab["categories"]
    )
    vocab_ids = {c["id"] for c in vocab["categories"]}
    results = {}
    # resume support
    import os
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    todo = [fid for fid in evidence if fid not in results]
    print(f"Assigning {len(todo)} features ({len(results)} already done), {threads} threads...")

    def work(fid):
        return fid, assign_one(client, evidence[fid], vocab, vocab_block)

    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(work, fid): fid for fid in todo}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                _, res = fut.result()
            except Exception as ex_err:
                res = {"category": None, "confidence": 0, "error": str(ex_err)[:80]}
            results[fid] = res
            done += 1
            if done % 100 == 0:
                json.dump(results, open(out_path, "w"), indent=2)
                print(f"  {done}/{len(todo)}")
    json.dump(results, open(out_path, "w"), indent=2)
    return results
```

Wire into `main()`'s `assign` branch:
```python
    else:  # assign
        vocab = json.load(open(args.vocab))
        results = run_assign(evidence, vocab, args.out)
        from collections import Counter
        h = Counter(r["category"] for r in results.values())
        print("category counts:", dict(h.most_common()))
        unassigned = sum(1 for r in results.values() if r["category"] is None)
        print(f"unassigned (need review): {unassigned}")
```

- [ ] **Step 4: Run test, verify PASS**

Run: `python3 -m pytest tests/taxonomy/test_assign.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the assignment fan-out**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
AWS_PROFILE=default python3 scripts/sae/taxonomy/categorize.py \
  --mode assign \
  --evidence output/taxonomy_v2/evidence.json \
  --vocab output/taxonomy_v2/category_vocab.json \
  --out output/taxonomy_v2/assignments.json
```
Expected: progress every 100; final `category counts:` histogram + unassigned count. Resumable (re-run continues from saved progress).

- [ ] **Step 6: CHECKPOINT — sanity-check the distribution**

```bash
python3 -c "
import json
from collections import Counter
a=json.load(open('output/taxonomy_v2/assignments.json'))
v=json.load(open('output/taxonomy_v2/category_vocab.json'))
names={c['id']:c['name'] for c in v['categories']}
h=Counter(r['category'] for r in a.values())
for cid,n in h.most_common():
    print(f'{n:>4}  {names.get(cid,cid)}')
print('unassigned:', sum(1 for r in a.values() if r['category'] is None))
"
```
**Review with human:** No single category should hold >25% of features (that's a new junk drawer). If one does, the vocabulary needs splitting — go back to Task 3. Compare against the OLD junk drawers (Missed Tactics 372, Ignoring Free Pieces 130) — those should now be distributed across multiple mechanism-based categories.

- [ ] **Step 7: Commit**

```bash
git add scripts/sae/taxonomy/categorize.py tests/taxonomy/test_assign.py output/taxonomy_v2/assignments.json
git commit -m "taxonomy: assign features to controlled categories"
```

---

## Task 5: Category-aware chip generation

Now generate the short chip — last, with the category as context. The chip can be short because the category header carries meaning. Reads description + assigned category.

**Files:**
- Create: `scripts/sae/taxonomy/generate_chips.py`
- Test: `tests/taxonomy/test_chips.py`
- Output: `output/taxonomy_v2/chips.json`

- [ ] **Step 1: Write the failing test (chip validation, mocked)**

```python
# tests/taxonomy/test_chips.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts/sae/taxonomy"))
from generate_chips import validate_chip

def test_chip_rejects_generic_frame():
    # The exact failure mode we're fixing: "X ignores tactics" type chips
    assert validate_chip("Quiet move ignores tactics") is False
    assert validate_chip("Slow move misses tactics") is False

def test_chip_accepts_specific():
    assert validate_chip("Bishop hangs on g4") is True
    assert validate_chip("a-pawn push, ignores center break") is True
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/taxonomy/test_chips.py -v`
Expected: FAIL (`validate_chip` undefined).

- [ ] **Step 3: Implement chip generation + validation**

```python
# scripts/sae/taxonomy/generate_chips.py
"""Generate a short, specific chip per feature — category-aware, description-grounded.

The chip is the LAST step and may be short because the category header carries
context. We explicitly reject the generic frame ("X ignores/misses tactics")
that broke the original pipeline.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"

# The generic frame we are eliminating.
_GENERIC = re.compile(
    r"\b(ignor\w*|miss\w*|wast\w*)\b.*\b(tactic|tactics|tempo|urgency|crisis|threat)\b",
    re.I,
)


def validate_chip(chip):
    """False if the chip falls back into the banned generic frame."""
    if not chip or len(chip) > 60:
        return False
    return _GENERIC.search(chip) is None


def _client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def gen_chip(client, e, category_name):
    prompt = f"""Write a SHORT chip (3-6 words) naming this specific chess mistake. It will be displayed under the category header "{category_name}", so it does NOT need to repeat the category — it should add the SPECIFIC detail.

description: {e['description']}

BANNED: generic phrases like "ignores tactics", "misses tactics", "wastes tempo". Name the CONCRETE pattern (piece, square, mechanism).
GOOD examples: "Bishop hangs on g4", "a-pawn push over center break", "Queen to rim, allows fork".

Respond ONLY with JSON: {{"chip": "3-6 words"}}"""
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 60,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(resp["body"].read())
    text = result["content"][0]["text"]
    st, en = text.find("{"), text.rfind("}") + 1
    try:
        chip = json.loads(text[st:en]).get("chip", "").strip()
    except Exception:
        chip = ""
    return chip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    evidence = json.load(open(args.evidence))
    assignments = json.load(open(args.assignments))
    vocab = json.load(open(args.vocab))
    names = {c["id"]: c["name"] for c in vocab["categories"]}
    client = _client()

    import os
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [fid for fid in evidence if fid not in results]
    print(f"Generating chips for {len(todo)} features...")

    def work(fid):
        cat = assignments.get(fid, {}).get("category")
        cat_name = names.get(cat, "Uncategorized")
        chip = ""
        for _ in range(args.max_retries + 1):
            chip = gen_chip(client, evidence[fid], cat_name)
            if validate_chip(chip):
                break
        return fid, {"chip": chip, "valid": validate_chip(chip), "category": cat}

    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(work, fid): fid for fid in todo}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                _, res = fut.result()
            except Exception as e:
                res = {"chip": "", "valid": False, "error": str(e)[:80]}
            results[fid] = res
            done += 1
            if done % 100 == 0:
                json.dump(results, open(args.out, "w"), indent=2)
                print(f"  {done}/{len(todo)}")
    json.dump(results, open(args.out, "w"), indent=2)
    invalid = sum(1 for r in results.values() if not r.get("valid"))
    print(f"Done. Still-generic after retries (need review): {invalid}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify PASS**

Run: `python3 -m pytest tests/taxonomy/test_chips.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run chip generation**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
AWS_PROFILE=default python3 scripts/sae/taxonomy/generate_chips.py \
  --evidence output/taxonomy_v2/evidence.json \
  --assignments output/taxonomy_v2/assignments.json \
  --vocab output/taxonomy_v2/category_vocab.json \
  --out output/taxonomy_v2/chips.json
```
Expected: progress every 100; final count of still-generic chips. Resumable.

- [ ] **Step 6: CHECKPOINT — compare old vs new chips on the worst features**

```bash
python3 -c "
import json
lab=json.load(open('output/taxonomy_v2/_inputs/feature_labels.json'))
chips=json.load(open('output/taxonomy_v2/chips.json'))
# show 15 features whose OLD chip had the generic frame
import re
G=re.compile(r'(ignor|miss|wast).*(tactic|tempo)', re.I)
shown=0
for fid,c in chips.items():
    old=lab[fid]['chip']
    if G.search(old) and shown<15:
        shown+=1
        print(f'[{fid}] OLD: {old!r}')
        print(f'      NEW: {c[\"chip\"]!r}')
"
```
**Review with human:** new chips should name concrete patterns. If many are still vague, tighten the prompt and re-run (delete chips.json or specific entries first).

- [ ] **Step 7: Commit**

```bash
git add scripts/sae/taxonomy/generate_chips.py tests/taxonomy/test_chips.py output/taxonomy_v2/chips.json
git commit -m "taxonomy: category-aware specific chip generation"
```

---

## Task 6: Assemble the final taxonomy

Merge label + description + fingerprint + category + chip into one shippable `taxonomy_v2.json`, keyed by feature_id, with a category index.

**Files:**
- Create: `scripts/sae/taxonomy/assemble.py`
- Test: `tests/taxonomy/test_assemble.py`
- Output: `output/taxonomy_v2/taxonomy_v2.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/taxonomy/test_assemble.py
import json, os, sys, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_assembled_shape(tmp_path):
    out = tmp_path / "taxonomy_v2.json"
    r = subprocess.run([sys.executable,
        os.path.join(ROOT, "scripts/sae/taxonomy/assemble.py"),
        "--evidence", os.path.join(ROOT, "output/taxonomy_v2/evidence.json"),
        "--assignments", os.path.join(ROOT, "output/taxonomy_v2/assignments.json"),
        "--chips", os.path.join(ROOT, "output/taxonomy_v2/chips.json"),
        "--vocab", os.path.join(ROOT, "output/taxonomy_v2/category_vocab.json"),
        "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = json.load(open(out))
    assert "features" in t and "categories" in t
    f0 = t["features"]["0"]
    for k in ("feature_id", "chip", "title", "description", "category", "fingerprint"):
        assert k in f0, k
```

- [ ] **Step 2: Run test, verify it fails**

Run: `python3 -m pytest tests/taxonomy/test_assemble.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the assembler**

```python
# scripts/sae/taxonomy/assemble.py
"""Merge evidence + assignments + chips + vocab into the final taxonomy_v2.json."""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--chips", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    evidence = json.load(open(args.evidence))
    assignments = json.load(open(args.assignments))
    chips = json.load(open(args.chips))
    vocab = json.load(open(args.vocab))
    names = {c["id"]: c["name"] for c in vocab["categories"]}

    features = {}
    by_cat = defaultdict(list)
    for fid, e in evidence.items():
        cat = assignments.get(fid, {}).get("category")
        features[fid] = {
            "feature_id": int(fid),
            "chip": chips.get(fid, {}).get("chip", ""),
            "title": e["label"],            # the specific one-sentence title (was `label`)
            "description": e["description"],
            "category": cat,
            "category_name": names.get(cat),
            "fingerprint": e["fingerprint"],
            "verification": e["verification"]["verdict"],
            "old_chip": e["old_chip"],
        }
        by_cat[cat].append(int(fid))

    out = {
        "meta": {
            "sae": "maia3_sae_diff_2048_k32_v2",
            "n_features": len(features),
            "n_categories": len(vocab["categories"]),
            "source": "rebuild 2026-05-29 (title->categorize->chip)",
        },
        "categories": vocab["categories"],
        "category_index": {k: sorted(v) for k, v in by_cat.items()},
        "features": features,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Assembled {len(features)} features into {len(by_cat)} categories -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify PASS**

Run: `python3 -m pytest tests/taxonomy/test_assemble.py -v`
Expected: PASS.

- [ ] **Step 5: Build the final taxonomy**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
python3 scripts/sae/taxonomy/assemble.py \
  --evidence output/taxonomy_v2/evidence.json \
  --assignments output/taxonomy_v2/assignments.json \
  --chips output/taxonomy_v2/chips.json \
  --vocab output/taxonomy_v2/category_vocab.json \
  --out output/taxonomy_v2/taxonomy_v2.json
```
Expected: `Assembled ~1996 features into N categories`.

- [ ] **Step 6: Commit**

```bash
git add scripts/sae/taxonomy/assemble.py tests/taxonomy/test_assemble.py output/taxonomy_v2/taxonomy_v2.json
git commit -m "taxonomy: assemble final taxonomy_v2.json"
```

---

## Task 7: Quality gate + docs sync

Validate the rebuilt taxonomy against the old one and update research docs/inventory.

**Files:**
- Create: `scripts/sae/taxonomy/qa_report.py`
- Modify: `output/S3_INVENTORY.md`, `plan.md`, `knowledge.md`, `log.md`

- [ ] **Step 1: Write the QA report (no LLM, deterministic)**

```python
# scripts/sae/taxonomy/qa_report.py
"""Quality gate: compare rebuilt taxonomy to old labels. No LLM."""
import argparse
import json
import re
from collections import Counter

GENERIC = re.compile(r"(ignor|miss|wast).*(tactic|tempo|urgency|crisis)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--old-labels", required=True)
    args = ap.parse_args()
    t = json.load(open(args.taxonomy))
    old = json.load(open(args.old_labels))
    feats = t["features"]

    old_generic = sum(1 for f in feats if GENERIC.search(old[f]["chip"]))
    new_generic = sum(1 for f, v in feats.items() if GENERIC.search(v["chip"]))
    print(f"Generic chips — OLD: {old_generic}  NEW: {new_generic}  (lower is better)")

    cat_counts = Counter(v["category"] for v in feats.values())
    biggest = cat_counts.most_common(1)[0]
    print(f"Largest category: {biggest[1]} features ({biggest[1]/len(feats)*100:.0f}%) — should be <25%")
    print(f"Categories: {len(cat_counts)}")
    print(f"Unassigned: {cat_counts.get(None, 0)}")
    empty = [c["id"] for c in t["categories"] if cat_counts.get(c["id"], 0) == 0]
    if empty:
        print(f"WARNING empty categories: {empty}")

    # gate
    fail = []
    if new_generic > old_generic * 0.2:
        fail.append(f"too many generic chips remain ({new_generic})")
    if biggest[1] / len(feats) > 0.25:
        fail.append(f"junk-drawer category ({biggest[0]} {biggest[1]})")
    if cat_counts.get(None, 0) > len(feats) * 0.05:
        fail.append(f"too many unassigned ({cat_counts.get(None,0)})")
    if fail:
        print("QA FAILED:", "; ".join(fail))
        raise SystemExit(1)
    print("QA PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the QA gate**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
python3 scripts/sae/taxonomy/qa_report.py \
  --taxonomy output/taxonomy_v2/taxonomy_v2.json \
  --old-labels output/taxonomy_v2/_inputs/feature_labels.json
```
Expected: `QA PASSED`. If FAILED on junk-drawer → revisit vocabulary (Task 3). If FAILED on generic chips → tighten chip prompt (Task 5).

- [ ] **Step 3: Update S3_INVENTORY.md**

Add a note under the Maia 3 section recording (a) the truncated `all_positions_labeled_opus_final.json` on S3 (10,648) vs the complete notebook copy (19,342) — flag the S3 copy as stale; (b) the new `taxonomy_v2.json` output location.

- [ ] **Step 4: Update plan.md / knowledge.md / log.md**

- `plan.md`: mark "Run labeling pipeline on H1 model" queue item — note taxonomy rebuilt; next is deploy decision.
- `knowledge.md`: add the finding — *chip-first pipeline collapsed distinct features; descriptions were accurate, chips were lossy; rebuilt title→categorize→chip.* Note the truncated-S3-file gotcha.
- `log.md`: append session narrative (diagnosis → rebuild).

- [ ] **Step 5: Commit + push the complete rebuild**

```bash
git add scripts/sae/taxonomy/qa_report.py output/S3_INVENTORY.md plan.md knowledge.md log.md
git commit -m "taxonomy: QA gate + docs sync for v2 rebuild"
```
(Do NOT auto-push — let Sam push, per CLAUDE.md.)

---

## Self-Review notes

- **Spec coverage:** title (Task 6 promotes `label`→title) ✓; categorize from description (Tasks 3–4) ✓; chip last, category-aware (Task 5) ✓; verification that descriptions are trustworthy (Tasks 1–2) ✓; docs sync (Task 7) ✓.
- **Cost:** ~1 vocab call + ~2,000 assign calls + ~2,000 chip calls (+ retries) on Sonnet 4.6 = ~4,000 short calls. All on research account.
- **Resumability:** assign + chip steps checkpoint every 100 and resume from disk — safe against the Bedrock 503s seen this session.
- **Reversibility:** original labels untouched (`feature_labels.json` is read-only input); all new output in `output/taxonomy_v2/`.
- **Open risk flagged for human:** if Task 2's verdict histogram shows many `contradicted`, the descriptions are less reliable than measured — pause before categorizing on them.
