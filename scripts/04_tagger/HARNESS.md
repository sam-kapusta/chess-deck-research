# Tag harness — the reproducible gate for mistake-tagger changes

**Run before shipping any tagger detector change.** This is the tagger counterpart to the coaching-prompt
harness (`chess-deck-code/backend/scripts/prompt_lab/eval/`). The tagger is DETERMINISTIC (rule-based, no
LLM), so the gate is regression + self-consistency + a scale audit for *new* false positives — not a
blind A/B. `tag_harness.py` is the one-command entry; this doc is the process + the discipline.

Full method + the war stories behind every rule: `chess-deck-code/knowledge/2026-08-09-tagger-audit-harness.md`
(read it — it's why these layers exist and what each can actually catch).

## The two tiers

```
TIER 1 — LOCAL, deterministic, always-runnable   (tag_harness.py default; MUST be green to ship)
  regression.py            curated known-answer positions, POS + NEG per detector
  internal-consistency     the tagger contradicting ITSELF (material label on a mate line, evidence
                           net vs claimed win) — no board reasoning, so it can't be wrong about it

TIER 2 — SCALE AUDIT to discover NEW false-positive classes   (needs a corpus; run on touched detectors)
  sweep_judge.py           blind LLM judge (agy, subscription = free) over a stratified sample, ranked
                           by FLAG RATE (never raw hits). Board-verify every top-ranked label by hand.
```

## Run it

```bash
python3 tag_harness.py                 # TIER 1 gate — regression, green/red, exits non-zero on fail
python3 tag_harness.py --tags out.json # + internal-consistency over a run_corpus.py output
python3 tag_harness.py --judge-cmd     # print the TIER-2 command

# the pieces, directly:
python3 regression.py                                              # the known-answer gate
python3 run_corpus.py --sf /tmp/stockfish_data_v2.json --out out.json   # tag a corpus (L1+L2, fast)
python3 sample_tag_examples.py                                    # example dumps — read board facts, find bugs
python3 sweep_judge.py --enrich /tmp/fifa_enrich.json --per-tag 4 --max-positions 200 --out sweep.json
```

## The decision rule — a tagger change ships only if

1. **TIER 1 is green** — `regression.py` all pass + no internal self-contradiction.
2. **You added a POS *and* a NEG regression case** for the exact behavior you changed. A change with no
   new case is unshipped: a mutation sweep found **17 of 24 line detectors could be deleted and the suite
   stays green** — coverage is only where a bug was once found. `regression.py`: `SINGLE_MOVE_CASES` /
   `LINE_CASES`.
3. **TIER 2 shows no new false-positive cluster** on the detectors you touched (flag rate didn't rise),
   or you fixed the one it found. "Positions not having the right tags is as important as getting rid of
   the wrong ones" — check both directions.
4. **You board-verified any candidate finding.** A candidate is not a finding until: reproduced under the
   production path → board facts (computed independently) agree → it violates the taxonomy blurb → you
   tried to refute it → it survived.

## The discipline (every one of these was paid for in false positives)

- **The harness presents; it never judges.** Zero-assertion example dumps found 3 real bugs; the 24k-fire
  rule-based contradiction pass found 0 (+2 false alarms in my own checks). Print board facts computed
  *independently* of the detector; don't read the detector's reasoning back to itself.
- **Use the production entry point** — `tag_adapter.tags_for_eval(eval_dict)` (or `tagger.tag_mistake_full`)
  and **pass `classification=None`.** Forcing `classification="blunder"` overrides the real `win_drop>=10`
  gate and *fabricates* tags on non-mistakes (a Greedy Capture on a 5cp move). Any parameter production
  doesn't set is fiction.
- **Compare the judge against the position's FULL tag set, not one label** — "the tagger missed the mate"
  is only true if NO mate tag fired anywhere (the coach shows a position's tags together). Judging labels
  in isolation produced 4 straight false positives.
- **The judge is club-strength and hallucinates board facts confidently.** Trust its SUSPICION as a
  ranking signal, never its REASONS as evidence. Rank labels by flag rate over a stratified sample;
  aggregation is what makes the per-instance noise tolerable. Every top label still gets hand-verified.
- **Validate or exclude, never render.** Assert every FEN is legal and every ply parses; count exclusions
  out loud. Illegal hand-built FENs and broken fixtures produce confident nonsense.
- **Oversample atypical positions** — mate-available, mover-already-lost, all-checks lines. Detectors get
  tuned on normal middlegames; the bugs live in the odd contexts.

## Corpora

- **Local, known-answer:** `regression.py`'s inline cases — the reproducible gate, needs nothing remote.
  Add to it with every change.
- **Scale, real positions:** `fifa_blitz/fifa_enrich.json` (56,950 positions with 6-ply engine lines) —
  **remote on the `chess-poc` SageMaker notebook** (`~/SageMaker/fifa_blitz/`), NOT local. Adapter:
  `from_fifa_entry` (FIFA stores lines as SAN strings; `from_sf_entry` can't read it → silent empty
  lines). Its 6-ply cap is a known blind spot (see the line-length contract doc). Regenerate a fresh
  corpus with `run_corpus.py` over a `stockfish_data_v2.json`.
- Past audit output is committed: `output/tagger_audits/sweep_2026-08-11.json`, `judge_validation.json`.

## After a change

Edit the source HERE (research), then **`ship_tagger.py`** vendors it into both consumers
(`chess-deck-code/backend/{lambda/tag_moments,worker}/tagger/`); `check_tagger_sync.py` (a CI gate) fails
if the copies drift. A detector change also has to propagate to the frontend data artifacts — see
`chess-deck-code/knowledge/2026-07-17-tagger-data-sync-pipeline.md`. Editing the tagger alone ships nothing.
