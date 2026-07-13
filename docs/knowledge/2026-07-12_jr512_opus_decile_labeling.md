# jr512_k8 labeled by the Personas decile-Opus method + the "is 2048 too many" answer (2026-07-12)

Follows `2026-07-11_jumprelu_sweep_l7diff.md`. Answers Sam's question ("is 2048 too many features?")
by training a compact **512** JumpReLU with the CANONICAL scheme and labeling it with the real
SandstonePersonas autointerp method (not the earlier tagger-vote hack).

## The model: `jr512_k8_final`
Canonical JumpReLU (`train_jr_canonical.py`, ported from SandstonePersonas `model.py` JumpReLUSAEAuxK):
**target_l0 quadratic loss + AuxK revival + separate ~33× LR on log_threshold**. That threshold
param-group is the piece the earlier `train_jr_sweep.py` missed — it's why θ "couldn't climb" and
`l0_coeff` looked inert. With it, `target_l0` becomes the real sparsity knob.
- dict=512, l0_alpha=4.0 (0.1 default left L0~24; had to crank it — the "very tricky" part), bw=0.1,
  init_threshold=0.5, 60 epochs. Result: **L0≈9.3, FVU 0.30, 1 dead.**
- **Fire-rate shape:** on a fixed 512 dict, LOW L0 gives the best 0.1–5% distribution; raising L0 just
  grows blobs (>10%) without adding in-band features (measured: tl8→172 in-band/15 blobs, tl26→133/96).
  **Dict size, not L0, is the lever for more in-band features** — 2048 had 1695 in 0.1-5% purely
  because 4× more features populate the band. (Per-feature fire rate ≈ L0/dict.)

## Labeling: the SandstonePersonas decile method (`label_features_decile_opus.py`)
Faithful port of `opus_label_audit_2.py` (see `SandstonePersonas/knowledge/naming-rules.md`):
- **Two orthogonal axes.** STRENGTH verdict {good / diffuse / too_broad / wrong / polysemantic /
  noise} = is there a sharp concept at the top activators. REACH `good_until_decile` (D10=top10% only …
  D1=nearly all) = how deep it holds — INDEPENDENT of strength (a good concept can decay fast or hold
  deep). `broad_label` records the normal decay (precise → generic) so it's not punished.
- **Per-band signal = the existing 62,956 Opus per-position analyses** (tactical_motif + tags +
  blunder_summary), aggregated per activation decile. Deciles computed over the **~60k analyzed
  positions only** (Sam) so every band is fully backed (vs 35% coverage if deciling all 168k).
- Opus 4.8 medium/adaptive, 16 threads, JSONL checkpoint. ~20min for 512.

## Results
**252 good · 186 diffuse · 41 too_broad · 26 polysemantic · 7 noise** (0 errors).
- **Reach among the 252 good:** peak at D8–D9 (hold through top 20-30%); **28 hold to D4 or deeper**
  (the broad-reach gold — e.g. f204/f267 "Hanging Piece Left En Prise" @D2, f258 "King Walks Into
  Mating Net" @D3).
- Labels are genuinely **mechanism-level** — "Trading Into Lost Pawn Endgame", "Premature Passed Pawn
  Push", "Missed Back-Rank Mate", "Premature Exchange Squandering Advantage" — far beyond the
  tagger-vote's terse tags (which topped out at best_uci-only 1-ply detection).

## The answer to "is 2048 too many?"  → **yes, and 512 still has slack.**
155 distinct `good_label`s across the 252 good features, BUT the top ~10 are all "hanging piece"
variants — **~70 of 252 good features (~28%) re-encode "you left a piece hanging"** (Left Undefended /
En Prise / Overlooked / Ignored…). Even at 512 the dictionary spends a quarter of its good capacity on
near-duplicates of the single concept the l7-diff representation captures most strongly. So:
- The l7-diff rep has maybe **~150 genuinely distinct coaching concepts**, and even those have
  near-synonyms. This is the same collapse the k6/v7 + architecture-comparison work found.
- 2048 was inherited from the k6/v7 lineage, not justified for coaching. 512 loses redundancy, not
  concepts (same categories covered). A dedup/merge pass (decoder-cosine or label-embedding) would
  cut the ~70 hanging-piece features to a handful and get closer to the true concept count.

## Artifacts
- `output/jumprelu_l7diff/jr512_k8_final.pt` (+jr256), `final_results.jsonl` — models (gitignored .pt).
- `output/jumprelu_l7diff/labels_decile_jr512.json` — the 512 labels (verdict/reach/labels/desc/reason).
- `output/jumprelu_l7diff/labels_decile_jr512.html` — browsable explorer (verdict/reach/decay filters).
- Scripts (committed): `scripts/sae/train_jr_canonical.py`, `scripts/03_feature_labeling/label_features_decile_opus.py`.
- Also on chess-poc `~/SageMaker/jr_canon_out/`.

## Cross-check: Opus labels vs tagger-vote labels (on the 252 good features)
Compared the two label sources head-to-head (`labels_decile_jr512.json` vs `labels_jr512_k8.json`):
- **Q1 — 101/252 (40%) good features have NO tagger label.** Opus named a concept the tagger abstained
  on — mostly tempo/endgame-technique ("Pointless Check Losing Tempo", "Passed Pawn Endgame Conversion
  Error") that need multi-ply reading the tagger can't do from `best_uci` alone. Clear Opus win on breadth.
- **Q2 — 80/252 the tagger's concept words appear nowhere in Opus's label+desc.** Grouped by tagger
  label, 3 systematic patterns: Greedy Capture (25×), Missed Trade to Simplify (20×), Missed
  Overloading (15×). Trade-to-Simplify + Overloading are genuinely COMPLEMENTARY (tagger names the
  primitive Opus's prose glosses) → keep both.

### ⚠️ Tagger bug found via the cross-check: `greedy_capture` conflates greed with unsound sacrifice
The 25 "Greedy Capture" divergences were the tell. Pulled the actual boards for f21/f90/f310/f342/f418:
**every top-firing position is a Greek-Gift-style sacrifice — `Bxf7+` / `Bxh7+` / `Bxh3`, bishop for a
single pawn to expose the king with no follow-up.** Opus labels them correctly ("Unsound Sacrifice, No
Compensation" / "premature attack, unlike the sound Ng5"). The tagger calls them "Greedy Capture."
**Root cause:** `predicates.greedy_capture` fires on *played-is-a-capture + best-is-quiet* with **no
test for whether the capture GAINS or SHEDS material.** A B-for-P Greek Gift satisfies the predicate
(it captures a pawn; the sound move is quiet Ng5) but it's the OPPOSITE mistake — shedding material for
a failed attack, not greedily grabbing it. So `greedy_capture` merges two opposite errors that share a
surface (a capture the engine dislikes). Fix: require the played capture to be a net material GAIN
(SEE-positive, or victim > attacker), which excludes the sacrifices → they'd fall to an "Unsound
Sacrifice" detector (doesn't exist yet). Filed as a tagger issue. **Verdict: Opus CORRECTS the tagger
here — not complementary.**

## SAE-as-tagger-audit: full Stockfish + missing-detector discovery (2026-07-12)
The whole point of the SAE for the product: cluster positions by mistake concept, then find concepts
the RULE TAGGER can't name = missing detectors. Done end-to-end:
- **Why the earlier tagger-vote was capped at 1-ply:** the SAE cache only stores `best_uci` (the diff
  vector needs just the two board states, not the PV). NOT a tagger limitation — it was fed a
  PV-stripped cache. The product's `/tag-moments` DOES get full MultiPV lines.
- **Fix:** ran full Stockfish (depth 16 = prod worker's depth) on all ~60k analyzed positions,
  48 workers, ~10min → `sf_lines_60k.jsonl` (pv_uci + refutation_uci + eval_before/after). Box has
  SF16.1 compiled locally at `~/SageMaker/stockfish_compiled` (glibc 2.26 too old for official bins).
  Scripts: `scripts/sae/sf_batch_60k.py`, `scripts/03_feature_labeling/retag_and_gaps.py` (parallelize
  the tagging — 40-worker Pool; single-threaded was 30min+, parallel ~3min).
- **Result: with full lines the tagger explains 76% of positions** (up from ~35% on best_uci). So most
  of the earlier "gap" was the cache, not the tagger.

### The missing detectors (25 / 252 good features the FULL tagger still can't name) — 3 families:
1. **Passed-pawn / pawn-endgame tempo & conversion (~10, the biggest gap).** Opus: "Passed Pawn
   Endgame Conversion Error", "Mistimed Passed-Pawn Promotion", "King Mismanaged vs Passed Pawn",
   "Lost Tempo in Pawn Endgame". Tagger guesses garbage at ~10% conf (Greedy Capture, Wrong Pawn Race).
   **No detector for pushing/converting a passer at the wrong moment or king-in-pawn-race technique.**
   Word-freq of the gaps confirms it: pawn/tempo/endgame/passed/promotion dominate → endgame technique
   is the tagger's weakest area (it's tactics-first). Filed chess-coach#46.
2. **Pointless / premature checks that waste tempo (~5).** Opus: "Pointless Check Wasting Tempo" (×3+).
   Tagger mislabels Allowed Mate / Premature Attack. **No "aimless check" detector.** Filed chess-coach#47.
3. **Residual hanging-piece (~10)** — likely NOT true gaps: tagger fires low-conf Greedy Capture /
   Missed Trade to Simplify where Opus says "Hanging Piece En Prise" — the same greedy_capture
   conflation from #45, i.e. wrong-label not no-label. Covered by #45.

Artifacts: `output/jumprelu_l7diff/retag_full.json` (per-feature full-tagger vote vs Opus verdict),
`sf_lines_60k.jsonl` + `retag_full_postags.json` on chess-poc.

## ⚠️ "Tagger fires" ≠ "tagger correct" — the board-grounded audit (2026-07-12, Sam's push)
The "76% explained" number was MISLEADING — it counts whether the tagger FIRES, not whether the label
is right or deep. Built a board-grounded Opus judge (`scripts/03_feature_labeling/judge_tagger.py`):
per feature, show the judge the TOP-FIRING boards + SF best/refutation lines + per-position analysis +
BOTH candidate labels, and rule accurate / shallow / wrong with its OWN concept (can reject both — Opus
is NOT ground truth either). Validated on hand-checked cases before trusting it.

**Result on the 112 confident (≥0.30) tagger fires on good SAE features:**
- **33 accurate (29%)** · **39 shallow (35%)** · **40 wrong (36%)**, 2 direction-flips.
- So only ~29% of the tagger's CONFIDENT fires are both right AND deep. This is the real audit signal,
  not the fire-rate.
- **Neither label source is ground truth:** on f38 the tagger was directionally RIGHT ("Allowed
  [back-rank] Mate") and OPUS was WRONG ("Missed Back-Rank Mate" — flipped the direction). The judge
  reading boards caught it; label-vs-label couldn't.

**Failure patterns (→ filed #52 wrong, #53 shallow):**
- `Greedy Capture` = worst offender, ~11 confident-wrong, conflates 4+ concepts: unsound sacrifices
  (Greek Gifts, = #45), zwischenzug misses, losing exchanges into pawn endgames. Fires on
  "capture + best-quiet" with no material-gain/sac/intermezzo test.
- `Missed Trade to Simplify` (~5 wrong) fires on "missed free capture of a hanging piece" — the player
  missed WINNING material, not a trade decision.
- SHALLOW: "Missed Fork"→knight fork, "Advanced Pawn"→PASSED pawn, "Missed Mate"→back-rank,
  "Missed Prophylaxis"→king-pawn opposition draw. Right direction, specific concept flattened.
- **Tagger is reliable on DIRECT primitives** (Missed Mate, Hung Queen/Knight/Bishop, Missed Free
  Queen) — accurate+deep there. The INFERENTIAL labels (Greedy, Trade-to-Simplify, Overloading,
  Prophylaxis) are where it fires confidently-wrong.

Deliverable: `output/jumprelu_l7diff/judge_tagger.json` (per-feature board-grounded verdicts).
**The SAE's real value = auditing tagger CORRECTNESS/DEPTH, not just coverage.**

## ⚠️ Single-tag was a lossy frame — the honest coverage is MULTI-TAG + FAMILY (2026-07-12, Sam)
Two corrections to everything above (the single-top-tag numbers UNDERSTATE the tagger; supersede §"⚠️
Tagger fires ≠ correct" and the §"missing detectors" #3 hanging-piece claim).

**Correction 1 — score coverage by RECALL, not top-1 match.** Positions are multi-mistake and the
tagger emits multiple tags; collapsing to one top tag per feature threw away right answers. Re-judged
with a multi-tag judge (`judge_multi.py`): "is the true concept named by ANY of the feature's tags?"
→ **66.5% covered / 17.6% shallow / 15.9% not_covered** (vs the single-tag ~37%). And **36% of features
had a NON-top tag be the correct one** — direct proof the single-tag collapse was lossy.

**Correction 2 — piece-specific tags FRAGMENT one concept below the top-5 view.** `Missed Free
{Q,R,B,N,P}` and `Hung {Q,R,B,N}` split a single dominant concept across 5 chips, so none cracked the
judge's top-5 even when the FAMILY fired on 30-60% of a feature's top positions. Sam's fix: **parent
"family" tags for aggregation, keep the specific piece as the chip.** Added `tagger.family_of()` (single
source of truth, DIRECTION-PRESERVING — Missed Fork ≠ Allowed Fork) + a `families` map in
`build_mistake_taxonomy.py`. Re-tagged 60k with family votes (`retag_and_gaps.py` now emits
`tagger_top_family` / `tagger_family_votes`), re-judged with the family distribution:

| measurement | covered | shallow | not_covered |
|---|---|---|---|
| multi-tag, piece-level | 66.5% | 17.6% | 15.9% |
| **multi-tag + family rollup** | **84.3%** | 7.0% | **8.7%** |

Also moved median top-concept vote share **0.27 → 0.45** and "features <35%" **165 → 70** (of 252).
**The #3 "residual hanging-piece" claim above was WRONG** — those features fire Hung Material on 35-48%
of positions; they were fragmented, not un-detected. Sam called this ("I thought we had a lot of missed
free/winning/hung") and was right; I'd trusted the judge's top-5 input without checking tagger output.

### The 21 genuinely not_covered (8.7%) — real missing detectors, now tight
- **7 endgame technique** (king activity / opposition / promotion-race tempo) — chess-coach#50, still open.
- **5 pointless / tempo-losing check** — chess-coach#51, **DONE** (see below).
- 3 missed forcing check/mate sequence; 6 scattered (bad queen trade, zwischenzug-before-recapture,
  exposed-king-punished, squandered-won-endgame).

### pointless_check detector shipped (#51) → coverage 84.3% → 86.1% (2026-07-12)
Built `predicates.pointless_check`: played move is a NON-CAPTURE check, best is quiet, it's a mistake.
Mirror of greedy_capture / unsound_sacrifice (all played-direction). Fires 47–62% on the 5 target
features (f4/f85/f121/f129/f499) — **all 5 flipped not_covered → covered**. Corpus fire rate 2.28% of
the mistake-gated 60k. +3 regression cases (115/115). Full-tagger explained 76% → 78% of positions.

**Final coverage (multi-tag + family + pointless_check): covered 86.1% / shallow 7.4% / not_covered 6.6%.**

### The remaining 16 not_covered — two real follow-ups, NOT quick predicates
1. **Endgame technique (f43/f73/f125/f215/f267/f271, 6)** — best is a KING move but no tag dominates
   (missed_king_activity fires only 7–12%). The right move is distant/diagonal opposition, key squares,
   shouldering — position-specific technique. A generic "Missed King Move" would be a naked-rate
   catch-all (rejected on tag-value). Needs real detectors: generalize `lost_opposition` beyond direct
   same-line, key-square occupation, K+P shouldering, rook-endgame Lucena/Philidor. Filed on #50.
2. **Missed winning queen-check (f25/f27/f153/f179, 4)** — best is a check (f179: 78/80 queen checks)
   the player missed, currently MISLABELED `Missed Overloading` (confidently wrong). NOT a naked
   "Missed Check" predicate (that's naked-rate) — the fix is making the existing FORK/tactic motifs
   read the best-line PV so they catch the material-win/fork that follows the check. This is motif
   accuracy = #52/#53 territory, not a new predicate.
- Scattered: f111 bad queen trade, f143/f331 forcing sequences, f187/f380 missed tactics.

Artifacts: `judge_multi.json` (piece-level), `judge_multi_family.json` (family view) on chess-poc
`~/SageMaker/jr_canon_out/`. Scripts: `judge_multi.py`, `judge_multi_family.py`.

## Next (open)
- **Build the 2 real detectors:** pointless-check predicate (played is check, best isn't, eval drops) +
  endgame king-activity/opposition (K+P-specific). Covers 12 of the 21 gaps.
- **Dedup the hanging-piece cluster** to recover the true distinct-concept count.
- Optionally label the 256 model + compare (leaner still).
- The 62,956 Opus analyses cover only 35% of the l7 cache — a top-up batch would deepen low-decile
  reach signal, but Sam chose to use what exists.
