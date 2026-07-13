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

## 2048 as a blind-spot discovery pass (2026-07-13) — does more features find NEW tagger gaps?

Sam reframed the SAE's purpose: **it's a discovery tool for tagger blind spots**, not a size-selection
exercise. A feature the tagger scatter-fires on (no dominant tag) = a concept the tagger can't name.
More features = more chances to surface a gap. So: does 2048 expose gaps 512 didn't?

**Method (the trustworthy one — decoder cosine is meaningless here, Sam):**
1. Trained `jr2048_k8` matched to jr512 (same maia3-L7-blunderdiff cache, JumpReLU, target_l0=8; ONLY
   dict differs). FVU 0.276 (vs jr512 0.30), 2034 live features. `jr_canon_out/jr2048_k8.pt`.
2. Tagged all 2034 with the rule tagger (`retag_and_gaps.py --weights jr2048_k8.pt`).
3. Blind-spot candidates = `tagger_family_conf < 0.25` AND `tagger_covered_frac > 0.5` (tagger firing
   but no dominant concept = confused, not silent). → **183 candidates.**
4. Ran the **Opus decile pass** on those 183 (`label_features_decile_opus.py --only <ids>`). This is
   the only filter that separates a COHERENT gap (good/excellent → add a tag) from a DIFFUSE mess
   (no tag will help). Verdicts: **54 good, 24 too_broad, 6 polysemantic, 98 diffuse, 1 noise.**

**Result — the 54 coherent blind spots collapse to 2 clusters:**
- **~29 features = pawn/passed-pawn endgame technique** (promotion race, conversion, king+passer, tempo-in-race).
- **~14 features = tempo-loss / slow passive move** (murkier — overlaps missed_tempo_push/prophylaxis, some overfire-adjacent).

**Verdict: 2048 surfaced NO new concepts vs 512 — the SAME endgame gap, with ~5× more features
pointing at it (29 vs ~6 on jr512).** The concept ceiling is the SUBSTRATE (L7 best−blunder-diff mostly
encodes "material changed hands" + coarse endgame/tempo), not dict size. Going 512→2048 buys resolution
WITHIN concepts (which piece/square) and LOUDER ranking of the real gap, not new coaching concepts.
Also: 2048's redundancy is WORSE — Hung Material family = 72% of features vs 64% at 512.

**Actions:** keep 512 as the shipped size. The one blind spot worth closing = pawn/passed-pawn endgame
technique → chess-coach#50 (updated with the 29 feature-id regression set). If genuinely MORE concepts
are wanted, that's a substrate change (multi-layer activations, or positions beyond blunder-diff), not
a bigger dict. Artifacts on chess-poc: `jr_canon_out/jr2048_k8.pt`, `retag_2048.json`,
`labels_decile_jr2048_blind.json`.

## Fixing the endgame blind spot (2026-07-13) — fragmented concept, not missing detector

Sam: "can we fix the gaps we do have." Took the #1 blind spot (29 pawn-endgame features from the 2048
pass). **Root-caused before coding** (instrumented existing detectors on the 29 features' real boards):

- 36% of the real-mistake positions have best = a KING move; `missed_king_activity` fires on only ~7-12%.
  Its `toward_center OR toward_pawns` heuristic KILLS 72/163 king-move cases — the right endgame king
  move is *lateral* (opposition / key square), neither central nor toward pawns.
- The tagger ALREADY fires the right fragments on these positions (Wrong Pawn Race / Missed Prophylaxis
  / Bad Simplification / Missed King Activity / Lost Opposition) — it just SCATTERS them across 6-8 tags
  so none dominates. **This is fragmentation (family_of's job), not a missing detector.**

**Fixes (commit ce408a9 research, 8b426113 code):**
1. `_opposition_kind` + generalized `lost_opposition`: direct + distant + **diagonal** opposition (was
   direct-only; the SAE misses were ALL diagonal). Kept K+P-only — opposition is only decisive/teachable
   in a pawn ending; with pieces on, 2-sq king spacing is coincidence (rejected the naked-rate loosen).
2. **Position-gated `family_of(label, board=None)`**: new `Pawn Endgame Technique` family rolls the
   fragments up ONLY in a K+P (≤2 heavy) ending. Middlegame prophylaxis/simplification stay separate;
   static/label-only callers (taxonomy) never trigger it. retag passes the board per position.

**Result: 18/29 endgame features now have Pawn Endgame Technique as dominant family (was 0)**, confidence
often doubling (f851 .20→.46, f922 .23→.49, f1709 .14→.42). 118/118 regression. E2E-verified through
the worker adapter. Remaining 11 = ~5 rook endgames (Missed Open File — separate concept, #50 kept open)
+ ~6 genuinely-mixed hanging-piece near-endings (correct to leave).

**Method lesson (the important one):** a low-tagger-confidence SAE feature is NOT automatically a missing
detector. Two distinct causes: (a) **fragmentation** — right tags fire, scattered → fix with a family
rollup; (b) **genuine gap** — no right tag exists → add a detector. Diagnose which BEFORE coding by
instrumenting the existing predicates on the feature's real boards. The endgame gap was (a). The
generalized opposition was a small real (b). Chasing (b) when it's (a) wastes effort (I nearly built 3
new detectors; the fix was one family + one geometry generalization).

## The tempo-loss cluster is a DIFFERENT shape → chess-coach#55 (not fixed here)
The other ~14 coherent blind features Opus calls "Slow Move Missing a Forcing Tactic." Unlike the endgame
cluster, these scatter across genuinely DIFFERENT tactics (Missed Overloading/Mate/Battery/Pin/Fork) —
the only shared thing is "a forcing move existed, a quiet one was played." NOT a family (would merge
distinct tactics); a naive "Missed Forcing Move" tag is near-naked-rate. Likely a motif PV-reading
problem (#52/#53): the specific tactic tags fire WEAKLY because the tagger can't see the multi-ply
forcing line. Filed #55, deferred.

## Are the tags EXACT or catch-alls? Validate against Opus ground truth (2026-07-13)

Sam's test (from #54): a tag on many features carries little info about WHICH feature. Applied it
properly — for each tag, pull the OPUS good_labels of the features it dominates (ground truth, not
position-overlap which can't tell same-concept-different-board from truly-distinct) and count distinct
concepts under it:

| tag / family | #feats dominated | distinct Opus concepts | verdict |
|---|---|---|---|
| **Pawn Endgame Technique** (new) | 30 | 1 (all "passed-pawn endgame") | **EXACT** |
| **Pointless Check** (new) | 6 | 1 ("check") | **EXACT** |
| Missed Fork | 4 | 1 | EXACT |
| Missed Free Material | 40 | ~1 (85% hang) | EXACT |
| **Hung Material** (pre-existing) | 134 | **15** (57% hang) | **CATCH-ALL** |

**The tags I added are exact, not catch-alls** — "Pawn Endgame Technique" looks big only because the
SAE made ~30 near-duplicate features of one concept; one faithful label beats 30 scattered fragments.
That's the RIGHT behavior for a redundant SAE.

**The real catch-all is the pre-existing `hung_material`** — it fires on ANY net material loss, so it
absorbed 28 passed-pawn features (a promoted enemy queen = +8 material = "you hung material") plus
king-safety-into-mate ("lost material to the attack"). 15 distinct Opus concepts under one tag.

**Fix (Sam: "allowed the pawn to promote != hung material"):** `hung_material` subtracts the opponent's
PROMOTION gain from net/peak loss — a queening passer is a lost PAWN RACE, tagged by the endgame
fragments, not "Hung Material". Only fires on pieces the opponent CAPTURED.
- passed-pawn features labeled Pawn Endgame Technique: **0 → 30**
- passed-pawn features mislabeled Hung Material: **28 → 11**
- Hung Material catch-all: **134 → 115 features**

**Method (the durable lesson):** to judge if a tag is a catch-all, measure it against the INDEPENDENT
ground truth (Opus per-feature labels), not against itself or geometry. A tag is a catch-all iff the
features it dominates span multiple Opus concepts. By that test, adding a coarse-LOOKING tag is fine if
the underlying features are genuinely one concept (SAE redundancy); the sin is a tag spanning DIFFERENT
concepts (Hung Material). Next target: Hung Material still mixes hang + mate-attack + endgame at 115.

## Targeting the Hung Material catch-all (2026-07-13) — 134 → 87 features, 57% → 76% pure

Sam: "lets target hung material." Broke down its 115 (then-current) features by Opus ground truth:
77 genuine hangs, 38 mislabeled (11 pawn-endgame, 10 king-safety/mate, 6 sac/greed, rest scattered).
`hung_material` fires on ANY net material loss, so it absorbs anything that ends in lost material.

Fixed the two CLEAN, root-cause mechanisms (both = "the player didn't HANG a piece, something else
happened that nets as material loss"):
1. **Promotion** (already done above): opponent's queening passer = +8 → subtract the promotion gain.
2. **Played sacrifice** (SEE guard): 85% of the sac/greed cluster's hung-fires were a **SEE<0 PLAYED
   capture** — the player CHOSE to shed material (unsound_sacrifice/greedy_capture own it), hung_material
   was out-voting them. Genuine-hang features: only 4% SEE<0. `hung_material` now returns [] on a SEE<0
   played capture. ("Hung" = left a piece to a quiet capture, NOT initiated a losing exchange.)

**Cumulative: Hung Material 134 → 87 features, 57% → 76% genuine-hang purity.** 28 features moved to
their true concept. commits 5aaf049 (research) / 0b4879a8 (code). 121/121 regression, shipped.

**Stopped at 76% — the residual 21 have NO clean mechanism; pushing would break real hangs:**
- 7 king-safety/mate: material genuinely captured (real Hung Queen, 5% actual mate). Opus reads the
  CAUSE; mechanically it's a hang. Demoting these = breaking correct tags on a judgment call. Leave.
- 3 fork ("Walked Into Knight Fork"): fork wins material → nets as hung. That's motif-accuracy (#52/#53
  — make the fork motif read the PV), not a hung_material guard.
- 2 MISSED sacrifices (best WAS the sac); ~5 scattered.

**Method reinforced:** fix a catch-all by finding the mechanical reason the wrong features land there
(promotion=+8, played-SEE<0=sacrifice) and guarding THAT — not by relabeling on the Opus concept
(which would demote genuine cases sharing the same mechanics). When no mechanism separates the residual
from the true positives, STOP; the remaining error is a different problem (motif accuracy), not this tag.

## King Safety as a multi-tag CO-concept (2026-07-13) — "it can be both"

Sam, on the 7 king-safety features left in Hung Material: keeping Hung Queen is right (material IS
lost), but a king-safety concept should ALSO fire — "it's not Hung Queen OR King Attack, it can be
both." This is the multi-tag principle: a feature is a COMBO; the retag's `tagger_family_votes`
distribution already records 2+ co-concepts (not just the argmax `tagger_top_family`).

Measured: king-safety tags DO fire on these features, scattered across ~8 own-king tags (Exposed King,
Allowed Kingside/Queenside/f2f7 Attack, Allowed Double Check, Lost Castling Rights, Allowed Pin-to-King)
so none dominated — SAME fragmentation as pawn-endgame. Added a **King Safety family** (label-based,
direction-scoped to OWN king endangered; Enemy King Exposed / Missed Kingside Attack excluded as the
opposite skill; Allowed Mate kept its own concept — Sam's call).

**Overfire caught mid-fix (the important part):** 'Pawn Move Exposed King' fires on ANY pawn move near a
king. Including it inflated King Safety to 40-51/200 on ~29 plain HANGING-PIECE features. Decomposed the
family vote member-by-member: real king features hold 14-40 WITHOUT Pawn Move Exposed King; noise
features collapse 40 -> 2-18. Dropped it from the family (still its own chip). **Strong-KS precision
(features with KS>=40 that are actually Opus king/mate/exposure) 34% -> 83%.**

Result: 6/7 target features dual-concept (Hung Material + King Safety both top of family_votes).
Families now: Missed Free Material, Hung Material, Missed Exchange, Missed/Allowed Fork, Pawn Endgame
Technique, King Safety. commits c208a1f (research) / e200ed7f (code).

**Method lesson (adds to the catch-all one above):** when building a FAMILY, validate each member
against ground truth, not just the family aggregate. A single trigger-happy member (Pawn Move Exposed
King) can make the whole family look like it fires broadly when the real signal is narrow. Decompose the
family vote by member on both a REAL set and a NOISE set; keep only members that stay strong on real and
collapse on noise.

## Making exposed_king_pawn a real detector (2026-07-13)

Sam: "can we make Pawn Move Exposed King better?" (it was the King Safety family's noise source). It
fired on ANY pawn within 1 file of the king anywhere -> 9.8% of corpus. Failure modes measured: 34%
king in CENTRE (no shelter), 16% uncastled, 6% captures, 35% legit.

Tightened to the concept: (1) king CASTLED (reuse `_king_is_castled`), (2) NON-capture push, (3) pawn
in the shelter zone (<=1 file, <=2 ranks from king) ADVANCING toward the enemy. **9.8% -> 3.8%**;
surviving fires are exactly castled-king shelter pushes. +4 regression (125/125).

**Kept OUT of the King Safety family even after tightening** — re-measured: adding it back drops family
precision 83% -> 62% (2 non-king features join, no new king ones). Two independent questions, both
validated: "is the tag accurate as a standalone chip" (yes now) vs "is it good enough to define a
family aggregate" (still no). A tag can pass the first and fail the second. commits 74592da / 728410ab.

## #52 / #53 / #55 resolution (2026-07-13)

**#52 — trade_to_simplify SEE gate (FIXED, commit 2868f3a).** 'Missed Trade to Simplify' was a catch-all
on 32 good features; on f305/183/39/426/376 the best move captures a HANGING piece (winning material),
mislabeled as a simplifying trade. 95% of wrong fires had best-capture SEE>=2. Gate: fire only on an EVEN
exchange (SEE<2); a winning grab is Missed Free X. All 5 flip to Missed Free Rook/Queen; catch-all 32 -> 2.
(Greedy->Unsound-Sacrifice half was already fixed by the earlier SEE gate.) +2 regression.

**#53 — shallow labels (SUBSTANTIALLY ADDRESSED, closed).** knight-fork already fixed (fork-by-piece);
passed-vs-advanced now owned by the Pawn Endgame Technique family (#50); back-rank mate fires as a co-tag
(f38: Allowed Back-Rank Mate 54 votes) but only 2/40 mate features are nameable, so not promoted over the
reliable eval-Mate. Specific concept is dominant-tag / dominant-family / present-co-tag respectively.

**#55 — forcing-move oversight (WON'T-FIX, closed).** The premise is half-wrong: across the 14 features,
best move is quiet 50% of the time (check 30%, capture 18%), so a forcing-move gate mislabels half. Tags
are maximally scattered (every feature a different top tactic at 7-12/40) — no concept to promote via
PV-reading. Opus reach median D6.5 (diffuse). And ~70% of the cluster is ALREADY covered by pointless_check
(#51) + Pawn Endgame family (#50) + missed_tempo_push. The residual is the SAE's diffuse 'slow move, better
move existed' drift direction — a substrate limit, not a missing detector; a tag would be naked-rate.

**Coherence after all fixes (jr512 v7): 76% top-match / 86% multi-tag** on the 252 good features. The
remaining ~14% is substrate-bound (diffuse features + L7-blunderdiff ceiling), not addressable by more tags.
