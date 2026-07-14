# SAE-driven tagger gap audit — full session record (2026-07-12 → 07-14)

**What this is:** the end-to-end record of using the SAE as an audit instrument to find what the
rule-tagger misses/mislabels, and the detectors + fixes that came out of it. Companion to
`2026-07-12_jr512_opus_decile_labeling.md` (which has the blow-by-blow); this is the durable summary.

## The method (this is the reusable part)

The SAE is **not** the product labeler — the rule tagger (`scripts/04_tagger/`) is. The SAE is a
**discovery tool for tagger blind spots**. Two independent label systems over the same 60k blunder
positions:
- **Opus per-position analysis** (`all_positions_labeled_opus.json`) — reads the board, produces its own
  motif/tags/summary. Never sees the tagger.
- **Rule tagger** — deterministic predicates/motifs.

Pipeline to find gaps:
1. Train an SAE on the substrate (`maia3_l7only_v2_dedup` = Maia-3 L7 best−blunder activation diff).
2. **Label every feature** with the Opus decile method (`label_features_decile_opus.py`): verdict
   ∈ {good, diffuse, too_broad, polysemantic, noise} + `good_until_decile` reach. Independent of tagger.
3. **Re-tag** the feature's top positions (`retag_and_gaps.py`) → per-feature tag + family vote dist.
4. **Board-grounded judge** (`judge_multi_family.py`): Opus reads boards + the tagger's full tag
   distribution, rules each good feature **covered / shallow_only / not_covered** (recall, not top-1).
5. Cluster the not_covered/shallow → build detectors ONLY for clusters that are coherent + teachable.

**Every detector:** real-board TDD in `regression.py` + corpus-overfire check (target ≪ 10%; real
tactics are 0.3–3%) BEFORE shipping via `ship_tagger.py`.

## SAE state

- **jr512_k8** (`jr_canon_out/jr512_k8_final.pt`) — 512 dict, JumpReLU, target_l0=8, FVU 0.30.
  FULLY labeled: **252 good / 186 diffuse / 41 too_broad / 26 poly / 7 noise.**
- **jr2048_k8** (`jr_canon_out/jr2048_k8.pt`) — same substrate, 2048 dict, FVU 0.276. FULLY labeled:
  **909 good / 796 diffuse / 201 too_broad / 103 poly / 39 noise.**
- **Verdict on dict size:** 2048 = same ~7 concepts as 512, MORE redundant (Hung Material family 64%→72%
  of features). More dict buys resolution within concepts, not new concepts. **Concept ceiling is the
  SUBSTRATE (L7 blunder-diff mostly encodes "material changed hands"), not dict size.** Keep 512 as the
  characterized SAE; 2048 was the wider audit net.
- **ARTIFACTS LIVE ONLY ON chess-poc NOTEBOOK** (`jr_canon_out/`): the .pt weights, all labels_decile_*,
  retag_*, judge_2048_* JSON. NOT in git, NOT in S3 (bucket `chess-stage-a-140023406996` errors
  NoSuchBucket). **This is the top risk — one notebook death loses the labeling work.**

## Coverage / the honest numbers (jr2048, 909 good features)

Three different "match" claims — don't conflate:
- **Recall (some tag names it): 91–92% of GOOD features.** This is what the product uses (multi-chip moments).
- **Exact top-tag match: ~47% of good, ~20% of ALL features.** (The #54 discriminativeness goal.)
- **Of ALL 2048 features, only 44% are even "good"** (coherent) — 56% are diffuse/too_broad/poly/noise
  with no crisp concept to match. No tag layer fixes those; it's the substrate.

Board-judge on 909 good: **~91.7% covered / ~5.5% shallow / ~2.8% not_covered.** The ~2.8% is
substrate-bound (diffuse tempo/slow-move, unnameable clusters).

## Detectors BUILT this session (all TDD'd + corpus-checked + shipped)

| detector | concept | corpus | note |
|---|---|---|---|
| `pointless_check` | played an aimless check, best was quiet | 2.3% | #51 |
| `missed_attacking_check` | missed a forcing (non-mate, non-capture) check | 4.8% | co-tag; mate-delivered excluded |
| `missed_greek_gift` | missed Bxh7+/Bxf7+ bishop SAC on castled king | 0.31% | mirror of unsound_sacrifice |
| `missed_zwischenzug` | right capture, wrong order (insert forcing check first) | 0.80% | reads best_line_san |
| `recapture_exposes_king` | pawn recapture (hxg4) opens line to own castled king | 0.63% | joins King Safety family |
| Missed/Allowed Mate PV-depth | fire from best/refutation line reaching mate, not just eval sentinel | — | #56 |

## Detectors FIXED (precision / catch-all cleanup)

- `lost_opposition` → generalized to direct + **distant + diagonal** opposition (was direct-only), K+P-only.
- `greedy_capture` / `trade_to_simplify` → **SEE gates**: a SEE<0 played capture is a sacrifice not a grab;
  a best capture that WINS material (SEE≥2) is Missed Free X, not a "simplifying trade" (catch-all 32→2).
- `hung_material` → two guards: opponent **promotion** gain doesn't count as a hang (lost pawn race);
  a **SEE<0 played capture** is a sacrifice, not a hang. Hung Material catch-all **134→87 features, 57%→76% pure**.
- `exposed_king_pawn` → tightened to **castled king + non-capture + shelter-pawn advance** (9.8%→3.8%).
- `missed_overloading` → require the best LINE to **win ≥2 material** (was geometry-only). **9.96%→3.26%**;
  unmasked 6 features (→ Missed Attacking Check / Missed Fork / Hung Material).

## Concept FAMILIES added (multi-tag rollup for aggregation, NOT the displayed chip)

`family_of(label, board=None)` rolls piece/variant tags to a concept parent. Families now:
Missed Free Material, Hung Material, Missed Exchange, Missed/Allowed Fork, **Pawn Endgame Technique**
(position-gated: only in a K+P ending), **King Safety** (own-king-endangered; `Pawn Move Exposed King`
deliberately EXCLUDED — too trigger-happy, dropped family precision 83%→62%).

## Key DECISIONS + why (so they don't get re-litigated)

- **Multi-tag is native + correct.** A feature is a combo (hung queen AND king attack); `tagger_family_votes`
  carries all co-tags. The "mislabel" only exists if you force a single argmax. Product shows multiple chips.
- **Precedence scheme (specific-over-material) — MEASURED + REJECTED.** Hurt at every threshold
  (75%→62–75% top-match). Material co-fires are REAL (a fork wins material), not over-fire.
- **Maskers = 1 real bug + argmax whack-a-mole.** Overloading was a crisp over-fire bug (had a material
  test). The rest (Prophylaxis 8.2%, Pawn Break 7.3%, Defensive Resource…) are positional tags with no
  crisp gate — tightening one just promotes the next to the argmax. A masker is only a *fixable bug* when
  the over-firing tag has a correctness test its top-spot violates.
- **Won't-fix (substrate-bound):** tempo/slow-move (#55), perpetual-check (signature absent, 1/80), the
  diffuse 56% of features. These need a richer substrate (multi-layer / non-diff activations), not tags.

## Open follow-ups (see also GitHub issues)

- **#42 — DEPLOY.** None of this session's tagger work is in production. The ECS worker + tag_moments
  Lambda still run the pre-session tagger. Everything is shipped (vendored + in sync, 142/142) but NOT deployed.
- **Prophylaxis (8.2%) + Pawn Break (7.3%) precision pass** — genuinely too high; own effort, framed as
  "tighten this detector," NOT as "maskers." No crisp gate yet — needs board study.
- **Artifact backup** — get jr2048 labels/judge/weights off the notebook into git/S3 (see risk above).
- **#54 discriminativeness** — exact-top-match is ~47%; capped by SAE redundancy, not tagger. Needs SAE
  dedup (merge the ~122 hung-piece duplicate features) to be meaningfully improvable.

## Files / where things are

- Detectors: `scripts/04_tagger/predicates.py` (+ `motifs.py`, `tagger.py`, `chesslib_util.py`).
- Regression: `scripts/04_tagger/regression.py` (142 cases). Taxonomy build: `build_mistake_taxonomy.py` (174 tags).
- Labeling: `scripts/03_feature_labeling/label_features_decile_opus.py`, `retag_and_gaps.py`, `judge_multi_family.py`.
- Ship to product: `../chess-deck-code/backend/scripts/ship_tagger.py` (vendors to worker + tag_moments Lambda + frontend taxonomy).
