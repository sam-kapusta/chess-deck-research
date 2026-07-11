# The last fully-labeled SAE — `btk_2048_k6_nol2` / labels v7 (2026-06-05)

**What this is:** the record of the FINAL SAE we labeled end-to-end before the rule-based tagger became
the product direction (~2026-06-15). If someone asks "what was the last SAE we labeled and how,"
this is it. Distinct from the *coherence winner* `btk_2048_k16_nol2` (better structure, **never
labeled** — see `../../knowledge.md` "Current state"). Two different threads; the LABELED one is k6/v7.

## The model
- **Weights:** `btk_2048_k6_nol2.pt` (16MB) — BatchTopK, dict=2048, **k=6**, z-score-only (NO L2).
  S3: `s3://chess-stage-a-140023406996/sae-weights/` (see `output/S3_INVENTORY.md`). Training recipe
  recovered/reproduced in `scripts/sae/train_l7only_nol2.py`.
- **Representation:** Maia3 **79M layer-7**, mean-pooled over 64 squares, **best − blunder diff** —
  `L7[after Maia-best move] − L7[after played/blunder move]`, 1024-dim. This is "the difference
  between the best move and the played move," encoded as layer-7 activations.
- **Cache:** `chess-stage-a/cache/maia3_l7only_v2_dedup.pt` (168,132 × 1024, deduped from 168,669).
  **Notebook-only, not in S3** (expensive to re-extract Maia activations). Rebuild via `build_l2l7_v2.py`
  + `build_l7_only.py` on chess-poc.

## Why k6 + l7only + no-L2 (the three decisions)
- **l7only over the alternatives (2026-06-01 bakeoff):** four best−blunder diff caches compared. The
  real dissociation was **model, not diff** — `board_diff`/`option_a` use the ONNX probe → "mush";
  `l7only`/`l2l7` use the 79M PyTorch checkpoint (`maia3_79m_fixed.pt`) → coherent features that carve
  *mistakes* (mate-in-one, hung piece), not position types. Sam picked **l7only** by eye from
  `feature_boards.html` — simpler than l2l7, can't be contaminated by amplified-L2 noise-laundering.
  (The 79M "fixed" checkpoint matters — the ONNX path had the wrong move vocab, see
  `[[project_maia3_best_move_extraction]]`.)
- **k6 (structural sweet spot):** 3 independent methods agree — mass-band 0.1–10% = 61.5% (most among
  0-dead models), raw-Gini U-shaped min at k6, sparse-probe concept-isolation flat/decreasing above k6.
- **No L2:** z-score-only preserves magnitude = mistake severity; L2 erases it. (Deliberate divergence
  from SandstonePersonas — chess diffs aren't unit-sphere.)

## The labeling: v3 → v7, five iterations (the over-specification fix)
Labeled over 2026-06-04/05. Each version fixed a defect in the prior; the durable lesson is **v7**:
- **v7 = peak + median profiles.** `build_peak_median_profiles.py` samples **10 peak + 10 median
  (p40–60) Opus-covered boards per feature**; the labeler reads BOTH. Fixes the over-specification /
  over-claim defect where labeling from top-N-activation boards alone describes the unrepresentative
  tip of a feature that fires on thousands of positions. (Same lesson as the frequency-ceiling /
  "never characterize a high-fire feature by top-N" finding in `../../knowledge.md`.)
- Earlier versions in the lineage: v2 (neutral framing), v3 (5-word chips), v5 (xhigh thinking), v6
  (debias + refutation context, kept as fallback). The arc was Sam repeatedly pushing "you're
  over-prescriptive / look at median positions, not just peak."

## Artifacts (all in git unless noted)
| File | What |
|------|------|
| `output/relabel_v7_d2048_k6.json` | **The v7 feature labels** (the deliverable) |
| `output/peak_median_profiles_d2048_k6.json` | 10-peak + 10-median boards/feature (v7 input) |
| `output/feature_buckets_v3_d2048_k6.json` | 2035 features → 12 buckets (5 unassignable, largest 18%) |
| `output/atlas/taxonomy_v3_d2048_k6.html` | Atlas tree — **`output/atlas/` is gitignored**, notebook-only |
| `output/game_v7_169764992210.json` | v7 applied to a real game (cabbagelover, 1518) — the probe that later exposed the "can't say WHY mechanically" ceiling (`../../knowledge.md` 2026-06-06) |
| `scripts/sae/train_l7only_nol2.py` | training recipe |
| `scripts/03_feature_labeling/build_peak_median_profiles.py` | the v7 profiler |

## Why it stopped here
The k6/v7 labels were the last full SAE labeling pass. Applying v7 to a real game (2026-06-06) surfaced
the architectural ceiling — the SAE can say "self-weakening/overextending committal move" but never
"pin, knight→queen" (mean64 pooling averaged away which-piece/which-square before the SAE saw it). That
plus the eligible-denominator methodology pushed the product to the **rule-based tagger** (2026-06-15),
which demoted the SAE to vocabulary seed + regression set. So k6/v7 is the high-water mark of the
labeled-SAE line. See `../../knowledge.md` "What the SAE can and cannot tell you" + "TAGGER + FIFA".
