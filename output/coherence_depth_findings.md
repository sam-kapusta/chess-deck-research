# Coherence Depth — do feature names hold to the 70-80th percentile of activation? (d2048_k6)

**Question (Sam):** verify features are coherent — name consistent with the feature down to the 70/80th
percentile of *its own* activation (not just the top-10). Is Claude's label `confidence` good enough?

## Method (objective, no LLM — `coherence_depth.py`)
For each feature, find its identity axis (the single most-concentrated SEE axis at the peak ≥0.9·max:
material_kind / moved / captured / best_captured / best/played_check / own_piece / phase). Then measure
whether that exact axis+value PERSISTS in lower activation bands (≥0.8·max, ≥0.7·max). Coherent = identity
holds; incoherent = decays toward base rate (the f127 pattern).

## Result — 2,048 k6 features
| verdict | count | % |
|---------|-------|---|
| **holds to 0.7·max** (name valid through 70th pct) | 1,676 | **82%** |
| holds to 0.8 only | 194 | 9% |
| **peak-only** (decays before 0.8 — distrust) | 178 | **9%** |

**82% of features keep their identity to the 70th percentile.** The 178 peak-only are the ones to
flag/exclude — and they are mostly the high-fire blobs:
- f1487 "Passive Move Hangs Piece" (fires 34%): material=safe 100%@peak → 38%@0.8 → 19%@0.7
- f1439 "Hanging Piece Blunder": material=down 100% → 7%
These look monosemantic at the top-10 but fire on unrelated positions deeper in. Classic f127 trap.

Coherent-deep examples (trustworthy at depth):
- f952 "Piece Left Hanging" (own=major): 100%@peak → 100%@0.7
- f1372 "Missed Back-Rank Mate" (best=check): 100% → 100%

## Is Claude's labeling `confidence` a good coherence proxy? — NO
- correlation(confidence, coherence-to-0.7) = **0.16** (noise).
- conf 50-70 features have the SAME coherence (0.79) as conf 70-80 (0.77). Only 90+ trends up (0.88).
- peak-only (incoherent) features got **mean confidence 78** — Claude did NOT flag them.
- **Why:** Opus labeled from the top-12 boards + aggregate stats only — it never saw the 70-80th pct
  positions, so confidence reflects "does the peak look clean," not "does the name hold deep." Exactly
  the f127 blind spot.
- **Implication:** filter/trust features by the OBJECTIVE coherence-depth metric, NOT by Claude confidence.

## Artifacts
`coherence_depth_d2048_k6.json` (git+S3): per-feature identity axis, peak/0.8/0.7 %, verdict.
Script: `scripts/labeling/coherence_depth.py`.

## k4 vs k6 — coherence comparison (apples-to-apples, same test)
| | k4 (d1024) | k6 (d2048) |
|---|-----------|-----------|
| holds to 0.7·max | 851 (**83%**) | 1,676 (**82%**) |
| holds to 0.8 only | 93 (9%) | 194 (9%) |
| peak-only (incoherent) | 80 (8%) | 178 (9%) |

**Verdict:** coherence RATE is tied (~82-83%, within noise) — k6's extra capacity neither dilutes nor
improves per-feature quality. k6 wins on absolute count: **1,676 coherent features vs k4's 851** (~2×),
covering more distinct mistake-types (58% of k6 features are concepts k4 lacks). For a coaching product
that filters to coherent features and wants broad coverage, k6 is the pick — more good features, same quality.
