# k4 vs k6 — Interpretability Head-to-Head (fixed-taxonomy)

**Date:** 2026-06-03
**Goal:** Does d2048_k6 (sweet-spot, more capacity) produce more *interpretable* features than
d1024_k4 (small, clean), measured by how cleanly each SAE's features map onto the FIXED 11-bucket
taxonomy we built this session?

## Honest framing (state in the writeup)
This compares two specific models — **d1024_k4** (dict 1024, k=4, the one we already labeled) vs
**d2048_k6** (dict 2048, k=6, the structural sweet-spot). Both `k` AND `dict-size` differ, so a
result means "*this* k6 beats/loses-to *this* k4," NOT "k=6 beats k=4 in the abstract." A controlled
k-isolation (d1024_k4 vs d1024_k6) is a separate question we are deliberately not asking here.

## Live feature counts (measured)
- d1024_k4: 1024 live, **446** in useful band (0.1–5% fire)
- d2048_k6: 2048 live, **761** in useful band
- The test subject = the **+315 extra band features** k6 has. Are they new concepts, finer splits, or noise?

## Method — Approach C (fixed taxonomy, the experimental control)
Label k6 with the SAME integrated pipeline k4 got, then assign to the SAME 11 buckets (do NOT
re-derive a new taxonomy). This isolates the variable: we measure how well each SAE's features fill
one fixed coaching vocabulary, not how two different taxonomies compare.

Steps (all mirror k4):
1. `compute_feature_see_stats.py --model k6 --dict 2048` → `see_stats_d2048_k6.json`
   (normalized cohorts ≥0.7/0.8max, net-material, trajectory, all distributions). No LLM.
2. Build `d2048_k6_profiles.json` (top-15 positions/feature). Check Opus coverage; reuse existing
   54.8k position labels — generate only if a feature has <5 covered (report any gap, don't silently skip).
3. `label_features_integrated.py` → `feature_labels_integrated_d2048_k6.json` (~761 Opus calls).
4. Assign each k6 feature to one of the existing 11 buckets (same prompt/rules as k4). Allow an
   **"unassignable"** verdict — a feature that fits NO bucket is recorded as such, not force-fit.
5. `audit_buckets.py` objective cross-check + semantic grade on flagged.

## Verdict — 5-metric table, k4 vs k6 side by side
| metric | computed from | decider? |
|--------|---------------|----------|
| **Diffuse rate** | % features with no concentrated SEE signal (unnameable) | ★ |
| **Bucket coherence** | within-bucket mechanism purity (mistake_type + material agreement) | ★ |
| Audit pass-rate | % surviving objective+semantic audit | |
| Blob mass | activation share in >10%-fire features | |
| Distinct coverage | # buckets/sub-buckets populated with real (non-diffuse) features | |
| **Unassignable rate** | % features that fit NO existing bucket (k6's new concepts) | report |

**Verdict rule:** k6 "wins on interpretability" if it has lower diffuse rate AND comparable-or-better
coherence while filling the taxonomy at least as completely. If k6 is richer (more distinct coverage)
but more diffuse/bloblier, state the tradeoff plainly — no forced winner.

## The payoff question — where do the +315 extra features come from?
After labeling, classify each k6 feature against the k4 set (behavioral correlation or chip/bucket
match):
- **finer split** — k6 feature behaviorally overlaps a k4 feature (same concept, sub-divided)
- **new concept** — k6 feature has no k4 analog AND is interpretable (real added vocabulary)
- **blob / diffuse** — k6 feature is high-fire or has no concentrated signal (capacity wasted)
The breakdown of the 315 into {finer-split, new-concept, blob} IS the interpretability answer:
more "new concept" = k6's capacity earns its keep; more "blob/split" = k4 is sufficient.

## "Unless buckets don't work for it"
If k6's unassignable rate is high, that's a FINDING (k6 surfaces concepts the k4-derived taxonomy
can't house), reported as its own number — not a penalty folded into diffuse.

## Deliverable
A short comparison doc + the side-by-side 5-metric table + the +315 breakdown, committed. Informs
the k decision; ships nothing.

## Files
- New: `see_stats_d2048_k6.json`, `feature_labels_integrated_d2048_k6.json`,
  `feature_buckets_k6_into_v2.json` (k6 features → existing 11 buckets), `output/k4_vs_k6_comparison.md`.
- Reused as-is: `buckets_v2_d1024_k4.json` (taxonomy), all k4 outputs, the pipeline scripts.
