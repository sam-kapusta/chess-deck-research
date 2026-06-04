# k4 vs k6 — Interpretability Head-to-Head (fixed-taxonomy)

**Date:** 2026-06-03. **Method:** Approach C — label d2048_k6 with the same integrated pipeline as
d1024_k4, assign to the SAME 11-bucket taxonomy (allow "unassignable"), compare. Spec:
`docs/superpowers/specs/2026-06-03-k4-vs-k6-interpretability-headtohead-design.md`.

**Framing (honest):** compares two specific models — **d1024_k4** (dict 1024, k=4) vs **d2048_k6**
(dict 2048, k=6). Both k AND dict differ, so this is "*this* k6 vs *this* k4," not k=4-vs-k=6 in the abstract.

## 5-metric table
| metric | k4 (d1024) | k6 (d2048) | verdict |
|--------|-----------|-----------|---------|
| live features | 1,020 | 2,034 | k6 has 2× |
| diffuse rate (conf<40) | ~0% | ~0% | tie |
| unassignable to taxonomy | 0% (0) | 1% (12) | tie — taxonomy houses both |
| blob mass (>10% fire) | 21% (4 feats) | 23% (7 feats) | ~tie |
| buckets populated | 11/11 | 11/11 | tie |
| **new distinct concepts vs k4** | — | **1,181 (58%)** | **k6** |

## Where do k6's extra ~1,000 features come from? (behavioral corr to nearest k4 feature)
| | count | % of k6 |
|---|-------|---------|
| NEW concept (corr <0.4, no k4 analog) | 1,181 | 58% |
| finer split of a k4 concept (0.4–0.7) | 514 | 25% |
| redundant with k4 (≥0.7) | 339 | 17% |

Even among k6's useful-band features (772, fire 0.1–5%): 349 new / 225 split / 198 redundant.

## Verdict
**k6 wins on interpretable vocabulary at no cleanliness cost.** Same ~0% diffuse, same ~22% blob mass,
fills the same 11 buckets, only 1% unassignable — but adds ~1,180 genuinely new mistake-concepts k4
can't express (only 17% redundant). The structural worry that "higher k = more blobs" does NOT translate
to interpretability loss here: the extra features are mostly real, distinct concepts.

**Caveat:** "new concept" = low activation-correlation to any k4 feature; not every one of the 1,181 was
independently verified coherent (some may be rare/noisy). But flat blob-mass + diffuse-rate argue they're
not predominantly junk.

**Implication for the k decision:** if richer coaching vocabulary is the goal, k6 is the better SAE —
it covers the same taxonomy more finely plus surfaces new mistake-types, without getting bloblier or
more diffuse. d1024_k4 remains the choice only if a small, minimal feature set is preferred over coverage.

## Bucket distribution (k6 into the 11 buckets)
Left Piece Hanging 441 · Endgame Technique 338 · Missed Tactic 242 · King Safety 190 · Missed Hanging 170 ·
Missed Check/Mate 159 · Premature Trade 137 · Passive Play 133 · Pointless Check 102 · Greedy Capture 62 ·
Unsound Aggression 53 · UNASSIGNABLE 12. (Proportions mirror k4 — same shape, ~2× the features.)

## Artifacts
- Labels: `feature_labels_integrated_d2048_k6.json` (git+S3). Stats: `see_stats_d2048_k6.json` (git+S3).
- Bucket assignment: `feature_buckets_k6_into_v2.json` (git+S3). Breakdown script: `/tmp/k6_vs_k4_breakdown.py` (notebook-run, logic in this doc).
