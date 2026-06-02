# Blob-Concentration Experiments (overnight 2026-06-02)

**Question:** The k=16 v2 SAE concentrates activation magnitude in ~32 broad "blob" features
(fire on >10% of corpus), drowning out specific mistake-features. Two hypotheses:
1. **Architecture/k**: BatchTopK forces k active slots → fills with broad features. Lower k → fewer blobs.
2. **Sample size**: 168k too small to split broad directions into specifics. More data → fewer blobs.

**Blob metric** (per SAE, on 20k corpus sample): calibrate threshold = mean k-th-largest activation;
n_blob = features firing >10%; pct_top_is_blob = fraction of positions whose top feature is a blob;
spec_top = median activation of top *non-blob* feature; plus n_dead, FVU, useful-feature count.

## Exp 1 — k-sweep (fixed 168k corpus, 200 epochs)

| k | dead | blobs(>10%) | %top=blob | spec_act_p50 | FVU | useful(≥0.1%) | active 1-10% |
|---|------|-------------|-----------|--------------|-----|---------------|--------------|
| 4 | **1800** | 5 | 37% | 0.37 | 0.48 | 196 | 78 |
| 8 | 0 | 12 | 57% | 0.30 | 0.38 | 452 | 138 |
| 12 | 1 | 22 | 69% | 0.27 | 0.32 | 676 | 204 |
| 16 | 1 | 32 | 77% | 0.25 | 0.29 | 983 | 244 |
| 24 | 0 | 68 | 84% | 0.21 | 0.25 | 1262 | 302 |
| 32 | 0 | 89 | 92% | 0.18 | 0.22 | 1501 | 363 |

**Finding: blob concentration is monotonic in k — it IS fundamentally a k problem.** Every step
down in k reduces blob count, blob-domination, and strengthens specific features. No free lunch:
lower k also means worse reconstruction (FVU) and fewer distinct useful features. k=4 collapses
(1800 dead — too sparse for a 2048 dict).

**Sweet spot: k=8.** Blob-minimizing while alive (0 dead): 12 blobs vs 32 at k=16, top-is-blob
57% vs 77%, specific features fire strongest of any viable config (0.30). Cost: FVU 0.38, only
452 useful features. If coaching needs a modest set of clean, high-precision features, k=8 > k=16.
If it needs broad coverage (many distinct mistake types), k=16 trades precision for count.

## Exp 2 — corpus-size sweep (fixed k=16) [RUNNING]
