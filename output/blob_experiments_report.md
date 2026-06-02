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

| corpus (k=16) | blobs(>10%) | %top=blob | spec_act_p50 | FVU | useful(≥0.1%) |
|---------------|-------------|-----------|--------------|-----|---------------|
| 42k | 36 | 85% | 0.19 | 0.33 | 1408 |
| 84k | 41 | 84% | 0.21 | 0.30 | 1175 |
| 126k | 36 | 76% | 0.24 | 0.29 | 1053 |
| 168k (full) | 32 | 77% | 0.25 | 0.29 | 983 |

**Finding: corpus size is a weak lever, NOT the blob driver.** Across 4× data (42k→168k),
blob count barely moves (36→41→36→32) and stays in the 30s-40s. Contrast the k-sweep where
blobs ranged 12→89. There IS a mild positive trend — specific-feature activation improves
0.19→0.25 and top-is-blob drops 85%→77% as data grows — so more data helps a little, but
nowhere near enough to fix blobs alone. (Note: useful-feature count *drops* with more data
because the threshold calibrates higher; not a quality regression.)

## Verdict

**The blob problem is fundamentally a k problem, not a data problem.**

- **k is the dominant lever** (blobs 12→89 across k=8→32). Lower k → fewer/weaker blobs,
  stronger specific features, monotonic.
- **Corpus size is a weak secondary lever** (blobs ~32-41 across 4× data). More data nudges
  specific features up slightly but won't solve blobs.

**Recommendation:** if blob concentration / feature specificity is the priority for coaching,
**use k=8, not k=16.** k=8 is the lowest viable k (k=4 collapses to 1800 dead), gives the
fewest blobs (12) and strongest specific features (0.30) of any alive config. The tradeoff is
fewer total useful features (452 vs 983) and worse reconstruction (FVU 0.38 vs 0.29) — acceptable
if coaching needs a clean, precise, modest feature set rather than broad coverage.

The planned 1M-position Lichess corpus is still worth building for *coverage* (more distinct
rare mistakes) but should NOT be expected to fix blob concentration — that requires lower k.

## Artifacts
- `blob_metric.py` — reusable metric (calibrate threshold, n_blob, pct_top_blob, spec_act, FVU)
- `make_subsamples.py` — corpus subsampling for Exp 2
- `blob_sweep_k.jsonl` — Exp 1 raw (k=4,8,12,16,24,32)
- `blob_sweep_corpus.jsonl` — Exp 2 raw (42k,84k,126k,168k)
- Weights on notebook: btk_2048_k{4,8,12,24}_v2_weights.pt, btk_2048_k16_{42,84,126}k.pt
