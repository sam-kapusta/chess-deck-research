# SAE Hyperparameter Sweep Results (2026-05-28)

Comprehensive sweep of BatchTopK SAE hyperparameters on Maia 3 blunder activations (200K × 512-dim, diff pooling, L2 normalized).

## Methodology

**Selection criterion** (from Kozaczuk Personas paper + Chanin "Sparse but Wrong"):
1. Sweep k at fixed dict size
2. Measure decoder cosine similarity (avg max pairwise cosine between decoder vectors)
3. Find the "elbow" where cosine flattens
4. Within that flat region, pick the k that maximizes interpretable features (0.1-1% fire rate)
5. Secondary check: max feature fire rate must stay ≤ 20% (no hubs)

## Full Sweep: Optimal k per Dictionary Size

### Dict=32
| k | Dead | Avg Fire | Max | Cos | Interp (0.1-1%) | FVU |
|---|------|----------|-----|-----|-----------------|-----|
| 1 | 0 | 3.1% | 19% | 0.503 | 8 | 0.663 |
| 2 | 0 | 6.2% | 19% | 0.424 | 2 | 0.560 |
| **3** | **0** | **9.4%** | **19%** | **0.405** | **0** | **0.526** |
| 4 | 0 | 12.5% | 23% | 0.366 | 0 | 0.508 |
| 6 | 0 | 18.8% | 37% | 0.308 | 0 | 0.487 |
| 8 | 0 | 25.0% | 55% | 0.280 | 0 | 0.474 |

**Optimal: k=3** (last before max > 20%)

### Dict=64
| k | Dead | Avg Fire | Max | Cos | Interp | FVU |
|---|------|----------|-----|-----|--------|-----|
| 1 | 1 | 1.6% | 17% | 0.561 | 28 | 0.632 |
| 2 | 0 | 3.1% | 13% | 0.443 | 11 | 0.526 |
| 3 | 0 | 4.7% | 15% | 0.375 | 2 | 0.487 |
| 4 | 0 | 6.2% | 19% | 0.358 | 1 | 0.465 |
| **6** | **0** | **9.4%** | **20%** | **0.325** | **0** | **0.439** |
| 8 | 0 | 12.5% | 29% | 0.263 | 1 | 0.421 |
| 12 | 0 | 18.8% | 52% | 0.222 | 1 | 0.396 |

**Optimal: k=6** (last before max > 20%)

### Dict=128
| k | Dead | Avg Fire | Max | Cos | Interp | FVU |
|---|------|----------|-----|-----|--------|-----|
| 2 | 0 | 1.6% | 19% | 0.447 | 56 | 0.505 |
| 3 | 0 | 2.3% | 19% | 0.402 | 32 | 0.459 |
| 4 | 0 | 3.1% | 19% | 0.369 | 18 | 0.430 |
| 6 | 0 | 4.7% | 19% | 0.327 | 4 | 0.397 |
| **8** | **0** | **6.2%** | **19%** | **0.294** | **4** | **0.377** |
| 12 | 0 | 9.4% | 23% | 0.245 | 0 | 0.350 |
| 16 | 0 | 12.5% | 40% | 0.210 | 0 | 0.331 |

**Optimal: k=8** (last before max > 20%)

### Dict=256
| k | Dead | Avg Fire | Max | Cos | Interp | FVU |
|---|------|----------|-----|-----|--------|-----|
| 2 | 11 | 0.8% | 12% | 0.437 | 120 | 0.486 |
| 4 | 0 | 1.6% | 13% | 0.397 | 95 | 0.401 |
| 6 | 0 | 2.3% | 14% | 0.362 | 46 | 0.364 |
| **8** | **0** | **3.1%** | **19%** | **0.322** | **17** | **0.341** |
| 12 | 0 | 4.7% | 21% | 0.277 | 5 | 0.313 |
| 16 | 0 | 6.2% | 23% | 0.239 | 0 | 0.295 |

**Optimal: k=8** (last before max > 20%)

### Dict=512
| k | Dead | Avg Fire | Max | Cos | Interp | FVU |
|---|------|----------|-----|-----|--------|-----|
| 4 | 1 | 0.8% | 12% | 0.393 | 307 | 0.383 |
| 6 | 0 | 1.2% | 19% | 0.372 | 266 | 0.344 |
| 8 | 0 | 1.6% | 19% | 0.354 | 188 | 0.318 |
| 10 | 0 | 2.0% | 19% | 0.325 | 123 | 0.301 |
| **12** | **0** | **2.3%** | **19%** | **0.304** | **75** | **0.288** |

**Optimal: k=12** (all tested stay under 20% — pick lowest cos)

### Dict=2048
| k | Dead | Avg Fire | Max | Cos | Interp | FVU |
|---|------|----------|-----|-----|--------|-----|
| 8 | 1885 | 0.4% | — | 0.160 | 3 | 0.371 |
| 12 | 252 | 0.6% | — | 0.250 | 1054 | 0.284 |
| **16** | **1** | **0.8%** | **—** | **0.254** | **1531** | **0.260** |
| 20 | 0 | 1.0% | — | 0.250 | 1482 | 0.244 |
| 24 | 0 | 1.2% | — | 0.241 | 1286 | 0.232 |
| 32 | 0 | 1.6% | — | 0.226 | 658 | 0.213 |
| 48 | 0 | 2.3% | — | 0.206 | 39 | 0.186 |

**Optimal: k=16** (cos flattens at k=12-48; within that, k=16 maximizes interpretable features)

## Summary: Optimal per dict

| Dict | k | Cos | Max Fire | Avg Fire | FVU |
|------|---|-----|----------|----------|-----|
| 32 | 3 | 0.405 | 19% | 9.4% | 0.526 |
| 64 | 6 | 0.325 | 20% | 9.4% | 0.439 |
| 128 | 8 | 0.294 | 19% | 6.2% | 0.377 |
| 256 | 8 | 0.322 | 19% | 3.1% | 0.341 |
| 512 | 12 | 0.304 | 19% | 2.3% | 0.288 |
| 2048 | 16 | 0.254 | — | 0.8% | 0.260 |

## Diminishing Returns Analysis

Marginal FVU gain per additional feature:
- 0→32: 0.0148/feat (massive)
- 32→64: 0.0027/feat (big drop but still meaningful)
- 64→128: 0.0010/feat (second elbow)
- 128→256: 0.0003/feat (flat)
- 256→2048: 0.00003/feat (negligible)

**Natural breakpoints:** 64 (first plateau) and 128 (second plateau). After 128, each additional feature contributes almost nothing to reconstruction.

## Elo Discrimination (Continuous Activation Strength)

Measures: for each feature, ratio of mean activation between low-Elo (1000-1400) and high-Elo (1800-2200) blunders. "Vary >1.5×" = features where one band activates 50%+ stronger.

| Dict | k | % vary >1.5× | % vary >2× | Mean ratio |
|------|---|-------------|-----------|------------|
| 32 | 3 | 9% | 3% | 1.19 |
| 32 | 8 | 3% | 0% | 1.15 |
| 32 | 16 | 0% | 0% | 1.11 |
| 32 | 32 | 0% | 0% | 1.04 |
| 64 | 6 | 14% | 6% | 1.42 |
| 64 | 16 | 9% | 2% | 1.20 |
| 64 | 32 | 5% | 0% | 1.13 |
| **128** | **8** | **18%** | **10%** | **1.52** |
| 128 | 16 | 14% | 6% | 1.31 |
| 128 | 32 | 10% | 2% | 1.21 |
| **256** | **8** | **31%** | **15%** | **1.81** |
| 256 | 16 | 24% | 10% | 1.50 |
| 256 | 32 | 17% | 4% | 1.30 |

**Key findings:**
1. **Lower k = better Elo discrimination.** At every dict size, the optimal structural k also gives the best Elo signal. Features become more specific → more rating-dependent.
2. **dict=32/64 have almost no Elo signal** (≤14%). Too coarse — broad categories fire equally at all ratings.
3. **Elo signal starts at dict=128** (18%) and is strong at dict=256 (31%).
4. **Implication for hierarchy:** Top level (32-64 features) is for display grouping only. Rating discrimination happens at the 256+ level.

## Alignment with Kozaczuk Personas Paper

| | Personas (Open e-commerce) | Personas (MovieLens) | Ours (Chess blunders) |
|--|--|--|--|
| Input dim | 128 | 256 | 512 |
| Dict size | 512 (4×) | 1024 (4×) | 2048 (4×) |
| L0 | ~23 | ~80 | 16 |
| Fire rate | 4.5% | 7.8% | 0.8% |
| Decoder cosine | ~0 | ~0 | 0.254 |

Both hit the 3-10% fire rate sweet spot at their optimal operating points. Our lower fire rate (0.8%) reflects the domain — chess blunders have more distinct patterns than shopping behavior, so each feature is more specific.
