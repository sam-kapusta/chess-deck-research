# BTK 2048 k=64 + Aux Loss — Baseline Results (2026-04-12)

## SAE Training
- Config: BatchTopK, dict=2048, k=64, aux_coeff=1/32, dead_threshold=50
- Data: 200K Lichess puzzles, 5 epochs
- Dead features: 213/2048 (10%) — down from 1161/2048 (57%) at k=32
- Active: 1835

## Labeling
- Model: Haiku (enriched FENs, Bedrock Batch job `82opgo09ltc8`)
- 1961 features labeled
- coaching_useful=True: 1089 (Sonnet's initial guess, pre-detection-scoring)

## Detection Scoring — 3 Conditions

### 1. Haiku + Raw FENs (baseline)
- Mean BA: 0.571 | Median: 0.567
- STRONG (≥0.8): 148 | HOLDS (≥0.7): 379 | WEAK+ (≥0.6): 813 | FAILED: 1148
- Top-200 mean: 0.841

### 2. Sonnet + Raw FENs
- Mean BA: 0.577 | Median: 0.567
- STRONG: 178 | HOLDS: 375 | WEAK+: 834 | FAILED: 1127
- Top-200 mean: 0.854
- Delta vs Haiku: +0.006 mean (negligible — judge quality doesn't matter)

### 3. Haiku + Enriched FENs (Stockfish + python-chess)
- Mean BA: 0.619 | Median: 0.600
- STRONG: 289 | HOLDS: 644 | WEAK+: 1135 | FAILED: 826
- Top-200 mean: 0.883
- Delta vs raw: +0.048 mean, +141 STRONG, +265 HOLDS

### Enrichment impact by category
| Category | Raw | Enriched | Delta |
|----------|-----|----------|-------|
| back_rank | 0.574 | 0.732 | +0.158 |
| captures | 0.478 | 0.617 | +0.139 |
| deflection | 0.511 | 0.651 | +0.140 |
| checkmate | 0.589 | 0.689 | +0.100 |
| fork | 0.557 | 0.624 | +0.067 |
| check | 0.550 | 0.604 | +0.054 |
| forcing_moves | 0.509 | 0.534 | +0.025 |
| endgame_technique | 0.798 | 0.739 | -0.059 |

## Comparison with Production (k=32)
| Metric | k=32 (prod) | k=64 + aux |
|--------|-------------|------------|
| Dead | 1161 (57%) | 213 (10%) |
| Features ≥ 0.7 BA (raw) | 89 | 379 |
| Features ≥ 0.7 BA (enriched) | — | 644 |
| coaching_useful (prod filtered) | 218 | TBD |

## Sonnet+Thinking Labeling (2026-04-12)
- Model: Sonnet 4 + 4K thinking tokens (Bedrock Batch job `pztzjp2jzh8v`)
- 1,872/1,961 successfully parsed (89 errors)
- Polysemantic flagged: 572/1,872 (30.6%)
  - 486/572 poly are medium confidence, only 7 high — poly correlates with uncertainty
- Confidence: high=1,146, medium=647, low=79
- Mono + high-confidence: 1,139 features (best candidates for coaching)
- Top categories: forcing_moves(559), checkmate(229), fork(220), check(123), endgame_technique(122)
- Labels are more tactical/specific than Haiku (e.g. "Forced checkmate delivery" vs "Standard d4 opening")

### Detection Scoring (Sonnet labels + enriched FENs)
- Batch job: `ac6bc19768ax` (Haiku judge, enriched FENs, Sonnet-generated labels)
- Status: pending — compare against Haiku-label enriched BA (0.619) when done

## Detection Scoring Summary — 4 Conditions

| Condition | Judge | FENs | Mean BA | Top-200 | HOLDS | STRONG |
|-----------|-------|------|---------|---------|-------|--------|
| Haiku + raw v1 | Haiku | Raw | 0.494 | 0.494 | 4 | 0 |
| Haiku + raw v2 | Haiku | Raw | 0.499 | 0.651 | 38 | 0 |
| Sonnet + raw | Sonnet | Raw | 0.500 | 0.642 | 27 | 0 |
| **Haiku + enriched** | **Haiku** | **Enriched** | **0.619** | **0.883** | **644** | **289** |

Note: v1 had 92% parse failures (Haiku writing essays). v2 fixed with prefill "[".

**Key finding:** Enrichment is the dominant factor (+0.120 BA). Judge model irrelevant (+0.001).

## Job ARNs
- Labeling (Haiku, raw): `82opgo09ltc8`
- Labeling (Haiku, enriched): `2lg4j0j3xf91`
- Labeling (Sonnet+thinking): `pztzjp2jzh8v`
- Detection (Haiku, raw): `m3531jyvb81s`
- Detection (Sonnet, raw): `wi7fkoejtif7`
- Detection (Haiku, enriched): `cvrbvrpaykib`
- Detection (Sonnet labels, enriched): `ac6bc19768ax` (pending)

## Files
- Profiles: `s3://chess-stage-a-140023406996/sae-eval/profiles_btk_2048_k64.json`
- SAE weights: `s3://chess-stage-a-140023406996/output/k_sweep/sae_btk_2048_k64.pt`
- Enrichment cache: `output/fen_enrichment_cache.json` (17,923 FENs, MD5 keys)
- Ground truths: `output/k64_baseline/sae_detect_gt_*.json`
- Sonnet labels: `output/k64_baseline/labels_sonnet_think.json`
