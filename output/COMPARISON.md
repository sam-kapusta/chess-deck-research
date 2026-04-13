# SAE Variant Comparison — Final Results (2026-04-12)

## Detection Scoring (Haiku judge + enriched FENs + Sonnet+thinking labels)

| Config | Mean BA | Top-200 | HOLDS (≥0.7) | STRONG (≥0.8) | FAIL (<0.5) | Poly% | Active |
|--------|---------|---------|--------------|---------------|-------------|-------|--------|
| **2048 k=64 + aux** | **0.632** | **0.886** | **659** | **325** | **293** | 30.6% | 1,835 |
| 4096 k=64 + aux | 0.566 | 0.824 | 566 | 159 | 824 | 3.6% | 3,017 |
| 4096 k=32 + aux | 0.563 | 0.829 | 537 | 155 | 854 | 3.3% | 2,908 |
| 2048 k=32 + aux | 0.557 | 0.776 | 284 | 70 | 515 | 3.7% | 1,864 |

## Winner: 2048 k=64 + aux

Best on every detection metric. 0.632 mean BA, 325 STRONG features, only 293 FAIL.

## Key findings

1. **k=64 >> k=32 on detection.** Despite k=32 having much lower polysemantic rate (3.5% vs 30.6%), k=64 labels are more detectable. Lower polysemanticity ≠ better labels for detection.

2. **2048 >> 4096 per-feature.** 4096 has more total features but each one scores worse. The extra dictionary capacity doesn't help detection quality.

3. **Polysemantic rate is misleading.** Sonnet flags 30.6% of k=64 features as polysemantic, but Haiku can still distinguish them from negatives (BA 0.632). The "polysemantic" features might be *general* (fire across contexts) while still being *one concept*.

4. **Best pipeline:** Train BTK 2048 k=64 + aux → Profile → Label with Sonnet+thinking + enriched FENs → Score with Haiku + enriched FENs → Filter by BA ≥ 0.6.

## Structural metrics (for reference)

| Config | Dead | Active | FVU | c_dec | L0 |
|--------|------|--------|-----|-------|----|
| 2048 k=32 + aux | 184 (9%) | 1,864 | 0.128 | 0.045 | 32 |
| 2048 k=64 + aux | 213 (10%) | 1,835 | ~0.082 | 0.036 | 64 |
| 4096 k=32 + aux | 1,188 (29%) | 2,908 | 0.126 | 0.041 | 32 |
| 4096 k=64 + aux | 1,079 (26%) | 3,017 | 0.092 | 0.035 | 64 |

## Job ARNs

| Job | ID | Model | Records |
|-----|-----|-------|---------|
| 2048 k=64 labeling (Sonnet) | pztzjp2jzh8v | Sonnet+thinking | 1,961 |
| 2048 k=64 detect (Sonnet labels) | ac6bc19768ax | Haiku | 1,872 |
| 2048 k=32 labeling (Sonnet) | 9tve7y1jz72h | Sonnet+thinking | 1,918 |
| 4096 both labeling (Sonnet) | jd44898kiujk | Sonnet+thinking | 6,703 |
| All 3 new detect scoring | wma1a33zjoze | Haiku | 8,580 |
