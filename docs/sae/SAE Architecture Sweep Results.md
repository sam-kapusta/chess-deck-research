# SAE Architecture Sweep Results (2026-04-05)

## Setup
- **Encoder:** DeepMind 270M chess encoder
- **Training data:** 20K Lichess puzzles (correct move), per-token activations
- **Agreement test:** 498 games, 17,532 positions with both played + best moves
- **Agreement metric:** % of positions where feature fires on BOTH played and best move (when it fires on either). High = positional. Low = move-specific.

## Key Metric Definitions
- **Features:** Total features with ≥20 fires (alive and measurable)
- **PurePos (80%+):** Positional features — fire regardless of move choice
- **PureMov (<20%):** Move-specific features — fire on one move but not the other
- **Messy (20-80%):** Mixed features — unreliable for coaching
- **Miss:** Low agreement + fires on best move 1.5x more than played. Things the player doesn't do.
- **Over:** Low agreement + fires on played move 1.5x more than best. Things the player does wrong.
- **Coaching:** Miss + Over = total coaching signals
- **Clean%:** (PurePos + PureMov) / Features
- **Noise:** Features with >20% fire rate (too broad to be specific)

## Results — BatchTopK (the only viable architecture)

| Config | Features | PurePos | PureMov | Messy | Miss | Over | Coaching | Clean% | Noise |
|--------|----------|---------|---------|-------|------|------|----------|--------|-------|
| btk_1024_k8 | 133 | 2 (2%) | 77 (58%) | 54 (41%) | 25 | 32 | 57 | 59% | 0 |
| btk_1024_k16 | 218 | 6 (3%) | 138 (63%) | 74 (34%) | 42 | 33 | 75 | 66% | 0 |
| **btk_1024_k32** | **319** | **9 (3%)** | **223 (70%)** | **87 (27%)** | **47** | **58** | **105** | **73%** | **0** |
| btk_2048_k16 | 258 | 6 (2%) | 170 (66%) | 82 (32%) | 39 | 54 | 93 | 68% | 0 |
| btk_2048_k32 | 395 | 7 (2%) | 272 (69%) | 116 (29%) | 72 | 64 | 136 | 71% | 0 |
| btk_2048_k64 | 503 | 10 (2%) | 340 (68%) | 153 (30%) | 89 | 63 | 152 | 70% | 1 |
| btk_4096_k32 | 432 | 9 (2%) | 295 (68%) | 128 (30%) | 62 | 69 | 131 | 70% | 0 |
| btk_4096_k64 | 532 | 13 (2%) | 365 (69%) | 154 (29%) | 97 | 67 | 164 | 71% | 2 |

## Results — Other Architectures (not viable)

| Config | Features | PurePos | PureMov | Messy | Coaching | Clean% | Noise |
|--------|----------|---------|---------|-------|----------|--------|-------|
| v1_2048 (L1) | 1963 | 1599 (81%) | 230 (12%) | 134 (7%) | 74 | 93% | **1698** |
| gated_2048 | 2048 | 1806 (88%) | 4 (0%) | 238 (12%) | 4 | 88% | **2016** |

## Key Findings

### 1. BatchTopK is the only viable architecture
V1 and Gated produce "clean" numbers (93%, 88%) but it's fake — nearly all features fire >20% (noise). They can't produce specific features. BTK produces zero noise features.

### 2. Higher k/dict ratio = cleaner features
- 1024×k=32 (3.1%) → 73% clean (best ratio)
- 2048×k=64 (3.1%) → 70% clean
- 2048×k=32 (1.6%) → 71% clean
- 4096×k=64 (1.6%) → 71% clean

The ~3% ratio hits a sweet spot. But the difference between 70% and 73% is small.

### 3. More features = more coaching signals (diminishing returns)
- 1024×k=32: 105 coaching signals from 319 features
- 2048×k=32: 136 coaching signals from 395 features (+30%)
- 4096×k=64: 164 coaching signals from 532 features (+21% more)

### 4. Positional features are rare (~2-3%) across ALL configs
The encoder learns move representations, not position representations. This is fundamental — it takes (position + move) as input.

### 5. Maia SAE is complementary, not competing
Maia takes position only (no move input) → 100% positional features by definition. Encoder SAE takes position + move → ~70% move features. They operate on different axes.

## Recommendation
**2048×k=32 for production.** Best balance:
- 395 features (enough coverage)
- 136 coaching signals (most actionable)
- 71% clean (low messy middle)
- 0 noise
- Already trained on 150K puzzles (sweep used 20K for comparison)

If we want max coaching signals and don't mind more features to manage: 4096×k=64 (164 signals) but requires retraining on full data.

## Coherence Sweep (position similarity of top-firing positions)

| Config | Features | Mean | Median | High% (>0.8) | Low% (<0.6) |
|--------|----------|------|--------|-------------|-------------|
| orig_2048_k1 | 30 | 0.797 | 0.835 | 60% | 10% |
| orig_2048_k4 | 85 | 0.790 | 0.857 | 62% | 9% |
| btk_1024_k8 | 157 | 0.722 | 0.807 | 51% | 24% |
| btk_1024_k16 | 262 | 0.645 | 0.753 | 42% | 34% |
| btk_2048_k16 | 313 | 0.644 | 0.764 | 43% | 35% |
| btk_1024_k32 | 385 | 0.578 | 0.676 | 34% | 46% |
| btk_2048_k32 | 466 | 0.565 | 0.642 | 33% | 47% |
| btk_4096_k32 | 533 | 0.557 | 0.644 | 34% | 49% |
| btk_2048_k64 | 619 | 0.495 | 0.412 | 23% | 60% |
| btk_4096_k64 | 656 | 0.483 | 0.397 | 24% | 61% |

**Key finding:** Coherence trades off against k. Lower k = fewer features, each more coherent (positions look similar). Higher k = more features, less coherent (fires across diverse positions sharing a narrow move quality). For coaching diffs, lower coherence is actually fine — you WANT features that detect move qualities across diverse positions.

## Temporal + Polysemanticity

All BTK configs: avg stickiness 0.02-0.04 (features don't persist across moves), 81-94% pure decoder directions, 0% polysemantic. Encoder features are "spiky" (fire on one move, gone the next) and clean in decoder space.

V1/Gated: stickiness 0.83/0.94 (fire on everything = always "sticky"), noise.

## both_vs_one Ratio — Best Predictor for Coaching Diffs

The ratio = one_only / both. High ratio = feature fires on one move but not the other in the same position. This is what coaching needs — features that distinguish played from best.

**83% of features in blunder diffs have ratio > 2.** Of those, 78% have clear move-type signals (captures, checks, piece types). These are the labelable, coaching-useful features.

Filter: ratio > 2 gives ~99 features on 500 games. Expect ~200 with Lichess-scale data.

## Previous Results (150K training, original SAEs)
These were tested with the original game analysis data, not the sweep:

| Config | Features | PurePos | PureMov | Messy | Clean% |
|--------|----------|---------|---------|-------|--------|
| 2048_k1 | 28 | 2 (7%) | 12 (43%) | 14 (50%) | 50% |
| 2048_k4 | 71 | 2 (3%) | 34 (48%) | 35 (49%) | 51% |
| 2048_k16 | 154 | 6 (4%) | 96 (62%) | 52 (34%) | 66% |
| 2048_k32 | 203 | 10 (5%) | 118 (58%) | 75 (37%) | 63% |

Note: 150K training gives slightly different distributions than 20K, but rankings are consistent.
