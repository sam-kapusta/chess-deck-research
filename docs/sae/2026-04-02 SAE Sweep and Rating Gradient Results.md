# SAE Sweep & Rating Gradient Results
*2026-04-02*

## TL;DR

SAE features map to real chess concepts AND differentiate rating levels. The rating gradient reveals a clear perception shift: lower-rated Maia fixates on uncertainty/danger/material, higher-rated sees coordination/activity/targeted threats.

## Setup

- **SAE:** 2048 features, k=32, trained on 200K general positions at rating 1900, 50 epochs
- **Concept labeling:** 54 computable chess concepts (game phase, material, king safety, pawns, piece activity) correlated with 2048 features across 10K positions
- **Rating gradient:** Same 5K positions run through Maia at 1100/1400/1700/1900, SAE activations compared

## Concept Labels

799/2048 features (39%) have |correlation| > 0.08 with at least one concept.

**Strongest features:**
| Feature | Correlation | Concept |
|---------|------------|---------|
| F253 | +0.48 | Fianchetto bishop |
| F1205 | +0.44 | Queenside castling |
| F754 | +0.42 | Open files |
| F994 | +0.39 | Center pawns + locked pawns |
| F288 | +0.37 | Pawn shield + castled kingside |

**Distribution:** 27% have |corr| ≥ 0.1, 9.3% ≥ 0.15, 3.9% ≥ 0.2, 1% ≥ 0.3. No features ≥ 0.5 — features are compositional (multiple concepts per feature), not pure detectors.

**Top concept coverage:** opening (119 features), open files (101), piece count (80), endgame (42), minor pieces (35), in check (23), king center (22), knight outpost (20).

## Rating Gradient

**Higher-rated Maia activates more features:** 698 (1100) → 713 (1400) → 733 (1700) → 748 (1900) fire at >1% rate. Richer representation at higher ratings.

295 features increase with rating, 387 decrease.

### What changes with rating (concept-labeled features)

**Concepts that INCREASE with rating:**
- Opening understanding (9 features)
- Passed pawns recognition
- Pawn shield awareness
- Rook on 7th (piece activity)
- Endgame patterns

**Concepts that DECREASE with rating:**
- Open file fixation (5 features)
- General mobility sensitivity
- Fianchetto fixation
- King center detection

### Unlabeled gradient features (manual investigation)

The most interesting features have NO concept correlation — they encode something beyond our 54 concepts.

| Feature | 1100→1900 | What it detects |
|---------|-----------|----------------|
| F1801 ↓ | 20.1% → 7.7% (2.6x) | "Uncertainty" — unresolved tension, uncommitted kings |
| F1637 ↓ | 8.1% → 4.5% (1.8x) | "Crisis" — checks, king attacks, decisive moments |
| F585 ↓ | 3.0% → 1.6% (1.9x) | "Material crisis" — heavy imbalances |
| F948 ↑ | 3.3% → 4.2% | "Coordination" — complex piece interactions |
| F208 ↑ | 0.4% → 1.2% (3x) | "King danger" (targeted) — exposed kings that matter |
| F973 ↑ | 1.8% → 2.6% | "Piece activity" — quality over quantity |

## Coaching Translation

The rating gradient gives us this coaching narrative:

**1100 perception:** "Is this scary? Am I losing material?"
- Everything feels uncertain → many "uncertainty" features fire
- Fixated on crisis and danger → "crisis" features dominate
- Material counting is primary evaluation → "material crisis" features

**1900 perception:** "How do my pieces work together? Where is the REAL danger?"
- Fewer uncertainty features → positions feel more understood
- Targeted danger assessment → knows WHICH threats matter
- Piece coordination as primary evaluation → "coordination" features

**For cabbagelover5566 (1800):** The delta from 1800→2000 would be:
- Further reduction in F1801 (uncertainty) — develop positional intuition
- More F948 (coordination) — see piece relationships in complex middlegames
- More F208 (targeted king danger) — distinguish real threats from phantom ones

## K-Sweep Results

Higher k = better concept interpretability. Contradicts Sandstone k=32 finding.

| k | FVU | EV% | Labeled | %>0.1 | %>0.2 | Gradient feats |
|---|-----|-----|---------|-------|-------|----------------|
| 16 | 0.309 | 69% | 25.8% | 16.5% | 1.0% | — |
| 32 | 0.245 | 76% | 38.1% | 25.0% | 1.6% | 33% |
| 64 | 0.194 | 81% | 51.1% | 35.4% | 3.2% | — |
| **128** | **0.155** | **84%** | **59.9%** | **44.8%** | **6.8%** | **84%** |

**Recommendation: k=128** for concept interpretability. More features per position (128 vs 32) but much better labeling. Can always take top-N most activated for LLM prompts.

## Encoder SAE — Test Invalid

Encoder SAE (DeepMind 270M) showed zero concept correlations at both k=256 and k=32. BUT: the test was likely invalid because encoder embeddings (27K) were extracted with failures/skips from the JSONL but no FEN file was saved — position alignment is broken. Also, mean-pooling destroys the spatial info our concepts measure.

**Verdict:** Can't conclude encoder is useless. Needs re-extraction with FEN tracking.

## Decisions Made

1. **General positions > puzzles for training.** Puzzles bias toward tactical features. General positions match Maia's training distribution.
2. **Euclidean coherence is broken** for high-dim hidden states. Use concept-correlation instead.
3. **k=32 is optimal** for coaching (1.6% fire rate = specific tags). k=128 labels more but at 6.2% fire rate = too broad.
4. **Maia is the proven model for coaching SAE.** Encoder needs proper re-testing.

## Files

- **Production checkpoint:** `output/maia_sae_2048_k32_v2.pt` (best for coaching)
- Concept labels (k=32): `output/maia_sae_2048_k32_v2_concept_labels.json`
- K-sweep: `output/k_sweep_results.json`, `concept_per_k_results.json`
- Rating gradient (k=32): `output/rating_gradient_analysis.json`
- Rating gradient (k=128): `output/rating_gradient_k128.json`
- Feature investigation: `output/feature_investigation.log`
- Cached activations: `cache/maia_acts_{rating}_20000.pt` (4 ratings × 20K)
