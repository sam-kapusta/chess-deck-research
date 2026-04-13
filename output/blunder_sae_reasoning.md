# Blunder SAE — Design Decisions & Quick Validations

## Goal
Train an SAE on chess blunder moves to discover interpretable "mistake pattern" features. Use these to tell players *what kind of mistake* they keep making, not just *that* they made a mistake.

## Data
- **Source:** Lichess/chess-position-evaluations (HuggingFace, 844M rows)
- **Filter:** eval drop ≥ 200cp from best move to played move (16.1% hit rate)
- **200K blunder positions** from 1.24M scanned positions (7.2M rows)
- Each position has: FEN, blunder_uci (the bad move), best_uci, cp_loss

## Key Design Decision: Move Token Only

### Problem
First attempt trained on all 77 encoder tokens (64 board squares + extras). Fire rates were 20-31% — way too broad. Features detected "what kind of position is this" rather than "what kind of bad move is this."

### Fix
The DeepMind 270M chess encoder takes `[FEN tokens, move_action, return_bucket]` as input (79 tokens). The hidden state at position 77 (the move action token) encodes the encoder's understanding of *this specific move in this specific position*. Production already uses only hidden[77].

Training the SAE on move-token-only activations (200K × 1024 instead of 200K × 77 × 1024):
- Reduced fire rates from 20-31% to 0.8-3.1%
- Reduced cache from 60GB to 804MB
- Reduced training time from 5 min to 8-14 seconds
- Features now represent move patterns, not position patterns

### Validation
Simple phase/piece breakdown of features confirmed move-token features are specific:
- F353: 100% endgame, 100% king moves — "wrong king move in minor piece endgame"
- F265: 70% opening, queen/bishop moves — "wrong development move in opening"
- F1090: middlegame pawn/rook, 0% captures — "quiet pawn push when tactics available"
- F262: 55% captures, 15% checks — "missed decisive capture or check"

All-token features were generic ("this is a middlegame position") — no move-type specificity.

## BatchTopK: Why Not L1?

### L1 problems
1. **Shrinkage** — L1 penalty pulls all activation magnitudes toward zero, even when a feature should fire strongly. Systematic underestimation.
2. **Dead features** — features that don't win early in training never recover (no gradient signal)

### BTK advantages
- k is directly interpretable (mean features per position)
- No magnitude shrinkage — activations are learned freely
- Batch-level k constraint allows variable L0 per position (some blunders activate more features than others)
- Aux loss (1/32 coefficient) recovers dead features from the residual

### What about L1 + resampling?
Resampling/ghost grads fix the dead feature problem (#2) but not shrinkage (#1). Scaled L1 (per-feature coefficients) reduces shrinkage but adds hyperparameters. BTK sidesteps both cleanly.

For chess SAEs specifically, BTK produced zero noise features (>20% fire rate) across all configs, while V1 (L1) and Gated both produced 1700+ noise features.

## Natural Sparsity Analysis

### Question
Does k=64 match the natural sparsity of blunder move-token activations, or are we forcing an arbitrary constraint?

### Quick test
Computed pre-topk activations (before the k-selection) on the trained 4096 k=64 SAE:

| Metric | Value |
|--------|-------|
| Pre-topk features > 0 per position | **318 mean, 313 median** |
| Energy in top-16 | 27.5% |
| Energy in top-32 | 42.2% |
| Energy in top-64 | 60.5% |
| Energy in top-128 | ~75% (estimated) |

### Interpretation
- 318 features naturally activate, but the energy is concentrated in the top few
- k=64 captures 60% of the total activation energy — the long tail is individually weak
- k=32 captures 42% (less complete), k=128 captures ~75% (more complete but potentially more polysemantic)
- The "natural k" for 80%+ energy would be ~100-150

### Decision
We're testing k=32, k=64, and k=128 on 4096 dict. All five move-token variants being labeled with Sonnet+thinking:

| Config | Alive | FVU | FR Median | Energy% | Status |
|--------|-------|-----|-----------|---------|--------|
| 2048 k=32 | 2,031 | 0.115 | 0.87% | ~42% | Labeling |
| 2048 k=64 | 2,033 | 0.093 | 2.00% | ~60% | Labeling |
| 4096 k=32 | 4,009 | 0.107 | 0.35% | ~42% | Labeling |
| 4096 k=64 | 4,027 | 0.085 | 0.84% | ~60% | Labeling |
| 4096 k=128 | 4,092 | 0.066 | TBD | ~75% | Profiling |

The labeling results (% confident labels, detection accuracy) will determine which k/dict_size combination produces the most interpretable blunder features.

## Relevance to Sandstone Persona Pipeline

The same architecture (BTK + aux loss) and methodology (profile → label → detection score) is used for both chess and Sandstone customer personas. Key transferable insights:

1. **Token selection matters.** Training on all tokens vs. the specific token of interest produces very different fire rates and feature quality. Sandstone's `avg_9_18_27` mean-pooling may have similar issues — the new `1024d_matryoshka` MLP-compressed representation is analogous to move-token selection.

2. **k is a tradeoff, not a quality metric.** Higher k = more complete but potentially more polysemantic. Lower k = more selective but loses information. The natural sparsity analysis (pre-topk energy distribution) can guide k selection for any domain.

3. **Fire rate is partially mechanical** (k/dict_size) but BatchTopK's batch-level constraint allows natural variation. Per-feature fire rates are meaningful.

4. **Cheap structural tests first.** Dead features, FVU, c_dec, fire rate distribution — all computed in seconds. Kill bad configs before expensive labeling.
