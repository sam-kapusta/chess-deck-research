# Maia 3 SAE Training Plan

## Goal

Train a Sparse Autoencoder on Maia 3's internal representations to get fast, local tactical theme classification for blunders. When a player makes a mistake, SAE features fire to identify the tactical motif they missed (hanging piece, back-rank mate, fork, etc.).

**Success = features represent mistake patterns holding the player back, not structural facts about the position.**

## Architecture

- **Model:** Maia 3 (8-layer transformer, 512-dim per square, Elo-conditioned)
- **Probe layer:** `/model/transformer/layers.7/Add_2_output_0` (final residual after layer 7 FFN)
- **Raw activation shape:** (N, 64, 512) — 64 squares × 512 hidden dims
- **SAE architecture:** BatchTopK (Sandstone version — batch-level top-k at train+eval, gradient projection for decoder norm, k warmup, AMP)

### Maia 3 Input Format (Critical)

Maia 3 always expects **white-to-move** orientation:
- Input: `(64, 12)` one-hot piece placement. Channels 0-5 = white PNBRQK, 6-11 = black pnbrqk.
- Square ordering: rank-1-first (a1=0, b1=1, ..., h8=63).
- **Black-to-move positions MUST be mirrored** — flip board vertically + swap piece colors.
- Elo inputs: `elo_self` (side to move = blunderer), `elo_oppo` (opponent).
- No castling/en passant planes — model handles these implicitly.

When pooling from-square for mirrored positions, the UCI from-square must also be mirrored (e.g., e2→e7).

## Experiment Matrix

Run both pooling modes, compare:

| Variant | Pooling | Input dim | Dict size | k |
|---------|---------|-----------|-----------|---|
| A | from-square (blunder UCI) | 512 | 2048 | 32 |
| B | mean (all 64 squares) | 512 | 2048 | 32 |

Both use:
- L2 normalization on input (prevents endgame domination)
- k_aux: 256, aux_alpha: 1/32
- lr: 0.0003, batch_size: 4096, n_epochs: 50
- sparsity_warmup_steps: 500
- **Mixed-Elo training data** (see Data section)

If dead features are high at k=32, try **k=16** (not k=64 — input dim is half of encoder SAE).

## Data

**Source:** Existing `blunder_acts_200k.pt` metadata (FENs + blunder UCIs + best UCIs + cp_loss). Same positions that are already Gemini-labeled.

**Elo conditioning:** The Lichess position-evaluations dataset doesn't include player ratings per position. Assign each position a random Elo sampled uniformly from 600-2600 for `elo_self`, and an independent random Elo for `elo_oppo`. This covers the full Lichess rating range (absolute beginners to super GMs). The random assignment is saved in the activation cache for reproducibility.

**Why random is fine:** Running each position at a random Elo is equivalent to running the same position at every Elo and sampling one — Maia's representation of a position at a given Elo is deterministic. With 200K positions × uniform Elo, the SAE sees the full joint distribution of (position, Elo). Future version: re-collect blunders from `Lichess/standard-chess-games` (which has `WhiteElo`/`BlackElo` per game) for real player-position-Elo triples.

**Extraction:** Run FENs through Maia 3 with proper preprocessing (mirror for black, rank-1-first square ordering) at each position's assigned Elo → save both pooling variants.

## Validation Battery

### 1. Elo Sensitivity Test
Run 100 positions through Maia at Elo 1200, 1500, AND 1800 (same positions, different conditioning). Compare which features fire at each Elo. Because training data is mixed-Elo, the SAE has seen the full Elo range — this is a legitimate in-distribution comparison.

Expected:
- Tactical features fire MORE at low Elo (player misses them)
- Some features should be Elo-invariant (positional structure)
- If features DON'T change with Elo → SAE is learning position structure, not human blind spots

### 2. Multi-axis Feature Validation
For each feature, measure:
- **Elo distribution:** Does it fire more at low Elo? (Tactical) vs uniformly? (Structural)
- **Severity correlation:** Correlation with cp_loss. Tactical features should correlate.
- **Game phase distribution:** Opening/middlegame/endgame split. Features firing only in endgames regardless of mistake type = likely structural.

Features that pass all three axes (Elo-sensitive, severity-correlated, not phase-dominated) are the tactical candidates worth labeling.

### 3. Pooling Comparison
If from-square and mean-pool produce nearly identical features (high cosine similarity between matched decoder vectors), from-square isn't adding signal. Primary comparison metric: **label interpretability** (can you name a tactical theme from top-20 positions?). Secondary: structural metrics (FVU, dead features).

## Success Criteria

- Fire rates in 0.5-3% range
- Features change meaningfully with Elo (sensitivity test)
- Severity correlation for majority of active features
- At least 50% of features are labelable as tactical themes
- Features represent mistakes holding players back, not position structure
- Features cover standard drill categories: hanging pieces, forks, pins, skewers, back-rank, discovered attacks

## Labeling Strategy

Reuse existing Gemini per-position tactical analyses (5,851 positions already analyzed). Pipeline:
1. Profile Maia SAE → top-20 positions per feature
2. Look up each position's existing Gemini analysis
3. If coverage is good (most top-20 have existing analyses), feed to Sonnet for synthesis
4. If coverage is thin, run additional Gemini position analyses to fill gaps

## Pipeline Steps (on chess-poc)

```bash
# 1. Extract activations (both modes) — uses mixed random Elos
python scripts/maia3_activations.py \
  --from-cache ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
  --pool from-square --elo-mode random \
  --output ~/SageMaker/chess-stage-a/cache/maia3_blunder_from_sq.pt

python scripts/maia3_activations.py \
  --from-cache ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
  --pool mean --elo-mode random \
  --output ~/SageMaker/chess-stage-a/cache/maia3_blunder_mean.pt

# 2. Train SAE (both variants) — use Sandstone BatchTopK
python scripts/sae/train_maia3_sae.py \
  --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_from_sq.pt \
  --dict-size 2048 --k 32 --output ~/SageMaker/chess-stage-a/output/maia3_sae_from_sq.pt

python scripts/sae/train_maia3_sae.py \
  --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_mean.pt \
  --dict-size 2048 --k 32 --output ~/SageMaker/chess-stage-a/output/maia3_sae_mean.pt

# 3. Elo sensitivity test
python scripts/sae/elo_sensitivity.py \
  --sae ~/SageMaker/chess-stage-a/output/maia3_sae_from_sq.pt \
  --positions ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
  --elos 1200,1500,1800 --limit 100

# 4. Profile (top-20 positions per feature)
python scripts/labeling/label.py profile \
  --sae ~/SageMaker/chess-stage-a/output/maia3_sae_from_sq.pt \
  --activations ~/SageMaker/chess-stage-a/cache/maia3_blunder_from_sq.pt

# 5. Label — reuse existing Gemini position analyses + Sonnet synthesis
```

## What Still Needs to Be Written

1. ~~`scripts/sae/train_maia3_sae.py`~~ — DONE (committed)
2. ~~`scripts/sae/elo_sensitivity.py`~~ — DONE (committed)
3. **Fix `maia3_activations.py` preprocessing:**
   - Mirror board for black-to-move (flip vertically + swap colors)
   - Correct square ordering (rank-1-first: a1=0, h8=63)
   - Mirror from-square UCI index for black positions
   - Add `--elo-mode random` (uniform 1100-1900, saved for reproducibility)

## Endgame Concern

L2 normalization on inputs addresses this: positions with fewer active squares (endgames) produce lower-magnitude activations that would otherwise be dominated by richer middlegame positions. L2 norm puts all positions on the unit sphere regardless of complexity — SAE learns direction (tactical theme) not magnitude (board complexity).

If endgame features still dominate after L2 norm, fallback: filter training data to positions with ≥10 pieces.

## Connection to Production

Once the best variant is identified and labeled:
1. Ship a modified `maia3_simplified.onnx` with **two outputs**: (1) original policy logits, (2) layer 7 residual stream. Same model file, one extra output tensor — no extra inference pass.
2. Convert SAE weights to small ONNX model (~2M params). Runs in the same Web Worker as Maia 3.
3. At inference: player blunders → Maia outputs policy + residual → pool from-square → L2 norm → SAE fires → feature → tactical theme label.
4. Server-side path preserved: `.pt` weights can be loaded by Lambda for player profiling and batch analysis of past games.
