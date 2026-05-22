# Maia 3 SAE Training Plan

## Goal

Train a Sparse Autoencoder on Maia 3's internal representations to get fast, local tactical theme classification for blunders. When a player makes a mistake, SAE features fire to identify the tactical motif they missed (hanging piece, back-rank mate, fork, etc.).

## Architecture

- **Model:** Maia 3 (8-layer transformer, 512-dim per square, Elo-conditioned)
- **Probe layer:** `/model/transformer/layers.7/Add_2_output_0` (final residual after layer 7 FFN)
- **Raw activation shape:** (N, 64, 512) — 64 squares × 512 hidden dims
- **SAE architecture:** BatchTopK (Sandstone version — batch-level top-k at train+eval, gradient projection for decoder norm, k warmup, AMP)

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
- Elo conditioning: 1500 (player's approximate level for blunder positions)

## Data

**Source:** Existing `blunder_acts_200k.pt` metadata (FENs + blunder UCIs + best UCIs + cp_loss). Same positions that are already Gemini-labeled.

**Extraction:** Run those FENs through Maia 3 at the blunder player's Elo → save both pooling variants.

## Validation: Elo Sensitivity Test

After SAE is trained, run 100 positions through Maia at Elo 1200 AND Elo 1800. Compare which features fire at each Elo. Expected:
- Tactical features (hanging piece, fork) should fire MORE at low Elo (player misses them)
- Some features should be Elo-invariant (positional structure)
- If features DON'T change with Elo → SAE is learning position structure, not human blind spots

## Success Criteria

- Fire rates in 0.5-3% range (same as working DeepMind encoder SAE)
- Features change meaningfully with Elo (the validation test above)
- At least 50% of features are labelable as tactical themes after Gemini labeling
- Features cover standard drill categories: hanging pieces, forks, pins, skewers, back-rank, discovered attacks

## Pipeline Steps (on chess-poc)

```bash
# 1. Extract activations (both modes)
python scripts/maia3_activations.py \
  --from-cache ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
  --pool from-square --elo 1500 \
  --output ~/SageMaker/chess-stage-a/cache/maia3_blunder_from_sq.pt

python scripts/maia3_activations.py \
  --from-cache ~/SageMaker/chess-stage-a/cache/blunder_acts_200k.pt \
  --pool mean --elo 1500 \
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

# 5. Label (Gemini — same pipeline as before)
# Uses existing blunder position enrichments
```

## What Still Needs to Be Written

1. `scripts/sae/train_maia3_sae.py` — port Sandstone's BatchTopK with:
   - Raw W_enc/W_dec/b_enc/b_dec parameterization
   - Gradient projection for decoder norm
   - k warmup (1 → 32 over 500 steps)
   - AMP (fp16 autocast + GradScaler)
   - L2 normalization on input
   - Input dim 512 (not 1024)

2. `scripts/sae/elo_sensitivity.py` — run same positions at multiple Elos, compare feature activations

3. Update `maia3_activations.py` — add L2 norm option (currently saves raw)

## Endgame Concern

L2 normalization on inputs addresses this: positions with fewer active squares (endgames) produce lower-magnitude activations that would otherwise be dominated by richer middlegame positions. L2 norm puts all positions on the unit sphere regardless of complexity — SAE learns direction (tactical theme) not magnitude (board complexity).

If endgame features still dominate after L2 norm, fallback: filter training data to positions with ≥10 pieces.

## Connection to Production

Once the best variant is identified and labeled:
1. Convert SAE weights to ONNX (runs alongside Maia 3 in the frontend)
2. At inference: player blunders → run position through Maia → extract from-square activation → SAE fires → feature → tactical theme label
3. Ship via same `ship_sae_version.py` pipeline (with `model: "maia3"` config)
