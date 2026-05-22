# Maia 3 SAE — Normalization Decision

## Context

Training a BatchTopK SAE on Maia 3 diff vectors (to_square - from_square activations, 512-dim). Tested three normalization approaches.

## Results

| Normalization | FVU | Active >0.5% | In 0.5-3% | Max activation |
|---|---|---|---|---|
| Z-score + L2 (Sandstone) | 0.191 | 2019 | 1887 | 0.86 |
| Z-score only | 0.188 | 2014 | 1860 | 20.9 |
| **Raw (none)** | **0.118** | 1943 | 1790 | not checked |

## Decision: Raw (no normalization)

Matches the BatchTopK paper (arXiv:2412.06410) which applies no input normalization. The SAE learns `b_dec` to handle means and encoder weights adapt to dimension scales.

## Why NOT Z-score

Our raw diff vectors have per-dimension std ranging 0.5–2.75 (5.5x ratio). Z-score equalizes them, which suppresses high-variance dimensions. But those high-variance dimensions are the most information-rich — dimension 240 (highest Elo sensitivity in our tests) is also high-variance. Z-scoring dilutes the strongest tactical signal.

The SAE can learn to handle non-uniform scales via encoder weights. It just needs the natural structure.

## Why NOT L2

L2 normalization puts all inputs on the unit sphere (norm=1). This:
- Caps max activation at ~1.0 (encoder norms ~0.8–1.3 after training)
- Destroys per-sample magnitude info (a dramatic tactical blunder and subtle positional error look the same)
- Solves a problem we don't have (variable-length aggregation)

Sandstone uses L2 because their inputs are time-decay weighted averages of variable-length purchase sequences. L2 removes the "how much did this customer buy" confound so features encode "what kind of stuff" not "how much stuff." We have single-position vectors — no aggregation, no volume confound.

## Why Sandstone uses Z-score + L2

File: `SandstonePersonas/.../scripts/01_train/train_sae_chunked.py` lines 131-138.

Their pipeline:
1. Customer embeddings are averages of variable-length token sequences
2. Z-score equalizes dimensions (1024-dim encoder output)
3. L2 normalizes to unit sphere (removes volume/sequence-length signal)
4. Norms saved as weights (unused in loss — artifact)

They want share-of-behavior features, not rate-of-behavior. L2 enforces this. Their system works with activations in [0, ~1].

For rate-of-behavior, they post-hoc multiply activations by `W_L` (the time-decay weight sum per customer). See `docs/handoff/universe_wl_handoff.md` in SandstonePersonas.

## Why the BatchTopK paper uses no normalization

The paper (Bussmann et al., NeurIPS 2024 workshop) trains on raw GPT-2 / Gemma 2 2B residual stream activations. No normalization mentioned. LLM residual streams have non-uniform dimension scales — the model's internal scaling IS information. SAE learns around it.

## Hub feature concern

BatchTopK has no anti-hub mechanism (paper is silent on this). High-variance dimensions produce features with larger encoder norms → those features win the batch-level top-k competition more often → higher fire rates. This creates hubs.

Empirically: ~9 features fire >10% (hubs). We exclude these from coaching (same as encoder SAE). The remaining 91% are in healthy 0.5-3% range.

L2 might reduce hubs (by equalizing input magnitudes) but at the cost of worse reconstruction and capped activations. Not worth the tradeoff.

## Inference-time normalization

At inference (production), apply the same normalization as training — which means no normalization. Feed the raw diff vector directly to the SAE. This is simpler than Sandstone's pipeline (no need to store/apply mean+std stats).

If we want to compare activation strength across features (for display), divide by each feature's 95th percentile post-hoc. But that's a display concern, not a training concern.
