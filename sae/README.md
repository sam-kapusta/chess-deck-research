# Chess SAE — Sparse Autoencoder Feature Extraction

Decompose the DeepMind 270M chess encoder's knowledge into interpretable features for coaching.

## Production SAE

**`puzzle_2048_k32_v1`** — puzzle-trained encoder SAE, 2048 features, k=32, BatchTopK.

- Trained on 100K Lichess puzzles (correct moves)
- Extraction: move-token hidden[77] from encoder's 79-token output
- 395 features labeled by Sonnet 4 from 20 FEN examples each
- Labels at: `backend/lambda/sae_features/versions/puzzle_2048_k32_v1/labels.json`
- Versioned architecture — see CLAUDE.md "SAE Labels" section

## Architecture

```
labels.json (ground truth, ~1MB)
  → build_frontend_labels.py → saeLabels.json (frontend, 38KB)
  → relabel.py → llm_stream slim labels (15KB)
  
Lambda outputs: {feature_id, strength, v} — no baked labels
Frontend resolves by feature_id at render time
```

## What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| Mean-pooled encoder SAE | AUCs looked good but features were misleading | Mean-pooling destroys per-square spatial info |
| Blunder-trained encoder SAE | 27% confident labels | Blunder moves too diverse for clean clusters |
| Maia blunder SAE | 45% confident | Better than random but < puzzle encoder |
| Diff SAE (best - blunder activation) | 16% confident | Tautological labels |
| MLP projection (encoder → LLM) | Mode collapse | Info asymmetry between modalities |

## Historical: Maia SAE (not in production)

`maia_sae_2048_k32_v2.pt` was used in early Fargate worker integration. Maia detects WHAT (spatial/concrete patterns — piece placement, structure). Encoder detects WHY (abstract concepts — threats, tactics, strategy). They're complementary but the encoder SAE alone is sufficient for coaching features.

Maia SAE files in this directory are historical artifacts. The production pipeline uses the puzzle encoder SAE exclusively.

## Files

**Production (versioned):**
- `backend/lambda/sae_features/versions/puzzle_2048_k32_v1/labels.json` — ground truth
- `backend/lambda/sae_features/versions/puzzle_2048_k32_v1/sae_weights.npz` — model weights
- `backend/lambda/sae_features/versions/puzzle_2048_k32_v1/config.json` — version metadata

**Historical Maia files (this directory, not in production):**
- `maia_2048_k32_final_labels.json` — 551 Maia labels (historical)
- `maia_2048_k32_concept_labels.json` — concept correlations (historical)
- `k_sweep_results.json`, `rating_gradient_k32.json` — Maia analysis (historical)

**Research scripts:**
- `research/scripts/label_sae_features.py` — canonical labeling script (Sonnet, 20 FENs)
- `research/scripts/lichess_rich_profiler.py` — canonical profiler (100K positions)

## Regeneration

```bash
# On SAIS notebook (chess-poc, ml.g5.4xlarge):

# Profile features (100K positions → top 20 FENs per feature):
python3 research/scripts/lichess_rich_profiler.py

# Label features (Sonnet 4, ~$5):
python3 research/scripts/label_sae_features.py \
  --profiles output/lichess_rich_profiles.json \
  --output output/labels.json

# Rebuild downstream:
python3 backend/scripts/relabel.py
```
