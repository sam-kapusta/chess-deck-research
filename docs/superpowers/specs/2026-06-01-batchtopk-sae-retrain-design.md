# BatchTopK Chess Mistake SAE — Design Spec
**Date:** 2026-06-01
**Status:** Approved

## Problem

The existing l7only SAE (2048/k16, plain per-position top-k) produced labels that looked good on stats (median confidence 72) but failed manual inspection. Feature f897 got confidence 100 from Opus despite its top activators being a grab-bag (Ke7, a5, duplicated Bxf7, Rf8). Root causes:

1. **Plain per-position top-k** — every position forced to use exactly 16 features, smearing weak directions across the dictionary. No dead-feature revival mechanism.
2. **Non-unit-norm decoder** — activation magnitudes not comparable across features; the "78% weak" finding was a magnitude confound, not real weakness.
3. **No verification step** — Opus writes confident stories from any 10 positions; nothing checked whether a chip actually predicted activation.

SandstonePersonas (same author, different team) solved all three. This spec ports their BatchTopK model + normalization + T1 structural eval to chess, keeping a lightweight chess-specific labeling pass.

## Goal

- **Now (B):** interpretability — honest structural metrics + objective per-feature stats that show whether the SAE carves real, distinct mistake-concepts.
- **Eventually (A):** coaching — the trustworthy filterable feature set this produces feeds cabbagelover's coaching pipeline.

## Scope

**In scope:**
- Port `BatchTopKSAE` + `train_batchtopk` + exact normalization into chess-deck-research
- Retrain on existing deduped l7only diff cache (168,132 positions — do NOT re-extract Maia activations)
- T1 structural eval with gate criteria
- Per-feature stats from top-100 activations (objective, no LLM)
- Opus 2-pass labeling: Pass-1 gap positions → Pass-2 chip+description from top-20
- Atlas updated with stats panel + chips

**Out of scope:**
- Re-extracting Maia-3 layer-7 activations or rebuilding diffs
- Detection scoring (can add later for coaching phase)
- Weighted dataloader (confirmed dead code in SandstonePersonas — weights are unpacked and ignored)
- Relabeling old l7only features (new SAE = new features, old labels moot)
- SandstonePersonas infra (S3/Andes/Cradle/inference pipeline)

## Pipeline — 5 scripts, run in order on chess-poc

```
[cache: maia3_l7only_v2_dedup.pt, 168,132 positions]
         │
    01_train_btk.py       → btk_2048_k32_weights.pt
                            btk_train_stats.json
         │
    02_eval_t1.py         → t1_btk_2048_k32.json
                            GATE: stop if dead>50 or FVU>0.15
         │
    03_feature_stats.py   → feature_stats_btk_2048_k32.json
         │
    04_label_positions.py → all_positions_labeled_opus.json (gap only)
         │
    05_label_features.py  → feature_labels_btk_2048_k32.json
         │
    [atlas update]        → l7only_atlas.html gains stats panel + chips
```

## Section 1 — Training (01_train_btk.py)

### Model: BatchTopKSAE (from SandstonePersonas `sae/model.py`)

Copy the class verbatim. Key properties:
- Selects top `batch_size × k` activations **across the whole batch** (not per-position)
- AuxK loss revives features dead > `n_batches_to_dead` batches
- k-warmup from 1 → k over `sparsity_warmup_steps` steps
- `make_decoder_weights_and_grad_unit_norm()` enforced every step → decoder rows stay unit-norm → activation magnitudes comparable across features

### Hyperparameters (from `exp_btk_2048_k32.json`)

```python
dict_size = 2048
k = 32
k_aux = 256
aux_alpha = 0.03125   # = 1/32
lr = 3e-4
batch_size = 4096
n_epochs = 50         # may need 150-200 if loss hasn't plateaued — watch curve
seed = 123
enc_dtype = "fp32"
beta1, beta2 = 0.9, 0.99
sparsity_warmup_steps = 500   # NOT 10000 — the 2048/k32 config uses 500
```

### Normalization (must match exactly, in order)

```python
mean = x.mean(axis=0)
std  = x.std(axis=0); std[std == 0] = 1.0
x_std  = (x - mean) / std               # per-dim z-score
norms  = np.linalg.norm(x_std, axis=1, keepdims=True).clip(1e-8)
x_norm = x_std / norms                  # per-sample L2-normalize
```

- Compute mean/std on the full corpus (no train/val split — 168k is too small to waste 17k)
- Save mean/std to `btk_train_stats.json` — required for all downstream inference
- T1, stats, and labeling all run on the full corpus using the same saved mean/std

### Training loop (from `train_batchtopk`)

- AMP enabled (`use_amp=True`, fp16 forward/backward via `torch.autocast`)
- After `scaler.unscale_()` and BEFORE `scaler.step()`: call `model.make_decoder_weights_and_grad_unit_norm()`
- Log every 500 steps: loss, L2, aux_loss, L0, current k
- k-warmup: `model.k = max(1, int(1 + (32-1) * step / 500))` for steps ≤ 500

### Epoch count caveat

168k positions / batch_size 4096 = ~41 batches/epoch. 50 epochs = ~2,050 steps. Their runs process millions of rows. If loss is still declining at epoch 50, bump to 150-200. T1 FVU is the arbiter.

## Section 2 — T1 structural eval (02_eval_t1.py)

Port `evaluate_t1()` from `scripts/08_metrics/compute_t1.py` verbatim. Run on full corpus (not held-out val).

### Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| FVU | `MSE / np.var(x)` | Population variance, flattened full tensor |
| L0 | `mean(sum(acts > 0, axis=1))` | Should be ≈32 |
| dead | `freq == 0` | |
| near_dead | `0 < freq < 0.001` | |
| useful | `freq >= 0.001` | |
| very_active | `freq >= 0.05` | |
| mean_decoder_cosine | Over alive features only | |
| redundant_pairs | `|cos| > 0.5` | |
| collapsed_norms | `‖w_dec‖ < 0.01` | |
| mean_bimodality | Mean CV of nonzero acts per feature | |

### Gate criteria

Stop and investigate before proceeding to labeling if:
- `dead > 50`
- `FVU > 0.15`
- `L0 < 20 or L0 > 50`
- `redundant_pairs > 500`

If L0 ≈ 32 and FVU is clean (0.05–0.12), proceed.

## Section 3 — Feature stats (03_feature_stats.py)

For each of 2048 features, compute from its **top-100 activating positions** in the corpus:

```python
{
  "fid": int,
  "n_activating": int,           # total positions where feature is in top-k
  "top100_acts": [float],        # activation values, sorted desc

  # Piece type of blunder move (from python-chess)
  "piece_types": {"bishop": int, "knight": int, "pawn": int, ...},
  "piece_type_pct": {"bishop": float, ...},

  # Side to move
  "side_white_pct": float,

  # Eval trajectory (white-relative, from enrichment cache)
  "traj_already_losing_pct": float,   # eval_before < -150 (white-relative losing side)
  "traj_made_worse_pct": float,       # losing AND eval got worse for mover
  "traj_threw_winning_pct": float,    # eval_before > +150, eval_after < +50

  # cp_loss distribution
  "cp_loss_p50": float,
  "cp_loss_p90": float,
  "cp_loss_mean": float,

  # Motif histogram (from Opus join, None where not labeled)
  "motif_hist": {"hanging_piece": int, "fork": int, ...},
  "motif_coverage_pct": float,        # fraction of top-100 with Opus label

  # Phase
  "phase_hist": {"opening": int, "middlegame": int, "endgame": int},
}
```

Eval trajectory uses `metadata.is_white` + enrichment cache `eval_before/eval_after` (white-relative Stockfish strings). Already in the 34k enrichment cache for most positions; gap positions inherit from Pass-1 enrichment.

## Section 4 — Labeling (04_label_positions.py + 05_label_features.py)

### Pass-1: label gap positions (04_label_positions.py)

Reuse `label_all_positions_opus.py` pattern exactly (the proven script that made the 34k). Gap = top-15 profile positions not yet in `all_positions_labeled_opus.json`. Enrichment cache already covers 34k positions — new gap expected to be small (a few thousand at most, depending on how new weights fire vs old).

- Model: `us.anthropic.claude-opus-4-6-v1`
- Thinking: enabled, budget_tokens=4096
- max_tokens: 8000 (confirmed safe from overnight run)
- Concurrency: 60
- Resume-safe: save every 50

### Pass-2: chip+description per feature (05_label_features.py)

Reuse `label_features_pass2.py` pattern exactly. Top-20 activating positions per feature (from deduped profiles), each with full enrichment + Pass-1 analysis.

Output schema per feature:
```json
{
  "chip": "3-5 word punchy title",
  "label": "one sentence summary",
  "description": "full paragraph with evidence counts",
  "move_pattern": "geometric description of the moves",
  "why_bad": "common reason these moves are mistakes",
  "sub_patterns": ["variant 1", "..."],
  "categories": ["broad", "mid", "specific"],
  "confidence": 0-100
}
```

## Section 5 — Atlas update

Add a stats panel to `l7only_atlas.html` reading `feature_stats_btk_2048_k32.json`. Per feature:
- Piece type bar chart (horizontal, top-3 types)
- Side% indicator (White / Black / mixed)
- Trajectory breakdown (threw winning / made losing worse / other)
- cp_loss p50/p90
- Phase histogram
- Motif top-3 (from Opus join coverage)

Chips and descriptions from `feature_labels_btk_2048_k32.json` replace the old label file. Same atlas URL, same server.

## Output files

| File | Location | Description |
|------|----------|-------------|
| `btk_2048_k32_weights.pt` | notebook + S3 | Trained SAE weights |
| `btk_train_stats.json` | notebook + S3 | Mean/std for normalization |
| `t1_btk_2048_k32.json` | `output/` | T1 structural metrics |
| `feature_stats_btk_2048_k32.json` | `output/` | Per-feature objective stats |
| `feature_labels_btk_2048_k32.json` | `output/` | Chips + descriptions |
| `l7only_atlas.html` | `output/` | Updated explorer |

## Key things NOT to copy from SandstonePersonas

- `create_weighted_dataloader` weights — confirmed dead code, weights are unpacked and never used. Use plain shuffled DataLoader.
- T2 `compute_t2` — uses `relu(z)` without top-k gating, overcounts active features for BatchTopK. Skip T2 entirely.
- Global optimizer state reset in resampling — resets ALL params, not just dead features. Not needed: AuxK handles dead features.
- Naive JSON regex fallback `{[^{}]*...}` — fails on nested JSON. Use the proven `parse_json_response` from our existing pipeline.
