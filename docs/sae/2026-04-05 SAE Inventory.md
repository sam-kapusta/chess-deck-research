# SAE Inventory — All Models Built 2026-04-05

**Location:** chess-research notebook, `/home/ec2-user/SageMaker/chess-stage-a/output/`
**Encoder:** DeepMind 270M (bidirectional, no causal mask), weights at `cache/deepmind_270m_params.npz`
**Move map:** `cache/move_to_action.json` (1968 actions)

## Training Data

| File | Size | Description |
|------|------|-------------|
| `good_correct.npy` | 10K × 1024 | Mean-pooled activations, correct (best) moves, from Lichess puzzles |
| `bad_correct.npy` | 10K × 1024 | Mean-pooled activations, opponent's moves |
| `meta_correct.json` | 10K entries | Puzzle themes for each position (used for labeling) |
| `bulk_activations.npy` | 400K × 1024, 1.6GB | Mean-pooled activations from 200K Lichess evals + 200K puzzles |
| `bulk_meta.json` | 400K entries | Themes/source for each position ("eval" or puzzle themes) |

## Mean-Pooled SAEs (trained on 20K positions)

Architecture: `encoder → mean_pool([77, 1024]) → [1024] → SAE(1024, dict_size, k)`

| Checkpoint | Dict | k | Fire% | Alive | Notes |
|-----------|------|---|-------|-------|-------|
| `sae_correct_2048_k32.pt` | 2048 | 32 | 1.6% | ~2048 | **Original best.** Fork 0.825 AUC (misleading). mateIn3 on FEN1, fork on FEN2, mateIn1 on FEN3. |
| `sae_correct_2048_k64.pt` | 2048 | 64 | 3.1% | ~2048 | Decent, some dilution |
| `sae_correct_4096_k64.pt` | 4096 | 64 | 1.6% | ~1000 | Worst — missed mate on all 3 FENs |
| `sae_correct_8192_k64.pt` | 8192 | 64 | 0.8% | ~1000 | Good mate detection, weak fork |
| `sae_correct_8192_k128.pt` | 8192 | 128 | 1.6% | ~2000 | mateIn2 on FEN1/FEN3, discoveredAttack |

## Hyperparameter Sweep SAEs (trained on 20K positions)

| Checkpoint | Dict | k | Fire% | Alive | Notes |
|-----------|------|---|-------|-------|-------|
| `sae_correct_8192_k32.pt` | 8192 | 32 | 0.39% | 1042 | mateIn3+pin on FEN1, pin on FEN3. Specific but misses some tactics |
| `sae_correct_8192_k16.pt` | 8192 | 16 | 0.2% | 519 | Too sparse, generic labels |
| `sae_correct_4096_k32.pt` | 4096 | 32 | 0.78% | 980 | **Excellent.** mateIn2 on FEN1, mateIn1 on FEN3 |
| `sae_correct_4096_k128.pt` | 4096 | 128 | 3.1% | 2971 | Bad — no mate/fork detection |
| `sae_correct_16384_k64.pt` | 16384 | 64 | 0.39% | 2763 | **Best labels.** mateIn1, skewer, backRankMate |
| `sae_correct_16384_k128.pt` | 16384 | 128 | 0.78% | 6232 | mateIn2, backRankMate, good |
| `sae_correct_16384_k32.pt` | 16384 | 32 | 0.2% | 1105 | Fork on FEN2 but lost mate on FEN1/FEN3 |
| `sae_correct_32768_k32.pt` | 32768 | 32 | 0.1% | 1311 | Ultra-specific: discoveredCheck, doubleCheck. Misses common patterns |

## Bulk-Trained SAEs (trained on 400K positions)

| Checkpoint | Dict | k | Alive | Notes |
|-----------|------|---|-------|-------|
| `sae_bulk_2048_k32.pt` | 2048 | 32 | 2048 | Full utilization |
| `sae_bulk_8192_k64.pt` | 8192 | 64 | 5367 | 5x more alive than 20K version |
| `sae_bulk_16384_k32.pt` | 16384 | 32 | 2084 | 2x more alive than 20K version |
| `sae_bulk_16384_k64.pt` | 16384 | 64 | 2855 | discoveredAttack, discoveredCheck, queensideAttack labels |
| `sae_bulk_32768_k32.pt` | 32768 | 32 | 2277 | mateIn2, enPassant |
| `sae_bulk_32768_k64.pt` | 32768 | 64 | 3081 | **backRankMate, mateIn1, mateIn2** on FEN3 |

## Per-Token SAE (trained on 50K puzzles × 77 tokens)

Architecture: `encoder → [77, 1024] → SAE on each token independently`

| Checkpoint | Dict | k | Alive | Notes |
|-----------|------|---|-------|-------|
| `sae_pertoken_16384_k64.pt` | 16384 | 64 | 2183 | **Spatial features.** 511 narrow features (fire on ≤5 squares). F4673=king detector, F10653=queen position. Back-rank features fire on 8th rank in diff mode. |

## Key Findings

### Mean-pooled SAEs
- **Invalidated for tactics.** Fork AUC 0.825 is misleading — ground truth check showed 5/10 non-fork puzzles also have forks. Detects statistical correlates, not spatial tactics.
- **Mate detection probably real** (0.999 AUC, F12415 fires on 56/57 mate positions). Mate is structural, survives pooling.
- **Bigger dict = more specific labels** but same fundamental problem. 16384_k64 says "backRankMate" instead of "mate" but still can't tell you where.
- **More training data helps alive count** but doesn't fix the spatial problem.

### Per-Token SAE
- Trained on individual tokens but learned mostly global features (top-10 by activation fire everywhere)
- **511 narrow features exist** — fire on 1-5 squares, piece-specific (king, queen detectors)
- **Spatial signal is in the diff** (good move - bad move), not absolute activations
- F7272 fires on positions with dominant queens — "queen is powerful" detector across all 10 top positions
- Good-move features: F1994 (fires ONLY on good moves, diff=+8.18), F12415 (mate, +6.99)
- Bad-move features: F11427 (back rank vulnerability, fires more on bad moves)

### Evaluation on 3 Game FENs (game 166660084296)

Position 1: White played Kd1?? (should Qxg8)
Position 2: Black played Bg4+ (great move)
Position 3: White played f3 (should Qxg8)

Best results by SAE type:
- **sae_correct_2048_k32:** mateIn3 on FEN1, fork on FEN2, mateIn1 on FEN3. Consistent.
- **sae_correct_16384_k64:** mateIn1, skewer, backRankMate. Most specific labels.
- **sae_bulk_32768_k64:** backRankMate on FEN3. Best single label.
- **sae_pertoken_16384_k64:** Back-rank features (F11427) light up entire 8th rank on Qxg8 diff. Most informative diff.

### Not Yet Built
- **Diff SAE:** Train on (good_move_acts - bad_move_acts). Every feature is about move quality by construction. Best idea from the session.
- **Tag correlation:** Script ready (`tag_correlation.py`), game moments need uploading.
- **Final 3-SAE comparison:** Script ready (`final_compare.py`), needs connectivity.

## Scripts on Notebook

| Script | Purpose |
|--------|---------|
| `compare_all_saes.py` | 5 original SAEs on 3 FENs |
| `sae_sweep_v2.py` | Hyperparameter sweep (6 configs) |
| `extract_bulk.py` | Extract 400K activations from evals+puzzles |
| `train_bulk_saes.py` | Train 6 SAEs on 400K data |
| `train_pertoken_sae.py` | Per-token SAE training + evaluation |
| `investigate_pertoken.py` | Top 10 features across 200 puzzles |
| `investigate_v2.py` | Good/bad move diff + narrow features |
| `test_fork_reality.py` | Fork ground truth check (invalidated mean-pooled) |
| `final_compare.py` | Side-by-side: 2048_k32 vs bulk_16384_k64 vs pertoken_16384_k64 |
| `tag_correlation.py` | Correlate features with coaching tags from real games |
