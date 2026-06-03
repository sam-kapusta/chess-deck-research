# S3 Asset Inventory

Bucket: `s3://chess-stage-a-140023406996/` · Account 140023406996 (research, `default` profile)
**Last reconciled to ground truth (`aws s3 ls --recursive`): 2026-05-31**

**How to read:** organized by *what the asset is*, not when it was added. Every entry is
tagged with where it actually lives: `[S3]` = in this bucket · `[notebook]` = chess-poc
local disk only (NOT in S3) · `[git]` = chess-deck-research repo. If S3 has an object this
doc doesn't list, the doc is wrong — S3 is ground truth.

**Scope:** this doc tracks the **current Maia3 line of work**. Older eras (Maia2,
DeepMind-270m puzzle/move-token SAEs, contrastive bridge experiments) are superseded — see
[Legacy](#legacy) at the bottom. Don't build on legacy assets without a reason.

---

## 1. Datasets / training inputs

The thing every SAE trains on. The Maia3 SAEs train on a **diff cache**: 200K × 512-dim
vectors, one per blunder position.

| Asset | Location | Size / count | What it is |
|-------|----------|--------------|------------|
| `sae/cache/maia3_blunder_diff.pt` | **[S3]** | 419MB · 200K×512 | **v1** diff cache, `h[to_sq]−h[from_sq]` from Maia3 L7. ⚠️ Has the Black-to-move label-inversion bug (see note). Trained the shipped diff SAE. |
| `maia3_blunder_diff_v2.pt` | **[notebook]** `~/SageMaker/chess-stage-a/cache/` | 429MB · 200K×512 | **v2 = corrected** diff cache. Fixes the inversion. **NOT in S3** — only on chess-poc. Trained matryoshka_v2. |
| `real_game_blunder_positions.json` | **[notebook]** same dir | 48MB · 200K | Source positions for the v2 cache: fen, blunder_uci, cp_loss, eval_before/after, ply, elos, time_control. **No best_uci.** (`cache/blunder_positions.json` is a symlink to this.) |
| `data/stage_a_all_positions.jsonl` / `_uci.jsonl` | [S3] | 9.3MB each | Stage-A position pool (FEN + UCI variants). |

> **The v1 bug (state once):** v1 ranked Black-to-move candidate moves by raw White-POV cp
> instead of flipping sign, so for Black positions the "blunder" and "best" labels invert.
> `cache_blunder_activations_v2.py` Phase 1 fixes the sort (`x[1] if is_black else -x[1]`).
> v2 model features align with the 19,342 Opus analyses by index; v1 features do not.
> **Open item for any rebuild (best_uci for 200k):** the v2 source
> (`real_game_blunder_positions.json`) and the v2 tensor cache both have **no best_uci**.
> The only place 200k best moves exist is the **v1** cache metadata
> (`maia3_blunder_diff.pt`) — but those are produced by the buggy sort, so Black-to-move
> entries are swapped. The three "what-was-missed" constructions (option_a / board_diff /
> l2l7) all need correct best_uci for 200k. Resolution before rebuild: either re-derive
> best_uci for the 200k v2 positions with the corrected `cache_blunder_activations_v2.py`
> Phase 1, or recompute via Stockfish. Don't reuse v1 metadata's labels as-is.

## 2. SAE weights — Maia3 (current)

### BatchTopK L7-only SAEs — `[S3]` `sae/weights/` — **CURRENT FOCUS (2026-06-01)**

These are the BatchTopK SAEs trained on the **l7only v2 diff cache** (168,132 positions, 1024-dim, Maia3 79M layer-7 best−blunder mean-pool diff). Architecture matches SandstonePersonas exactly: BatchTopK + unit-norm decoder + AuxK.

| File | Size | Notes |
|------|------|-------|
| `btk_2048_k16_v2_weights.pt` | 17MB | **Selected for labeling.** 2048/k16, 200 epochs, n_batches_to_dead=126. FVU=0.287, 1036 useful features (≥0.1%), 1012 near-dead. |
| `btk_2048_k16_v2_weights_stats.json` | 43KB | Train mean/std for normalization. Required for inference. |
| `btk_2048_k32_v2_weights.pt` | 17MB | 2048/k32, 200 epochs. FVU=0.218, 1466 useful features. Broader but less precise than k=16. |
| `btk_2048_k32_v2_weights_stats.json` | 43KB | Train mean/std for k=32. |

**Inference:** use `encode_threshold(x, θ)` not BatchTopK at eval time. θ calibrated via `scripts/evaluation/calibrate_threshold.py` (k-th largest activation method). k=16 θ=0.0806 → L0≈15.7. See `output/btk_2048_k16_v2_calibration.json`.

**Cache:** `chess-stage-a/cache/maia3_l7only_v2_dedup.pt` (168,132 × 1024, deduped from 168,669). Only on notebook — not in S3 (expensive to re-extract Maia activations). To rebuild: `build_l2l7_v2.py` + `build_l7_only.py` on chess-poc.

**Status:** k=16 Pass-1 labeling running overnight (2026-06-01). Pass-2 (feature chips) chains after. Labels → `feature_labels_btk_2048_k16_v2.json`.

### z-score-only SAEs — `[S3]` `sae/weights/` — **CHOSEN LINE (2026-06-02)**

Dropping L2 (z-score only) nearly tripled coherent features (990 vs 350). These are the chosen models.

| File | Size | Notes |
|------|------|-------|
| `btk_2048_k16_zscore.pt` | 17MB | k=16, z-score only (NO L2). 990 coherent (48%) by dual-axis SEE probe. SUPERSEDED as the pick (see k-sweep below) but kept. |
| `btk_2048_k16_zscore_stats.json` | 43KB | mean/std for z-score normalization. |
| `btk_2048_k32_zscore.pt` | 17MB | k=32 z-score comparison. 786 coherent (38%). |

**Normalization for these: z-score ONLY (do NOT L2-normalize at inference).** This is a deliberate divergence from SandstonePersonas (which uses z-score+L2) — chess diffs are magnitude-meaningful (severity), customer embeddings aren't. See `output/blob_experiments_report.md` § DECISIVE COMPARISON.

**k-sweep (2026-06-02): k=6 is the sweet spot, not k=16.** Full z-score-only sweep + sparse probing
(3 independent methods) chose `btk_2048_k6_nol2.pt`. Mass-band 0.1–10% = 61.5% (most among 0-dead
models); raw-Gini U-shaped min at k6; sparse-probe concept-isolation flat/decreasing above k6 (k16
splits hang_queen 0.81→0.70@1). See `plan.md` Current State + `log.md` 2026-06-02 (cont.).

**Notebook-only experiment models (reproducible from `scripts/sae/train_maia3_sae.py` + cache, NOT in S3):**
- z-score-only sweep: `btk_2048_k{4,6,8,10,12,16,32}_nol2.pt` — **k6 is the chosen model.**
- dict-size: `btk_1024_k{4,6,8}_nol2.pt` (d1024_k4 = 0 dead, 452 feats, recovers k4 concentration).
- older: k-sweep `btk_2048_k{4,8,12,24}_v2_weights.pt` (z-score+L2), corpus-subsamples `btk_2048_k16_{42,84,126}k.pt`, `btk_2048_k16_raw.pt`.
Regenerate via `--no-l2 --no-val-split --n-batches-to-dead 126 --seed 42` (args in `blob_experiments_report.md`).

### Diff SAEs — `[S3]` `sae/maia3/`
| File | Size | Notes |
|------|------|-------|
| `maia3_sae_diff_2048_k32_l2_200ep.pt` | 8MB | **Shipped winner.** L2 norm, 200ep, FVU 0.191, 0 dead, 2007 labeled. Trained on **v1** cache. |
| `maia3_sae_diff_2048_k32_v2.pt` | 8MB | z-score only, FVU 0.188. |
| `maia3_sae_diff_2048_k32_raw.pt` | 8MB | no norm, FVU 0.118 (best recon, fewer labelable). |
| `maia3_sae_diff_2048_k32.pt` | 8MB | L2 50ep — deprecated by the 200ep version. |
| `maia3_sae_diff_2048_k32_lr1e3.pt` / `_256_k16.pt` | 8MB / 1MB | LR experiment / micro test. |

### Matryoshka SAEs — `[S3]` `sae/weights/matryoshka/` (on v1 cache)
| File | Size | Notes |
|------|------|-------|
| `maia3_matryoshka_2048_k16_p64_256_2048.pt` | 8MB | Config A (recommended): elbows 64/256, 0 dead. |
| `maia3_matryoshka_perlevel_2336_p32_288_2336_k3_8_16.pt` | 9MB | Per-level k, 0 dead, FVU 0.209. |
| (+ 6 more sweep configs) | 8–11MB | Branching/level experiments. See `docs/knowledge/matryoshka-sae.md`. |

### Matryoshka V2 — `[S3]` `sae/weights/matryoshka_v2/` (CORRECT data)
| File | Size | Notes |
|------|------|-------|
| `matryoshka_v2_L3_p128_640_2688_k8_12_16.pt` | 10.5MB | **Best on v2.** Groups [128,512,2048], k=[8,12,16]. |
| `matryoshka_v2_H1_p32_288_2336_k3_8_16.pt` | 9MB | H1 on v2: [32,256,2048], k=[3,8,16]. |
| `sweep_v2_k16_d2048.pt` | 8MB | Standard k=16 baseline on v2. |
| `full_sweep_v2_results.json` | small | All v2 sweep metrics. |

> ⚠️ Trained on the **v1** cache (label bug): all `sae/maia3/maia3_sae_diff_*` and all
> `sae/weights/matryoshka/*`. Trained on **v2** (correct): `sae/weights/matryoshka_v2/*`.

## 3. Labels & profiles — Maia3

| Asset | Location | What it is |
|-------|----------|------------|
| `sae/maia3/l2_labels_sonnet.json` | [S3] | 2007 feature labels (Sonnet 4.6 + thinking) for the diff SAE. |
| `sae/maia3/l2_feature_profiles.json` / `_v2.json` | [S3] | Top-20 example positions per feature (v1 / v2). |
| `sae/maia_labels/maia_2048_k32_final_labels.json` (+ haiku/sonnet/concept variants) | [S3] | Older Maia label passes. |
| `output/taxonomy_v2/taxonomy_v2.json` | [git] | **Ship artifact:** 1,996 features → 20 coaching categories + chips. See knowledge.md § Taxonomy. |
| `output/labels_matryoshka_v2_H1_top32.json` | [git] | 32 top-level v2 features labeled (conf 62–91). |
| `all_positions_labeled_opus.json` | [notebook] | **54,763** Opus-4.6 per-position analyses (position_description, tactical_motif, tags, blunder_summary, refutation). Keyed `FEN\|move`. Shared across all SAE variants. Grows via `label_positions_btk.py`. |
| `sae/labels/fused_names_k4.json` / `fused_names_d1024_k4.json` | [S3] | k4 fused names (Opus motif + SEE facts). **SUPERSEDED method** (fragments concept, got f127 direction backwards). Slim in git: `output/fused_names_*_slim.json`. |
| `sae/labels/feature_labels_see_d1024_k4.json` | [S3]+[git] | **CURRENT label method.** d1024_k4, 1020/1024 features. Opus reads top-12 boards holistically + top-500 SEE-on-both-moves aggregate as raw data. 629 distinct chips (Missed Hanging Piece, Hung Own Queen, Missed Knight Fork...). Produced by `label_features_see.py`. |
| `sae/labels/see_stats_d1024_k4.json` | [S3]+[git] | Per-feature SEE-on-both-moves stats over top-500: best_wins_material_pct / blunder_hangs_own_pct / piece dist. Grounds the labeler; from `compute_feature_see_stats.py`. |
| `sae/eval/sparse_probe_results.json` | [S3] | k-sparse probe (SEE concepts × k4/6/8/16), bal_acc/F1 @p=1..32. Slim in git: `output/eval/sparse_probe_results.json`. `see_labels_168k.npz` (notebook) = the 168k SEE ground-truth concept labels. |

## 4. Stockfish / enrichment data

| Asset | Location | Count | Notes |
|-------|----------|-------|-------|
| `sae/cache/maia3_stockfish_data.json` | [S3] | 18,027 | Stockfish depth-18 analysis keyed `fen\|uci`: best_uci, cp_loss, top_lines, refutation_lines, threat. **Subset, not 200k.** |
| `sae/maia3/stockfish_data.json` | [S3] | ~19k | Duplicate of the above (15.6MB). |
| `output/maia3_sae/stockfish_data_v2.json` | [notebook] | 19,362 | Same format, v2 positions. |
| `sae/cache/position_enrichment_cache.json` | [S3] | — | 17MB enrichment cache. |
| `sae/cache/sweep_blunders_2000.json` | [S3] | 2000 | Blunder sweep eval set. |

> No single file has best moves for all 200k. Stockfish files cover only ~18–19k. The
> only 200k best_uci is in v1 cache metadata, produced by the buggy sort (see §1 note).

## 5. Opus / Gemini position analyses

| Asset | Location | Count | Use? |
|-------|----------|-------|------|
| `all_positions_labeled_opus.json` | **[notebook]** `~/SageMaker/` | **~47k** (growing) | ✅ CANONICAL. Originally 19,342; expanded to 34,186 (2026-05-31 overnight) then to ~47k (2026-06-01 overnight Pass-1 for btk k=16 profiles). Keyed `fen\|uci`; `analysis.{position_description, best_moves_analysis, tactical_motif, blunder_summary}`. Backed up as `all_positions_labeled_opus.bak2.json`. |
| `sae/cache/all_positions_labeled_opus_final.json` | [S3] | 10,648 | ❌ TRUNCATED upload — DO NOT USE. Pull canonical from notebook. |
| `sae/cache/opus_english.json` | [S3] | — | 76MB English analyses (canonical S3 copy). |
| `sae/cache/all_gemini_positions.json` | [S3] | 5,829 | Gemini tactical analysis. |
| `sae/cache/batch_input_maia3.jsonl` | [S3] | — | 37MB batch labeling input. |

## 6. Scripts in S3 — `[S3]` `sae/cache/`

Pipeline scripts parked in S3 (also should be in `[git]` `scripts/sae/`):
- `cache_blunder_activations_v2.py` — two-phase blunder cache builder (Phase 1 CPU filter + best_uci derivation from Lichess multi-move cp ranking, Phase 2 GPU encode). **Authoritative provenance for the diff cache** — and the script to re-run for correct 200k best_uci (its Phase 1 fixes the Black-to-move sort).
- `build_stockfish_data.py` — Stockfish depth-18 MultiPV=3 enrichment → `stockfish_data.json`.
- `train_maia3_sae.py` — SAE training.
- `enrich_all_positions.py`, `profile_l2_for_labeling.py`, `run_cov.py`, `run_pipeline_steps_2_to_4.sh`.

## 7. Encoder weights & misc

| Asset | Location | Size | Notes |
|-------|----------|------|-------|
| `cache/maia3_simplified.onnx` | [S3] | 43.6MB | Maia3 model (8-layer, 512-dim) for ONNX inference. |
| `maia3_with_probe.onnx` | [notebook] | — | Maia3 with L7 probe output (used by build scripts). |
| `maia3_79m_fixed.pt` | [notebook] | — | 79M PyTorch Maia3 (for L2/L7 multi-layer constructions). |
| `cache/move_to_action.json` | [S3] | small | UCI move → token ID map. |
| `sae/cache/feature_norm_constants.json` | [S3] | 0.2MB | Normalization constants. |

## 8. Deprecated / deleted

**Option A / Board Diff / L2+L7 (2026-05-31) — DELETED (v1-corrupted).**
These experimental "what-was-missed" SAEs were trained on the v1 cache and deleted from
both S3 and the notebook. Removed weights: `maia3_option_a_2048_k{8,16,32}.pt`,
`maia3_board_diff_2048_k32.pt`, `maia3_l2l7_2048_k{8,16,32}.pt`, `maia3_l7only_*`.
Removed caches: `maia3_option_a_diff.pt`, `maia3_board_diff_both.pt`,
`maia3_l2l7_concat.pt`. Build/train scripts kept in `[git]` `scripts/sae/new_sae_architecture/`
(annotated with v1 warnings) — **repoint to v2 + correct best_uci before any rebuild.**
Any probe/eval finding from these (board_diff mean-gap, L2/L7 split, f432/f46 examples)
was on v1 data and needs re-verification on v2.

---

## Legacy

Superseded eras, kept in S3 but not part of current work. Don't build on these without a
specific reason.

- **DeepMind-270m puzzle SAEs** — `[S3]` `sae-weights/sae_btk_*` (puzzle), `sae_btk_blunder_mt_*` (blunder move-token), profiles in `sae-eval/`, `detection-scoring/`. The pre-Maia3 production line.
- **Maia2 SAEs** — `[S3]` `sae/weights/sae_maia2_btk_*` (512–4096, k4–128). Maia2-era, replaced by Maia3.
- **"real_btk" realgames SAEs** — `[S3]` `sae/weights/sae_real_btk_*`. Earlier real-game experiment.
- **sae_checkpoints/** — `[S3]` 5 misc 2048_k32 checkpoints (encoder/bulk/combined/correct). Exploratory.
- **Contrastive bridge / Stage-A training** — `[S3]` `scripts/`, `output/phase1/`, `output/contrastive/`, `models/encoder_270m_fp16.*`. Encoder-projection experiments.
- **Large base assets** — `[S3]` `embeddings/encoder_embeddings.npy` (8.2GB), `encoder/` + `cache/deepmind_270m_params.npz` (~1GB each), `models/qwen2.5-7b/`, `models/gemma-4-E4B-it/`. Base encoders/LLMs.
- **`corpus/`** — `[S3]` 20,027 per-game analysis JSONs across `analyzed/`, `analyzed_puzzle_sae/`, `analyzed_movetoken/`, `analyzed_with_sae/` (~5k each). Game-level annotation outputs.

## Using a model

1. `aws s3 cp s3://chess-stage-a-140023406996/<path>.pt . --profile default`
2. `ckpt = torch.load(path, map_location='cpu', weights_only=False)`
3. Build SAE with `ckpt['dict_size']`, `ckpt['k']`; load `W_enc`/`b_enc`/`W_dec`/`b_dec`.
4. Normalize input: `(x − ckpt['mean']) / ckpt['std']`, then L2 to unit sphere (for `_l2_` models).
5. Forward: `recon, acts = sae(x_norm)`.
