# S3 Asset Inventory — SAE Models (2026-04-14)

Bucket: `s3://chess-stage-a-140023406996/`
Account: 140023406996 (research, default profile)

## SAE Weights — Puzzle-trained (production)

```
s3://chess-stage-a-140023406996/sae-weights/
  sae_btk_2048_k64.pt       ← WINNER (BA=0.632, deploy this)
  sae_btk_2048_k32_aux.pt   ← BA=0.557, low poly (3.7%)
  sae_btk_4096_k64_aux.pt   ← BA=0.566, most features (3,017)
  sae_btk_4096_k32_aux.pt   ← BA=0.563, most selective
```

All trained on 200K Lichess puzzles, 5 epochs, BatchTopK + aux loss (1/32).

## SAE Weights — Blunder-trained (move-token, experimental)

```
s3://chess-stage-a-140023406996/sae-weights/
  sae_btk_blunder_mt_1024_k16_aux.pt  ← alive=1023, FVU=0.155
  sae_btk_blunder_mt_1024_k32_aux.pt  ← alive=1016, FVU=0.127
  sae_btk_blunder_mt_2048_k16_aux.pt  ← alive=2040, FVU=0.144
  sae_btk_blunder_mt_2048_k32_aux.pt  ← WINNER: alive=2031, FVU=0.115, 1080 unique labels
  sae_btk_blunder_mt_2048_k64_aux.pt  ← alive=2033, FVU=0.093
  sae_btk_blunder_mt_4096_k32_aux.pt  ← alive=4009, FVU=0.107, 1914 unique labels
  sae_btk_blunder_mt_4096_k64_aux.pt  ← alive=4027, FVU=0.085
  sae_btk_blunder_mt_4096_k128_aux.pt ← alive=4092, FVU=0.066
  sae_btk_blunder_mt_8192_k32_aux.pt  ← alive=8024, FVU=0.101
```

All trained on 200K Lichess blunder move tokens (≥200cp loss), 10 epochs, BTK + aux.
Move-token = hidden[77] from DeepMind 270M encoder (matches production pipeline).
Winner (k=32): 2048 k=32 — best balance of unique labels (1,080) and quality (65% uniqueness).

## SAE Weights — Blunder k=8 sweep (2026-04-14)

```
s3://chess-stage-a-140023406996/sae-weights/
  sae_btk_blunder_mt_2048_k8_aux.pt  ← 202 alive, FVU=0.238, 90% dead
  sae_btk_blunder_mt_2048_k8.pt      ← 207 alive, FVU=0.236, 90% dead
  sae_btk_blunder_mt_1024_k8.pt      ← 175 alive, FVU=0.240, 83% dead
  sae_btk_blunder_mt_512_k8.pt       ← CANDIDATE: 143 alive, FVU=0.250, 72% dead, 59% energy
```

Key finding: at k=8, aux loss doesn't matter (same results ± noise). Dict size barely matters
(all produce ~150-200 alive features). 512 captures most energy per feature (59% vs 37%).
Need to label 512 variant to see if 16 coaching themes emerge from 143 features.

Format: PyTorch dict with `encoder_weight`, `encoder_bias`, `decoder_weight`, `pre_bias`, `k`, `dict_size`, `mean`, `std`.

## Profiles (top-20 examples per feature)

```
s3://chess-stage-a-140023406996/sae-eval/
  profiles_btk_2048_k64.json
  profiles_btk_2048_k32_aux.json
  profiles_btk_4096_k64_aux.json
  profiles_btk_4096_k32_aux.json
```

## Taxonomy (in chess-deck-research repo, not S3)

```
chess-deck-research/output/
  feature_taxonomy_v2.json          ← Blunder SAE: 10 categories, 3,529 features, coaching questions
  puzzle_taxonomy_v1.json           ← Puzzle SAE: 12 raw clusters (use labels.json coaching_category instead)
  categorization_experiments.md     ← Full 29-experiment log with findings
```

## Labels (in chess-deck-research repo, not S3)

```
chess-deck-research/output/
  k64_baseline/labels_sonnet_think.json   ← 2048 k=64 labels
  k32_aux_baseline/labels_sonnet_think.json ← 2048 k=32 labels
  labels_4096_k32_sonnet.json             ← 4096 k=32 labels
  labels_4096_k64_sonnet.json             ← 4096 k=64 labels
```

## Activation Caches (on notebook, not S3)

```
/home/ec2-user/SageMaker/chess-stage-a/cache/
  puzzle_acts_200k.pt              (~30GB, 200K×77×1024, all tokens)
  blunder_acts_200k.pt             (~60GB, 200K×77×1024, all tokens, ≥200cp loss)
  blunder_move_token_200k.pt       (804MB, 200K×1024, move token only)
  blunder_positions.json           (31MB, 200K blunder metadata from HuggingFace)
```

## Encoder Weights

```
/home/ec2-user/SageMaker/chess-stage-a/cache/
  deepmind_270m_params.npz   ← DeepMind 270M chess encoder
  move_to_action.json        ← UCI move → token ID mapping
```

## SAE Weights — Maia 3 (2026-05-22)

```
s3://chess-stage-a-140023406996/sae/maia3/
  maia3_sae_diff_2048_k32_l2_200ep.pt  ← WINNER: L2 norm, 200 epochs, FVU=0.191, 0 dead, 2007 labeled
  maia3_sae_diff_2048_k32_v2.pt        ← z-score only, FVU=0.188
  maia3_sae_diff_2048_k32_raw.pt       ← no norm, FVU=0.118 (best reconstruction, fewer labelable)
  maia3_sae_diff_2048_k32.pt           ← L2 50ep (deprecated by 200ep version)
  maia3_sae_diff_2048_k32_lr1e3.pt     ← higher LR experiment
  maia3_sae_diff_256_k16.pt            ← micro test
  l2_labels_sonnet.json                ← 2007 feature labels (Sonnet 4.6 + thinking)
  l2_feature_profiles.json             ← top-20 positions per feature
```

Trained on 200K Lichess blunder diff vectors (to_sq - from_sq) from Maia 3 layer 7 residual.
Input: 512-dim, L2 normalized (z-score + unit sphere). Mixed random Elo 600-2600.
Labels: 80% unique, 66% confidence ≥ 0.7. Categories: hanging_pieces (34%), king_safety (12%), forks (10%), pawn_endgames (8%), trapped_pieces (7%).

## SAE Weights — Maia 3 Matryoshka (2026-05-28)

```
s3://chess-stage-a-140023406996/sae/weights/matryoshka/
  maia3_matryoshka_2048_k16_p64_256_2048.pt     ← Config A (RECOMMENDED): elbows at 64/256, 0 dead, max 23%
  maia3_matryoshka_2048_k16_p32_128_512_2048.pt ← Config B: 4 levels, max 30%
  maia3_matryoshka_2048_k16_p32_96_224_480_992_2048.pt ← Config E: 6 Bussmann-style levels, max 26%
  maia3_matryoshka_2336_k20_p32_288_2336.pt     ← Config F: groups 32/256/2048, best top-level cos (0.230)
  maia3_matryoshka_2720_k22_p32_160_672_2720.pt ← Config C2: 1:4 branching, k=22
  maia3_matryoshka_2720_k24_p32_160_672_2720.pt ← Config C3: 1:4 branching, k=24, best progressive recovery
```

All trained on 200K blunder diff vectors, L2 normalized, BatchTopK + aux loss, 200 epochs.

```
s3://chess-stage-a-140023406996/sae/weights/matryoshka/
  maia3_matryoshka_perlevel_2336_p32_288_2336_k3_8_16.pt  ← BEST: per-level k, 0 dead, FVU=0.209
  maia3_matryoshka_perlevel_2336_p32_288_2336_k2_6_16.pt  ← H2: conservative top (5 dead at prefix-32)
```

Per-level k enforcement (novel modification). Groups [32, 256, 2048], each at its validated k.
See `docs/knowledge/matryoshka-sae.md` for full comparison and methodology.

## SAE Weights — Maia 3 Matryoshka V2 (CORRECT DATA, 2026-05-28)

```
s3://chess-stage-a-140023406996/sae/weights/matryoshka_v2/
  matryoshka_v2_H1_p32_288_2336_k3_8_16.pt   ← H1 on v2: [32,256,2048] groups, k=[3,8,16]
  matryoshka_v2_L3_p128_640_2688_k8_12_16.pt ← L3 on v2 (BEST): [128,512,2048], k=[8,12,16]
  sweep_v2_k16_d2048.pt                       ← standard k=16 baseline on v2
  full_sweep_v2_results.json                  ← all v2 sweep metrics
```

**v2 = corrected data** (`maia3_blunder_diff_v2.pt`, fixes the Black-to-move inverted-label bug).
v2 model features match the 18K Opus analyses 100% by index — labelable. v1 models are not.
Labels: `output/labels_matryoshka_v2_H1_top32.json` (32 top-level features, conf 62-91).

## SAE Weights — Maia 3 k-sweep (2026-05-28)

```
s3://chess-stage-a-140023406996/sae/cache/
  k_sweep_summary.json  ← Full results for k=[8,12,16,20,24,32,48] at dict=2048
```

k=16 validated as optimal (1531 interpretable features, 0 dead, cos=0.254).

## Maia 3 Activation Cache

```
s3://chess-stage-a-140023406996/sae/cache/
  maia3_blunder_diff.pt            ← 200K × 512 diff vectors (to_sq - from_sq), mixed Elo
  all_gemini_positions.json        ← 5829 unique positions with Gemini tactical analysis
  maia3_simplified.onnx            ← Maia 3 model (8-layer transformer, 512-dim)
```

## Opus Pass-1 Position Analyses (⚠️ S3 copy is truncated)

```
CANONICAL: chess-poc:/home/ec2-user/SageMaker/all_positions_labeled_opus.json  ← 19,342 analyses (79.5MB)
TRUNCATED: s3://chess-stage-a-140023406996/sae/cache/all_positions_labeled_opus_final.json  ← only 10,648 (40MB) — DO NOT USE
```

The S3 `_final.json` upload was truncated to 10,648 of 19,342 entries. Pull the complete file from the chess-poc notebook (`sais -n chess-poc download all_positions_labeled_opus.json <dest>`). Keyed by `fen|uci`; fields `analysis.{position_description, best_moves_analysis}`. (2026-05-29: cost a long debugging detour.)

## Maia 3 Taxonomy v2 (rebuilt 2026-05-29)

```
chess-deck-research/output/taxonomy_v2/
  taxonomy_v2.json        ← 1,996 features → 20 coaching categories + specific chips (SHIP ARTIFACT)
  REBUILD_REPORT.md       ← distribution, method, example relabels, coherence validation
  category_vocab.json     ← 20-category controlled vocabulary (reused from chess_blunder_taxonomy_v2)
  evidence.json           ← per-feature description + structural fingerprint + verification
```

Built TITLE→CATEGORIZE→CHIP from the accurate Pass-2 `description` field (NOT the lossy chips). Generic chips 398→0, no junk drawer. Pipeline in `scripts/sae/taxonomy/`. See knowledge.md § "Taxonomy rebuild".

## To use any model

1. Download weights: `aws s3 cp s3://chess-stage-a-140023406996/sae-weights/<file>.pt .`
2. Load: `torch.load(path, map_location='cpu', weights_only=False)`
3. Create SAE: `SAE(1024, ckpt['dict_size'], ckpt['k'])`
4. Load weights: `sae.encoder.weight.data = ckpt['encoder_weight']` etc.
5. Normalize input: `(hidden - ckpt['mean']) / ckpt['std']`
6. Forward: `recon, acts = sae(normalized_input)`

Labels are in the research repo. Profiles are on S3. Everything needed to deploy or experiment with any variant.

## New SAE Architecture — Option A / Board Diff / L2+L7 (2026-05-31)

### Weights
```
s3://chess-stage-a-140023406996/sae/weights/
  maia3_option_a_2048_k32.pt    ← h[best_to]-h[blunder_to] diff, layer-7 ONNX, 512-dim input
  maia3_board_diff_2048_k32.pt  ← mean64(after_best-after_blunder), layer-7 ONNX, 512-dim input
  maia3_l2l7_2048_k32.pt        ← concat(L2_mean64_diff,L7_mean64_diff), 79M PyTorch, 2048-dim input
```
All: BatchTopK, dict=2048, k=32, 200 epochs on 200k v1 blunder positions.

### Activation Caches (v1 positions, 200k each)
```
s3://chess-stage-a-140023406996/sae/cache/
  maia3_option_a_diff.pt      (~391MB) — h[best_to]-h[blunder_to], layer-7 ONNX
  maia3_board_diff_both.pt    (~415MB) — mean64(after_best-after_blunder), layer-7 ONNX
  maia3_l2l7_concat.pt        (~1.5GB) — concat L2+L7 mean64 diffs, 79M PyTorch
```

### Probe Results (2026-05-31)
- board_diff best overall: mean gap 0.038 across 6 taxonomy categories
- L2 (layer 2, 79M) better for positional mistakes; L7 better for tactical timing
- Eval on 4 real positions: board_diff correctly fired "Recapture leaves piece hanging"
  for both bishop_f5 and queen_h4 positions (feature f46 = genuine coaching signal)

### Large files moved to S3
```
s3://chess-stage-a-140023406996/sae/cache/
  maia3_stockfish_data.json     ← stockfish eval data for positions
  labels_2048_k64_canonical.json ← canonical labels for 2048-k64 SAE
  batch_input_maia3.jsonl       ← batch labeling input file
  opus_english.json             ← 19k Opus English analyses (canonical copy)
```
