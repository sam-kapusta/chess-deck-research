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

## Maia 3 Activation Cache

```
s3://chess-stage-a-140023406996/sae/cache/
  maia3_blunder_diff.pt            ← 200K × 512 diff vectors (to_sq - from_sq), mixed Elo
  all_gemini_positions.json        ← 5829 unique positions with Gemini tactical analysis
  maia3_simplified.onnx            ← Maia 3 model (8-layer transformer, 512-dim)
```

## To use any model

1. Download weights: `aws s3 cp s3://chess-stage-a-140023406996/sae-weights/<file>.pt .`
2. Load: `torch.load(path, map_location='cpu', weights_only=False)`
3. Create SAE: `SAE(1024, ckpt['dict_size'], ckpt['k'])`
4. Load weights: `sae.encoder.weight.data = ckpt['encoder_weight']` etc.
5. Normalize input: `(hidden - ckpt['mean']) / ckpt['std']`
6. Forward: `recon, acts = sae(normalized_input)`

Labels are in the research repo. Profiles are on S3. Everything needed to deploy or experiment with any variant.
