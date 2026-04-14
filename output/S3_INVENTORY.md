# S3 Asset Inventory — SAE Models (2026-04-13)

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
Winner: 2048 k=32 — best balance of unique labels (1,080) and quality (65% uniqueness).

Format: PyTorch dict with `encoder_weight`, `encoder_bias`, `decoder_weight`, `pre_bias`, `k`, `dict_size`, `mean`, `std`.

## Profiles (top-20 examples per feature)

```
s3://chess-stage-a-140023406996/sae-eval/
  profiles_btk_2048_k64.json
  profiles_btk_2048_k32_aux.json
  profiles_btk_4096_k64_aux.json
  profiles_btk_4096_k32_aux.json
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

## To use any model

1. Download weights: `aws s3 cp s3://chess-stage-a-140023406996/sae-weights/<file>.pt .`
2. Load: `torch.load(path, map_location='cpu', weights_only=False)`
3. Create SAE: `SAE(1024, ckpt['dict_size'], ckpt['k'])`
4. Load weights: `sae.encoder.weight.data = ckpt['encoder_weight']` etc.
5. Normalize input: `(hidden - ckpt['mean']) / ckpt['std']`
6. Forward: `recon, acts = sae(normalized_input)`

Labels are in the research repo. Profiles are on S3. Everything needed to deploy or experiment with any variant.
