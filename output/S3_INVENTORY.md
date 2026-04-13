# S3 Asset Inventory — SAE Models (2026-04-12)

Bucket: `s3://chess-stage-a-140023406996/`
Account: 140023406996 (research, default profile)

## SAE Weights

```
s3://chess-stage-a-140023406996/sae-weights/
  sae_btk_2048_k64.pt       ← WINNER (BA=0.632, deploy this)
  sae_btk_2048_k32_aux.pt   ← BA=0.557, low poly (3.7%)
  sae_btk_4096_k64_aux.pt   ← BA=0.566, most features (3,017)
  sae_btk_4096_k32_aux.pt   ← BA=0.563, most selective
```

All trained on 200K Lichess puzzles, 5 epochs, BatchTopK + aux loss (1/32).
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

## Activation Cache

```
s3://chess-stage-a-140023406996/cache/  (on notebook, not S3)
  /home/ec2-user/SageMaker/chess-stage-a/cache/puzzle_acts_200k.pt  (~30GB)
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
