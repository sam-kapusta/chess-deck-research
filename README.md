# Chess SAE Research

Sparse Autoencoder research on the DeepMind 270M chess encoder. Training SAEs to decompose encoder activations into interpretable chess features for coaching.

## Repo Structure

```
plan.md            Current state, experiment queue, beliefs
log.md             Session-by-session history of what happened
findings.md        Validated results with evidence tables
learnings.md       Indexed insights with evidence pointers
scripts/           All pipeline code (training, profiling, labeling, caching)
output/            Results, labels, metrics, S3 inventory
docs/              Research papers, SAE analysis, reference material
archive/           Old code and superseded data
```

## Current Focus (2026-04-13)

**Blunder SAE experiment.** Training SAEs on 200K blunder move tokens (≥200cp loss from Lichess eval dataset) to discover "mistake pattern" features. Five variants being labeled with Sonnet+thinking.

**Production SAE:** `puzzle_2048_k64_v1` — puzzle-trained, BA=0.632, 218 features served.

See `plan.md` for full state.

## Pipeline

1. **Cache activations** — `scripts/sae/cache_move_token.py` or `cache_activations.py`
2. **Train SAE** — `scripts/sae/train_blunder_sae.py` (BTK + aux loss)
3. **Profile** — `scripts/encoder/profile_sae.py` (top-20 FEN examples per feature)
4. **Label** — `scripts/evaluation/batch_label_and_score.py label` (Sonnet+thinking via Bedrock Batch)
5. **Detection score** — `scripts/evaluation/batch_label_and_score.py score` (does the label match the positions?)

## Infrastructure

- **Encoder:** DeepMind 270M (searchless_chess), bidirectional transformer, 1024-dim hidden
- **Notebook:** chess-poc (ml.g6.16xlarge, L4 GPU, 242GB RAM)
- **Account:** 140023406996 (default profile)
- **S3:** `s3://chess-stage-a-140023406996/sae-weights/` — see `output/S3_INVENTORY.md`
- **Labeling:** Bedrock Batch with Sonnet+thinking

## Key Design Decisions

- **BatchTopK over L1** — no shrinkage, controllable sparsity, zero noise features
- **Move token only** — hidden[77] from encoder, not all 77 tokens. Matches production.
- **Aux loss** — 1/32 coefficient, dead_threshold=50. Recovers dead features.
- **Two-phase caching** — CPU download/filter, then GPU batch encode. Enables fast retraining.

See `output/blunder_sae_reasoning.md` for full reasoning.
