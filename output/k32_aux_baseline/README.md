# BTK 2048 k=32 + Aux Loss — Baseline Results (2026-04-12)

## SAE Training
- Config: BatchTopK, dict=2048, k=32, aux_coeff=1/32, dead_threshold=50
- Data: 200K Lichess puzzles, 5 epochs
- Dead features: 184/2048 (9%) — down from 1,161/2048 (57%) at k=32 without aux
- Active: 1,864
- FVU: 0.128, c_dec: 0.045, L0: 32.0

## Profiling
- 1,918 alive features profiled (130 dead on eval sample)
- 50,000 puzzle positions, top-20 examples per feature
- Fire rate: mean=16.20%, median=6.83%

## Labeling (pending)
- Model: Sonnet 4 + 4K thinking tokens
- Job: `blgxhwvvr7rh`
- Enriched FENs (Stockfish + python-chess)
- 12,324 FENs enriched for this variant

## Detection Scoring (pending)
- Will run after labels parse
- Same method as k=64: Haiku judge + enriched FENs

## Comparison target
- 2048 k=64 + aux (Sonnet labels): Mean BA 0.632, HOLDS 659, STRONG 325

## Job ARNs
- Labeling: `blgxhwvvr7rh` (Sonnet+thinking, enriched)
