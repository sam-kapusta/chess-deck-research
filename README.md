# Chess Encoder Research

Wire DeepMind's 270M chess encoder (2895 Elo) into Qwen2.5-7B for chess position understanding.

**Active work lives in `.lab/chess/`** — see `plan.md` for current state, `findings.md` for evidence.

## Current Status (2026-04-01)

LoRA v2 training (norm-fixed) running on chess-research notebook. 10K smoke test, eval+ablation will auto-run on completion.

**Key findings so far:**
- Encoder contains eval direction signal at 67% (F1)
- NTP Phase 1 destroys that signal to 53% (F2)
- Contrastive Phase 0 preserves 92% of signal — 61.4% (F3b)
- LayerNorm alone gives 60x norm mismatch with Qwen — must explicitly scale (F10)
- LoRA v1 was catastrophically broken (NaN masked by nan_to_num). v2 fixes norms.

## Architecture

```
FEN → tokenize (77 tokens)
  → DeepMind 270M encoder (frozen) → [77, 1024]
  → Contrastive projection MLP (1024→3584, 16M params, frozen)
  → LayerNorm + scale to match Qwen embedding norm (~0.94)
  → prepend to Qwen2.5-7B input embeddings
  → LoRA-64 fine-tuned Qwen generates structured UCI output
```

## Training Pipeline

| Step | What | Status |
|------|------|--------|
| 0. Contrastive | InfoNCE alignment (200K evals, 3 epochs) | **Done** — preserves 92% signal |
| 1. LoRA Phase 2 | LoRA-64 on Qwen + frozen contrastive projection | **v2 in progress** (v1 failed: F10) |
| 2. Scale data | 50K → 427K mixed | Blocked on Step 1 |
| 3. RL | GRPO with Stockfish rewards | Future |

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| Lichess evals | 200K (from 13M) | Contrastive Phase 0 + eval prediction |
| Lichess puzzles | 200K (from 5.8M) | Tactical patterns |
| Mixed (evals + puzzles + our positions) | 50K subset | LoRA Phase 2 training |
| Eval positions | 500 held-out | Evaluation |

## Infrastructure

- **Account:** 140023406996 (personal dev)
- **S3:** `s3://chess-stage-a-140023406996/`
- **Notebook:** chess-research (ml.g5.xlarge, A10G 23GB) — $1.40/hr
- **Scripts:** on notebook at `/home/ec2-user/SageMaker/chess-stage-a/scripts/`

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_lora_v2.py` | LoRA Phase 2 with norm scaling fix |
| `train_contrastive.py` | Contrastive Phase 0 (InfoNCE) |
| `eval_lora.py` | Eval + ablation for LoRA models |
| `debug_lora.py` | Diagnostic: norms, generation tests, token analysis |
| `chess_model.py` | Pure PyTorch encoder (no JAX deps) |
| `fen_tokenizer.py` | FEN tokenizer (77 tokens) |
| `projection_layer.py` | 2-layer MLP projection |
| `probe_projection.py` | 5-fold CV probing of encoder/projection signal |

## Encoder Probing Results

| Probe | Accuracy |
|-------|----------|
| Check detection | 97.7% |
| Game phase | 94.8% |
| Piece identification | 92.6% |
| Material count | r=0.99 |
| Tactical volatility | r=0.78 |
| Attack detection | 77% (+26% over baseline) |
| Eval direction (all tokens) | 66.9% |

The encoder knows abstract search-level concepts — not just FEN-parseable features.

## Findings Summary

See `.lab/chess/findings.md` for full evidence tables.

| # | Finding | Key number |
|---|---------|------------|
| F1 | Encoder encodes eval direction | 67% accuracy |
| F2 | NTP Phase 1 destroys signal | 53% (worse than random projection) |
| F3 | NTP and signal preservation in tension | SimReg failed |
| F3b | Contrastive Phase 0 preserves signal | 61.4% (92% of raw) |
| F4 | Encoder shifts Qwen into chess mode | Qualitative |
| F5 | Gradient checkpointing needs use_cache=False | 37% VRAM reduction |
| F6 | Phase 1 has severe diminishing returns | 80% learning in 10% steps |
| F7 | Information asymmetry is root cause | Cross-modal research |
| F8 | Contrastive projection 26x larger than Qwen | LayerNorm required |
| F9 | LayerNorm dtype mismatch crashed FSDP | Must stay float32 |
| F10 | LayerNorm gives 60x norm mismatch | Must scale after LN |
