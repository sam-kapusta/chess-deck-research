# Encoder Fusion Architecture Research

**Date:** 2026-03-28
**Question:** What's the best architecture for fusing a chess encoder with an LLM for coaching commentary?

---

## Cross-Domain Projection Patterns

Every successful domain encoder + LLM system follows the same pattern:
1. Frozen domain encoder extracts features
2. Trainable projection maps features into LLM embedding space
3. LLM generates text grounded in those features

### Projection Architectures Compared

| System | Domain | Encoder | Projection | Trainable | Data | Key Insight |
|--------|--------|---------|------------|-----------|------|-------------|
| LLaVA v1 | Vision | CLIP ViT-L | Linear | ~1M | 753K | Simplest possible works |
| LLaVA v1.5 | Vision | CLIP ViT-L/336 | **2-layer MLP** | ~4M | 1.2M | MLP > linear, surprisingly data-efficient |
| BLIP-2 | Vision | ViT-G | Q-Former (32 queries) | 188M | Large | Overkill for compact encoders |
| SALMONN | Audio | Whisper + BEATs | Window Q-Former + LoRA | 33M | 2.3M | Window-level reduces sequence length |
| MolReGPT | Molecular | None (retrieval) | In-context learning | 0 | 0 | Skip projection entirely with retrieval |
| chess-ai-tutor | Chess | ResNet CNN (72M) | Vision pad injection | LoRA | 15K | Already working, active development |

### Recommendation for Chess

**2-layer MLP (LLaVA v1.5 pattern).** Reasons:
- Chess encoder output (77 tokens × 1024 dim) is compact — no need for Q-Former compression
- MLP beats linear projection (proven by LLaVA v1 → v1.5 ablation)
- Only ~4M trainable parameters in the projection
- Data-efficient: LLaVA 1.5 achieved SOTA with only 1.2M examples total

**Dimension mapping:** Chess encoder (1024) → Qwen 4B (4096). The MLP upscales 4x:
```
Linear(1024, 4096) → GELU → Linear(4096, 4096)
```
~16M parameters. Same ratio as LLaVA (CLIP 1024 → LLaMA 4096). Well-understood.

**Q-Former is overkill** because:
- Designed for vision where encoder outputs are large (257 patches × 1024)
- Chess encoder is already 77 tokens — similar scale to LLM sequence lengths
- Q-Former adds 188M trainable params vs 4M for MLP

---

## Training Procedure

Based on LLaVA + SALMONN patterns:

### Stage 1: Feature Alignment (MLP only)
- **Freeze:** Chess encoder (DeepMind 270M) + LLM (Qwen 4B)
- **Train:** 2-layer MLP projection only (~16M params)
- **Data:** (FEN → encoder hidden states, coaching commentary) pairs
- **Goal:** Teach the MLP to map chess representations into LLM embedding space
- **Duration:** ~30-60 minutes on 1×A10G
- **Estimated data needed:** 15K-50K examples
- **Hyperparameters (from LLaVA):**
  - Learning rate: 2e-3 (high — just the projection)
  - Batch size: 128
  - Epochs: 1-3
  - Optimizer: Adam
  - Scheduler: cosine with 3% warmup

### Stage 2: Instruction Tuning (MLP + LoRA)
- **Freeze:** Chess encoder
- **Train:** MLP + LoRA on LLM
- **Data:** Instruction-formatted coaching examples
- **Goal:** Teach the LLM to generate coaching text from chess encoder features
- **Duration:** Longer — LoRA on 4B model
- **Estimated data needed:** 100K-500K examples

### Stage 3 (Optional): RL with Verifiable Rewards
- Following chess-ai-tutor's GRPO approach
- 6 reward functions: format, legality, reasoning, eval term accuracy, coaching tone, educational quality
- Pure REINFORCE (beta=0), no reference model
- This is the polish phase — SFT should work first

---

## The chess-ai-tutor Reference Implementation

**github.com/helloworld0909/chess-ai-tutor** is the closest existing implementation:

- **Encoder:** 72M ResNet CNN (pretrained on SF15 eval regression)
- **Base LLM:** Qwen3.5-4B
- **Projection:** Vision pad token injection (encoder embeddings replace `<|vision_pad|>` tokens)
- **Training:** SFT → GRPO with 6 reward functions
- **Data:** 15K training samples from Lichess + SF15 annotations
- **Key reward (R3b, weight 0.35):** Validates model correctly interprets classical eval terms

### What they do differently from our proposed approach:
1. **CNN encoder vs transformer encoder** — their ResNet is 72M params, our DeepMind model is 270M. Ours should produce richer representations.
2. **Pretrained on eval regression** — their CNN was trained to predict Stockfish classical eval terms. The DeepMind model was trained on move prediction at 2895 Elo — much more chess knowledge.
3. **15K training samples** — surprisingly small. LLaVA used 753K-1.2M. But chess-ai-tutor's GRPO phase compensates for less data with reward-shaped learning.
4. **Stockfish classical eval terms** — they ground coaching in specific terms (mobility +0.32, king safety -0.15). We'd need to decide: ground in eval terms (like them) or in raw encoder representations (like LLaVA).

### Key takeaway from chess-ai-tutor:
The **reward functions** are the real innovation. R3b (SF15 term accuracy, weight 0.35) forces the model to correctly interpret position features. This is the chess equivalent of the "correctness" metric in CCC's GCC-Eval. Without it, the model generates fluent but hallucinated coaching.

---

## Alternative: Skip the Encoder, Use Retrieval

MolReGPT showed you can skip the encoder entirely:
1. For each position, retrieve 3-5 similar positions from the Lichess studies dataset
2. Include their human commentaries as few-shot examples
3. Let the LLM generate commentary for the current position

**Pros:** No training, no encoder, no projection. Works immediately.
**Cons:** Limited by retrieval quality. Can't "understand" the position — just pattern-matches similar ones.

This could be a strong baseline to compare the encoder fusion approach against.

---

## Alternative Encoders

### DeepMind 270M (proposed)
- 16 layers, 1024 embedding dim, 8 heads
- Trained on 10M games, 15B Stockfish annotations
- 2895 Elo — strongest chess representations available
- Apache 2.0
- **Pro:** Deepest chess understanding
- **Con:** JAX, needs conversion. 270M params is large.

### DeepMind 9M
- 8 layers, 256 embedding dim, 8 heads
- Same training data but much less capacity
- **Pro:** Tiny, fast inference, easy to convert
- **Con:** 256 dim is small — would need upscaling in the projection

### Leela Chess Zero
- Various sizes (15-layer T82 is well-studied for interpretability)
- Trained by self-play (not distillation from Stockfish)
- **Pro:** Well-understood internals (Jenner et al. probing studies), native PyTorch available
- **Con:** Doesn't learn from game data like DeepMind model — learns from self-play

### Karvonen's Chess GPT (50M)
- 8 layers, 512 dim, trained on 16M Lichess games in PGN
- Learns board state internally (99.6% accuracy via probes)
- **Pro:** PyTorch native, smallest, well-studied internally
- **Con:** Weakest chess play, PGN-based (not FEN)

### chess-ai-tutor's ResNet CNN (72M)
- Custom architecture, pretrained on SF15 eval regression
- **Pro:** Already proven to work with LLM fusion
- **Con:** CNN not transformer — different representation style. Trained on eval terms, not move prediction.

### Recommendation
**Start with DeepMind 9M for rapid prototyping** (tiny, same architecture as 270M, quick to validate the pipeline). Then swap in 270M once the pipeline works. The 9M→270M swap should be a config change, not an architecture change.

### Why DeepMind 270M Over Leela
- **Stronger:** 2895 Elo. Leela varies by network version.
- **Simpler:** Pure transformer (LLaMA-style). Leela has custom architectures.
- **License:** Apache 2.0. Leela is GPL-3.0 (would infect derivative work).
- **Research:** Compositional generalization validated (2510.20783). Shows genuine understanding, not memorization.
- **Python:** Clean JAX code, straightforward conversion to PyTorch. Leela's Python bindings are stale/unmaintained.
- **One-time cost:** The JAX→PyTorch conversion is done once, then it's a standard PyTorch nn.Module forever.

---

## Data Requirements Estimate

Based on cross-domain patterns:

| Stage | LLaVA (vision) | SALMONN (audio) | chess-ai-tutor | Our estimate |
|-------|----------------|-----------------|----------------|-------------|
| Pretrain | 595K | 2.3M | 15K (SFT) | 50K-100K |
| Finetune | 665K | Included above | GRPO on same | 50K-100K |
| Total | 1.2M | 2.3M | 15K | 100K-200K |

Chess is a simpler domain than vision or audio (fixed vocabulary, deterministic rules, compact state). The data required may be much less than initially estimated:

- **chess-ai-tutor achieved working coaching with just 15K samples** (SFT phase)
- **LLaVA 1.5 is "surprisingly data-efficient"** — 1.2M total but they note you don't need billions
- **TextME (2602.03098) achieves zero-shot cross-modal transfer** using only text descriptions, no paired supervision — exploits geometric structures of pretrained encoders
- **Projection layers are consistently data-efficient** across all domains studied

**Revised estimate: 15K-50K paired examples may be enough for Stage 1.** This means the existing Jhamtani dataset (298K) alone is more than sufficient, before even adding MATE or generated data. The bottleneck is likely quality, not quantity.

---

## Data Strategy Update

**Lichess studies bulk export doesn't exist.** No API for searching/downloading studies in bulk. The cloud2 agent is working on scraping via the Lichess web interface or individual study export endpoints, but volume is uncertain.

**Alternative data sources (no scraping needed):**

| Source | Size | Format | Quality | Effort |
|--------|------|--------|---------|--------|
| **Lichess evaluations DB** | 362M positions | JSONL (FEN + Stockfish PVs) | Engine evals, no language | Zero — just download |
| **Jhamtani (GameKnot)** | 298K (move, comment) pairs | Text | Natural coaching language | Zero — already published |
| **MATE dataset** | 1M positions | Parquet | Strategy + tactics annotations | Zero — on HuggingFace |
| **ChessCOT** | 4.5M positions | HuggingFace | Chain-of-thought in UCI | Zero — on HuggingFace |
| **Generated via Claude** | Unlimited | Custom | Coaching voice, controlled quality | API cost (~$50-100 for 30K) |
| **Lichess studies** | Unknown (50K+ target) | PGN + comments | Natural coaching, varied quality | Scraping effort |
| **Your 3,843 games** | ~15K moments | Custom JSON | Has behavioral tags (unique) | Already exists |

**Recommended approach:**
1. Start with Jhamtani (298K) + MATE (1M) — free, published, no scraping
2. Supplement with generated data from Claude on your tagged games (unique data)
3. Add Lichess studies if scraping yields enough quality
4. Use Lichess evaluations DB for eval grounding (not language, but position data)

The key insight: **you don't need Lichess studies to start training.** Jhamtani + MATE + generated data is enough for Stage 1. Studies are a nice-to-have for coaching voice diversity.

## Baseline: Retrieval-Augmented Commentary (No Training)

Before investing in encoder fusion, implement a simpler baseline:

1. Index all Jhamtani commentary pairs by position similarity (material, piece placement, pawn structure)
2. For each new position, retrieve 3-5 most similar positions from the index
3. Include their human commentaries as few-shot examples in the LLM prompt
4. LLM generates commentary for the current position grounded in real examples

**Nobody has published this approach for chess.** It's a novel contribution with zero training cost.

**Pro:** No encoder, no training, no projection. Works immediately. Quality comes from retrieval.
**Con:** Limited by how similar the retrieved positions are. Novel positions get worse examples.

This is the floor to beat. If encoder fusion doesn't significantly exceed retrieval-augmented commentary, the simpler approach wins.

## Novel Paper Ideas (Emerging from This Research)

Based on everything researched, there are at least 3 publishable contributions nobody has made:

1. **Chess Encoder + LLM Fusion for Coaching Commentary** — using the DeepMind 270M as a domain encoder with LLaVA-style projection to Qwen. First to wire a 2895 Elo chess model to a language model.

2. **Behavioral Tag-Driven Chess Coaching** — using cross-game pattern detection (your 33 tags) to ground LLM coaching in behavioral analysis, not just positional analysis. The CCC paper does positional concepts; you do behavioral concepts. Complementary.

3. **Human-Calibrated Chess Explanation** — using Maia-2 predictions to frame coaching relative to the player's skill level. "Most 1800s would play X, the move that separates 1900 from 1800 is Y."

Any one of these is a paper. All three together is a system paper.

## Open Questions

1. **Should the chess encoder output be pooled or kept as a sequence?** LLaVA keeps all image patches as separate tokens. We could keep all 77 FEN tokens, or pool to a single 1024-dim vector. The chess-ai-tutor injects per-move embeddings at vision_pad positions (sequence approach).

2. **What's the right representation layer?** The DeepMind model has 16 layers. The last hidden layer (before policy/value heads) is the obvious choice, but intermediate layers might encode different concepts (McGrath's what-when-where finding).

3. **Does the DeepMind model need FEN input or can we use board tensors?** The tokenizer converts FEN to 77 integer tokens. But for the projection, we might want the model's internal representation of the position, not its next-token-prediction logits.

4. **How to handle the action-value framing?** The DeepMind model is trained to predict P(return_bucket | state, action). For feature extraction, we want state representations, not action-conditioned ones. See "Encoder Extraction Guide.md" for detailed analysis — test both FEN-only (77 tokens) and full sequence (79 tokens with dummy action/return) on the 9M model first.

5. **Are the DeepMind model's representations coaching-relevant?** Yes, based on prior work:
   - McGrath (PNAS 2022) probed AlphaZero (also trained on values) and found 209 chess concepts linearly encoded — material, king safety, mobility, threats, pawn structure, all there.
   - Karvonen found board state at 99.6% accuracy in chess GPTs trained on PGN.
   - The SAE paper (NeurIPS 2024) found strategic concepts (pins, forks, checks) in chess model internals.
   - The DeepMind model was trained on **distilled** Stockfish evaluations — to predict move values, it MUST encode positional concepts internally. Distillation-trained models tend to produce smoother, more interpretable representations than self-play models.
   - **Key advantage:** the model's concept space aligns with how engines evaluate positions — the same framework human coaches use. This should make the representations naturally coaching-friendly.

6. **Multi-signal fusion.** For combining chess encoder + Maia + tags, use simple concatenation (SALMONN pattern): project each signal into the LLM's embedding space, concatenate as prefix tokens. No need for complex gated or cross-attention fusion — the LLM's self-attention handles the integration.
