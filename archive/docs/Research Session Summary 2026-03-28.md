# Research Session Summary — 2026-03-28

## What was researched

Deep dive into encoder fusion architectures for chess coaching. The question: how to wire the DeepMind 270M chess model (2895 Elo) into Qwen 4B for coaching commentary.

## Key findings

### Architecture Decision: LLaVA 1.5 pattern (2-layer MLP)
- **Source evidence:** LLaVA v1/v1.5, BLIP-2, SALMONN all use encoder → projection → LLM
- **MLP beats linear** (LLaVA v1 → v1.5 ablation)
- **Q-Former is overkill** for chess's compact 77-token input
- **Dimension mapping:** Chess 1024 → Qwen 4096 (4x upscale, same as LLaVA CLIP→LLaMA)
- **~16M trainable params** in the projection

### Data Requirements: Lower than expected
- **chess-ai-tutor achieved coaching quality with 15K samples** (SFT phase)
- **TextME paper:** zero-shot cross-modal transfer possible by exploiting geometric structures
- **Revised estimate:** 15K-50K paired examples for Stage 1
- **Jhamtani (298K) alone is sufficient** before adding MATE or generated data

### Encoder Quality: Validated
- **DeepMind 270M shows genuine compositional understanding** — not memorization (Compositional Generalization paper)
- **McGrath probed 209 concepts** in AlphaZero (similar training) — all linearly encoded
- **Representations are coaching-friendly** — trained on Stockfish value distillation, aligns with how coaches evaluate positions

### Novelty: Confirmed
- **Nobody has used a game-playing neural network as an encoder for language generation** — in any game
- **CCC paper is closest** but uses concept vectors via SVMs, not direct representation projection
- **Retrieval-augmented chess commentary** also novel — no published work

### Reference Implementation: chess-ai-tutor
- Qwen 4B + 72M CNN + LoRA + GRPO with 6 verifiable reward functions
- R3b reward (SF15 term accuracy, weight 0.35) is the correctness enforcer
- Only 15K training samples
- Active development (March 2026)

## What was produced

### Documents
- `Encoder Fusion Architecture Research.md` — full architecture comparison, data strategy, alternatives
- `Encoder Extraction Guide.md` — DeepMind model conversion guide, weight mapping, evaluation plan

### Cloud2 Agent Status
- Running for 37+ minutes, created `crawl_lichess_studies.py` on SAIS
- Agent is working on both workstreams in parallel
- 270M.zip on SAIS is corrupt (needs re-download) — flagged in `270M_STATUS.md`

### Open Items for Sam
1. **270M weights:** Need fresh presigned URL from S3 → download to SAIS
2. **Cloud2 agent:** Running but hasn't written research_plan.md yet — check tmux session
3. **Maia-2:** Still not integrated in MCP (code written, needs `pip install maia2` in MCP venv)
4. **Feigned Discovery Prompting:** Applied to prompts.py + server.py, needs deploy + testing

## What's next

1. **Cloud2 agent completes** encoder conversion (may need 9M model first due to 270M zip corruption)
2. **Cloud2 agent completes** Lichess dataset extraction
3. **Validate encoder** using probing evaluation (board state, strategic concepts, similarity clustering)
4. **Build projection layer** — 2-layer MLP (1024 → 4096)
5. **Stage 1 training** — freeze encoder + LLM, train MLP on Jhamtani data
6. **Stage 2 training** — unfreeze LoRA, instruction tune
7. **Evaluate** — compare coaching quality with vs without encoder
