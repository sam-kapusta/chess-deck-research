# Qwen × SAE Label Training Plan

## The Idea

Train Qwen to predict SAE feature labels directly from encoder hidden states. No FEN text input — encoder tokens are the only information channel. This forces Qwen to learn what the encoder representations mean.

## Why This Is Different From Previous Attempts

Previous bridge attempts failed because of **information asymmetry** — Qwen could read the FEN text and learned to ignore the encoder projection. Mode collapse to "generic chess mode."

This time:
1. **No FEN input.** Encoder hidden states are the sole input. Qwen must learn from them or fail.
2. **SAE labels as targets, not coaching text.** Short, concrete outputs: "zwischenzug", "queen fork", "capturing hanging piece". Not open-ended generation.
3. **Per-token SAE is proven.** 394 features with real chess concepts (54% high/medium confidence). The labels are good.
4. **Training data is free.** Run encoder on any Lichess position, get SAE features, look up labels. Millions of training examples.

## Architecture

```
Encoder (pos + move) → [77, 1024] per-token hidden states
  → Projection (1024 → Qwen dim) per token
  → Qwen 2.5-0.5B (or 1.5B)
  → Output: "fork, exposed king, pawn break"
```

**Why small Qwen?** This isn't open-ended reasoning. It's classification through a language model. 0.5B should be enough for outputting 3-8 word labels. Smaller = faster training, faster inference.

**Why not just classification head?** A language model can output variable-length descriptions and combine concepts. "Zwischenzug exploiting hanging piece" isn't a fixed class — it's a composition. Also, once Qwen learns the encoder→concept mapping, it can generate novel descriptions.

## Training Data

### Source 1: Puzzle positions (existing)
- 100K Lichess positions with encoder activations (cached on chess-poc)
- Each fires ~32 SAE features with labels
- ~394 unique labels, high quality

### Source 2: Blunder positions (building now)
- 50K blunder positions with encoder activations (caching in progress)
- Same SAE → labels
- Different distribution — what bad moves look like

### Format
```json
{
  "encoder_hidden": [77, 1024],  // input
  "target_labels": ["queen fork", "exposed king", "forcing check"],  // top 3-5 labels by activation strength
  "move_type": "best",  // or "blunder"
  "cp_loss": 0  // or 250 for blunders
}
```

### Data generation pipeline
1. Stream Lichess positions
2. Encode with DeepMind 270M → [77, 1024]
3. Run SAE → 32 active features
4. Look up labels for each feature
5. Sort by activation strength, take top 5
6. Save (hidden_states, label_list) pair

Can generate 100K+ examples per hour on chess-poc.

## Training Plan

### Phase 1: Projection alignment (freeze Qwen)
- Train only the projection layer (1024 → Qwen dim)
- Contrastive loss: positions with same SAE labels should project to similar Qwen embeddings
- 50K positions, ~30 min on A10G
- Validates that encoder signal reaches Qwen's input space

### Phase 2: Fine-tune Qwen (LoRA)
- LoRA rank 16-32 on Qwen
- Input: projected encoder tokens (no FEN)
- Output: comma-separated labels, sorted by strength
- Standard NTP loss
- 100K positions, ~1-2 hours on A10G

### Phase 3: Evaluate
- Hold-out 5K positions
- Metrics:
  - **Label accuracy:** does Qwen produce the correct SAE labels?
  - **Novel compositions:** does it combine labels in meaningful ways?
  - **Blunder description:** given encoder(pos, blunder_move), does it describe what's wrong?
- Qualitative: show 20 positions to Sam, rate whether Qwen's output is useful

## What Success Looks Like

**Minimum viable:** Qwen correctly predicts top-3 SAE labels >60% of the time from encoder tokens alone.

**Home run:** Qwen outputs useful tactical descriptions that go beyond the fixed label vocabulary. "Knight fork winning the exchange on f7" from hidden states alone.

**Ship it:** Qwen replaces SAE label lookup at inference. One forward pass: encoder → projection → Qwen → coaching concepts. No SAE needed at runtime.

## What Could Go Wrong

1. **Projection collapse (again)** — mitigated by removing FEN entirely
2. **Label set too small** — 394 labels might not be rich enough. Fix: use explanation field too, not just label
3. **Overfitting** — Qwen memorizes position→label mappings. Fix: large diverse training set from Lichess
4. **Qwen too small** — 0.5B can't learn the mapping. Fix: try 1.5B or 3B

## Dependencies

- [x] Per-token SAE with labels (done — 394 features, 54% confident)
- [x] Encoder activation caching pipeline (done)
- [ ] Blunder activation cache (in progress — 50K positions)
- [ ] Encoder blunder SAE trained + labeled
- [ ] Training data generation script
- [ ] Qwen fine-tuning script

## Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| SAE labels → Claude prompt (current) | Works now, no training | Claude doesn't understand features, expensive per call |
| Qwen × SAE labels (this plan) | Learns real understanding, fast inference, can compose | Needs training, might fail |
| Full LLaVA bridge | Most flexible | Failed before (info asymmetry), heavy training |
| Probes → text | Simple | Fixed vocabulary, no composition |
