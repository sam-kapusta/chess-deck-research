# Prior Art: Non-Vision Encoder-LLM Bridges

**Date:** 2026-04-04  
**Type:** Literature review

## Key Finding

The encoder → projection → LLM architecture is now standard across MANY modalities, not just vision. Audio, music, scientific data, and time series all use the same pattern. Our chess bridge is not novel architecture — it's applying a proven pattern to a new domain.

## Most Relevant Papers

### NExT-GPT (2023, 760 citations)
- **Architecture:** Multimodal adaptors (1% of params) connect frozen encoders to frozen LLM
- **Key insight:** Projection layers are small and cheap. The encoder and LLM do the heavy lifting. Only 1% of total params are trained.
- **Relevance:** Validates our approach — frozen DeepMind encoder, small MLP projection, LoRA on Qwen. The projection doesn't need to be complex.

### MIDI-LLaMA (2026)
- **Architecture:** MusicBERT encoder → projection → Llama-3-8B
- **Training:** TWO phases — feature alignment (encoder→LLM mapping) THEN instruction tuning
- **Key insight:** The phased approach works for symbolic, structured, non-visual domains. Music is analogous to chess: structured, symbolic, non-visual, requires understanding temporal patterns.
- **Relevance:** Directly validates our phased curriculum (phases 1-2 = alignment, phases 3-5 = instruction tuning). If it works for music, it should work for chess.

### Steer-MoE (2025)
- **Architecture:** Mixture-of-Experts router between encoder and LLM
- **Key insight:** Different input types need different projection paths. A fixed MLP treats all inputs the same. MoE selects the right transformation based on content.
- **Relevance:** For chess, endgame positions might need different projection than middlegame tactics. Worth trying if the fixed MLP plateau persists.

### Layer-wise Attention Pooling (2025)
- **Architecture:** Attention across encoder layers, not just final layer
- **Key insight:** Different encoder layers encode different levels of abstraction. Attending across layers captures multi-scale features.
- **Relevance:** We've only used the final layer of DeepMind. Early layers have spatial info (E12 partially showed this). Middle layers have tactical computation. Attending across all 16 layers could give richer input to the projection.

## Implications for Our Architecture

1. **Our basic approach is validated.** Frozen encoder + small projection + LoRA LLM works across many domains. 1% tunable params is sufficient.

2. **Phased training is the standard.** Feature alignment → instruction tuning. Not novel — it's how MIDI-LLaMA, LLaVA, and others do it. Our curriculum (win prob → strategy → coaching) follows the same pattern.

3. **Potential improvements if we plateau:**
   - MoE projection: different paths for different position types
   - Layer-wise attention: attend across all 16 encoder layers, not just final
   - Cross-attention resampler: compress 78 tokens into fewer summary tokens (Flamingo-style)

4. **Our domain is actually EASIER than audio/music:**
   - Chess has finite, well-defined state (64 squares, 32 pieces)
   - Evaluation is objective (Stockfish ground truth)
   - The encoder is very strong (2895 Elo)
   - The MATE dataset has 592K annotated training examples

If the approach works for music (unbounded, subjective, no ground truth), it should work for chess.

## Papers to Read More Deeply

- NExT-GPT full paper — their multimodal adaptor training procedure
- MIDI-LLaMA — their two-phase training details and loss curves
- Steer-MoE — how the MoE router is implemented
