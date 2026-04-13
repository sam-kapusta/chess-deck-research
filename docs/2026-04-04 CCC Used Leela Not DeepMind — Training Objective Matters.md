# CCC Used Leela, Not DeepMind — Training Objective Matters

**Date:** 2026-04-04  
**Type:** Research finding — critical context for diff probe experiment

## CCC's Probe Results
Linear SVMs on Leela Chess Zero layer 40 achieved **0.91 accuracy** on 22 chess concepts (material, mobility, king_safety, etc.). NAACL 2025.

## Why This Might Not Transfer to DeepMind

**Leela was trained with self-play (MCTS + policy/value network).** Self-play requires understanding WHY a position is good — which pieces are active, where the king is vulnerable, what pawn breaks exist. The model needs structural understanding to improve through self-play. The representations encode concepts BECAUSE the training demanded it.

**DeepMind was trained with supervised action-value prediction.** Given (position, move), predict win probability. The model only needs to output a NUMBER. It doesn't need to understand WHY the win probability is what it is — just what the number is. The representations are optimized for evaluation accuracy, not concept decomposition.

## The Prediction

| Probe test | Leela (CCC) | DeepMind (ours) |
|-----------|-------------|-----------------|
| Linear probe for evaluation | ~0.91 | **0.85-0.95** (confirmed: 85% 3-class, 83% 128-bucket) |
| Linear probe for concepts | **0.91** | **??? (untested — the diff probe experiment)** |
| If concepts fail | N/A | Falls to 0.5-0.6 (near random for 5 classes) |

CCC's 0.91 was achievable because Leela's representations are structured around concepts. Our encoder's might not be.

## If Diff Probes Fail

It's not a failure of the approach — it's a mismatch between encoder and task. DeepMind's encoder was trained for evaluation, not concept understanding. The fix would be:

1. **Use Leela instead of DeepMind.** Leela's representations are proven to encode concepts (CCC's result). We'd need to run Leela inference, which is heavier (MCTS) but conceptually cleaner.
2. **Fine-tune DeepMind with concept supervision.** Add a concept classification head and train the last few encoder layers on MATE labels. Risky (catastrophic forgetting) but direct.
3. **Accept evaluation-only bridge + separate concept extraction.** The bridge carries win probability. Concept extraction (hand-coded tags, Maia features) handles strategy. Different tools for different signals.

## If Diff Probes Succeed

The training objective argument is wrong — DeepMind's representations DO encode concepts despite being trained only on evaluation. This would be a genuinely interesting finding: evaluation training creates concept structure as a byproduct.

This would mean: optimizing for "how good is this move?" implicitly teaches the model "why is this move good?" — because knowing why helps predict how good. The concepts emerge as intermediate computations that serve evaluation.

## Implications for the Product

If DeepMind can't do concepts → hybrid architecture: bridge for eval + tags for concepts.
If DeepMind CAN do concepts → unified bridge: encoder carries everything, LLM verbalizes it.

The diff probes decide which product we're building.
