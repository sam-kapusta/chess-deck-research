# Implicit Look-Ahead: Does the Encoder See Future Moves?

**Date:** 2026-04-04  
**Type:** Research finding + experiment design

## The Finding

Jenner et al. (2024, arxiv 2406.00877) proved that Leela Chess Zero implements implicit look-ahead: a linear probe on Leela's hidden states predicts the optimal move **2 turns ahead** at 92% accuracy. The model doesn't just encode the current position — it encodes the CONSEQUENCES of moves.

Attention heads move information "forward and backward in time" — from squares of future moves to squares of earlier ones. The model computes the continuation internally.

## Does DeepMind's Encoder Do This?

DeepMind's 270M was trained to predict win probability for (position, move) pairs. To predict win probability accurately, the model needs to evaluate what happens AFTER the move — the opponent's best response, your follow-up, etc. This is implicit search.

If the model encodes "after Nf5, the opponent plays Qe2, then Nxe3 wins the exchange" in its hidden states, we can extract that with probes. The coaching implication is huge: instead of just "the best move is Nf5 because it's a fork," the encoder can tell the LLM "Nf5 leads to Qe2 Nxe3, winning the exchange."

## The Experiment

**Probe: predict the PV continuation from encoder hidden states.**

For each position in `lichess_evals_200k.jsonl`:
- FEN, best_move_uci, pv_line_uci (Stockfish PV)
- Encode (position + best_move) → hidden states
- Probe target: the SECOND move of the PV line (opponent's response)

If a linear probe achieves >30% accuracy on predicting the opponent's response (vs ~2% random across ~1800 possible moves), the encoder encodes look-ahead.

**Stronger test:** predict the THIRD move (your follow-up after opponent's response). If this works too, the encoder is doing deep implicit search.

## Why This Matters More Than Strategy Categories

Strategy categories tell you WHAT kind of advantage exists. Look-ahead tells you HOW the advantage unfolds — the concrete sequence of moves. For coaching:

- Strategy only: "Nf5 is a tactical shot exploiting king safety"
- Strategy + look-ahead: "Nf5 forks the king and rook. After the forced Qe2, you play Nxe3 winning the exchange. The rook on a1 then dominates the open file."

The second is dramatically better coaching. And the encoder might already encode it — we just haven't tested.

## Priority

This is a LATER experiment — the rebalanced strategy classification (item 2b in queue) comes first. But if strategy classification works AND look-ahead probes succeed, the encoder's coaching potential is much larger than we thought.

**Add to queue as item after phase 3 (text generation).**
