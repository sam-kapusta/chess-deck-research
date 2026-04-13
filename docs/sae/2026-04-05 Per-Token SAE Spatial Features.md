# Per-Token SAE: Spatial Features

**Date:** 2026-04-05
**Checkpoint:** `output/sae_pertoken_16384_k64.pt` on chess-research notebook
**Training:** 50K Lichess puzzles × 77 tokens × 2 epochs. 2183/16384 alive features.

## What This Is

SAE trained on individual encoder tokens instead of mean-pooled vectors. Each of the 77 tokens (1 side-to-move + 64 board squares + 12 metadata) is a separate training example. The SAE doesn't know which token it's looking at — spatial grounding comes from evaluation, not training.

## Key Findings

### 511 narrow features (fire on ≤5 squares per position)

Out of ~2183 alive features, 511 consistently fire on 1-5 specific squares. A quarter of alive features are spatially specific.

### Piece-specific features confirmed

| Feature | Avg width | Fires on | Interpretation |
|---------|-----------|----------|----------------|
| F4673 | 1.0 | g8(k), g1(K) — always the king | **King detector** |
| F10653 | 1.0 | f2(Q), d2(Q), g4(Q) | **Queen position** |
| F2065 | 1.0 | d2(Q), c7(r), c8(R) — major pieces | **Major piece on key square** |
| F3301 | 1.0 | Fires in 48/50 positions | **Ubiquitous single-square** |
| F2284 | 1.0 | Fires in 26/50 positions | **Common single-square** |

### Good-move vs bad-move features

Compared encoder(position, solution_move) vs encoder(position, random_legal_move) across 100 puzzles:

**Features MORE active on good moves (what good moves have):**
- F1994: good=8.18, bad=0.0 (diff=+8.18) — fires ONLY on good moves
- F2341: good=6.06, bad=0.03 (diff=+6.03)
- F5446: good=6.73, bad=1.25 (diff=+5.47)
- F8665: good=5.33, bad=0.10 (diff=+5.23)
- F12415 (mate detector): good=+6.99 on good moves

**Features MORE active on bad moves (what bad moves lack):**
- F11427: good=0.36, bad=5.12 (diff=-4.76) — the back-rank feature
- F9831: good=1.31, bad=3.65 (diff=-2.34)
- F4673 (king detector): good=0.77, bad=2.18 (diff=-1.41)

Interpretation: F11427 fires on bad moves because those moves leave the back rank vulnerable. F1994 fires only on good moves because those moves achieve something specific the bad moves don't.

### Back-rank features are real (in differential mode)

From the 3-FEN game analysis, features that Qxg8 activates MORE than f3/Kd1:
- F11427: fires on a8, b8, c8, d8, e8, f8 with activations >10 — entire 8th rank
- F5621, F9831, F1781: same pattern, all 8th rank features

These features didn't look spatial when examined in absolute mode (fired on 103/200 positions). But in differential mode (good move minus bad move), they clearly highlight the back rank.

### Global features dominate top-10 by total activation

The most active features (F12415 mate, F8471 general, F7272 queen-dominant, etc.) fire on many squares per position. They encode position-level concepts like "mating attack exists" or "queen is powerful" — same thing the mean-pooled SAE detected, just spread across tokens.

The spatial signal is in the less active, narrower features and in the differential between good and bad moves.

## Implication for Diff SAE

The per-token SAE shows the encoder HAS spatial and move-quality information in its per-token representations. But most features learn global concepts because that's what minimizes reconstruction loss.

A diff SAE — trained on (good_move_activations - bad_move_activations) — would force every feature to be about the move choice. No "this is an endgame" features because endgame-ness is the same for both moves. Only "what the good move achieves that the bad move doesn't."

## Files

- Checkpoint: `output/sae_pertoken_16384_k64.pt`
- Investigation log: `output/investigate_v2.log`
- Top-10 feature analysis: `output/investigate_pertoken.log`
- Training log: `output/pertoken_sae.log`
