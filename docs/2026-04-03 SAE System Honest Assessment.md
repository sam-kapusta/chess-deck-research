# SAE System — Honest Assessment

**Date:** 2026-04-03

## What's working

- **PF-ICF × enrichment** produces real, specific player weakness profiles. "Unclear coordination" at 47.6x across 122 games is not noise.
- **Per-game narratives** from Stockfish + SAE tell coherent stories: phase trajectory, recurring themes, critical moments.
- **Clustering** reveals the SAE's true vocabulary: ~12 concept groups, not 689 random labels.
- **Endgame strength / middlegame coordination weakness** profile matches what a human coach would say about an 1800 player.

## What's not working

- **SAE features describe positions, not mistakes.** "Unclear coordination" fires on the position, not on the error. The same feature fires whether you blunder or play brilliantly in that position — the enrichment ratio just tells you it happens to co-occur with YOUR mistakes.
- **Tactical detection is weak.** The SAE can't tell you "missed fork" — it says "position with multiple piece interactions." The hand-coded tags are better for this.
- **Labels are noisy.** "Piece sacrifice" doesn't detect sacrifices. "Unclear coordination" is a catch-all. 40+ of the top 107 enriched features describe the same underlying thing (underdeveloped opening).
- **Single-rating Maia limitation.** We train on 1800 Maia. This tells us what an 1800 sees, not what an 1800 misses. The delta between rating levels is more useful for coaching but we're not using it.

## What the SAE is actually good for

1. **Position type labeling** — "this is a rook endgame with passed pawns" (replaces manual game phase detection)
2. **Cross-game pattern aggregation** — "you make mistakes in development positions" (PF-ICF across 502 games)
3. **LLM grounding** — pre-filtered tags prevent hallucination about what the position is about
4. **Drill queue construction** — select positions from weakness clusters for targeted practice

## What the SAE cannot do

1. **Name the specific mistake** — that's Stockfish PV analysis + hand-coded tags
2. **Explain why you made the mistake** — that requires intent modeling (LLM)
3. **Tell you what a better player would see** — that requires multi-rating Maia comparison
4. **Detect tactical motifs** — forks, pins, skewers are structural, not representational

## The right architecture

```
Stockfish ──→ WHAT went wrong (eval drop, best move, classification)
Hand-coded tags ──→ NAME of the mistake pattern (missed_fork, premature_push)  
SAE features ──→ POSITION CONTEXT (what kind of position, what matters here)
Maia findability ──→ DIFFICULTY (should you have found this at your rating?)
PF-ICF profile ──→ PERSONALIZATION (is this your known weakness?)
LLM ──→ SYNTHESIS (explain all of the above in human language)
```

No single layer replaces another. Each answers a different question.

## What matters most for player improvement

**The drill system, not the coaching text.** A player improves by practicing positions, not reading analysis. The SAE's highest-leverage use is building a smart drill queue sorted by weakness cluster × Maia findability. Everything else (coaching narrative, player profile, per-game analysis) is secondary to "did the player practice the right positions?"

## Open questions

- Would a 2200 Maia SAE produce better coaching features for 1800 players? (detecting what they should see but don't)
- Can we use the rating gradient (1100→1900 feature activations) to identify "features that matter more as you improve"?
- Is 2048 features the right SAE size, or would 1024 with cleaner clusters work better?
- Should we re-run Haiku labeling with cluster context? ("You are labeling a feature in the Rook Endgame cluster. Given these 5 FENs...")
