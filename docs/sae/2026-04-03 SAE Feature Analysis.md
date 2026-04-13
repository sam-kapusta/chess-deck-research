# SAE Feature Analysis — cabbagelover5566

**Date:** 2026-04-03
**Data:** 502 games, 3,012 moments, 32 SAE features per moment (~96K observations)

---

## Key Findings

### 1. SAE features describe positions, not mistakes

The top features across ALL moments are generic position descriptors: "Bishop pair" (143%), "Knight outpost" (94%), "Rook endgame" (87%), "Passed pawns" (86%). These fire on nearly every position. They tell you WHAT the board looks like, not WHAT you did wrong.

**Problem for v3 tags:** If "Knight outpost" fires on 94% of moments, it's useless as a tag. A tag needs to differentiate — "you keep making mistakes in X type of position" requires X to be specific.

### 2. Blunder vs inaccuracy: features barely differentiate

Blunders and inaccuracies have nearly identical top-5 feature profiles. "Knight outpost" is #1 for both (44% vs 34%). The features don't know you blundered — they just describe the board.

**One signal that works:** "Precise calculation" appears 9.4% of blunders but not in the top 10 for inaccuracies. The SAE is detecting positions that REQUIRE calculation — and you're blundering in them. That's actionable: "you make your worst mistakes in positions that require precise calculation."

### 3. Win/loss differentiation is real but subtle

Features over-represented in losses:
- **Promotion race** (9.67x) — you lose promotion races
- **Undefended pieces** (3.98x) — you leave pieces undefended in losses
- **Queen-rook coordination** (2.5x) — you mishandle Q+R endgames

Features over-represented in wins:
- **Full middlegame** (2.32x) — you play better in complex middlegames
- **Tactical threats** (1.66x) — positions with active threats favor you
- **Isolated pawn** (1.63x) — you handle IQP positions well

**This is coaching gold.** "You win when the position is complicated. You lose when it simplifies to an endgame with promotion chances." That's a player profile.

### 4. Worst mistake positions

Positions where you make the worst mistakes (lowest win% after):
- **Unsupported center** (avg 41.8% win, 131 observations) — you consistently mishandle central pawn tension
- **Overextended center** (avg 44.5%, 53 obs)
- **Full middlegame** (avg 44.3%, 98 obs) — complex positions where you SHOULD do well but sometimes catastrophically fail

Positions where mistakes are mild:
- **King pawn technique** (avg 67.9%) — even your mistakes here aren't bad
- **Multiple threats** (avg 61.1%) — you play well enough that mistakes don't cost much

### 5. Hand-coded tags vs SAE: different signals

The tag `undeveloped_pieces` co-occurs with SAE features "Underdeveloped pieces" and "Bishop development" — they detect the same thing. But `missed_capture` co-occurs with "Knight outpost" and "Rook activity" — the SAE describes WHERE you missed it, the tag says WHAT you missed.

**Verdict:** SAE features are not a replacement for hand-coded tags. They're complementary. Tags = error type. SAE = position context. Together: "You missed a capture (tag) in a position with an active knight outpost (SAE) — this type of position requires looking at ALL pieces, not just the obvious ones."

### 6. Deep dive: SAE tells a game narrative

**Game chesscom:146123780236** (Caro-Kann, 13 mistakes, lost):
- Moves 9-14: "Underdeveloped position" fires on every mistake — the SAE is screaming "you haven't developed!"
- Moves 25-36: Features shift to "Central tension" + "Pawn structure" — the game became about pawns, and you kept making non-pawn moves
- Moves 38-53: "Pawn endgame" dominates — you reached an endgame you don't know how to play
- Move 59: "King centralization" (strength 13.4!) — the SAE's strongest signal all game, and you traded pieces instead of centralizing your king
- Move 70: "Zugzwang technique" — an endgame concept you evidently haven't studied

**The narrative:** Development problems → pawn structure mishandling → lost endgame you didn't understand. That's a coaching story, built without an LLM.

### 7. "Knight outpost" problem

"Knight outpost" fires on 94% of moments and dominates every analysis. It's not that YOUR games are about knight outposts — it's that the Maia SAE feature for "knight outpost" is over-broad. It probably fires on any position with a knight. This is a labeling issue, not a data issue. The feature might actually be detecting "knight activity" or "piece development" but was labeled as "knight outpost."

This means: **the top 5 features per moment are polluted by over-frequent features.** The signal is in features 5-32, where more specific concepts live.

---

## Verdict: Can SAE replace hand-coded tags?

**No, not directly.** But they serve a different purpose that's equally valuable:

| | Hand-coded tags | SAE features |
|---|---|---|
| **Answers** | What you did wrong | What the position demanded |
| **Specificity** | High (missed_fork is precise) | Medium (top features too generic, tail features specific) |
| **Actionable** | Directly ("drill forks") | Indirectly ("study these position types") |
| **Scalable** | No (manual detection rules) | Yes (learned, zero-cost) |
| **Coverage** | 48 concepts | 695 concepts |

**Best architecture:** Tags for error detection (what), SAE for position context (where/why), LLM for synthesis (explain it).

## Next steps
1. Filter out over-frequent features (fire rate >50%) before aggregation
2. Look at features 5-32 for more specific signals
3. Test: can SAE features alone predict which tag should fire? (classification task)
4. Build the hybrid: tag + top 3 non-generic SAE features per moment
