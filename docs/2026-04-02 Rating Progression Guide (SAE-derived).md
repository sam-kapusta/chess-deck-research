# Rating Progression Guide (SAE-derived)
*2026-04-02 — from Maia SAE 2048/k=32 rating gradient analysis*

## How this was made

Ran the same 5,000 positions through Maia at ratings 1100, 1400, 1700, and 1900. SAE features that change activation rate with rating reveal what higher-rated players "see" that lower-rated don't.

## 1400 → 1900: What to learn

**Pay MORE attention to (higher-rated sees these more):**

| Concept | Signal strength | Why it matters |
|---------|----------------|----------------|
| Passed pawns | +++ | Higher-rated spots passed pawns earlier and plans around them |
| Rook activity | +++ | Recognizes rook scope (presence, open files, 7th rank) |
| King safety (pawn shield) | +++ | Evaluates king safety by pawn structure, not just "is it castled" |
| Exposed king | ++ | Spots when king position creates real vulnerability |
| Castling patterns | ++ | Understands implications of castle direction (not just "did they castle") |
| Back rank awareness | + | Recognizes back rank weakness as ongoing theme |
| Endgame patterns | + | More attuned to endgame positions and their unique demands |
| Early middlegame plans | + | Better sense of what the middlegame requires |

**Pay LESS attention to (lower-rated over-focuses on these):**

| Concept | Signal strength | What's happening |
|---------|----------------|-----------------|
| Center pawn tension | --- | Fixated on d4/e4 pawn battles, missing bigger picture |
| Fianchetto patterns | -- | Distracted by bishop structure details |
| Queen presence | -- | Over-weights "queens are on the board" (queen ≠ everything) |
| Rook on open file | -- | Notices the obvious (rook + open file) but misses nuance |
| Locked pawn structure | -- | Focuses on pawn immobility rather than piece play |
| Raw piece count | - | Counting pieces rather than evaluating their activity |
| Position flexibility | - | Sensitive to number of options rather than quality of options |

## The story

**1100 perception:** "Am I in danger? How many pieces do I have? Are my pawns stuck?"
- Fixated on material counting, center pawns, obvious structures (fianchetto, open file)
- Over-sensitive to how many legal moves exist (flexibility ≠ quality)

**1900 perception:** "Is my king safe? Where is the real weakness? What's my pawn structure creating?"
- Evaluates king safety through pawn shield quality, not just castling
- Spots passed pawns as long-term plans
- Understands rook scope (7th rank, activity) beyond just "open file"
- Recognizes endgame transitions early

## For cabbagelover5566 (~1800)

The 1800→2000 push means:
1. **Passed pawns** — you probably see them but don't plan around them enough
2. **King safety depth** — go beyond "is it castled?" to "is the pawn shield intact?"
3. **Rook scope** — not just "put rook on open file" but understanding rook activity holistically
4. **Stop over-weighting center pawns** — the d4/e4 battle matters less than you think past the opening

## Methodology

- Maia-2 (human move prediction model) at 4 rating levels
- 2048-feature SAE with k=32 (BatchTopK, 200K training positions)
- 54 chess concepts computed from FEN (game phase, material, king safety, piece activity, pawn structure)
- Point-biserial correlation between feature activations and concept presence
- Rating gradient: Spearman correlation of fire rate with [1100, 1400, 1700, 1900]
