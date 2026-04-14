# Chess Coaching Taxonomy — Book Comparison (2026-04-14)

Comparing our SAE-derived categories against 5 classical chess instruction books.

## Sources

| Book | Author | Year | Focus |
|------|--------|------|-------|
| My System | Nimzowitsch | 1925 | Positional elements (9) + positional play (6) |
| Chess Strategy | Lasker | 1911 | Opening, middle game, end game strategy |
| How to Reassess Your Chess | Silman | 2010 | 7 imbalances for position evaluation |
| Complete Strategy A-to-Z | Silman | 1998 | Alphabetized chess concept glossary |
| Simple Chess | Stean | 1978 | Outposts, open files, weak pawns |

Plus: Lichess puzzle themes (60 themes, industry standard for tactical categorization)

## Our SAE Taxonomy (16 categories)

Derived from 34 experiments on blunder SAE (2048 k=32, validated; 512 k=8 pending).

| # | Category | Type | Coverage |
|---|----------|------|----------|
| 1 | Hanging Pieces | Tactical | 100% |
| 2 | Overloaded Defenders | Tactical | 99.8% |
| 3 | King Safety | Tactical/Positional | 99.5% |
| 4 | Discovered Attacks | Tactical | 99.2% |
| 5 | Pawn Endgames | Endgame | 97.9% |
| 6 | Passed Pawns | Endgame/Pawn | 96.8% |
| 7 | Back Rank | Tactical | 94.9% |
| 8 | Rook Endgames | Endgame | 93.7% |
| 9 | Pins | Tactical | 92.0% |
| 10 | Forks | Tactical | 91.8% |
| 11 | Skewers | Tactical | 33.6% |
| 12 | Quiet Moves | Tactical | 29.3% |
| 13 | Checkmate Patterns | Tactical | 25.4% |
| 14 | Trapped Pieces | Tactical | 17.1% |
| 15 | Sacrifice | Tactical | 10.5% |
| 16 | Other | Mixed | varies |

## Silman's 7 Imbalances (the gold standard)

1. **Superior Minor Piece** — bishops vs knights
2. **Pawn Structure** — weak pawns, passed pawns
3. **Space** — territory control
4. **Material** — piece value
5. **Control of Key File** — rook activity
6. **Control of Weak Square** — outposts
7. **Lead in Development** — tempo, development lead
8. **Initiative** — who's calling the shots
9. **King Safety** — king vulnerability

## Nimzowitsch's 9 Elements

1. Center and Development
2. Open Files
3. 7th and 8th Ranks
4. The Passed Pawn
5. On Exchanging
6. Endgame Strategy
7. The Pin
8. Discovered Check
9. The Pawn Chain

## Coverage Matrix

| Concept | Our SAE | Silman | Nimzowitsch | Lasker | Lichess |
|---------|---------|--------|------------|--------|---------|
| **Hanging/Undefended Pieces** | ✓ Hanging Pieces (678) | Material | — | Objects of Attack | ✓ Hanging Piece |
| **Overloaded Defenders** | ✓ (471) | — | — | Deflection | ✓ Deflection |
| **Forks** | ✓ (157) | — | — | Elementary Combos | ✓ Fork |
| **Pins** | ✓ (160) | — | ✓ Ch.7 | Pins & Discovered | ✓ Pin |
| **Skewers** | ✓ small (19) | — | — | — | ✓ Skewer |
| **Discovered Attacks** | ✓ (283) | — | ✓ Ch.8 | — | ✓ Discovered Attack |
| **Back Rank** | ✓ (182) | King Safety | ✓ Ch.3 (7th/8th) | King Safety | ✓ Back Rank Mate |
| **King Safety** | ✓ Exposed King (345) | ✓ #9 | — | Breaking King's Side | ✓ Exposed King |
| **Passed Pawns** | ✓ (423) | ✓ Pawn Structure | ✓ Ch.4 | End-Game | ✓ Advanced Pawn |
| **Rook Endgames** | ✓ (302) | Control of Key File | ✓ Ch.2 Open Files | End-Game | ✓ Rook Endgame |
| **Pawn Endgames** | ✓ (533) | — | ✓ Ch.6 | End-Game | ✓ Pawn Endgame |
| **Trapped Pieces** | ✓ small (14) | — | — | — | ✓ Trapped Piece |
| **Quiet Moves** | ✓ small (16) | — | — | — | ✓ Quiet Move |
| **Sacrifice** | ✓ small (6) | — | — | — | ✓ Sacrifice |
| **Checkmate Patterns** | ✓ puzzle only | — | — | — | ✓ 19 named mates |
| **Pawn Structure** | ✗ MISSING | ✓ #2 | ✓ Ch.9,11,12 | ✓ Pawn Skeleton | ✗ |
| **Space** | ✗ MISSING | ✓ #3 | ✓ Ch.10 | — | ✗ |
| **Minor Piece Quality** | ✗ MISSING | ✓ #1 | ✓ Ch.13 | — | ✗ |
| **Exchanging** | ✗ MISSING | ✓ Material | ✓ Ch.5 | ✓ 138 paragraphs | ✗ |
| **Initiative** | ✗ MISSING | ✓ #8 | — | Balance of Attack | ✗ |
| **Outposts/Weak Squares** | ✗ MISSING | ✓ #6 | — | Objects of Attack | ✗ |

## Key Insight

**Our SAE captures tactical errors. The books teach positional understanding.**

The SAE detects WHAT WENT WRONG in a specific move. The books teach HOW TO EVALUATE a position to avoid mistakes. These are complementary:

- "You left a piece hanging" (our SAE) vs "Evaluate material imbalances" (Silman)
- "You missed a discovered attack" (our SAE) vs "Control open files" (Nimzowitsch)
- "Back rank was weak" (our SAE) vs "King safety is an imbalance" (Silman)

For a complete coaching product, we need BOTH:
1. SAE features for "what went wrong per move" (tactical profiling)
2. Positional imbalance framework for "what should you have been thinking about" (strategic coaching)

Our tags system (14 behavioral + 13 tactical) partially covers the positional side.

## Gaps to Address

1. **Pawn Structure** — all 3 major books devote 2-3 chapters to this. Our SAE has features for it but they're buried in other categories. Need to check.
2. **Exchanging/Simplification** — Nimzowitsch Ch.5, Lasker 138 paragraphs. "When to trade" is a major skill at 1800. Not in our SAE or tags.
3. **Space/Territory** — Silman's #3 imbalance. Positional concept the SAE probably can't detect from single moves.
4. **Initiative** — "Who's calling the shots" is Silman's #8. Closest to our "Forcing Moves" but more strategic.
