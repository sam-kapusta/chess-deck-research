# FIFA skill card — mid-level CLUSTER scheme (review draft, 2026-06-22)

**The idea (Sam):** a 3rd hierarchy level between the 6 groups and 116 features. Individual
features are mostly too sparse to score per rating band; CLUSTERS of related features
("Pins", "Exchanges", "Hung Pieces") are dense enough AND more specific than the 6 groups.

**How built:** clustering workflow (chess-pedagogy lens) → deterministic density validation on
the real 11-band corpus (60k moves/band). 30 clusters, **27 scoreable** (combined ≥30 fires in
every band), all 116 features covered. `scoreable=false` = show under its group but no own per-band score.

**Decisions made (flag if you disagree):**
- Trapped-piece tags (Trapped Rook/Knight/Bishop/Queen) MERGED into Combinative Motifs (trapping is a tactic, and they're rare alone).
- Mate-variants (Arabian/Smothered/Hook/etc.) KEPT under Missed/Allowed Forced Mates — scoreable only in aggregate (variants too rare individually).
- "Created Doubled/Isolated/Backward Pawn" = Pawn Structure Weaknesses, marked scoreable=false (these are the non-drillable STATE tags).
- Clusters stay WITHIN one parent group (missed→Offensive, allowed→Defensive). A cross-group 'mechanism' view (one Pins holding missed+allowed+failed) was considered but NOT used — flag if you want it.

Combined totals/min-band are summed from the real corpus. `minbf`=min fires in any single band (the stability metric).


## Offensive Tactics  (7 clusters)

### Missed Pins  — 2568 fires, minband 62
- Missed Pin (to King)  (786 fires)
- Missed Pin  (681 fires)
- Missed Pin (to Queen)  (591 fires)
- Missed Pin (to Rook)  (510 fires)

### Missed King Attacks  — 2022 fires, minband 58
- Enemy King Exposed  (944 fires)
- Missed Kingside Attack  (673 fires)
- Missed f2/f7 Attack  (246 fires)
- Missed Queenside Attack  (159 fires)

### Missed Combinative Motifs  — 1696 fires, minband 46
- Missed Clearance  (415 fires)
- Missed Deflection  (380 fires)
- Missed Zwischenzug  (210 fires)
- Missed Trapped Rook  (163 fires)
- Missed Interference  (115 fires)
- Missed Attraction  (114 fires)
- Missed Trapped Knight  (113 fires)
- Missed Trapped Bishop  (111 fires)
- Missed Trapped Queen  (75 fires)

### Missed Forks  — 1620 fires, minband 34
- Missed Fork  (1008 fires)
- Missed Combination → Fork  (612 fires)

### Missed Sacrifices  — 1619 fires, minband 59
- Missed Sacrifice  (1619 fires)

### Missed Discovered Attacks & Skewers  — 1212 fires, minband 31
- Missed Discovered Attack  (982 fires)
- Missed Skewer  (133 fires)
- Missed Double Check  (84 fires)
- Missed X-Ray  (13 fires)

### Missed Forced Mates  — 781 fires, minband 8  ⚠️ group-only (not scoreable)
- Missed Mate  (733 fires)
- Missed Back-Rank Mate  (32 fires)
- Missed Arabian Mate  (6 fires)
- Missed Anastasia's Mate  (4 fires)
- Missed Double Bishop Mate  (2 fires)
- Missed Hook Mate  (2 fires)
- Missed Dovetail Mate  (1 fires)
- Missed Smothered Mate  (1 fires)

## Defensive Tactics  (7 clusters)

### Allowed Pins  — 3152 fires, minband 83
- Allowed Pin  (1023 fires)
- Allowed Pin (to King)  (873 fires)
- Allowed Pin (to Queen)  (647 fires)
- Allowed Pin (to Rook)  (609 fires)

### Allowed Combinative Motifs  — 2912 fires, minband 80
- Allowed Deflection  (528 fires)
- Allowed Clearance  (431 fires)
- Allowed Attraction  (430 fires)
- Allowed Zwischenzug  (398 fires)
- Allowed Capture of Defender  (305 fires)
- Allowed Trapped Rook  (237 fires)
- Allowed Trapped Knight  (166 fires)
- Allowed Trapped Bishop  (162 fires)
- Allowed Interference  (145 fires)
- Allowed Trapped Queen  (110 fires)

### Allowed Forks  — 2854 fires, minband 52
- Allowed Fork  (2015 fires)
- Allowed Combination → Fork  (839 fires)

### Allowed Mates  — 2060 fires, minband 23  ⚠️ group-only (not scoreable)
- Allowed Mate  (1904 fires)
- Allowed Back-Rank Mate  (118 fires)
- Allowed Arabian Mate  (9 fires)
- Allowed Hook Mate  (8 fires)
- Allowed Anastasia's Mate  (6 fires)
- Allowed Smothered Mate  (6 fires)
- Allowed Dovetail Mate  (4 fires)
- Allowed Double Bishop Mate  (3 fires)
- Allowed Boden's Mate  (2 fires)

### Allowed King Attacks  — 2010 fires, minband 33
- Allowed Kingside Attack  (1385 fires)
- Allowed f2/f7 Attack  (378 fires)
- Allowed Queenside Attack  (247 fires)

### Allowed Discovered Attacks & Skewers  — 1781 fires, minband 55
- Allowed Discovered Attack  (1366 fires)
- Allowed Skewer  (251 fires)
- Allowed Double Check  (146 fires)
- Allowed X-Ray  (18 fires)

### Allowed Sacrifices  — 1645 fires, minband 50
- Allowed Sacrifice  (1645 fires)

## Calculation  (2 clusters)

### Greedy Captures  — 4576 fires, minband 118
- Greedy Capture  (4576 fires)

### Backfired Tactics  — 2913 fires, minband 67
- Failed Discovered Attack  (1531 fires)
- Failed Pin  (1037 fires)
- Failed Fork  (345 fires)

## Piece Safety  (2 clusters)

### Hung Pieces  — 9330 fires, minband 116
- Hung Knight  (2148 fires)
- Allowed Hanging Piece  (1939 fires)
- Hung Bishop  (1670 fires)
- Hung Queen  (1399 fires)
- Hung Rook  (1391 fires)
- Hung Material  (783 fires)

### Missed Free Material  — 5094 fires, minband 100
- Missed Free Pawn  (1504 fires)
- Missed Free Knight  (924 fires)
- Missed Free Bishop  (836 fires)
- Missed Hanging Piece  (609 fires)
- Missed Free Rook  (523 fires)
- Missed Free Queen  (436 fires)
- Missed Capture of Defender  (202 fires)
- Missed Capture (Pawn)  (60 fires)

## Positional  (9 clusters)

### King Safety & Castling  — 8238 fires, minband 171
- Pawn Move Exposed King  (3719 fires)
- Missed Castling  (1768 fires)
- Allowed Castling  (1300 fires)
- Exposed King  (1144 fires)
- Lost Castling Rights  (307 fires)

### Pawn Advances & Tempo  — 4114 fires, minband 143
- Missed Tempo Push  (2248 fires)
- Allowed Advanced Pawn  (1133 fires)
- Missed Advanced Pawn  (733 fires)

### Exchanges  — 3789 fires, minband 72
- Missed Bishop-Knight Exchange  (1135 fires)
- Missed Pawn Trade  (907 fires)
- Missed Knight Exchange  (519 fires)
- Missed Bishop Exchange  (507 fires)
- Missed Queen Exchange  (409 fires)
- Missed Rook Exchange  (312 fires)

### Prophylaxis  — 3755 fires, minband 108
- Missed Prophylaxis  (3755 fires)

### Pawn Breaks  — 3214 fires, minband 91
- Missed Pawn Break  (3214 fires)

### Outposts  — 2443 fires, minband 77
- Missed Outpost  (1246 fires)
- Allowed Outpost  (1197 fires)

### Open Files  — 2262 fires, minband 72
- Missed Open File  (2262 fires)

### Premature Trades  — 1726 fires, minband 35
- Premature Trade  (1726 fires)

### Pawn Structure Weaknesses  — 392 fires, minband 7  ⚠️ group-only (not scoreable)
- Created Backward Pawn  (373 fires)
- Created Isolated Pawn  (19 fires)

## Endgame  (3 clusters)

### King & Pawn Technique  — 1279 fires, minband 294
- Wrong Pawn Race  (773 fires)
- Missed King Activity  (473 fires)
- Lost the Opposition  (33 fires)

### Passed Pawns  — 1132 fires, minband 224
- Missed Passed Pawn  (982 fires)
- Rook Behind Passer  (150 fires)

### Promotion & En Passant  — 931 fires, minband 140
- Allowed Promotion  (393 fires)
- Missed Promotion  (208 fires)
- Missed En Passant  (137 fires)
- Allowed En Passant  (114 fires)
- Allowed Underpromotion  (40 fires)
- Missed Underpromotion  (39 fires)