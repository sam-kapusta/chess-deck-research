# Maia 3 SAE Taxonomy v2 — Rebuild Report

**Source:** rebuild 2026-05-29 (title->categorize->chip, Sonnet 4.6 on research account)

- Features: 1996  |  Categories used: 20/20
- Generic chips: OLD 398 → NEW 0
- Largest category: Slow Play Punished (20%) — no junk drawer (<25%)

## Method

Chip-FIRST pipeline (old) collapsed distinct features into junk-drawer categories because categories were assigned from lossy 2-4 word chips. Rebuilt TITLE→CATEGORIZE→CHIP: the accurate per-feature `description` (verified against the board) is the source of truth; each feature assigned to one of 20 checkpoint-stable coaching categories (reused from chess_blunder_taxonomy_v2), then a specific chip generated last, category-aware, with the generic frame banned.

## Distribution

| Category | Features | % |
|----------|---------:|--:|
| Slow Play Punished | 408 | 20% |
| Pawn Moves Ignore Threats | 309 | 15% |
| Piece Abandons Defense | 283 | 14% |
| Piece Lands Badly | 196 | 10% |
| King Walks Into Danger | 148 | 7% |
| Greedy Captures | 142 | 7% |
| Rook Misplacement | 94 | 5% |
| Captures Backfire | 87 | 4% |
| Pawn Moves Weaken King | 76 | 4% |
| Checks Lose Tempo | 63 | 3% |
| King Safety Ignored | 34 | 2% |
| Passed Pawn Blindness | 33 | 2% |
| Retreating Errors | 29 | 1% |
| Ignoring Threats | 28 | 1% |
| Unsound Sacrifices | 25 | 1% |
| Piece Trapping | 20 | 1% |
| Fork Vulnerability | 7 | 0% |
| Moving Hangs Pieces | 6 | 0% |
| Back Rank Weakness | 4 | 0% |
| Pieces Left Undefended | 4 | 0% |

## Example relabels (old → new)

- **[0]** `Slow flank pawn ignoring tactics` → `a/b pawn push gifts tempo` _(Pawn Moves Ignore Threats)_
- **[10]** `Piece placed en prise` → `Bishop steps onto attacked square` _(Piece Lands Badly)_
- **[1045]** `Seductive Knight Centralization Blunders` → `Knight jumps to e4/e5 undefended` _(Piece Lands Badly)_
- **[556]** `Idle Pawn Push Ignoring Crisis` → `Pawn push over urgent piece action` _(Pawn Moves Ignore Threats)_
- **[12]** `Passive bishop retreat wastes tempo` → `Bishop retreats when concrete action needed` _(Slow Play Punished)_
- **[64]** `Queen repositions, misses tactics` → `Queen repost ignores concrete demand` _(Slow Play Punished)_

## Coherence validation

Every category's structural signature matches its definition: greedy_captures 88% capture, captures_backfire 89% capture, checks_lose_tempo 73% check, unsound_sacrifices 83%cap+42%chk, king_walks_into_danger 98% king, slow_play_punished 88% quiet, pawn categories 99-100% pawn, rook_misplacement 100% rook.

## Known limits

- 4 low-population categories (moving_hangs_pieces 6, pieces_left_undefended 4, back_rank_weakness 4, fork_vulnerability 7): features route to the sharper `piece_abandons_defense`/`piece_lands_badly` instead. Kept as checkpoint-stable vocab; not mis-routing.
- ~33/1996 (2%) chips name no specific square/piece — these are genuinely quiet 'slow play' features where the move character IS nondescript; forcing specificity would fabricate it.
