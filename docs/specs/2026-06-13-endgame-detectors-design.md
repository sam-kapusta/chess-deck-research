# Endgame Mistake Detectors — Design (2026-06-13)

Builds on `2026-06-13-tagger-coverage-gaps-findings.md` (the naked-mistake analysis). That doc measured
~1,500 real mistakes (cp ≥ 150) get zero explanatory tag, concentrated in quiet positional/endgame moves.
This spec adds four endgame detectors to start closing that gap.

## What we're building

Four new predicate functions in `scripts/04_tagger/predicates.py`, registered in `ALL_PREDICATES`, each
emitting one new "Endgame"-category tag. Same shape as the existing `endgame_type` / `pawn_structure`
predicates: pure `(Mistake) → [(label, direction, evidence)]`. Shipped to the Lambda + ECS worker via
`backend/scripts/ship_tagger.py` (verbatim copy — research is the source of truth). No frontend change:
the tags render automatically as Review chips through `buildMistakeChips` (category "Endgame" → endgame
chip color).

| Tag | Direction | Corpus coverage of naked mistakes |
|---|---|---|
| **Missed King Activity** | missed | 72 (5% of all naked; ~60% of endgame king-best naked) |
| **Lost the Opposition** | missed | ~7 (rare, but the most teachable endgame concept) |
| **Missed Passed Pawn** | missed | 34 (2%) |
| **Rook Behind Passer** | missed | 19 (1%) |

Together ~130 naked mistakes gain a real teaching tag.

## The firing rule — NO causal gate

Decided with Sam: **fire when the theme is present** — the *best move exhibits the theme AND the played
move did not*. Two conditions. No third "is this actually the mistake / is the moment otherwise naked"
gate. If the result is noisy (a tag fires on a moment that's really about a hung piece), we prune by
reading real outputs later — not by gating up front.

Rationale (Sam's framing): these are **drill-filter categories** — "would someone want to drill a
collection filtered by this tag?" Each of the four answers yes. The point is to mark positions of a
*type*, so firing whenever the type is present is correct; over-precision would shrink the drill set.

> **Scope note — drill wiring does NOT exist yet.** Confirmed via code exploration: the Drill / Practice /
> Library filters read **SAE features → SAE subcategories**, never `mistake_tags`. `mistake_tags` render
> only as Review chips. So these tags ship as **Review chips now**; the "drill my bishop endgames" payoff
> needs a separate tag→drill bridge (a later project, out of scope here). Building the detectors is the
> prerequisite for that bridge regardless.

## Endgame phase gate

Reuse the existing `phase(m)` predicate's "Endgame" determination (`npieces <= 12 or non_pawn <= 4`,
`predicates.py:54-64`). Do NOT duplicate the threshold — call a shared helper.

- **King Activity, Opposition, Rook-Behind-Passer:** Endgame phase only.
- **Passed Pawn:** may fire in middlegame too (passed pawns matter before the endgame). No phase gate —
  gated by the passed-pawn geometry itself.

## Per-detector specification

All four resolve the best move via the existing `_best_move(m)` and played via `_played_move(m)`
(`predicates.py:23,42`). All use `chess.square_distance` (Chebyshev) from `chesslib_util`.

### 1. `missed_king_activity(m)` → "Missed King Activity"

Fires when: Endgame phase; best move is a **king move**; best move is **not** a response to check
(`not board_before.is_check()` — escaping check is defense, not activity); the king moves **toward the
center OR toward the nearest enemy pawn**; and the played move is not that same king move.

- "Toward center" = `center_dist(to) < center_dist(from)`, where `center_dist(sq)` = min Chebyshev
  distance to {d4,e4,d5,e5}.
- "Toward enemy pawns" = `nearest_enemy_pawn_dist(to) < nearest_enemy_pawn_dist(from)`.
- Evidence: `f"best {m.best_san} activates the king (toward {center|pawns})"`.

Real corpus fixtures (positives):
- `8/8/6p1/5pkp/8/4N3/4K3/8 w - - 0 51` — played Ng2, best **Kf3** (cp 211)
- `8/8/3k2pp/p2P1p2/2K2P1P/1p3P2/1P6/8 w - - 0 35` — played Kb5, best **Kd4** (cp 247)

### 2. `lost_opposition(m)` → "Lost the Opposition"

Fires when: **pawn-only endgame** (every non-king piece is a pawn); best move is a king move that lands
**in direct opposition** to the enemy king — exactly 2 squares away (`square_distance == 2`) on the same
file OR same rank; and the played move did not take that opposition.

- Direct opposition only in v1 (same file/rank, distance 2). Distant/diagonal opposition deferred —
  rarer and fuzzier.
- Evidence: `f"best {m.best_san} takes the opposition"`.

Real corpus fixtures (positives):
- `8/8/8/3k1p1p/3P2pP/4K1P1/8/8 w - - 2 53` — played Kf4, best **Kd3** (cp 487)
- `8/8/3k2pp/p2P1p2/2K2P1P/1p3P2/1P6/8 w - - 0 35` — played Kb5, best **Kd4** (cp 247)

Note: King Activity AND Opposition can BOTH fire on the same position (Kd4 above is both). That's
accepted — no dedup between them; they're different lenses and both are valid drill filters.

### 3. `missed_passed_pawn(m)` → "Missed Passed Pawn"

Fires when: best move is a **pawn move** that results in a **passed pawn** on the destination square
(no enemy pawn on the destination file or adjacent files, on any rank ahead of it); and the played move
is not that pawn move. No phase gate.

- `is_passed(board, sq, color)`: scan files {f-1, f, f+1}, all ranks ahead in the color's direction; if
  any enemy pawn → not passed.
- Covers both *creating* a new passer and *advancing* an existing one (both are "push the passer").
- Evidence: `f"best {m.best_san} makes/advances a passed pawn"`.

Real corpus fixtures (positives):
- `8/8/4p3/5k1K/p1pP1P2/P6P/1P6/8 w - - 1 38` — played Kh4, best **h4** (cp 414)
- `r4rk1/p1p2ppp/2q1p3/3pP3/5P2/2B3P1/PPQ4P/R3R1K1 b - - 0 19` — played Rab8, best **d4** (cp 396, middlegame — intentional, no phase gate)

### 4. `rook_behind_passer(m)` → "Rook Behind Passer"

Fires when: Endgame phase; best move is a **rook move** landing on a file that contains a **passed pawn**
(either color), with the rook **behind** that pawn (Tarrasch: behind your own passer to push it, behind
the enemy's to stop it); and the played move did not do this.

- "Behind" relative to the pawn's promotion direction: white passer → rook on a lower rank; black passer
  → higher rank.
- Evidence: `f"best {m.best_san} puts the rook behind the passed pawn"`.

Real corpus fixtures (positives):
- `7R/8/p1r4p/3pp2P/8/4KP2/k7/8 w - - 0 44` — played Rg8, best **Rd8** (cp 227)
- `5R2/4k1K1/4P3/p4P2/1r6/8/2p5/8 w - - 4 54` — played Rf7+, best **Rc8** (cp 472)

## Shared helpers (add to `chesslib_util.py`)

Write once, reuse across detectors (DRY — none of these exist yet; verified `chesslib_util` has only
`square_distance`, `is_king_move`):
- `center_distance(sq) -> int` — min Chebyshev to {d4,e4,d5,e5}.
- `nearest_enemy_pawn_distance(board, sq, pov) -> int` — min Chebyshev to any `not pov` pawn (99 if none).
- `is_passed_pawn(board, sq, color) -> bool` — the file-scan above.
- `is_pawn_only_endgame(board) -> bool` — every non-king piece is a pawn.

## Taxonomy

Add the 4 labels to the taxonomy generator `scripts/04_tagger/build_mistake_taxonomy.py` (category
"Endgame"), regenerate `output/mistakeTaxonomy.json`, and re-ship (ship_tagger copies the taxonomy to
`frontend/src/data/mistakeTaxonomy.json`).

**`categorize()` MUST change — verified bug, not optional.** `categorize()` (`tagger.py:190`) is
first-match, and the existing King Safety branch (`"king"`, line 206) and Positional branch (`"pawn"`,
line 210) fire BEFORE the Endgame branch (line 208). Tested: today these labels misroute —
"Missed King Activity" → King Safety, "Missed Passed Pawn" → Positional, "Rook Behind Passer" → Other.
Fix: add a dedicated endgame-label branch with **distinctive phrases** (NOT bare "king"/"pawn") placed
**immediately before** the Tactical branch (line 201), so it wins:

```python
    # Endgame mistake tags — distinctive phrases, checked early so the "king"/"pawn" substring
    # branches below don't mis-claim them (categorize is first-match).
    if any(w in l for w in ["king activity", "opposition", "passed pawn", "passer", "behind passer"]):
        return "Endgame"
```

"Lost the Opposition" already routes correctly (line 208 "opposition"), but this branch makes all four
explicit and order-independent. After the change, re-verify all four → "Endgame".

## Testing (TDD — failing test first, then implement)

Each detector follows the skewer/exchange precedent:
1. **Regression positives** in `regression.py`: the real corpus FENs above. Construct a `Mistake` with the
   fen, played_uci, best_uci, best_san, and assert the detector returns the expected label.
2. **Regression negatives:** a position where the theme is absent (e.g. a king move *away* from center →
   King Activity silent; a non-pawn-endgame → Opposition silent). Assert `[]`.
3. **Corpus before/after firing count** (not a unit test — a sizing script, like `/tmp/endgame_candidates.py`
   this session): report total fires + a sample, so we SEE the noise level before trusting it. Record the
   numbers in the findings doc.
4. Update `regression.py`'s `total` count for the new cases (the `extra_*` pattern).

## Out of scope
- The tag→drill filter bridge (separate project; drill currently filters on SAE subcategories only).
- Distant/diagonal opposition, triangulation, zugzwang (engine-required), pawn breakthrough.
- Any causal/"is it the real mistake" gate — explicitly rejected; prune by reading outputs instead.
- Display ranking / show-more (the separate tag-display brainstorm, still parked).

## Verification before claiming done
- `python3 scripts/04_tagger/regression.py` → all pass including new cases.
- Corpus firing-count script run; numbers recorded; eyeball ~10 sample fires per detector for obvious
  false positives (the real test — per the metric lesson, coaching value is eyeball-only).
- `ship_tagger.py` run; confirm Lambda + worker copies updated and taxonomy JSON synced.
- Spot-check one detector end-to-end through `tag_adapter.tags_for_eval` on a fixture FEN.
