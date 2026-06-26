# Endgame reorg by material type + book-derived feature backlog (2026-06-26)

## Decision: organize the Endgame FIFA group by MATERIAL TYPE

The 5 concept clusters (King & Pawn Technique, Passed Pawns, Endgame Simplification, Rook
Technique, Promotion & En Passant) are being replaced by **material-type** clusters, matching how
every practical endgame book is structured (Dvoretsky *Endgame Manual*, de la Villa *100 Endgames
You Must Know*) and mirroring the Openings group (Sicilian vs Caro-Kann → Rook vs Pawn endgame).

**Why material type, not concept:**
1. It's how players think about and study endgames ("I'm bad at rook endgames").
2. **It fixes the non-monotonicity.** Material type is its OWN denominator — blunders-in-rook-
   endgames ÷ moves-in-rook-endgames — a clean population, unlike the concept clusters whose
   denominator (all endgame moves) didn't match where the concept arises. Rook Technique was the
   broken cluster precisely because of this; material-typing dissolves the problem.
3. Concept tags (opposition, rook-behind-passer, 7th rank, square rule…) become **drill-detail
   features INSIDE** each material cluster, not the top-level scoring unit.

**Material types (from `pull_endgame_material.py` `material_type()`):**
Pawn (K+P) · Rook (R+P) · Queen (Q+P) · Minor (B/N/B+N/BB) · RookMinor (R + minor) · Heavy (Q+R/pieces) · Other.
Final cluster set = whichever types come back monotonic + high-volume on the 200k corpus.

## Features ADDED this pass (easy to label, expected monotonic)

Built into `predicates.py` (verify-on-corpus gating before shipping):
- **Missed Protected Passer** — best creates a passer defended by a friendly pawn (Dvoretsky
  §"The Protected Passed Pawn"). Decisive in pawn endgames.
- **Missed Square Rule** — defending king steps into the passer's "square" to catch it; played
  steps outside and lets it queen (Dvoretsky §"The Rule of the Square"). Pure geometry.

Plus the existing endgame detectors that become material-cluster features:
opposition (`lost_opposition`), `wrong_pawn_race`, `missed_king_activity`, `missed_passed_pawn`,
`missed_connected_passers`, `outside_passer`, `rook_behind_passer`, `rook_to_seventh`,
`rook_cut_off_king`, `missed_active_rook`, `rook_to_open_file_endgame`, `push_to_promote`,
`missed_promotion`/`underpromotion`, `en_passant`, `bad_simplification`, `trade_to_simplify`,
`wrong_king_direction`.

## FUTURE feature ideas (from Dvoretsky TOC — deferred: harder to label or need engine reasoning)

### Pawn endgames
- **Key Squares** — king must reach a specific square to win K+P vs K. Needs a key-square table per
  pawn file/rank. Tractable (tablebase-like rule) but fiddly.
- **Corresponding / Mined Squares, Triangulation** — needs opposition-network reasoning; hard to
  detect from a single best-vs-played move.
- **Réti's Idea / The Floating Square** — the diagonal king march that catches a pawn AND supports
  your own. Beautiful but rare; needs two-goal geometry.
- **Shouldering / Outflanking / Zigzag / Pendulum** — king-path techniques; detectable as "best king
  move gains opposition/space while played retreats" but noisy.
- **Breakthrough** — pawn sac (e.g. b6 in the b/c/a vs a/b/c structure) that forces a passer. Crisp
  pattern, worth building: best move is a pawn push INTO contact that the opponent can't hold.
- **Reserve Tempi / Zugzwang** — needs "every move worsens the position" = engine eval of all
  replies. Engine-dependent.
- **Stalemate Refuge / Semi-Stalemate** — defensive saves; low volume.

### Minor-piece endgames
- **Wrong Bishop (rook-pawn + wrong-colored bishop = draw)** — HIGH value, very teachable, crisp:
  K+B+rook-pawn where the bishop doesn't control the promotion square. Strong future build.
- **Opposite-Colored Bishop drawing technique** — recognize OCB + extra pawn(s) = often drawn;
  detect "traded into OCB when winning" / "missed the OCB fortress." Distinct skill (de la Villa ch.8).
- **Bishop vs Knight: open position / pawns on both wings** — bishop superiority. Detect "best
  activates bishop on the long diagonal hitting both wings."
- **Knight vs Bishop: closed position / bad bishop** — knight superiority; fix the pawns on the
  bishop's color.
- **Domination / Knight Forks in the endgame** — trapping a knight or bishop.

### Rook endgames (richest section — most future value)
- **Lucena (building a bridge)** — the winning method with R+P vs R, pawn on 7th, king in front.
  Crisp, iconic, very teachable. Strong future build.
- **Philidor (third-rank defense)** — the drawing method. Pair with Lucena.
- **Rook + rook-pawn special cases** (a/h pawn drawishness).
- **Cutting the king off along a rank/file** (we have file/rank cut-off via `rook_cut_off_king` —
  could specialize "cut off N files away").
- **Rook activity vs passive rook** (have `missed_active_rook`) and **Rook behind the passer**
  (have `rook_behind_passer`).
- **Vancura position** (drawing with rook-pawn).

### Queen endgames
- **Q+P vs Q** — winning/defensive tactical tricks, perpetual-check defense. Detect "missed
  perpetual" (a drawing resource) — overlaps with a general perpetual-check detector.
- **Active queen / centralization**.

## Notes
- Build order by expected value: **Wrong Bishop, Lucena, Breakthrough, Philidor** are the highest-
  value future adds (iconic, teachable, crisp to detect).
- All future detectors gate the same way: build → measure per-band rate on the 200k corpus → keep
  only if monotonic + non-trivial volume. Concept overlap is fine; they live as drill features inside
  the relevant material cluster.

## RESULT: material-type monotonicity confirmed (partial corpus, ~330k endgame moves)

Per-material blunder rate by band (600→2800), from the partial pull (lower bands fully populated):

| Type        | 600-800 → 2600-2800 | Verdict |
|-------------|----------------------|---------|
| Pawn        | 20.8% → 8.3%         | clean monotonic decline (steepest — beginners much worse at K+P) |
| Rook        | 12.7% → 4.4%         | **clean** — the cluster that was BROKEN as "Rook Technique" is now monotonic, because material type is its own denominator |
| RookMinor   | 14.1% → 4.7%         | clean |
| Minor       | 14.2% → 4.8%         | clean |
| Heavy (Q+R) | 11.9% → 7.7%         | declines, flatter (Q+R stays sharp at all levels) |
| Queen       | 5.8% → 5.6%          | flat / low-volume (17k); queen endings rare + drawish → weak skill signal. May drop or ship flat. |

**Conclusion: the material-type reorg is correct and fixes the non-monotonicity.** 5/6 types decline
cleanly with rating; the one weak type (Queen) is low-volume and drawish. The deciding mechanism is
that material type is its OWN denominator (blunders-in-rook-endgames ÷ rook-endgame-moves), the same
property that made the Openings groups clean.

### HF download gotcha (constraint learned)
On chess-poc, `hf_hub_download` to the default `~/.cache/huggingface` (on `/home`) fails at the final
blob rename ("No such file or directory: ...incomplete") — the `/home` fs nukes in-progress blobs.
FIX: set `HF_HOME=/tmp/hfcache` (clean fs). All corpus pulls on chess-poc should use this.

## Book detectors built + measured (186k blunder corpus)

| Detector | Fires | Verdict |
|----------|-------|---------|
| Missed Square Rule | 284 | KEPT (Pawn Endgames feature) |
| Missed Protected Passer | 228 | KEPT (Pawn Endgames feature) |
| Missed Breakthrough | 148 | KEPT (Pawn Endgames feature) |
| Missed Wrong-Bishop Draw | 7 | DROPPED — exact K+B+rook-pawn material almost never occurs in real rapid |
| Missed Lucena Bridge | 4 | DROPPED — exact R+P-vs-R-king-in-front config too rare as a played blunder |

Lesson: iconic *theoretical* endgames (Lucena, Wrong Bishop) are too RARE as real-game blunders to
ship as features (4-7 fires in 186k) — they're great teaching positions but not real-game-frequency
mistakes. Concept/technique detectors that arise across many positions (square rule, protected passer,
breakthrough) clear the volume bar. Future iconic-position ideas should be served as curated DRILL
POSITIONS, not corpus-scored features.

## Drill features for the empty material clusters (measured 186k corpus)

The Queen/Minor/Rook+Minor/Heavy material clusters scored but had NO drill detail. Built + measured 3:

| Detector | Fires | Trend | Verdict |
|----------|-------|-------|---------|
| Missed Perpetual (Queen) | 1575 | 429→48 clean monotonic | KEPT → Queen Endgames feature. Beginners miss perpetual-check draws constantly. |
| Missed Rook Activity (R+Minor) | 484 | flat-ish, usable | KEPT → Rook + Minor Endgames feature |
| Missed Bishop Activity (Minor) | 85 | 2→15 RISES with rating | DROPPED — backwards (stronger players reach clean minor endings more; population artifact, no clean denom fix even at lower thresholds). |

Still featureless (scored, no drill detail — acceptable, better than misleading):
- **Minor-Piece Endgames** — bishop/knight activity is the obvious concept but the "active minor"
  signal rises with rating (you have to REACH the clean ending first). Would need a per-eligible
  denom like the rook tags, or an engine-based "your minor got dominated" detector. Deferred.
- **Heavy-Piece Endgames** (Q+R) — really late-middlegame; concepts (centralization, perpetual on the
  heavy side) overlap tactics. Deferred.

## CORRECTION: Bishop Activity is a real signal (eligible-denominator miss rate)

Earlier I dropped Missed Bishop Activity as "rises with rating." That was a DENOMINATOR ERROR — I
measured raw fires ÷ all-minor-endgame-moves, which is diluted by every non-bishop move and lets the
"beginners reach minor endgames less" population effect dominate.

The correct question (Sam): "WHEN the chance exists, do beginners miss it more?" Measured the
eligible-denominator miss rate = misses ÷ positions where a >=4-mobility bishop move is available
(pull_concept_miss_rates.py, ~1500 eligible/band):

  Bishop Activity miss%:  13.8 → 11.2 → 11.2 → 10.0 → 8.7 → 7.5 → 7.1 → 6.6 → 5.3 → 6.2 → 3.8
  Rook Activity   miss%:  11.3 → 12.3 → 11.2 → 9.3 → 9.0 → 7.3 → 7.9 → 7.1 → 5.7 → 4.3 → 3.1

Both fall ~3-4x beginner→master (one tiny top-band wobble = small-sample noise). So beginners DO
blunder these more when the opportunity is on the board → both are legitimate skill signals.

RESULT: Missed Bishop Activity ADDED BACK to Minor-Piece Endgames. Lesson: for a "missed-X" drill
feature, judge it on the ELIGIBLE denominator (positions where X is available), not raw count or the
material-type denominator — otherwise the population effect masks the real miss-rate signal.
