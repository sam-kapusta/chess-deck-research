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

## Re-examined ALL enrichment-dropped features — 2 more were real signals

After Bishop Activity, checked every feature dropped on the ENRICHMENT metric (fires-on-blunders vs
good-moves) using the correct eligible-denominator miss-rate test (pull_concept_miss_rates.py):

| Feature | Old verdict | Eligible miss-rate by band (600→2800) | Real? |
|---------|-------------|----------------------------------------|-------|
| Missed Doubled Rooks | dropped (0.8x enrich) | 19.9% → 3.4% (miss/eligible) | YES — ~6x fall. Re-added → Missed Combinative Motifs (Offensive). |
| Pawn Grab While Undeveloped | dropped (0.4x enrich) | 2.6% → 0.1% (took_bad/eligible) | YES — ~26x fall. Re-added → Greedy Captures (Calculation). |
| Bishop Activity | dropped (non-mono raw) | 13.8% → 3.8% | YES (already re-added) |
| Rook Activity (R+Minor) | (kept) | 11.3% → 3.1% | confirms |

Also fixed: Missed Doubled Rooks had been mis-swept into the Rook ENDGAMES cluster during the material
reorg, but it categorizes as Missed Tactic (Offensive) and fires across all phases — moved to Missed
Combinative Motifs.

### THE LESSON (proven 3×)
The **enrichment metric is the wrong test for a drill feature.** "Does this pattern fire more on
blunders than good moves" conflates pattern-frequency with skill-differentiation. A pattern that's the
best move equally often in good play and blunder positions (enrichment ~1) can STILL be something
beginners miss far more — that's the skill signal. The correct test is the **eligible-denominator
band miss-rate**: among positions where the concept is available, does the weaker player err more?
Enrichment nearly cost Bishop Activity and DID cost Doubled Rooks + Pawn Grab. Going forward, judge
every missed-X / played-bad-X feature on eligible miss-rate by band, not enrichment.

The not-rescuable drops stay dropped: Wrong-Bishop (7 fires) and Lucena (4 fires) — rarity can't be
denominator-fixed; serve those as curated drill positions instead.

## Round 2: thin clusters WERE under-built (eligible-miss-rate test)

Sam asked "are the low clusters just under-built, not data-limited?" Measured 6 untested candidates
(pull_concept_miss_rates2.py, eligible miss-rate by band). Answer: YES, under-built — 5/6 are real:

| Candidate | miss% 600→2800 | Verdict |
|-----------|----------------|---------|
| Knight Activity (Minor) | 11.5 → 4.3 | real (~2.7x) |
| King Activity (Minor)   | 12.9 → 4.7 | real |
| King Activity (Rook)    | 11.0 → 3.6 | real (~3x) |
| Minor Activity (R+Minor)| 11.7 → 2.4 | real (~5x) |
| Queen Activity (Queen+Heavy) | 10.7 → 5.7 | real (~2x), weaker |
| King Activity (Queen)   | 5.4 → 2.2 bumpy | weak/noisy — queen endgames low signal |

("falls=False" flags are all the single thin-2600-band wobble; curves are clearly declining.)

KEY INSIGHT: **King Activity is a UNIVERSAL endgame skill** (Shereshevsky's #1 principle) — belongs in
every material cluster, not just Pawn. To build:
- Knight Activity → Minor-Piece (pairs w/ Bishop Activity)
- Minor Activity → Rook+Minor (pairs w/ Rook Activity)
- Queen Activity → Queen + Heavy (Sam: queen mistakes in any queen endgame, not just Q+P)
- King Activity → add to Rook / Minor / RookMinor / Heavy clusters (already in Pawn via missed_king_activity)

## SHIPPED (2026-06-27): activity detectors + universal King Activity

3 new pure detectors in predicates.py via `_activates_piece(m, ptype, gain=4)` (best move moves a
piece of `ptype`, gains ≥4 attack squares, played differs):
- `missed_knight_activity`  → "Missed Knight Activity"  → Minor-Piece cluster
- `missed_minor_activity`   → "Missed Minor Activity"   → Rook+Minor cluster (bishop OR knight)
- `missed_queen_activity`   → "Missed Queen Activity"   → Queen + Heavy clusters

King Activity (`missed_king_activity`, already in Pawn) added to Rook/Minor/RookMinor/Heavy feature
lists — universal endgame skill. categorize(): generalized to route any "activity" label → Endgame.
Removed mis-placed "Missed Doubled Rooks" from Rook cluster (it's Offensive Tactics).
All 6 material clusters re-aggregated, monotonic. Local volume (200k): Knight 98, Minor 269, Queen 871.
Research 9aa5d2b, code cf6c5445.

## Cross-group thin-cluster audit (2026-06-27) — the lesson generalizes

Applied the round-2 "thin = under-built, not data-limited" check to ALL 6 scored groups. Single-feature
clusters and their per-band spread (beginner_rate − master_rate, isotonic-smoothed, shipped anchors):

| Group | Thin cluster (1 feat) | mono? | spread /k | verdict |
|-------|----------------------|-------|-----------|---------|
| Positional | Prophylaxis  | yes | 12.3 | strong — expand (Nimzowitsch My System) |
| Positional | Pawn Breaks  | yes |  9.1 | strong — expand (minority attack, levers) |
| Positional | Open Files   | yes |  4.5 | modest — expand cautiously (7th rank, file seizure) |
| Offensive  | Missed Sacrifices | — | — | already pure router; sac is rare-by-nature |
| Defensive  | Allowed Sacrifices | — | — | mirror of above |
| Calculation| Premature Trades  | — | — | check next |
| Calculation| Tactical Resources (Desperado) | — | — | rare-by-nature; likely drill-position not corpus |

KEY: Positional is the most under-built scored group — 3 single-feature clusters all monotonic with
real spread. Same conclusion as endgame: build more detectors, don't drop. Candidate detectors to
build + measure on eligible-miss-rate (chess-poc):
- Prophylaxis family: missed luft/escape-square, missed defensive trade, missed blockade (mid-game)
- Pawn Breaks family: missed minority attack, missed pawn lever, missed central break (…d5/…e5/c5/f5)
- Open Files family: missed rook-to-open-file (mid-game, not just endgame), missed 7th-rank seizure,
  missed file-contest (doubling on a half-open file)
Next: build these, measure eligible-miss-rate by band, keep the monotonic ones, attach to the cluster.
