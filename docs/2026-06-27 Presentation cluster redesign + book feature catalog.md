# Presentation cluster redesign + book feature catalog (2026-06-27)

Mandate (Sam, before going AFK): "keep naming features from the books we reviewed, get strong cluster
names like we discussed that align with common mistakes and what books recommend so they're
presentable. Features as many as we can get, clusters optimized for presenting and volume not just
splitting up by feature count."

## The principle (decided this session)

There are TWO layers and we'd been conflating them:

1. **Scoring layer** — the group bars (Offensive / Defensive / Calculation / Positional / Endgame).
   Must have COMPLETE coverage so the FIFA score is honest. Every label counts toward its group.
2. **Presentation / drill layer** — the line items a player actually sees + drills. The display unit
   should be the **named, book-recognized concept** (Fork, Pin, Skewer, Mate, Discovered Attack,
   Prophylaxis, King & Pawn endgame), NOT an academic umbrella ("Combinative Motifs",
   "Discovered Attacks & Skewers").

### Rules for a cluster being its own presentable line item
A standalone cluster (even a SINGLETON) earns a card iff:
- **Volume** — enough corpus fires to score reliably (not noise). [thresholds set by volume job below]
- **Book-named** — a player/coach recognizes it by name (it's in Silman / Nimzowitsch / de la Villa /
  Dvoretsky / a tactics trainer's motif list).
- **Spread** — monotonic, differentiates skill (beginners err more). Already verified for most.

Feature-COUNT is irrelevant. A singleton is fine if it's big + book-named (Sam: "okay to have some
clusters be singletons if large enough/expressed in books"). Prophylaxis = one concept Nimzowitsch
wrote a chapter on; it does NOT need siblings bolted on to justify a card.

### Rare motifs → pooled rollup, not hidden
The long tail (Interference, Clearance, Attraction, Zwischenzug, Battery, X-Ray, Double Check) is too
rare to be its own card but must still COUNT toward the score. Pool into ONE honest rollup
("Other Combinations" / "Other Tactics") rather than burying the common ones under it. Promote the
common + named ones (Fork, Pin, Skewer, Discovered Attack, Trapped Piece, Deflection, Overloading)
to their own line; pool the rest.

Sam's litmus: "if i think of offense i think of — Missed Fork, Missed Pin, Missed Mate, Missed
Skewer, etc." The display must match that mental model.

### "Allowed X" framing
"Allowed X" (you let the opponent do it) drills worse than "Missed Y" (you failed to do it). The
defensive group is currently ALL passive "Allowed". The book workflow's best new idea is
**Active Defense** — the MISSED-half of defense: the resources you failed to USE (unpin, interpose,
remove the attacker, counter-sac, cross-check). That's a coherent, presentable drill unit and fills
a real hole. Keep the "Allowed X" buckets for scoring; surface Active Defense for drilling.

## Existing label inventory (ground truth — what the tagger can already emit)

Motifs (motifs.py), each in missed/allowed direction:
  fork, pin, skewer, discoveredAttack, deflection, clearance, attraction, interference, zwischenzug,
  trappedPiece, overloading, battery, xRayAttack, doubleCheck
Named mates: backRank, smothered, arabian, anastasia, boden, dovetail, hook, doubleBishop, + generic
  mate + faster-mate.
Material (predicates): Hung {piece}, Hung Material, Missed Free {piece}, Greedy Capture, Pawn Grab
  While Undeveloped, Missed Capture (Pawn), Missed Capture of Defender.
Exchanges: Missed {piece} Exchange, Missed Bishop-Knight Exchange, Premature Trade.
Backfired (calc): Failed Fork, Failed Pin, Failed Discovered Attack.
Positional: Missed Prophylaxis, Missed Pawn Break, Missed Open File, Missed Outpost/Allowed Outpost,
  Missed Piece Activation, Missed Tempo Push, Created Doubled/Isolated/Backward Pawn, Bad
  Simplification, Pawn Move Exposed King, Lost Castling Rights, Missed Blockade.
King attack: Enemy King Exposed, Missed/Allowed Kingside Attack, Missed/Allowed Queenside Attack,
  Missed/Allowed f2/f7 Attack.
Endgame (by material now): opposition, pawn race, passers (passed/connected/outside/protected),
  square rule, breakthrough, promotion family, en passant, rook technique (7th/cutoff/active/open
  file/behind passer), piece activity (king/bishop/knight/minor/queen), perpetual, simplify.
Threat: Ignored Threat, Missed Defensive Resource, Missed Desperado.

## Book feature catalog — workflow w5zuk548s (66 agents, 243 concepts cataloged, 60 specced)

### TOP TIER — build now (real gaps, plausible strong gradient)
| Feature (presentable name) | Detector sketch | Cluster | Dir |
|---|---|---|---|
| **Missed Pin Exploitation** ("pile on the pin") | enemy ≥N pinned by a pov ray-piece, currently held (atk≤def); best is a non-capturing prep adding a NEW attacker (ideally a pawn) on the pinned piece; played didn't | Pins | missed |
| **Missed Unpinning Resource** | mover has a pinned piece (abs via is_pinned / rel via ray-scan); best leaves it no-longer-pinned (capture pinner / king-step / interpose / rear-piece move); played left pin standing | Active Defense | missed |
Both reuse `_pin_target`. Verify: pile-on denominator size; unpinning incidental-unpin overfire.

### SECOND TIER — standalone, medium freq (measure eligible-miss-rate first)
| Feature | Net-new slice vs existing | Cluster | Dir |
|---|---|---|---|
| **Over-extended Pawn** | push left a permanent hole on a half-open file (NOT caught by backward_pawn, which needs enemy pawn on stop sq). Risk: masters over-extend more by count → band test may invert. RUN IT. | Pawn Structure | played |
| **Released Central Tension** | the PUSH half (e4-e5 locking center) — no detector covers it. Even-trade only, no material confound. Dedup vs premature_trade. | Pawn Breaks / Center | played |
| **Remove the Guard (of castled king)** | even-trade slice (Nxf6 removing h7-defender as a trade not a sac). Geometry self-selects: victim must guard a king-zone square. | King Attack | missed |

### SECOND TIER — standalone, low freq (build only if corpus count clears floor)
| Feature | Note | Cluster | Dir |
|---|---|---|---|
| **Missed Blockade (jail the passer)** | cleanest geometry (knight, empty front, quiet, safe). Dedup before Prophylaxis/Outpost. Measure on full move corpus. | Positional (sibling Outpost) | missed |
| **Missed Shield Sacrifice** | symmetric twin of shipped f7/f2. Must suppress generic Missed Sacrifice + Enemy King Exposed co-fire or it's a relabel. | Sacrifices (named subtype) | missed |
| **Traded the Fianchetto Bishop** | clean geometry (B on g7/b7/g2/b2, king on wing, even trade, best kept it). | King Safety | played |
| **Surrendered the Center** | has_duo before/after = causal attribution, no eval heuristic. Co-fire audit vs premature_trade/greedy. | Center | played |
| **Castled Into the Attack** | refutation-causation gate separates from coincidental drops. Verify fire count (castles happen early). | King Safety | played |
| **Missed Interposition (block the check)** | exact geometry, near-zero misfire. | Active Defense | missed |
| **Missed Counter-Sacrifice (give it back)** | clean mechanism; denominator band-confounded (masters rarely sit up-material-but-equal). | Active Defense | missed |

### SUB-LABELS only (~100% co-fire existing detector — enrich evidence string, NO new scan)
Removing-the-Attacker (extend missed_defensive_resource); minor-choice family (good/bad bishop,
bad minor, B-vs-N-by-structure, knight-superior-closed → sub-labels on capture_or_exchange's Missed
Minor Exchange); Missed Cross-Check (annotation on in-check missed tactic); Pin-the-Defender
(refine Missed Pins); half-open-file pressure / exploit backward-doubled / IQP blockade (evidence on
Open File / Outpost); strike-the-base / minority-storm context (branches in missed_pawn_break);
rook-lift / g-file / h-file / opposite-castling-race / uncastled-king (sub-labels on King Attack /
Open File — their unique slice is the slow prep move the win_drop gate strips).

### CURATED DRILL positions (iconic, too rare / not corpus-detectable)
Named attacking combos: Greek Gift (Bxh7+), Double-Bishop Sac, Knight sac f5/e6/d5/g7, Exchange sac,
Positional exchange sac (Petrosian), Rook-lift attacks, g/h-file fianchetto attacks, pawn-storm,
hook creation. Iconic technique: Underpromotion, Stalemate swindle, Minority Attack (Carlsbad),
IQP play, Hanging pawns, Space (Maroczy/French/Benoni), Luft/back-rank, "which minor to keep".
Not-detectable (search/multi-move/judgment): Windmill, Fortress, Two-weaknesses, Overprotection,
Restraint, Zugzwang/triangulation, Corresponding squares, Reti maneuver, Mysterious rook move,
named structure plans, sham-vs-real sac, focal points, initiative, reserve tempo, Q+P vs Q.
Cognitive/meta (Rowson/Hendriks): time-trouble, blinkered vision, no-plan, confirmation bias,
playing the board not the opponent — coaching content, NOT board detectors. Park.

### DROP / already shipped (no new signal)
Double Check (have doubleCheck), X-Ray (have xRayAttack, can't band-test), Capture-of-Defender
missed-direction (FAILS band test: flat), f7/f2 (have attackingF2F7). Create-passed-from-majority
(subset of missed_passed_pawn), activate-worst-piece (subset of missed_piece_activation), bishop-pair
surrender (rarely clears gate), knight-on-rim / space / pawn-storm-seed (sub-threshold or subsets).

## Proposed presentable cluster scheme (FILL volume numbers from job)

### Offensive Tactics — display by named tactic
- **Missed Mate** (rename from "Missed Forced Mates") — generic + all named mates pooled in
- **Missed Fork**
- **Missed Pin** (+ Pin Exploitation when built)
- **Missed Skewer** (SPLIT OUT of "Discovered Attacks & Skewers")
- **Missed Discovered Attack** (SPLIT OUT; Double Check + X-Ray pool under it or Other)
- **Missed Trapped Piece** (rook/knight/bishop/queen pooled — "Trapped Piece" is one named concept)
- **Missed King Attack** (kingside/queenside/f7 + Remove-the-Guard)
- **Missed Sacrifice** (+ named sac subtypes as evidence)
- **Other Combinations** (rollup: Deflection, Clearance, Attraction, Interference, Zwischenzug,
  Overloading, Battery — promote any that clear volume to their own line)

### Defensive Tactics
- **Allowed Mate / Fork / Pin / Skewer / Discovered / King Attack** (mirror; scoring + "what hit you")
- **Active Defense** (NEW — the missed-half: Unpinning, Interposition, Remove-Attacker,
  Counter-Sac, Cross-Check). Best new presentable cluster in the set.
- **Threat Awareness** (Ignored Threat + generic Missed Defensive Resource — the leftover)
- **Other Combinations Allowed** (rollup)

### Calculation
- **Hung Pieces** (high vol, high spread 43/k — keep, it's great)
- **Missed Free Material** (concrete)
- **Greedy Captures**
- **Backfired Tactics** (Failed Fork/Pin/Discovered — good self-aware drill; keep singleton-ish)
- **Premature Trades**, **Tactical Resources / Desperado** (modest but named; keep)

### Positional
- **King Safety & Castling**, **Prophylaxis** (singleton OK), **Pawn Breaks**, **Pawn Structure**,
  **Outposts**, **Open Files**, **Exchanges** (which-minor-to-keep sub-labels), **Missed Blockade**(new),
  **Pawn Center** (Surrendered Center + Released Tension — only if both clear floor)

### Endgame — rename to book convention
- **King & Pawn Endgames** (from "Pawn Endgames"), **Rook Endgames**, **Queen Endgames**,
  **Minor-Piece Endgames**, **Rook + Minor Endgames** (score-only? clunky name), **Heavy-Piece**
  (score-only). Spotlight the recognizable ones; keep the clunky-named ones scoring-only.

## TODO when volume job lands
- [ ] Fill per-label + per-cluster fire counts → set promote/pool threshold
- [ ] Decide which Combinative motifs promote to own line vs pool
- [ ] Decide Skewer/Discovered split volumes support it
- [ ] Rewrite clusters.json + aggregation to the presentable scheme
- [ ] Add `spotlight`/display metadata distinguishing scored-bar vs drill-card
- [ ] Build top-tier pin detectors; measure

## VOLUME DATA (from shipped fifaSkillRatings.json — production 200k corpus, no re-pull)

The shipped JSON already holds per-feature `by_band` fires + band denominators (200k/band). So any
re-grouping is a pure OFFLINE recompute — `fifa_pipeline/recluster.py` takes a cluster→features scheme
and rebuilds every band rate/anchor/smoothed_rate. No corpus re-pull needed. (Killed the 22-min
full-tagger volume job once I realized this.)

Key volume findings (total_fires, 200k corpus):
- The "Combinative Motifs" umbrella HID the two highest-volume tactics in the system:
  **Overloading 17,520** and **Battery 8,387 missed / 19,411 allowed** — both bigger than Fork/Pin/Mate,
  both book-named. Burying them was the core mistake Sam flagged.
- Big enough for own card: Overload, Battery, Mate, Pin, Fork, King Attack, Sacrifice, Discovered
  Attack, Trapped Piece (pooled 1565).
- Rare → pool into "Other Combinations": Clearance/Deflection/Zwischenzug/Interference/Attraction
  (4275 missed / 6032 allowed combined). Named mates (193) fold into the generic Mate card.
- Skewer: 442/787 fires, low but rate falls cleanly ~12x (0.415→0.035 /k). Real signal, just thin.
  KEEP as own card (it's in Sam's explicit mental list: Fork/Pin/Mate/Skewer), rank low by volume.

## PRESENTABLE SCHEME (scheme_presentable.json) — recompute verified, almost all MONO

37 clusters across the 4 rebuilt groups (Endgame + Openings preserved). spotlight=false = score-only
(feeds the group bar, not shown as a drill card). Spread is /1000 moves, beginner−master.

OFFENSIVE: Missed Mate(20.0) Overload(16.5) King Attack(8.0) Pin(7.9) Battery(7.1) Sacrifice(6.7)
  Fork(5.8) Discovered(3.4) Trapped Piece(1.6) Skewer(0.4) | Other Combinations [score-only](2.7)
DEFENSIVE: Allowed Battery(22.6) Mate(22.2) Threat Awareness(9.3) Active Defense(8.8) Fork(8.4)
  Pin(8.1) King Attack(6.5) Discovered(4.1) Trapped Piece(2.0) Skewer(0.5) | Allowed Sacrifice
  [score-only](6.0), Other Combinations Allowed [score-only](3.5)
CALCULATION: Hung Pieces(43.0) Missed Free Material(22.3) Greedy Captures(17.2) Backfired(8.2)
  Desperado(6.6) Premature Trades(4.5)
POSITIONAL: King Safety(34.0) Prophylaxis(12.3) Exchanges(12.3) Pawn Breaks(9.1) Piece Activity(9.4)
  Outposts(6.0) Pawn Structure(5.9) Open Files(4.5)

NEW clusters introduced: **Active Defense** (Unpinning + Interposition + Defensive Resource — the
missed-half of defense, the workflow's best idea), **Missed Overload** & **Missed Battery** (promoted
out of Combinative Motifs), **Missed/Allowed Trapped Piece**, **Missed Skewer**, **Missed Discovered
Attack** (split from the umbrella), **Piece Activity** (Tempo/Advanced Pawn/Piece Activation merged).
Renamed: "Missed Forced Mates"→"Missed Mate", "Tactical Resources"→"Desperado", "Pawn Advances &
Tempo"→"Piece Activity", "Pawn Structure Weaknesses"→"Pawn Structure".

### Still TODO (needs Sam's eye / measurement)
- The new detectors (Pin Exploitation, Unpinning, Interposition) contribute 0 fires until measured on
  corpus — attach is wired, volume lights up after a chess-poc pull.
- spotlight flag needs frontend consumption (group bars use ALL clusters incl score-only; drill surface
  filters spotlight=true). useSkillCard DISPLAY_ORDER + clusterLiveScore unaffected (score by cluster).
- Endgame rename to "King & Pawn / Queen / Minor-Piece" book convention (separate, pending Sam OK).
- This scheme is a DRAFT in scheme_presentable.json — NOT yet shipped to fifaSkillRatings.json.
