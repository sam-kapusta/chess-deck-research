# Tagger Coverage Gaps — Findings (2026-06-13)

> **Status: exploration, not a spec.** Written under `/goal` AFK latitude. Sam asked: "brainstorm
> optimal tags and things we could potentially be missing — endgame, more subtle mistakes etc."
> This is the evidence + ranked options. The actual fix needs Sam to pick a direction first.

Companion to the rule-based tagger (`scripts/04_tagger/`, design `2026-06-07-rule-based-mistake-tagger-design.md`)
and the in-flight tag-DISPLAY brainstorm (which is a separate concern — what to *show*; this is what
the tagger can *detect* at all).

## The one number that matters

Ran the tagger's own 19,362-moment corpus (`output/mistake_tags.json`) and measured the **naked-mistake
rate**: a real mistake (cp_loss ≥ threshold) that gets **zero explanatory tags** — only phase/game-state
context ("Endgame", "Blunder While Equal"). A naked mistake is one the coach can name *that* it happened
but not *why*.

| Severity band | n | naked | naked % |
|---|---|---|---|
| inaccuracy (50–100cp) | 461 | 111 | **24.1%** |
| mistake (100–200cp) | 2,710 | 423 | **15.6%** |
| blunder (200–400cp) | 8,265 | 957 | **11.6%** |
| huge (400+cp) | 6,699 | 255 | 3.8% |

**~1,500 real mistakes (cp ≥ 150) currently get no explanation.** The tagger is strong on big tactical
swings (huge band only 3.8% naked — those are hanging pieces, mates, forks, all covered) and weak on the
quieter the mistake gets. That's the shape of a tactically-complete, positionally-thin detector set.

## What the naked mistakes actually ARE

Characterized all 1,505 naked mistakes (cp ≥ 150):

- **92% — the best move is QUIET** (no capture, no check). There is no tactic to detect because the
  *correction* isn't tactical. It's a better square, a king step, a prophylactic move.
- **Phase split:** 701 middlegame, 497 opening, 307 endgame. Endgame is only 20% by count but it's the
  band with the **highest naked RATE** — endgames are disproportionately quiet-positional.
- **Best-move piece:** pawn 26%, rook 21%, knight 14%, **king 14%**, queen 13%, bishop 12%. King moves
  are 14% of corrections but kings are ~3% of pieces — a massive over-representation. **King-activity
  mistakes are a real, concentrated, untagged class.**

Concrete endgame examples (all naked today — only "Endgame / Blunder While X / <type> Endgame"):

```
8/8/6p1/5pkp/8/4N3/4K3/8 w     played Ng2  best Kf3   (+211cp)  — king must advance, not shuffle the N
8/8/3nk1p1/5p1p/3K3P/p1N3P1/P7/8 w  played Na4  best Ke3  (+214cp)  — king centralization
8/8/8/3k1p1p/3P2pP/4K1P1/8/8 w   played Kf4  best Kd3   (+487cp)  — wrong king route / opposition
8/8/3k2pp/p2P1p2/2K2P1P/1p3P2/1P6/8 w  played Kb5  best Kd4 (+247cp) — king route in pawn endgame
```

These are **opposition / king-activity / king-route** errors — textbook endgame teaching, and the single
clearest gap. A coach looking at these wants to say "your king went the wrong way," and we say nothing.

## Where the gaps came from (vs the upstream we ported)

We ported tactical motifs from lichess-puzzler `cook.py`. The upstream `TagKind` enum (vendor/model.py)
is a 57-theme vocabulary. What we did NOT port:

**Upstream detectors we skipped (exist in cook.py):**
1. `quietMove` — best move is positional/prophylactic (not check/capture/attack/pawn-push/king-move).
   **This is the single highest-value miss** — it's literally the "92% of naked mistakes" category, and
   the upstream already has the detector. It says "the right move was quiet" — turning a naked mistake
   into at least a *category* of explanation.
2. `defensiveMove` — the quiet move is purely defensive (last-move-of-line variant).
3. `zugzwang` — engine-required (null-move eval comparison). Named endgame concept.
4. `collinearMove` — ray piece aligns on an enemy ray piece's line without capturing (battery/x-ray).
5. eval-magnitude (`crushing`/`advantage`/`equality`) and line-length (`oneMove`/`short`/`long`) tags —
   low coaching value, skip.

**Stubbed in our code, never fires:** `overloading` (`return False`). Real motif, just unimplemented.

**Absent from BOTH upstream and us (the positional/endgame frontier):**
- **King activity / opposition / king route** (endgame) — *highest measured value per the corpus*
- Passed-pawn creation / blockade failure; pawn breakthrough
- Open-file control (rook on open file)
- Bad bishop (blocked by own pawns)
- Prophylaxis failure; tempo loss (distinct from our "Wrong Move Order")
- Exchange sacrifice (strategic R-for-minor — distinct from tactical "Sacrifice")
- Trade management (trading the wrong piece — your only active piece)

## Precision bugs found + fixed this session (separate from coverage)

While reviewing real outputs Sam flagged, two crisp **mislabel** bugs surfaced and were fixed (same
discipline as a coverage gap is different work — these are "the tag we DO emit is wrong," not "we emit
nothing"):

1. **Skewer pawn-floor** — `skewer_line` fired on any ray capture of a *pawn* on a square a major piece
   vacated along the ray (a deflection/discovered-attack, not a skewer). 105/286 reconstructable fires
   (37%) were pawn-grabs. Floor: captured back piece must be ≥ a minor. All 181 piece-back fires kept.
2. **Exchange equal-value gate** — `capture_or_exchange` labeled a higher-value piece taking a defended
   lower-value one (Q×defended-B) as "Missed Bishop Exchange." That's a sacrifice, not an even trade —
   and `sacrifice_line` already names it, so the label was wrong AND contradictory. 310/1274 Exchange
   fires (24%) were this misfire; 207 co-occurred with "Missed Sacrifice." Gate: suppress when
   attacker > victim+0.5. All 964 genuine trades kept.

Both have regression cases now. The lesson for the coverage work below: **every detector needs a corpus
before/after firing count + a regression case** — both bugs were invisible until measured against the
19K corpus, and neither had a regression case before.

## What we already cover well (don't re-build)

Top explanatory tags by corpus frequency: Lost Material to Combination (4,007), Bad Capture (3,520),
Hung Material (3,030), Missed Sacrifice (3,030), Wrong Move Order (2,450), Pawn Move Exposed King (2,038),
Missed Clearance (1,283), Allowed Mate (1,236)… The tactical + material + king-safety axes are dense.
Endgame *type* naming (Rook/Knight/Bishop/Pawn/Q+R, same/opp-color bishop) exists but is **info-only** —
it names the board, not the mistake.

## Measured: how much would `quietMove` + king-activity actually close?

Prototyped the upstream `quietMove` definition (best move: no check given/escaped, no capture, doesn't
attack a non-pawn piece, not an advanced pawn push, not a king move) against all 1,505 naked mistakes
(cp ≥ 150). Breakdown of what the *best move* is:

| Best-move kind | count | % of naked |
|---|---|---|
| **quiet** (would tag "Quiet Move Was Best") | 759 | **50%** |
| **king move** (upstream excludes — needs the king-activity detector) | 130 | **9%** |
| **forcing** (capture / check / attacks a piece) | 616 | **41%** |

So **`quietMove` alone tags 50%** of the naked set, and **+ king-activity → ~59%**.

**⚠️ Update (2026-06-13, after review with Sam): the 50% is a mirage — don't chase it.** `quietMove`
covers 50% *precisely because it's contentless* — it fires on the absence of a tactic, telling the player
nothing about why their move was bad. The naked-rate metric counts any tag as "covered," so a descriptor
scores like a real explanation. That's the metric's flaw, not quietMove's strength. The two kinds of tag:
**explanations** ("you lost the opposition" — teach something) vs **descriptors** ("quiet move was best" —
describe the board). The metric can't tell them apart; a human reading outputs can. **The right metric is
"does the tag teach something true about THIS mistake," which is eyeball-only.** See revised ranking below.

**The 41% forcing residual is murkier — be honest about it.** Breaking it down: 417 "best attacks a
piece," 127 "best gives check," 72 "escapes a check." But "best move attacks a piece" (e.g. Qd7→Qd8)
usually just means the best move *makes a threat* — not that there's a clean named tactic (fork/pin/
skewer) we failed to fire on. A few are genuine detector false-negatives worth auditing, but most are
"the engine's move happens to be active." **Do not treat the forcing-naked set as a tactical-detector
bug to chase** — it's mostly the irreducible "the best move was just better, no nameable theme" case,
which is the LLM-narrative's job, not a rule's. (Sanity-checked the samples: Qd7/Qd8, Qf2/Qc3 — no clean
motif to name.)

## SHIPPED (2026-06-13): four endgame detectors

Built + shipped the "BUILD" tier below. Each fires when the best move exhibits the theme AND the played
move didn't — **no causal gate** (Sam's call: fire-when-present, prune noise by reviewing outputs). All
render as "Endgame" chips on Review (the drill-filter bridge is still a separate project). Spec:
`2026-06-13-endgame-detectors-design.md`. Corpus firing rates (19,342 moments), all sane (<3%, no
pov-parity-style runaway):

| Detector | Fires | Eyeball verdict |
|---|---|---|
| Missed King Activity | 489 (2.5%) | clean on real endgame king mistakes |
| Lost the Opposition | 43 (0.2%) | clean, all pawn endgames, real opposition geometry |
| Missed Passed Pawn | 556 (2.9%) | geometry correct, but **main noise source** — fires on captures in crowded middlegames where the passer is a side effect; watch in outputs |
| Rook Behind Passer | 85 (0.4%) | clean |

Regression 63/63 (8 new endgame cases). `quietMove` deliberately NOT built (router-only, per below).

## DECIDED ranking (2026-06-13, with Sam) — by coaching value, NOT naked-rate

The principle we settled on: **a tag must teach something true about THIS mistake.** Descriptors that
just characterize the board don't count, no matter how much naked-rate they "cover."

**BUILD — these name a concept the player can learn:**

1. **King-activity / opposition** (endgame, NEW). Covers only 9% by the metric, but every fire is real
   content: "your king went the wrong way," "you lost the opposition." Concept the player learns;
   opposition is precisely computable (kings on same file/rank/diagonal, odd squares between, side-to-move
   loses it). **The actual winner** — smaller number, real teaching. Catch: precision without overfiring
   on quiet-but-fine king moves is genuine work. Prototype against the 4 endgame FENs above FIRST.
2. **Passed-pawn creation / blockade**. Real, nameable, statically computable (no enemy pawns on the file
   or adjacent files ahead). Build it.

**ROUTER, NOT A CHIP:**

3. **`quietMove`** — do NOT ship as a user-facing tag. It's contentless as an explanation. Its real value
   is as an internal flag ("this mistake is positional, not tactical → hand to the LLM to narrate"). The
   50% coverage is the metric rewarding noise. Use it to steer, never to display.

**DROP / DEFER:**

4. **`overloading`** (stubbed `return False`) — SKIP. "It's already in the enum" is not a reason. Reliable
   overload detection is hard (prove a piece had two duties, one removed); cook.py stubbed it for that
   reason. Only revisit with a clean detection idea.
5. **Open-file rook / bad bishop** — DEFER. Same trap as quietMove: detecting "there's a bad bishop"
   is easy, but proving "*this move* was a mistake *because* of it" is hard. The easy version is a
   descriptor. Low value-per-risk.
6. `zugzwang` (engine-required), `collinearMove`, `defensiveMove`, exchange-sacrifice, trade-management,
   prophylaxis — defer (engine coupling or too fuzzy to make precise).

**Everything else quiet → the LLM narrative**, not a rule. Rules are good at tactics/material (crisp
facts), bad at positional judgment — exactly where STANDARDS.md's "keyword rules are often more accurate"
inverts. Don't stack fuzzy positional rules to chase coverage.

## The metric lesson (why the ranking flipped)

The naked-rate metric (% of mistakes with ≥1 tag) **rewards noise** — a contentless descriptor scores
like a real explanation. It made `quietMove` look like the top pick (50%) when it's the weakest. The
honest metric is "does the tag teach something true about this mistake," which is **eyeball-only** — Sam
reading ply 15 / ply 50 caught more than the 19K-corpus script did. Use the corpus for *firing counts /
overfire sizing* (where it's great — see the skewer & exchange fixes), NOT for ranking coaching value.

## Verification notes for whoever builds this

- The naked-rate measurement is reproducible: `output/mistake_tags.json` (19,362 moments) + the bucketing
  in this session's `/tmp/corpus_gaps.py` / `naked_chars.py`. Re-run after adding any detector to watch
  the naked rate drop — that's the success metric.
- Every new positional detector MUST add a regression case (the dir had none for skewer until today) AND
  a corpus before/after firing count, like the skewer floor did (105/286 measured). No detector ships on
  one hand-picked example.
- King-activity is the one to prototype against the four endgame FENs above first — if it can't cleanly
  tag those without firing on quiet won positions, it's not ready.
