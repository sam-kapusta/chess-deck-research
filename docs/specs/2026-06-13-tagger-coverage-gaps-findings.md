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

So **`quietMove` alone tags 50%** of the naked set, and **+ king-activity → ~59%**. That's the headline:
two detectors cut the naked-mistake count roughly in half.

**The 41% forcing residual is murkier — be honest about it.** Breaking it down: 417 "best attacks a
piece," 127 "best gives check," 72 "escapes a check." But "best move attacks a piece" (e.g. Qd7→Qd8)
usually just means the best move *makes a threat* — not that there's a clean named tactic (fork/pin/
skewer) we failed to fire on. A few are genuine detector false-negatives worth auditing, but most are
"the engine's move happens to be active." **Do not treat the forcing-naked set as a tactical-detector
bug to chase** — it's mostly the irreducible "the best move was just better, no nameable theme" case,
which is the LLM-narrative's job, not a rule's. (Sanity-checked the samples: Qd7/Qd8, Qf2/Qc3 — no clean
motif to name.)

## Ranked options (what to build, by value÷effort)

**Tier 1 — do these; highest value, upstream code exists or logic is simple:**

1. **`quietMove` / "Quiet Move Was Best"** (port from cook.py). **Measured: tags 50% of all naked
   mistakes (759/1505).** ~30 LOC, upstream-proven. Caveat: it's a weak explanation ("the right move was
   positional") — but a weak explanation beats a naked mistake, and it's the floor the better detectors
   build on. **Recommended first** — biggest single drop in the naked rate for the least code.

2. **King-activity / opposition detector** (endgame, NEW). **Measured: +9% (130/1505)** — and these are
   the moves `quietMove` deliberately won't touch, so it's purely additive. A focused detector: in an
   endgame (≤ a piece-count threshold),
   if the best move is a king move toward the center / toward the action / taking the opposition, and the
   played move wasn't → "Missed King Activity" / "Lost the Opposition." Opposition is computable (kings on
   same file/rank/diagonal, odd squares between, side-to-move loses opposition). Highest *measured*
   coaching value. Needs care to avoid overfiring.

**Tier 2 — real value, more design:**

3. **Implement `overloading`** (currently stubbed `return False`). It's already in the taxonomy and
   referenced by `_MATE_OUTRANKS`; finishing it is honest completion, not new scope.
4. **Passed-pawn creation / blockade** (endgame + middlegame). Computable from pawn structure
   (no enemy pawns on the file or adjacent files ahead). Pairs naturally with the endgame push.
5. **Open-file rook** / **bad bishop** — classic positional axes, fully static-computable.

**Tier 3 — defer; high effort or low marginal value:**

6. `zugzwang` (engine-required — needs the static analyzer's Stockfish, doable once that lands but
   couples the tagger to eval). `collinearMove`, `defensiveMove` (niche). Exchange sacrifice, trade
   management, prophylaxis (fuzzy — hard to make precise without false positives).

## The key decision for Sam

**Is the goal coverage (fewer naked mistakes) or precision (every tag trustworthy)?** They pull opposite
ways:

- The naked-mistake gap is overwhelmingly **quiet positional/endgame** moves. Closing it means detectors
  that reason about *position*, not material — which are inherently fuzzier and overfire more (cf. the
  skewer floor, the pin preexisting-gate — every positional detector has needed a guard).
- Alternatively: **accept that quiet positional mistakes get no tactical tag**, and instead surface the
  *eval swing + best move* plainly ("Best was Kd3 — king activity"), letting the LLM coach narrate the
  why from the position rather than forcing a rule-based label. This dodges the overfire risk entirely
  and may be the *better* product answer for the quiet 92% — rules are good at tactics, bad at judgment.

My lean: **port `quietMove` (cheap, safe, upstream-proven) + build the endgame king-activity detector
(highest measured value), and for everything else quiet, lean on the LLM narrative rather than stacking
fuzzy positional rules.** That matches the existing instinct in the codebase — keyword rules for what's
crisp (STANDARDS.md: "keyword rules are often more accurate" — but that's for *tactical/material* facts;
positional judgment is exactly where it inverts).

## Verification notes for whoever builds this

- The naked-rate measurement is reproducible: `output/mistake_tags.json` (19,362 moments) + the bucketing
  in this session's `/tmp/corpus_gaps.py` / `naked_chars.py`. Re-run after adding any detector to watch
  the naked rate drop — that's the success metric.
- Every new positional detector MUST add a regression case (the dir had none for skewer until today) AND
  a corpus before/after firing count, like the skewer floor did (105/286 measured). No detector ships on
  one hand-picked example.
- King-activity is the one to prototype against the four endgame FENs above first — if it can't cleanly
  tag those without firing on quiet won positions, it's not ready.
