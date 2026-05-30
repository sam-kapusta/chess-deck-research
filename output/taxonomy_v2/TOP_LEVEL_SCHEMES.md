# Top-Level Category Schemes — exploration (2026-05-30)

Sam asked: explore *multiple ways* to divide the ~2000 features at the **top level**.
Sub-clusters deferred. This is the design-space map, grounded in real counts +
fire-rate share (model: `maia3_sae_diff_v2_2048_k32_l2`, fire rates from
`firerate_flat_v2_k32.npy`).

## The central finding that shapes everything

**Semantic clustering (bge-m3, bottom-up) over-weights ONE concept.** ~14 of the 20
emergent categories are variants of *"played a slow/natural/plausible move when the
position demanded forcing action"* ("Slow Play Punished", "Autopilot Over Forcing
Action", "Routine Moves at Critical Moments", "Passive Drift", "Missing The Critical
Moment"…). That's not 14 lessons — it's **one dominant lesson** with many sub-flavors.

Why: at ~1800, the #1 blunder mode genuinely IS "quiet move when action was needed."
The SAE learned many neurons for it. So **pure label-text clustering gives a lopsided
top level** (one giant theme + specifics). The even-spread ward cut only looks balanced
because it *splits that one theme* into many — which isn't coaching-meaningful at the top.

**Implication:** the top level needs a deliberate *axis choice*. The mechanism axis
(below) is the fix — it asks "what concretely went wrong" first, so "a slow move that
hangs a piece" is filed under the actionable lesson (hangs a piece), not under "slow play."

---

## Scheme 1 — Concrete Mechanism (RECOMMENDED top level)

Priority-ordered: classify by what *actually happened on the board*, most actionable first.
This is the only scheme that's both coaching-meaningful AND reasonably spread.

| Category | feats | %feat | %fire |
|---|--:|--:|--:|
| Hangs a piece | 821 | 41% | 42% |
| Greedy capture | 296 | 15% | 14% |
| King into danger | 219 | 11% | 10% |
| Abandons defense | 192 | 10% | 12% |
| Endgame error | 183 | 9% | 8% |
| Pointless check | 129 | 6% | 5% |
| Unsound sacrifice | 58 | 3% | 3% |
| Bad pawn push | 40 | 2% | 3% |
| Bad capture/trade | 37 | 2% | 2% |
| Slow/quiet move (pure) | 19 | 1% | 1% |

- **Strength:** every category is a lesson a coach assigns drills for. "Hangs a piece" being 41% is *true* and *useful* — it's this player's biggest leak.
- **Weakness:** "Hangs a piece" is large. If we want it smaller, it splits cleanly by HOW (moved-into-attack vs left-behind vs abandoned-guard) — that's the sub-cluster level.
- **Note:** keyword-classified here as a fast proxy; final version needs LLM/agent assignment for precision (the keyword version mis-files some edge cases).

## Scheme 2 — Player-Facing Themes (fewest, most marketable)

6 broad "what to work on" buckets. Good for a product UI / coach-facing summary.

- **Piece Safety** (hangs + abandons + moves-into-attack) ≈ 50%
- **Don't Be Greedy** (greedy captures + bad trades) ≈ 17%
- **King Safety** (king-into-danger + weakening pushes) ≈ 13%
- **Play With Urgency** (the slow-play cluster) ≈ 10%
- **Endgame Technique** ≈ 8%
- **Tactical Alertness** (pointless checks, unsound sacs) ≈ small

- **Strength:** a player sees 6 things, instantly grasps them.
- **Weakness:** Piece Safety ≈ half. Acceptable for a summary, too coarse for training.

## Scheme 3 — Coaching Skill (Heisman-style "what skill failed")

Board vision / Calculation / King safety / Time management / Material / Piece activity /
Pawn play / Endgame. **Tested by embedding-assignment → collapsed (68% into one bucket).**
The skills overlap too much in feature space to assign cleanly. **Rejected as a top level**
unless built by agents reading holistically — and even then the categories bleed.

## Scheme 4 — Game Phase (opening / middlegame / endgame)

Tested → **97% middlegame.** The blunder dataset is overwhelmingly middlegame positions,
so phase is nearly constant. **Rejected** — no discriminating power. (Could be a *secondary*
tag, not a top level.)

## Scheme 5 — Intent (overreaching / too passive / failed defense / faulty conversion)

Tested → 75% "overreaching." The framing doesn't match how the features separate.
**Rejected** as primary; "too aggressive vs too passive" is interesting as a *cross-cut*
attribute but not the spine.

## Scheme 6 — Severity / Frequency (fire-rate bands)

Not a *category* axis — it's an ordering. Useful as a **secondary sort** within any scheme:
- Pervasive (>5% fire): 44 features, 12% of all firing — the highest-leverage to fix.
- Common (1-2%): 1204 features, 50% of fire — the bulk.
- Very rare (<0.5%): 12 features — long tail.

Use this to rank *which features/categories matter most*, layered on Scheme 1 or 2.

---

## Scheme 7 — Thinking Error (CREATIVE: why the brain failed) ⭐

Organize by the *cognitive* mistake, not the board mistake. Psychologically framed,
tells the player about their **habits**. Naturally well-balanced.

| Category | feats | %feat | %fire |
|---|--:|--:|--:|
| Autopilot — didn't stop to think | 761 | 38% | 40% |
| Oversight — didn't see the threat | 467 | 23% | 23% |
| Temptation — grabbed the bait | 465 | 23% | 22% |
| Overconfidence — overreached | 154 | 8% | 8% |
| Passivity — too timid | 84 | 4% | 4% |
| Miscalculation — saw it, got it wrong | 65 | 3% | 3% |

- **Strength:** the most *coaching-resonant* framing — "you play on autopilot at critical
  moments" is a habit a player can actually change. Balanced (top 38%). Maps to known
  improvement psychology (Heisman's "hope chess" / "real chess").
- **Weakness:** softer to verify per-feature than mechanism; "autopilot" is inferred.

## Scheme 8 — Piece You Mishandled (CREATIVE: drill by piece) 

The most evenly-spread axis of all. Dead-simple to act on ("work on your rook play").
This is the axis bge-m3 clustering naturally discovered (clusters split by piece).

| Category | feats | %feat |
|---|--:|--:|
| Pawn handling | 550 | 28% |
| Rook handling | 386 | 19% |
| Knight handling | 284 | 14% |
| Bishop handling | 280 | 14% |
| King handling | 251 | 13% |
| Queen handling | 245 | 12% |

- **Strength:** perfectly balanced, trivially assignable (we have dom_piece), intuitive drills.
- **Weakness:** "rook mistakes" isn't a *lesson* — a player doesn't think "I'm bad at rooks,"
  they think "I hang pieces." Piece is a great **secondary** cut (sub-cluster within a lesson),
  weak as the spine. (This is exactly why we embed mechanism, not piece, for clustering.)

## Scheme 9 — The Lesson (action-framed) — collapses to 2

"What to do instead": **Check your pieces are defended (49%)** + **Look for forcing moves
first (47%)** dominate; everything else <3%. Too coarse for a top level, but a striking
finding: **96% of this player's blunders reduce to two meta-lessons.** Good as the headline
of a coaching report, not as a navigable taxonomy.

---

## The schemes side by side (balance + verdict)

| Scheme | #cats | largest | spread | best for | verdict |
|---|--:|--:|---|---|---|
| 1 Concrete Mechanism | ~10 | 41% | ok | drills, coach diagnosis | **strong** |
| 2 Player Themes | 6 | ~50% | coarse | product UI roll-up | good as roll-up |
| 7 Thinking Error | 6 | 38% | **good** | habit-change coaching | **strong / most creative** |
| 8 Piece | 6 | 28% | **best** | piece-specific drills | good as 2nd axis |
| 3 Coaching Skill | 8 | 68% | bad | — | rejected (collapses) |
| 4 Game Phase | 3 | 97% | bad | secondary tag | rejected |
| 5 Intent | 4 | 75% | bad | cross-cut attribute | rejected |
| 6 Severity | bands | — | — | "fix first" sort | secondary layer |
| 9 The Lesson | 2 | 49% | n/a | report headline | too coarse |

## Recommendation — depends on PURPOSE (the real question)

I'm not going to pick one for you, because the right top-level axis is decided by what
the taxonomy is *for*, not by the data. Three live options, each best for a different goal:

- **If it's coach-facing diagnosis / drill assignment → Scheme 1 (Concrete Mechanism).**
  Actionable, data-honest. "This player hangs pieces 41% of the time" is the headline.
- **If it's player-facing habit coaching → Scheme 7 (Thinking Error).** ⭐ My pick if I
  had to choose one — it's balanced (38% top), the most creative, and "you play on
  autopilot at critical moments" is a habit a player can actually change. Maps to real
  improvement psychology.
- **If it's a product UI / browsable atlas → Scheme 2 (6 themes)** as a roll-up, with
  **Scheme 8 (piece)** or mechanism as the sub-level, and **Scheme 6 (fire-rate)** as the
  "fix this first" sort.

**Cross-cutting, regardless of spine:** layer fire-rate (severity) as the priority sort,
and keep piece as a secondary facet. Drop phase/intent/skill as top levels — tested, they
collapse (97% / 75% / 68% into one bucket).

### The question only you can answer
**What is this taxonomy for?** Coach diagnosis, player drills, product UI, or a research
map? That decides the spine. Tell me and I'll build that one out with proper LLM/agent
assignment (the keyword classifications here are fast proxies — real version needs the
holistic agent pass per category + the 3 QC passes).

### Two judgment calls within whichever you pick
1. **"Hangs a piece" / "Autopilot" is ~40%.** Keep as the true #1 leak, or split? Data
   supports either. I lean keep-at-top + split into sub-clusters underneath.
2. **Granularity:** 6 categories (clean, marketable) vs ~10 (more precise). 6 reads better;
   10 loses less.
