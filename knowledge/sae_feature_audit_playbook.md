# SAE Feature Audit Playbook — find tagger holes by auditing top-firing features

**What this is:** the repeatable process for using a labeled SAE as an audit instrument to find what the
rule tagger MISSES (no detector) or MISLABELS (wrong/catch-all), then fixing it. Run this over any
feature set (or all of them) to find holes. Distilled from the 2026-07-12→14 sessions
(`2026-07-14_sae_tagger_gap_audit_session.md` has the specific findings; THIS is the method).

**Core stance:** the SAE is NOT the product labeler — the rule tagger (`scripts/04_tagger/`) is. The SAE
is a *discovery tool*. It groups blunder positions by a learned direction; some groups are crisp chess
concepts the tagger should name, some are diffuse/outcome clusters that no tag can fix. The audit tells
you which is which.

---

## The pipeline (scripts, in order)

All run on chess-poc (`sais -n chess-poc`). Substrate = `maia3_l7only_v2_dedup` (Maia-3 L7 best−blunder
activation diff). Weights backed up at `s3://chess-sae-weights-140023406996/sae/weights/`.

1. **Label every feature** — `03_feature_labeling/label_features_decile_opus.py`
   Opus reads each feature's decile bands (TIP=top-100, D10..D1) of per-position analyses and assigns a
   VERDICT ∈ {good, diffuse, too_broad, polysemantic, noise} + `good_until_decile` (reach). Independent
   of the tagger. Output: `labels_decile_<sae>.json`. ~30-40 min for 2048 features (Opus, throttled).
   Only ~44% come back `good` (a crisp concept); the rest are the substrate's diffuse mass.

2. **Re-tag the features** — `03_feature_labeling/retag_and_gaps.py`
   Runs the full tagger on the 60k, aggregates each feature's top-200 positions into `tagger_top`,
   `tagger_votes`, `tagger_top_family`, `tagger_family_votes`, `tagger_covered_frac`. Output:
   `retag_<sae>.json`. NOTE: it FILTERS `direction=="info"` tags — descriptive tags (phase, severity,
   conversion) won't appear in the vote tally; that's intentional (it measures MISTAKE coverage).

3. **Board-grounded judge** — `03_feature_labeling/judge_multi_family.py`
   Opus reads each good feature's boards + the tagger's FULL tag distribution and rules
   **covered / shallow_only / not_covered** (recall — does ANY tag name the concept, not top-1). This is
   the coverage deliverable. Output: `judge_<sae>.json`.

4. **Cluster the gaps** — group `not_covered` + `shallow_only` by concept (regex on Opus labels, or by
   hand). Build a detector ONLY for a cluster that is (a) coherent, (b) teachable, (c) real on the board.

---

## The three verdicts you'll get, and what each means

| verdict | meaning | action |
|---|---|---|
| **covered** | some tag names the concept | fine — but check if the TOP tag is right (discriminativeness) |
| **shallow_only** | a generic tag fires, specific concept missed (fork→knight-fork) | parametrize the existing detector |
| **not_covered** | no tag names it | either a MISSING detector (build) or a DIFFUSE feature (don't) |

**"not_covered" splits two ways — this is the key judgment:**
- **Coherent gap** → Opus verdict `good` + the boards share ONE teachable concept → build a detector.
- **Diffuse** → verdict `diffuse`, OR the boards are a grab-bag → do NOT build (naked-rate trap).

---

## THE LESSONS (what makes this analysis good vs bad)

### 1. Read the FENs, not the Opus summary. ← the biggest one (2026-07-14)
Opus's per-position `blunder_summary`/`tactical_motif` is a decent SIGNAL but NOT ground truth for
whether a *detector* is correct. Verifying Prophylaxis "looked fine" by summaries — but pulling 10 FENs
and playing them out revealed 29% prevented a CHECK and 19% were king moves (both = king-safety, not
prophylaxis). **Always dump `FEN | played | best-line | refutation-line | eval` and reason through the
chess yourself before trusting a detector.** Use `dump_feature_boards.py` / a quick verify script.

### 2. Judge a detector on the POSITIONS it fires on, not on SAE feature-counts.
"This tag tops only 2 good features" is circular — the SAE may just not have a clean feature for a real
concept (substrate limit). Prophylaxis IS a real concept even though ~0 SAE features are cleanly it.
Pull the detector's actual corpus fires and check if THOSE positions are the concept.

### 3. A tag is a CATCH-ALL when the features/positions it dominates span MULTIPLE Opus concepts.
Measure it: for each tag, cluster the Opus labels of what it tops. `Hung Material` topped 134 features /
15 concepts (57% actually hanging) — a catch-all. `Missed Fork` topped 4 / 1 concept — exact. Fix a
catch-all by finding the MECHANISM that pulls wrong features in (promotion=+8 material; SEE<0 played
capture; geometry-only overload) and gating THAT.

### 4. Over-fire rate is a smell, not a verdict. A real tactic fires 0.3–3% of the mistake corpus.
Anything ≥7% is almost certainly naked-rate (fires on "the engine wanted something else"). BUT: measure
WHY before gating. Overloading at 9.96% was a genuine bug (geometry-only, no material win → gate on
best-line winning ≥2). Pawn Break at 7-8% was NOT a bug (it fires on real breaks) — its high rate is real.

### 5. Gate on the CONCEPT, never on an arbitrary threshold (Sam's rule).
A "don't fire in close/decided games" rate cap is bad — it's arbitrary and drops real fires. A concept
gate is defensible: "prophylaxis prevents a NON-CAPTURE, NON-CHECK plan and the best move is a quiet
non-king move." Same result on rate, but every exclusion has a chess reason. If you can't name the
chess reason for an exclusion, it's the wrong gate.

### 6. Multi-tag is native and correct — don't force a single top tag.
A feature is a combo (Hung Queen AND King attack). `tagger_family_votes` carries all co-tags. A
"mislabel" often means the argmax picked a valid co-concept, not that a tag is wrong. Check the judge's
`alt_better` flag: it separates real masking (top tag is wrong) from valid-co-concept (both true).
A masker is only a *fixable bug* when the over-firing top tag has a correctness test it violates
(overloading → did it win material?). Otherwise it's an argmax/display preference — don't chase it
(precedence reweighting was measured and REJECTED: hurt at every threshold).

### 7. Diffuse ≠ "LLM couldn't find it." Often it's genuinely formless, OR it's an OUTCOME cluster.
575/796 diffuse features are diffuse even at the TIP (no concept). Many "structured diffuse" features
cluster by OUTCOME/severity (squandered a won game via 20 different moves) or by the "flat activation"
of a do-nothing move — SUBSTRATE artifacts, not chess concepts. Don't build move-tags for these;
describe them (see descriptive axes below) or leave them.

### 8. Every detector, before shipping: real-board TDD + corpus-overfire check.
Add cases to `04_tagger/regression.py` (POS from a real corpus board + NEGs for the exclusion classes).
Run `python3 scripts/04_tagger/regression.py` (must stay green). Measure the corpus fire rate. Then
`build_mistake_taxonomy.py`, then ship via `../chess-deck-code/backend/scripts/ship_tagger.py`.

### 9. Coaching MESSAGE matters as much as firing. Name the concrete thing, not jargon.
"Missed Prophylaxis: had a one-move threat" → useless. "Rd1 covers d4, preventing the opponent's d4" →
the player learns the actual plan. Evidence strings should name the concrete move/square/plan.

### 10. The delete-vs-gate-vs-keep discriminant (2026-07-15 audit — the single most useful rule).
Rank a tag's fires by **sole-explain %** (how often it's the ONLY explain tag on a move) and judge by
*mechanism type*:
- **DELETE** a FUZZY heuristic at **≤2% sole-explain.** Fuzzy = side-of-board proximity (`_side_attack`),
  `is_attacked_by`-without-SEE (ignored_threat, missed_defensive_resource), "moving piece is attacked"
  (desperado). At ≤2% sole it's redundant when right (a sharper tag names it) and wrong when unique
  (phantom: reading the sole cases, the opponent doesn't actually win the flagged thing). Killed this way:
  Allowed Battery, Kingside/Queenside Attack (×4), Ignored Threat, Missed Defensive Resource, Missed
  Desperado.
- **KEEP** a PRECISE motif even at **4-8% sole-explain**, even if it co-fires a sharper result tag. Precise
  = real pin/discovery/sacrifice/tempo-push geometry in the ACTUAL line. Multi-tag is native; the mechanism
  is a distinct lesson (Allowed Discovered Attack co-fires Hung Queen, but "you allowed a discovery" teaches
  something Hung Queen doesn't). Kept: Allowed Sacrifice (81% real material investment), Allowed Discovered
  Attack, Allowed Pin, Missed Tempo Push (11% sole), Pawn Move Exposed King (8% sole).
- **GATE** when a real concept has an incidental-fire class: name the chess reason, never a rate cap.
  Castling → best MOVE is the castle (not castling appearing later in the line). Open File → best move is
  a QUIET rook move (capture/check that lands on an open file is a tactic). Premature Trade → played
  capture SEE≥0 (a material-losing trade is greedy/sac's job). Outpost → first pov move quietly makes it.

### 11. "Best move IS a capture" ≠ "the lesson is material" — use SEE over the whole recapture line, not piece values.
`capture_or_exchange` mislabeled combination-initiating captures: Q takes a defended knight where the
recapturer is itself re-won nets +3 by SEE, but queen(9)>knight(3) made the value-heuristic call it a
"sacrifice" and stay silent (95 corpus positions with NO tag). Piece-value comparison can't see the
follow-up recapture; SEE can. Rule: SEE<0 = sacrifice, SEE≥1 = wins material (Missed Free X), SEE==0 =
even trade. Same lesson killed `missed_desperado`: 90% of its fires were SEE≥0 (plain winning captures,
not doomed-piece salvage). **Judge a material tag by SEE-over-the-line, not raw piece values.**

### 12. "First pov move IS the motif" gates the whole class of "motif appears somewhere in the line" catch-alls.
The motif detectors (`detect_line`) fired if the motif appeared ANYWHERE in the best/refutation line.
For a *positional* motif (castling, outpost) that means it fired on any line that eventually castles —
73% of Missed Castling had the best MOVE be something else, castling 2-4 plies deep. Gate: `povn[0]` (the
first pov move) IS the motif. Killed the incidental 73%/51% while keeping the real "you should have
castled/outposted NOW" residue. (Tactical motifs like fork/sac are fine deep — they ARE the punishment.)

### 13. "The SAE has a feature for X" ≠ "this detector is the right way to tag X." Verify against the corpus, not the labels.
Sam asked "aren't there kingside-attack SAE features?" — yes, 54. But all 54 were **covered by a sharper
tag** (Mate / Attacking Check / Greek Gift / Sacrifice) in their vote distributions. The concept was real
and SAE-visible, but it resolved to those sharper tags, not to a side-of-board heuristic. Deleting the
coarse motif lost no SAE-validated concept. Corollary: low per-feature `tagger_covered_frac` on a THEME
(passed-pawn/endgame-tempo, 22 low-covered features) looked like a gap, but running those endgames against
the corpus showed 95% coverage — the low SAE coverage was the hard multi-move TAIL, not an unnamed concept.
**A single hand-picked counterexample (or a direct corpus check) beats a feature-label impression.**

### 14. Validate a finder against a known-positive before trusting its emptiness. ← the battery lesson.
I "proved" battery isn't a real concept with a finder that required the battery's back piece to DIRECTLY
attack the target — impossible by definition (the front piece blocks the line). It matched 0 real batteries,
and I read that as "no batteries exist." Sam gave one FEN and it collapsed. Rebuilt with correct xray
geometry → 1.0% corpus, a genuine detector. **When proving "X isn't in the data," first confirm the
detector fires on a hand-picked X.** (See `feedback_validate_finder_before_proving_absence` in memory.)

### 15. Coverage is good (94%); the uncovered 6% is the ceiling, not a to-do list.
The "what are we missing" audit: 6% of real mistakes get no explain tag, ~91% are quiet best-moves. The ONE
nameable cluster was **development** (opening) → built `missed_development` (349 uncovered → real tag, 1.9%,
13% sole). Everything else uncovered is multi-move king/pawn tempo + diffuse middlegame subtlety — no static
single-position gate (why `missed_stalemate` has ~0 corpus residue; the eval knows, no rule can name it
without a tablebase/search). Don't manufacture tags for formless mass — describe it (info axes) or leave it.

---

## Descriptive axes (characterize a feature even when it's not a coaching lesson)

For diffuse/outcome features, we can't NAME a mistake type but we CAN describe them. Three info tags
(direction=info, Meta category) exist:
- `conversion_outcome` — result-band transition (Winning→Losing, Winning→Drawn, Even→Losing).
- `blunder_severity` — Sharp Blunder (win-drop≥30%) vs Slow Bleed (<15% AND balanced). SATURATION-guarded
  (a missed mate while +M5 is a tiny drop but NOT a bleed).
- `move_difficulty` — Only Good Move Missed (n_good_moves≤1) vs Careless Blunder (≥4). Needs `n_good_moves`
  (MultiPV≥6 depth-14, moves within ~100cp of best; cache at s3://.../sae/n_good_moves.json).

Aggregate these over a feature's top-100 → a **feature descriptor catalog**
(`output/sae_labels/feature_descriptor_catalog.json`): "what is this feature" on severity × conversion ×
difficulty, even for uncoachable ones.

---

## To run the audit over ALL top-firing features (the goal)

1. Ensure the SAE is labeled (step 1) + retagged (step 2) + judged (step 3). jr2048 already is
   (`output/sae_labels/`). Re-run only if the tagger changed materially since.
2. Pull the `not_covered` + `shallow_only` lists from `judge_<sae>.json`. Cluster by Opus concept.
3. For each cluster ≥~5 features: dump 10 FENs (step-1 lesson), read the boards, decide coherent-gap vs
   diffuse. Coherent → build (TDD + overfire + message + ship). Diffuse → note it, move on.
4. For `covered` features where the TOP tag ≠ concept: check `alt_better`. Real masker with a crisp
   correctness test → fix. Otherwise leave (multi-tag).
5. ALSO audit the loudest EXISTING tags for catch-all/over-fire: rank tags by corpus fire rate; anything
   ≥7% → read its fires (lesson 1), quantify the false-fire class, gate on the concept (lesson 5).
6. Record what you built + what you deliberately DIDN'T (with the reason) so it's not re-litigated.

## Substrate ceiling — what NO amount of this finds
The concept ceiling is the substrate (L7 best−blunder-diff mostly encodes "material changed hands").
Missing, and unreachable from single-position SAEs: **strategic/positional plans** (multi-move, ~7
features total) and **conversion/squander** (a game-trajectory property). Those need a different signal
(multi-layer activations, or trajectory metrics off the win% curve), not more tags or a bigger dict.
