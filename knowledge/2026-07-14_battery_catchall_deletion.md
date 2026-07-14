# Battery: deleted as a catch-all, then REBUILT correctly (2026-07-14)

> **OUTCOME (read this first):** `allowed_battery` stays **deleted** (it was a real 9%-corpus catch-all).
> `missed_battery` was deleted too — but that was **partly wrong**, and Sam caught it. My "battery isn't a
> real concept" proof used a **broken finder** (it required the battery's back piece to *directly* attack
> the target, which is impossible by definition — the front piece blocks the line). With correct **xray**
> geometry, battery IS a real, detectable concept: `missed_battery` was **rebuilt** (xray + quiet-move +
> defended-target gates), fires **1.0%** of the corpus (real-tactic band), 163/163 regression. The
> narrative below documents both the original (correct) catch-all finding AND the error + rebuild. See
> the "REBUILD" section at the bottom for the corrected detector; **the lessons from the mistake are the
> most valuable part of this doc.**

**What this is:** the audit that removed both battery detectors, plus the correction. A worked example of
lesson 3 (catch-all) + lesson 5 (gate/delete) — AND a cautionary tale about proving a negative with a
finder you didn't validate.

## The claim they made
A "battery" = two heavy/sliding pieces aligned on a file/diagonal (Q+R, Q+B, R+R) attacking a real target.
`missed_battery` fired when the best MOVE formed one; `allowed_battery` when the opponent's REFUTATION LINE
formed one the best move would have prevented. Both had already been "fixed" once (2026-07-11: allowed_battery
was narrowed from scanning all legal moves to walking only the refutation line). The rate stayed high.

## Why they're catch-alls (the evidence — all on chess-poc, 60k `sf_lines` mistake corpus, N=59598)

| Test | Allowed Battery | Missed Battery |
|---|---|---|
| Corpus fire rate | **9%** | **4%** | ← ≥7% = naked-rate smell (lesson 4) |
| SAE features it TOPS / distinct Opus concepts | 118 / **81** | 13 / 13 | ← catch-all signature (lesson 3) |
| Opus verdicts of those top features | 61 diffuse, 20 too_broad, 31 good | 6 diffuse, 2 too_broad, 5 good |
| Appears in vote tally on N of 2035 features | **903 (44%)** | 111 | ← fires on ~half of ALL features |
| Co-fires a SHARPER mistake tag | 76% | 74% | ← battery is mechanism, not the lesson |
| Battery is the ONLY explain tag | **4%** | **5%** | ← almost never the real story |

The Opus concepts the "battery" features actually encode: *Hanging Piece Left En Prise, Missed Tactical
Shot, Missed King-Attack Tactic* — **never "battery."** Two heavy pieces lining up on an enemy piece is a
geometric coincidence present in nearly every middlegame/refutation line; it rode along on whatever the real
mistake was.

## Read the FENs (lesson 1) — the deciding step
Dumped `FEN | played | best-line | refutation` for the fires, bucketed by refutation type:
- **allowed_battery**: 17% mate refutations (`Qxg2#`, `Rf1#`), 23% big material swings, 59% "positional."
  Read 10 of the *positional* ones by hand: **0/10 were a battery lesson** — they were missed material wins
  (Qxg5 wins a rook), hung pieces (Nxe5 → Qa5+ zwischenzug), or diffuse drift. The doubled rooks / aligned
  Q+R in the refutation were incidental.
- **missed_battery**: even after removing best-lines that mate or win ≥2 material, the residue was defensive
  best moves (Qd7, Be7) and forcing checks — the alignment was a side effect of the move's real purpose.

## Delete, not gate — the judgment (lesson 5)
Prophylaxis and pawn_break got concept GATES because after gating there was a real residue of teachable
positions. Battery has **no residue**: only 4-5% of fires had battery as the sole explain tag, and reading
those showed diffuse non-battery positions. If you can't name the chess residue a gate would preserve, the
gate is wrong — so this was a straight delete (matches "prefer removing over adding"). A battery IS a real
chess concept; this CORPUS just never presents it as the primary lesson (when a best move genuinely forms a
decisive battery it's already winning material / mating, and the sharper tag catches it).

## "Are we replacing battery?" — checked from BOTH directions, answer is no (2026-07-14)
Sam pushed on whether deletion left a coverage hole. Verified instead of asserted:

1. **Sole-explain gap (bottom-up).** Reconstructed the deleted detectors, found the fires where battery was
   the ONLY explain tag: 1,475 total, but **1,061 sub-threshold (wd<10%)**; only **414 real mistakes** now
   untagged. Read them — clustered by the missed (best) move: **75% "missed a quiet positional improvement"**,
   ~19% missed check, ~14% missed capture. **None are battery lessons** — battery had been labeling
   missed-quiet-move / missed-tactic positions with the wrong name. Deletion didn't lose coverage; it stopped
   MIS-coaching. (The 75% quiet-improvement bucket is the substrate ceiling, not a battery hole.)
2. **Tight battery finder (bottom-up, the strict test).** Quiet best move + converging under-defended
   Q+R/R+R/Q+B battery on a real target: **only 20 positions in all 60k.** 15 already carry a sharper tag
   (the attack/piece the battery wins); the 5 battery-only ones are ALL sub-threshold (wd 4-9%). Zero
   genuine battery-only mistakes clear the 10% line.
3. **SAE labels (top-down, the discovery instrument).** Grepped all **7,635 Opus-authored** feature labels
   across jr2048 + jr512: **"battery" appears 0 times** (also "doubled rooks" 0). Opus named the neighbors
   (Open File, File Control, Hanging Queen on Open File) but never a battery direction. If the concept lived
   in the activations, the discovery tool would have surfaced it.

**Why no residue:** a battery is a MULTI-MOVE buildup (stack now, break through later). Our signal is a
single-position best−blunder diff — it encodes the tactic a battery ENABLES, not the slow setup. Same
substrate ceiling as strategic/positional plans (`sae_feature_audit_playbook.md` §ceiling). A "battery v2"
detector would only re-catch the 20 and re-mislabel the 414 — strictly worse than no tag. **Not replacing it.**

## What changed
- `predicates.py`: removed `missed_battery`, `allowed_battery`, `_battery_hits_target`, `_BATTERY_TARGETS`;
  dropped both from `ALL_PREDICATES`. Left a removal note above `missed_overloading`.
- `tagger.py`: removed "battery" from `categorize()`'s tactical-motif branch; fixed the `_bare_motif`
  docstring example (was `'Allowed Battery' → 'Battery'`).
- `build_mistake_taxonomy.py`: removed the two Battery `add(...)` lines. 184 → **182 tags**.
- `regression.py`: retargeted the twin-collapse fixture from Battery to Doubled Rooks; removed the
  allowed_battery over-fire NEG case. **157 → 156 cases, all green.**
- Verified on chess-poc: functions absent, **0 Battery fires** in the corpus, loud board now tops out at
  Allowed Pawn Capture 11% (SOLID, separately audited).

---

# Missed Open File — GATED, not deleted (2026-07-14, same audit)

The contrast case that shows why battery was a delete and this is a gate.

**SAE picture (NOT a catch-all):** tops only **19 features** (vs battery's 118) and appears in the vote
tally on **7% of features** (vs battery's 44%). It's a real positional concept with two incidental-fire
bugs bolted on — not a formless catch-all.

**FEN read (lesson 1), bucketed by what the best move actually IS:**
- **CAPTURE (~11%)**: best move is `Rxc1` / `Rxe4` — a material tactic that happens to land the rook on an
  open file. "Open file" incidental; a material tag is the real lesson.
- **CHECK (~11%)**: best move is `Rd5+` / `Rg8+` — a forcing check. Endgames are full of open files, so
  nearly any rook check qualified. The lesson is the check, not the file.
- **QUIET (~77%)**: the genuine file-occupation concept (Rd1, Rfe8 to activate/pressure).

**The gate (lesson 5 — name the chess reason):** Missed Open File is a QUIET file-occupation lesson → the
best move must be neither a capture nor a check. Two lines in `missed_open_file`:
`if b.is_capture(bm): return []` and `if after_best.is_check(): return []`.

**Result:** raw **6.4% → 4.9%** (dropped ~867 incidental capture/check fires); the 4.9% residue is all
genuine quiet rook-to-open-file. Regression `159/159` (added POS quiet + NEG-capture + NEG-check, corpus
FENs). This is a GATE because there IS a real residue to keep — the exact judgment battery failed.

---

# Allowed Mate + Hung* — legit-loud, NOT buggy (2026-07-14, same audit)

The other outcome besides delete/gate: **loud but correct.** A high fire rate is a *smell*, not a verdict
(lesson 4) — some tags are loud because the mistake is genuinely common at 1800.

**Allowed Mate (7%, 4752 fires):** verified every fire either reaches checkmate in the refutation PV or has
a mate-score `eval_after`. **0% NO_MATE.** No hidden false-fire class. Ship as-is.

**Hung\* (Rook/Queen/Knight/Bishop/Material, ~15-17k fires across the family):** `hung_material` is by
design a **net-material-loss-over-the-refutation-line** detector (handles DELAYED hangs = loss realized a
few moves deep). So a 1-move SEE is the WRONG yardstick — 37% "no free 1-move capture" is expected because
many hangs are multi-move (`Qc8+ Qe8 Qxe8+` forces the queen). Reading the hardest bucket (the 37%) by
hand: ~11/12 were genuine multi-move hangs. Confirmed legit-loud (matches the detector's own "~48/52 clean"
note). Filed **issue #58** for the one edge: `peak_victim` naming can call an even rook trade "Hung Rook"
when the true net loss is a pawn (transient peak from a recaptured piece). Rare, low-priority — the peak
logic is correct for genuinely-hung-with-partial-compensation pieces; the fix must not break that.

## Wide-audit scorecard so far (loudest tags, mistake corpus N=59598)
| Tag | Rate | Verdict | Action |
|---|---|---|---|
| Allowed Pawn Capture | 11% | SOLID | none |
| Allowed Battery | 9% | catch-all | **deleted** |
| Allowed Mate | 7% | legit-loud | none |
| Missed Pawn Break | 6% | (prior session: gated) | — |
| Hung Rook/Q/N/B/Material | 4-6% | legit-loud | edge #58 filed |
| Missed Open File | 6.4→4.9% | real + 2 bugs | **gated** (no capture/check) |
| Missed Battery | 4% | catch-all | **deleted** |

Three outcomes seen: **delete** (no concept residue), **gate** (real concept + incidental bugs),
**leave** (loud but correct). Next: continue the playbook over mid-rate tags + the `not_covered` gap list.

---

# REBUILD — `missed_battery` restored with correct geometry (2026-07-14, same day)

Sam gave a concrete counterexample: `rnbqk1nr/pp3ppp/2p5/2bpP3/4P3/3B1P2/PPP3PP/RNBQK1NR b` — "I know this
allows a battery." After `...Qb6` Black has Bc5 (front) + Qb6 (back) stacked on the b6→f2 diagonal. That is
a textbook Q+B battery, and my "no real battery exists in the data" conclusion couldn't explain it.

## The error (this is the lesson)
My "tight battery finder" (the one that returned "20 in 60k, all junk") had this filter: *the back piece
must ALSO directly attack the target.* **That is impossible for a battery by definition** — the front
piece sits between the back piece and the target, blocking the back piece's line. A battery's whole point
is the *latent* second attacker (revealed only when the front piece moves/captures). So my finder silently
matched only **convergence** (two pieces hitting a square via different lines), never a real **stacked
battery**, and I used its empty result to "prove" battery isn't a concept. **I proved a negative with an
instrument that was structurally incapable of finding positives, and didn't validate it against a known-
positive first.**

## The fix — correct XRAY geometry
`_find_battery(board, target, color)`: find a front slider that directly attacks the target, then step
along the SAME line PAST the front piece; the first piece behind it, if a compatible slider (Q/R on
straight lines, Q/B on diagonals), is the back piece. That's the battery. `_new_battery_from_move` requires
the battery to be NEW (not pre-existing), the moved piece to be part of it, and the **target to be DEFENDED**
(an undefended target = plain hanging piece, not the battery lesson).

## Why it's a real detector now (not the old catch-all)
- **Corpus fire rate 1.0%** (was 9%). Squarely in the real-tactic band (0.3–3%, lesson 4).
- Characterized the 57 sole-lesson fires BEFORE writing the gate (lesson 1/2): **53/57 target a DEFENDED
  piece/pawn**, **48/57 are pure positional pressure** (best line never even captures the target). That is
  the concept: double a slider behind another onto a point the opponent defends only once.
- Read 12 fresh sole-lesson fires by hand: all genuine (Qb1→h7 Q+B battery, Qf6→f2, Qd5→f7, Bd6→h2). ✓
- Targets INCLUDE pawns (h7/g7/f7, h2/f2) — the canonical battery aims at a defended kingside pawn. I
  first over-restricted to pieces-only and it missed the Qd5→f7 case; added pawns back. King is excluded
  (a battery "on the king" is a check — the quiet-move gate already handles that).
- Gates (all concept-based): best move QUIET (no capture, no check), correct xray, defended target.
- `allowed_battery` stayed DELETED — the old refutation-line scan was the genuine 9% catch-all; no
  characterization pass has shown a real "opponent builds a battery against you" residue. Don't re-add on
  reflex. `missed_battery` re-added to `ALL_PREDICATES`, `categorize()`, taxonomy (183 tags). Regression
  163/163 (POS1 Bc5+Qb6, POS2 Bc4+Qd5, NEG capture, NEG lone-rook-no-back-piece).

## LESSONS (the durable takeaways — Sam asked to store these)
1. **Validate a finder against a known positive before trusting its emptiness.** A negative result only
   means something if the instrument can produce positives. I never fed my "tight finder" a position I
   KNEW was a battery — if I had, it would have returned nothing and exposed the bug immediately. **When
   proving "X doesn't exist in the data," first confirm your detector fires on a hand-picked X.**
2. **Know the concept's geometry before coding it.** A battery is a latent/xray stack, NOT two pieces both
   attacking a square. I coded the wrong mental model (convergence) and it looked plausible. Re-derive the
   definition from a real example, not from intuition.
3. **The SAE "no battery label" was a true fact used for a false conclusion.** Opus really never labels a
   battery feature — but that's because a battery is a MULTI-MOVE plan the single-position substrate can't
   encode (confirmed: the engine wants Nc3–Ne2–f4 buildup in Sam's POS2). "The SAE can't see it" ≠ "the
   rule tagger can't detect it." The tagger works on explicit geometry and CAN catch what the SAE misses.
   Don't let a substrate-ceiling result veto a geometry-based detector.
4. **"Prefer removing over adding" is right, but deletion still needs the same proof burden as a gate.** I
   was correct to delete the CATCH-ALL, but I over-generalized to "the concept is empty" without the
   evidence that claim required. Delete the broken detector; don't delete the concept unless you've looked
   for it correctly.
5. **Sam's counterexample > my corpus statistic.** One concrete FEN ("I know this is a battery") beat a
   60k-position scan, because the scan was instrumented wrong. A single hand-verified example is a
   powerful check on an automated finding — take it seriously, don't defend the aggregate.

## Scorecard correction
The row `Missed Battery | 4% | catch-all | deleted` above is **superseded**: Missed Battery is REBUILT at
1.0% (real). Only `Allowed Battery` remains deleted.
