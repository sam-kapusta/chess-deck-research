# Battery detectors deleted — naked-rate catch-alls (2026-07-14)

**What this is:** the audit + decision that removed `missed_battery` and `allowed_battery` from the rule
tagger. First "delete" outcome of the wide SAE-tagger audit (`sae_feature_audit_playbook.md`). A worked
example of lesson 3 (catch-all) + lesson 5 (gate on concept, or delete if there's no concept residue).

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
