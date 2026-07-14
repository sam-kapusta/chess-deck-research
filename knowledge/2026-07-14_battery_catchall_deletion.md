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

## Follow-on
Next unaudited loud tag: **Missed Open File (4%)** — same suspicion (does the rook move actually seize a
useful open file, or is "open file" incidental?). Then spot-check Allowed Mate (7%) + Hung* (4-6%) are
legit-loud not buggy.
