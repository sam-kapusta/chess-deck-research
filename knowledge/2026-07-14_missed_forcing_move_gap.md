# Missed-forcing-move gap — closed 237→6 (2026-07-14)

**What this is:** the audit + two fixes that closed the "player missed a check or a material-winning
capture, and NO tag fired" coverage gap. Follow-on from the wide loud-tag audit
(`2026-07-14_battery_catchall_deletion.md`). Applied the battery lesson: re-derived the gap from scratch
(didn't trust an earlier rough "~57" estimate), read FENs, root-caused each detector before touching it.

## The gap
Positions where the best move is a **check** or a **material-winning capture (SEE≥1)**, the move is a real
mistake (win_drop≥10%), and **no explain tag fired**: **237 positions.** Split by classifying best-vs-played:
- `best=check, played=check` (100) — player checked, but a DIFFERENT check was best. ("wrong check")
- `best=capture, played=quiet` (95) — missed a material-winning capture.
- `best=cap+chk / quiet-check` (rest) — missed forcing moves of other shapes.

## Fix 1 — capture_or_exchange: gate on SEE, not piece values (Gap B captures)
**Root cause:** the defended-capture branch called a capture a "sacrifice" (`return []`) whenever the
attacker outvalued the victim by *piece value* (`VAL[att] > VAL[vic]+0.5`). But that heuristic can't see
the recapture sequence. `Qxf6` winning a defended knight where the recapturer is itself re-won nets **+3
by SEE**, yet queen(9)>knight(3) made it look like a queen sac → silent. 95 corpus mistakes had no tag.

**Fix:** decide the defended-capture branch by **static exchange eval**:
- `SEE < 0` → sacrifice (excluded, as before — protects the original Qxe4+ queen-for-bishop case, SEE −6).
- `SEE ≥ 1` → wins material → `Missed Free X` ("wins material (nets +N)").
- `SEE == 0` → even trade → `Missed X Exchange` (unchanged).

**Verified (60k):** 1420 newly firing (all SEE≥0), **0 newly silenced**. 109 of them are capture-mates
(`Qxg7#` etc.) — ALL co-fire a Mate tag (0 mislabeled as sole "Missed Free Pawn"), so multi-tag handles
them. Commit `7a6ef1c`. regression +2 (POS defended-capture-nets-material, NEG queen-for-bishop-sac).

## Fix 2 — new `wrong_check` detector (Gap A)
**Root cause:** `missed_attacking_check` deliberately excludes the played-also-check case
(`if b.gives_check(pm): return []`), so "you checked, but a DIFFERENT check was best" had no detector.

**Fix:** `wrong_check` — the third case of the check family:
| detector | played | best | lesson |
|---|---|---|---|
| `pointless_check` | a check | a quiet move | shouldn't have checked |
| `missed_attacking_check` | not a check | a check | should have checked |
| **`wrong_check`** | a check | a DIFFERENT check | right idea, wrong check |

Gates: both give check (bm≠pm), best is non-capture non-mate (Missed Mate / capture_or_exchange own those),
played is a non-capture check. Message names the concrete better check: *"you checked with Ra8+, but the
rook check Re8+ was stronger."* Teachable point: **calculate all candidate checks, don't fire the first.**
Category = Calculation. Only reaches past the win_drop gate, so the played check genuinely lost ≥10% win%.

**Verified (60k):** 1.42% fire rate (real-tactic band), 79 sole-explain positions (were untagged). Commit
`a9fb690`. regression +4 (2 real-corpus POS, NEG best-is-mate, NEG played-not-a-check).

## Result
Gap **237 → 6** (97% closed). The residual 6 are one-off edge cases (e.g. a capture-check that wins only
after a multi-move regroup; a wrong-check where the PLAYED move is itself a capture-check, excluded by the
material guard) — not a coherent pattern, not worth a detector. Noted here rather than chased (no silent cap).
The quiet-check half of the gap was already fully covered by `missed_attacking_check` (measured 0 residual).

## Lesson reinforced
Re-derived the gap from scratch instead of trusting the earlier "~57" number — good, because it was
actually 237 and split into two distinct concepts needing two different fixes. Read FENs + root-caused each
detector (which gate declines them) BEFORE editing. Same discipline that the battery episode forced.
regression 161 → 169 across both fixes; all green.
