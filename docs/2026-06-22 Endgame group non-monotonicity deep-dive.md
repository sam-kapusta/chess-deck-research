# Endgame group non-monotonicity — deep-dive (2026-06-22)

**TL;DR.** The Endgame skill-card group rises into the mid bands (peaks 7.21% at 1400-1600, above the
5.80% beginner rate) and breaks the rate→score inversion. The cause is **NOT** Missed Passed Pawn, and
**NOT** the endgame-moves denominator. The cause is **numerator contamination**: the production scoring
counts the endgame-**TYPE** orient/info tags (Rook Endgame, Pawn Endgame, Knight Endgame, Queen Endgame,
Bishop Endgame, Queen + Rook Endgame) as if they were endgame **mistakes**. These are board-context
classifiers that fire on *any* blunder that happens to occur in an endgame, regardless of whether an
endgame skill was the lesson. They make up **58.8% of the Endgame numerator** and their rate is hump-shaped
in rating, which is exactly what wobbles the curve. The task spec already says these should be excluded
from scoring; `categorize()` routes them to `"Endgame"` anyway via a substring match, so they leak in.

**Fix (one line):** exclude endgame-TYPE labels from the Endgame scoring group — count only the 8 EXPLAIN
labels. After the fix: beginner 3.25% → master 0.99% (3.3× fall-off), and the inversion is restored (the
peak band IS the beginner band, no mid-band exceeds it).

All numbers below are measured on the actual rebuilt corpus: 55,574 rapid blunders, current shipped tagger
(`~/SageMaker/tagger_run`), built exactly as `fifa_skill_ratings.py` builds the `Mistake` (white-POV evals,
no re-flip), 0 tagging errors. Scripts: `~/SageMaker/endgame_diag.py` (per-label + phase), `endgame_diag2.py`
(deduped EXPLAIN-vs-TYPE moment split). Raw outputs: `endgame_diag.json`, `endgame_diag2.json`,
`endgame_per_moment.json`.

---

## 1. The headline reproduced

Production Endgame rate = (moments where ANY Endgame-group label fired) / endgame-moves-reached, per band.

| band | Endgame moments | endmoves | rate % |
|---|---:|---:|---:|
| 600-800 | 445 | 7668 | **5.80** |
| 800-1000 | 666 | 10105 | 6.59 |
| 1000-1200 | 617 | 10233 | 6.03 |
| 1200-1400 | 762 | 11162 | 6.83 |
| 1400-1600 | 871 | 12082 | **7.21 ← peak** |
| 1600-1800 | 704 | 12914 | 5.45 |
| 1800-2000 | 778 | 12711 | 6.12 |
| 2000-2200 | 667 | 13607 | 4.90 |
| 2200-2400 | 695 | 14542 | 4.78 |
| 2400-2600 | 576 | 15192 | 3.79 |
| 2600-2800 | 453 | 17116 | 2.65 |

Matches `fifa_skill_ratings.json` to the digit. Peak (7.21%) is **above** the beginner rate (5.80%), so the
mid-band score = 0 under rate→score inversion. Confirmed problem.

---

## 2. Per-label decomposition — the stated prime suspect is innocent

Per-label rate (label fires / endmoves × 100) across the 11 bands, for every label that `categorize()`
routes to the Endgame group. Bold = peak band.

### EXPLAIN labels (the 8 that the spec says SHOULD be scored)

| label | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 | shape |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Missed Passed Pawn | **1.37** | 1.04 | 0.85 | 0.99 | 0.83 | 0.76 | 0.76 | 0.60 | 0.54 | 0.41 | 0.33 | **falls cleanly** |
| Missed King Activity | 0.42 | 0.41 | 0.42 | 0.34 | **0.54** | 0.36 | 0.36 | 0.29 | 0.36 | 0.24 | 0.19 | mild mid-bump |
| Wrong Pawn Race | 0.52 | 0.66 | 0.72 | 0.68 | **0.82** | 0.73 | 0.54 | 0.48 | 0.51 | 0.43 | 0.29 | mid-bump |
| Lost the Opposition | 0.01 | 0.02 | **0.08** | 0.03 | 0.04 | 0.05 | 0.02 | 0.01 | 0.01 | 0.00 | 0.01 | tiny (noise floor) |
| Rook Behind Passer | 0.07 | 0.16 | 0.09 | 0.15 | 0.10 | 0.15 | 0.08 | **0.18** | 0.12 | 0.07 | 0.05 | tiny (noise floor) |
| Allowed Promotion | 0.46 | **0.58** | 0.51 | 0.30 | 0.44 | 0.29 | 0.23 | 0.23 | 0.19 | 0.13 | 0.09 | falls (early peak) |
| Missed Promotion | 0.21 | **0.33** | 0.21 | 0.21 | 0.22 | 0.13 | 0.13 | 0.08 | 0.12 | 0.09 | 0.07 | falls (early peak) |
| Missed En Passant | **0.33** | 0.23 | 0.12 | 0.16 | 0.11 | 0.09 | 0.09 | 0.04 | 0.06 | 0.02 | 0.04 | falls |
| Allowed En Passant | **0.25** | 0.15 | 0.18 | 0.13 | 0.11 | 0.07 | 0.09 | 0.04 | 0.01 | 0.02 | 0.02 | falls |
| Missed Underpromotion | **0.12** | 0.04 | 0.03 | 0.02 | 0.05 | 0.03 | 0.02 | 0.00 | 0.03 | 0.02 | 0.01 | falls (noise floor) |
| Allowed Underpromotion | **0.10** | 0.03 | 0.05 | 0.04 | 0.03 | 0.02 | 0.03 | 0.01 | 0.03 | 0.01 | 0.02 | falls (noise floor) |

**Missed Passed Pawn falls cleanly from 1.37% to 0.33%.** It is the single largest EXPLAIN label but it does
**not** rise in the mid bands — so it cannot be the source of the mid-band hump. The prior knowledge-doc note
("main noise source — fires on crowded middlegames") is about *precision*, not about this monotonicity bug.

### TYPE / orient info tags (the ones the spec says to EXCLUDE — but production includes)

| label | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 | shape |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Rook Endgame | 1.43 | **2.54** | 2.01 | 2.39 | 2.15 | 1.49 | 1.82 | 1.57 | 1.48 | 1.18 | 0.75 | hump |
| Pawn Endgame | 0.30 | 0.30 | 0.40 | 1.02 | 1.13 | 1.01 | **1.34** | 0.80 | 0.77 | 0.53 | 0.36 | strong mid-hump |
| Knight Endgame | 0.07 | 0.42 | 0.22 | 0.40 | 0.46 | **0.57** | 0.41 | 0.41 | 0.39 | 0.28 | 0.18 | hump |
| Queen + Rook Endgame | 0.73 | 0.57 | **0.78** | 0.58 | 0.26 | 0.37 | 0.54 | 0.26 | 0.25 | 0.35 | 0.17 | wobble |
| Queen Endgame | 0.14 | 0.20 | 0.60 | 0.58 | **0.69** | 0.38 | 0.38 | 0.25 | 0.30 | 0.30 | 0.17 | mid-hump |
| Bishop Endgame (+variants) | ~0.4 | ~0.5 | ~0.3 | ~0.3 | **~0.9** | ~0.4 | ~0.5 | ~0.5 | ~0.5 | ~0.5 | ~0.5 | hump |

Every TYPE tag is hump-shaped (peaks in the 800–1800 region), not monotone. They dominate the numerator.

---

## 3. The decisive split: EXPLAIN vs TYPE moments (deduped, per band)

Counting **moments** (a blunder counted once if it has ≥1 label of each kind), so this is apples-to-apples
with the production numerator.

| band | PROD rate % (type+explain) | TYPE-tag rate % | **EXPLAIN-only rate %** |
|---|--:|--:|--:|
| 600-800 | 5.80 | 2.93 | **3.25** |
| 800-1000 | 6.59 | 4.21 | 3.25 |
| 1000-1200 | 6.03 | 4.20 | 2.67 |
| 1200-1400 | 6.83 | 5.26 | 2.41 |
| 1400-1600 | **7.21** | 5.07 | 3.01 |
| 1600-1800 | 5.45 | 4.01 | 2.25 |
| 1800-2000 | 6.12 | 4.75 | 2.08 |
| 2000-2200 | 4.90 | 3.51 | 1.94 |
| 2200-2400 | 4.78 | 3.36 | 1.95 |
| 2400-2600 | 3.79 | 2.76 | 1.45 |
| 2600-2800 | 2.65 | 1.95 | **0.99** |

Totals across all bands: **7,234** production Endgame moments, of which **5,120** carry a TYPE tag and only
**2,977** carry an EXPLAIN tag. **4,257 moments (58.8% of the numerator) are "TYPE-only"** — a blunder in an
endgame with NO endgame-technique tag at all (a hung piece / missed fork that merely *happened* in a rook
endgame). The TYPE-only rate is itself hump-shaped (peaks 4.42% at 1200-1400), and it is what drags the
production curve up in the middle.

**EXPLAIN-only is no longer inverted:** beginner 3.25% is the peak; master 0.99%; **3.3× fall-off**. No
mid-band exceeds the beginner band, so the rate→score inversion is restored and mid-band players no longer
score 0.

---

## 4. Ruling out the two stated hypotheses

**Hypothesis (b) — denominator-population mismatch (a tag fires OFF-endgame but is divided by
endgame-moves). DISPROVEN as the cause.**

- Phase audit of every Endgame-group label (FEN rule: endgame iff ≤12 pieces OR ≤4 non-pawn/non-king
  pieces). The TYPE tags are **0.0% off-endgame** by construction — they only fire in the endgame phase.
  The off-endgame firing is concentrated in a few EXPLAIN labels: **Missed Passed Pawn 57.5% non-endgame**
  (506/982 middlegame, 59 opening), En Passant ~89–91% non-endgame, Underpromotion ~59–70% non-endgame.
- But fixing that mismatch does **not** fix the curve. Counterfactual "gate the whole Endgame group to
  actual-endgame positions" still peaks at 1400-1600 (6.43%, still above the 3.87% beginner band):

  | band | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
  | EG-phase-gated rate % | 3.87 | 5.31 | 4.98 | 5.91 | **6.43** | 4.99 | 5.41 | 4.43 | 4.36 | 3.54 | 2.47 |

  Still inverted. So the denominator mismatch is real for Missed Passed Pawn (a precision problem) but is
  **not** the monotonicity bug.

**Hypothesis (a) — ONE noisy label (Missed Passed Pawn). DISPROVEN.** Removing Missed Passed Pawn entirely
makes the curve *slightly worse*, still peaking at 1400-1600:

  | band | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 | 2600 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
  | rate %, no-MPP | 4.66 | 5.87 | 5.47 | 6.22 | **6.76** | 4.99 | 5.61 | 4.49 | 4.48 | 3.53 | 2.50 |

  Because MPP's own rate falls with rating, dropping it removes a *down-sloping* component and the residual
  hump (the TYPE tags) shows through more, not less.

**Hypothesis (c) — genuine late-developing endgame skill. SMALL residual only.** After excluding TYPE tags,
EXPLAIN-only still has a minor secondary bump at 1400-1600 (2.41→3.01%), driven by **Wrong Pawn Race**
(peak 0.82%) and **Missed King Activity** (peak 0.54%). These are advanced techniques that plausibly *do*
develop late, so a small genuine effect is present — but it is second-order: it does not invert the curve
(the global peak is still the beginner band), so it does not break scoring.

---

## 5. Root cause

**(a-class, but it is a whole label CLASS, not one label, and it is a numerator-definition bug, not a
denominator bug.)**

`to_group()` puts a label in the Endgame scoring group whenever `categorize(label) == "Endgame"`.
`categorize()` returns `"Endgame"` for the endgame-TYPE orient tags too, via this branch
(`tagger.py:256`):

```python
if "endgame" in l or any(w in l for w in (... "passed pawn", "passer", ... )):
    return "Endgame"
```

The substring `"endgame"` catches "Rook Endgame", "Pawn Endgame", etc. — which are `info`-direction board
classifiers, ungated by win-drop, firing on every blunder that lands in an endgame phase regardless of the
mistake type. The spec ("the endgame-TYPE info tags are excluded from scoring") intends these to be orient
tags for drill-bucket filtering, **not** skill-rate inputs. But `fifa_skill_ratings.py` has no filter for
direction or for TYPE labels, so they enter the numerator and contribute 58.8% of it. Their hump-shaped
rating profile (more mid-rated players reach typed endgames and still blunder there for ordinary tactical
reasons) is what produces the mid-band rise.

This is a duplicate of the documented `categorize()` substring gotcha
(`2026-06-13-tagger-precision-and-endgame.md` §"categorize() endgame case") — same root, different
consequence: there it mis-routed *labels*, here it mis-counts *scores*.

---

## 6. Recommended fix

**Primary (resolves the non-monotonicity): exclude endgame-TYPE info tags from the Endgame scoring group.**
Score only the 8 EXPLAIN labels. Two equivalent ways to implement in `fifa_skill_ratings.py`:

- Filter by direction at the numerator: a tag only counts toward a group if `direction != "info"`. (Cleanest
  — also future-proofs against any other orient tag leaking into any group. The other 5 groups are unaffected
  because they have no info-direction members.)
- Or explicitly drop labels matching the TYPE set (anything ending in `"Endgame"`).

Result (measured): Endgame rate becomes beginner 3.25% → master 0.99%, a clean 3.3× fall-off; the peak is
the beginner band, so the inversion is restored and mid-band players stop scoring 0.

**Secondary (precision, not monotonicity): gate Missed Passed Pawn to the endgame phase.** It fires 57.5%
in non-endgame positions (crowded middlegames where the passer is incidental). It does not cause the bug,
but it mislabels middlegame pawn pushes as an endgame skill. Add the `_is_endgame(m)` gate that every
sibling endgame detector already has (`missed_king_activity`, `rook_behind_passer`, `wrong_pawn_race` all
call it; `missed_passed_pawn` is the only one that doesn't — predicates.py:457, "No phase gate"). Same
applies to En Passant (~90% off-endgame) and Underpromotion if they are kept in the group.

**Do NOT** drop Missed Passed Pawn (it is the largest legitimate EXPLAIN label and falls cleanly with rating)
and do NOT change the endgame-moves denominator (it is correct for EXPLAIN labels; the problem was always the
numerator).

**Residual to accept:** after the fix, a small mid-band bump remains in Wrong Pawn Race / Missed King
Activity. It is plausibly a real late-developing-skill effect and it does not invert the curve, so it needs
no action now. Flag for eyeball review if precision work continues.
