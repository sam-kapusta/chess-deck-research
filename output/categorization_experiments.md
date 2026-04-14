# Blunder SAE Categorization Experiments (2026-04-14)

## Setup
- SAE: 2048 k=32 move-token blunder SAE
- Data: 10K blunder positions (200K available)
- Labels: `labels_blunder_mt_k32.json` (1,630 quality features)

---

## Experiment 1: Binary category overlap
**Hypothesis:** Sonnet's 22 categories separate positions into distinct groups.
**Test:** For each category, check what % of positions have ANY feature from that category firing.
**Result:** FAILED. Every category fires on nearly every position:
- hanging_pieces: 99.6%, deflection: 98.0%, multiple_threats: 95.6%
- Zero positions are exclusive to one category.
**Interpretation:** With hundreds of features per category, at least one fires on almost everything. Binary fire is not discriminating.

## Experiment 2: Activation strength dominance  
**Hypothesis:** Categories separate by activation STRENGTH, not binary presence.
**Test:** For each position, which category has the strongest total activation?
**Result:** PARTIALLY CONFIRMED. Categories do separate:
- hanging_pieces: 34%, deflection: 27%, multiple_threats: 13%, king_attack: 9%, endgame: 9%
- But top category only captures 37% of activation (mean). 86% of positions have NO majority-dominant category.
**Interpretation:** Blunders are genuinely multi-category. One category is primary but not dominant.

## Experiment 3: Top-3 category combinations
**Hypothesis:** Certain category combinations define blunder archetypes.
**Test:** For each position, rank categories by total activation. Count top-3 combos.
**Result:** Top-3 combos are almost all permutations of {hanging_pieces, deflection, multiple_threats}:
- 7.8%: hanging > deflection > multiple_threats
- 6.2%: deflection > multiple_threats > hanging
- 6.0%: deflection > hanging > multiple_threats
**Interpretation:** These three categories aren't separate coaching concepts for tactics. They're three names for "missed a tactic."

## Experiment 4: Blunder severity
**Hypothesis:** Bigger blunders activate different feature patterns.
**Test:** Split by cp_loss: mild (<300), medium (300-700), severe (>700). Compare primary categories.
**Result:** CONFIRMED. Severity shifts the distribution:
- Mild: hanging (28%) + deflection (27%) — basic tactical errors
- Medium: hanging jumps to 37% — more blatant oversights
- Severe: endgame_technique becomes #1 at 27% — worst blunders are endgame mistakes
**Interpretation:** Severity is a meaningful axis. Could separate "basic safety" from "advanced technique" coaching.

## Experiment 5: Game phase
**Hypothesis:** Features separate by game phase (opening/middlegame/endgame).
**Test:** Classify positions by piece count. Check primary category per phase.
**Result:** STRONG SEPARATION:
- Opening: 41% hanging_pieces — development errors
- Middlegame: 32% deflection — overloaded pieces
- Endgame: 34% endgame_technique + 19% passed_pawns — technique failures
**Interpretation:** Phase is the strongest natural separator for coaching categories.

## Experiment 6: Phase-specific vs phase-neutral features
**Hypothesis:** Some features fire almost exclusively in one phase.
**Test:** For each feature, compute % of fires in each phase. Flag >80% as phase-specific.
**Result:**
- 174 endgame-specific features (>80% endgame)
- 134 opening-specific features (>80% opening)
- 52 middlegame-specific features (>80% middlegame)
- 191 phase-neutral features (15-55% each phase)
- ~1,079 partially phase-specific
**Interpretation:** ~360 features are strongly phase-specific. 191 are truly generic. The rest are in between.

## Experiment 7: Hierarchical clustering (25 groups)
**Hypothesis:** Fire-pattern clustering produces clean coaching categories.
**Test:** Cosine distance + Ward hierarchical clustering, cut at 25.
**Result:** PARTIALLY CONFIRMED:
- Endgame clusters are tight (Jaccard 0.2-0.3, 75-90% category-pure)
- Tactical clusters are loose (Jaccard 0.02-0.04, 20-30% category-pure)
- One mega-cluster of 769 features (38% of all)
**Interpretation:** Endgames cluster, tactics don't.

## Experiment 8: Louvain community detection
**Hypothesis:** Graph-based community detection handles fuzzy boundaries better.
**Test:** Weighted graph (Jaccard edges ≥0.05) + Louvain.
**Result:** Similar to hierarchical:
- 9 real communities, ~60 singletons
- Endgame communities: 30-69 features, Jaccard 0.13-0.22 — tight
- Tactical communities: 111-469 features, Jaccard 0.005-0.03 — loose blobs
**Interpretation:** Same pattern. The algorithm doesn't matter — endgames separate, tactics don't.

## Experiment 9: Clique-based grouping (Jaccard ≥0.3)
**Hypothesis:** Strict cliques (all pairs ≥0.3) find tight coaching groups.
**Test:** Greedy clique finding with minimum all-pairs Jaccard 0.3.
**Result:** FAILED as a grouping method:
- 75 groups of 2+ features, but 1,410 singletons (87%)
- The threshold is too strict — almost nothing qualifies.
**Interpretation:** 0.3 clique requirement is too harsh for the tactical features.

## Experiment 10: Greedy set cover
**Hypothesis:** A minimum set of features can cover all blunder positions.
**Test:** Greedily pick feature covering most uncovered positions.
**Result:** 22 features cover 95%. But top features fire 15-20% — too broad.
- Top 10 = 31.5% coverage, top 25 = 48.5%, top 50 = 60.5%
**Interpretation:** Coverage ≠ coaching value. Broad features cover more but teach less.

## Experiment 11: Top-1 feature analysis
**Hypothesis:** Individual features are more discriminating than categories.
**Test:** For each position, which single feature has the strongest activation?
**Result:**
- 880/2048 features are ever the #1 strongest (57% never primary)
- Top 25 features cover 49% of positions as primary
- But top-1 carries only 9.7% of total activation — blunders are flat across features
**Interpretation:** No single feature dominates any position. The 32 active features per position are roughly equal weight.

## Experiment 12: Phase-specific clustering (PENDING)
**Hypothesis:** Endgame-specific features cluster cleanly, phase-neutral don't.
**Prediction:** Endgame cluster mean Jaccard >0.15, phase-neutral <0.05.
**Test:** Split by phase, cluster separately. Script: `phase_cluster_test.py`
**Result:** (running)

---

## Emerging picture
1. **Phase is the primary axis** for organizing blunder coaching
2. **Endgame features** form clean, specific coaching topics
3. **Tactical features** overlap heavily — "hanging piece" and "overloaded defender" fire on the same positions
4. **Severity correlates with phase** — severe blunders are disproportionately endgame mistakes
5. **Categories work by activation strength**, not binary fire — primary category is meaningful, secondary isn't
6. **~880 features are ever primary** — the other 1,168 add context but never dominate
