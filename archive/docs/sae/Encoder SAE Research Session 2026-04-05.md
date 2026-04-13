# Encoder SAE Research Session — 2026-04-05

## What we set out to do
Figure out which encoder SAE features are useful for coaching, label them, and wire them into production.

## What actually happened
We invented several new evaluation metrics, ran a full architecture sweep, disproved our own hypotheses, and ended up with a much clearer picture of what SAE features can and can't do.

---

## Key Findings

### 1. The Agreement Test (new metric)
For each position, check: does a feature fire on BOTH the played move and the best move, or only one?
- **High agreement (>80%)** = positional feature (describes the board regardless of move)
- **Low agreement (<20%)** = move-specific feature (fires on one move but not the other)
- **Middle (20-80%)** = noisy / polysemantic

**Result across 17,532 positions:** 83% of features are move-dependent. Only ~5% are purely positional. The encoder fundamentally encodes (position + move) pairs, so the SAE learns move qualities, not position types.

### 2. Architecture Sweep Results

**BTK is the only viable architecture.** V1 and Gated produce features that fire on everything (1700+ noise features).

| Config | Features | PureMov% | Clean% | Coaching Signals | Noise |
|--------|----------|----------|--------|-----------------|-------|
| btk_1024_k8 | 133 | 58% | 59% | 57 | 0 |
| btk_1024_k16 | 218 | 63% | 66% | 75 | 0 |
| **btk_1024_k32** | **319** | **70%** | **73%** | **105** | **0** |
| btk_2048_k16 | 258 | 66% | 68% | 93 | 0 |
| btk_2048_k32 | 395 | 69% | 71% | 136 | 0 |
| btk_2048_k64 | 503 | 68% | 70% | 152 | 1 |
| btk_4096_k32 | 432 | 68% | 70% | 131 | 0 |
| **btk_4096_k64** | **532** | **69%** | **71%** | **164** | **2** |
| v1_2048 | 1963 | 12% | 93%* | 74 | **1698** |
| gated_2048 | 2048 | 0% | 88%* | 4 | **2016** |

*V1/Gated "clean%" is misleading — nearly all features are noise.

**Key insight:** k/dict ratio of ~3% is optimal for clean features (1024×k=32 wins at 73%). But for max coaching signals, 4096×k=64 gives 164 signals vs 105.

Full results saved: `lab/chess/docs/sae/SAE Architecture Sweep Results.md`

### 3. Agreement Does NOT Predict Coaching Value
We tested whether "clean" features (pure move or pure position) are more useful for coaching than "messy" ones. **Result across 2,509 blunders:**

- Clean features: avg consistency 0.42, 40% high consistency
- Messy features: avg consistency 0.61, 62% high consistency

**Messy features were MORE consistent.** Our consistency metric was confounded by sample size — rare features appear consistent by chance, common features appear inconsistent by statistics.

### 4. F2012 Deep Dive — A Cautionary Tale
Spent hours investigating F2012 (14% fire rate, 30% agreement). Found:
- Not "quiet maneuvering" (34% counter-examples with active moves)
- Not positional (30% agreement = 70% move-dependent)
- Not captures, not checks, not any single chess concept
- **Conclusion:** Some SAE features don't map to human chess concepts. The encoder learned its own representations.

### 5. Features That Actually Work
From game 122008167175 (19 blunders), examining feature diffs:

**Clearly interpretable:**
- F77: queen checks (8/8 examples are queen checks)
- F307: opening pawn moves (8/8 are e4/d4/c3)
- F928: material captures (7/8 are captures)
- F1436: tactical forcing moves (4/8 checks, forks)
- F1002: brilliant endgame technique (6/6 brilliant)

**Not interpretable:**
- F2012: fires on everything, no pattern
- F245: error-prone moves (fires on mistakes but no clear concept)
- F1738: broad middlegame feature

**Coaching-useful features from blunder diffs:**
- Ply 16: Missed F77 (queen check was available)
- Ply 33: Missed F837 + F1538 (tactical forcing moves)
- Ply 49: Missed F1459 (trapped piece exploitation)
- Ply 61: Missed F556/F867/F915 (endgame technique)

### 6. Maia SAE is Complementary, Not Competing
Maia takes position only (no move input) → its SAE features are 100% positional by definition. Encoder SAE takes position + move → ~70% move features. They operate on different axes entirely.

- **Maia SAE:** "This position has back-rank weakness"
- **Encoder SAE:** "The best move was a tactical check, your move was a quiet retreat"

### 7. Coherence Test (new metric)
For each feature, take the 50 positions where it fires strongest, compute pairwise cosine similarity.

**Result:** 45% of features have low coherence (<0.6). 29% have high coherence (>0.8).

**Surprise:** Features that are clearly "one concept" to humans (F77 queen checks) have LOW coherence (0.536) because queen checks happen in diverse positions. Features like F245 have HIGH coherence (0.914) despite being noisy — they fire on similar-looking positions even though the "concept" isn't clear.

**Implication:** Coherence measures positional similarity, not conceptual clarity. It's a different axis from agreement.

### 8. Why So Many Dead Features?
With 2048 dict and k=32: 728 alive, 1320 dead. The encoder's representation doesn't have 2048 distinct directions. It has ~500-700 real patterns. More dict size = more dead features (4096 has 3535 dead). This is normal for SAEs.

---

## What's Running Now
1. **Coherence sweep** — coherence metric across all 14 SAE checkpoints
2. **Temporal + polysemanticity sweep** — stickiness and decoder correlation across all checkpoints
3. **Lichess-scale profiler** — 100K Lichess positions with multi-PV evals → SAE features → Sonnet labels

## What's Next
1. **Finish labeling** — Sonnet 4.6 labeling on Lichess data (100K positions instead of 500 games)
2. **Coaching A/B test** — Take 50 blunders, generate coaching with and without SAE feature diffs
3. **Pick final config** — Combine agreement + coherence + temporal + coaching test to choose winner
4. **Train production SAE** — Retrain winner config on 150K puzzles (or explore adding blunders to training data)

## Metrics Inventory

| Metric | What it measures | What it's good for | What it's NOT good for |
|--------|-----------------|-------------------|----------------------|
| **Agreement** | Position vs move | Filtering architecture (BTK vs V1) | Predicting coaching value |
| **Coherence** | Position similarity | Finding position-type features | Finding move-type features |
| **Temporal** | Stickiness across moves | Distinguishing structural vs tactical | TBD |
| **Polysemanticity** | Decoder correlation | Finding pure vs mixed features | TBD |
| **T1 structural** | Dead/alive, L0, redundancy | Confirming SAE health | Choosing between healthy SAEs |
| **T2 concentration** | Phase/classification HHI | Finding phase-specific features | Didn't differentiate configs |
| **Fire rate** | How often feature activates | Filtering noise (>40%) | Everything else |
| **Coaching test** | Does it help Claude coach? | THE REAL TEST | Not built yet |

## Disproven Hypotheses
1. ~~Agreement predicts coaching value~~ — messy features are more consistent in coaching diffs
2. ~~F2012 = quiet positional maneuvering~~ — 34% counter-examples, fires on captures too
3. ~~Clean features are good, messy ones are bad~~ — no correlation in the direction we expected
4. ~~k=16 is optimal~~ — k=32 produces more features at similar quality
5. ~~Coherence = interpretability~~ — queen checks have low coherence, noise has high coherence

## Confirmed Findings
1. **BTK is the only viable architecture** — V1/Gated produce noise
2. **The encoder learns move qualities, not position types** — 83% move-dependent
3. **Maia SAE is complementary** — purely positional, different axis
4. **Some features clearly map to chess concepts** — F77, F307, F928, F1436
5. **Feature diffs between played/best moves contain coaching signal** — blunder analysis confirms
6. **No single metric predicts coaching value** — need the actual coaching test
