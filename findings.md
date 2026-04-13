# Chess Lab — Findings

## Maia SAE (production)

### Concept Labels
799/2048 features labeled (39%) via point-biserial correlation with 54 concepts across 10K positions. Strongest: F253 (fianchetto, 0.48), F1205 (queenside castling, 0.44), F754 (open files, 0.42).

### Rating Gradient (1100→1900)
295 features increase with rating, 387 decrease. Higher-rated Maia activates more features total: 698→748 fire >1%.

| Rating | Sees MORE | Sees LESS |
|--------|-----------|-----------|
| 1100→1900 | Pawn shield, passed pawns, rook on 7th, endgame patterns | Uncertainty/tension, crisis signals, material fixation |

**Coaching implication:** Rating progression = reactive → structural. "Stop counting material, start evaluating piece activity."

### Tag Correlations
SAE features detect coaching tags. Top: `undeveloped_pieces` r=0.65, `passive_rook` r=0.34, `weakened_pawn_shield` r=0.33. F1438 is a "missed tactic" super-feature (fires on quiet_when_winning, missed_check, missed_overloaded_piece simultaneously).

**Key insight:** Tags say WHAT went wrong. SAE features say WHAT KIND of position. Together: "You missed a check (tag) in a position with tactical opportunities (F1438)."

### k=32 wins for coaching — OVERTURNED then PARTIALLY RESTORED (2026-04-12)
~~k=128 labels more features (60% vs 38%) but 6.2% fire rate is too broad.~~ Without aux loss, k=32 is too sparse (57% dead, 35% FAILED). **With aux loss, k=32 is viable again** (9% dead, 1,864 active). Whether k=32+aux produces better labels than k=64+aux is untested — labeling comparison in progress.

---

## Detection Scoring & K-Sweep (2026-04-11)

### T3b Detection Scoring — production labels (puzzle_2048_k32_v1)

**Method:** Per feature, 15 positive FENs + 15 negative FENs → Haiku judges "does label match?" → balanced accuracy. Adapted from Sandstone T3b framework.

**Overall: Mean BA = 0.650** (Sandstone baseline = 0.690)

| Tier | Count | % |
|------|-------|---|
| HOLDS (BA > 0.75) | 89 | 23% |
| WEAK (0.60-0.75) | 169 | 43% |
| FAILED (BA < 0.60) | 137 | 35% |

**By category (best → worst):**

| Category | Avg BA | n |
|----------|--------|---|
| checkmate | 0.801 | 6 |
| endgame_technique | 0.753 | 95 |
| passed_pawn | 0.715 | 20 |
| quiet_moves | 0.518 | 12 |
| fork | 0.554 | 5 |
| captures | 0.563 | 31 |

**Fire rate correlation:** Spearman r=-0.166 (p=0.0009). Higher fire rate = worse detection. 5-10% fire rate features: avg BA=0.609, 6/10 FAILED.

**Confidence tracks detection:** high=0.710, medium=0.596, low=0.600.

**Script:** `research/scripts/detection_scoring.py`

### Contrastive relabeling — negligible improvement

Relabeled 169 WEAK features using contrastive prompts (positive + negative FENs). Result: **+0.005 mean BA** on relabeled features. 78 improved, 66 degraded. Not a labeling problem — polysemantic features can't be captured by any single label.

### K-sweep with c_dec proxy ("Sparse but Wrong" paper)

**c_dec** = mean |cos(d_i, d_j)| across decoder weight pairs. Minimized at the "correct" L0 per Chanin & Garriga-Alonso 2025.

**Result: c_dec decreases monotonically — no minimum found up to k=256.**

| k | c_dec | dead | alive | L0 | FVU |
|---|-------|------|-------|-----|-----|
| 8 | 0.085 | 1427 | 621 | 8.0 | 0.186 |
| 16 | 0.066 | 1352 | 696 | 16.0 | 0.148 |
| **32 (current)** | **0.052** | **1161** | **887** | **32.0** | **0.112** |
| 64 | 0.042 | 899 | 1149 | 64.0 | 0.084 |
| 128 | 0.035 | 345 | 1703 | 128.0 | 0.060 |
| 256 | 0.031 | 15 | 2033 | 255.1 | 0.045 |

**Implication (without aux loss):** k=32 is too sparse — 57% dead. But this is fixed by aux loss (see below): k=32+aux → 9% dead, 1,864 active. The k-sweep above was run WITHOUT aux loss.

**Script:** `research/encoder/scripts/sweep_k_cdec.py`

### BTK + aux loss sweep (2026-04-12)

Trained on 200K puzzles, 5 epochs, aux_coeff=1/32, dead_threshold=50. Full results in "Aux loss fixes k=32" section below.

### FEN enrichment improves detection scoring (2026-04-12)

Added Stockfish eval + python-chess tactical annotations to detection FENs. Same Haiku judge, same features — only the FEN context changed.

| Condition | Mean BA | STRONG (≥0.8) | HOLDS (≥0.7) | WEAK+ (≥0.6) | Top-200 |
|-----------|---------|---------------|--------------|--------------|---------|
| Haiku + raw FENs | 0.571 | 148 | 379 | 813 | 0.841 |
| Sonnet + raw FENs | 0.577 | 178 | 375 | 834 | 0.854 |
| Haiku + enriched FENs | **0.619** | **289** | **644** | **1135** | **0.883** |

**Enrichment impact by category (biggest gains):**

| Category | Raw | Enriched | Delta |
|----------|-----|----------|-------|
| back_rank | 0.574 | 0.732 | +0.158 |
| deflection | 0.511 | 0.651 | +0.140 |
| captures | 0.478 | 0.617 | +0.139 |
| checkmate | 0.589 | 0.689 | +0.100 |
| fork | 0.557 | 0.624 | +0.067 |

**Conclusion:** Enrichment matters more than judge quality. Haiku+enriched beats Sonnet+raw on every metric.

### Polysemantic metric is invalid (2026-04-12)

Phase/piece diversity (entropy of phase distribution + piece distribution) was used to flag polysemantic features. Validation: took 20 features with HIGHEST poly diversity scores, asked Sonnet to read all 15 examples and judge monosemantic vs polysemantic.

**Result: 19/20 (95%) are monosemantic.** The metric measures *generality* (fires across phases/pieces) not *polysemanticity* (fires on unrelated concepts). A feature detecting "forks" fires across all phases — high diversity — but is one clean concept.

The other session's claim of "60% polysemantic" is likely "60% general." Real polysemanticity is ~5%.

**Data:** `research/output/k64_baseline/`

### Sonnet+thinking labels > Haiku labels (2026-04-12)

Same features, same enriched FENs, same Haiku judge — only the labels changed.

| Labels | Mean BA | Top-200 | HOLDS | STRONG | FAIL |
|--------|---------|---------|-------|--------|------|
| Haiku | 0.619 | 0.883 | 644 | 289 | 360 |
| **Sonnet+thinking** | **0.632** | **0.886** | **659** | **325** | **293** |

Delta: +0.013 BA, +36 STRONG, -67 FAIL. Sonnet labels are more specific ("Forced checkmate delivery" vs "Standard d4 opening responses") and more detectable.

**Polysemantic audit:** 572/1,872 (30.6%) flagged polysemantic. But 486/572 are medium confidence, only 7 high — poly flag correlates with labeling uncertainty, not genuine polysemanticity.

**Job:** `pztzjp2jzh8v` (labeling), `ac6bc19768ax` (detection scoring)

### Aux loss fixes k=32 dead features (2026-04-12)

| Config | Aux | Dead | Active | FVU | c_dec | L0 |
|--------|-----|------|--------|-----|-------|----|
| 2048 k=32 | no | 1,161 (57%) | 887 | 0.112 | 0.052 | 32 |
| **2048 k=32** | **yes** | **184 (9%)** | **1,864** | **0.128** | **0.045** | **32** |
| 2048 k=64 | yes | 213 (10%) | 1,835 | ~0.082 | 0.036 | 64 |
| **4096 k=32** | **yes** | **1,188 (29%)** | **2,908** | **0.126** | **0.041** | **32** |
| 4096 k=64 | yes | 1,079 (26%) | 3,017 | 0.092 | 0.035 | 64 |

Key tradeoffs:
- k=32 vs k=64: similar active counts at 2048 (~1,850). k=32 is more selective (L0=32) but worse reconstruction (FVU 0.128 vs 0.082). Question: does selectivity produce better labels?
- 2048 vs 4096: 4096 has ~60% more active features. Question: new concepts or thinner splits?
- Dead features aren't waste — they're unused dictionary capacity. 4096 with 29% dead = 2,908 active > 2048 with 9% dead = 1,864 active.
### 4-variant comparison — 2048 k=64 + aux wins (2026-04-12)

| Config | Mean BA | Top-200 | HOLDS | STRONG | FAIL | Poly% |
|--------|---------|---------|-------|--------|------|-------|
| **2048 k=64** | **0.632** | **0.886** | **659** | **325** | **293** | 30.6% |
| 4096 k=64 | 0.566 | 0.824 | 566 | 159 | 824 | 3.6% |
| 4096 k=32 | 0.563 | 0.829 | 537 | 155 | 854 | 3.3% |
| 2048 k=32 | 0.557 | 0.776 | 284 | 70 | 515 | 3.7% |

k=64 wins despite higher polysemantic rate. k=32 features are less polysemantic but less detectable.
2048 > 4096 per-feature quality. Extra dict capacity produces more features but each one is worse.

**Detection scoring job:** `wma1a33zjoze` (8,580 features, Haiku + enriched FENs)
**Full comparison:** `chess-deck-research/output/COMPARISON.md`

### k=32 produces dramatically less polysemantic labels than k=64 (2026-04-12)

| Config | Features | Poly% | High conf | Med | Low |
|--------|----------|-------|-----------|-----|-----|
| 2048 k=32 + aux | 1,906 | **3.7%** | 1,022 | 657 | 227 |
| 2048 k=64 + aux | 1,872 | 30.6% | 1,146 | 647 | 79 |
| 4096 k=32 + aux | 3,270 | **3.3%** | 1,844 | 1,061 | 365 |
| 4096 k=64 + aux | 3,404 | **3.6%** | 1,883 | 1,084 | 437 |

k=32 features fire on 32 positions per input vs 64. Each feature has to be more selective about when it fires → less concept mixing → lower polysemantic rate. This is consistent across dict sizes.

**Jobs:** `9tve7y1jz72h` (2048 k=32), `jd44898kiujk` (4096 both)

---

## Encoder SAE — Per-Token Puzzle SAE (active research)

### Mean-pooled SAE was a dead end
Fork AUC 0.825 is misleading — manual check showed 5/10 non-fork puzzles also have forks. Mean-pooling destroys spatial info. 18 configs tried, none coaching-useful.

### Per-token SAE works
Spatial grounding confirmed: features fire on specific squares (8th rank for back rank threats, queen square for attacks). The fix for mean-pooling.

### Correct training: puzzles only, correct move, per-token, BatchTopK
K-sweep (150K puzzles):

| Config | Alive | Loss | Themes |
|--------|-------|------|--------|
| k=4 | 190/2048 | 0.237 | 47 |
| k=16 | 477/2048 | 0.136 | 47 |
| **k=32** | **728/2048** | **0.104** | **52** |

### Game comparison — coaching-useful features found
Tested k=32 on game 166660084296:
- **F2012** fires on both blunders (Kd1??, f3), not on good moves. "Passive when you should attack."
- **F165** fires on Bg4+ — fork detection.
- **F492** fires on Qxg8 — discovered check detection.

### 502-game analysis — behavioral fingerprints
29,891 moves analyzed. Puzzle-vs-game divergence reveals real patterns:

| Feature | Puzzle rate | Game rate | Signal |
|---------|------------|-----------|--------|
| F2012 | 4.9% | 14.3% | Player plays too defensively (3x divergence) |
| F492 | 30.3% | 4.8% | Player misses tactical checks |
| F1459 | 33.1% | 3.6% | Player misses trapped pieces |

Noise features (F1790 at 51%, F1915 at 35%) dominate raw counts — filtered.

### Labeling: solved (2026-04-06)

**Previous attempts (all failed):** Theme stats → Haiku, puzzle FENs → Haiku/Sonnet, game diffs → Haiku/Sonnet. All produced garbage or vague labels.

**Root cause:** Two bugs stacked. (1) Sonnet never received FENs — only stats. (2) Sonnet outputs `**LABEL:**` with markdown bold, parser couldn't extract labels. Every previous attempt returned "unclear."

**Fix:** Give Sonnet FENs + SAN + evals + cp_loss + puzzle themes. Strip `**` before parsing. Canonical script: `research/scripts/label_sae_features.py`.

**Result (puzzle-trained encoder SAE):** 394 features labeled. 44 high, 238 medium, 86 low, 26 unclear. **72% confident** with real chess concepts: zwischenzug, queen fork, capturing hanging pieces, zugzwang, back rank mate, pawn breaks. Strong endgame bias from puzzle training data.

### Training data comparison (2026-04-06) — puzzles >> blunders (REVISITED 2026-04-13)

| SAE | Training Data | Features | Confident% | Best Labels |
|-----|--------------|----------|-----------|-------------|
| **Encoder puzzle** | 150K correct moves | 394 | **72%** | Zwischenzug, forks, zugzwang |
| Encoder blunder (old) | 50K blunder moves, k=32 no-aux | 1922 | 27% | Vague — "near-equal moves", "drawn positions" |
| Maia blunder | 100K pre-blunder positions | 2048 | 45% | Meta — "positions where blunders occur" |

**Why puzzles won before:** k=32 no-aux had 57% dead features. Blunders with proper k=64+aux may be different.

### Blunder SAE v2 structural results (2026-04-13)

Trained BTK 2048 k=64 + aux on 200K blunder positions (≥200cp loss from Lichess eval dataset).

| Metric | Blunder SAE v2 | Puzzle SAE (winner) |
|--------|---------------|---------------------|
| Dead | 89 (4.3%) | 213 (10%) |
| Alive | 1,959 | 1,835 |
| L0 | 64.0 | 64.0 |
| FVU | 0.129 | ~0.082 |
| c_dec | 0.034 | 0.036 |

**Structurally promising.** Fewer dead features than puzzle SAE, more alive, similar decoder separation. Higher FVU expected — blunders are more diverse than puzzle solutions. Whether features are interpretable (the old failure mode) is being tested via profiling + labeling.

**CP loss distribution:** 53% ≥300cp, 22% ≥500cp, 8% ≥1000cp. Wide range of blunder severity.

**Decision:** Structural metrics pass. Profiling + Sonnet labeling needed to determine if features are interpretable.

### Corpus baseline pipeline (2026-04-08)

**Key insight:** Everyone makes ~6 mistakes per game regardless of rating. The difference is severity — 22.4% avg win% loss at 1400 vs 14.9% at 2200.

| Band | Games | Mistakes/game | Avg win% loss |
|------|-------|---------------|---------------|
| 1400-1600 | 1,000 | 5.9 | 22.4% |
| 1600-1800 | 1,000 | 6.0 | 20.6% |
| 1800-2000 | 1,000 | 6.2 | 19.0% |
| 2000-2200 | 1,000 | 6.2 | 17.1% |
| 2200+ | 1,000 | 5.8 | 14.9% |

**Category differentiation (1400 vs 2200, per-game rate):**
- 1400s worse: passed pawns (+8.3), endgame (+6.1), hanging pieces (+3.6), checkmate (+3.9)
- 2200s worse: king attack (-2.8), evaluation (-2.1), quiet moves (-1.8), check (-1.4)

**Critical process failure:** Three SAE extraction runs with wrong method before discovering the production Lambda uses move token (hidden[77]), not mean-pool or per-token-all. Root cause: never read the actual production code (`backend/lambda/sae_features/app.py`) before building the corpus pipeline. **LESSON: always read the production code first.**

### Diff SAE (best-blunder activation diff) — dead end (2026-04-06)
Trained SAE on encoder(best_move) - encoder(blunder_move). Hypothesis: "what the blunder missed" would cluster cleanly. Dict sweep 1024→16384, best at 8192 (76% alive). But labels were 16% confident — all tautological ("forcing move missed", "direct attack missed"). The diff space encodes "better move was better" which is useless.

**Correct approach:** Use puzzle SAE with two forward passes. Run SAE on encoder(best_move) and encoder(played_move) separately. The diff in *feature labels* tells you what was missed — "best move = fork, your move = retreat." No diff-trained SAE needed.

### Feature redundancy: near-zero (2026-04-06)
Jaccard overlap analysis across all categories. Features with similar labels fire on DIFFERENT positions — they're subtypes, not duplicates.

| Category | Features | Max Jaccard | Redundant pairs |
|----------|----------|-------------|-----------------|
| Fork | 26 | 0.289 | 0 |
| Hanging pieces | 26 | 0.291 | 0 |
| Zugzwang | 23 | 0.302 | 1 |
| Pin/skewer | 16 | 0.034 | 0 |
| Check | 13 | 0.122 | 0 |

### Subtype taxonomy: 191 features with specific labels (2026-04-06)
Re-prompted Sonnet with category context to distinguish subtypes. 15 categories, 191 refined labels (27 high, 101 medium confidence).

| Category | Subtypes | Examples |
|----------|----------|---------|
| Fork | 25 | Knight fork with check, queen fork, pawn fork, f7 sacrifice fork |
| Passed pawn | 25 | King escorting pawn, rook check escort, underpromotion, mutual race |
| Endgame technique | 25 | BN mate, two knights draw, opposition, bishop fortress |
| Hanging pieces | 22 | Capture with check, ignoring hang for stronger move, multiple hanging |
| Zugzwang | 21 | Triangulation, opposition, pawn breakthroughs, tempo moves |
| Checkmate | 14 | Smothered mate, sacrificial mating attack, back rank, BN coordination |
| King attack | 15 | Pawn storm, piece sacrifice on shelter, queen-led attack, uncastled king |
| Defense | 14 | Interposition, counterattack, king reaching key square, blockade |
| Pin/skewer | 10 | Check-first skewer, discovered skewer, trapped queen, pin exploitation |
| Check | 9 | Capture-check, perpetual check, battery check |
| Sacrifice | 6 | Attraction, deflection, pawn shelter destruction, speculative |
| Back rank | 2 | Check exploiting trapped king, piece landing with promotion |

Full output: `poc/output/lichess_subtype_labels.json`

**Reframe still holds:** Labels are neutral descriptors. Coaching interpretation happens at inference via feature diffs (best move vs played move). SAE describes, Claude coaches.

See: `website/research/SAE Labeling — Descriptive Not Prescriptive.md`

### The encoder sees multi-move tactics that tags can't
Kd1?? allows a 4-move forcing sequence ending in a fork. Tags check one move deep and miss it. The encoder's activations differentiate Kd1 from the best move — it sees the full consequence without step-by-step calculation. The challenge is extracting this into coaching language.

---

## Maia vs Encoder — Complementary, Not Competing (confirmed 2026-04-05)

**Maia SAE:** Production. Position only (no move input) → 100% positional features. "What kind of position is this?"

**Encoder SAE:** Research. Position + move input → ~70% move features, ~5% positional, ~25% mixed. "What kind of move is this?"

They operate on different axes. Maia says "tactical position with fork potential." Encoder says "the best move was a forking move but you played a quiet retreat." Not redundant.

---

## Architecture Sweep (2026-04-05)

### BTK is the only viable architecture
V1 (L1) and Gated SAEs produce 1700+ noise features (>20% fire rate). Only BatchTopK produces specific features (zero noise across all configs).

### Optimal config — superseded (2026-04-12)
~~1024×k=32 or 4096×k=64~~ — this was before aux loss. With aux loss, all configs produce viable active feature counts. Current candidates: 2048 k=32, 2048 k=64, 4096 k=32, 4096 k=64 (all with aux). Detection scoring will determine the winner.

### 83% of encoder SAE features are move-dependent
The encoder encodes (position + move) pairs. No hyperparameter changes this. The ~5% positional features are exceptions (back-rank mate positions, pure endgames).

---

## Evaluation Metrics (2026-04-05)

### Agreement test — best for architecture selection
Measures: does feature fire on both played and best move, or only one?
- High = positional, Low = move-specific, Middle = noise
- Clearly differentiates BTK (70% pure move) from V1 (12%) and Gated (0%)
- Does NOT predict coaching value (disproven with 2,509 blunder test)

### Coherence — measures position similarity, NOT interpretability
Pairwise cosine similarity of top-firing positions.
- Queen checks (F77) = LOW coherence (0.536) — checks happen in diverse positions
- Noisy feature (F245) = HIGH coherence (0.914) — fires on similar-looking positions
- Useful for finding position-type features, misleading for move-type features

### No proxy metric predicts coaching value
Agreement, coherence, T1 structural, T2 concentration — none reliably predict which features help Claude coach better. The only real test is the coaching A/B test (not yet done).

---

## F2012 Deep Investigation — Cautionary Tale

Spent hours trying to label F2012 (14% fire rate, 30% agreement). Hypotheses tested and falsified:
1. "Defensive/stabilizing" — 34% counter-examples with active moves
2. "Maneuvering in closed positions" — fires 17% on captures
3. "Position type" — 30% agreement means it's 70% move-dependent

**Conclusion:** F2012 is a computational feature that doesn't map to any single chess concept. The encoder learned it for its own reasons. Not all SAE features are human-interpretable.

**What does work:** F77 (queen checks), F307 (opening pawns), F928 (captures), F1436 (tactical forcing) — these are clearly one thing, easily labeled from 8 examples.

---

## Methodological Notes

### Labeling requires Lichess-scale data
498 games (30K moves) gives only 186 features with enough data to profile. Lichess eval dataset provides millions of positions with best/alternative moves — orders of magnitude more signal per feature.

### Labeling requires actual positions, not statistics
Theme enrichment + top squares is insufficient context for LLM labeling. Show the actual moves, their classifications, and piece types. Let Sonnet find the pattern independently.

### The coaching A/B test is the only real evaluation
All metrics are proxies. The real question: "does adding SAE feature diffs to Claude's prompt produce better coaching?" This requires: labeled features → feature diffs on blunders → Claude coaching with/without → human rating. Not yet done.
