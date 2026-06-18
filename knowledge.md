# Chess Deck Research — Package Knowledge

Research-package-specific concepts. Shared cross-package concepts (SAE pipeline, DDB schema, handoff contract, core gotchas) live in [`../../knowledge.md`](../../knowledge.md) — authoritative; if you're reading something here that's also there, this file is out of date.

## Current state (2026-06-02)

**Production SAE:** `realgames_512_k8_v1` — 500 features, deployed.
**Chosen research model:** `btk_2048_k16_zscore.pt` (S3) — BatchTopK, k=16, **z-score ONLY (no L2)**, on Maia3 79M layer-7 best−blunder diffs. **990 coherent features (48%)** by dual-axis SEE probe. Beats z-score+L2 (350) and k=32-z-score (786).
**UNLABELED** — the existing Opus labels (`feature_labels_btk_2048_k16_v2.json`) are for the *z-score+L2* model; indices differ, do NOT reuse. Relabel needed with a both-axes prompt (include computed best-move + hang signatures).
**Next milestone:** relabel chosen model (both axes), rebuild profiles + atlas, then ship-vs-expand decision.

**KEY 2026-06-02 findings (supersede the k=16-v2/L2 entries below):**
- **Drop L2.** z-score-only nearly triples coherent features. L2 erases magnitude = mistake severity.
- **Coherence must measure BOTH moves** of the diff (blunder + best). Best-move axis dominant (634/990). One-sided (blunder-only) probing caused a whole session of false "it's noise" conclusions.
- **k=16 > k=32** (990 vs 786 coherent, z-score). k=8 z-score in progress.
- **Position-descriptor axes (phase/direction/severity/trajectory) are leaky** — features concentrate on them via corpus base rate, not mistake structure. Only piece-identity, what-hangs (SEE), and best-move-character are trustworthy coherence axes. Refutation-motif is moot (0 features, and Maia never computes refutations anyway).

## What the SAE can and cannot tell you (2026-06-06 — three experiments)

Triggered by applying v7 k6 to a real game and asking "why doesn't it say *why* the move was
bad, mechanically (e.g. 'allowed pin, knight→queen')?" Three experiments answered it. The ceiling
is **architectural, not statistical** — none of these are fixed by more data or higher elo.

**1. "Hangs" overclaims severity (label defect, not model defect).** The chip says "Hangs bishop"
on moves where the piece is only *threatened / about to be chased*, not lost. Verified on move 10
(Bb5): zero attackers on the bishop after the move, a −105cp tempo slip, yet labeled "Hangs." Root
cause: the diff `L7[best]−L7[blunder]` encodes "this move was worse than best" but not *by how much
/ whether material is actually lost*. The labeler reaches for the strongest verb. **Fix (cheap, not
yet done):** gate the verb on signal we already have — `cp_loss` + `blunder_hangs_own_pct` →
hangs (material lost) / drops / **misplaces·chases-out** (cp_loss<~150, no material change) /
weakens (positional). A labeling-prompt change, in the v7 lineage's spirit.

**2. Elo conditioning sharpens *category* but never reaches *mechanism* (pooling ceiling).** Ran the
game through the encoder at 2400/2400 vs the players' real ~1518. Per-move top-activation change was
NOT a uniform OOD inflation: most moves *dropped* (m20 −3.7, m50 −3.5, m56 −2.5); the one soft
inaccuracy that *sharpened* was move 17 d4 (+3.58, the largest swing). At 2400 the spurious "hangs"
features **dropped out** (f1684/f996/f1439/f521 gone) and were replaced by a more coherent theme —
f319 "King Safety › self-weakening / deflects defender" + f351/f1114 "committal move squanders the
better side." The shared anchor f672 "Missed Central Pawn Break" *strengthened* (4.0→5.6). So:
strong conditioning makes the read **cleaner and more coherent**, BUT the best it produces is
"self-weakening/overextending committal move," never "pin, N→Q." `mean64` pooling averaged away
"which knight / which file" before the SAE ever saw it — no elo setting recovers deleted info.
Side benefit worth keeping: **strong-elo conditioning is a free denoiser** for soft inaccuracies
(removes junk "hangs" features) with no retrain. NOTE: the SAE was trained on ~1500 diffs, so 2400
input is OOD *for the SAE itself* — a properly 2400-*trained* SAE would likely be cleaner still.

**3. Sample size is NOT the driver of low consistency (decisive).** Median feature trained on only
118 diff-examples; 40% on <100; cache is 168,132 diffs. BUT consistency is **flat across
example-count**: features with <50 examples have mean consistency 65.4, features with >1500 have
63.2. **Pearson r(log10 examples, consistency) = −0.002.** If thin data caused the messiness, the
under-trained features would be the messy ones — they are not. This is the signature of an intrinsic
ceiling. More games → same median-60 consistency, just over more vectors. Two caveats: (a) this
measures *quality of existing features*, not *coverage* — rare patterns (1-per-5000-games) could
still be absent from 168K; (b) all diffs come from the same ~3,843-game pool, so a *position-
diversity* limit is invisible to this within-cache test (would need 2K-vs-4K-game SAEs to probe).

### The design fork (what to actually change if pursuing "what was the model thinking")
The high-elo-everything path is a **local minimum** — too strong to model the 1500's actual cognition
(kills coaching), too un-localized to beat Stockfish at mechanism (SF refutation lines already give
"pin N→Q" perfectly). The two real forks:
- **Coaching / blind-spots → elo CONTRAST, not level.** The gold is "what does 2400 represent here
  that 1500 doesn't" = `L7[pos@2400] − L7[pos@1500]` (or the crosscoder framing). Keep the 1500;
  the upgrade is the *difference* between weak and strong Maia. An elo-conditioned model is *for*
  this; training on one elo throws the contrast away.
- **Mechanism / "why is d4 bad" → UN-POOLED, decision-point extraction.** Requires a new cache
  (square-resolved L7, no mean64). Gate it behind a cheap probe FIRST: ~50 positions with known
  pins/forks, extract L7 without pooling, ask "does a linear probe find a pin direction?" If no,
  Maia doesn't represent tactics explicitly and goal (B) should be abandoned — do not spend GPU
  before that probe returns yes. See [[direction_arbiter_is_board_not_see]] for the related lesson
  that mechanism must come from the board/refutation, not SEE heuristics.

Artifacts: `output/game_v7_169764992210.json` (1518), `output/game_v7_2400_169764992210.json`
(2400), `output/game_v7_k4_169764992210.json` (k4). Game 169764992210 = cabbagelover White, an
endgame-technique collapse, useful as the canonical "soft inaccuracy + losing endgame" probe game.

## Architecture — what works (and why)

### Seed STABILITY — instability is init-basin, fixed by data-driven init (2026-06-17) ⭐

The big methods finding. SAE seed-instability (different seeds → different dictionaries; literature
treats this as near-unsolved, cf. Fel et al. Archetypal SAE ICML 2025) is on our Maia blunder-diff data
**almost entirely an INITIALIZATION-basin artifact**, and cheaply fixable:
- random init, different seeds: MMCS 0.41, only ~6–9% of features reproduce (>0.7 cos).
- shared (identical) init, different data order: MMCS **0.95** → instability is NOT data/objective, it's
  which random basin init lands in.
- **k-means init, SAME centroids, different seeds: MMCS 0.89** — BUT the bias test (Sam) showed this is
  partly circular: **DIFFERENT centroids + different seeds → MMCS 0.63.** So data-driven init captures
  real seed-independent structure (0.47→0.63 genuine), but ~half the 0.89 was shared-init, not
  data-determination. NOT seed-independent "truths."
- **Action:** for a reproducible production SAE, fix the centroids once and reuse → legitimately 0.89
  ("features from this data + this fixed init"). Honest cross-independent-run number is ~0.63.
- Archetypal SAE (the literature fix) only nudged it (0.41→~0.50) AND cost sparsity/reconstruction — not
  worth it vs init.

Full writeup + the 3 systematic-debugging root-causes (overcomplete L1-shrinkage collapse;
threshold-can't-climb → stuck L0; per-seed candidate-set confound) + JumpReLU canonical impl notes:
`docs/2026-06-17 SAE Stability — init-basin instability and the data-driven-init fix.md`.

### BatchTopK (SandstonePersonas pattern) — CURRENT

- **BatchTopKSAE** from SandstonePersonas (`sae/model.py`): batch-level top-k (not per-position), unit-norm decoder enforced every step, AuxK dead-feature revival.
- **Key hyperparameters (2048/k16):** k_aux=128, aux_alpha=0.03125, lr=3e-4, batch=4096, 200 epochs, warmup=500 steps, seed=123.
- **Normalization:** z-score per-dim then L2 per-sample. Save mean/std as `_stats.json` alongside weights — required for all inference.
- **Inference:** use calibrated threshold θ (NOT BatchTopK at eval — batch-dependent). Compute θ = mean of k-th largest activation per position across corpus. See `scripts/evaluation/calibrate_threshold.py`. k=16 θ≈0.0806 → L0≈15.7.
- **AuxK note:** AuxK never fires on 168k corpus because every feature activates multiple times per batch (4096 × 16 = 65k active slots across 2048 features). This is correct behavior — features aren't dying, they're just rare. The 1012 near-dead features (0<freq<0.1%) are likely real low-frequency concepts, not broken.
- **k selection:** k=16 chosen over k=32 for precision (coaching use-case). k=16 has 63 very-active vs 147 for k=32; 744 middlegame-dominated features vs 450. Per Noyan/Jonathan: k is the precision/recall lever. k=32 fires on loosely-related concepts; k=16 is more selective.

### Representation: Maia3 79M layer-7 diff

- `maia3_79m_fixed.pt` (79M params, 8-layer transformer, 1024-dim). **Not the ONNX probe** — that produces mush. This specific checkpoint is required.
- Layer 7 activations hooked via `model.transformer.layers[6]` (0-indexed). Mean-pooled over 64 board squares → 1024-dim per position.
- Diff = (best-move position activations) − (blunder-move position activations). Captures "what Maia sees differently between right and wrong."
- Cache: `maia3_l7only_v2_dedup.pt` (168,132 × 1024). Deduped from 168,669 (537 duplicate fen|uci removed).

### Frequency-ceiling fix (2026-06-01) — the high-frequency feature problem

When applying k=16 v2 to the 10 test positions, ~6 features dominate raw activation ranking but are **coarse, near-content-free detectors**:
- f101 (fires 33% of corpus, labeled "Queen Hanging to Bishop" — actually fires on hung bishops too)
- f1487 (44%! "Quiet move ignoring hanging piece"), f952 (29%), f98 (39%), f959 (38%)

These fire on 6-8 of the 10 test positions and outrank the SPECIFIC, correctly-labeled features that actually match each mistake (f504 "Queen left en prise" 4%, f2027 "Knight capture enables pawn fork" 0.7%, f735 "Queen Abandons Post" 7%).

**Two problems:** (1) Opus mislabels high-frequency features with specific names — a 33%-corpus feature can't be "Queen Hanging to Bishop." (2) Ranking by raw activation surfaces blobs over specific features.

**Fix (first pass): hide features with corpus fire-rate >10% before ranking.** Surfaces specific features for ranking. BUT see correction below — this was too blunt.

**CORRECTION (2026-06-02, after Sam flagged "33% can't be hung queens"):** The blobs are NOT uniform, and characterizing them by top-60 positions is WRONG (a 33%-fire feature hits 55k positions; top-60 is the unrepresentative tip). Measured Q/R-hang rate across activation bands instead:
- **f101 = graded-real:** "high-value piece hangs" at 75% in top-2% activation, decaying to 10% (base rate) at threshold. Real concept, but ONLY meaningful at high activation. The magnitude IS the confidence.
- **f1487 = flat-noise:** ~11% hang rate at EVERY activation band including top-2%. Highest-firing feature (44%) and carries zero signal. Discard.

**Right fix: per-feature high-activation gating + flat-noise detection, NOT global fire-rate filter and NOT lower k.** Lower k (the k-sweep "k=8" rec) would kill graded-real features like f101 along with noise. Keep features whose high-activation band is coherent; gate each at the activation where concept-rate exceeds base rate; discard flat features. See `output/blob_experiments_report.md` for full curves. **Method lesson: never characterize a high-fire SAE feature by top-N — measure property-rate across the full activation distribution.**

The k-sweep + corpus-sweep experiments (`blob_experiments_report.md`): blob *count* is monotonic in k (12→89 across k8→k32) and weakly affected by corpus size. But "blob count" conflates graded-real and flat-noise features, so minimizing it isn't the right objective — coherence-at-high-activation is.

**Labeling improvement for next pass:** feed Opus the fire rate; instruct "if >20% of corpus, label as COARSE pattern, not specific." See [[reference-personas-sae-pipeline]].

### Older architecture note

- **BatchTopK** is the only viable SAE architecture. L1/Gated produce noise on blunder move tokens.
- **Move-token (hidden[77])** from DeepMind 270M encoder. Not mean-pooled, not per-position. See shared knowledge.md § SAE Feature System for the full extraction pipeline.
- **Aux loss** fixes dead features at any k.

### Architecture decision: why 2048_k64

13 SAE configs compared (512/1024/2048/4096 × multiple k values). Key findings:

- **Filtered recall@N is similar across architectures** (~40% at top-5 for 1-10% features). The advantage of 2048 is *diversity* — 918 useful features vs 116 at 512_k8. More specific coaching patterns to surface.
- **Activation strength is continuous, not binary.** Use full strength values for scoring. Threshold at ~0.5 to filter garbage.

### What we measured but doesn't predict label quality

- Reconstruction quality (FracVar): 1024 negative, 2048 at 81%. Doesn't predict label quality.
- Severity correlation with cp_loss: dominated by catastrophic blunders (9000+ cp). On moderate blunders (300-1000cp), drops to 0.07.
- Decoder cosine: zero pairs above 0.5 at any dict size. Features well-separated by construction.
- Golden feature independence: all collapse to ~6 groups regardless of architecture.

## Labeling

### What works
- **Sonnet 4.6 thinking** (4K budget, 16K max tokens) — best label quality. More detailed sub-patterns than Opus.
- **Gemini 3.1 Pro → Sonnet synthesis** pattern — Gemini analyzes positions, Sonnet synthesizes the pattern across positions.
- **5,851 Gemini-analyzed positions** cover all 2042 features (10-20 examples each).
- **Top-20 examples by activation strength** — well above 0.7 threshold, clear signal.

### Taxonomy (for `2048_k64`)

Production taxonomy (`realgames_512_k8_v1`, 7 domains / 24 subcategories) lives in `../chess-deck-code/knowledge.md`. The `2048_k64` re-labeling produced its own 7 categories emergent from the data:

| Category | Features | % |
|----------|----------|---|
| Endgame Technique | 660 | 32% |
| Tactical Oversight | 544 | 27% |
| Piece Safety | 299 | 15% |
| Mate Awareness | 266 | 13% |
| Calculation | 217 | 11% |
| King Safety | 55 | 3% |

~30 subcategories after chip name consolidation (still needs cleanup — 749 raw chip names with near-duplicates).

**Subcategory is the right coaching granularity.** Category too broad, individual feature too narrow (label drifts across positions).

## Player Profiling

### The coaching metric
**Continuous subcategory score vs rating band baseline.**
- Score = mean activation strength across player's blunders (includes zeros).
- Ratio = player / baseline. >1 = worse than peers, <1 = better.
- Show top 5 subcategory leaks ranked by ratio.

### Validated on cabbagelover5566 (1800 rapid)
- **Top leaks:** Autopilot 2.12x, Missed Captures 1.87x, Missed Tactics 1.64x
- **Strengths:** Endgame 0.59x, King Safety 0.47x
- Matches known playing style — tactical oversight, not strategy.

## Research-only infrastructure

Shared S3 paths live in `../../knowledge.md` § S3 layout. This table covers artifacts that are *only* relevant to research workflows:

| What | Where |
|------|-------|
| Gemini analyses (5.8K positions) | `output/position_analyses.json` |
| Architecture comparison results | `output/` with per-architecture suffixes |
| Shared encoder code | `scripts/shared/chess_encoder.py` |
| S3 inventory | `output/S3_INVENTORY.md` |
| Labeling pipeline procedure | `PIPELINE.md` |

## Research Dead Ends

- **Maia SAE (puzzle-trained)** — trained on puzzle positions, detected positions not mistakes. Hub contamination high. **Revisited 2026-04-23 with blunder-filtered data:** 200K blunder activations, 2048_k64 config, hub contamination 4%, features are specific. Not a dead end — labeling in progress.
- **MLP projection (encoder→LLM)** — information asymmetry. LLM ignores encoder when FEN text is present.
- **Per-blunder fire rate baselines** — flat across ratings. Need per-game rates instead. (Current: baselines are blunder-only fire rates; rating signal comes from blunders-per-game. Note in `../../knowledge.md` § gotchas.)
- **dict=1024 for production** — degenerate case (dict=input_dim). Negative reconstruction. Severity signal might be artifact. Abandoned in favor of 2048_k64.
- **Diff SAE** — trained on (played move – best move) diff. Produced tautological labels ("better move was better").
- **Blunder encoder SAE** — blunder moves too diverse to cluster (27% confidence).

## Maia 3 SAE (2026-05-22)

Separate from the DeepMind 270M encoder SAE above. Uses Maia 3 layer-7 residual stream, diff pooling (to_sq - from_sq), L2 normalized.

### Architecture
- **Model:** Maia 3 (8-layer transformer, 512-dim, Elo-conditioned)
- **Probe layer:** `/model/transformer/layers.7/Add_2_output_0`
- **Pooling:** diff (to_sq - from_sq) — best for tactical clustering
- **Normalization:** L2 (z-score + unit sphere) — more features labelable than raw
- **SAE:** BatchTopK 2048 dict, k=32, aux loss, 200 epochs
- **Training data:** 200K Lichess blunders (≥200cp per Lichess cloud eval, depth 40+), mixed Elo 600-2600

### Gemini labeling — what worked (the $3 batch)

**Successful batch:** `chess-sae-position-analysis` (job `batches/hzeagvatdornkr6swwh7qd7iy328h2br2vtz`)

Input format (proven, ~460 tokens/position):
```
System: "You are a chess grandmaster analyzing blunder positions. For each position, explain:
1. What the player was trying to do (their intent)
2. What goes wrong after this move (the refutation/punishment)
3. The specific point of failure (which square, piece, or tactical motif was missed)
Be concrete and specific. Name squares, pieces, and tactical patterns."

User: "Analyze this chess blunder:
Position (FEN): {fen}
Move played: {uci}
Centipawn loss: {lichess_cp_loss}
This move was a blunder that lost {lichess_cp_loss} centipawns of evaluation. Explain what happened."
```

Output schema (6 fields, free-form — NOT enums):
- `intent` — STRING (1-2 sentences)
- `blunder_trace` — STRING (2-3 sentences)
- `point_of_failure` — STRING (1 sentence)
- `best_move_rationale` — STRING
- `position_context` — STRING (only_move / thematic / normal)
- `tags` — ARRAY of STRING (free-form tactical themes)

Key facts:
- **No Stockfish lines in prompt.** Just FEN + UCI + cp_loss. Gemini analyzes positions natively.
- **Uses Lichess cp_loss** (depth 40+), NOT our Stockfish depth-18 cp_loss.
- **Thinking enabled** (~2200 thinking tokens per position).
- **~250 output tokens** per position (JSON schema constrains output).
- **Cost:** ~$3 for 5,851 positions (~$0.50/1K positions).
- **Model:** `gemini-3.1-pro-preview`

### Gemini labeling — what failed

1. **Flash batch (accidentally submitted, $15 wasted):** Used `gemini-2.5-flash` with a different 5-field schema (enums for `tactical_motif` and `severity`). Got cancelled but still charged. Don't use Flash for this task.

2. **19.5K input file on chess-poc** (`gemini_batch_input.jsonl`, 19,511 lines): Built for the cancelled Flash batch. Uses the WRONG schema (5 fields with enums). Do NOT use this as a template.

3. **SF-enriched Format B** (built in this session as `build_batch_input_maia3.py`): Includes `top_lines`, `refutation_lines`, `eval_delta`. Untested, 36MB, bloated input (~1500 tokens/position vs 460). The successful batch proved SF lines aren't needed — Gemini reads FENs natively.

### Stockfish enrichment — depth disagreement problem

Our Stockfish runs at depth 18. The Lichess positions were verified at depth 40+ (cloud analysis with tablebases).

- **47% of positions:** SF depth-18 doesn't see the blunder (cp_loss < 100 when Lichess says ≥ 200)
- **29% of positions:** SF says `played == best` (move IS the best move at depth 18)
- Root cause: endgame positions and deep tactical combos invisible at depth 18
- The proven Gemini batch sidestepped this entirely by NOT including SF lines — just the Lichess cp_loss and letting Gemini figure it out

### Geometric labeling — dead end

Attempted python-chess based geometric analysis (detect forks, pins, hanging pieces, abandoned defense from board position). Results:

- `moved_to_attacked` fires on nearly everything (trivial)
- Fork detection overcounts (counts all attacked pieces, not just from the threat piece)
- Pin/overloaded detection is fragile (false positives from `board.attackers()` logic)
- **60% of positions get no signal** (the mechanism requires multi-move calculation)
- **Verdict: useless for mechanism detection.** The SAE features encode multi-move patterns that can't be reduced to one-ply geometry.
- Trust SAE clusters + Gemini narrative, not heuristic geometry.

### Structural taxonomy — useful as filtering layer only

From SF-derived stats (piece played, phase, check/capture rates, cp_loss), can assign features to ~15 categories at 98% coverage. But these are **filters, not coaching categories** — "Knight Errors" contains 282 different mechanisms.

- Good for: summary level ("you had 3 knight errors")
- Bad for: drill level ("practice knight forks") — too broad
- Gemini's per-position analysis is needed to get mechanism-level labels

### SAE structure findings

- **41 hub features** (fire 5-19% of all positions) — position-type indicators, excluded from labeling. High mutual co-firing (Jaccard up to 0.64).
- **2007 specific features** (~1.6% fire rate each) — tactical patterns. Low mutual co-firing among specifics.
- Sonnet confidence score correlates with structural purity (0.97 piece purity at conf≥0.9 vs 0.64 at conf<0.6). Confidence is real signal.
- Sonnet's "hanging_pieces" category is too broad (covers 34% of features, 12/20 structural clusters).

### Opus labeling pipeline (2026-05-27)

Two-pass pipeline using Opus 4.6 on Bedrock (capped thinking 4096 tokens, concurrency 20):

1. **Pass 1:** 19,216 per-position analyses (19K unique positions × Stockfish depth-18 enrichment × Opus). 17h wall time on chess-poc. **Canonical copy is on chess-poc: `/home/ec2-user/SageMaker/all_positions_labeled_opus.json` (79.5MB, 19,342 entries).** ⚠️ The S3 copy `sae/cache/all_positions_labeled_opus_final.json` (40MB) is **TRUNCATED to 10,648 entries** — do NOT use it; pull from the notebook. (Cost us a long debugging detour 2026-05-29.)
2. **Pass 2:** 2,000 feature-level labels synthesized from 10 position analyses each. 80min. Output: `s3://chess-stage-a-140023406996/sae/labels/maia3_feature_labels_opus.json`
3. **Rerun:** 604 low-confidence features relabeled with geometric context (diff vector stats, from-file/rank distribution). Only 36 improved — low confidence = genuinely polysemantic, not missing info.

### Taxonomy rebuild — chip-first was lossy (2026-05-29)

The shipped Pass-2 labels have accurate `description` and `label` (one-sentence) fields, but the 2-4 word `chip` was generic ("Quiet move ignores tactics") on ~400 features, and categories were assigned FROM those lossy chips → junk-drawer categories (old "Missed Tactics" = 372 features spanning every piece type). **The features are fine** (near-orthogonal decoders, accurate descriptions); only the chip + category layer was broken.

**Fix — rebuilt TITLE→CATEGORIZE→CHIP** (compression last, not first):
- `description` (verified accurate against the board) is the source of truth.
- Each of 1,996 features assigned to one of **20 checkpoint-stable coaching categories** (reused from `chess_blunder_taxonomy_v2`, since the coaching vocab is about chess not the SAE).
- Specific chip generated LAST, category-aware, generic frame banned.
- Result: generic chips 398→0, no junk drawer (largest 20%), every category's structural signature matches its definition (greedy_captures 88% capture, checks_lose_tempo 73% check, king_walks 98% king, slow_play 88% quiet).
- Output: `output/taxonomy_v2/taxonomy_v2.json` + `REBUILD_REPORT.md`. Pipeline: `scripts/sae/taxonomy/` (deterministic fingerprint/verifier/evidence/assemble/qa + `relabel_sonnet.py`).
- **Lesson:** compress last. A 2-4 word title seeded before categorization propagates its lossiness downward. [[feedback_verify_batch_formats]]-style: don't trust the chip; verify against the description/board.

> **⚠️ SUPERSEDED (2026-05-30) — `taxonomy_v2.json` is suspect, redo planned.** Two problems found after: (1) **categorization was still top-down** — each feature independently assigned to one of 20 pre-baked categories → magnet effect (Slow Play 408, Undefended 4). The correct method is bottom-up semantic clustering — see `docs/knowledge/taxonomy-method-persona.md`. (2) **Provenance bug:** the profile the labels came from (`l2_feature_profiles_v2.json`) was NOT reproduced by ANY checkpoint tested (flat k=32 l2_200ep/v2/base, nor H1 matryoshka) over the v2 cache — feature 3's top firings ≠ the profile's Bxf7+ set in all cases. So the labels' source model is unknown; treat per-feature categories as unverified. **Redo:** regenerate fresh profile + fire rates from the flat k=32 model, then bge-m3 cluster bottom-up. See plan.md "Current State (2026-05-30)".

Scripts: `scripts/labeling/label_features_pass2.py`, `scripts/labeling/label_features_pass2_rerun.py`

### Rating validation (2026-05-27)

56K blunders (6 rating bands, ~10K each) from `sweep_blunders_2000.json` run through Maia 3 → SAE on chess-poc.

**Key result: 73% of features (62% of fire rate) vary by rating band.** This is NOT flat like the DeepMind SAE — the Maia 3 Elo conditioning gives genuine rating signal.

Output: `s3://chess-stage-a-140023406996/sae/cache/sae_rating_validation.json`

### Feature hierarchy (2026-05-27)

Built from decoder cosine similarity (Ward hierarchical clustering):
- **k=26 natural gap** in dendrogram — 26 top-level categories
- **277 subcategories** (~7 features each) via sub-clustering within parents
- Labeled with Opus, plain coaching language

Key findings:
- **"Aggressive Bishop Abandons Post"** is the #1 diagnostic feature at FOUR rating transitions (1100→1900). Single most important skill below 1900.
- **Bishop mastery** is the consistent differentiator across ratings (drops most at every transition except 1900→2100 where Rook takes over)
- **"Capturing removes shielding pawn"** increases monotonically from 7% to 18% — the primary COST of improvement
- Top 100 features capture 27% of discrimination power; top 500 capture 70%
- Adjacent rating bands are 99.1-99.4% cosine similar — signal is subtle but real
- 60% of fire rate is coachable (varies by rating), 40% is universal (fires equally at all levels)
- Anti-correlations: fixing one pattern often introduces another (e.g., stopping pawn-hunting → starting to capture own shielding pawns)

Output files in git:
- `output/maia3_feature_hierarchy.json` — full hierarchy with feature assignments
- `output/maia3_taxonomy_k26.json` — 26 category names + cluster assignments
- `output/maia3_website_rates.json` — per-subcategory fire rates for all 6 bands
- `output/maia3_rating_profiles.json` — simulated player profiles by band
- `output/maia3_analysis_findings.json` — rating roadmap, fatal features, beginner markers

### cabbagelover5566 profiling (2026-05-27)

1,209 blunders from 502 games run through Maia 3 → SAE. Profile compared against 1800-2000 peer baseline.

Top weaknesses vs peers: "Pawn Captures That Open King" (2.3×), "Aimless Rook Repositioning" (1.8×), "Trading Away Dominant Pieces" (1.5×). Top strengths: "Greedy Material Grabs" (0.5×), "Queen Misplacement" (0.7×). Matches known profile from DeepMind SAE: tactical oversight, not strategy.

### Files on chess-poc

| File | What | Lines |
|------|------|-------|
| `gemini_batch_input.jsonl` | Input for CANCELLED Flash batch (wrong schema) | 19,511 |
| `gemini_batch_results_raw.jsonl` | Output from SUCCESSFUL Pro batch | 5,851 |
| `all_positions_labeled_opus.json` | Pass 1 results — **CANONICAL 19,342 analyses (79.5MB)**; S3 copy is truncated to 10,648 | 19,342 |
| `maia3_feature_labels_opus.json` | Pass 2 + rerun results (2K feature labels) | — |
| `position_enrichment_cache.json` | Stockfish depth-18 enrichment (19K positions) | — |
| `sae_rating_validation.json` | 56K positions × 6 bands fire rates | — |
| `feature_clustering.json` | Decoder cosine + co-firing clustering | — |
| `cabbagelover_profile.json` | Player profile for cabbagelover5566 | — |
| `stockfish_data.json` | SF depth-18 enrichment | 18,027 |
| `l2_feature_profiles.json` | Top-20 positions per feature | 2,048 features |
| `l2_labels_sonnet.json` | Sonnet 4.6 labels | 2,007 features |

### Next: rebuild batch in proven format

Need to rebuild `build_batch_input_maia3.py` to match the PROVEN format:
- Simple prompt (FEN + UCI + Lichess cp_loss, no SF lines)
- 6-field schema (intent, blunder_trace, point_of_failure, best_move_rationale, position_context, tags)
- Free-form tags, no enums
- Thinking enabled
- Estimated cost for 18K: ~$9 (18K × $0.50/1K)

## Deep-dive docs

| Doc | Date | What |
|-----|------|------|
| [`docs/knowledge/normalization.md`](docs/knowledge/normalization.md) | 2026-05-22 | SAE input normalization — why raw (no norm) beats Z-score and L2 for Maia 3 diff vectors. Sandstone comparison. |

## Maia3 Layer-7 Representation Space (May 2026)

### What the current v2 SAE actually encodes
The `maia3_blunder_diff_v2.pt` cache uses `before - after-blunder` at the destination square
(NOT `blunder - best` as originally intended). The best move is NOT in the representation.
- Confirmed: `to_sq(before-blunder)` corr=0.83 vs stored diff; `blunder-best` corr≈0.02
- The v2 metadata has `best_uci: None` for all 200k positions

### Option A: repr(after-best) - repr(after-blunder)
Tested on v1 cache (which has best_uci). **(derived on v1 data — needs re-verification on v2.
The v1 cache has the Black-to-move label-inversion bug; every gap/correlation below is suspect
until reproduced on the v2 cache.)** Key findings:

**cp_loss signal:**
- `h_diff.mean64 -> cp_loss: r≈0.07` (5-fold CV) — weak, but expected
- `value(best)[3D] -> cp_loss: r≈0.51` — value head has real signal but too low-dim for SAE
- cp_loss weakness is NOT a bug — Maia3 sees "fork available" as coherent regardless of severity

**Win-prob encoding:**
- `h_after_blunder.mean64 -> wp_white: r=0.94` — layer-7 strongly encodes win-prob
- `h_before.mean64 -> wp_white: r=0.25` — before-move position is weaker
- Layer-7 is highly eval-informative but eval concentrates AFTER moves, not in the diff

**Tactical clustering in h_diff (the key result):**
- capture gap=0.122, fork gap=0.039, quiet gap=0.017
- All positive — positions where the best move was a fork cluster coherently in h_diff space
- This is the geometric prerequisite for an SAE to produce "missed fork" features

**Value_diff (3D) clustering:**
- fork gap=0.149, quiet gap=0.065, capture gap=-0.048
- Value_diff clusters forks and quiet differently from h_diff — they measure different things
- h_diff = tactical character; value_diff = eval consequence

### What an SAE trained on Option-A h_diff would learn
Features organized by **what kind of move was missed** (fork, free capture, quiet tactical strike).
NOT organized by severity. This is the right taxonomy for coaching — "you missed a fork" is 
actionable. Severity (cp_loss) can be added as a statistic per feature, not as a feature axis.

### Right pooling
Mean over all 64 squares is fine for h_diff. For single-position win-prob, mean64 gives r=0.94.
Per-token scalar norms max at r=0.23 — no dominant single-token eval signal.

### Architecture note
Maia3 probe ONNX input: `tokens [batch, 64, 12]` (board only, no move token).
Output probe: `layers.7/Add_2_output_0 [batch, 64, 512]`.
Value head: `logits_value [batch, 3]` — convention `[black_win, draw, white_win]`.

### To build the correct Option-A SAE
1. Use v1 source positions (have best_uci + player elos)
2. For each position: push blunder_uci → board_bl, push best_uci → board_bs
3. Run Maia3 probe on both at player's elo
4. Diff: `h_bs.mean64 - h_bl.mean64` → [512] vector per position
5. Train BatchTopK SAE on these vectors
6. Features will be tactically meaningful (fork/capture/quiet separate)

## 79M Maia3 model probe results (May 2026)

Tested `maia3-79m` (1024-dim, 8 layers, GAB) against existing ONNX probe (512-dim).
**(derived on v1 data — needs re-verification on v2. Probes ran against the v1 blunder cache
with the label-inversion bug; the board_diff / L2 / L7 gap numbers below are suspect until
reproduced on the v2 cache.)**

**Key finding: 79M is NOT better for SAE purposes.**
- h[sq] global context cosines: 0.92–0.998 across wildly different positions
  (old ONNX was 0.76–0.97 — slightly better)
- Larger model = more square-identity-dominated per-square activations
- fork gap with mean64_board_diff: 0.041 (79M) vs 0.039 (old ONNX) — essentially same
- capture gap with to_sq_diff: 0.148 (79M) vs 0.272 (old ONNX) — old ONNX wins

The square-identity dominance is architectural, not a model-size issue.
More parameters improved policy accuracy, not per-square positional information.

**Best construction confirmed: mean64(h_after_best - h_after_blunder)**
- Fork gap: ~0.039–0.041 (hard ceiling with this approach)
- Capture gap: ~0.058–0.122
- Better than all single-square approaches for forks
- Currently building 200k cache with old ONNX

**Fork ceiling (~0.04) is a fundamental limit of per-square representations.**
Forks require understanding "this piece simultaneously attacks two squares" which
isn't captured by any single square's activation or their mean.

## maia3 repo notes
- Install: `git clone https://github.com/CSSLab/maia3.git && pip install -e .`
- Load: `MAIA3Model(cfg)` with `SimpleNamespace` cfg; load_state_dict with renamed keys
  (checkpoint uses `smolgen_*`, model expects `gab_*` — rename before loading)
- Forward: `model(tokens, self_elos, oppo_elos)` where tokens=[B,64,97] from
  `ds.get_historical_tokens([tok]*8, cfg, 0,0,0,0)` with `tok=ds.tokenize_board(board)`
- Hook last layer: `model.transformer.layers[-1].register_forward_hook(...)`

### MAJOR CORRECTION (2026-06-02): coherence must measure BOTH moves of the diff

The SAE diff = `L7[after maia-best] − L7[after blunder]`. Features can cohere on the best-move
side, the blunder side, or both. Every coherence probe I ran (motif-join, piece-hang, SEE) only
looked at the BLUNDER move → systematically called best-move-coherent features "noise."

Dual-axis probe (`dual_coherence.py`, real SEE + maia_best): of 1322 candidate features,
184 cohere on blunder-axis only, **458 on best-move-axis only**, 71 both, 609 neither.
**713/1322 (54%) coherent — best-move axis has 2.5× more than blunder.** The best-move features
are the "you missed a good move" mistakes (missed capture/check/quiet-correction) — first-class
coaching categories invisible to a blunder-only probe.

This SUPERSEDES the blob-pessimism above. The SAE is not mostly noise. Labeling must describe
both what was played and what Maia would have played. Use z-score-only k=16 (btk_2048_k16_nol2.pt)
— same blob behavior as L2 but ~2× more useful features (magnitude preserved).

**Method law:** a coherence probe on a feature defined over a DIFFERENCE must measure both sides.
One-sided probing = guaranteed false negatives. This blind spot cost most of the 2026-06-01/02 session.
