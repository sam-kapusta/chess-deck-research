# Chess Encoder

> **CORRECTION (2026-05-31): the option_a / board_diff / l2l7 architecture search below is INVALID.**
> All three SAEs were trained on the v1 blunder cache (`maia3_blunder_diff.pt`, Black-to-move
> label-inversion bug). The weights, caches, labels, profiles, and eval artifacts have been
> deleted from S3, the notebook, and git. No conclusion below (board_diff "leading candidate",
> f46 "Recapture leaves piece hanging", >50%-coherent expectations) holds. Build/train scripts
> kept under `scripts/sae/new_sae_architecture/` but must be repointed to the v2 cache before
> any rerun. Next step if revisiting: rebuild all three constructions on v2 data.

## Current State (2026-06-01 end) — BatchTopK k=16 labeling overnight

**Active overnight:** Pass-1 Opus labeling gap positions for k=16 v2 profiles (13,208 positions, ETA ~2.5h from ~22:00). Pass-2 (feature chips) chains after. Outputs: `all_positions_labeled_opus.json` (growing to ~47k), `feature_labels_btk_2048_k16_v2.json`.

**What's done this session:**
- BatchTopK SAE trained (matches SandstonePersonas exactly): 2048/k16 v2, 200 epochs. Weights in S3.
- Calibrated inference threshold θ=0.0806 (k-th largest method). All eval scripts updated to use threshold not BatchTopK.
- Feature stats computed with extended fields: piece type (blunder + best move), is_capture, is_check, piece_left_hanging. Both k=16 and k=32.
- Atlas (`l7only_atlas.html`) working with clickable boards + chess.com links.
- corpus builder (`cache_real_game_blunders.py`) identified — streams `Lichess/standard-chess-games`, pulls 200k blunders with elo + eval trajectory. Plan to run 1M position build.

**Key decisions made:**
- k=16 over k=32 for coaching precision (Noyan/Jonathan recommendation, paper confirms k=16 best for downstream CE on Gemma 2B)
- BatchTopK at inference is wrong — must use calibrated threshold (per arXiv:2412.06410 §3 + inference_example.py)
- AuxK not firing is correct behavior for 168k corpus (features don't die, they're just rare)
- Inference threshold method: mean of k-th largest activation per position (not min-positive, which collapses to 0 on small corpora)

**Next session:**
1. Check overnight results — download `feature_labels_btk_2048_k16_v2.json`, coherence stats
2. Run T1 with threshold inference on k=16 v2
3. Re-run feature stats with threshold inference (already uploaded, pending run)
4. Update atlas with stats panel + k=16 labels
5. Launch 1M Lichess corpus build (`cache_real_game_blunders.py --n-positions 1000000`)
6. Commit `cache_real_game_blunders.py` to git (currently only on notebook via S3_INVENTORY §6)

## Current State (2026-05-31 end) — v2 REBUILD running (superseded by 2026-06-01)

**Active work:** rebuilding the 4 SAE constructions on v2 (corrected) data + Maia-best moves.

**Key unblock this session:** the 3 constructions need a *best move* per position; v2 cache
dropped best_uci and the only 200k best_uci (v1 metadata) is bug-affected. **Decision: best =
Maia3 policy argmax @ elo 2600** (not Stockfish) — the SAE reads Maia's activations, so Maia's
own human-best is the consistent, coaching-relevant target. Building `maia_best_200k.json`.

**Maia-best extraction gotcha (cost a debug cycle):** must use the maia3 package primitives —
`get_all_possible_moves()` = **4352**-move vocab (NOT the 1968-move `move_to_action.json`, which
is the DeepMind-270m model), `tokenize_board` (mirrors board for black-to-move),
`get_legal_moves_mask`, and `mirror_move` on the chosen move for black. Wrong vocab → 3%
Stockfish agreement (chance). Correct → 50% agreement + passes start/hanging-Q/M1 sanity.

**Unattended pipeline (2 screens on chess-poc):**
- `maiabest`: `build_maia_best.py` → `maia_best_200k.json` (~110 pos/s, ~30min)
- `pipeline`: `wait_and_run.sh` → `run_all_v2.sh`: build option_a + board_diff + l2l7 caches →
  slice l7only → train 4 SAEs @ **k=16** → `eval_v2_html.py` → `eval_v2.html` + results JSON

**4 constructions (all v2 + maia_best@2600, conditioned at player's real elo):**
- option_a: `h[best_to]-h[blunder_to]` before-board (ONNX, 512d)
- board_diff: `mean64(h_after_best - h_after_blunder)` (ONNX, 512d)
- l2l7: `concat(L2_mean64_diff, L7_mean64_diff)` (79M PyTorch, 2048d)
- l7only: L7 half of l2l7 (1024d)

**Scripts (in /tmp locally + ~/SageMaker on chess-poc; commit to scripts/sae/ before trusting results):**
`build_maia_best.py`, `build_option_a_v2.py`, `build_board_diff_v2.py`, `build_l2l7_v2.py`,
`train_sae_v2.py` (generic), `eval_v2_html.py`, `run_all_v2.sh`, `wait_and_run.sh`.

**Decision after eval:** if one SAE clearly fires the right features on the 10 test positions →
start labeling it. Drop degenerate positions (maia_best==blunder, ~14%).

 — SAE Feature Pipeline

## Current State (2026-05-30)

### Taxonomy design — LOCKED via /grill-with-docs (2026-05-30)

Decisions (don't re-litigate):
1. **Shape:** 2-level, **category → cluster → feature**. Fresh artifact, replaces the old prod taxonomy (`realgames_2048_k64_v1` domain/subcategory schema is stale — not bound to it).
2. **Top-level axis = mistake-type ("what kind of mistake can a player make"), FLAT.** No "missed-win vs losing-blunder" halves wrapper at the top (considered, rejected — Sam thinks in mistake-types, not in that split).
3. **Data-driven, not hand-imposed.** The ~11 seed categories below are the *target shape*; the real categories come from reading each feature. Merge/drop/rename by what the data supports. Anything under ~2% folds into a neighbor.
4. **Offensive-miss mistakes are first-class top-level categories**, not sub-flavors of a "slow play" blob. The big bge-m3 "Slow Play Punished / Autopilot" mass (~61% of features) is NOT one junk-drawer category — it's several real "you had a line and played a nothing-move" categories collapsed under one bad name. This was THE key correction (Sam: "the entire Slow Play Punished category is what I've been talking about where you missed a good move").
5. **CRITICAL METHOD NOTE:** keyword classification is UNUSABLE for assigning these categories — 77% of features match 3+ category keywords, and commission-vs-offensive-miss are keyword-entangled (same feature counts as "hung a piece" or "missed a capture" depending on which keyword you check first). Counts MUST come from reading each feature's description (LLM/agent judgment of the PRIMARY mistake), not regex. Every keyword-based count this session was an artifact — ignore them.

**Seed category vocabulary (~11, data may revise):**
- Commission: Hung a Piece · Walked Into a Tactic (got forked/pinned) · Greedy Capture · Exposed Your King · Bad Trade/Simplification · Abandoned a Defender
- Offensive miss: Misplayed an Attack (had attack/initiative, failed to convert to mate OR winning material) · Missed a Capture · Missed a Tactic
- Other: Endgame Error · Missed a Defensive Resource


### Sub-cluster level = COACHING TOPICS, not fine slivers (decision 2026-05-30, revises earlier)

When asked "would you use these 12 endgame clusters to coach?", answer was NO — semantic
clustering fragments ONE skill into many (e.g. 5 near-dup "king race" clusters). A coach
organizes by SKILL TO LEARN, few meaningful topics. So:

- **Structure: category -> coaching TOPIC -> feature** (2-level). The cluster level = ~3-6
  coaching topics per category (what a coach actually teaches), NOT ~12-15 fine slivers.
- **SUPERSEDES the earlier "sub-clusters in 2-15 range" goal** — that target pulled the wrong
  way (kept slivers apart; coaching wants them merged). Topic size can be large (60+ feats) if
  it's one real skill.
- Method per category: read fine clusters -> drop misfits (boundary rule) -> CONSOLIDATE into
  the few coaching topics a teacher would name -> verify coherence.
- **Endgame Error done (template):** 3 topics — King & Pawn Technique (65), Passed Pawns &
  Promotion (36), Endgame Piece Activity (32). See output/taxonomy_v2/endgame_final.json.


**Model is pinned:** `maia3_sae_diff_v2_2048_k32_l2` (flat k=32, v2 corrected data). Verified labels align + fire rates in `output/taxonomy_v2/firerate_flat_v2_k32.npy`. See below for the earlier provenance hunt.

**Scheme exploration done:** `output/taxonomy_v2/TOP_LEVEL_SCHEMES.md` + `schemes_atlas.html` (the latter's category assignments are keyword-based → stale; structure/UI is the keeper). `chess_taxonomy_atlas.html` is STALE (old top-down 20-cat) — regenerate after assignment.

**Next:** robust reading-based assignment of 1996 features to the seed categories (small agent batches ≤10 to avoid the StructuredOutput stall seen at batch=50), then sub-cluster within each, then atlas. Then 3 QC passes (misfit reconciliation, member verification, coherence bar).

### ⚠️ Taxonomy needs a clean redo on the FLAT k=32 model (earlier 2026-05-30 notes)

Sam wants the taxonomy built on the **flat k=32 SAE** (`maia3_sae_diff_2048_k32_l2_200ep.pt` — 200ep, the "2007 labeled" champion per S3_INVENTORY), with **semantic sub-clusters inside each category** and **fire rates** (per feature, summed per cluster + category). Two problems with the existing `taxonomy_v2.json` must be fixed:

1. **Categorization was done top-down (WRONG ORDER).** Each feature was independently dropped into one of 20 pre-baked categories → magnet effect: Slow Play Punished got 408, Pieces Left Undefended got 4. The persona-atlas method (see `docs/knowledge/taxonomy-method-persona.md` — paste from Sam) is the correct one: **cluster FIRST on label-text semantics (bge-m3), let categories emerge bottom-up, one agent regroups within each.** No imposed buckets.
2. **No sub-clusters, no fire rates.** Both were asked for from the start; the flat→atlas only had category→feature.

### ⚠️⚠️ PROVENANCE BUG — taxonomy_v2 labels' source model is UNKNOWN (verified 2026-05-30)

`l2_feature_profiles_v2.json` (the profile the 2007 Opus labels were built from) was **NOT reproduced by ANY checkpoint I tested**: not flat k=32 (l2_200ep / v2 / base), not the H1 perlevel matryoshka. Verified via: forward-pass v2 cache (`maia3_blunder_diff_v2.pt`, idx 137471 == profile feat3 ex0 Bxf7+ ✓ so the *cache* is right), check whether feature 3's top firings = the profile's Bxf7+ set. ALL candidates gave 0/10.

**Implication:** the labels in `taxonomy_v2.json` are bound to a profile of unknown model origin → treat `taxonomy_v2.json` as **suspect, not a foundation.** Don't cite its per-feature category as ground truth.

**Clean path (next session):** regenerate from scratch on the flat k=32 model — fresh profile (`extract` top-20 per feature over v2 cache, flat top-k=32, z-score→L2 norm), join the 19K Opus English by `fen|uci`, then bge-m3 cluster → emergent categories → fire rates. One known model end-to-end = reproducible.

**Established facts (verified this session, trustworthy):**
- v2 corrected cache: `chess-poc:~/SageMaker/chess-stage-a/cache/maia3_blunder_diff_v2.pt` (200K×512, has `metadata` with fen/blunder_uci/cp_loss, NO stored mean/std — compute z-score then L2 per `label_v2_features.py`).
- Normalization that label scripts use: `x=(raw-mean)/std; x=x/||x||`.
- bge-m3 + sklearn available locally → semantic clustering runs locally, no Bedrock.
- 19K Opus English: `chess-poc:~/SageMaker/all_positions_labeled_opus.json` (19,342, keyed fen|uci). S3 `_final.json` truncated to 10,648 — DON'T use.

**Matryoshka SAE:** Per-level-k, dict=[32,256,2048]=2336, k_per_level=[3,8,16]. Zero dead, FVU=0.209. See `docs/knowledge/matryoshka-sae.md`. (Separate track; Sam wants flat k=32 for the taxonomy right now.)

**Next steps (taxonomy redo, in order):**
1. Generate fresh profile + per-feature fire rate from FLAT k=32 (`l2_200ep`) over v2 cache. Verify feat-N top firings look sane.
2. Embed each feature's label-text (chip+description, or fresh Opus labels) with bge-m3.
3. Agglomerative cluster → ~280 sub-clusters; name each from members.
4. Group sub-clusters → emergent categories (one agent per group, holistic, "name the type of mistake"). NOT independent per-feature assignment.
5. Fire rate summed per sub-cluster + per category (reach % + sum-rate %).
6. QC: misfit reconciliation + member verification + coherence bar (≥ the persona method's 0.593-style cutoff).
7. Rebuild atlas: category → sub-cluster → feature, fire rates at each level.

---

## Previous State (2026-05-23)

**Production SAE:** `puzzle_2048_k32_v1` — filtering by `coaching_useful` flag + `detection_accuracy >= 0.6`. 218 features served.

**Puzzle SAE champion:** BTK 2048 k=64 + aux, BA=0.632 — ready to deploy (Queue item 2).

**Blunder SAE winner: MT 2048 k=32** — 1,080 unique coaching labels, 65% label uniqueness, 1.56% median fire rate.

**NEW — Maia 3 SAE v2 (2026-05-24): RETRAINING on correct data.**
- v1 had a critical bug: ~50% of training positions (Black-to-move) had inverted blunder/best labels from the Lichess eval dataset sort. See knowledge.md § "Gemini labeling" for full details.
- v2 uses real-game blunder data (`blunder_positions.json`, 200K positions, already cached) which correctly identifies blunders for both colors.
- **Status:** Maia 3 activation extraction RUNNING on chess-poc (PID 22253, 16/s, ETA ~3h from 21:30 UTC May 24). Output: `maia3_blunder_diff_v2.pt`.
- **Next after activations:** Run `bash /home/ec2-user/SageMaker/run_pipeline.sh` to chain train + profile. Then Gemini CLI labeling locally.

**OLD Maia 3 SAE v1 (2026-05-22):** BTK 2048 k=32, diff pooling, L2 normalized. INVALIDATED by data bug.
- 2007 features labeled (Sonnet 4.6 + thinking). Labels unreliable due to mixed blunder/best-move training.
- Structural analysis showed: 41 hub features, 91% classified into 15 structural categories.
- Geometric python-chess labeling attempted — dead end (too shallow, mostly noise).
- Gemini batch ($0.05 actual cost, April promotional?) produced 5,851 position labels that cover all 2048 features.
- Cost audit: published rates would be ~$67-200 for similar batch. Batch API may have been free during preview.

Full sweep (9 variants trained, 5 labeled):

| Config | Alive | FVU | FR Med | Quality | Unique Labels | Verdict |
|--------|-------|-----|--------|---------|---------------|---------|
| 1024 k=16 | 1,023 | 0.155 | 1.56% | — | — | Too coarse |
| 1024 k=32 | 1,016 | 0.127 | 3.12% | — | — | Too coarse, 71% of 2048 missed |
| **2048 k=32** | **2,031** | **0.115** | **0.87%** | **1,670** | **1,080** | **WINNER** |
| 2048 k=16 | 2,040 | 0.144 | 0.78% | — | — | Unlabeled |
| 2048 k=64 | 2,033 | 0.093 | 2.00% | 2,984 | — | More redundant |
| 4096 k=32 | 4,009 | 0.107 | 0.35% | 3,447 | 1,914 | +834 unique but 44% redundant |
| 4096 k=64 | 4,027 | 0.085 | 0.84% | 2,984 | — | Diminishing returns |
| 4096 k=128 | 4,092 | 0.066 | 2.09% | 2,711 | — | Too many broad features |
| 8192 k=32 | 8,024 | 0.101 | — | — | — | Research only |

**Key findings:**
- Move-token-only (hidden[77]) fixed fire rates: 20-31% all-token → 0.8-3.1% move-token
- 60% high-confidence labels across all variants (up from 27% in old blunder SAE)
- Pairwise Jaccard 0.12-0.19 across variants — SAEs find different decompositions
- Within categories: features are unique (Jaccard <0.5), but labels are the bottleneck (40% get generic names)
- 1024 too small (misses 71% of 2048 coverage), 4096 diminishing returns (44% redundant)
- Top blunder categories: hanging pieces (20%), endgame technique (17%), passed pawns (11%), deflection (12%)

**Repo structure:** Everything in chess-deck-research now. See README.md.

## Beliefs
- [CONFIRMED] BTK is the only viable SAE architecture
- [CONFIRMED] Move-token (hidden[77]) >> mean-pooled or per-token-all
- [CONFIRMED] Puzzle-trained wins over blunder-trained (72% vs 27%)
- [CONFIRMED] Features are subtypes not duplicates (Jaccard ~0)
- [CONFIRMED] Versioned architecture works — swap SAE = new version dir
- [OVERTURNED] ~~k=32 gives right specificity~~ → k=32 too sparse without aux (57% dead). With aux, k=32 works fine (9% dead, 1,864 active).
- [CONFIRMED] Aux loss reduces dead features dramatically (57% → 10% at k=64)
- [CONFIRMED] FEN enrichment improves detection scoring (+0.048 mean BA, +141 STRONG features)
- [CONFIRMED] Judge quality (Haiku vs Sonnet) doesn't matter for detection scoring (+0.006, negligible)
- [CONFIRMED] Enrichment matters more than judge model for detection scoring
- [OVERTURNED] ~~Phase/piece diversity measures polysemanticity~~ → 95% false positive rate. Measures generality not polysemanticity.
- [OVERTURNED] ~~Dead features are bad~~ → Dead = unused capacity. Optimize for active count, not dead %.
- [MEASURED] 2048 k=64 + aux: Sonnet labels + enriched → mean BA 0.632, HOLDS 659, STRONG 325
- [CONFIRMED] Sonnet+thinking labels > Haiku labels: +0.013 BA, +36 STRONG, -67 FAIL
- [CONFIRMED] Aux loss fixes k=32 too: 57% dead → 9% dead (same effect as k=64)
- [MEASURED] 2048 k=32 + aux: 1,864 active, FVU=0.128, c_dec=0.045
- [MEASURED] 4096 k=64 + aux: 3,017 active, FVU=0.092, c_dec=0.035
- [CONFIRMED] k=32 + aux labels much less polysemantic than k=64: 3.5% vs 30.6%.
- [CONFIRMED] But k=64 wins on detection scoring despite higher poly rate (0.632 vs 0.557 BA). Poly ≠ quality.
- [CONFIRMED] 2048 >> 4096 per-feature detection quality. Extra dict capacity doesn't help.
- [MEASURED] Final: 2048 k=64 + aux = BA 0.632, 659 HOLDS, 325 STRONG. Winner.
- [UNTESTED] SAE feature diffs improve coaching output (A/B test needed)
- [CONFIRMED] Move-token-only >> all-token for blunder SAE (fire rate 2% vs 31%)
- [CONFIRMED] Blunder move tokens produce viable SAE structure (0.4-2.1% dead, FVU 0.066-0.115)
- [MEASURED] Pre-topk: 318 features naturally activate, top-64 = 60% energy, top-128 = ~75%
- [OVERTURNED] ~~Puzzles >> blunders for SAE training~~ Previous test was k=32 no-aux all-token. Move-token + aux changes the picture. Labeling pending.
- [UNTESTED] Blunder SAE features are interpretable (labeling will determine)

## Queue

### 0. Sonnet+thinking labeling (DONE)
- 1,872/1,961 parsed. 1,139 mono+high-confidence. 30.6% poly (correlated with uncertainty).
- Detection scoring: mean BA 0.632 (+0.013 vs Haiku labels), 325 STRONG (+36), 293 FAIL (-67)
- Sonnet labels measurably better than Haiku labels. Use Sonnet+thinking going forward.

### 1. Pick k and dict_size (DONE — 2048 k=64 wins)
All 4 variants profiled, labeled (Sonnet+thinking), and scored (Haiku + enriched):

| Config | Mean BA | HOLDS | STRONG | FAIL |
|--------|---------|-------|--------|------|
| **2048 k=64** | **0.632** | **659** | **325** | **293** |
| 4096 k=64 | 0.566 | 566 | 159 | 824 |
| 4096 k=32 | 0.563 | 537 | 155 | 854 |
| 2048 k=32 | 0.557 | 284 | 70 | 515 |

**Winner: 2048 k=64 + aux.** Best on every detection metric. Deploy this.

### 2. Deploy 2048 k=64 + aux as production SAE (NEXT)
Full plan: `lab/chess/website/plans/2026-04-12-deploy-sae-k64.md`
1. Convert weights .pt → .npz (Lambda uses numpy, not PyTorch)
2. Build labels.json (merge Sonnet labels + detection scores + profiles, set coaching_useful)
3. Create version dir `puzzle_2048_k64_v1/` with config.json, labels.json, sae_weights.npz
4. Read k from config.json instead of hardcoding 32 in app.py
5. Update active_version.json, run relabel.py
6. Smoke test locally
7. Deploy CDK

### 3. Blunder-trained SAE experiment (IN PROGRESS — labeling)
- **Hypothesis:** Move-token SAE on blunder moves clusters "what kind of mistake" patterns
- **Data:** 200K blunders (≥200cp loss) from Lichess eval dataset, move-token cache (804MB)
- **Scripts:** `cache_move_token.py`, `train_blunder_sae.py`, `profile_sae.py --move-token-only`
- **All 5 weights on S3** — see `output/S3_INVENTORY.md`
- Pipeline:
  1. ✅ 200K blunders collected from HuggingFace (16.1% hit rate, 13min)
  2. ✅ Move-token cache built (hidden[77] only, 804MB)
  3. ✅ 5 variants trained (2048×{k32,k64} + 4096×{k32,k64,k128}), 8-14s each
  4. ✅ All profiled — fire rates 0.35-3.15% median (all under 5% target)
  5. ✅ Labeling complete: all 3 batches (k=32, k=64, k=128). ~60% high confidence across all.
  6. ✅ Pairwise Jaccard: SAEs find different features (mean best 0.12-0.19)
  7. ✅ Quality filter: 2048 k=32 = 1,670 passing, 1,080 unique labels (65%)
  8. ✅ Within-category analysis: features are unique (Jaccard <0.5) but labels are bottleneck (40% generic)
  9. ✅ Dict size sweep: 1024 too coarse (misses 71%), 4096 diminishing returns (44% redundant)
  10. 🔄 **NEXT: Cluster fire patterns into 20-30 coaching categories**
  11. Relabel with coaching taxonomy (short_label, coaching_advice, theme/subtopic)
  12. Detection scoring on 2048 k=32
- **Winner: 2048 k=32** — best balance of unique labels and quality
- **Presentation problem:** 1,080 features → need 20-30 coaching categories → 5-6 player-facing themes
- **Approach:** Cluster fire patterns (not labels) via cosine similarity + hierarchical clustering. Then name clusters with Sonnet. Categories should map to Heisman's mistake taxonomy.
- See `output/blunder_sae_reasoning.md` for full design rationale.

### 4. Coaching taxonomy
- Cluster 2048 k=32 fire patterns into ~25 coaching subtopics
- Map subtopics to ~6 player-facing themes (piece safety, tactical awareness, endgame play, etc.)
- Relabel features with fixed taxonomy + short labels + coaching advice
- Taxonomy must be stable across SAE architectures (it's about chess, not features)

### 5. Deploy puzzle + blunder SAEs
- Puzzle SAE: 2048 k=64, BA=0.632 — deploy plan at `lab/chess/website/plans/2026-04-12-deploy-sae-k64.md`
- Blunder SAE: 2048 k=32 — deploy alongside after coaching taxonomy is set

### 6. Coaching A/B test
- 50 blunders. Coaching with vs without SAE feature context. Sam rates.

## Constraints
- chess-poc: ml.g6.16xlarge (L4 + 256GB RAM), account 140023406996
- Bedrock calls: account 140023406996, default profile
- Bedrock Batch supports thinking (tested 2026-04-12, needs ≥100 records)
- Opus doesn't support Bedrock Batch

## Pipeline (repeatable)
1. Cache activations (cache_activations.py — run once per 200K puzzles, ~15 min)
2. Train SAE (sweep scripts in chess-deck-research/scripts/sae/)
3. Eval structural metrics (eval_sae_checkpoint.py — dead, L0, FVU, c_dec)
4. Profile (profile_sae.py — top-20 examples per feature, ~5 min per SAE)
5. Enrich FENs (enrich_fens.py — Stockfish + python-chess, cached)
6. Label (batch_label_and_score.py label — Sonnet+thinking via Bedrock Batch)
7. Detection score (batch_label_and_score.py score — Haiku + enriched FENs)
8. Auto-flag coaching_useful (BA ≥ 0.6, FR ≤ 3.0, not polysemantic)

All scripts in `chess-deck-research` repo. Run on notebook via git pull.

## After Maia3 space investigation (May 2026)

### Findings (see knowledge.md for full detail)
- Current v2 SAE: diff is before-after-blunder, NOT blunder-best. Features describe what you played.
- Option-A (repr_best - repr_blunder): tactical clustering is real. Fork gap=0.04, capture gap=0.12.
- An SAE trained on Option-A WILL produce missed-fork/missed-capture/missed-quiet-tactic features.
- cp_loss doesn't cluster in this space — expected, not a bug. Severity = statistic per feature.

### Decision pending from Sam
Build Option-A cache (~3.5hr on chess-poc) and retrain SAE, OR continue finishing the
taxonomy categorization on the current v2 SAE first.

### If rebuilding Option-A cache
Script needed: encode v1 positions (200k, have best_uci + player elos), run Maia3 ONNX
on both resulting boards at player elo, diff mean64, save. Then retrain BatchTopK SAE.

## Current work (2026-05-31) — INVALID (v1-corrupted, see correction at top of file)

### SAE architecture search

> All three SAEs below were trained on the v1 cache (label-inversion bug) and have been deleted.
> Findings unverified. To revisit, rebuild on the v2 cache.

**Goal:** Find SAE where >50% of features have coherent coaching labels.
**Done criterion:** Feature descriptions match what a coach would say ("you hung your bishop," not "insufficient data").

**Three new SAEs built and trained:** (DELETED — v1-corrupted)
- `maia3_option_a_2048_k32.pt` — h[best_to] - h[blunder_to], layer-7 ONNX, 512-dim
- `maia3_board_diff_2048_k32.pt` — mean64(after_best - after_blunder), layer-7 ONNX, 512-dim
- `maia3_l2l7_2048_k32.pt` — concat(L2_mean64_diff, L7_mean64_diff), 79M PyTorch, 2048-dim

**Probe findings:** (derived on v1 data — needs re-verification on v2)
- board_diff best overall for mistake-type separation (mean gap 0.038 across 6 taxonomy categories)
- L2 better for positional mistakes, L7 better for tactical timing — l2l7 captures both
- v2 SAE (current): h[to_sq]-h[from_sq] of blunder — encodes what you played, not what you missed

**In progress:**
- Encoding 18k Opus positions through all 3 SAEs (chess-poc, ~80min remaining)
- After: Sonnet labeling pass using Opus descriptions as source
- After: Eval on 4 real positions from cabbagelover games

**Test positions (output/test_positions.json):**
- bishop_f5: 9...Bf5 → hung piece (463cp)
- queen_h4: 17.dxe4 → left piece undefended (453cp)
- knight_e4_trap: 5...Nxe4 → walked into tactic (331cp)
- qd5_king_exposed: 8...Qd5 → misplayed attack (425cp)

**Decision tree:**
- If l2l7 hits >50% coherent labels AND fires right features on test positions → use l2l7
- If none hit 50% → rebuild on v2 data (same constructions, v2 positions that match the 18k Opus labels)
