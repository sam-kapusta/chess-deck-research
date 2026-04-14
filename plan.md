# Chess Encoder — SAE Feature Pipeline

## Current State (2026-04-13)

**Production SAE:** `puzzle_2048_k32_v1` — filtering by `coaching_useful` flag + `detection_accuracy >= 0.6`. 218 features served.

**Puzzle SAE champion:** BTK 2048 k=64 + aux, BA=0.632 — ready to deploy (Queue item 2).

**Blunder SAE winner: MT 2048 k=32** — 1,080 unique coaching labels, 65% label uniqueness, 1.56% median fire rate.

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
