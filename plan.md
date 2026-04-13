# Chess Encoder — SAE Feature Pipeline

## Current State (2026-04-12, session 2)

**Production SAE:** `puzzle_2048_k32_v1` — filtering by `coaching_useful` flag + `detection_accuracy >= 0.6`. 218 features served.

**Best candidate:** BTK 2048 k=64 + aux loss
- Dead: 213/2048 (10%), Active: 1835
- Best detection: Sonnet labels + enriched FENs → Mean BA 0.632, HOLDS 659, STRONG 325, Top-200 0.886
- 1,872 Sonnet+thinking labels completed. 1,139 mono + high-confidence.
- 30.6% polysemantic (Sonnet audit) — correlates with uncertainty, not actual polysemanticity.

**New finding (2026-04-12):** Aux loss fixes k=32 dead features too.
- 2048 k=32 + aux: 184 dead (9%), 1,864 active — was 1,161 dead (57%) without aux
- k=32 vs k=64 with aux: similar active count (~1,850), k=32 more selective (L0=32 vs 64)
- 4096 k=64 + aux: 3,017 active features, but 26% dead
- 4096 k=32 + aux: 1,188 dead (29%), 2,908 active, FVU=0.126, c_dec=0.041

**Key insight:** Dead features aren't bad — they're unused capacity. 4096 with 26% dead = 3,017 active > 2048 with 10% dead = 1,835 active. Optimize for active count and quality, not dead %.

**Architecture:** Versioned single source of truth. See CLAUDE.md "SAE Labels" section.

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

### 3. Blunder-trained SAE experiment (IN PROGRESS)
- Previous finding (S32): blunder SAE at k=32 no-aux = 27% confident labels. But that was before aux loss.
- Hypothesis: k=64 + aux on blunder positions might produce useful "mistake pattern" features
- **Train on the played (bad) move** — cluster what players do wrong, not what they should've done
- Data: Lichess position evaluations (844M rows, HuggingFace `Lichess/chess-position-evaluations`)
  - Filter: eval drop ≥ 200cp (-2.0) from best move to played move
  - Encode: FEN + played_move (the bad move) → encoder → hidden[77] → cache
  - Cache 200K blunder activations
- Scripts committed to chess-deck-research:
  - `scripts/data_prep/cache_blunder_activations.py` — streams HF, filters, encodes, caches
  - `scripts/sae/train_blunder_sae.py` — loads cache, trains BTK + aux, saves checkpoint
- **Status (2026-04-13):** Structural metrics pass. Profiling in progress.
  - dead=89 (4.3%), alive=1959, L0=64, FVU=0.129, c_dec=0.034
  - Comparable to puzzle SAE (10% dead, FVU=0.082, c_dec=0.036)
  - S3: `s3://chess-stage-a-140023406996/sae-weights/sae_btk_blunder_2048_k64_aux.pt`
- Pipeline:
  1. ✅ Scripts written and committed
  2. ✅ 200K blunders cached (Phase 1: 13min download, Phase 2: 40min encode, 60GB cache)
  3. ✅ BTK 2048 k=64 + aux trained (5 epochs, 306s)
  4. ✅ Structural metrics pass
  5. ✅ All-token profiling: fire rates too high (31% / 20%). Root cause: training on all 77 tokens instead of move token.
  6. ✅ **Move-token-only retraining** — built cache_move_token.py, retrained on hidden[77]
     - MT 2048 k=64: alive=2,033, **fire rate 3.15% mean, 2.00% median** ✅
     - MT 4096 k=64: alive=4,027, **fire rate 1.59% mean, 0.84% median** ✅
  7. 🔄 Labeling: move-token batch `mjgqyjem1w28`, all-token batches `wtewr9qxt9gy` + `63ouxzbuzjh2`
  8. Detection scoring (after labels)
  - Fixed IAM: SageMaker role can now PassRole for Bedrock Batch
- Also test: 100K puzzles + 100K blunders mixed training
- Cheapest test first: structural metrics only (~5 min). If promising, full pipeline.

### 4. Coaching A/B test
- 50 blunders. Coaching with vs without SAE feature context. Sam rates.

### 5. Coaching LLM (BLOCKED on good labels)
- Gemma 4 E2B fine-tuned on SAE labels → coaching commentary

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
