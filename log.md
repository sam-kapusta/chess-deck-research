# Chess Lab — Log

## 2026-04-14 (session 3 continued) — Blunder SAE full sweep + categorization

**Big session.** 9 SAE variants trained, 5 labeled, full comparison, winner selected, categorization explored.

### Results
- **Winner: 2048 k=32 move-token** — 1,080 unique labels, 65% uniqueness, 1.56% median FR
- All variants got ~60% high-confidence labels (up from 27% in old blunder SAE)
- Move-token fix was key: all-token → 20-31% FR, move-token → 0.8-3.1% FR
- 1024 too coarse (misses 71% of 2048), 4096 diminishing returns (44% redundant), 8192 structural fine but overkill

### Labeling
- 3 Bedrock Batch jobs: k=64 (`mjgqyjem1w28`), k=32 (`ypr3017mqa9s`), k=128 (`9m6cs1aioq3k`) — all completed
- Top categories: hanging_pieces (20%), endgame_technique (17%), passed_pawn (11%), deflection (12%)
- Within-category Jaccard: features are distinct (<0.5) but labels are bottleneck (40% get generic names)

### Analysis
- Pairwise Jaccard (full matmul, no sampling): 0.12-0.19 mean across SAE pairs. Features are unique across SAEs.
- Pre-topk energy: 318 features naturally activate, top-64 = 60% energy, top-128 = 75%
- Greedy set cover: 22 features cover 95% of positions BUT top features fire 15-20% (too broad)
- Fire-pattern clustering: endgame features cluster cleanly, tactical features don't (positions overlap)
- Key insight: "overloaded defender" and "hanging piece" fire on overlapping positions — same blunder from different angles

### Categorization (in progress)
- Sonnet's categories are good but flat (22 categories, no hierarchy)
- Dedup at Jaccard 0.8 + clique grouping at 0.3 — script running
- Endgame features → clean coaching topics. Tactical features → need tags not categories.
- Explored: Heisman mistake taxonomy, greedy set cover, hierarchical clustering, decoder-direction clustering
- Decision: dedup → group → relabel groups with coaching-focused prompt (short_label, coaching_advice, theme/subtopic)

### Infrastructure
- Repo consolidated: everything in chess-deck-research (plan, log, findings, learnings, scripts, output, docs, archive)
- Cleaned hooks: 11 → 4 (session-start, post-compact, drift-nudge, anti-poll)
- Updated /organized skill: added S3, git commit, naming, two-phase, cheap-before-expensive habits
- Fixed IAM: SageMaker role can now PassRole for Bedrock Batch
- CLAUDE.md updated to point to chess-deck-research for SAE research

### Scripts committed
- `compare_saes.py` — full matmul Jaccard
- `quality_filter.py` — confidence + FR + mono filter
- `within_category_jaccard.py` — redundancy within categories  
- `pretopk_energy.py` — natural sparsity analysis
- `label_breakdown.py` — category comparison
- `cluster_features.py` — hierarchical clustering
- `greedy_feature_selection.py` — set cover
- `dedup_and_group.py` — Jaccard dedup + clique grouping
- `label_blunder_coaching.py` — coaching-focused labeling prompt
- `cache_move_token.py` — extract hidden[77]

### Next session
- Check dedup_and_group.py results (running on notebook)
- Relabel grouped features with coaching prompt
- Detection scoring on 2048 k=32
- Deploy puzzle SAE (Queue item 2 still waiting)

## 2026-04-13 (session 3) — Blunder SAE experiment

- Starting blunder-trained SAE experiment (Queue item 3 from plan.md)
- Scripts: `cache_blunder_activations.py` (two-phase: download+filter → batch encode), `train_blunder_sae.py`
- Fixed original caching script: wrong BASE path, no batching, print bug at 5K boundary
- Rewrote as two-phase pipeline: Phase 1 = CPU-only HuggingFace streaming, Phase 2 = batched GPU encoding
- Notebook: chess-poc (ml.g6.16xlarge, L4 GPU, 242GB RAM)
- Phase 1: 200K blunders from 1.24M positions (7.2M rows) at 247/sec, 16.1% hit rate, 809s total
- Phase 2: 400K forward passes (blunder + best), batch_size=64, ~20 min per pass
  - Blunder encoding: 200K in 1208s
  - Best encoding: 200K in 1220s
  - Cache saved: 60GB (`blunder_acts_200k.pt`)
- Training BTK 2048 k=64 + aux on 15.4M blunder activations (200K × 77 tokens), 5 epochs, 306s
  - ep0: mse=0.151, dead=1 → ep4: mse=0.136, dead=3
  - Final eval: dead=89 (4.3%), alive=1959, L0=64, FVU=0.129, c_dec=0.034
  - **Structural metrics pass** — comparable to puzzle SAE (10% dead, FVU=0.082, c_dec=0.036)
  - Fewer dead, more alive, higher FVU (blunders harder to reconstruct but features are well-separated)
- Weights saved to S3: `s3://chess-stage-a-140023406996/sae-weights/sae_btk_blunder_2048_k64_aux.pt`
- Profiling: 2,048 alive, mean fire rate 31.1% (higher than puzzle SAE — blunders more diverse)
- Profiles saved: `s3://chess-stage-a-140023406996/sae-eval/profiles_btk_blunder_2048_k64.json`
- Fixed IAM: added PassRole permission to ChessResearchSageMakerRole for BedrockBatchInferenceRole
- Sonnet+thinking labeling batch submitted: `wtewr9qxt9gy` (2,048 features), combined batch `63ouxzbuzjh2` (5,900)
- **High fire rate (31% mean for 2048, 20% for 4096) is too high** — target is <5%
- Root cause: trained on all 77 tokens (64 board + extras), but production uses only hidden[77] (move token)
  - Board tokens = "what position is this" → fires broadly across position types
  - Move token = "what kind of move is this" → should fire selectively on move patterns
- Previous "move-token-only" attempt used wrong token (index 76 = last FEN token, not index 77 = move token)
- Built `cache_move_token.py`: extracts only hidden[77] from encoder output, ~400MB cache
- Also trained 4096 k=64 on all tokens: dead=836 (20%), alive=3260, FVU=0.125
- Correct move-token pipeline: cache_move_token.py extracts hidden[77] from encoder output
  - Move-token cache: 804 MB (vs 60GB for all-token cache)
  - Training instant: 8-14 seconds for 200K activations
- **Move-token results:**
  - 2048 k=64: dead=9, alive=2039, FVU=0.093, **fire rate mean=3.15%, median=2.00%** ✅
  - 4096 k=64: dead=26, alive=4070, FVU=0.085, profiling in progress
  - Move-token fixed the fire rate problem (was 31% all-token → 3.15% move-token)
- **Move-token profiling results:**
  - MT 2048: 2,033 alive, 15 dead, **fire rate mean=3.15%, median=2.00%** ✅
  - MT 4096: 4,027 alive, 69 dead, **fire rate mean=1.59%, median=0.84%** ✅
  - Both within <5% target. 4096 is more selective (sub-1% median)
- Uploaded to S3: `sae_btk_blunder_mt_2048_k64_aux.pt`, `sae_btk_blunder_mt_4096_k64_aux.pt`
- Move-token labeling batch submitted: `mjgqyjem1w28` (6,060 features)
- Also running all-token Sonnet labeling batches: `wtewr9qxt9gy` (2048), `63ouxzbuzjh2` (both) — for comparison
- Updated /organized skill with S3, git commit, naming conventions

## 2026-04-12 (session 2) — Sonnet labels, detection comparison, k=32 aux, 4096 sweep

- Sonnet+thinking labeling completed (`pztzjp2jzh8v`): 1,872/1,961 parsed, 30.6% poly flagged
- Detection scoring with Sonnet labels (`ac6bc19768ax`): BA 0.632 (+0.013 vs Haiku labels), 325 STRONG
- Full 5-condition detection comparison: enrichment >> judge quality >> label quality
- 4096 k=64 + aux completed: 3,017 active, 1,079 dead (26%), c_dec=0.035
- 4096 k=128 + aux crashed epoch 3 (GPU OOM on eval)
- 2048 k=32 + aux: 184 dead (9%), 1,864 active — aux fixes k=32 same as k=64 (57% → 9%)
- 4096 k=32 + aux: reran solo (earlier crash was memory pressure from sequential sweep). 2,908 active, 1,188 dead (29%), FVU=0.126, c_dec=0.041
- Organized research scripts into chess-deck-research GitHub repo, proper git workflow
- Scripts committed: eval_sae_checkpoint.py, parse_batch_results.py, profile_sae.py, sweep_k32_aux.py, profile_and_label_all.py
- Profiling 3 variants running (2048 k=32, 4096 k=32, 4096 k=64) — all from git repo scripts
- Next: upload profiles → Sonnet+thinking labeling batch → enriched detection scoring → compare all 4
- Created /organized skill: "never run code on remote that isn't a committed script"
- Consolidated drift-nudge hook to invoke /organized every 50 tool calls
- Notebook resized to g6.16xlarge (256GB) for 30GB cache loading, don't downsize yet
- Key insight: dead features aren't bad, active count is what matters
- 4096 profiling fixed: encoder weight key auto-detection (linear/w vs query/kernel), batch_size=32
- 4096 k=32 profiled: 3,287 alive, 809 dead, fire rate mean=10.35% median=2.00%
- 4096 k=64 profiling in progress (16K/50K)
- 2048 k=32 labeling batch submitted (`9tve7y1jz72h`), Sonnet+thinking, stuck in Scheduled >1hr
- Fixed batch_label_and_score.py: max_tokens must be > thinking.budget_tokens
- All 4 variants labeled with Sonnet+thinking: k=32 poly=3.5%, k=64 poly=30.6%
- Detection scoring complete: 2048 k=64 wins (BA=0.632, 659 HOLDS, 325 STRONG)
- k=32 less polysemantic but worse on detection — poly ≠ quality
- 4096 worse per-feature than 2048 — extra dict capacity doesn't help
- All results in chess-deck-research/output/COMPARISON.md

## 2026-04-12 (session 1) — k=64 + aux loss, enrichment pipeline, polysemantic validation

- Trained BTK 2048 k=64 with auxiliary dead-feature loss on 200K puzzles. Dead: 57% → 10%.
- Added aux loss to all 3 training scripts (train_tactics_sae, train_encoder_sae, train_and_profile_all)
- Built FEN enrichment pipeline (enrich_fens.py): Stockfish engine pool (8 parallel) + python-chess tactical annotations (forks, pins, back rank, overloaded, skewers, discovered attacks, promotion threats, eval delta)
- Enrichment cache: 17,923 unique FENs enriched, cached to disk
- Detection scoring 3-way comparison: Haiku+raw (0.571), Sonnet+raw (0.577), Haiku+enriched (0.619). Enrichment wins, judge doesn't matter.
- Updated batch_label_and_score.py: prefill forces array output (fixed 92% parse failure), enrichment wired to both labeling and scoring, persistent output to research/output/
- Polysemantic validation: 19/20 features flagged by diversity metric are actually monosemantic. The metric is invalid — measures generality not polysemanticity.
- 10-feature comparison: Sonnet+thinking > two-pass > Haiku 1-pass for label specificity ("Royal fork" vs "Winning captures")
- Confirmed Bedrock Batch supports thinking (needs ≥100 records)
- Submitted Sonnet+thinking labeling on all 1,961 features with polysemantic audit (job pztzjp2jzh8v)
- Production SAE Lambda updated: filters by coaching_useful flag, threshold raised to BA ≥ 0.6, fire_rate ≤ 3.0. 218 → will update after new labels.
- Built cache_activations.py and train_from_cache.py for fast SAE iteration (skip re-encoding)
- All baseline data saved to research/output/k64_baseline/ with README

## 2026-04-11 — Detection Scoring + K-Sweep + "Sparse but Wrong"

- Built T3b detection scoring framework (`research/scripts/detection_scoring.py`), adapted from Sandstone's evaluation methodology
- Scored all 395 production labels: mean BA=0.650, 35% FAILED (near-random)
- Fire rate vs BA: Spearman r=-0.166 — common features are polysemantic and unlabelable
- Tried contrastive relabeling on 169 WEAK features: +0.005 BA — negligible. Problem is features not labels.
- Found "Sparse but Wrong" paper (Chanin & Garriga-Alonso 2025): c_dec proxy for optimal L0
- Ran k-sweep (8,16,32,64,128,256) on chess-poc: c_dec monotonically decreasing, k=32 confirmed too sparse (57% dead features)
- Training 4096×{128,256} SAEs on chess-poc — in progress
- Will profile all 4 variants, label via Bedrock Batch, detection score, pick champion
- detection_accuracy baked into production labels.json for frontend filtering
- Key insight: the "k=32 is ideal" belief was wrong. Higher k = more alive features = more specific = more labelable
- Trained 4096×{128,256} on chess-poc, profiled all 4 variants
- Created BedrockBatchInferenceRole IAM role for Bedrock Batch
- Submitted labeling batch (10,255 features, Haiku) — completed successfully
- Submitted detection scoring batch — completed but 90% parse failures (Haiku doesn't follow "return ONLY list" instructions)
- Serial scoring on btk_2048_k128: BA=0.600 (n=13, below production 0.650)
- **Result: new variants did NOT beat production on T3b.** But comparison is confounded — production used Sonnet labeling, new used Haiku. Need Sonnet labeling on new SAEs for fair comparison.
- Also confounded by 50K vs 150K training data.
- Next: relabel k=128/256 variants with Sonnet (same quality as production), retrain at 200K, re-score

---

## 2026-04-09 — SAE Feature Relabeling Sprint

**Problem:** Practice page showed "B+K checkmate in 154 games" — completely wrong. Top FENs were generic middlegame positions. Labels were hallucinated from 5 examples.

**Per-feature normalization:** Each feature's strength / its own historical max, threshold 0.2. Replaces per-moment normalization. `featureMaxStrengths.json` generated from DDB scan (128 features, max range 0.5-25.9).

**Relabeling pipeline:**
1. Downloaded canonical profiles from SAIS chess-poc: 395 features × 20 FEN examples
2. Sonnet 4 pass 1: all 395 features → 394 high confidence but 181 got "Creating Multiple Simultaneous Threats" (lazy)
3. Sonnet 4 pass 2: differentiation prompt on 230+ duplicates — "what makes THIS feature different?"

**Practice page fixes:** Hero → top 10 worst features (2×5 grid), removed blue accent, per-feature norm, small category merging, redistributed SAE colors.

---

## 2026-04-08 — Corpus baselines + Practice page

**Pipeline built and run:**
- Sampled 5,000 Lichess rapid games (1K per rating band: 1400-1600, 1600-1800, 1800-2000, 2000-2200, 2200+) from June 2016 monthly dump
- Stockfish depth 14 analysis on ml.c5.9xlarge (36 vCPU, ~2h, zero failures)
- SAE feature extraction on ml.g5.2xlarge (A10G, ~15 min with per-token)
- Three SAE runs: (1) Maia SAE mean-pooled (wrong SAE), (2) Puzzle SAE mean-pooled (wrong pooling), (3) Puzzle SAE per-token (still wrong — used all 77 tokens, not move token)
- Discovered production Lambda uses **move token (hidden[77])**, not mean-pool or per-token-all

**Critical findings:**
- Production pipeline: FEN + UCI move → 79 tokens → encoder → hidden[77] (move token) → SAE → top 5 features
- Corpus script was using mean-pooled 77 tokens → completely different activation distribution → non-comparable features
- DeepMind 270M checkpoint loading: tensorstore pip wheel doesn't include OCDBT driver. Fix: `zarr` driver over `ocdbt` kvstore, save as npz. Should have cached npz the first time it ever loaded.
- Per-token fires ~127 features per position (vs ~5 for move token). Categories saturate at 100% of games when using per-token.

**Practice page shipped:**
- Merged /patterns + /drill into single /practice page with "Drill" nav tab
- Hero card + 2x2 category grid + sidebar (accuracy/rating trends, radar, drill scores)
- Rating band toggle: My Games | 1400 | 1600 | 1800 | 2000 | 2200+
- PF-ICF ranking: categories sorted by multiplier (your % / baseline %)
- Library cards redesigned: pokemon card ratio, SAE feature moments, cburnett pieces

**What's still wrong:**
- Corpus features don't match user features (per-token-all vs move-token). Need re-run with FEN+move→hidden[77] to match Lambda.
- 6 categories missing from baseline (captures, deflection, opening, piece_activity, quiet_moves, zwischenzug)

## 2026-04-04 (Bridge experiments + SAE deep dive)

- DeepMind 270M loaded on GPU (360 pos/s). Discovered: model is BIDIRECTIONAL (use_causal_mask=False), our reimplementation was wrong (causal mask bug). All prior SAE/probe results used corrupted activations.
- Model expects 79 tokens (FEN + move + return_bucket), not 77. It's an action-value evaluator, not a move predictor.
- SAE experiments: trained on puzzles + general positions. Key finding — SAE is wrong tool for tactics, raw activations beat SAE on every theme.
- Bridge experiments v2-v4: move prediction plateaus at ~1.3 regardless of encoder config (causal/bidirectional, 1/77/79 tokens). Bottleneck is task difficulty.
- Win probability 3-class: loss 0.59 at step 360, below random 1.10. **Bridge works for evaluation signal.** Phased curriculum validated.
- MATE dataset discovered: 592K positions with English strategy/tactics annotations. Training data for phases 3-5.
- MATE model discovered: LLaMA-3.1-8B fine-tuned on MATE, MIT license. But it's a reader, not writer.
- CCC paper (NAACL 2025): 0.60 correctness with concept extraction → GPT-4. Our Path B baseline to beat.
- Hybrid architecture proposed: bridge + diff vector concept probes. Probes as guardrails for LLM.
- Lab infrastructure: scientist/engineer modes, file-watch hooks, agent definitions, loop-based monitoring.
- Compressed plan.md from 202 → 64 lines. Archived 8 stale docs.

## 2026-04-01 (SAE sweep + rating gradient session)

- Maia SAE 2048/k=32/200K/50ep completed. Coherence: 0.4% @1.2 (useless). Structural metrics healthy. → DECISION: coherence metric broken for high-dim hidden states, use concept-correlation instead.
- Concept-correlation labeling: 799/2048 features labeled with 54 chess concepts. 27% with |corr| ≥ 0.1. Top: fianchetto (0.48), queenside castling (0.44), open files (0.42). → see findings.md § Concept Labels
- Rating gradient analysis: 5K positions × 4 ratings (1100/1400/1700/1900). 295 features increase, 387 decrease with rating. Higher-rated Maia activates more features (698→748 at >1% fire rate).
- Feature investigation: top unlabeled gradient features reveal perception shift. 1100 Maia → uncertainty/crisis/material fixation. 1900 Maia → coordination/targeted danger/piece activity. → see findings.md § Rating Gradient
- Debated puzzles vs general positions for SAE training. DECISION: general positions (matches Maia's training distribution, covers all position types). Puzzles as diagnostic.
- k-sweep complete: k=16/32/64/128, all zero dead. k=128 best for concept interpretability (60% labeled vs 38% at k=32). Contradicts Sandstone k=32 finding.
- Encoder SAE v2 trained (dict=2048, k=32, 27K positions). Still zero concept correlations → BUT test likely invalid (FEN alignment broken, 27K/200K extraction with no FEN file saved).
- Steelmanned encoder finding: position alignment, mean-pooling spatial destruction, limited concept list. Downgraded from CONFIRMED to INVALIDATED. → see findings.md § Encoder SAE
- k=128 full concept labels: 1278/2048 features labeled (62.4%). piece_count dominates (382 features). Strong king safety, pawn structure, piece activity coverage.
- k=128 rating gradient: 940 increasing + 789 decreasing = 84% of features show rating gradient (vs 33% at k=32). Features are more specialized.
- DeepMind searchless_chess 270M model downloaded (JAX/Orbax format). adamkarvonen model was wrong (512-dim, not 1024). Encoder re-test blocked on JAX setup.
- Final coaching feature analysis: 60 features with both concept labels AND rating gradients. Clear pattern: 1100→1900 = counting→understanding.
- All results saved locally to research/sae/: k128 checkpoint, concept labels, gradients, sweep results.
- DECISION: k=128 on 20K positions is the production SAE. Better than k=32 on 200K.
- Puzzle diagnostic complete: SAE correctly differentiates tactical (check, forced) from positional (opening, pawn shield). Validates general-position training decision. → see findings.md § Puzzle vs General
- JAX + orbax installed. DeepMind 270M checkpoint downloaded but loading needs searchless_chess codebase. Deprioritized — Maia results are strong enough.
- All results saved locally to research/sae/.
- REVISED: k=32 is best for coaching (not k=128). Sam's insight: tags should fire 1-5% of positions. k=32 = 1.6% (good), k=128 = 6.2% (too broad).
- Rating Progression Guide written → docs/2026-04-02 Rating Progression Guide (SAE-derived).md
- Key coaching insight: 1400→1900 = stop over-focusing on center pawns, start evaluating king safety through pawn shields, spot passed pawns early, understand rook activity beyond open files.
- Autonomous pipeline queued: 200K extraction → opening concepts → FEN extraction → k=128 200K training.
- 552 production labels created (38 unique concepts) → research/sae/maia_2048_k32_concept_labels.json
- Sonnet interpretation: 551 features labeled via Bedrock ($1-2, 174s). 540 unique labels. Rating-aware framing: "Learn..." for increasing, "Beginners..." for decreasing. → research/sae/maia_2048_k32_final_labels.json
- Neutral labels created: 197 rating-aware labels ("Beginners...") rewritten to neutral position descriptions ("Kingside castling with standard pawn shield"). Rating-aware preserved in `rating_aware_labels` field.
- position_features.py updated: new SAE path (2048 k=32), new labels, removed threshold (all 32 topk features are meaningful).
- Tests updated for new paths. MCP checkpoint updated.
- All code changes ready for commit.
- Fargate worker wired: Dockerfile adds torch+maia2, copies SAE checkpoint+labels, server_worker.py computes `sae_features` per moment. Graceful degradation if torch unavailable. ~10 lines changed.
- Tag→SAE correlation: 27 tags tested on 10K moments. `undeveloped_pieces` r=0.30 (strongest). F1438 is a "missed tactic" multi-tag detector (quiet_when_winning + missed_check + missed_overloaded). → see findings.md § Tag → SAE
- Top FENs extracted: 2041/2048 features with ≥3 top-activating positions.
- Key unlabeled features (F1438/F1555/F1281/F886) interpreted with Sonnet + actual FENs. All now labeled.
- Opening concept correlation: only 2/10K position matches (dataset is mid/endgame, not openings). Not useful.
- 200K activations cached at 4 ratings. Notebook idle, all work complete.
- Final label count: 554→561 features labeled after full tag correlation (64 tags including computed positional + opening detection).
- KEY: SAE detects specific openings — Sicilian (r=0.38), French (r=0.38), Kings Pawn (r=0.43). Also bad_bishop (0.18), passive_rook (0.34), weakened_pawn_shield (0.33). undeveloped_pieces jumped to r=0.65 with more examples.
- Mechanical labeling: extended concept list to ~100 concepts (endgame types, pawn structures, piece coordination, strategic themes). 1885/2041 features labeled (92%). Remaining 156 are compositional/unique patterns.
- Haiku-refined all 2041 features using top 5 FENs + concept guidance ($1 total, 370s). 2008+ labeled, 1818 unique. 100% coverage.
- Final labels: specific chess concepts — "French Defense with closed center", "Rook endgame with outside passed pawn", "Tactical melee with multiple forcing moves".
- MCP engine updated: `get_sae_features` now returns labeled tuples (fid, strength, label). Labels loaded from production file.
- Two new MCP tools: `check_game_stockfish` (pure eval) and `check_game_full` (eval + SAE labels). Both accept SAN moves directly.
- Standalone game analyzer: `research/sae/analyze_game_sae.py` — tested on Ruy Lopez and Sicilian games.
- Analyzed Sam's game (chess.com/game/live/166695486394): 17...Qf6 was the critical miss (41% of 1900s find Rxc2). 21...Kh8 understandable (89% play it). Post-move-22 mistakes are engine-only.
- Maia rating comparison integrated into `check_game_full` MCP tool — for each mistake, shows "would a 1100/1400/1700/1900 find the right move?"
- `check_game_full` returns: eval, best move, SAE features, Maia rating comparison, coaching verdict per mistake.
- 8 categories + 695 chip labels (2-3 words) for UI display ($0.30 Haiku, 218s). Endgame 26%, Tactics 19%, Piece Play 12%, Opening 11%, Rooks 11%, Strategy 9%, King Safety 8%, Pawn Structure 4%.
- position_features.py returns chip + category per feature. Data structure: `{label, chip, category, strength, feature_id}`.
- Frontend wired: `sae_features` added to Moment type, passed through both `buildMomentPayload` and both `streamCoachOverview` call sites. Backward-compatible (empty array for pre-SAE games).
- Full end-to-end pipeline: Fargate→DynamoDB→Frontend→Backend prompts→LLM. All code ready.

## 2026-04-01 (research session — lab manager)

- Deep research on chess LLM landscape: 50+ papers, 5 competing implementations. Wrote comprehensive fusion research doc.
- Steelmanned against own thesis (CLIP gap is insurmountable) — evidence from multimodal literature says it's bridgeable. Changed position from 20/80 against to 60/40 for.
- Ablation results came in: encoder IS the signal (constant output without, varying with). 54.5% vs 50% = ~4.5% delta. Architecture validated, data is bottleneck.
- Key discovery: FEN tokenizer expands to 64 individual board square tokens. The mean-pooled contrastive training DESTROYED this spatial structure. The translator was only trained on eval direction (one scalar) via mean pooling.
- Wrote board reading test (`research/encoder/scripts/test_board_reading.py`). This is the blocking experiment — determines whether translator carries spatial info or just a faint eval residual.
- Brainstormed 6 data paths + curriculum approach. Sam proposed game-level eval ("find 6 critical moments") as the real benchmark for chess understanding.
- SAE vs translator debate: SAE decomposes encoder knowledge into interpretable features for any LLM. Translator preserves holism but LLM can barely read it. Both have merit — board reading test determines which path.
- Reorganized lab structure: merged chess-research lab, consolidated docs, renamed findings→learnings, updated lab skill, created workspace/lab/ symlinks.
- Docs written: What We're Building, What We Don't Know, Translator Signal Analysis, Data Paths Brainstorm, Board Reading Test spec.

## 2026-04-01 (building session)

- LoRA v1 training completed (25K steps, 434 min, loss 0.018). Output: adapter_model.safetensors 617MB + proj_norm.pt
- Eval: 100% degenerate output (`!!!...`). Every position, same output. → see learnings.md § F10
- Debug: logits all NaN, argmax(NaN)→token 0 (`!`). Text-only (no chess) also broken — LoRA weights corrupted.
- Root cause: LayerNorm gives per-token norm sqrt(3584)=59.8, Qwen expects 0.94. 60x mismatch. nan_to_num masked the NaN losses.
- v2 training launched (10K, scale_factor=0.0157). Step 0: chess_norm=0.94, text_norm=0.96. Loss: 2.97→1.39 at step 400. NaN: 3/450.
- Loss plateaued ~1.47-1.58 from step 400-950. Expected with 10K examples (overparameterized).
- Caught critical bug: eval_lora.py used `scale_factor` in generate() but it was local to main(). Would have crashed auto-eval. Fixed.
- Auto-eval watcher staged in screen. Background poll running locally.
- All docs updated: design doc, projection signal analysis, research README, STATUS.md, plan.md.
- LoRA v2 training complete: 5000 steps, 86.5 min, final loss 1.2035, 36 NaN (0.7%).
- Loss trajectory: 2.97→1.39(400)→1.55(1000, plateau)→1.30(3600)→1.20(5000). Cosine LR recovered from plateau.
- Eval running (200 positions). Partial results at 80 positions: eval_direction=61%, format=100%, move_legal=6%, best_move=0%.
- Model outputs structured text ("Eval: +1.1. Best: d7d6") but defaults to common moves (d7d6, d2d4). Eval direction above random.
- Ablation partial results (140/200): eval_direction oscillating 45-54%, centering at ~50%. Normal was 54.5%.
- Ablation output is CONSTANT across all positions — same text every time. Encoder IS the only position signal.
- Delta: ~4.5%. Real signal, but p≈0.10 and 0% move accuracy = not useful for coaching.
- DECISION: Park encoder projection. Pivot to Maia (L2). Notebook stopped. → docs/2026-04-01 Encoder Experiment Retrospective.md
- Read Sam's brainstorm doc → revised decision: architecture works, DATA is the bottleneck. Multi-task (Path D) with existing data.
- Extracted 21,341 multi-task moments from DynamoDB (4,073 games). 27 tags, 6 fields per moment. → research/data/multitask_moments.jsonl
- New plan: Maia (product) + multi-task encoder training (research) in parallel.
- Board reading test: 0/100. Model outputs training format regardless of prompt. Can't read pieces.
- CRITICAL: Eval direction bias check (E15). Training data is 55.3% positive. Always-positive gets 55.3%. Our model gets 54.5% — BELOW trivial baseline. The "4.5% delta vs ablation" was comparing against a parser artifact (ablation had no eval field → default 0 → ~50%).
- Revised understanding: encoder transmits position-varying signal (proven) but that signal doesn't beat trivial baselines (also proven). The architecture works for transmission, fails for useful prediction.
- Wrote 6 analysis docs capturing the full reasoning chain.
- Two cheap probes queued: per-token projected embeddings (spatial info?) and LLM hidden states (where does signal die?).
- Notebook still running ($1.40/hr). Should stop if probes aren't run today.
- Per-token probe results (raw encoder): occupied 68.6% (baseline 64.2%), piece type 65.1% (baseline 64.2%), color 66.0%. Piece type is barely above baseline — encoder doesn't have per-square info after attention layers.
- REFUTED: per-token alignment path. Encoder computes board-level features, not square-level. The transformer attention distributed spatial info across all tokens.
- Consolidated docs: 20 → 10. Archived 10 to archive/docs-2026-04-01/.
- Projected representation: occupied 66.6% (vs raw 68.6%). Piece type still computing when stopped.
- Partial projected result: 66.6% ≈ 68.6% raw = projection preserved what little per-token info exists.
- DECISION: Per-token alignment eliminated. Mean-pooled with richer signals + SAE are the remaining encoder paths.
- Notebook stopped. Next encoder experiment: multi-task training on 21K moments (data already extracted).
- Pivoted to SAE path. Ran 5K positions through existing Maia SAE. 1020/1024 features active.
- Interpreted 20 more features (total 46/1024). Features are meaningful: "king under attack," "centralized knight," "undefended pieces."
- Proof of concept: SAE features on 4 test positions match human chess understanding. Complex middlegame → knight/king attack features. Fork position → "converting advantage requires precision."
- Wrote SAE vs Projection comparison doc, SAE plan, research/sae/ directory with README + labels JSON.
- SAE sweep: 512/1024/2048/4096 on 20K Maia activations. 4096 wins: 84.8% explained variance, zero dead features, orthogonal decoder columns. All trained in <20s on CPU.
- 4096 SAE differentiates position types (endgame/opening/middlegame get different top features). Needs interpretation (4096 new feature IDs, no labels yet).
- Structural metrics all healthy: no dead features, near-zero decoder cosine sim, constant 6.25% fire rate across all sizes.
- Encoder SAE trained on notebook: 4096 (MSE 0.053) and 8192. Float32 fix required (half precision NaN on some positions).
- Both Maia SAE (local) and encoder SAE (SAIS) checkpoints downloaded locally to research/sae/.
- Notebook stopped. Total SAE work: 5 Maia sizes + 2 encoder sizes trained, all zero dead features.
- KEY RESULT: Feature overlap test on Maia 4096 SAE. K+P endgame vs tactical = 0/50 overlap. Endgame vs exposed king = 0/50. Features are highly position-specific and monosemantic.
- Profiled top 30 features across 3K positions. Clean phase split: opening features (eval ~0) vs late middlegame features (eval ±1). 4047/4096 features active.
- 35 features labeled for Maia 4096 SAE. Integration PoC: SAE correctly identifies "opening mistake — fundamental principles violated" for a position where player took pawn instead of recapturing knight.
- DeepMind encoder SAE: 4096 (MSE 0.053) and 8192 trained on notebook, checkpoints downloaded.
- Pipeline proven: Maia → hidden state → SAE → feature labels → coaching prompt enhancement. No LLM training needed.
- 500 features interpreted by Claude Haiku ($2, 700s). 233 unique labels. Top concepts: rook on 7th rank, king in center, centralized king in endgame.
- Maia SAE heavily focused on king safety (~100 features detect king position variants). Makes sense for 1800 Elo human move prediction.
- Started wiring SAE into Fargate worker Dockerfile, then reverted — premature.
- A/B TEST DONE: 2 positions, coaching with vs without SAE features. SAE improves framing — connects moves to position type principles. Rook endgame: "rook on the seventh is fundamental" (with SAE) vs generic "prioritize rook activity" (without).
- Prompt changes shipped: `build_moment_prompt` and `build_overview_prompt` now accept `sae_features` field. 3 lines each, backward-compatible.
- Euclidean distance in SAE space: 1.01-1.03x separation (useless — sparse vectors make everything equidistant). Jaccard on active feature sets: endgame↔opening = 2.43x separation. Adjacent phases: ~1.1x. Saved Jaccard insight as cross-project memory.
- Built `backend/shared/position_features.py` — clean module: `get_position_feature_labels(fen)` → ["Rook on seventh rank", ...]. Importable by worker, Lambda, or MCP. Tested, works.
- 100% label coverage on test positions. Rook endgame → "Rook on seventh rank". Opening → "Undeveloped pieces, king not castled".
- Tag↔SAE correlation: `undeveloped_pieces` tag → SAE "Undeveloped pieces, blocked" (10% co-occurrence). Tags and SAE capture different levels — tags say WHAT went wrong, SAE says WHAT KIND of position. Complementary.
- Verified prompts.py diff: Feigned Discovery rewrite + SAE feature support. 89 lines, clean. Not committed. Ready to commit+deploy next session.
- Process feedback written to `.lab/meta/`. MEMORY.md updated.
- Encoder SAE features interpreted: extracted 5K position FENs on GPU, downloaded locally, interpreted via Bedrock ($2). 500 features labeled. Heavy on king safety + "requires immediate defense" — GM-level urgency concepts.
- **1000 total features labeled** across both SAEs for $4 total. Maia (human modeling) + Encoder (deep analysis) = complementary coaching context.
- Unique concept analysis: encoder has 172 labels with no Maia match (urgency concepts: "requires immediate defense", "vulnerability", "coordination imbalance"). Maia has 160 labels with no encoder match (spatial: "rook on fifth rank", "pieces on starting squares", "king not castled"). Encoder is EVALUATIVE, Maia is DESCRIPTIVE.
- Gitignore updated: all .pt checkpoints, maia2_models/, large JSONL files excluded. Only text files would be committed.
- research/sae/README.md written: full documentation of files, results, integration, regeneration.
- KEY: Maia SAE vs Encoder SAE decoder comparison — max cosine similarity 0.113. ZERO features overlap. 4096/4096 unique in each. The two models encode completely different information. Fully complementary — using both gives 8192 unique features.

## 2026-03-31

- (Prior session) Contrastive Phase 0 trained (200K, 3 epochs). Preserved 92% of encoder signal. → learnings.md § F3b
- LoRA v1 launched on 50K mixed data, batch 2, 1.0 step/s
- FSDP crash root cause found: LayerNorm dtype mismatch → learnings.md § F9

## 2026-04-03 (SAE v3 Tag Analysis)

- Pulled 502 cabbagelover5566 games to local JSON (12.4 MB → 26.6 MB with SAE features)
- Computed SAE features (all 32 per moment) on 3,012 moments, 45 seconds on CPU, 0 errors
- Raw feature analysis: top features are too generic ("Bishop pair" 143%, "Knight outpost" 94%). Useless as tags.
- Blunder enrichment analysis: "Undefended piece" 11x enriched at blunders, "Knight fork" 5x, "Unsupported pawn" 4.6x. These are real mistake signals.
- PF-ICF scoring: ICF range 0.27–8.01. Generic features (Bishop pair ICF=0.27) properly downweighted. Rare features (Rook endgame maneuvering ICF=8.01) amplified.
- Combined PF-ICF × enrichment scoring produces actionable per-game narratives:
  - Game 1 (Caro-Kann, 13 mistakes): "Underdeveloped position" in 6/13 mistakes (4.5x enriched). Development → pawn problems → endgame collapse.
  - Game 2 (Sicilian, close loss): "Back Rank Weakness" in 4/6 mistakes. Collapsed after good opening.
  - Game 3 (messy win): "Rapid development" in 7/10 mistakes (14.8x enriched). Pushed pawns instead of developing.
- Key finding: enrichment ratio with min-count threshold (n≥10) eliminates small-sample noise (was getting 99x artifacts)
- Verdict: SAE features ARE the tags when properly filtered. PF-ICF × enrichment is the filter.
- Best-move position comparison attempted but needs init_models() call — deferred.
- Built `player_profile.py` — full cross-game analysis: weaknesses, strengths, opening-specific, phase-specific, time pressure, best-move comparison
- Player profile results (cabbagelover5566):
  - #1 weakness: "Unclear coordination" (47.6x enriched, 122 games) — 16.5% of mistakes vs 0.3% of good moves
  - #2: "Underdeveloped pieces" (20.5x, 88 games)
  - #3: "Piece sacrifice" (23.9x, 95 games) — positions demanding sacrifices overwhelm this player
  - #4: "Rapid development" (14.8x, 54 games) — behind in development
  - Strengths: King opposition (6.8x at good moves), Centralized kings, Rook on seventh — classic endgame competence
  - Opening patterns: Caro-Kann → coordination problems, Vienna → development speed, Sicilian → underdevelopment
  - Phase: Opening = underdevelopment, Middlegame = coordination + sacrifice themes, Endgame = rook play
- Best-move comparison: top 50 worst blunders compared current vs best-move position. "Knight outpost" (40%), "Rook endgame" (34%) appear after best move but not current — these are missed improvements.
- DECISION: PF-ICF × enrichment is the v3 tag system. Enrichment ratios are validated (raw counts confirm 47.6x for top weakness). Ready to wire into frontend.
- Scripts: `lab/chess/scripts/analyze_sae_features.py`, `lab/chess/scripts/game_deep_dive.py`, `lab/chess/scripts/player_profile.py`
- Analysis doc: `lab/chess/docs/2026-04-03 SAE Feature Analysis.md`

## 2026-04-03 (Design Language Implementation Sprint)

**CSS-only design language pass across all pages. No structural/component changes.**

### Global changes
- **Fonts:** Added Lora to Google Fonts import in index.html. Added `--font-display: 'Lora'` to theme.css. Changed `--font-sans` from Inter to DM Sans in both theme.css and theme.ts.
- **Nav:** Active tab changed from bottom-border underline to subtle pill highlight (background tint). Brand text now uses Lora serif.

### Per-page changes
- **Patterns (weaknesses.css):** Tag rows: removed card backgrounds, use border-bottom. Severity bars: 3px→2px. Section accents: 3px→2px. Chips: card-like→simple pills. Cost bars: 4px→3px. Positive card: thinner accent. All JetBrains Mono→tabular-nums.
- **Drill (drill.css):** Mode card active state: removed double-ring box-shadow. Category chips: tighter padding, smaller font. Active chip uses accent-text color.
- **Library (library.css):** Grade uses Lora 18px/600. Removed 3px focus box-shadow rings on search/select inputs.
- **Landing (landing.css):** Headline uses Lora 400 weight instead of DM Sans 700. Removed gradient text clip on highlight (now just accent color). Removed focus ring on import bar.
- **Players (players.css):** Search title uses Lora. Removed focus ring.
- **Review/Board (board.css, nav.css):** No structural changes — board CSS was already clean. Nav changes affect Review's appearance globally.

### Code audit findings
- 11 `any` types across 5 files (existing code, not new). Not changed during design pass.
- No TypeScript build available in this environment — changes are CSS-only and low-risk.

### What's NOT done (needs Sam's environment)
- Build verification (npm start / npm run build)
- Playwright testing (node not available)
- Players page: existing page got CSS pass but doesn't yet show patterns+openings combo view for showcase players
