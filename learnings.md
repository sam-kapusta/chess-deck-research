# Chess Lab — Learnings

Index of proven learnings. Short claim + evidence source. Full analysis in docs/.

**Status (2026-04-12):** S37-S40 added (enrichment, aux loss, dead features, polysemanticity). Architecture and Maia sections unchanged since 2026-04-06.

---

## Architecture

- **A1: Three-Layer Architecture** — Position understanding (L1) + Human modeling (L2) + Coaching language (L3). Build order: Maia → QLoRA → Encoder SAE. → docs/2026-04-01 Chess Reasoning Model Design.md
- **A2: Information Asymmetry** — LLM ignores encoder when FEN text present. Every successful projection (LLaVA, MolCA, PointLLM) has encoder as sole info channel. We're the only case where LLM has a text alternative to the same data. Three fixes: (a) remove FEN, (b) contrastive + auxiliary loss, (c) abandon projection, use probes→text. → docs/2026-04-01 Cross-Modal Projection Research.md
- **A3: Integration Is the Moat** — Nobody combines all three layers. Lichess has L1. CCC has L1+L3. Maia has L2. Three teams (Lee/Meta, CCC/POSTECH, us) independently converged on engine→concepts→LLM. → docs/2026-03-27 Papers - MATE CCC Lee Architecture Validation.md
- **A4: Maia + LLM Coaching Is an Open Lane** — Searched Semantic Scholar, GitHub, web. Nobody has combined Maia human move prediction with LLM coaching. → docs/2026-04-01 Maia Chess and Human-Like AI Research.md
- **A5: FEN Is Complete But Not LLM-Accessible** — LLMs collapse to random on novel positions (2601.16823). Encoder decompresses FEN into computed properties. Structured text handles ~90%, encoder adds the 10%. → docs/2026-03-27 Chess AI Landscape (comprehensive).md

## Encoder Experiments (runs 10-11, 2026-04-01)

**⚠ E1-E15 used WRONG causal activations (S13). Absolute numbers unreliable. Archived to docs/. Still-valid operational lessons:**
- **E5: Embedding Norm Must Be Explicitly Scaled** — LayerNorm gives 60x mismatch → NaN. Fix: multiply by target_norm/sqrt(dim).
- **E9: FSDP Crashes Were Dtype Bug** — LayerNorm in bf16 raises RuntimeError. Keep float32, cast after.

## Maia Integration

- **M1: Highest-ROI Immediate Step** — `pip install maia2`, 23M params, CPU 10ms/position, MIT. Zero training needed.
- **M2: Policy Distribution as Coaching Signal** — Difficulty score, rating boundary detection, personalized weakness ID from full move distribution.
- **M3: Tag Learnability (Empirical)** — 100 positions: missed_trapped_piece 72%, missed_capture 57%, premature_trade 20%. → docs/2026-04-01 Maia Chess and Human-Like AI Research.md
- **M4: Rating Gradient** — undeveloped_pieces +5.8% across 1400→2000 (rating-sensitive). quiet_when_winning +1.9% (rating-flat).
- **M5: Best Move Probability Is the Signal** — Player moves ~11% regardless of rating. Best move 36→40%. "These mistakes are below your level."
- **M6: Blunders More Learnable Than Inaccuracies** — 2,764 moments: Blunder P(best)=29% vs Inaccuracy 21.8%.
- **M7: Maia Captures 66% of Eval Variance** — r=0.811 with Stockfish. Encoder adds remaining 34%.
- **M8: Failure Modes** — Rating pool mismatch (~200pts), rare openings (check entropy), aggregate vs individual, staleness. All mitigatable.

## Player Profile (cabbagelover5566, 3,843 games)

- **P1: #1 Improvement Lever** — missed_capture: 42% learnable, 338 instances.
- **P2: Tilt Factor** — -9.3% accuracy post-blunder (1,419 normal vs 218 post-blunder moves).
- **P3: Winning Accuracy Drop** — Equal 91.1%, Winning 84.8%, Losing 90.0%. Focus management, not technique.
- **P4: Pattern Evolution 2021→2026** — Improved: undeveloped_pieces -4%, missed_check -3.4%. Worsened: missed_pawn_break +4.1%, premature_trade +3.3%.
- **P5: Black Is Weakest** — White 83%/60%win, Black 80%/47%win. 2.1x more undeveloped_pieces as Black.
- **P6: Danger × Learnability Matrix** — Fatal+learnable = DRILL NOW (left_piece_hanging). Recoverable+hard = don't worry yet.
- **P7: Compound Weaknesses** — back_rank+missed_skewer 2.9x lift, missed_simplification+quiet_when_winning 2.6x.
- **P8: Opening-Specific Profiles** — London: missed_check. Zukertort: allowed_pin+premature_trade.
- **P9: 45% of "Brilliant" Moves Aren't Impressive** — Maia >30% probability. Forced checkmates inflate the label.
- **P10: Time = Perceived Difficulty** — <2s: 94.6% accuracy. 30-60s: 76.7%. Slow+wrong+Maia says easy = pattern recognition gap.

## Alternative Paths

- **ALT1: Encoder→Probe→Text→Any LLM** — Extract encoder's knowledge via trained linear probes, render as text concepts, feed to Claude/any LLM. No projection, no GPU at inference, no LoRA. Works with existing production pipeline. Investment: days (probes already trained). → docs/2026-04-01 Cross-Modal Projection Research.md
- **ALT2: Priority Stack** — Maia first (2-3 days, `pip install maia2`, add to /tag-moments), landing page second (funnel fix), encoder third (research). If encoder fails: stop encoder, focus Maia + product. If encoder works: still deploy Maia first. → docs/2026-04-01 Next Step Priority Analysis.md

## Literature

- **L1: Feigned Discovery Prompting** — -24% quality without it (C1 paper's largest ablation). Position first, engine last. → docs/2026-03-30 Chess Encoder LLM Fusion Research.md
- **L2: Small Models Beat Frontier** — C1-4B beats Claude Sonnet (48.1% vs 25.6%, 178 tokens vs 3,227). Fine-tuned LLaMA-8B gets 95.2% on MATE vs o1-preview 76.6%. Full FT beats LoRA-64 by 2.1%, LoRA-16 by 8.8% — "knowledge acquisition in a novel domain benefits from updating all parameters." Theme-balanced sampling (K=50 rare themes, M=800 per theme) critical for coverage. → docs/2026-03-29 Small Model Fine-Tuning Evidence.md, docs/2026-03-27 Papers - MATE CCC Lee Architecture Validation.md
- **L3: MLP Projection Works Without CLIP** — 10+ modalities, MiniGPT-4 with 3.5K pairs, Apple MM1 "connector is negligible." → docs/2026-03-31 Cross-Domain Feasibility Research.md
- **L4: GameKnot Data Quality Is Bottleneck** — CARLSy thesis: polarized output, LoRA hurt, medium comments best. Jhamtani 298K is 30% noise. → docs/2026-03-29 Small Model Fine-Tuning Evidence.md
- **L5: Available Training Data** — Lichess popular ~25K (high quality, 640 studies, 550+ likes), Jhamtani 298K (noisy), ChessCOT 4.5M, puzzles 3.5M. Lichess for SFT quality, Jhamtani for alignment volume. → docs/2026-03-30 Chess Encoder LLM Fusion Research.md
- **L6: chess-sandbox Validates Our Architecture** — LC0 → 12 concepts → Claude. 97.9K concept-labeled positions on HuggingFace. Independent implementation of our approach. → docs/2026-03-30 Chess Encoder LLM Fusion Research.md
- **L7: chess-ai-tutor Struggling** — 656M positions, SF15 CNN, 3-stage pipeline, 6 rule-based rewards. Still can't board-read. But different problem — CNN from scratch vs pre-trained encoder. → docs/2026-03-30 Chess Encoder LLM Fusion Research.md
- **L8: Competitive Landscape** — Chess.com uses rating-adjusted win% classification (not centipawn), DecodeChess acquisition for template-based explanations. Lichess uses win% delta thresholds (30%=blunder, 20%=mistake). No competitor has LLM coaching + tags + human modeling. → docs/2026-03-27 Competitive Intelligence - Chess.com Lichess Chessable.md
- **L9: Modern Stockfish Has No Decomposable Eval Terms** — SF16+ is pure NNUE. The classical eval terms CCC used (mobility, king safety, threats from SF8) no longer exist as decomposable components. Our rule-based tag system is the correct approach — there's nothing to decompose. chess-ai-tutor uses SF15 classical eval specifically because SF16+ dropped them. → docs/2026-03-27 Chess AI Landscape (comprehensive).md
- **L10: CSS Lab Built the Foundation** — U of Toronto CSS Lab (Ashton Anderson) published 8 chess papers 2020-2026: Maia (KDD 2020) → behavioral stylometry (NeurIPS 2021) → individual modeling (KDD 2022) → Skill-Compatible AI (ICLR 2024) → Maia-2 (NeurIPS 2024) → ChessQA (2025) → Maia4All (2025) → C1 (2026). We're building on their research program. → docs/2026-04-01 Maia Chess and Human-Like AI Research.md
- **L11: LLMs Fundamentally Cannot Replace Engines** — Collapse to random on novel positions (2601.16823). 600% error under board rotation (Geometric Stability, 2512.15033). RL can't fix pretrained deficits (2507.00726). o1-mini reasoning doesn't transfer to chess. Pattern-matching, not spatial reasoning. → docs/2026-03-27 Chess AI Landscape (comprehensive).md

## SAE System Findings

### Still valid
- **S1:** Maia SAE detects positions, not mistakes. Enrichment measures co-occurrence, not causation.
- **S3:** Three-layer tag architecture: Stockfish PV + hand-coded tags + SAE context + personalization.
- **S4:** Drill system > coaching text. Players improve by practicing, not reading.
- **S5:** DeepMind 270M loadable. OCDBT → npz cache. 360 pos/s on GPU.
- **S11:** DeepMind model is action-value (position + move → win prob), not move predictor. 79 tokens.
- **S12:** Move token reshapes ALL position representations through bidirectional attention.
- **S13:** Our causal mask was wrong. All experiments before v3 used corrupted activations.
- **S16:** Bridge transmits eval: 85% 3-class, 83% 128-bucket.
- **S21:** Eval perspective: always `eval_stm = eval if side == 'w' else -eval`.
- **S23:** Encoder encodes strategic concepts linearly (material 0.94, king_safety 0.80).

### Superseded by per-token puzzle SAE (2026-04-05)
- **S6:** "SAE can't detect tactics" — OVERTURNED. Per-token puzzle SAE detects fork, check, passive play on real game positions. The problem was mean-pooling + wrong training data, not SAE fundamentally.
- **S7/S24/S27/S28:** Bridge/probe/resampler approaches — all superseded by SAE-as-text-labels approach.
- **S8/S9/S10:** Used wrong causal activations (S13). Numbers unreliable.
- **S14/S15:** Bridge experiments — superseded.

### New findings (2026-04-12)
- **S37:** Enrichment >> judge quality >> label quality for detection scoring. Enrichment: +0.120 BA. Judge (Haiku vs Sonnet): +0.001 BA. Label (Haiku vs Sonnet+thinking): +0.013 BA.
- **S38:** Aux loss is universal — fixes dead features at any k. k=32: 57%→9%, k=64: 44%→10%. Always use it.
- **S39:** Dead features aren't bad. They're unused dictionary capacity. 4096 with 26% dead = 3,017 active > 2048 with 10% dead = 1,835. Optimize for active count and label quality, not dead %.
- **S40:** Sonnet polysemantic audit: 30.6% flagged poly, but 486/572 are medium confidence. Poly flag correlates with labeling uncertainty, not genuine concept mixing.
- **S41:** Lower polysemantic rate ≠ better detection quality. k=32 has 3.5% poly vs k=64 30.6%, but k=64 wins on BA (0.632 vs 0.557). Features can be "general" and still be distinguishable.
- **S42:** Larger dictionaries don't improve per-feature quality. 4096 k=64 (3,017 features) scores 0.566 BA vs 2048 k=64 (1,835 features) at 0.632. More features ≠ better features.
- **S43:** Winner config: BTK 2048 k=64 + aux loss. Best on detection (0.632 BA, 325 STRONG), 1,139 mono+high-confidence labels, established pipeline.

### Findings (2026-04-06)
- **S30:** Sonnet outputs `**LABEL:**` with markdown bold. Must strip `**` before parsing. All labels before this fix were garbage.
- **S31:** Puzzle-trained encoder SAE produces 72% confident labels with real chess concepts. Canonical script: `research/scripts/label_sae_features.py`.
- **S32:** Blunder-trained SAEs don't work for labeling. Encoder blunders 27%, Maia blunders 45%. Blunder moves are too diverse to cluster.
- **S33:** Labeling requires FENs + SAN + eval + cp_loss. Without FENs, Sonnet can't analyze positions. Previous attempts never gave Sonnet positions.
- **S34:** Diff SAE (best-blunder activation diff) produces tautological labels (16%). The diff space encodes "better was better" — use puzzle SAE + two forward passes instead.
- **S35:** SAE features with the same category label are subtypes, not duplicates. Jaccard overlap near zero. 26 "fork" features = 25 distinct fork types.
- **S36:** Subtype labeling works: give Sonnet the category + other features' labels + FENs, ask what makes THIS one different. 191/282 features refined into specific subtypes.

### Reference (still valid but niche)
- **S17:** MATE dataset: 592K positions with English annotations. Useful if text generation needed.
- **S18:** BLEU/ROUGE anti-correlated for chess commentary. Use human judgment.
- **S19:** CCC baseline = 0.60 correctness.
- **S26:** MATE has 200 templates, not open text.

## Pipeline Learnings (2026-04-08)

- **PL1: Read production code before building pipelines.** Three SAE extraction runs used the wrong method (mean-pool, per-token-all) before discovering the Lambda uses `hidden[77]` (move token). The production code was 168 lines and answered the question on line 147. Cost: ~3 hours of compute + debugging.
- **PL2: Cache expensive conversions.** The DeepMind 270M checkpoint in OCDBT format requires tensorstore with the OCDBT driver (not in pip wheels). Converting to npz once with `zarr` driver over `ocdbt` kvstore solves it forever. Should have done this the first time.
- **PL3: Same SAE on different activations = garbage.** Maia SAE on 270M activations, puzzle SAE on mean-pooled, puzzle SAE on per-token-all — all produce different features. The SAE must be fed the exact same activation distribution it was trained on.
- **PL4: Per-token fires too many features for corpus aggregation.** 77 tokens × k=32 = up to 2464 activations per position. Every category saturates at 100% of games. Move-token (single 1024-dim vector) gives ~32 activations → meaningful frequency differences.

## Open Questions

1. ~~Is the per-token puzzle SAE the right interface?~~ **YES** — 72% confident, 191 specific subtypes. Production candidate.
2. **Can we skip the encoder entirely?** — Stockfish + tags + Maia already coach well. Encoder adds multi-move tactical depth but at significant complexity cost.
3. ~~Blunder SAE vs puzzle SAE~~ **ANSWERED** — Puzzle SAE wins. Blunder (27%), Maia blunder (45%), diff (16%) all worse.
4. ~~Are "fork" features redundant?~~ **NO** — Jaccard ~0. They're 25 distinct fork subtypes. Same across all categories.
5. **Does wiring SAE labels into Claude prompts improve coaching?** — The A/B test. Next step.
6. **Corpus re-run with move token.** Need FEN + best_move_uci → encoder → hidden[77] → SAE. Requires move_to_action.json mapping + best move for each moment (already in Stockfish data).
