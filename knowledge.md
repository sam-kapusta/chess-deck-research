# Chess Deck Research — Package Knowledge

Research-package-specific concepts. Shared cross-package concepts (SAE pipeline, DDB schema, handoff contract, core gotchas) live in [`../../knowledge.md`](../../knowledge.md) — authoritative; if you're reading something here that's also there, this file is out of date.

## Current state

**Production SAE:** `realgames_512_k8_v1` — 500 features, deployed. Research's job is keeping the next version in the pipeline.
**Next SAE:** `2048_k64` — 2042 features labeled (Sonnet 4.6 thinking, 79% high, 0 errors). Baselines still needed on `chess-poc`. Not yet deployed.

## Architecture — what works (and why)

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
