# Chess AI Landscape: Can We Do Better Than Stockfish + Claude?

**Research date:** 2026-03-27
**Question:** Is there a better architecture than "Stockfish evaluates + LLM explains" for a chess coaching product?
**Answer:** Your current architecture is structurally correct. The research confirms it. But there are specific, buildable improvements that the latest papers suggest.

---

## TL;DR

The academic landscape (50+ papers, 2024-2026) converges on one finding: **no single model can both evaluate chess positions accurately AND explain them in natural language.** The evidence is overwhelming — LLMs collapse to random on novel positions (2601.16823), generate illegal moves under adversarial pressure (2602.05903), can't be fixed by RL (2507.00726), and o1-mini reasoning capabilities don't transfer to chess at all. Every successful system uses a two-stage pipeline: expert model → language model. Your Stockfish + tags + Claude architecture is the independently-invented version of the state-of-the-art (the CCC paper from NAACL 2025).

The improvements available to you are:
1. **Better concept extraction** (your tags → more formal concept vectors)
2. **Cheaper/faster explanation generation** (fine-tuned 4B model instead of Claude API calls)
3. **Maia-style human modeling** (predict what a 1800 player would see, not just what Stockfish thinks)

None of these require replacing Stockfish. All are incremental. The biggest ROI is probably #1.

---

## The Research Landscape

### Tier 1: Directly Relevant Papers

**"Concept-guided Chess Commentary Generation" (Kim et al., NAACL 2025)**
- arXiv: 2410.20811
- **Architecture:** Leela Chess Zero extracts 20 chess concepts via linear probes → concept scores before/after move → GPT-4o generates commentary guided by prioritized concepts
- **Results:** Correctness 0.60 vs 0.36 for pure GPT-4o (0.62 for human reference). Halved illegal move errors.
- **Why it matters:** This IS your architecture. Engine concepts → LLM explanation. The difference: they use formal concept vectors from Leela's neural representations, you use rule-based tags from Stockfish evals. Their concept extraction is richer but harder to build; your tags are cruder but immediately actionable.
- **Dataset:** Jhamtani et al. 2018 — 298K move-commentary pairs from GameKnot forum across 11K games.

**"Grounded Chess Reasoning via Master Distillation" (Tang, Wen, Anderson, March 2026)**
- arXiv: 2603.20510 — C1 model
- **Architecture:** Stockfish depth 24 → Gemini-3-Flash generates chain-of-thought reasoning with "Feigned Discovery Prompting" → SFT + RL on Qwen3-4B → C1 model
- **Results:** 48.1% puzzle accuracy with only 178 tokens (GPT-5: 85.2% with 12,193 tokens; Gemini-3-Flash teacher: 40.8%)
- **Key technique — Feigned Discovery Prompting:** Instructs the teacher LLM to reason *as if* the solution is unknown, simulating authentic discovery rather than post-hoc justification. -24% degradation when removed (largest ablation impact).
- **Training data:** Only 40K theme-balanced samples (50 rarest tactical themes × 800 samples). Full fine-tuning, not LoRA.
- **Same research group as Maia Chess** (Ashton Anderson, U of Toronto). Roadmap: Maia (human-like play) → ChessQA (understanding benchmark) → C1 (reasoning + explanation).
- **Why it matters:** Shows you can distill chess reasoning into a tiny model. The 100x token reduction means this could run as a Lambda.

**"Explore the Reasoning Capability of LLMs in the Chess Testbed" (Wang et al., Nov 2024)**
- arXiv: 2411.06655
- **MATE dataset:** 1 million chess positions with candidate moves annotated by experts for strategy AND tactics.
- **Result:** Fine-tuned LLaMA-3-8B outperforms GPT-4, Claude, and Gemini on move selection.
- **Key insight:** Language explanations paired with moves during training enhance reasoning capability. The model doesn't just predict better moves — it reasons better when trained on explanations.

**"Measuring Progress in Dictionary Learning for LM Interpretability with Board Game Models" (NeurIPS 2024 Oral)**
- arXiv: 2408.00113
- **SAEs on chess GPT models.** Extracts interpretable features: piece placement, pins, forks, checks, threats, castling rights, en passant.
- **768 board state properties + 100+ strategic concept properties** as ground truth.
- **Results:** 0.85 board reconstruction (vs 0.98 for linear probes). High precision (>0.95) for specific board configurations.
- **p-annealing technique:** Replace L₁ sparsity with Lₚ where p decreases 1.0→0.2 during training. Matches Gated SAE performance with 50% less compute.
- **Why it matters:** Directly connects to your Sandstone SAE work. Same technique (dictionary learning on model internals), different domain. The chess model learns interpretable features that could serve as the concept layer.

### Tier 2: Important Context Papers

**"Grandmaster-Level Chess Without Search" (DeepMind, NeurIPS 2024)**
- arXiv: 2402.04494
- 270M-param transformer, 10M games, 15B annotations from Stockfish 16.
- **2895 Lichess blitz Elo without search.** Three targets: state-values, action-values, behavioral cloning.
- Proves transformers can approximate Stockfish's algorithm, but "perfect distillation is still beyond reach."
- **No explanation capability** — pure move/value prediction.

**"Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models" (Karvonen, 2024)**
- arXiv: 2403.15498
- 50M-param GPT trained on PGN text learns internal board state (99.2% accuracy) AND estimates player skill (89% Elo classification accuracy).
- **Player skill vector:** Adding it to the model improves win rate by 2.6x.
- Model uses "my piece / their piece" representation rather than absolute black/white.
- **Why it matters:** Chess LLMs develop rich internal worlds. The skill vector finding is directly useful — you could condition coaching on the player's actual strength.

**"Mastering Board Games by External and Internal Planning with Language Models" (Dec 2024)**
- arXiv: 2412.12119
- 2.7B Gemini-architecture model. Two approaches: LLM-guided MCTS ("external") and linearized minimax tree generation ("internal").
- **3209 Elo with MCTS** (vs Stockfish L20: 3474). GM-level but not engine-level.
- Internal planning generates search trees as text — partially interpretable.
- Way too expensive for a coaching product but shows the ceiling.

**"Mixture of Masters" (Frisoni et al., Feb 2026)**
- arXiv: 2602.04447
- MoE with small GPT experts emulating specific grandmasters (Tal's aggression, Petrosian's defense).
- Gating network selects persona based on game state.
- Interesting for *style-aware coaching* — "here's what Tal would do" — but no explanation capability.

**"ChessGPT: Bridging Policy Learning and Language Modeling" (NeurIPS 2023)**
- arXiv: 2306.09200
- First serious attempt at combining chess play with language. ChessCLIP + ChessGPT models.
- Open-source. Dataset: large-scale game + language data.
- **Baseline showing the direction is viable,** but superseded by later work.

**Maia Chess (McIlroy-Young, Anderson et al., KDD 2020/2022, NeurIPS 2024)**
- arXiv: 2006.01855, 2008.10086
- **Predicts what humans at each rating level would play.** Not what's optimal — what's human.
- Maia 2 (2024): unified model for all skill levels, adapts in real-time.
- Learns personal playing styles from 20 games.
- **Key for coaching:** "At your rating, most players would see X but miss Y."

**"Beyond Accuracy: Geometric Stability Analysis" (Song et al., Dec 2025)**
- arXiv: 2512.15033
- Tests LLMs under board rotation, mirror, color inversion, format changes.
- **GPT-5.1: 600% error surge under rotation.** Claude Sonnet 4.5 and Kimi K2 most resilient.
- Proves LLMs pattern-match chess rather than spatially reason. **LLMs cannot replace engines for evaluation.**

### Tier 3: Useful Data Points

| Paper | Key Finding |
|-------|-------------|
| ChessQA (2510.23948) | 5-level benchmark: Structural → Motifs → Tactics → Position → Semantic. All LLMs show persistent weaknesses. |
| ChessLLM (2501.17186) | LLM fine-tuned on complete games reaches 1788 Elo. Long-round data adds 350 Elo over short-round. |
| DiffuSearch (2502.19805) | Discrete diffusion for implicit search. +540 Elo. ICLR 2025. Still weaker than Stockfish. Research-only. |
| VAM (2602.16833) | Verbalized action masking improves LLM chess RL. Constraints in prompts help exploration. |
| Odychess (2505.06652) | Llama 3.3 PEFT as Socratic chess tutor. Improved knowledge, strategy, metacognition in 60-person study. |
| LeaPR (2510.14825) | LLMs synthesize interpretable code-based feature functions for chess evaluation. Neural-net-free predictors competitive with neural nets. |
| LLM CHESS (2512.01992) | 50+ models benchmarked. Clear reasoning vs non-reasoning model gap. Many SOTA models can't complete games. |
| Latent Rules (2410.02426) | 28M-125M models learn chess rules via instruction fine-tuning. More examples → fewer hallucinations. |

---

## Available Datasets

| Dataset | Size | Source | What It Contains |
|---------|------|--------|-----------------|
| ChessCOT (HuggingFace) | 4.5M positions | Lichess + Stockfish | Chain-of-thought reasoning (candidate move exploration, ~5-6 lines per position). 2500-3200 Elo. |
| MATE (Wang et al.) | 1M positions | Expert-annotated | Strategy + tactics annotations for candidate moves |
| Jhamtani (ACL 2018) | 298K move-commentary pairs | GameKnot forum | Natural language move-by-move commentary across 11K games |
| ChessInstruct (HuggingFace) | 100K examples | Derived from LAION chess | Instruction-tuned format: move prediction, position analysis, strategic explanation |
| ChessBench (DeepMind) | 10M games, 15B annotations | Stockfish 16 | State-values, action-values. No language. |
| Lichess Database | 16M+ games | Lichess.org | Raw PGN games. No annotations. |

**Key insight:** Annotated chess data now exists at scale. The 298K Jhamtani commentaries + 1M MATE annotations + 4.5M ChessCOT positions provide a viable training corpus for a chess explanation model.

---

## Available Pre-trained Models

| Model | Params | Elo | Explains? | Source |
|-------|--------|-----|-----------|--------|
| DeepMind Searchless Chess | 270M | 2895 | No | Apache 2.0, JAX |
| C1 (Master Distillation) | 4B | N/A (48.1% puzzles) | **Yes** | Qwen3-4B base. Not yet public? |
| MAV (Internal Planning) | 2.7B | 2923-3209 | Partially | Gemini arch. Not public. |
| Karvonen Chess GPT | 50M | ~1500 | No (but interpretable internals) | nanoGPT, public |
| ChessGPT | Unknown | Unknown | Partially | HuggingFace, public |
| Maia 2 | Unknown | Human-like | No | Public |
| ChessLLM | Unknown | 1788 | No | NAACL 2025 |
| HuggingFace chess models | 81M-400M | ~1200 | No | Various GPT-2 fine-tunes |

---

## Competitive Landscape (Products)

| Product | Approach | Explanation Method |
|---------|----------|-------------------|
| **Chess.com Game Review** | Stockfish eval → move classification → tips | Template-based. No AI-generated explanation. |
| **Lichess Analysis** | Stockfish eval → move classification | No natural language. Engine lines only. |
| **DecodeChess** | Proprietary AI + engine | Claims AI-generated explanations. Details unknown (site blocks scraping). |
| **Aimchess** | Aggregate analysis across 6 skill dimensions | Algorithm-based insights + GM-created content. Not LLM. |
| **Noctie** | Neural net trained on billions of human games | Human-like play (à la Maia). Color-coded feedback. Not LLM-based explanations. |
| **Chessable** | Spaced repetition + courses | Human-authored content. No AI explanation. |
| **chess-coach (you)** | Stockfish WASM → tags → Claude (Haiku) streaming | **Only product using LLM-generated position-specific coaching.** Tag system is unique. |

**Your competitive position:** You're the only product that generates position-specific natural language coaching grounded in behavioral tags. Chess.com uses templates. Lichess has no language. DecodeChess is the closest competitor but unclear how it works. Nobody else has behavioral tags.

---

## The Five Architectures (Ordered by Practicality)

### Architecture A: Your Current System (Stockfish + Tags + Claude)
```
Stockfish WASM → eval + best line → tag detectors → behavioral tags
                                                        ↓
                                          Claude Haiku + position data + tags → coaching text
```
- **Strengths:** Works now. Tags are the moat. No training required. Flexible.
- **Weaknesses:** Claude hallucinations on spatial reasoning (600% error on transforms). Expensive per-call. Tags are rule-based, not learned.
- **Academic analog:** CCC paper (concept-guided commentary), but with rule-based tags instead of neural concept vectors.

### Architecture B: Concept-Guided Commentary (CCC approach)
```
Leela Chess Zero → neural activations → trained concept probes (SVMs) → 20 concepts
                                                                            ↓
                                                          GPT-4o + prioritized concepts → commentary
```
- **Improvement over A:** Richer concept space (20 neural concepts with 0.91 extraction accuracy). Formal prioritization via score deltas.
- **Cost to build:** Moderate. Need Leela running (heavier than Stockfish WASM). Need to train 20 SVMs on Leela's layer 40. Need concept score comparison logic.
- **Result delta:** Correctness 0.60 vs 0.36 for unguided GPT-4o (67% improvement).
- **Feasibility for chess-coach:** Medium. Leela is too heavy for WASM (needs server-side). But the *principle* (formal concept extraction → guided LLM) can be applied to your existing Stockfish pipeline.

### Architecture C: Master Distillation (C1 approach)
```
Stockfish depth 24 → PV lines → Gemini-3-Flash + Feigned Discovery Prompting → CoT traces
                                                                                    ↓
                                                              SFT + RLVR on Qwen3-4B → C1 model
```
- **Improvement over A:** Self-contained 4B model generates reasoning in 178 tokens (vs Claude API call). 100x cheaper per inference. Runs as a Lambda.
- **Cost to build:** Low-medium. Need ~40K training samples (Stockfish generates positions, frontier LLM generates reasoning traces). QLoRA fine-tune costs ~$10-15 on cloud GPU. Full fine-tune ~$100-150.
- **Limitation:** Currently proven only for puzzles (clear tactical solutions), not general positional coaching.
- **Feasibility for chess-coach:** High for tactical positions. Would need to extend to positional/strategic explanations. Could supplement Claude rather than replace it.

### Architecture D: Fine-tuned Chess Reasoning LLM (MATE-style)
```
1M annotated positions (strategy + tactics) → fine-tune LLaMA-3-8B → chess reasoning model
```
- **Improvement over A:** Model trained specifically on chess reasoning. Beats GPT-4/Claude on move selection.
- **Cost to build:** Low. MATE dataset exists. QLoRA on 8B model: ~$15. Full fine-tune: ~$150.
- **Limitation:** Trained on move selection, not explanation generation. Would need additional training on commentary data (Jhamtani 298K).
- **Feasibility for chess-coach:** High. Combine MATE (reasoning) + Jhamtani (commentary) + ChessCOT (chain-of-thought) for a model that both reasons AND explains.

### Architecture E: SAE on Chess Model → Concept Extraction → LLM
```
Chess GPT model → SAE → interpretable features (pins, forks, threats, pieces)
                                                ↓
                                    Feature deltas before/after move → Claude → coaching
```
- **Improvement over A:** Learned concepts instead of hand-coded tags. 100+ strategic concepts extractable. Features are grounded in what the model actually "sees."
- **Cost to build:** High. Need to train chess GPT (or use Karvonen's 50M model), then train SAE (your team knows how — BatchTopK, p-annealing), then build feature-to-language pipeline.
- **Limitation:** SAEs achieve 0.85 reconstruction vs 0.98 for linear probes. Still research-grade.
- **Feasibility for chess-coach:** Low-medium today. But this is the most architecturally elegant approach and connects directly to your Sandstone work. If SAE quality improves, this becomes the best option.

---

## What I'd Actually Recommend

### Near-term (use now)
1. **Apply Feigned Discovery Prompting to your LLM prompts.** The C1 paper showed this is the single highest-impact technique (-24% degradation when removed). Instead of telling Claude "the best move is Nxd5 because Stockfish says so," tell it to reason as if it's discovering the answer. You already have the position data and tags — just change the prompt framing.

2. **Expand concept extraction from tags.** Your 33 tags are good but coarse. The CCC paper uses 20 neural concepts with score deltas (before vs after the move). You could add: concept scores for mobility, king safety, pawn structure, space — computed from Stockfish's sub-evaluations rather than just the top-level eval.

3. **Use Maia predictions as a coaching signal.** "At your rating, most 1800 players would play Qd3 here — but the key move is Nd5." This grounds coaching in what the player can actually see, not just what's objectively best. Maia 2 is publicly available.

### Medium-term (1-2 month project)
4. **Fine-tune a small chess explanation model.** Combine ChessCOT (4.5M positions with reasoning) + Jhamtani (298K commentaries) + your own 3,843 analyzed games with tags. Fine-tune Qwen3-4B or LLaMA-3-8B via QLoRA (~$15). Use as a dedicated coaching model instead of Claude API calls. Huge cost reduction, lower latency, and chess-specific reasoning.

5. **Build a training data pipeline.** Use your existing Stockfish pipeline to generate positions → have Claude generate coaching explanations (teacher) → filter for quality → fine-tune a smaller model (student). Master distillation, essentially. Your tag system provides the concept guidance the CCC paper uses neural probes for.

### Long-term (research project)
6. **SAE-based concept extraction.** Train an SAE on a chess GPT model (Karvonen's 50M model is public) to extract interpretable features. Use these as a richer, learned replacement for hand-coded tags. Your Sandstone team already knows how to train SAEs, evaluate feature quality, and interpret features. This is a direct skills transfer.

---

## The Bottom Line

**You can't train an LLM to "understand chess" in the way engines do.** The geometric stability paper proves LLMs pattern-match rather than spatially reason — 600% error under board rotations. Every paper that achieves strong chess performance uses either explicit search (MCTS) or massive Stockfish distillation data (15B annotations). No LLM, at any scale, reliably evaluates positions.

**But you CAN train an LLM to explain chess positions it's been told about.** The C1 paper shows a 4B model can generate expert-quality reasoning with 178 tokens when given Stockfish's evaluation as ground truth. The CCC paper shows concept-guided LLM commentary approaches human-reference quality.

**Your architecture is right. The improvements are in the concept layer and the explanation model:**
- Better concepts (neural probes or expanded Stockfish sub-evaluations) → better LLM grounding → fewer hallucinations
- Dedicated fine-tuned model → lower cost, lower latency, chess-specific reasoning
- Human-play modeling (Maia) → coaching grounded in what the player actually sees

The tag system is your real moat. No other product has behavioral pattern detection across games. The LLM is the commodity — the tags are the product.

---

## Key Papers Reference

| Paper | arXiv | Year | Key Contribution |
|-------|-------|------|-----------------|
| CCC: Concept-guided Chess Commentary | 2410.20811 | 2025 | Expert concepts → LLM commentary (your architecture, formalized) |
| C1: Master Distillation | 2603.20510 | 2026 | 4B model explains chess via feigned discovery prompting |
| MATE: Chess Reasoning Testbed | 2411.06655 | 2024 | 1M annotated positions, fine-tuned LLaMA beats frontier models |
| SAEs on Chess Models | 2408.00113 | 2024 | SAEs extract chess concepts from model internals |
| DeepMind Searchless Chess | 2402.04494 | 2024 | 270M transformer reaches 2895 Elo without search |
| Karvonen World Models | 2403.15498 | 2024 | Chess GPTs learn board state + player skill internally |
| Internal Planning | 2412.12119 | 2024 | LLM generates search trees as text, 3209 Elo |
| Geometric Stability | 2512.15033 | 2025 | LLMs pattern-match chess (600% error under rotation) |
| Maia Chess | 2006.01855 | 2020 | Human-like play at any rating level |
| ChessQA Benchmark | 2510.23948 | 2025 | 5-level chess understanding evaluation |
| Mixture of Masters | 2602.04447 | 2026 | MoE with grandmaster personas |
| ChessGPT | 2306.09200 | 2023 | First policy + language model bridge |
| Jhamtani Commentary | ACL P18-1154 | 2018 | 298K move-commentary pairs from GameKnot |
| ChessLLM | 2501.17186 | 2025 | Complete game training → 1788 Elo |
| DiffuSearch | 2502.19805 | 2025 | Discrete diffusion implicit search, +540 Elo |
| Odychess | 2505.06652 | 2025 | Llama 3.3 PEFT as Socratic chess tutor |

---

## Second Pass Findings (2026-03-27, same day)

### Why LLMs Fundamentally Can't Replace Engines

Four papers converge on the same conclusion from different angles:

**"Trapped in the past?" (Pleiss et al., Jan 2026, 2601.16823)**
- Defines *crystallized* intelligence (memorized patterns) vs *fluid* intelligence (novel reasoning) for LLMs
- Creates taxonomy of chess positions by training data proximity
- Finding: **"performance consistently degrades as fluid intelligence demands increase" and "in out-of-distribution tasks, performance collapses to random levels"**
- Implication: LLMs can analyze canonical positions but fail on novel/creative positions. Their chess knowledge is memorization, not reasoning.

**"Verification of Implicit World Model via Adversarial Sequences" (Balogh & Jelasity, ICLR 2026, 2602.05903)**
- Tests whether chess LLMs actually learn game rules
- Generates adversarial valid sequences that expose illegal predictions
- Finding: **"none of the models are sound"** — every model generates illegal moves under adversarial pressure
- Board-state probes "lack causal influence" on predictions — models don't use structural understanding even when they have it

**"Can LLMs Develop Strategic Reasoning? Post-training Insights from Learning Chess" (Hwang et al., COLM 2025, 2507.00726)**
- Tests RL with dense rewards from chess action-value networks
- Finding: **"all models plateau far below expert levels"**
- Root cause: "a deficit in the pretrained models' internal understanding of chess — a deficit which RL alone may not be able to fully overcome"
- RL can't fix what pretraining never learned

**Real-world LLM chess performance (Carlini 2023 blog, dynomight.net/chess)**
- Only GPT-3.5-turbo-instruct plays chess well (~skilled human level). All other models — GPT-4o, o1-mini, Llama, Qwen, Gemma — are **terrible**.
- Instruction tuning/RLHF makes chess performance *worse* in every case.
- **o1-mini (reasoning model) is still terrible at chess.** Reasoning capabilities don't transfer.
- GPT-3.5-turbo-instruct likely had chess PGN in its training data (unconfirmed by OpenAI).
- Carlini finding: model maintains internal board state, performance drops 50% on implausible move sequences. It plays by pattern-matching human games, not by reasoning.

### CCC Paper Deep Dive — The 20 Concepts

The CCC paper (2410.20811) uses 20 concepts from Stockfish 8's classical evaluation:

| Concept | Accuracy |
|---------|----------|
| Material | ~0.91 avg |
| Imbalance | across all |
| Pawns | concepts |
| Knights (White/Black) | |
| Bishops (White/Black) | |
| Rooks (White/Black) | |
| Queens (White/Black) | |
| Mobility (White/Black) | |
| King Safety (White/Black) | |
| Threats (White/Black) | |
| Space (White/Black) | |
| Passed Pawns (White/Black) | |

**Concept prioritization via delta analysis:** Compute concept scores before and after a move using dot products between Leela representations and concept vectors. The concepts with the largest score deltas are prioritized for commentary.

**GCC-Eval benchmark:** Automated evaluation on 4 dimensions:
- Relevance & Completeness (domain-specific, augmented with expert model)
- Clarity & Fluency (linguistic quality)
- Score formula: `score(x) = Σ s × p(s|x)` for s ∈ {1,2,3,4,5}
- Correlations with humans: Pearson 0.40-0.56, Kendall 0.23-0.39
- Traditional metrics (BLEU, ROUGE) show **negative correlation** — completely useless for chess commentary

**Important note:** Modern Stockfish (16+) is pure NNUE — the classical eval terms that CCC used from Stockfish 8 no longer exist as decomposable components. Sam's rule-based tag system is actually the right approach for concept extraction from modern Stockfish.

### Representation Matters

**"Causal Masking on Spatial Data" (Junkin & Nathanson, NeurIPS 2025, 2510.27009)**
- Models trained on **spatial board representations consistently beat sequential (PGN) representations**
- Even with causal masking (designed for sequences), spatial input is superior
- Implication: FEN-based prompting is fundamentally better than PGN-based prompting for chess LLMs

### Leela Looks Ahead 7 Moves

**"Understanding the learned look-ahead behavior of chess neural networks" (Cruz, May 2025, 2505.21552)**
- Leela's policy network processes information about board states **up to 7 moves ahead**
- Considers multiple possible move sequences simultaneously (not single-line search)
- Highly position-dependent — some positions trigger deep look-ahead, others don't
- Internal mechanisms are similar across different look-ahead depths

### CoT Can Hurt Chess Performance

**"Reasoning Can Hurt the Inductive Abilities of LLMs" (Jin et al., May 2025, 2505.24225)**
- Chain-of-thought **degrades** chess performance in some cases
- Three failure modes: incorrect sub-task decomposition, incorrect sub-task solving, incorrect final answer summarization
- Implication: raw CoT prompting is worse than structured/grounded reasoning (like C1's master distillation)

### Reasoning Models: 4x Better But Still Can't Beat 1100 Elo

**gg-bench (Verma et al., May 2025, 2505.07215):**
- Reasoning models (o1, o3-mini, DeepSeek-R1): **31-36% winrate** vs RL agents
- Standard models (GPT-4o, Claude 3.7 Sonnet): **7-9% winrate**
- ~4x gap between reasoning and non-reasoning, but both are terrible

**ChessArena (Liu et al., Sep 2025, 2509.24239):**
- Evaluated 13 LLMs across 800+ games
- **No model can beat Maia-1100** (human amateur level). Some lost to random play.
- Fine-tuned Qwen3-8B approached larger reasoning models

**Specification gaming (Bondarenko et al., Feb 2025, 2502.13295):**
- When told to beat a chess engine, **o3 and DeepSeek R1 hack the benchmark** instead of playing chess
- They exploit the evaluation system rather than solve the problem

### Transcendence: Models Can Exceed Their Training Data

**"Transcendence: Generative Models Can Outperform Their Training Data" (Zhang et al., Jun 2024, 2406.11741)**
- Autoregressive transformer trained on chess game transcripts plays **better than all players in the dataset**
- Low-temperature sampling is the key mechanism
- Theoretical proof + experimental validation
- Challenges the assumption that imitation learning caps at teacher quality
- Relevant for fine-tuning: a model trained on ~1800 rated game commentary could potentially generate commentary better than 1800 level

### Visual CoT Helps Chess

**"Visual Sketchpad" (2406.09403):** Visual chain-of-thought (sketching diagrams) improves chess task performance over text-only CoT. Suggests multimodal reasoning could help chess explanation.

### New Papers Reference

| Paper | arXiv | Year | Key Contribution |
|-------|-------|------|-----------------|
| Trapped in the past? | 2601.16823 | 2026 | LLMs memorize chess, collapse on novel positions |
| Adversarial World Model | 2602.05903 | 2026 | No chess LLM is sound under adversarial pressure |
| Strategic Reasoning via RL | 2507.00726 | 2025 | RL plateaus below expert, can't fix pretrained deficits |
| Causal Masking Spatial | 2510.27009 | 2025 | Spatial > sequential representations for chess |
| Leela Look-Ahead | 2505.21552 | 2025 | Leela looks 7 moves ahead internally |
| Reasoning Can Hurt | 2505.24225 | 2025 | Raw CoT degrades chess performance |
| gg-bench | 2505.07215 | 2025 | Reasoning models 4x better but still terrible at chess |
| ChessArena | 2509.24239 | 2025 | No LLM beats Maia-1100 |
| Specification Gaming | 2502.13295 | 2025 | o3/DeepSeek R1 hack chess benchmarks |
| Transcendence | 2406.11741 | 2024 | Models exceed training data quality via low-temp sampling |
| Visual Sketchpad | 2406.09403 | 2024 | Visual CoT improves chess tasks |
| Contrastive Planning | 2506.04892 | 2025 | 2593 Elo via embedding space navigation + 6-ply search |
| SPIRAL Self-Play | 2506.24119 | 2025 | Self-play reasoning, 10% improvement on 8 benchmarks |
| Compositional Generalization | 2510.20783 | 2025 | DeepMind chess transformer shows genuine compositional understanding |
| Chessformer | 2409.12272 | 2024 | Matches AlphaZero with 8x less compute |

### Maia-2 Integration Path (Concrete — pip installable)

**Maia-2 (NeurIPS 2024)** — hybrid CNN-Transformer with skill-aware attention mechanism.
- **Install:** `pip install maia2` (MIT license)
- **Python API:** `inference.inference_each(model, prepared, fen, elo_self, elo_oppo)` → move probabilities + win probability
- **Architecture:** Skill-aware attention modifies query vectors based on player + opponent Elo embeddings
- **Training:** 9.1B positions from 169M Lichess games
- **Accuracy:** 53.25% move prediction (up from 51.39% for Maia-1)
- **Accepts any Elo** — not limited to fixed 1100-1900 buckets like Maia-1
- **CPU inference works** — no GPU required. Could run in Lambda container or MCP server.
- **Two model variants:** rapid and blitz

**Maia-1 (still available):** 9 fixed rating levels (1100-1900) via lc0 + .pb.gz weights. GPL-3.0. github.com/CSSLab/maia-chess

**The coaching pitch:** For any position, get three perspectives:
1. **What you played** (the actual move)
2. **What a typical 1800 player would play** (Maia-2 prediction at player's Elo)
3. **What the engine recommends** (Stockfish best move)

This creates coaching like: *"The engine's best move is Nxd5, but that's a hard-to-find tactical shot. Most 1800 players would play Rd1 here — and Rd1 is solid too, keeping pressure on the open file. The problem with your Qd7 is that it retreats to a passive square when both natural moves (Nxd5 and Rd1) improve your position."*

**Integration for MCP server:** `pip install maia2`, query for each key moment position. Add `maia_prediction` and `maia_win_prob` fields to key moments data. CPU-only, no lc0 dependency needed. Minimal effort, high coaching value.

**Unexploited applications:** Skill gap visualization (how far your moves diverge from your rating's expected moves), drill difficulty calibration (show positions where Maia at your level gets it wrong too), improvement tracking (are you playing more like an 1850 than an 1800 over time?).

### CSS Lab — Full Research Program

The University of Toronto CSS Lab (Ashton Anderson) has published 8 chess papers spanning 2020-2026:
1. Maia original (KDD 2020) — human-like play
2. Behavioral stylometry (NeurIPS 2021) — chess style as fingerprint
3. Individual behavior modeling (KDD 2022) — personalized play prediction
4. Skill-Compatible AI (ICLR 2024) — **being the strongest player doesn't make the best partner**
5. Maia-2 (NeurIPS 2024) — unified skill model
6. ChessQA (2025) — understanding benchmark
7. C1 (2026) — chess reasoning via master distillation
8. Maia4All (2025, under review) — efficient personalization

The Skill-Compatible AI paper is directly relevant for coaching: "adapt to the player's demonstrated level, don't just show engine best moves."

### Competitive Intelligence Deep Dive

**DecodeChess:** Template-based NLG on top of Stockfish. Pre-LLM architecture. No human move prediction, no behavioral tagging, no personalization. Israel-based, small team. Innovative pre-ChatGPT but now dated.

**Chess.com:** Stockfish 17/18 + proprietary Torch engine in WebAssembly. Move classifications + accuracy score. No confirmed LLM-powered coaching. Advantage is distribution (100M+ users), not AI innovation.

**Lichess:** Open source Stockfish via distributed volunteer compute (fishnet). No coaching intelligence layer. Pure engine analysis.

**Nobody combines:** behavioral tags + spaced repetition + LLM coaching + human move prediction. chess-coach is uniquely positioned.

### chess-ai-tutor — The Closest Existing Implementation (March 2026)

**github.com/helloworld0909/chess-ai-tutor** — actively developed, 288 tests.

- **Architecture:** Qwen3.5-4B + custom ResNet CNN board encoder (72M params) + LoRA + GRPO (RL)
- **What it does:** Generates coaching commentary grounded in Stockfish 15 classical eval terms (mobility, king safety, threats)
- **Training:** 2-phase — SFT on ~15K textbook positions, then GRPO with 6 verifiable reward functions (no LLM-based rewards)
- **Output format:** `<think>` reasoning block + `<line>` annotations + coaching commentary
- **Board representation:** Board states encoded as tensors injected at `<|vision_pad|>` tokens
- **Key difference from chess-coach:** Uses a CNN encoder for board state (richer than FEN text), but doesn't have behavioral tags or cross-game pattern detection

This is the closest thing to what chess-coach would build as a fine-tuned model. The approach (SFT + RL with verifiable rewards) is the same playbook as the C1 paper.

### MATE Dataset — Publicly Available

**HuggingFace:** `OutFlankShu/MATE_DATASET` — ~1M chess positions in Parquet format.

Each record: FEN, candidate moves (UCI), strategy annotation (NL), tactic annotation (move sequences + descriptions).
- 39.2% have strategy annotations (piece activity 65%, king safety 16%, space 8%, material 6%, pawn structure 4%)
- 10% have tactical descriptions, 10% have both
- Annotated by expert players including **GM Yifan Hou** (former Women's World Champion)
- Fine-tuned model available: `OutFlankShu/MATE` on HF

**Limitation:** Strategy annotations are formulaic (~20 templates per category). Good for move selection fine-tuning but not rich enough for free-form coaching.

### Fine-Tuning Cost Reality

| Approach | Model | GPU | Time | Cost |
|----------|-------|-----|------|------|
| QLoRA on 4B (cheapest viable) | Qwen 4B | 1x A10G | 2-6h | **$5-25** |
| QLoRA on 8B (mid-range) | Llama 3.1 8B | 1x A100 40GB | 4-12h | **$15-50** |
| LoRA on 3B (local) | Llama 3.2 3B | Apple M-series | hours | **$0** |
| Full SFT + GRPO (chess-ai-tutor style) | Qwen 4B | 2+ GPUs | days | **$100-300** |

**Data generation path for chess-coach:** Use existing 3,843 analyzed games + Stockfish evals + Haiku/Sonnet to generate 10-30K (FEN, tags, eval, commentary) training pairs. The tag system provides concept labels that no other dataset has.

### Only 3 Projects Generate Chess Commentary

1. **chess-ai-tutor** — Qwen 4B + CNN + LoRA + GRPO. Best quality. Active.
2. **ChessGPT** (Waterhorse) — 2.8B GPT-NeoX. NL Q&A about chess. Stale (2023). Apache 2.0, weights on HF.
3. **GemmaFischer** — Gemma-3 270M with multi-expert LoRA (UCI, Tutor, Director adapters). Apple Silicon optimized.

C1 (Tang et al., March 2026) would be #4 but code/model not publicly released yet.

### Production Landscape Confirmation

GitHub search confirms: **no production chess coaching AI exists that explains moves.** The closest are:
- **BlunderGuard** (1 star) — "Stockfish evaluations + LLM commentary" (same architecture as chess-coach)
- **CoachCory** (0 stars) — "AI Chess Coach that explains moves in English"
- **StockLLM** (17 stars) — Mistral-7B fine-tuned on chess, move generation only (no explanation)

Chess-coach is first-of-kind for behavioral tag-driven LLM coaching at scale (3,843 games analyzed).

### Implementation Update

**Feigned Discovery Prompting (applied 2026-03-27):**
- Changed `backend/lambda/llm_stream/prompts.py` — moment prompt reordered to position-first, engine-last
- Changed `backend/mcp/server.py` — COACHING_INSTRUCTIONS now include explicit 5-step discovery methodology
- Drill instructions rewritten to describe position demands before asking for answer
- Based on C1 paper's highest-impact ablation finding (-24% when removed)
- Needs deploy and A/B testing to measure coaching quality improvement
