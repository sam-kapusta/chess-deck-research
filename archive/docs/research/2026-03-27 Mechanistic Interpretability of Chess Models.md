# Mechanistic Interpretability of Chess Neural Networks

**Research date:** 2026-03-27
**Purpose:** Survey of what's known about internal representations in chess-playing neural networks and chess-playing language models, with emphasis on techniques that connect to Sandstone's SAE work on behavioral embeddings.

---

## 1. Probing Studies on Chess Models

### AlphaZero: "Acquisition of Chess Knowledge in AlphaZero" (DeepMind, 2022)

**Paper:** McGrath, Kapishnikov, Tomasev, Pearce, Hassabis, Kim, Paquet, Kramnik. Published in PNAS (2022). arXiv: [2111.09259](https://arxiv.org/abs/2111.09259)

The landmark study on what a superhuman chess engine has learned internally. The team probed AlphaZero's neural network for a broad range of human chess concepts and tracked when/where they emerge during training.

**Concepts probed and found:**
- **Material balance** -- the network learns to count material, but interestingly undervalues it relative to human heuristics (AlphaZero is famously willing to sacrifice material for long-term dynamic compensation)
- **King safety** -- emerges as a clear internal representation; AlphaZero's pieces "swarm around the opponent's king with purpose and power"
- **Pawn structure** -- independently discovered and represented internally despite no human input
- **Piece mobility/activity** -- AlphaZero's dominant strategy is "maximizing the activity and mobility of its own pieces while minimizing the activity and mobility of its opponent's pieces" (per GM Matthew Sadler's analysis)
- **Openings** -- independently rediscovered common human opening motifs through self-play

**Methodology:** Combination of behavioral analysis (game-level patterns, commentary from GM Vladimir Kramnik) and representational analysis (probing internal activations). The paper is 69 pages with 44 figures -- exhaustive.

**Key insight:** Concepts emerge at different rates during training. The network doesn't learn everything at once; there's a developmental sequence analogous to how humans learn chess.

### Hammersborg & Strumke: Probing RL Chess Agents (2022)

**Paper:** "Reinforcement Learning in an Adaptable Chess Environment for Detecting Human-understandable Concepts." arXiv: [2211.05500](https://arxiv.org/abs/2211.05500)

Proposed a general method for probing which concepts self-learning chess agents internalize during training. Uses a lightweight chess environment designed for accessibility -- meant to lower the barrier for interpretability research on game-playing agents. Concept detection across training stages.

### Harang et al.: State-Based Evaluation Using Chess (2025)

**Paper:** "Tracking World States with Language Models: State-Based Evaluation Using Chess." arXiv: [2508.19851](https://arxiv.org/abs/2508.19851)

A more skeptical take. Rather than probing activations, they test whether LLMs actually maintain coherent internal game states by examining downstream legal move distributions. Finding: LLMs show "deficiencies in state-tracking" -- the world model is less robust than earlier probing studies suggested. Model-agnostic framework that doesn't require access to model internals.

---

## 2. Emergent World Models: From Othello-GPT to Chess-GPT

### Othello-GPT: The Original "Emergent World Models" Paper

**Paper:** Li, Hopkins, Bau, Viegas, Pfister, Wattenberg. "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task." ICLR 2023 Oral (top 5%). arXiv: [2210.13382](https://arxiv.org/abs/2210.13382)

The foundational work. Trained a GPT variant to predict legal Othello moves from move sequences alone -- no board state, no rules, just next-token prediction on move sequences.

**Key findings:**
- The model develops an **emergent nonlinear internal representation of the board state** despite never seeing a board
- Interventional experiments proved the representation is causal, not just correlational -- modifying it changes model outputs
- Generated **latent saliency maps** that translate predictions into human-understandable explanations
- This was the first rigorous demonstration that sequence models can build internal world models from pure prediction

### Neel Nanda's Linear Probe Discovery

**Paper:** Nanda, Lee, Wattenberg. "Emergent Linear Representations in World Models of Self-Supervised Sequence Models." arXiv: [2309.00941](https://arxiv.org/abs/2309.00941)

Nanda's key contribution: he showed the Othello-GPT world model is actually **linear**, not just nonlinear as Li et al. originally found.

**Key findings:**
- The board state is encoded as a linear representation distinguishing "my colour" vs "opponent's colour"
- This linear structure means you can manipulate the model's behavior with **simple vector arithmetic** -- add/subtract color vectors to flip the model's perception of board state
- The "my/their" framing (rather than "black/white") suggests the model learns a single reusable decision-making program

**Why this matters:** Linear representations are much easier to extract, understand, and manipulate than nonlinear ones. This finding opened the door to practical intervention techniques.

### Hazineh, Zhang, Chiu: Confirming Linear World Models

**Paper:** "Linear Latent World Models in Simple Transformers: A Case Study on Othello-GPT." arXiv: [2310.07582](https://arxiv.org/abs/2310.07582)

Independent confirmation that Othello-GPT encodes a linear representation of opposing pieces that **causally steers** decision-making. Explored how the relationship between internal representation and output depends on layer depth and model complexity.

### Chess-GPT: Extending to Real Games

**Paper:** Adam Karvonen. "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models." Conference on Language Modeling, 2024. arXiv: [2403.15498](https://arxiv.org/abs/2403.15498)

**Blog posts:**
- ["Chess-GPT's Internal World Model"](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html) (Jan 2024)
- ["Manipulating Chess-GPT's World Model"](https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html) (Mar 2024)

The critical extension of Othello-GPT to chess. Karvonen trained a 50M-parameter GPT on 5M real chess games (PGN notation, next-character prediction). The model achieves ~1300 Elo.

**Board state probing results:**
- Linear probes achieve **99.2% accuracy** classifying pieces into 13 categories (blank + 6 white + 6 black piece types)
- Board state reconstruction: **98.6% accuracy** with the 16-layer model
- Legal move generation: **99.8%** accuracy
- Consistent with Nanda's Othello finding: the model represents pieces as **"my piece / their piece"** rather than "white/black" -- a single reusable program for both sides

**Layer-by-layer analysis:**
- 8-layer network: reaches 98% board accuracy by layer 5
- 16-layer network: doesn't reach 98% until layer 11
- This suggests the model calculates many things **in parallel** rather than building board state ASAP -- it's computing board state, move legality, evaluation, and planning simultaneously

**Player skill estimation:**
- Classification probe distinguishing sub-1516 from above-2029 Elo: **89% accuracy** (vs 66% random baseline)
- The model estimates **latent player skill** to better predict what move a player of that level would make
- This is a genuinely novel finding -- the model doesn't just know chess, it models the player

---

## 3. Causal Interventions: Editing Chess-GPT's World Model

From Karvonen's intervention experiments:

### Skill Level Interventions
- Extracted skill vectors by subtracting low-skill from high-skill residual stream representations
- **Positive intervention** (make model play stronger): win rate improved from 69.6% to 72.3% on standard positions
- On **randomly initialized boards**: win rate jumped from 16.7% to **43.2%** (2.6x improvement)
- **Negative intervention** (degrade play): win rate dropped from 69.6% to 11.9%
- Insight: on unfamiliar positions, the model was predicting moves consistent with low-skill players. The skill vector causally controls play quality.

### Board State Interventions
- Identified strategically relevant pieces from the model's predicted moves, then **erased** corresponding board-state vectors from the residual stream
- Unmodified board legal move rate: ~99.8%
- Post-intervention legal move rate: **90.4-92%** (model adapts to the "edited" board)
- Without intervention on modified boards: only 40.5% legal moves
- Heatmap visualizations confirmed piece erasure, though neighboring piece representations became less distinct

### What This Proves
The internal world model is **causal, not just correlational**. You can:
1. Edit which pieces the model "thinks" are on the board, and it changes its moves accordingly
2. Adjust the perceived skill level, and the model plays stronger or weaker
3. The model genuinely constructs an internal simulation, not just surface pattern matching

---

## 4. Evidence of Learned Look-Ahead (Leela Chess Zero)

**Paper:** Jenner, Kapur, Georgiev, Allen, Emmons, Russell. "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network." arXiv: [2406.00877](https://arxiv.org/abs/2406.00877)

Studies Leela Chess Zero (Lc0), a transformer-based chess engine where each board square is a token.

**Three lines of evidence for internal search:**
1. **Causal importance:** Activations on squares of future optimal moves are unusually important causally -- the model "looks ahead" to future board states
2. **Attention heads:** Identified specific attention heads that transfer information bidirectionally through time, connecting future move squares with current positions
3. **Predictive probe:** A simple probe predicts the optimal move **2 turns ahead** with **92% accuracy** in positions where Leela identifies a single best line

**Significance:** This is an existence proof that neural networks learn to implement something like search internally, without being explicitly programmed to do so. The network doesn't just evaluate the current position -- it internally simulates future positions and uses those representations for decision-making.

---

## 5. SAEs (Sparse Autoencoders) on Chess Models

### The Key Paper: SAE Evaluation with Board Games

**Paper:** Karvonen, Wright, Rager, Angell, Brinkmann, Smith, Mayrink Verdun, Bau, Marks. "Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models." ICML 2024 MI Workshop Oral (top 5%), NeurIPS 2024 Main Conference. arXiv: [2408.00113](https://arxiv.org/abs/2408.00113)

**Blog:** ["Evaluating Sparse Autoencoders with Board Games"](https://adamkarvonen.github.io/machine_learning/2024/06/12/sae-board-game-eval.html) (Jun 2024)

This is the direct bridge between Anthropic-style SAE work and chess. The key insight: chess and Othello models have **known ground truth features** (piece positions, legal moves, check, etc.), making them ideal testbeds for evaluating whether SAEs actually find real features.

**What SAEs found in Chess-GPT:**

Individual SAE features corresponded to specific chess concepts:
- **"White Knight on F3"** -- achieved 100% precision at certain activation thresholds
- **"En passant capture available"** -- fired only when this tactical option existed
- **Individual piece location features** -- specific features for specific pieces on specific squares
- **Check and pinned pieces** -- detected with varying success
- Higher-level tactical/positional features (less clearly separated)

**Supervised evaluation metrics (novel contribution):**

1. **Coverage metric:** Measures monosemanticity. For each Board State Property (BSP), finds the best-classifying SAE feature and averages F1 scores.
   - Chess-GPT SAEs: **0.45 coverage** (layer 6)
   - Othello-GPT SAEs: **0.52 coverage**
   - Both substantially outperform random baselines

2. **Board Reconstruction metric:** Can complete board state be recovered from SAE features?
   - Chess-GPT: **0.85 F1**
   - Othello-GPT: **0.95 F1**
   - Linear probes (ceiling): 0.98-0.99

**P-annealing (novel training technique):**
- Standard SAE training uses L1 penalty (convex), which causes feature shrinkage
- P-annealing starts with L1 (p=1) and gradually decreases p toward nonconvex Lp (p<1) during training
- Better approximates true sparsity (L0)
- Significant improvements on both coverage and reconstruction metrics

**Honest limitation acknowledged:** "Lessons learned from board game models may not transfer to language models" -- the ground truth advantage is also a potential weakness for generalization.

### Contrastive SAEs for Chess Planning

**Paper:** Poupart. "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents." RLC 2024 Workshop. arXiv: [2406.04028](https://arxiv.org/abs/2406.04028)

Novel approach: instead of applying SAEs to individual hidden states, applies **contrastive SAEs** to pairs of game trajectories. This targets the sequential planning dimension that standard SAEs miss.

**Key difference from standard SAE work:** Analyzes trajectory pairs to extract features meaningful to planning strategy, not just static position evaluation. Includes automated feature taxonomy and sanity checks against spurious correlations.

### MoE-X: Intrinsically Interpretable Alternative

**Paper:** Yang, Venhoff, Khakzar, et al. "Mixture of Experts Made Intrinsically Interpretable." arXiv: [2503.07639](https://arxiv.org/abs/2503.07639)

Evaluated on chess and natural language tasks. Claims "interpretability surpassing even SAE-based approaches" by exploiting the inherent sparsity of Mixture-of-Experts architectures. MoE-X rewrites MoE layers as equivalent sparse large MLPs with enforced activation sparsity. An alternative to post-hoc SAE analysis.

---

## 6. Can Internal Representations Be Verbalized?

Multiple lines of evidence say yes, partially:

### Latent Saliency Maps (Li et al., 2023)
Othello-GPT's internal representations were translated into **latent saliency maps** showing which board positions the model attends to when making predictions. These are directly human-interpretable.

### Karvonen's Board State Heatmaps
Linear probe outputs from Chess-GPT visualized as heatmaps showing the model's "beliefs" about piece locations. These are spatially organized and intuitively readable -- you can literally see what the model thinks the board looks like.

### SAE Feature Descriptions
SAE features from chess models can be described in natural language: "knight on F3", "en passant available", "piece is pinned." This is analogous to Anthropic's auto-interpretation pipeline where an LLM describes what activates each SAE feature.

### Skill Vector as Verbalization
The extracted skill vector from Chess-GPT is itself a form of verbalized internal state -- it tells you the model's estimate of how strong the player is, expressed as an Elo-like continuous value.

### AlphaZero Concept Probes
DeepMind's probes on AlphaZero effectively verbalize internal concepts: "the network is currently encoding king safety at strength X in layer Y." The probe outputs translate activations into human chess concepts.

### Open Challenge
No one has yet built a full pipeline that takes a chess model's activations at a given position and produces a natural language explanation like "White has a strong attack because the knight on f5 controls key squares around the black king, and Black's pawns are overextended." The pieces exist (SAE features + auto-interpretation + position context), but the end-to-end pipeline hasn't been demonstrated.

---

## 7. Concept Activation Vectors (CAVs) in Chess

### TCAV Applied to AlphaZero
The DeepMind AlphaZero probing paper (McGrath et al., 2022) uses techniques directly inspired by TCAV (Testing with Concept Activation Vectors, Been Kim -- who is a co-author). They define chess concepts (material, king safety, mobility, pawn structure), collect positive/negative examples, train linear classifiers in activation space, and test whether the concept direction is causally meaningful.

This is essentially TCAV applied to a game-playing agent rather than an image classifier. The concept vectors are directions in AlphaZero's hidden representation space that correspond to human chess concepts.

### Karvonen's Skill Vectors
The skill estimation work is a direct application of the CAV idea: find the direction in activation space that separates high-skill from low-skill games, then use that vector for interventions. The "skill concept activation vector" improved win rates by up to 2.6x.

### Nanda's Color Vectors
The "my piece / their piece" linear representations in Othello-GPT are concept vectors for piece ownership. Simple vector arithmetic (add/subtract the ownership direction) flips the model's perception.

---

## 8. Chess Feature Visualization: What Do Neurons "See"?

### Board State Heatmaps
Karvonen produced heatmaps showing, for each square, the model's confidence about what piece occupies it. These are the most direct "feature visualizations" for chess models -- spatial maps of the model's internal board representation.

### SAE Feature Activation Patterns
From the board game SAE work: individual SAE features have characteristic activation patterns across chess positions. A "knight on F3" feature activates specifically when there's a knight on F3. These can be visualized as board-shaped heatmaps showing where/when each feature fires.

### Attention Pattern Visualization (Leela Chess Zero)
Jenner et al. visualized attention heads in Leela Chess Zero that implement look-ahead. Specific attention heads connect squares involved in future optimal moves, creating visible "information highways" between current and future board states.

### Layer-by-Layer Concept Emergence
Both Karvonen (Chess-GPT) and McGrath (AlphaZero) show how concept representations change across layers. Early layers encode low-level features (piece positions), later layers encode higher-level concepts (threats, plans, evaluation). This parallels findings in vision models (edges -> textures -> objects) and language models (tokens -> syntax -> semantics).

---

## Connection to Sandstone's SAE Work

The parallels between chess model SAEs and Sandstone's behavioral SAEs are striking:

| Dimension | Chess Model SAEs | Sandstone Behavioral SAEs |
|-----------|-----------------|--------------------------|
| **Input** | 512-dim residual stream activations from chess-GPT | 1024-dim customer behavioral embeddings |
| **Architecture** | Standard SAE, p-annealing | BatchTopK (leading candidate), dict_size=1024 |
| **Ground truth** | Piece positions, legal moves, check (known) | Purchase categories, HVA membership (partially known) |
| **Evaluation** | Coverage + board reconstruction (supervised F1) | Oracle retrieval recall on 6 HVAs |
| **Feature types** | "Knight on F3", "en passant available" | "Buys premium electronics", "Prime Video heavy user" |
| **Key challenge** | Higher-level concepts (strategy, plans) are harder to isolate | Feature interpretability unmeasured (proposed: LLM scoring) |
| **Dead features** | Not reported | Major issue (1024 dead at dict_size=2048, 4025 dead at 4096) |
| **Novel technique** | P-annealing (L1 -> Lp schedule) | Resampling (artificially props up features -- quality unknown) |

**Direct applicable lessons from chess SAE work:**

1. **Supervised evaluation is possible when you have ground truth.** Sandstone's HVA membership is analogous to chess's "is there a knight on F3?" -- use it.
2. **P-annealing might help with dead features.** The gradual convexity relaxation improved both coverage and reconstruction. Worth trying on Sandstone embeddings.
3. **Board reconstruction as a metric** maps directly to "can you reconstruct the customer's purchase category distribution from SAE features?" -- a concrete Sandstone eval.
4. **Coverage metric** (best F1 per known concept) is directly applicable to Sandstone with HVA labels as the known concepts.
5. **The honest caveat applies both ways:** chess SAE lessons may not transfer to language models, AND language model SAE lessons may not transfer to behavioral models. But behavioral embeddings are arguably closer to chess (structured, finite vocabulary of actions) than to natural language.

---

## Key Open Questions

1. **Can SAEs find strategic/planning features in chess models?** Current SAEs mostly find piece-position features (the easy stuff). Higher-level concepts like "attacking chances," "weak pawn structure," or "piece coordination" haven't been cleanly isolated.

2. **Does the p-annealing technique transfer to other domains?** It showed clear improvements on chess SAEs. Would it help with Sandstone's dead feature problem?

3. **End-to-end verbalization pipeline:** Can you go from chess model activations -> SAE features -> natural language position assessment? The pieces exist but haven't been connected.

4. **Leela Chess Zero + SAEs:** The look-ahead discovery suggests Leela has incredibly rich internal representations. No one has yet applied SAEs to Leela's activations. This could be a goldmine.

5. **Contrastive SAEs for behavioral sequences:** Poupart's trajectory-pair approach could be relevant for Sandstone's sequential purchase data -- comparing two customer journeys to extract discriminating behavioral features.

---

## Citation Index

| # | Paper | Year | Key Contribution |
|---|-------|------|-----------------|
| 1 | McGrath et al., "Acquisition of Chess Knowledge in AlphaZero" | 2022 | PNAS. Probed AlphaZero for material, king safety, mobility, pawn structure. [arXiv:2111.09259](https://arxiv.org/abs/2111.09259) |
| 2 | Li et al., "Emergent World Representations" (Othello-GPT) | 2023 | ICLR Oral. First proof of emergent world models in sequence models. [arXiv:2210.13382](https://arxiv.org/abs/2210.13382) |
| 3 | Nanda, Lee, Wattenberg, "Emergent Linear Representations in World Models" | 2023 | Found linear (not just nonlinear) board representations in Othello-GPT. [arXiv:2309.00941](https://arxiv.org/abs/2309.00941) |
| 4 | Hazineh, Zhang, Chiu, "Linear Latent World Models in Simple Transformers" | 2023 | Independent confirmation of linear world models with causal steering. [arXiv:2310.07582](https://arxiv.org/abs/2310.07582) |
| 5 | Karvonen, "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models" | 2024 | 99.2% board state accuracy, player skill estimation, causal interventions. [arXiv:2403.15498](https://arxiv.org/abs/2403.15498) |
| 6 | Jenner et al., "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" | 2024 | 92% probe accuracy on 2-move look-ahead in Leela Chess Zero. [arXiv:2406.00877](https://arxiv.org/abs/2406.00877) |
| 7 | Karvonen et al., "Measuring Progress in Dictionary Learning with Board Game Models" | 2024 | NeurIPS 2024. SAEs on chess/Othello models, p-annealing, supervised metrics. [arXiv:2408.00113](https://arxiv.org/abs/2408.00113) |
| 8 | Poupart, "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents" | 2024 | Contrastive SAEs on trajectory pairs for planning interpretation. [arXiv:2406.04028](https://arxiv.org/abs/2406.04028) |
| 9 | Hammersborg & Strumke, "RL in an Adaptable Chess Environment for Detecting Human-understandable Concepts" | 2022 | Probing framework for concept detection in chess RL agents. [arXiv:2211.05500](https://arxiv.org/abs/2211.05500) |
| 10 | Yang et al., "MoE-X: Mixture of Experts Made Intrinsically Interpretable" | 2025 | Evaluated on chess; claims interpretability surpassing SAEs. [arXiv:2503.07639](https://arxiv.org/abs/2503.07639) |
| 11 | Harang et al., "Tracking World States with Language Models" | 2025 | Skeptical take: LLMs show state-tracking deficiencies in chess. [arXiv:2508.19851](https://arxiv.org/abs/2508.19851) |

### Additional Resources
- Adam Karvonen's blog: [adamkarvonen.github.io](https://adamkarvonen.github.io/) -- 4 chess interpretability posts
- Code: [github.com/adamkarvonen/chess_llm_interpretability](https://github.com/adamkarvonen/chess_llm_interpretability) -- full probing/intervention toolkit
- Anthropic's SAE methodology (comparison): [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
