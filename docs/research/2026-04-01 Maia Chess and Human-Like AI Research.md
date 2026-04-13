# Maia Chess and Human-Like AI for Coaching

Deep research report. March 2026.

---

## 1. Maia-2: Unified Human Move Prediction (NeurIPS 2024)

**Paper:** "Maia-2: A Unified Model for Human-AI Alignment in Chess"
**Authors:** Zhenwei Tang, Difan Jiao, Reid McIlroy-Young, Jon Kleinberg, Siddhartha Sen, Ashton Anderson
**Venue:** NeurIPS 2024 | **arXiv:** 2409.20553

### Architecture

Hybrid CNN-Transformer with a novel **skill-aware attention mechanism**.

**Position encoding:** 18-channel 8x8 tensor (piece locations, turn, castling rights, en passant). Uses only current position (not 6 historical frames like Maia-1) -- valid under Markov assumption, reduces training complexity.

**CNN backbone (ChessResNet):** 12 residual blocks (K_Conv=12), each with two 3x3 convolutions, BatchNorm, ReLU, dropout 0.5, skip connections. Produces C_patch=8 channel feature maps.

**Channel-wise patching:** Each of the 8 feature channels is flattened to a 64-dim vector and linearly projected to d_att=1024. This is different from spatial patching in standard ViTs -- channels become sequence elements.

**Skill encoding:** Categorical embeddings (d_s=128) for both the active player and opponent. Rating buckets: 0-1000, 1000-1100, ..., 2000+. Separate embedding matrices for self and opponent. Concatenated to 256-dim vector.

**Skill-aware attention (the core innovation):**
```
Q*_k = Q_k + (e_a ⊕ e_o)W*
h_k = softmax(Q*_k K_k^T / √d_k) V_k
```
Skill embeddings modify the **query** vectors only (not keys or values). This means the model learns to "look for different things" depending on skill level, while the position representation stays constant. 2 attention blocks, 16 heads, d_h=64 per head.

**Three output heads:**
1. **Policy head** -- 1968 possible moves, cross-entropy loss
2. **Auxiliary info head** -- piece movements, captures, squares, checks (multi-hot binary CE)
3. **Value head** -- game outcome prediction (-1, 0, 1), 2-layer MLP with dim=128

### Training

- **Data:** 169M rapid-format games (9.1B positions) from Lichess, May 2018 - Nov 2023
- **Balancing:** Up to 20 games per (active, opponent) skill pair per 20K-game chunk
- **Batch size:** 8,192 positions
- **Compute:** ~1 week for 1 epoch on 2x A100 + 16 CPUs
- **LR:** 1e-4, weight decay 1e-5

### Results

| Metric | Maia-1 | Maia-2 | Delta |
|--------|--------|--------|-------|
| Avg move prediction accuracy | 51.39% | **53.25%** | +1.86pp |
| Perplexity (bits) | 4.67 | **4.07** | -0.60 |
| Monotonic positions | 1% | **27%** | +26pp |
| Transitional positions | 17% | **22%** | +5pp |

**Breakdown by skill:**
- Skilled (<=1599): 51.72%
- Advanced (1600-1999): 54.15%
- Master (2000+): 53.87%

**Ablation:** Skill-aware attention accounts for ~73% of the gains. Additional training data accounts for ~27%.

**Linear probing:** Skill-dependent concepts (board evaluation, piece values) show increasing recognition post-attention at higher skill levels. Skill-independent concepts (castling legality) stay flat. The attention mechanism selectively amplifies skill-relevant features.

### Open Source

**Yes. MIT License.** `pip install maia2` -- fully usable as a Python library.

```python
from maia2 import model, dataset, inference

# Load pretrained model (auto-downloads weights)
maia2_model = model.from_pretrained(type="rapid", device="gpu")

# Per-position inference with Elo conditioning
prepared = inference.prepare()
move_probs, win_prob = inference.inference_each(
    maia2_model, prepared, fen, elo_self=1500, elo_oppo=1600
)

# Batch inference
data = dataset.load_example_test_dataset()
data, acc = inference.inference_batch(data, maia2_model, batch_size=1024)
```

**GitHub:** github.com/CSSLab/maia2
**Dependencies:** PyTorch 2.4, chess, einops, numpy, pandas
**Model variants:** "rapid" and "blitz"
**Device:** CPU or GPU

---

## 2. Maia Integration Options

### Direct Python Library (Maia-2)
`pip install maia2` -- the easiest path. Feed FEN + Elo, get move probabilities and win probability. MIT license allows commercial use.

### Maia-1 via Leela Chess Zero
GitHub: github.com/CSSLab/maia-chess. 9 pre-trained models (Elo 1100-1900) as Leela Chess pb.gz weights. Requires `lc0` engine with `go nodes 1` (no search). Python wrapper: `LeelaEngine` class. GPL-3.0 license.

### Lichess Bots
Three bots on Lichess: Maia 1100, 1500, 1900. Can play against them but no programmatic analysis API.

### Web Platform
maiachess.com -- play, analyze, puzzles, "Bot or Not" Turing test. Sign in via Lichess. No REST API exposed.

### What's Missing
No hosted REST API. No streaming analysis service. To use in a product, you'd run the model locally or on your own GPU instance. The `maia2` Python package is the integration point.

---

## 3. CSS Lab (U of Toronto) Complete Chess Research Output

**Lab:** Computational Social Science Lab
**Director:** Ashton Anderson, Dept. of Computer Science, University of Toronto
**Focus:** AI + society, human-AI alignment, mechanistic interpretability

### All Chess Papers (chronological)

**1. Aligning Superhuman AI with Human Behavior: Chess as a Model System** (KDD 2020)
- McIlroy-Young, Sen, Kleinberg, Anderson
- arXiv: 2006.01855
- The original Maia paper. Modified AlphaZero trained on human games to predict moves, not win. Created 9 rating-specific models (1100-1900). Showed Maia predicts human moves far better than Stockfish. Also developed a blunder prediction model.

**2. Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess** (NeurIPS 2021)
- McIlroy-Young, Wang, Sen, Kleinberg, Anderson
- arXiv: 2208.01366
- Transformer-based method identifies individual chess players from their moves alone. 98% accuracy among thousands of candidates using 100 games. Embeddings reveal structural patterns in human chess style. Generalizes from amateur to grandmaster.

**3. Mimetic Models: Ethical Implications of AI that Acts Like You** (AIES 2022)
- McIlroy-Young, Wang, Sen, Kleinberg, Anderson
- Ethical framework for AI that imitates individual behavior. Raises privacy concerns about behavioral identification.

**4. Learning Models of Individual Behavior in Chess** (KDD 2022)
- McIlroy-Young, Wang, Sen, Kleinberg, Anderson
- arXiv: 2008.10086
- Fine-tunes Maia to individual players, significantly improving per-player move prediction accuracy. Validates via stylometry (player identification from move sequences). Key advance: from population-level to individual-level behavior modeling.

**5. Designing Skill-Compatible AI: Methodologies and Frameworks in Chess** (ICLR 2024)
- Hamade, McIlroy-Young, Sen, Kleinberg, Anderson
- arXiv: 2405.05066
- See section 4 below.

**6. Maia-2: A Unified Model for Human-AI Alignment in Chess** (NeurIPS 2024)
- Tang, Jiao, McIlroy-Young, Kleinberg, Sen, Anderson
- arXiv: 2409.20553
- See section 1 above.

**7. C1: Grounded Chess Reasoning in Language Models via Master Distillation** (2026)
- Tang, Wen, Grief-Albert, Elgabra, Yang, Dong, Anderson
- arXiv: 2603.20510
- 4B parameter language model that generates step-by-step chess explanations. Distills expert engine reasoning into natural language chain-of-thought. 48.1% accuracy on chess problems, outperforming all open-source and most proprietary models. Two orders of magnitude fewer tokens than baselines.

**8. Learning to Imitate with Less (Maia4All)** (2025, under review)
- Tang, Jiao, Xue, McIlroy-Young, Kleinberg, Sen, Anderson
- arXiv: 2507.21488
- Achieves individual behavior modeling with only 20 games (down from 5,000). Two-stage optimization: prototype-enriched model bridges population-to-individual, then ability-level initialization refines embeddings. Demonstrated beyond chess on "idiosyncratic LLMs."

### Non-Chess but Relevant
- LLM reliance evaluation (CHI 2025)
- Human creativity with LLMs (CHI 2025)
- ChatBench evaluation framework (ACL 2025)
- SPIN neural compression (ACL Findings 2024)
- Nature 2021 paper on political polarization

### Current Team (chess-relevant)
- **Zhenwei (Joseph) Tang** -- 1st-year PhD, LLMs for teaching/explanation. Lead on Maia-2 and C1.
- **Difan Jiao** -- 1st-year PhD, mechanistic interpretability. Co-author on Maia-2 and Maia4All.

---

## 4. Skill-Compatible AI (ICLR 2024)

**Paper:** "Designing Skill-Compatible AI: Methodologies and Frameworks in Chess"
**Authors:** Karim Hamade, Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, Ashton Anderson
**Venue:** ICLR 2024 | **arXiv:** 2405.05066

### Core Idea
"Achieving superhuman performance alone is not sufficient" when AI must interact with less-skilled partners. Skill-compatibility is "a tangible trait that is qualitatively and measurably distinct from raw performance." An AI that's skill-compatible with a 1200-rated player adapts to their suboptimal decisions and compensates, rather than just being strong.

### Three Methodologies
The paper develops three approaches to explicitly create skill-compatible AI agents in complex decision-making settings. These agents understand and adapt to suboptimal partner decisions.

### Two Chess Frameworks
Two collaborative chess variants serve as test domains for evaluating skill-compatibility.

### Key Result
Skill-compatible agents **outperform AlphaZero-based chess AI** at collaboration despite being individually weaker at standard chess. Being the best player doesn't make you the best partner.

### Coaching Implications
This is directly relevant to coaching. A coach who understands *your* level of play and adapts recommendations accordingly is more effective than one that just shows the engine's best move. The paper formalizes this intuition: the best coaching AI isn't the strongest engine -- it's the one most compatible with the student's current skill level.

**Connection to our product:** Our tag system already does this implicitly. We don't just say "Stockfish says Nf5" -- we explain WHY the player's actual move was wrong in terms they can understand. The skill-compatible framework suggests we could go further: adapt the depth and type of explanations to the player's demonstrated skill patterns.

---

## 5. Human Move Prediction as a Coaching Tool

### The Core Idea
"What would a player at your level play here?" is a powerful teaching mechanism because it creates a mirror. The player sees what their statistical twin would do, then compares it to what they actually did and what's optimal.

### Research Supporting This

**Maia-2** directly enables this. Given a position and a player's Elo, it predicts the probability distribution over all legal moves. You can answer:
- "What would a 1200-rated player do here?" (most likely move at that Elo)
- "What would a 1800-rated player do here?" (same position, different Elo)
- "At what rating does the correct move become the most likely choice?"

**Learning Models of Individual Behavior** (KDD 2022) goes further -- not "what would a 1500 play" but "what would THIS specific player play." Fine-tuned Maia models capture individual tendencies.

**Maia4All** (2025) makes individual modeling practical -- only 20 games needed instead of 5,000.

### Products Using This
- **maiachess.com** -- play against Maia at your level, get human-aware analysis
- No other commercial product appears to use human move prediction for coaching directly

### Unexploited Coaching Applications
1. **Skill gap visualization:** Show the probability distribution at the player's Elo vs 200 points higher. The moves that diverge most are where improvement lives.
2. **"You would have played..."** -- predict the player's likely move before showing what they actually played. Creates self-awareness.
3. **Difficulty calibration:** If Maia predicts a player at your level would miss this tactic 80% of the time, it's a fair drill. If only 5% would miss it, it's trivial for your level.
4. **Improvement tracking:** As a player improves, their actual moves should converge toward higher-Elo Maia predictions. Track this over time.

---

## 6. Personalized Chess Coaching AI -- Academic Research

### Papers Found

**1. Odychess: Adaptive Chess Teaching with Generative AI** (2025)
- Giralt Hernandez, Bueno Perez
- arXiv: 2505.06652
- Fine-tuned Llama 3.3 as a Socratic chess tutor using PEFT. Quasi-experimental study (N=60). Significant improvements in chess knowledge, strategic understanding, and metacognitive skills. "Constructivist and dialectical principles" -- the AI asks guiding questions rather than giving answers.

**2. Game Intelligence: Theory and Computation** (2023)
- Seven
- arXiv: 2302.13937
- Framework for measuring player intelligence in games. Analyzed 1B+ chess moves including top 20 GMs. Uses engine as reference oracle to quantify strategic deviations. Not directly coaching but provides a measurement framework.

### Gap in the Literature
There is remarkably little academic work on *personalized* chess coaching AI that adapts to individual weaknesses. Most research falls into:
- Move prediction (Maia line)
- Puzzle generation (engine-derived)
- General tutoring systems (not chess-specific)

Our product (behavioral tags from actual games, spaced repetition of personal weak spots, tag-driven coaching prompts) appears to be genuinely novel in the academic landscape. The closest academic work is the Odychess Socratic tutor, but it doesn't do weakness-specific adaptation.

---

## 7. DecodeChess -- Technical Intelligence

### What It Claims
DecodeChess markets itself as generating "rich, human-like" natural language explanations of chess positions. The tagline is about understanding why a move is good, not just what the best move is.

### Technical Details (Limited)
DecodeChess is extremely guarded about their technology. Key findings:

- **Engine:** Almost certainly Stockfish-based (or was historically). Their explanations layer on top of engine analysis.
- **Explanation generation:** Pre-LLM approach. Likely rule-based / template-based NLG sitting on top of engine evaluations. The system maps engine output (eval, PV lines, tactical motifs) to templated natural language descriptions. This is NOT an LLM generating text.
- **Architecture:** Server-side analysis (not client-side WASM). Positions sent to their servers for processing.
- **Founded:** Israel-based company. Small team.
- **Pricing:** Freemium model with limited free analyses per day.

### What We Know They DON'T Do
- No human move prediction (they're engine-optimal only)
- No behavioral tagging (they explain the best move, not why you played your move)
- No spaced repetition / drill system
- No cross-game pattern detection
- No individual player modeling

### Assessment
DecodeChess is a **template-based explanation system on top of Stockfish.** It was innovative when it launched (pre-ChatGPT era) because nobody else was generating English explanations of chess moves. Now that LLMs can generate much richer, more contextual explanations, their approach is dated. Their moat is narrow: the chess-specific templates produce consistent, structured output, but they can't adapt to player level or personal patterns.

---

## 8. Chess.com's AI Features

### Engine Stack
Chess.com uses multiple engines:
- **Stockfish 17.1 / 18** -- primary analysis engine. Full (~108MB) and lite (~6MB) variants. WASM-based for client-side analysis. NNUE (neural network utility evaluation).
- **Torch** -- Chess.com's proprietary closed-source engine. Versions 2 and 4. ~73MB full, ~6MB lite. Development team includes authors of Ethereal, Koivisto, Berserk, and Dragon. Regularly places 2nd to Stockfish in Computer Chess Championships.
- **Komodo** -- acquired May 2018 (Elo 3300+). Incorporated MCTS (Monte Carlo tree search) methods.

### Game Review System
- **Move classifications:** Brilliant, Great, Best, Excellent, Good, Book, Inaccuracy, Mistake, Blunder, Forced, Missed Win
- **Accuracy score:** Proprietary calculation based on centipawn loss, likely using a sigmoid-like win% model similar to Lichess's approach
- **Analysis depth:** Varies by membership tier (free gets basic, Diamond gets deep)
- **WDL predictions:** Win/Draw/Loss probability shown per move

### AI / LLM Features
Chess.com has been cautious about LLM integration. What's visible:
- Move explanations exist but quality/depth unclear (may still be template-based)
- No confirmed LLM-powered natural language coaching in Game Review
- Their "AI" branding appears to refer to engine-based features, not generative AI
- Proctor (fair play / anti-cheat) uses ML for cheating detection

### Tech Stack
Java, JavaScript, PHP. Microservices architecture. WebAssembly for client-side engine execution.

### Assessment
Chess.com's analysis is fundamentally **Stockfish in a browser** with a good UI, classification system, and accuracy score. They have not (publicly) shipped LLM-powered coaching explanations. Their competitive advantage is distribution (100M+ users), not AI innovation. Torch is interesting as a proprietary engine but doesn't change the coaching experience.

---

## 9. Lichess Analysis -- Under the Hood

### Architecture
- **Open source:** github.com/lichess-org/lila (main app, Scala)
- **Engine:** Stockfish (latest version), using NNUE evaluation
- **Distributed compute:** fishnet (github.com/lichess-org/fishnet) -- volunteer-donated servers run Stockfish analysis
- **Client assignment:** Faster clients get user-facing analysis jobs. Slower clients get background/system queue jobs.
- **Protocol:** Outgoing HTTP requests only. ~64 MiB RAM per CPU core.

### Analysis Features
- Server-side analysis via fishnet cluster (donated compute)
- Client-side analysis in browser (Stockfish WASM)
- Opening explorer backed by massive game database
- Collaborative studies for shared analysis
- Tablebase integration (including recent 8-piece tablebases)

### What Lichess Does NOT Have
- No LLM-powered explanations
- No natural language coaching
- No behavioral tagging
- No personalized weakness detection
- No spaced repetition drills
- No human move prediction (Maia bots exist separately for play, not analysis)

### Assessment
Lichess is the gold standard for **open, free chess analysis** but it's purely engine-based. No coaching intelligence layer. Their advantage is the open-source community and volunteer compute model. 5M+ games per day analyzed.

---

## Implications for Chess Coach

### What Maia-2 Unlocks
1. **"A player at your level would play..."** -- add Maia-2 predictions to our coaching prompts. Show what the statistical average at their Elo would do, making the gap between their play and improvement targets concrete.
2. **Drill difficulty calibration** -- use Maia-2 to estimate P(correct) at the player's Elo for each drill position. Prioritize positions where P(correct) is 30-70% (zone of proximal development).
3. **Skill progression tracking** -- compare actual moves to Maia predictions at current vs target Elo over time.
4. **Better tag context** -- "87% of players at your level miss this" is more compelling than "Stockfish says this is better."

### Integration Path
`pip install maia2` works today. Feed FEN + Elo, get move probabilities. CPU inference is viable. Could run:
- **In the browser** -- not easily (PyTorch model, no WASM export)
- **As a Lambda** -- yes, with a container image. Cold start might be slow.
- **As a batch job** -- process all drill positions offline, store Maia predictions alongside Stockfish evals
- **In the MCP server** -- most natural fit. Add Maia predictions to game analysis.

### Our Moat vs. the Field
Nobody else is combining:
1. Behavioral tagging (WHY you went wrong, not just WHAT was wrong)
2. Spaced repetition on personal weak spots
3. LLM coaching that receives tags + position data as context
4. Cross-game pattern detection

Maia-2 could add a 5th dimension: human move prediction calibrated to your skill level. The CSS Lab has the research but hasn't built a coaching product. Chess.com has the users but hasn't shipped LLM coaching. DecodeChess has explanation templates but no personalization. We're the only ones connecting tags to drills to coaching.

### C1 Watch List
The CSS Lab's C1 model (4B params, chain-of-thought chess reasoning) is worth monitoring. If open-sourced, it could replace or augment our Haiku-based coaching with a chess-specialized model that generates grounded explanations.
