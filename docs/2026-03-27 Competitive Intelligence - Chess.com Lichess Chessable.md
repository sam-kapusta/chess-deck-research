# Engine + Explanation Landscape Research

**Date:** 2026-03-27
**Purpose:** Competitive landscape analysis of how chess coaching products handle the "engine + explanation" problem.

---

## 1. Chess.com Game Review

**URL:** chess.com/analysis

### Engine
- **Stockfish 18** (primary, 108MB full / 7MB lite) and **Stockfish 17.1** (75MB full / 7MB lite)
- Also offers **Torch 2** and **Torch 4** as alternative engines (73MB full / 6MB lite each)
- All engines available in multi-threaded and single-threaded variants
- Analysis runs **client-side in the browser** (WASM), not server-side

### Move Classification (Classification V2)
Uses an **Expected Points Model** -- NOT centipawn loss. Converts engine eval + player rating into win probability (0.00 to 1.00), then classifies based on expected points lost:

| Classification | Expected Points Lost |
|---|---|
| Best | 0.00 |
| Excellent | 0.00-0.02 |
| Good | 0.02-0.05 |
| Inaccuracy | 0.05-0.10 |
| Mistake | 0.10-0.20 |
| Blunder | 0.20-1.00 |

**Special classifications** beyond the threshold model:
- **Brilliant**: A good piece sacrifice; best/near-best move, not in a bad position, wasn't already completely winning
- **Great**: Turns a losing position to equal, or equal to winning. More generous for lower-rated players.
- **Miss**: Failing to capitalize on opponent errors

Key insight: Classification is **rating-adjusted** -- the same move might be classified differently for a 1200 vs a 2200 player.

### Explanation Generation
- **"Move Explanations"** is a **Diamond-tier premium feature** (most expensive tier)
- Chess.com acquired **DecodeChess** (Israeli startup) -- the explanation technology appears to be DecodeChess-derived
- DecodeChess's approach was **rule-based/expert system on top of Stockfish**, NOT LLM-generated
- DecodeChess used pattern recognition + chess knowledge rules to convert engine analysis into natural language
- The explanations are **deterministic and template-structured** -- they identify tactical/positional themes and map them to predefined explanation patterns
- No evidence of LLM integration for explanations as of 2026

### Architecture
- Client-side WASM Stockfish for analysis
- Server-side analysis for Game Review (runs on Chess.com infrastructure, queued)
- Diamond members get deeper server-side analysis
- Explanations generated post-analysis by a separate explanation engine (DecodeChess heritage)

### What's Notable
- The Brilliant move classification is a **marketing masterstroke** -- it drives social sharing and engagement
- Rating-adjusted classification is a genuine coaching insight (same centipawn loss matters more at lower levels)
- Explanations are **gated behind highest paywall** -- this is the scarcest feature

---

## 2. Lichess Analysis

**URL:** lichess.org | **Source:** github.com/lichess-org/lila (open source, AGPL)

### Engine
- **Stockfish** (latest) for standard chess, **Fairy-Stockfish** for variants
- **Dual analysis modes:**
  - **Client-side:** Stockfish WASM runs in user's browser for real-time analysis (unlimited, free)
  - **Server-side:** Powered by **fishnet** -- a distributed volunteer computing network written in Rust
- fishnet clients request analysis batches from Lichess servers; faster machines get user-queue (interactive), slower get system-queue (background)
- No fixed depth limit -- analysis continues until time/resource targets are met

### Move Classification (from source code: `modules/tree/src/main/Advice.scala`)
Uses **win percentage delta**, NOT centipawn loss directly:

```scala
private val winningChanceJudgements = List(
  .3 -> Advice.Judgement.Blunder,
  .2 -> Advice.Judgement.Mistake,
  .1 -> Advice.Judgement.Inaccuracy
)
```

- Converts engine eval to win percentage via `WinPercent.winningChances(cp)`
- Calculates delta between consecutive positions from the player's perspective
- **>=30% win probability drop = Blunder**
- **>=20% = Mistake**
- **>=10% = Inaccuracy**

For mate sequences, different thresholds apply:
- MateCreated (opponent gets forced mate): Blunder unless you were already badly losing (< -700cp = Mistake, < -999cp = Inaccuracy)
- MateLost (you lose a forced mate): Blunder unless resulting position is still very good (> 999cp = Inaccuracy, > 700cp = Mistake)

### Accuracy Calculation (from source: `AccuracyPercent.scala`)
Exponential decay model:
```
accuracy = 103.167 * exp(-0.0435 * winDiff) + (-3.167) + 1
```
Where `winDiff` = deterioration in winning chances (0-100%). Game-level accuracy uses weighted + harmonic mean of per-move accuracies.

### Explanation Generation
- **No natural language explanations.** Lichess shows:
  - Engine eval numbers
  - Best move arrows
  - "Learn from your mistakes" feature: replays positions where you made errors and asks you to find the best move
  - Opening reference (from 2M+ titled player games database)
- Completely transparent -- all source code is open
- Philosophy: provide raw engine data, let the player interpret

### What's Notable
- 100% free, no paywalled analysis features
- The win-percentage approach (not centipawn) is the same fundamental insight as Chess.com's, just with different thresholds
- No explanation = the explicit gap that products like DecodeChess and Chess Deck are trying to fill
- fishnet's distributed architecture is elegant -- volunteers donate compute, Lichess pays $0 for server-side analysis

---

## 3. Chessable / MoveTrainer

**URL:** chessable.com (acquired by Chess.com via Play Magnus Group, 2022, ~$80M deal)

### Spaced Repetition System
- Implements **spaced repetition** for chess opening/tactic memorization
- Algorithm details not publicly documented, but consistent with **SM-2 variant** (standard in the field before FSRS)
- Key mechanic: Review intervals increase as you demonstrate recall. "The gap between reviews will steadily increase" as memory strengthens
- **MoveTrainer 2.0** rebuilt in React (from older tech), enabling mobile apps

### How It Works
1. User selects a course (opening repertoire, tactics set, endgame patterns)
2. Learns lines by playing through them on an interactive board
3. MoveTrainer schedules reviews -- positions come back at increasing intervals
4. If you get it wrong, interval resets; if right, interval grows
5. PRO feature: "Difficult Moves" tracking highlights chronic problem positions

### Coaching Approach
- **No AI-generated explanations.** Coaching comes from:
  - Course author annotations (text/video by titled players)
  - Community comments
  - "Ask a Master" feature for stuck positions
  - Analysis board integration
- Explanations are **human-authored, pre-written** by course creators
- No per-game analysis or personalized coaching

### What's Notable
- Chessable solved the "how do you remember openings" problem, not the "how do you understand your mistakes" problem
- Their SRS is for **rote memorization of move sequences**, not for understanding WHY a move is good
- The content is the moat -- 500+ GM-authored courses
- No game analysis or personalized feedback at all
- Now owned by Chess.com, so integration with Game Review could theoretically happen

### Comparison to Chess Deck's Drill System
- Chessable: memorize authored content at scheduled intervals
- Chess Deck: review YOUR OWN mistakes from YOUR games, tagged with WHY you went wrong, scheduled via FSRS-6
- Chessable is a textbook; Chess Deck is a personalized mistake journal

---

## 4. DecodeChess

**URL:** decodechess.com (appears defunct/closed -- Chess.com profile "permanently closed")

### What It Was
An Israeli startup that built a **natural language explanation engine for chess positions**. The core claim: "explains the chess engine's top moves" in human-readable language.

### Technology (based on available evidence)
- **NOT LLM-based.** Built before the LLM era (pre-GPT-3)
- Architecture was a **rule-based expert system layered on top of Stockfish**:
  1. Stockfish analyzes position, produces eval + best lines
  2. Pattern recognition engine identifies tactical/positional themes (pins, forks, weak squares, pawn structure, piece activity, etc.)
  3. Rule-based natural language generator converts detected patterns into structured English explanations
  4. Explanations organized by category: "Why this move is good", "What are the threats", "What are the plans"
- ChessBase covered it: "DecodeChess explains the chess engine's top moves"
- The explanations were **deterministic** -- same position always produces same explanation
- Quality was reportedly good for tactical explanations, weaker for deep strategic concepts

### Acquisition
- Acquired by Chess.com (exact date unclear, likely 2023-2024)
- Technology appears integrated into Chess.com's Diamond-tier "Move Explanations" feature
- The decodechess.com domain and Chess.com account are now closed

### What's Notable
- Proved the market exists for "explain the engine move"
- Rule-based approach means explanations are reliable but rigid -- can't adapt to player level or ask follow-up questions
- The fact that Chess.com acquired it (and paywalled it) validates that explanation is the premium layer
- Pre-LLM architecture means it's probably already outdated compared to what LLMs can do

---

## 5. ChessCoach (Open Source Project)

**URL:** github.com/chrisbutner/ChessCoach | **License:** GPLv3

### What It Is
An open-source neural network chess engine that combines **MCTS (Monte Carlo Tree Search) with a large neural network** and includes **natural language commentary generation**.

### Architecture
- **Engine:** Large neural network for position evaluation + MCTS for move selection
- **~3450 Elo strength** (beats humans at 2850 but loses to Stockfish 14)
- ~10.5k lines C++ (engine) + ~3.7k lines Python (neural network)
- Custom search method: **SBLE-PUCT** (Selective Backpropagation and Linear Exploration) to reduce tactical blunders
- Training: 44 million self-play games, 700,000 training batches, TensorFlow 2

### Commentary Generation
- Feeds chess knowledge into a **secondary neural network** for English commentary
- Uses **COVET sampling** (tweaked nucleus sampling "focused on correctness-with-variety")
- Author's honest assessment: *"It is not very insightful and often wrong but shows some promise"*
- Commentary is generated by the same model that plays chess -- not a separate language model

### Results
- Search speed: 125,000 nodes/second
- Self-play: 2,360 games/hour
- Running a Lichess bot (@PlayChessCoach)

### What's Notable
- Ambitious attempt to have one model both play chess AND explain its reasoning
- The "often wrong" commentary highlights the fundamental problem: chess engines don't think in concepts, they think in positions
- Proves that generating chess commentary from a chess model alone (without a language model) produces poor results
- The architecture predates LLMs -- an LLM layer on top would likely produce much better commentary

---

## 6. Chessvision.ai

**URL:** chessvision.ai

### What It Does
Computer vision platform that **scans and recognizes chess positions** from:
- Screenshots of websites
- Photos of printed books and diagrams
- Digital board images
- Video frames

### Technology
- **Image recognition / CV** to detect chess positions from any visual source
- Converts detected positions to FEN notation
- Integrates with Chess.com analysis board for evaluation
- Linked YouTube videos matching scanned positions

### Products
- Browser extensions (Chrome, Firefox, Safari)
- Mobile apps (iOS, Android)
- eBook Reader for chess books
- Video analysis app
- Community bots (Discord, Twitter, Reddit)

### What's Notable
- Solves a different problem: input, not explanation
- The CV pipeline (image -> board state -> analysis) is genuinely useful for studying from books
- No original analysis or coaching -- just bridges physical/image chess to digital analysis tools
- Complementary to coaching tools, not competitive

---

## 7. Other AI-Native Chess Products (2023-2026)

### Noctie.ai
**URL:** noctie.ai | **Pricing:** EUR 14/mo or EUR 100/yr

- AI chess training platform with **humanlike AI opponent** trained on billions of human games
- NOT a traditional engine -- trained to play like humans at specific rating levels
- Features: color-coded move evaluation, opening practice, custom position setup
- **Mistake-based learning:** generates puzzle decks from user errors (similar concept to Chess Deck drills)
- Interactive lessons, daily puzzles, weekly scenarios
- Technology: neural network trained on human games (similar to Maia concept)

### Aimchess
**URL:** aimchess.com | **Pricing:** Free tier / $7.99/mo / $57.99/yr

- Analyzes recent games across 6 competency areas: Tactics, Endgame, Advantage Capitalization, Resourcefulness, Time Management, Opening Performance
- **13 specialized lesson types** including personalized puzzle drills from user mistakes
- Claims "personalized puzzles increase rating 31% faster" (UBC study)
- Compares individual performance to others at same rating
- Weekly personalized study plans
- **Coach features:** coaches can access student game data, generate mistake lists

### Chessify
**URL:** chessify.me | **Users:** 60K+ PRO users, 300+ GMs

- Cloud-based engine analysis (up to 1 billion nodes/second for Stockfish)
- Multiple engines: Stockfish 18, Lc0, asmFish, SugaR, Koivisto, Berserk, others
- Plugin for ChessBase, Fritz, HIARCS, SCID
- 9.7M+ game database, 6-piece Syzygy tablebases
- Official FIDE partner (44th Chess Olympiad)
- **Pure engine power** -- no explanation or coaching layer. For serious players who want raw compute.

### ChessMood
**URL:** chessmood.com

- 500+ hours of GM-authored video courses
- Opening, middlegame, endgame structured learning
- Instructors stream their games using the taught openings
- Traditional human coaching at scale -- no AI explanation
- 3,000+ students, 8+ years operating

### Pawnalyze
**URL:** pawnalyze.com

- Monte Carlo simulation for tournament predictions
- **Elocator:** AI position complexity analyzer (rates how hard a position is for humans)
- Analytics/prediction tool, not a coaching tool
- Covered by NYT, NPR

### ChessGPT (the app)
**URL:** chessgpt.app

- Play chess against GPT-4 with real-time commentary
- Sends FEN to GPT-4 API, gets move + strategic feedback via chat
- Vanilla JS + Chess.js frontend, REST API backend
- Falls back to random moves when GPT returns illegal moves (which happens often)
- More of a demo/toy than a coaching product

---

## 8. Academic Research: Engine + Explanation

### Maia Chess (Toronto, 2020)
**Paper:** arxiv.org/abs/2006.01855

- Retrained AlphaZero on human games instead of self-play
- **Predicts what humans will actually play**, not the best move
- Tunable accuracy at different rating levels
- Also predicts when players will blunder
- Key insight: modeling human behavior is fundamentally different from optimizing play
- **Coaching implication:** Could identify moves where a student will likely go wrong, not just where the engine says they went wrong

### Chess Commentary Generation (Meta, 2022)
**Paper:** arxiv.org/abs/2212.08195 | **Authors:** Andrew Lee, David Wu, Emily Dinan, Mike Lewis (Meta)

- **Hybrid approach:** symbolic reasoning engine + controllable language model
- Key finding: commentaries "preferred by human judges over previous baselines"
- Insight: language models alone lack grounding; symbolic engines alone lack fluency. The **combination** works.
- This is essentially the theoretical validation of the architecture Chess Deck uses (Stockfish analysis -> tag detection -> LLM explanation)

### ChessGPT (NeurIPS 2023)
**Paper:** arxiv.org/abs/2306.09200

- Bridges policy learning (from game data) with language modeling
- Built ChessCLIP and ChessGPT models + large-scale dataset
- Full evaluation framework for assessing LLM chess ability
- Demonstrates that integrating game knowledge with language capability produces better chess understanding than either alone

### Amortized Planning with Transformers (DeepMind, 2024)
**Paper:** arxiv.org/abs/2402.04494

- Trained transformers (up to 270M params) on ChessBench (10M games with Stockfish 16 annotations)
- Achieved **Lichess blitz Elo 2895** (GM level) via pure supervised learning, no search
- "A remarkably good approximation of Stockfish's search-based algorithm can be distilled into large-scale transformers"
- Proves transformers can learn chess reasoning, which means LLMs could theoretically reason about chess positions, not just pattern-match

### LLM Chess Playing (adamkarvonen, 2023)
**Repo:** github.com/adamkarvonen/chess_gpt_eval

- GPT-3.5-turbo-instruct: <0.1% illegal move rate, plays at a reasonable level
- GPT-4: consistently loses due to illegal moves, not bad strategy
- Llama models: hallucinate illegal moves constantly
- NanoGPT trained on Stockfish games: ~1200 Elo
- Key finding: instruction-tuned models are worse at chess than base models

---

## 9. How Top Platforms Generate Explanations

### The Spectrum (from fully template-based to fully AI-generated)

| Approach | Products | Pros | Cons |
|---|---|---|---|
| **No explanation** | Lichess, Chessify | Transparent, fast | Useless for learning |
| **Human-authored** | Chessable, ChessMood | High quality, nuanced | Doesn't scale, not personalized |
| **Rule-based expert system** | DecodeChess (Chess.com) | Deterministic, reliable | Rigid, can't adapt, pre-LLM era |
| **Neural commentary** | ChessCoach OSS | Novel approach | "Often wrong", low quality |
| **Hybrid: engine + rules + LLM** | Chess Deck | Adaptive, personalized, can have conversations | Depends on prompt quality, LLM can hallucinate |
| **Pure LLM** | ChessGPT app | Natural conversation | Hallucinates badly, illegal moves, wrong analysis |

### What Actually Works (based on evidence)
1. **Pure LLM fails.** GPT-4 can't reliably play legal chess. Asking it to explain positions without engine grounding produces confident hallucinations.
2. **Pure rules work but feel robotic.** DecodeChess proved the market but the explanations are formulaic.
3. **The Meta research validates hybrid.** Symbolic reasoning (engine) + language model = best results in human evaluation.
4. **Chess Deck's architecture is the right one:** Stockfish analysis -> deterministic tag detection (behavioral + tactical) -> LLM coaching grounded in tags + position data. The tags prevent hallucination. The LLM provides fluency and adaptability.

---

## 10. What Chess Coaches Actually Value

Based on forum discussions, coaching platform designs, and the gap between products and human coaching:

### What Human Coaches Do That Tools Don't
1. **Pattern recognition across games.** "You keep playing Nc3 too early in the Sicilian" -- not about one game, about your habits. (Chess Deck's tag system does this.)
2. **Rating-appropriate explanations.** A 1200 needs different language than a 2000. (Chess.com's rating-adjusted classification is a nod to this; Chess Deck's LLM can adapt.)
3. **Prioritization.** "Forget about endgames for now, your biggest leak is premature trades in the middlegame." (Chess Deck's tag frequency/recurrence scoring does this.)
4. **Emotional coaching.** "You tilted after that blunder on move 15 and the rest of the game was sloppy." (Chess Deck's `game:tilt_after_blunder` tag does this.)
5. **Opening repertoire guidance.** "Stop playing the King's Indian, your style suits the Catalan better." (No product does this well with AI.)
6. **Training plan creation.** Coaches assign targeted exercises. (Aimchess attempts this; Chessable provides content but not personalization.)

### The Gap Products Miss
- **Why over what.** Every product can tell you WHAT the best move is. Almost none explain WHY your move was bad in terms you can internalize. "You dropped a piece" is what. "You're consistently pushing pawns before developing pieces" is why.
- **Cross-game patterns.** Individual game review is table stakes. The value is in seeing patterns across 50+ games.
- **Actionable drills from your mistakes.** Not random puzzles -- YOUR positions where YOU went wrong.
- **Conversational depth.** "But what if they play Bf4 instead?" -- only possible with LLM-based coaching.

### Where Chess Deck Stands

| Capability | Chess.com | Lichess | Chessable | Aimchess | DecodeChess | Chess Deck |
|---|---|---|---|---|---|---|
| Engine analysis | Yes (Stockfish) | Yes (Stockfish) | No | Basic | Stockfish | Stockfish WASM |
| Move classification | Win prob model | Win prob model | N/A | Custom 6-area | N/A | Win prob (Lichess-style) |
| Single-game explanation | Rule-based (paid) | None | Human-authored | Template-based | Rule-based | LLM (Haiku) |
| Cross-game patterns | Basic stats | None | None | 6 competencies | None | 33 behavioral/tactical tags |
| Spaced repetition drills | Puzzle rush (generic) | Puzzle storm (generic) | MoveTrainer (authored) | Personalized puzzles | None | FSRS-6 from own games |
| Conversational coaching | None | None | None | None | None | Streaming LLM chat |
| "Why did I go wrong" | Limited | None | None | Category-level | Position-level | Tag + LLM per moment |

---

## Key Takeaways for Chess Deck

1. **The tag system is the real moat.** No other product labels mistakes with behavioral WHY. DecodeChess detected tactical themes but not behavioral patterns like `quiet_when_winning` or `premature_trade`. Aimchess has 6 broad categories. Chess Deck has 33 specific tags.

2. **Hybrid architecture is validated.** Meta's 2022 paper proves engine + symbolic reasoning + language model > any single approach. Chess Deck already has this: Stockfish -> tags (symbolic) -> LLM (language).

3. **Personalized drills from your own games is rare.** Noctie and Aimchess generate puzzles from your mistakes, but without the behavioral tagging or FSRS scheduling. Chessable's SRS is for memorizing authored content, not your mistakes.

4. **LLM hallucination is the known risk.** ChessGPT (the app) demonstrates the failure mode. Chess Deck's approach of grounding LLM output in engine analysis + tags + position data is the correct mitigation. The enriched position.py work (2026-03-27) directly addresses this.

5. **Cross-game analysis is the frontier.** Chess.com has basic stats. Aimchess has 6 competencies. Chess Deck's tag aggregation across 3,843+ games is genuinely novel.

6. **Chess.com's paywall on explanations validates the business model.** They charge their highest tier (Diamond) for move explanations. This is the feature with the most willingness to pay.

7. **No product has conversational coaching.** Chess Deck's `/moment`, `/ask`, and drill coaching endpoints are unique. The ability to ask "but what about Bf4?" after getting an explanation doesn't exist anywhere else.
