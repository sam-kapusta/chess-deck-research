# Maia4All — Data-Efficient Individual Behavior Modeling in Chess

Paper read 2026-04-26. PDF: [papers/maia4all-preprint2025.pdf](papers/maia4all-preprint2025.pdf).

---

## Paper

**Title:** Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess
**Authors:** Zhenwei Tang, Difan Jiao, Eric Xue, Reid McIlroy-Young, Jon Kleinberg, Siddhartha Sen, Ashton Anderson
**Affiliations:** UToronto, Harvard, Cornell, Microsoft Research
**URL:** https://www.cs.toronto.edu/~ashton/pubs/maia4all-preprint2025.pdf
**Base model:** Maia-2 (NeurIPS 2024)

---

## TL;DR

Models an individual chess player's style with **20 games** (800 positions) instead of the 5,000 games previously required — a **250× data-efficiency gain** over Maia-Individual. Two-stage fine-tuning: *enrich* Maia-2 on a curated prototype player set, then *democratize* to unseen players via a prototype-matching meta-network that produces a warm-start embedding.

At 20 games unseen: **53.22%** move-prediction accuracy vs Maia 51.32% / Maia-2 51.46%. At 2,500 games: **53.81%**. Gains scale smoothly with available data and hold across Skilled/Advanced/Master bins.

---

## Why It Matters

Previous state of the art (Maia-Individual, 2022) required **5,000 games per player** to beat base Maia. Fewer than 1% of Lichess players have that many. Direct fine-tuning with 1,000 games made the model *worse* than baseline. Maia4All is the first individual model that works for the long tail.

For us: this is the closest paper to a production-ready "learn my style" path. Our current SAE feature approach gives interpretable player fingerprints from a handful of games; Maia4All gives a *predictive* move-level model from a handful of games. Different slices of the same problem.

---

## Method

### Base: Maia-2 (summary)

- 18-channel 8×8 position input (piece, turn, castling, en-passant)
- ResNet backbone (12 blocks) → channel-wise patching → 2 transformer blocks with **skill-aware attention**
- Skill embedding matrix `E_P ∈ ℝ^{11×128}` — 11 rating bins (< 1100, 1100–1199, …, > 2000)
- Skill embedding modifies the *query* in attention: `Q* = Q + (e_a ⊕ e_o)W*`
- Three heads: policy (1968 moves), value, auxiliary
- Markov assumption — current position only, no history frames

### Stage 1 — Enrichment

**Idea:** expand the 11 population embeddings into thousands of per-player embeddings, then fine-tune the whole model.

- Select `K = 100` prototype players per skill level × 11 levels = **1,100 prototype players**
- Prototype selection criteria: rich game history + balanced across skill bins (avoid population bias)
- Initialize `E_I[i] ← E_P[r(i)]` — every prototype starts from its strength's population embedding
- Fine-tune φ and `E_I` jointly to minimize standard next-move CE loss on prototype games
- Output: **Prototype-Enriched Maia-2** with universal params φ' that are now responsive to individual-level variation

**Key finding:** enrichment is load-bearing. Maia4All-Strength (skip enrichment, just init from population embedding) only beats Maia-2 by ~0.5pp. Prototype-enriched version beats by 2.5pp. The discriminative prototype structure teaches φ' to care about individual variation.

### Stage 2 — Democratization

For unseen player u ∈ 𝒰, freeze φ' and optimize only a 128-d embedding `e_u`.

**Warm-start options:**
1. **Strength-Init** — `e_u ← E_P[r(u)]`, just use the player's rating bin embedding. Simple baseline.
2. **Prototype-Init** — use a **Prototype Matching Network (PMN)** to pick the closest prototypes and blend their embeddings.

**PMN architecture:**
- Input: history of player's moves (before + after action positions from Maia-2 ResNet features)
- Stacked transformer layers with mean pooling over history → single history embedding
- Classification head with |I| = 1,100 outputs — trained with CE to predict which prototype generated the history
- At inference: take top-k softmax-weighted prototype embeddings, average → `e_u` init

Then fine-tune only `e_u` on unseen player's limited history.

**Why this helps:** next-move prediction is a *generative* task and hard to do data-efficiently. Prototype matching is a *discriminative* task over a fixed class set — much easier, and the learned embedding space is already organized by style, so the init is already close to the right answer.

---

## Results

### Table 1 — Unseen players, varying history size

| Model | 20K pos (500g) | 8K pos (200g) | 2K pos (50g) | 800 pos (20g) |
|-------|---------------:|--------------:|-------------:|--------------:|
| Maia | 0.5132 | 0.5132 | 0.5132 | 0.5132 |
| Maia-2 | 0.5146 | 0.5146 | 0.5146 | 0.5146 |
| Maia-2-Strength | 0.5195 | 0.5196 | 0.5193 | 0.5189 |
| Maia4All-Strength | 0.5308 | 0.5298 | 0.5279 | 0.5249 |
| **Maia4All-Prototype** | **0.5365** | **0.5348** | **0.5334** | **0.5322** |

### Table 2 — Unseen players with 100K positions (≈2,500 games)

| Model | Skilled | Advanced | Master | Overall |
|-------|--------:|---------:|-------:|--------:|
| Maia | 0.4996 | 0.5099 | 0.5285 | 0.5132 |
| Maia-2 | 0.5008 | 0.5158 | 0.5364 | 0.5146 |
| Maia-2-Strength | 0.5071 | 0.5212 | 0.5400 | 0.5199 |
| Maia4All-Strength | 0.5226 | 0.5376 | 0.5478 | 0.5336 |
| **Maia4All-Prototype** | **0.5261** | **0.5408** | **0.5554** | **0.5381** |

### Table 3 — Prior-informed init only (no democratization fine-tune)

| Init | 20K / 8K / 2K / 800 positions |
|------|------------------------------:|
| Strength-Init | 0.5008 (flat across all) |
| Prototype-Init | 0.5180 / 0.5175 / 0.5173 / 0.5167 |

Prototype-Init beats Strength-Init before any fine-tuning happens — the PMN warm start is already stylistically on-target.

### Table 4 — Freeze vs optimize φ' during democratization

| φ' | 20K pos acc | 800 pos acc | 20K perplexity | 800 perplexity |
|----|------------:|------------:|---------------:|---------------:|
| Optimized | 0.5395 | 0.5297 | 4.1949 | 4.3753 |
| **Frozen** | 0.5365 | **0.5322** | **4.2295** | **4.2988** |

**Freezing φ' wins in low-resource.** Acts as regularization — fewer parameters to overfit. Also preferable for scalability: serving per-player φ' is a deployment non-starter.

---

## Other Interesting Findings

### Behavioral Stylometry (free)

Because φ' is frozen and `E_I` lives in a shared embedding space, prototype embeddings are directly comparable. PMN gets **89% player-ID accuracy from 20 games, 1-shot against 1,100 candidates**. Rivals/beats the dedicated stylometry work from McIlroy-Young 2021.

### Prototype Embedding Structure

t-SNE of prototype-informed unseen embeddings forms a linear structure with Skilled at one end, Master at the other, Intermediate filling the middle. Variance around each cluster center represents individual style variation beyond skill.

### Hyperparameter Sensitivity

- **Prototype distribution** must be uniform across skill levels. Low/mid/high-biased selection hurts accuracy and raises perplexity.
- **Number of prototypes N per level:** sweet spot at ~100. Too few = poor coverage of style space; too many = harder prototype-matching classification problem (diminishing returns past N=100).

### LLM Generalization (case study)

Same framework applied to LLaMA 3.1 8B + LoRA for author style mimicry on Project Gutenberg.

- 100 prototype authors, ModernBERT-based PMN, 94.7% prototype-classification accuracy
- 2-step (enrich + democratize) beats 1-step direct fine-tuning across 1K/2K/3K training tokens
- Demonstrates the prototype-enrichment pattern travels beyond chess

---

## Reproducibility Details

- **Hyperparameters:** LR 1e-4, weight decay 1e-5, batch 8192 positions, 12 conv blocks, 2 attention blocks, 256 intermediate channels, 128-d player embedding, 16 heads × 64 per head, d_att=1024, **N=100 players/bin**
- **Filtering:** min 10 ply, max 300, drop positions with < 30s remaining clock
- **Position perspective:** board-flipped so all analysis is from White's POV
- **Dataset:** Lichess Blitz (data-rich, comparable ratings), test positions = last 2048 from 2023
- Code/weights not yet released (as of preprint)

---

## Connections to Chess Deck Work

### How this intersects with our SAE feature system

Both approaches produce a low-dim player representation from limited data. Different properties:

| | Maia4All embedding | Our SAE profile |
|--|-------------------:|----------------:|
| Dimension | 128 (dense) | 500 (sparse, ~8 active) |
| Interpretability | opaque | labeled features (domain/subcategory) |
| What it predicts | next move (generative) | blunder patterns (behavioral) |
| Data needed | 20 games | 1+ game |
| Transferable to LLM | not directly | yes (via labels as text) |

**The research question:** does Maia4All's 128-d embedding contain information our 500-d SAE profile misses, and vice versa? If they agree on player style, we can drop one. If they disagree, the disagreement is the signal.

### Potential uses

1. **"Play like me" bot.** Maia4All gives us an actual move-prediction policy, not just a style fingerprint. Plug it into `/play` as an opponent trained on the user's own games.
2. **Better drill selection.** PMN similarity → find players with similar style, use their mistakes as drill seeds.
3. **Warm-start for coaching LLM.** Prototype-matching to a known showcase player = "your play resembles Magnus's at 1500" narrative hook.
4. **Stylometry for showcase auth.** 89% 1-shot from 20 games is suspiciously accurate — could verify a claimed chess.com account actually belongs to the user.

### Open questions / what the paper doesn't answer

- How does it behave on non-Blitz time controls? (Rapid/Classical not tested.)
- Does the 128-d embedding encode interpretable axes, or does it need PCA/probing to extract style?
- Does fine-tuning on 20 *post-coaching* games vs 20 random games shift the embedding? (Useful for measuring coaching impact.)
- No code released yet — reproducing would require retraining Maia-2 + 1,100 prototype fine-tunes. Probably weeks on our GPU budget.

---

## Bottom Line

Strong paper. The enrichment → democratization recipe is elegant and the data efficiency gain is real. The prototype-matching PMN is the clever bit — it turns an untractable generative few-shot problem into a tractable discriminative retrieval problem, then uses retrieval to warm-start the generative model.

For us, it's most useful as **architectural inspiration for how to personalize over sparse data** rather than something to integrate immediately. Code isn't released, and even if it were, reproducing on our infra is a month-scale project. But the *pattern* (prototype set → enrichment → freeze + warm-start) is directly applicable to any personalization work we do on top of our SAE system.
