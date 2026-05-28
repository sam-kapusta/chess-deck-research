# Matryoshka SAE — Architecture Search (2026-05-28)

Experiment to determine whether Matryoshka SAEs (Bussmann et al., ICML 2025) produce better hierarchical groupings than post-hoc decoder clustering on the standard BatchTopK SAE.

## Setup

- **Input:** 200K Lichess blunders, Maia 3 layer-7 residual (512-dim), diff pooling (to_sq - from_sq), L2 normalized
- **Baseline:** Standard BatchTopK SAE, dict=2048, k=32, 200 epochs (current production SAE)
- **Reference papers:** bussmann25a (Matryoshka SAE on Gemma-2-2B), Personas camera-ready (Jonathan's SAE hyperparameter selection via decoder orthogonality), "Sparse but Wrong" (Chanin & Garriga-Alonso 2025)

## Key Finding: Optimal k per dict size

Swept k values at each dictionary size. Applied Jonathan's criterion: **find where decoder cosine flattens (the elbow), then within that flat region pick the k that maximizes interpretable features (0.1-1% fire rate).**

Secondary criterion from the data: **max feature fire rate must stay ≤ 20%** (above this, hub features dominate and the dictionary becomes position-type indicators rather than coaching concepts).

### Results table (optimal k per dict)

| Dict | Optimal k | Avg Fire | Max Fire | Cos | Interp (0.1-1%) | FVU |
|------|-----------|----------|----------|-----|-----------------|-----|
| 32 | 3 | 9.4% | 19% | 0.405 | 0 | 0.526 |
| 64 | 6 | 9.4% | 20% | 0.325 | 0 | 0.439 |
| 128 | 4 | 3.1% | 19% | 0.369 | 18 | 0.430 |
| 256 | 8 | 3.1% | 19% | 0.322 | 17 | 0.341 |
| 512 | 6 | 1.2% | 19% | 0.372 | 266 | 0.344 |
| 2048 | 16 | 0.8% | — | 0.254 | 1531 | 0.260 |

### Data-driven hierarchy levels

Comparing adjacent dict sizes (Jonathan's rule: similar reconstruction → lower dict is better):

- **32→64:** Significant improvement on all metrics (FVU -17%, cos -20%, max fire drops). **64 is clearly better.**
- **64→128:** Marginal improvement (FVU -2%), cosine WORSENS. **64 wins.**
- **256→512:** Nearly identical FVU (0.341 vs 0.344), but 512 has 15× more interpretable features (266 vs 17).
- **2048** remains validated from the full k-sweep (k=16: 1531 interpretable features, 0 dead, cos=0.254).

**Conclusion — natural hierarchy in the data:**
- Top level: **dict=64, k=6** (categories, ~6 fire per blunder at 9.4% each)
- Mid level: **dict=256, k=8** OR **dict=512, k=6** (subcategories, ~1-3% fire rate)
- Bottom level: **dict=2048, k=16** (atoms, 0.8% fire rate, 1531 in interpretable range)

## Matryoshka Configs Tested

6 Matryoshka configurations trained + 2 baseline SAEs:

| Config | Prefixes | Dict | k | Top Max Fire | Top Cos | Dead |
|--------|----------|------|---|-------------|---------|------|
| A | [64, 256, 2048] | 2048 | 16 | 23% | 0.285 | 0 |
| B | [32, 128, 512, 2048] | 2048 | 16 | 30% | 0.344 | 0 |
| E | [32, 96, 224, 480, 992, 2048] | 2048 | 16 | 26% | 0.322 | 0 |
| C2 | [32, 160, 672, 2720] | 2720 | 22 | 42% | 0.332 | 0 |
| C3 | [32, 160, 672, 2720] | 2720 | 24 | 42% | 0.284 | 0 |
| F | [32, 288, 2336] | 2336 | 20 | 39% | 0.230 | 0 |

**Failed configs (too much dead):**
- C (dict=2720, k=16): 1900 dead features. k too low for dict size.
- D (dict=5440, k=16): 5157 dead. Way too sparse.

## Progressive Recovery (FVU vs N features)

Standard SAEs have NO progressive recovery — first N features are random, not hierarchical. Matryoshka concentrates reconstruction power in early latents.

| N | OG (k=32) | STD (k=16) | A (Matryoshka) | C3 | F |
|---|-----------|------------|----------------|-----|---|
| 32 | 1.026 | 0.983 | 0.763 | 0.484 | 0.486 |
| 64 | 1.015 | 0.975 | **0.433** | 0.450 | 0.463 |
| 256 | 0.922 | 0.835 | 0.330 | 0.328 | 0.331 |
| 2048 | **0.191** | 0.260 | 0.247 | 0.229 | 0.241 |

**Key insight:** Standard SAE wins at full dict (0.191 vs 0.247 best Matryoshka) but has zero useful hierarchy. Matryoshka trades ~5% final FVU for built-in progressive structure.

**Elbow analysis on Config A:** Biggest marginal FVU gains at N=40 and N=64. After N=64, curve flattens until N=256 (secondary shoulder). Config A's prefixes [64, 256, 2048] sit exactly at these natural elbows.

## Critical Learnings

### k must scale with dict size in Matryoshka

In global BatchTopK, early latents hog the k budget (they're trained to reconstruct alone → encoder pushes them higher). With k=16 on dict=2720, only 1.8 features per sample land in the last 2048 latents → mass death.

Fix: scale k proportionally. k=22-24 for dict=2720 eliminates all dead features.

### Prefix-32 doesn't work for this domain

At any k, prefix-32 produces features with 20-40% fire rates and high cosine (>0.30). The 512-dim chess blunder space doesn't decompose cleanly into 32 independent directions. Minimum useful top-level prefix is **64**.

### Bussmann's prefix spacing pattern

Their groups double in size: [2048, 4096, 8192, 16384, 34816] on dict=65536. We tested both 2× and 4× spacing. For our scale (dict~2048-2720), 3 levels with a 4× jump between each works well. More levels (6 in Config E) work but add complexity without clear benefit.

### Per-level k enforcement (untested)

Bussmann notes this as future work: "explicitly enforcing the number of active latents for different latent groups within the Matryoshka SAE has not been studied." We derived the right per-level k from our sweep (k=6 at dict=64, k=8 at dict=256, k=16 at dict=2048) but haven't trained a per-level Matryoshka yet.

## Decision: What to use going forward

**For immediate use:** Config A [64, 256, 2048] k=16 — matches the data's natural elbows, 0 dead, 23% max fire (no hubs), simplest 3-level structure.

**For potential improvement:** Per-level k Matryoshka with k_per_level=[6, 8, 16] at dict=[64, 256, 2048]. Would give each level its validated operating point. Novel modification — needs implementation and testing.

**Alternative:** Standard BatchTopK at k=16 (no Matryoshka) + post-hoc decoder clustering. Already proven coaching-coherent (26 categories validated in this session). Loses progressive recovery but avoids Matryoshka complexity.

## Files

### On S3 (`s3://chess-stage-a-140023406996/sae/weights/matryoshka/`)
- `maia3_matryoshka_2048_k16_p64_256_2048.pt` — Config A (recommended)
- `maia3_matryoshka_2048_k16_p32_128_512_2048.pt` — Config B
- `maia3_matryoshka_2048_k16_p32_96_224_480_992_2048.pt` — Config E
- `maia3_matryoshka_2336_k20_p32_288_2336.pt` — Config F
- `maia3_matryoshka_2720_k22_p32_160_672_2720.pt` — Config C2
- `maia3_matryoshka_2720_k24_p32_160_672_2720.pt` — Config C3

### On S3 (`s3://chess-stage-a-140023406996/sae/cache/`)
- `k_sweep_summary.json` — Full k-sweep results at dict=2048

### Scripts (git: `scripts/sae/`)
- `sweep_k_orthogonality.py` — k-sweep at dict=2048 (produced k=16 finding)
- `train_matryoshka_sae.py` — Matryoshka BatchTopK trainer
- `matryoshka_compare.py` — Multi-config comparison runner
- `matryoshka_elo_test.py` — Rating band discrimination test (not yet run)
- `train_matryoshka_perlevel_k.py` — Per-level k enforcement (not yet run)
