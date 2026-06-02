# Blob-Concentration Experiments (overnight 2026-06-02)

**Question:** The k=16 v2 SAE concentrates activation magnitude in ~32 broad "blob" features
(fire on >10% of corpus), drowning out specific mistake-features. Two hypotheses:
1. **Architecture/k**: BatchTopK forces k active slots → fills with broad features. Lower k → fewer blobs.
2. **Sample size**: 168k too small to split broad directions into specifics. More data → fewer blobs.

**Blob metric** (per SAE, on 20k corpus sample): calibrate threshold = mean k-th-largest activation;
n_blob = features firing >10%; pct_top_is_blob = fraction of positions whose top feature is a blob;
spec_top = median activation of top *non-blob* feature; plus n_dead, FVU, useful-feature count.

## Exp 1 — k-sweep (fixed 168k corpus, 200 epochs)

| k | dead | blobs(>10%) | %top=blob | spec_act_p50 | FVU | useful(≥0.1%) | active 1-10% |
|---|------|-------------|-----------|--------------|-----|---------------|--------------|
| 4 | **1800** | 5 | 37% | 0.37 | 0.48 | 196 | 78 |
| 8 | 0 | 12 | 57% | 0.30 | 0.38 | 452 | 138 |
| 12 | 1 | 22 | 69% | 0.27 | 0.32 | 676 | 204 |
| 16 | 1 | 32 | 77% | 0.25 | 0.29 | 983 | 244 |
| 24 | 0 | 68 | 84% | 0.21 | 0.25 | 1262 | 302 |
| 32 | 0 | 89 | 92% | 0.18 | 0.22 | 1501 | 363 |

**Finding: blob concentration is monotonic in k — it IS fundamentally a k problem.** Every step
down in k reduces blob count, blob-domination, and strengthens specific features. No free lunch:
lower k also means worse reconstruction (FVU) and fewer distinct useful features. k=4 collapses
(1800 dead — too sparse for a 2048 dict).

**Sweet spot: k=8.** Blob-minimizing while alive (0 dead): 12 blobs vs 32 at k=16, top-is-blob
57% vs 77%, specific features fire strongest of any viable config (0.30). Cost: FVU 0.38, only
452 useful features. If coaching needs a modest set of clean, high-precision features, k=8 > k=16.
If it needs broad coverage (many distinct mistake types), k=16 trades precision for count.

## Exp 2 — corpus-size sweep (fixed k=16) [RUNNING]

| corpus (k=16) | blobs(>10%) | %top=blob | spec_act_p50 | FVU | useful(≥0.1%) |
|---------------|-------------|-----------|--------------|-----|---------------|
| 42k | 36 | 85% | 0.19 | 0.33 | 1408 |
| 84k | 41 | 84% | 0.21 | 0.30 | 1175 |
| 126k | 36 | 76% | 0.24 | 0.29 | 1053 |
| 168k (full) | 32 | 77% | 0.25 | 0.29 | 983 |

**Finding: corpus size is a weak lever, NOT the blob driver.** Across 4× data (42k→168k),
blob count barely moves (36→41→36→32) and stays in the 30s-40s. Contrast the k-sweep where
blobs ranged 12→89. There IS a mild positive trend — specific-feature activation improves
0.19→0.25 and top-is-blob drops 85%→77% as data grows — so more data helps a little, but
nowhere near enough to fix blobs alone. (Note: useful-feature count *drops* with more data
because the threshold calibrates higher; not a quality regression.)

## Verdict

**The blob problem is fundamentally a k problem, not a data problem.**

- **k is the dominant lever** (blobs 12→89 across k=8→32). Lower k → fewer/weaker blobs,
  stronger specific features, monotonic.
- **Corpus size is a weak secondary lever** (blobs ~32-41 across 4× data). More data nudges
  specific features up slightly but won't solve blobs.

**Recommendation:** if blob concentration / feature specificity is the priority for coaching,
**use k=8, not k=16.** k=8 is the lowest viable k (k=4 collapses to 1800 dead), gives the
fewest blobs (12) and strongest specific features (0.30) of any alive config. The tradeoff is
fewer total useful features (452 vs 983) and worse reconstruction (FVU 0.38 vs 0.29) — acceptable
if coaching needs a clean, precise, modest feature set rather than broad coverage.

The planned 1M-position Lichess corpus is still worth building for *coverage* (more distinct
rare mistakes) but should NOT be expected to fix blob concentration — that requires lower k.

## Artifacts
- `blob_metric.py` — reusable metric (calibrate threshold, n_blob, pct_top_blob, spec_act, FVU)
- `make_subsamples.py` — corpus subsampling for Exp 2
- `blob_sweep_k.jsonl` — Exp 1 raw (k=4,8,12,16,24,32)
- `blob_sweep_corpus.jsonl` — Exp 2 raw (42k,84k,126k,168k)
- Weights on notebook: btk_2048_k{4,8,12,24}_v2_weights.pt, btk_2048_k16_{42,84,126}k.pt

## FOLLOW-UP (2026-06-02): Are blobs actually bad? — NO, mostly coarse-but-real

Scored all 32 k=16 blobs for coherence over top-60 positions (hang-piece concentration +
opus-motif concentration). `blob_coherence.py`.

**Result: blobs split three ways, NOT uniformly bad:**
- **~9 REAL coarse concepts** (coherence >0.35): f101 (high-value piece hangs, 92% hanging_piece
  motif — mislabeled "Queen Hanging to Bishop" but the concept is real), f78 (knight fork),
  f1822 (premature trade, 100% capture), f1563/f1154 (king safety), f655, f1487. These are
  legitimate broad coaching categories a coach WOULD teach.
- **~17 borderline** (0.2-0.35), heavily clustered on king_safety motif — king-safety blunders
  are common and the SAE spreads them across several overlapping broad features. Coherent theme,
  redundantly encoded.
- **~5 genuine mush** (f98, f343, f1091, f1112, f2041): low hang concentration, scattered motifs,
  <20% motif coverage. f98 (39% fire) is the worst — content-free.

**This overturns the overnight "use k=8 to kill blobs" recommendation.** Killing all blobs would
discard the MOST useful coaching categories (hung piece, fork, premature trade). The real problem
was never blob existence — it was that Opus **mislabeled** coherent broad features with overly
specific names (a 33%-corpus feature can't be "Queen Hanging to Bishop"; it's "high-value piece
left hanging").

**Corrected action items:**
1. Keep k=16 (its larger useful-feature set is fine — blobs are mostly real concepts)
2. Relabel coherent blobs by fire rate: feed Opus the fire-rate, instruct ">20% corpus = label
   the COARSE pattern broadly" (e.g. "high-value piece hangs" not "Queen Hanging to Bishop")
3. Discard only the ~5 true-mush features (f98, f343, f1091, f1112, f2041)
4. For display/coaching: the >10% blobs become a "coarse category" layer; specific (<10%) features
   are the fine lessons. Two-tier, not blob-removal.

## CORRECTION (2026-06-02): top-60 was unrepresentative — measure across activation range

Sam flagged: "33% of moves being king/queen hanging doesn't make sense." Correct — and it
exposed a measurement error. The coherence scores above were computed on each feature's
**top-60** positions. But a 33%-fire feature activates on ~55,000 positions; the top-60 are
the extreme tip and wildly unrepresentative.

Measured Q/R-hang rate across activation bands (`blob_activation_decay.py`, `blob_body.py`):

**f101 — REAL concept, but activation-graded:**
| band | activation | Q/R hang |
|------|-----------|----------|
| top 2% | >0.72 | 75% |
| 2-10% | >0.55 | 53% |
| 10-30% | >0.35 | 38% |
| 30-60% | >0.19 | 22% |
| 60-100% | >0.08 | 10% (~base rate) |

f101 genuinely means "high-value piece hangs" — but ONLY at high activation. At the firing
threshold it's at base rate (noise). The magnitude IS the confidence.

**f1487 — actually MUSH (flat across all bands):** Q/R-hang ~11% at every activation level,
including top 2%. Activation carries zero information. Highest-firing feature (44%) and it's
noise. The chip "Quiet move ignoring hanging piece" is simply wrong. Discard.

**Revised understanding (supersedes the "blobs are coarse-but-real" follow-up above):**
Blobs are NOT one kind of thing. Two distinct types, distinguishable only by the activation-decay
SHAPE (not by top-60, not by fire rate):
1. **Graded-real** (f101): coherent concept at high activation, decays to base rate. Usable with
   a per-feature HIGH activation cutoff (e.g. f101 above ~0.55 = "piece hangs").
2. **Flat-noise** (f1487): no concept at any activation level. Discard.

**The right fix is per-feature activation thresholding + flat-noise detection, NOT global
fire-rate filtering and NOT lower k.** Lower k (overnight rec) would remove graded-real features
like f101 along with the noise — wrong. The principled move: keep features whose high-activation
band is coherent; gate each at the activation level where its concept-rate exceeds base rate;
discard features that are flat (f1487-type).

Method lesson: NEVER characterize a high-fire SAE feature by its top-N examples. Measure the
property-rate across the full activation distribution. Top-N of a noise feature can look as clean
as top-N of a real one.

## MAJOR CORRECTION (2026-06-02): coherence probe was one-sided — measured blunder, not best move

The diff the SAE trains on is `L7[after best move] − L7[after blunder]`. A feature can be
coherent because its positions share a **best-move** pattern, a **blunder-move** pattern, or both.
All my coherence probes (motif-join, piece-hang, SEE) only measured the BLUNDER move. Sam: "you're
not capturing both, it's a bad crude proxy."

Rebuilt the probe to measure BOTH moves' signatures (`dual_coherence.py`), real SEE for hangs,
maia_best (100% dedup-cache coverage) for the best move. Of 1322 candidate features (fire 0.2-15%),
at ≥60% concentration:

| coherent on | count |
|-------------|-------|
| blunder-axis only (what I measured all night) | 184 |
| **best-move-axis only (probe was BLIND to these)** | **458** |
| both axes | 71 |
| neither | 609 |

**713 of 1322 (54%) are coherent — and the dominant axis is the BEST move, not the blunder.**
2.5× more features organize around "what Maia would have played" than "what the player did."

Examples of best-move-axis features (previously called "noise"):
- f7: best move always a queen capture (Qx 100%) → "missed a queen capture"
- f2/f15/f35: best move a rook move or check (100%) → "missed a checking/rook resource"
- f0/f16: best move quiet pawn (cap- chk- 100%) → "right move was quiet, you played something flashy"

These are the **offensive-miss / "you missed a good move"** mistakes — first-class coaching
categories. The blunder move shows "nothing hangs" because the mistake is failure-to-act, not
losing material. A blunder-move probe structurally cannot see them.

**FINAL VERDICT (supersedes all earlier pessimism):** The SAE is NOT mostly noise. ~54% of
candidate features cleanly encode a move pattern. The investigation kept concluding "noise"
because every probe measured only the blunder move. Labeling must describe BOTH what was played
AND what was missed (the best move). The z-score-only k=16 model has these 713 coherent features.

Method lesson (the big one): when a feature is defined on a DIFFERENCE of two things, the
coherence probe must measure both sides of the difference. Measuring one side and concluding
"incoherent" is a guaranteed false negative — it cost most of this session.

## DECISIVE COMPARISON (2026-06-02): normalization is the dominant lever

Dual-axis coherence (real SEE, both moves) over ALL 2048 features, identical probe:

| model | coherent/2048 | % | bl-only | best-only | both |
|-------|---------------|---|---------|-----------|------|
| **z-score only, k=16** | **990** | **48%** | 255 | 634 | 101 |
| z-score+L2, k=16 | 350 | 17% | 41 | 278 | 31 |
| z-score+L2, k=32 | 336 | 16% | 43 | 263 | 30 |

**Two confirmed findings:**
1. **Drop L2 (Sam's call, proven): z-score-only nearly TRIPLES coherent features (350→990).**
   L2 projects every diff to the unit sphere, erasing magnitude. Magnitude is part of what makes
   a mistake-pattern coherent. Adding L2 on top of z-score destroys ~65% of coherent features.
2. **k=16 > k=32**, but normalization dominates — both L2 models are ~16-17% regardless of k.
   Normalization is a far bigger lever than k.

**RECOMMENDED MODEL: z-score-only, k=16 (`btk_2048_k16_nol2.pt`), 990 coherent features (48%),
labeled on both blunder + best-move axes.** Coherence is best-move-dominated (634 best-only) —
"you missed a good move" mistakes.

Note: earlier numbers used a fire-rate prefilter (0.2-15%) that gave different denominators per
model (confound). This table is all-2048, identical denominator — the trustworthy comparison.

## Reproduction (chosen model)

```bash
# on chess-poc, from ~/SageMaker
python3 scripts/sae/train_maia3_sae.py \
  --activations chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
  --dict-size 2048 --k 16 --k-aux 128 --aux-alpha 0.03125 \
  --lr 3e-4 --batch-size 4096 --n-epochs 200 --warmup-steps 500 \
  --n-batches-to-dead 126 --no-val-split --no-l2 \
  --output chess-stage-a/output/maia3_sae/btk_2048_k16_zscore.pt
# eval: z-score input ONLY (no L2), calibrate threshold = mean k-th-largest, then dual_2x2.py / see_coherence.py
```
Cache `maia3_l7only_v2_dedup.pt` = 79M-Maia L7 mean64 best−blunder diff, deduped 168,132×1024
(notebook only; rebuild via build_l2l7_v2.py + build_l7_only.py if lost).

## k-axis complete at z-score (2026-06-02)

| model (z-score only) | coherent/2048 | % |
|----------------------|---------------|---|
| **k=8**  | **1042** | **50%** |
| k=16 | 990 | 48% |
| k=32 | 786 | 38% |

Monotonic: lower k → more coherent (down to k=8; k=4 collapses with 1800 dead). k=8 marginally best
(1042 vs 990, ~5% more). But the margin is small — normalization (z-score vs L2: 990 vs 350, 3×) is
the dominant lever; k=8-vs-k=16 is secondary. k=8 is sparser (8 active/position vs 16) — possibly
cleaner per-position but may miss secondary patterns. **Open decision: k=8 vs k=16 for the chosen model.**
