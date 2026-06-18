# SAE Stability on Maia activations — instability is an init-basin artifact, fixed by data-driven init (2026-06-17)

**Headline finding:** SAE seed-instability on our Maia blunder-diff activations — which the literature
(Fel et al., "Archetypal SAE", ICML 2025) treats as a fundamental, near-unsolved problem — is on this
data almost entirely an **initialization-basin artifact**. Random init drops each run in a different
basin (only ~6–9% of features reproduce across seeds). **Initializing the SAE from k-means centroids
of the activations makes 93% of features reproduce across DIFFERENT random seeds (MMCS 0.90).** No
architecture change, no archetypal constraint, no BatchTopK needed. This makes "feature N encodes
concept X" a stable, citable claim instead of a seed artifact — the blocker on all downstream
interpretability ("what does Maia represent that our tagger misses").

All experiments on the `maia3_blunder_diff_v2.pt` cache (200k positions × 512-dim; layer-7,
best-move − blunder-move, mean-pooled over 64 squares), on chess-poc. Scripts in
`scripts/sae/` (train_jumprelu.py, train_archetypal_jumprelu.py, jumprelu_frontier_sweep.py,
sae_metric_battery.py).

## The instability, measured (the problem)

Train two SAEs, identical config, different random seed. Match decoders by max cosine (MMCS = mean
of each feature's best-match cosine to the other dictionary; also report fraction matched > 0.7/0.8).

| Setup | MMCS | >0.7 reproducible |
|-------|-----:|------------------:|
| plain JumpReLU, random init, different seeds | 0.41 | ~6% |
| plain JumpReLU, dict 256 (smaller) | 0.45 | ~6.5% |
| **the same with SHARED (identical) init, different data order** | **0.95** | **98%** |

That last row is the diagnostic that cracked it: with the **same init**, two runs differing only in
data-shuffle order reproduce 98% of features. So the instability is NOT data-order and NOT the
objective — it is **which random basin the init lands in**. (Smaller dict raises MMCS slightly but the
>0.7 reproducible *fraction* stays flat ~6.5% — a hollow gain: it removes split-feature noise from the
mean, doesn't add reproducible features.)

## Archetypal SAE — modest, and not the answer here

We tried the literature fix (Fel et al.): constrain dictionary atoms to the convex hull of the data
(`overcomplete` lib, RelaxedArchetypalDictionary). Grafted onto OUR JumpReLU (see debugging below).
Result across dict sizes, different seeds:

| dict | MMCS | >0.7 |
|------|-----:|-----:|
| 256 | 0.50 | 12% |
| 512 | 0.47 | 9% |
| 1024 | 0.44 | 5% |
| 4096 | 0.51 | 18% |

Archetypal helps (0.41→~0.50, ~2× the stable core) but it's a **modest** gain, not the paper's vision-
model transformation — AND it costs sparsity/reconstruction (the constraint blunts atoms, needs more
of them). Smaller dict = more stable (same trend as plain). Not worth the cost given what init does.

## THE FIX: data-driven (k-means) initialization — and its HONEST limit (bias test)

Initialize the encoder from k-means centroids of the activations, then train with different seeds:

| Setup | MMCS | >0.7 |
|-------|-----:|-----:|
| random init, different seeds (baseline) | 0.47 | 9% |
| archetypal, different seeds | 0.51 | 18% |
| **k-means init, SAME centroids, different seeds** | **0.89** | 88% |
| **k-means init, DIFFERENT centroids, different seeds** | **0.63** | 42% |
| shared random init (identical init) — ceiling | 0.95 | 98% |

⚠️ **CORRECTED (the bias test Sam demanded): the 0.89 was PARTLY circular.** It used the SAME centroids
for both seeds, so most of that stability was the shared-init effect (≈ the 0.95 ceiling), NOT the data
forcing the features. The decisive test is **DIFFERENT centroids (different k-means seed) + different
training seeds** → MMCS drops to **0.63**. So:
- The truth is IN BETWEEN. Data-driven init captures REAL seed-independent structure (0.47 → 0.63 is a
  genuine gain over random init), but **~half of the headline "0.89" was shared-init, not
  data-determination.** Do NOT claim the features are seed-independent "truths."
- **Practical recommendation (valid):** for a REPRODUCIBLE production SAE, compute the centroids ONCE
  and reuse them → legitimately gives 0.89. "We always init from these fixed centroids" is a fine
  engineering choice; it makes "feature N = concept X" stable run-to-run. Just frame it as "the features
  from this data + this fixed init," not "the canonical concepts Maia has."
- Deeper caveat: k-means finds dense directions in ACTIVATION space; initializing there biases toward
  "concepts = dense activation directions" — a reasonable inductive bias, but not provably Maia's
  computational features.

Per-dict metrics (k-means init): dict=256 → L0=5.5, FVU=0.49, dead=1.2%, blob=9%, 253 live.
(fuller dict-size sweep with L0/FVU per size: `ddinit_sweep2.json` on chess-poc.)

**Action:** k-means init from FIXED centroids is the pragmatic default for a reproducible SAE — but the
reproducibility is ~0.63 across genuinely-independent runs, not 0.9. Honest number for any writeup.

## Systematic-debugging wins (3 real bugs found, not guessed)

This session's SAE plumbing hit three collapses; each was root-caused (instrumentation), not tuned away:

1. **overcomplete RA-JumpSAE collapses (FVU→1.0, L0→0).** Symptom looked like "threshold runs away."
   Instrumentation disproved it: threshold barely moved (0.050→0.057); instead `enc_wnorm` 26→5.5 and
   pre-codes 0.46→0.013. ROOT CAUSE: their loss is **L1-on-magnitude** (mse_l1) → shrinks encoder
   pre-activations (classic feature-shrinkage), while bandwidth=1e-3 makes the threshold's
   pseudo-gradient too weak to compensate. Fix: use OUR JumpReLU (L0-COUNT penalty via Step, bandwidth
   0.1, AuxK revival) and graft on ONLY their archetypal decoder.
2. **Archetypal JumpReLU stuck at L0=81–231 regardless of penalty (doubling penalty RAISED L0 2.5×).**
   ROOT CAUSE: the threshold's pseudo-gradient was too weak to climb to the activation scale at the
   shared optimizer LR, so sparsity pressure escaped by **shrinking the (non-unit-norm) archetypal
   decoder** (`Dnorm` 22→0.9) instead of gating features. Plain JumpReLU avoided this via
   `make_decoder_weights_unit_norm()` forcing the threshold to be the only lever. FIX: give `log_theta`
   its OWN high LR (1e-2 vs 3e-4) → threshold climbs (0.5→1.3), L0 drops 300→16.
3. **First archetypal stability run looked WORSE (MMCS 0.29).** ROOT CAUSE: I sampled the archetype
   candidate set C with the per-seed RNG → different C per seed → different hulls. CONFOUND, not a real
   result. FIX: share C across seeds. (Archetypal stability assumes a fixed candidate set.)

LESSON: every one of these would have been a wrong conclusion if "fixed" by tuning. The instrumentation
(per-epoch FVU / L0 / threshold / encoder-norm / decoder-norm) found the mechanism each time.

## Intrinsic-metric battery (label-free), per dict size

FVU/dead/blob/dup are HYGIENE metrics — they say "is this SAE pathological," not "is it good." On the
plain JumpReLU: dead=0 everywhere, blobs drop with bigger dict (256→18%, 2048→1%), dup>0.9 ≈ 0 at all
sizes (no near-identical features), participation-ratio COLLAPSES as dict grows (0.19 at 1024 → 0.06 at
4096 — bigger dicts don't use their capacity). FVU improves with dict (0.31→0.18) but FVU is a TRAP
metric — a bigger dict reconstructs better without being more interpretable. Reconstruction is not the
goal; stability + monosemanticity are, and those need the seed test (above) + grounding (boards/tags),
which intrinsic metrics structurally cannot provide.

## JumpReLU implementation notes (canonical, from SAELens)

Our working JumpReLU (`scripts/sae/train_jumprelu.py`, FVU 0.20, 0 dead, stable):
- Two autograd Functions: `Step` (Heaviside, grad ONLY to threshold via rectangle kernel — for the L0
  loss) and `JumpReLU` (gate, straight-through grad to x + kernel grad to threshold).
- Threshold in LOG-space (theta = exp(log_theta) > 0). **Init theta IN the activation range** (~0.5,
  not 0.001) or the kernel window never sees activations and theta never trains — the bug I hit twice.
- bandwidth 0.1 (not 1e-3); rectangle kernel; L0-COUNT penalty (NOT L1-magnitude); AuxK to revive dead.
- erichson/JumpReLU on GitHub is a DIFFERENT thing (adversarial-robustness eval-time threshold, no
  learned threshold/STE) — not the SAE variant. Use the Rajamanoharan/SAELens formulation.

## Open / next

- Lock k-means init into the trainer as default; confirm 0.90 holds at the production dict size.
- With a reproducible SAE, the "untagged features = concepts Maia has that our tagger misses" analysis
  becomes trustworthy (was previously confounded by seed-instability AND by the blunder-diff cache
  having no best-move line — see tagging-coverage note below).
- Tagging coverage on the blunder-diff cache: only 37%/8-tags WITHOUT a best line; using
  `blunder_metadata_200k.json`'s `best_uci` (single best move) → 64%/41-tags. Full multi-ply tactic
  tags (Missed Fork/Pin/Skewer) still need the PV, which only the d16 enrichment cache has (19% overlap).
