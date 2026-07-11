# JumpReLU SAE on the l7 best−blunder diff — the hyperparameter that actually matters (2026-07-11)

**What this is:** a JumpReLU SAE trained on the SAME representation as the last labeled BatchTopK model
(`btk_2048_k6_nol2`, see `2026-06-05_last_labeled_sae_k6_v7.md`) — Maia3 79M layer-7 mean-pooled
**best − blunder diff**, 1024-dim, `chess-stage-a/cache/maia3_l7only_v2_dedup.pt` (168,132 × 1024,
z-scored). Goal: good models with per-feature fire rates in the **1–5% band** (Sam), L0 flexible.

## THE finding: `init_threshold` is the dominant knob — NOT `l0_coeff`, NOT `bandwidth`

The JumpReLU threshold θ (per-feature, learned in log-space) **cannot travel far during training** —
its gradient only flows through a narrow straight-through kernel window `[θ−bw/2, θ+bw/2]`. Measured
pre-activation nonzero **median = 0.565**, p90 = 1.63 (probe_scale.py). Consequences, all measured:

- **`l0_coeff` from 0.01→0.1 (10×) barely moved anything** — θ stuck at ~init, mean L0 pinned ~940,
  all 2048 features fire >10%. This is the documented "threshold can't climb → stuck L0" root cause
  (2026-06-17 seed-stability doc). The penalty is irrelevant when the gradient path is empty.
- **`bandwidth` (0.02 → 0.5) was nearly irrelevant** too — same starved-gradient story.
- **`init_threshold` was everything.** θ settles near the pre-act median (~0.6) but only if it STARTS
  close. You must **initialize θ where you want the cut**, not expect it to get there.

| init θ | 1–5% band | blobs>10% | dead | FVU | θ settled to |
|--------|-----------|-----------|------|-----|--------------|
| 0.06   | ~40       | ~1550     | 0    | 0.004 | 0.08 (barely moved → dense) |
| 0.30   | 1388      | 352       | 0    | 0.083 | 0.60 |
| 0.40   | 1670      | 232       | 0    | 0.103 | 0.75 |
| 0.50   | 1707      | 154       | 0    | 0.122 | 0.90 |

Higher θ → more features specific (1–5%), fewer blobs, worse reconstruction. A clean monotone dial.
**Sam's first pass died at θ=0.5 with bw=0.001** — but that death was the tiny bandwidth (empty kernel
→ θ frozen high → everything below → all dead), NOT θ=0.5 itself. With bw≥0.02 + AuxK revival, θ=0.5
is the sparsest healthy model (0 dead). Scale-matching bw to the activation range is what fixed it.

## The winner (for the 1–5% target): `jr_thr0.40_l00.02`
`init_threshold=0.4, bandwidth=0.2, l0_coeff=0.02, dict=2048, 60 epochs`. Per-feature fire rate
**median 2.44%, IQR 1.90–3.57%** — the bulk of the dictionary sits dead-center in the target band.
FVU 0.103, **0 dead**, all 2048 active, 232 blobs. Best specificity/reconstruction balance.
- **Sparser alternative:** `jr_thr0.50_l00.02` — median fire 1.79%, only 154 blobs, FVU 0.122.
- **Best reconstruction:** `jr_thr0.30_l00.02` — median 3.69% (p75 6.16% leaks past 5%), FVU 0.083.

## Artifacts
- Weights: `output/jumprelu_l7diff/jr_thr0.{30,40,50}_l00.02.pt` (local, gitignored — 16MB each) +
  on chess-poc `~/SageMaker/jr_sweep_out/`. Each `.pt` has `state_dict`, `mean`, `std`, `config`.
- Full sweep: `output/jumprelu_l7diff/jr_sweep_results.jsonl`.
- Scripts (committed): `scripts/sae/train_jr_sweep.py` (JumpReLU + AuxK + fire-rate/FVU/θ eval),
  `scripts/sae/jr_diag.sh` (bw×thr matrix), `scripts/sae/jr_sweep.sh` (the thr×l0 sweep).

## ⚠️ S3 discrepancy found
`output/S3_INVENTORY.md` documents `s3://chess-stage-a-140023406996` for weights, but that bucket
**does not exist** from the notebook's creds (NoSuchBucket). The l7 cache lives only in the notebook's
LOCAL `~/SageMaker/chess-stage-a/` dir, not S3. Weights backed up to local `output/jumprelu_l7diff/`
instead. The inventory is stale re: the weights bucket — reconcile when convenient.

## Next: labeling (Sam's hint — use the tagger as the starting point)
Per-feature: pull top-firing positions from the cache, run `scripts/04_tagger/tagger.tag_mistake_full`
on each, and name the feature by its DOMINANT tagger label (deterministic, free, grounded — replaces
the old Opus-reads-20-boards pipeline for the labelable subset). Features with no dominant tag stay
low-confidence (the mechanism-ceiling cases). Not yet run.
