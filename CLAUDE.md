# Chess Deck — Research Package

Research package of the chess-deck workspace. SAE training, labeling, evaluation, player-profiling pipelines.

**Umbrella:** [`../../CLAUDE.md`](../../CLAUDE.md) maps both packages and the research→code handoff.
**Shared concepts:** [`../../knowledge.md`](../../knowledge.md) — SAE extraction pipeline, version tagging, DDB schema, S3 layout, handoff contract, core gotchas. Authoritative. Read before anything else in this package.

## Package Docs

| Doc | What |
|-----|------|
| `plan.md` | Current state, queue, backlog (research only — cross-package work in `../../plan.md`) |
| `knowledge.md` | **Research-specific** — architecture comparisons, labeling pipeline, player profiling. Cross-package concepts are in `../../knowledge.md`. |
| `log.md` | Session narratives, lessons from failed approaches |
| `PIPELINE.md` | Labeling procedure — step-by-step from raw encoder outputs to shipped label JSON |
| `README.md` | Package intro, setup |
| `output/S3_INVENTORY.md` | **Source of truth** for what's in S3 (`s3://chess-stage-a-140023406996/`). Update when you add/remove weights, labels, or cache artifacts. |

## AWS account

**This package runs on account `140023406996`, profile `default`.** Research, not production.

- Bedrock batch (Haiku, Sonnet, Opus, Gemini) for labeling + analysis
- S3 (`s3://chess-stage-a-140023406996/`) for weights, labels, cached activations
- SageMaker notebooks: `chess-poc` (GPU — training + eval), `curly-lock` (encoder research), `short-heart` (personas)

**Never run research scripts on account `934822760657` (chess-deck production).** That account is deploy-only for the sibling package `../chess-deck-code/`.

## The handoff to code

One-way. Full contract in `../../knowledge.md` § handoff.

Short version:
1. Train SAE → `s3://chess-stage-a-140023406996/sae/weights/sae_{family}_{dict}_k{k}.pt`
2. Label features → `output/labels_{name}.json` (committed to git) + uploaded to S3
3. Run baselines on chess-poc → `output/baselines_{name}.json`
4. `../chess-deck-code/backend/scripts/ship_sae_version.py` reads from S3 + research output, produces a versioned Lambda bundle, CDK deploys. Research never touches that script directly — it's the code package's responsibility.

Research code should **not** read from or write to production DynamoDB or the chess-deck AWS account.

## Naming conventions (non-negotiable — apply mid-session, don't defer)

- Weights: `sae_{family}_{dict}_k{k}.pt` (e.g. `sae_real_btk_2048_k64.pt`)
- Labels: `labels_{weights_name}.json`
- Metrics: `{metric}_{weights_name}.json`
- Versions that ship to production: `realgames_{dict}_k{k}_v{N}` (e.g. `realgames_512_k8_v1`)

Consistent naming is what keeps `ship_sae_version.py` working and `S3_INVENTORY.md` readable.

## Skills

Global skills from `~/.claude/CLAUDE.md`. For this package specifically:
- `/notebook` — exec on chess-poc / curly-lock / short-heart. GPU is shared, `nvidia-smi` before launching training.
- `/organize-research` — end-of-session wrap-up specific to research (weights in S3, scripts in git, `S3_INVENTORY.md` current, reproduction test on the session's main result).
- `/organized` (archived — see `~/.claude/skills/_archive/organized/`) — predecessor of `/organize-research`.

## Commands

```bash
# Run a script locally (uses default profile → research account)
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
python scripts/...

# Run on chess-poc (GPU)
sais -n chess-poc exec 'cd ~/SageMaker && python scripts/...'

# S3 access
aws s3 ls s3://chess-stage-a-140023406996/sae/weights/
```

## Current Priority

See [plan.md](plan.md) for queue, `../../plan.md` for cross-package priorities.
