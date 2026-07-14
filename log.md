# Chess Lab — Log

## 2026-07-14 — wide SAE-tagger audit: loudest tags (delete / gate / leave)

Ran the SAE-feature-audit playbook wide over the loudest tagger fires, reading actual FENs (not Opus
summaries). Three outcomes, all committed:

- **Deleted both battery detectors** (`missed_battery` 4%, `allowed_battery` 9%) — naked-rate catch-alls.
  Allowed Battery TOPPED 118 SAE features across **81 distinct Opus concepts** (Hanging Piece, Missed
  Tactic — never "battery") and appeared in 44% of vote tallies; 74-76% co-fired a sharper tag; only 4-5%
  had battery as the sole explain tag, and those FENs were diffuse drift. 184→182 taxonomy tags.
  **CORRECTION (later same day):** deleting `missed_battery` was partly wrong. Sam gave a counterexample
  (`...Qb6` builds Bc5+Qb6 on f2) and my "no battery exists" proof used a BROKEN finder — it required the
  battery's back piece to DIRECTLY attack the target, impossible by definition (front piece blocks the
  line), so it never matched a real stacked battery. **Rebuilt `missed_battery`** with correct XRAY
  geometry + quiet-move + defended-target gates: fires **1.0%** of corpus (real-tactic band), 57 verified
  sole-lesson positions, 53/57 defended-target, 48/57 pure positional pressure. `allowed_battery` stays
  deleted (it was the genuine 9% catch-all). Taxonomy back to 183. Lessons recorded in the knowledge doc
  (validate a finder against a known positive before trusting emptiness; SAE-can't-see-it ≠ tagger-can't-
  detect-it; one hand-verified FEN beats a mis-instrumented 60k scan).
- **Gated Missed Open File** (6.4%→4.9%) — a REAL positional concept (tops only 19 features, 7% of votes),
  but ~23% of fires were tactics that just landed a rook on an open file: captures (Rxc1 wins material)
  and endgame checks (Rd5+). Concept gate: best move must be neither a capture nor a check. Dropped ~867
  incidental fires; residue is genuine quiet file-occupation.
- **Left Allowed Mate + Hung\* alone** — loud but correct. Allowed Mate (7%) verified **0% NO_MATE**.
  Hung\* is a net-material-loss-over-line detector (multi-move hangs expected); ~92% clean on the hardest
  bucket. Filed **issue #58** for a rare peak-victim naming edge (even rook trade → "Hung Rook").

Regression 156→**163** (3 open-file + 4 battery cases; battery twin fixture retargeted to Doubled Rooks),
all green. Knowledge: `knowledge/2026-07-14_battery_catchall_deletion.md` (delete/gate/leave scorecard +
the battery error + rebuild + lessons). Commits `b390302` (battery delete), `61544b6` (open-file gate),
+ battery rebuild commit. Not pushed. Not yet shipped to prod.

## 2026-07-14 (cont.) — missed-forcing-move gap: 237→6 (two fixes)

Continued the audit into the "missed a check/winning-capture, no tag fired" gap. Re-derived from scratch
(didn't trust an earlier rough estimate) — 237 positions, splitting into two concepts:
- **Fix 1 (`7a6ef1c`):** `capture_or_exchange` now gates the defended-capture "sacrifice" exclusion on
  SEE, not piece values. A queen taking a defended knight that nets +3 over the recapture was wrongly
  called a sac and silenced (95 positions). SEE<0=sac (excluded), ≥1=Missed Free X, ==0=Exchange. +1420
  firing, 0 silenced; 109 capture-mates all co-fire Mate.
- **Fix 2 (`a9fb690`):** new `wrong_check` detector — you checked, but a DIFFERENT check was best (the
  missing third case beside pointless_check / missed_attacking_check). 1.42% fire, 79 sole-explain,
  Calculation category. Message names the better check.
- Verified the quiet-check half was already covered by missed_attacking_check (0 residual). Gap 237→6;
  residual 6 = negligible edge cases, noted not chased.

regression 161→**169**. Knowledge: `knowledge/2026-07-14_missed_forcing_move_gap.md`. Both fixes applied
the battery lesson (re-derive, read FENs, root-cause each detector before editing). Not pushed.

## 2026-07-11 — JumpReLU sweep on the l7 best−blunder diff (Sam AFK, autonomous)

Task: train the same/similar setup as the last labeled SAE (k6/v7 l7-diff) but with JumpReLU; sweep
hyperparameters for good models with 1–5% per-feature fire rates.

- Refreshed research creds (`ada`), re-authed chess-poc (SAIS token was expired — the recurring
  gotcha). GPU idle, l7only cache present (168,132×1024). **Midway expired → couldn't read the
  SandstonePersonas tanh-loss reference; used the proven canonical JumpReLU script on the notebook
  instead of reconstructing the loss from memory.**
- **Root-caused the tricky hyperparameters:** measured pre-act nonzero median = 0.565; θ can't travel
  far during training (narrow straight-through kernel), so `init_threshold` is the DOMINANT knob —
  `l0_coeff` (10× sweep) and `bandwidth` barely move anything. Sam's death config (θ=0.5, bw=0.001)
  died from the tiny bandwidth freezing θ high, not θ=0.5.
- Diagnostic matrix (bw×thr) → real sweep (thr{0.3,0.4,0.5}×l0{0.02,0.06}, bw=0.2, 60ep, dict=2048).
  Monotone frontier, **0 dead across all configs.** Winner for the 1–5% target: **`jr_thr0.40_l00.02`**
  — per-feature fire median 2.44%, IQR 1.9–3.6%, FVU 0.103, 232 blobs. thr0.5 sparser (median 1.79%),
  thr0.3 best FVU (0.083, but p75 leaks >5%).
- Weights: local `output/jumprelu_l7diff/` + chess-poc `~/SageMaker/jr_sweep_out/`. **S3 gotcha:** the
  inventory's `chess-stage-a-140023406996` bucket doesn't exist from the notebook — backed up local.
- Full detail: `docs/knowledge/2026-07-11_jumprelu_sweep_l7diff.md`. NEXT: label via tagger-vote on
  top-firing positions (Sam's hint).
- **Labeling (tagger-vote):** encode cache through winner → per-feature top-200 firings → dominant
  tagger tag. Took 3 passes: v1/v2 compromised by a stale-tagger import (two copies on the notebook,
  sys.path shadowing; "Allowed Battery" overfired 284 feats). v3 (fixed tagger, verified pre-run):
  **570/2048 labeled, 135 at ≥0.5 conf.** Median conf 0.34 = features genuinely polysemantic (the
  mechanism-ceiling); tagger abstains on ~72%. Labels: `output/jumprelu_l7diff/feature_labels_jr_thr0.40.json`.
  PERF: ~62min/model, CPU-bound (python-chess board build) — cache Board objects to speed up.

## 2026-06-05 — labeling v3→v7: debias, refutation, xhigh, peak+median (the over-specification fix)

Long session iterating the d2048_k6 feature labeler through five versions, each fixing a defect
Sam caught by reading boards. The through-line: **labeling from the top-10 (peak) activations
systematically over-specifies** — peak boards are the most extreme and piece-homogeneous, so the
label inherits a specificity the feature doesn't actually have.

- **v2 (debias):** v1's prompt said "prefer Missed X if a capture existed" → 275 features had
  "Missed X" chips while the played move actually hung material. Removed the steer. missed_win
  1020→905, hung_own 478→707. (Audited f19/f745 with Stockfish — and learned the `drop≈gap`
  heuristic is tautological garbage; the real hang-vs-miss tell is played-vs-best CAPTURE.)
- **v3 (5-word mechanism chips):** dup chips 57%→78% unique. Became the basis for LLM clustering
  (decoder + chip-embedding clustering both FAILED — produced muddy "Hangs" blobs; the coaching
  theme lives in the label, so Opus-reads-labels clustering is the method). 12 categories,
  validated 3× on disjoint 200-samples. Atlas rebuilt as a navigable SPA (was 176MB inline-SVG;
  now 1.6MB, boards client-side).
- **v4 (REJECTED):** added refutation_analysis BUT also result-framing — over-steered f882 to
  generic "Leaves piece undefended". Lesson: refutation good, result-framing bad. Removed.
- **v5 (refutation + confidence + opus-4-8 xhigh):** the model+thinking were the real unlock.
  Labelers had been running opus-4-6 with NO thinking; switched to opus-4-8 `thinking:adaptive` +
  `output_config.effort:xhigh` (the settings.json config). Recovered f882, fixed f1536 ("Hangs
  knight" → "Hangs piece, knight involved" — it was only 75% knight). Confidence flag now
  discriminates (was all-high before).
- **v6 (second pass on 482 flagged):** re-ran cons≤70 features showing v5's guess, told to look
  harder / stay honestly-uncertain. Fixed f1717 (false "Squanders winning position" → "Passive
  misses tactic" — trajectory showed it was tense/losing, not winning). 312 stayed honestly flagged.
- **v7 (peak+median — RUNNING):** the root-cause fix. `build_peak_median_profiles.py` samples 10
  peak + 10 median (p40-60) Opus-covered boards/feature; v7 labels from both + shows v6 as
  head-start. f103 ("Hangs queen to knight fork", cons 90, NOT flagged) → "Hangs to knight fork
  (often queen)" once it saw the median rook-forks. Chip form "Core (often X / major piece)";
  label narrates top→median broadening. Validated f103/f952/f882/f1536/f1717.

**Sam's catches drove all of it:** "is missed win bad? they're all blunders" (split by what's
missed), "I don't want you biasing it" (debias + no-rules), "f1536 only 75% knight" (piece
over-claim), "look at median positions not just peak" (the v7 insight), "you're over-prescriptive"
(trimmed prompts to minimal — facts in SEE block, not paragraphs).

**Redundancy proven a non-issue:** the 92 "hangs queen" features are decoder-distinct (cosine ~0.05)
and fire on disjoint boards (Jaccard ~0) — not the SAE wasting capacity, just same-named distinct
features. `decoder_overlap.py` + `firing_overlap.py`.

**Game application (cabbagelover5566, game 169732298592):** built `encode_game_blunders.py` (Maia3
L7-diff → k6 SAE → taxonomy, replicating corpus build). Found per-position diagnosis must read at
the CLUSTER level — a "queen knight-fork" feature fired on the player's ROOK fork (OOD structural
match at ~23% of the feature's peak). Updated the `/analyze-game` skill with the chess.com
callback-decode + the Maia3/v3 mapping step.

**State at session end:** v7 full run in progress on chess-poc. v6_merged committed as fallback.
plan.md + README updated. Next: finish v7 → re-assign → re-cluster → category-fit audit → atlas → game.

## 2026-06-04 — full d2048_k6 relabel with all-fields method + pipeline reorg

**The fix that mattered:** the old labeler (`label_features_integrated.py`) was feeding Opus only
~2.5 of the 7 per-position Opus fields — it dropped `best_moves_analysis` and `move_intent`. So
when SEE (single-ply, blind to traps/multi-move tactics) said `blunder_hangs_own`, Opus had no
way to see that the *best* move was a capture the player skipped → it labeled missed-X features as
"passive"/"hung". Caught earlier via f1487 (a rook trap SEE can't detect), f745, f950.

Wrote `scripts/03_feature_labeling/relabel_all_fields.py` (production version of the validated
20-feature prototype): top-10 boards with WHAT WENT WRONG + BEST MOVE + INTENT, one-line SEE floor,
single Opus call/feature, emits `consistency`. Verified parity on the 20 fids first (17/20 changed,
f1487→"Missed free capture", f745→"Missed winning capture" cons 80, f952 stays hung cons 90) before
the full run.

**Full run:** 2035/2047 labeled, 26 min on chess-poc, zero throttles. **1910 chips changed (94%)**.
`missed_win` = 1020/2035 (50%) — the direction error was dictionary-wide, exactly the f1487 bug
repeated across half the dictionary. 95 features ≤70 consistency flagged for review.

**Scaling decision recorded:** single-call + consistency-flag, NOT 3-vote consensus. The earlier
instability (f745 flipping missed↔hung) was a prompt bug (bloated 6-line SEE caveat), already fixed
by trimming to one line. 3-vote would pay 3× to fix a solved problem; the emitted `consistency`
already flags mixed features for free.

**Reorg:** moved the feature-labeling arc into `scripts/03_feature_labeling/` (Personas NN_stage
convention) with a README documenting run order + the SEE-is-single-ply lesson. Older one-off
labelers (gemini, pass2, btk, synthesize) left in `scripts/labeling/` — not deleted, flagged for a
later cleanup pass.

**Re-bucket (same session):** rebuilt the taxonomy bottom-up on the new labels.
- Mechanical clustering (Titan embeddings + agglomerative) was a DEAD END: prose labels share the
  "Player consistently..." scaffold so everything is cosine-near everything; no flat distance cut
  gives usable groups (0.45→156 clusters, 0.55→one 958-feature blob, 0.75→collapse). Chip-only
  embedding separates better but still 150+ clusters. Recorded, not pursued.
- What worked: Opus derives the top-level set from a 200-feature sample (chip+label + the
  self-inflicted-vs-omission SEE tell), reusing the old 11 as a non-forced starting hypothesis.
  Ran on **3 disjoint 200-samples** — all three independently converged on the same 11-12 buckets
  within a few % each. The one disagreement (is "Abandoning Defensive Duty" its own bucket) went
  2-of-3 yes → kept. Final: 12 buckets (`buckets_v3_d2048_k6.json`).
- Assigned all 2035 (`feature_buckets_v3_d2048_k6.json`): only 5 unassignable, biggest bucket 18%,
  NO catch-all — the omissions split into Missed Hanging (14%) / Missed Tactic (18%) / Missed
  Check-Mate (8%). The original "missed_win = 50% of dictionary" problem is resolved by carving on
  WHAT was missed. fire% >> feat% in Left-Hanging & Missed-Tactic = those buckets hold the blobs.
- The assigner has NO rules block (unlike old assign_to_buckets.py) — buckets + evidence + the
  self-inflicted/omission axis only. That axis is what stops "missed a better move" becoming a
  catch-all (every blunder had a better move).

Sam drove the key corrections this session: "is missed win a bad category? they're all blunders"
(→ split by what was missed), "I don't want you biasing it at all" (→ unbiased emergent run + no
rules in assigner), and "try a third one" (→ 3-sample convergence as the validation).

**Debias + sub-bucket (same session, cont'd):** while sub-bucketing, found the v1 relabel was
direction-biased — its prompt said "prefer Missed X if a capture was available," producing 275
features where the chip says "Missed" but the played move hangs own material. Sam pushed back
("is missed win bad? they're all blunders"; "I don't want you biasing it"; "use nearly the same
prompt without the bias"). Audited f19/f745 with Stockfish — and my FIRST audit heuristic
(drop-vs-gap) was garbage (those quantities are tautologically equal; gave confident-wrong
verdicts). The real tell: PLAYED-move-captures vs BEST-move-captures. v2 neutral relabel
(`relabel_all_fields_v2_neutral.py`) reran all 2035: missed_win 1020→905, hung_own 478→707.
Re-assigned to v3 buckets (now ~56/33/10 self-inflicted/omission/endgame), sub-bucketed
mechanically by piece/theme, rendered the browsable tree. Lesson saved as memory
[[project_direction_arbiter_is_board_not_see]]. Tree: `output/atlas/taxonomy_v3_d2048_k6.html`.

**Open:** blob handling (General-Tactic + Left-Hanging subs hold high-fire blobs); review 130 flagged.

## 2026-06-01 (session) — Maia3 v2 SAE bakeoff → l7only winner → overnight labeling

**Big session. Two wrong claims made and corrected mid-session — recorded because the corrections are the lesson.**

### Bakeoff: 4 v2 SAEs (k=16, 2048) on Maia-best@2600 diffs
- Caches: `board_diff` (ONNX, 512d), `option_a` (ONNX, 512d), `l7only` (79M PyTorch, 1024d), `l2l7` (79M concat, 2048d). All from `maia_best_200k.json` (199,433 → 168,669 after dropping no-op diffs).
- First eval (motif-match vs Opus join) was **CONTAMINATED** — scored Opus-join *coverage*, not SAE quality. board_diff "won" 4/10 only because its top features landed on the ~9.6% of positions that overlap the 19k Opus labels. DISCARDED. Lesson (again): never score against a join that barely connects.
- Built honest instrument instead: `output/feature_boards.html` — each test position's top-firing feature expanded to its top-12 activating boards as clickable chess.com diagrams. No proxy, judge by eye.

### Two corrected errors (the real lessons)
1. **"l2l7 collapsed to 450 live features"** — WRONG. The 450 was liveness measured over only the 10 test positions with a strict top-k threshold. Measured over 40k corpus positions, both l7only and l2l7 use all 2048 features. Artifact of sampling on 10 positions.
2. **"Diffs cancel signal, raw activations preserve it"** — WRONG. ALL FOUR caches are best−blunder diffs (l7only included). Read the build scripts: the real dissociation is **model**, not diff. board_diff/option_a use the ONNX probe (`maia3_with_probe.onnx`) → mush. l7only/l2l7 use 79M PyTorch (`maia3_79m_fixed.pt`) → coherent. The "fixed" 79M ties to the [maia3-best-move-extraction] memory — ONNX path had wrong move vocab.

### Decision
- Sam manually inspected `feature_boards.html`: **l7only and l2l7 both good** (features carve *mistakes* — mate-in-one, hung piece — not just position types). board_diff/option_a mush.
- l7only = literally the L7 half of l2l7 (`activations[:,1024:]`). L2-vs-L7 weight split is 50/50 but only because normalization amplifies L2's 6.5×-smaller variance — possible noise-laundering. Sam preferred **l7only** by eye (simpler, can't be contaminated by amplified L2). DECIDED: label l7only.

### Labeling (proven 2-pass pattern, Opus 4.6 + Stockfish trajectory)
- Pattern: `enrich_all_positions.py` (Stockfish depth-18 → `eval_before -> eval_after`, punish_type, n_good_moves, top-3 best/refutations) → `label_all_positions_opus.py` (Pass-1, per-position) → `label_features_pass2.py` (Pass-2, synthesize top-10 per feature). Stockfish trajectory IS injected per-example in Pass-2 prompt.
- Stage A: l7only top-15 profiles, 2048 feats, 14,824 gap positions lacking enrichment.
- Stage B: Stockfish-enriched 14,793 gap positions (depth 18, 48 workers, ~26min). Cache 19,362 → 34,186.
- **GATE (Pass-2 pilot on 200 already-covered features): PASSED.** Median confidence 72, 0 junk chips, 195/200 unique, sharp named mistakes (f71 "Premature Bxf7+ in Modern Defense", f466 "Premature fxe5 allows Qh5+"). Low-conf ones self-report as vague → confidence is calibrated/trustworthy. Caveat: pilot subset skews to opening blunders — check corpus phase skew tomorrow.
- Throughput fix: Pass-1 ETA was 19h @ concurrency 20 / max_tokens 32000. Real output ~700 tokens; 32k over-reserves Bedrock TPM quota. Changed to max_tokens 8000 + concurrency 60 (thinking budget 4096 preserved) → **ETA 5.6h, throttles=0**. Same prompt/model/quality.

### Overnight (running at session end)
- `overnight` screen on chess-poc: Pass-1 (14,885 positions, ~5.6h) → Pass-2 `--resume` (~1,848 remaining features, ~45min).
- Outputs: positions → `all_positions_labeled_opus.json` (backed up to `all_positions_labeled_opus.bak_*.json`); features → `maia3_l7only_feature_labels.json`.
- Morning TODO: coherence stats on full 2048, sample-read, check opening-phase skew, then commit labels + update knowledge/plan. NOT yet committed to git or S3.


## 2026-04-14 (session 3 continued) — Blunder SAE full sweep + categorization

**Big session.** 9 SAE variants trained, 5 labeled, full comparison, winner selected, categorization explored.

### Results
- **Winner: 2048 k=32 move-token** — 1,080 unique labels, 65% uniqueness, 1.56% median FR
- All variants got ~60% high-confidence labels (up from 27% in old blunder SAE)
- Move-token fix was key: all-token → 20-31% FR, move-token → 0.8-3.1% FR
- 1024 too coarse (misses 71% of 2048), 4096 diminishing returns (44% redundant), 8192 structural fine but overkill

### Labeling
- 3 Bedrock Batch jobs: k=64 (`mjgqyjem1w28`), k=32 (`ypr3017mqa9s`), k=128 (`9m6cs1aioq3k`) — all completed
- Top categories: hanging_pieces (20%), endgame_technique (17%), passed_pawn (11%), deflection (12%)
- Within-category Jaccard: features are distinct (<0.5) but labels are bottleneck (40% get generic names)

### Analysis
- Pairwise Jaccard (full matmul, no sampling): 0.12-0.19 mean across SAE pairs. Features are unique across SAEs.
- Pre-topk energy: 318 features naturally activate, top-64 = 60% energy, top-128 = 75%
- Greedy set cover: 22 features cover 95% of positions BUT top features fire 15-20% (too broad)
- Fire-pattern clustering: endgame features cluster cleanly, tactical features don't (positions overlap)
- Key insight: "overloaded defender" and "hanging piece" fire on overlapping positions — same blunder from different angles

### Categorization (in progress)
- Sonnet's categories are good but flat (22 categories, no hierarchy)
- Dedup at Jaccard 0.8 + clique grouping at 0.3 — script running
- Endgame features → clean coaching topics. Tactical features → need tags not categories.
- Explored: Heisman mistake taxonomy, greedy set cover, hierarchical clustering, decoder-direction clustering
- Decision: dedup → group → relabel groups with coaching-focused prompt (short_label, coaching_advice, theme/subtopic)

### Infrastructure
- Repo consolidated: everything in chess-deck-research (plan, log, findings, learnings, scripts, output, docs, archive)
- Cleaned hooks: 11 → 4 (session-start, post-compact, drift-nudge, anti-poll)
- Updated /organized skill: added S3, git commit, naming, two-phase, cheap-before-expensive habits
- Fixed IAM: SageMaker role can now PassRole for Bedrock Batch
- CLAUDE.md updated to point to chess-deck-research for SAE research

### Scripts committed
- `compare_saes.py` — full matmul Jaccard
- `quality_filter.py` — confidence + FR + mono filter
- `within_category_jaccard.py` — redundancy within categories  
- `pretopk_energy.py` — natural sparsity analysis
- `label_breakdown.py` — category comparison
- `cluster_features.py` — hierarchical clustering
- `greedy_feature_selection.py` — set cover
- `dedup_and_group.py` — Jaccard dedup + clique grouping
- `label_blunder_coaching.py` — coaching-focused labeling prompt
- `cache_move_token.py` — extract hidden[77]

### Next session
- Check dedup_and_group.py results (running on notebook)
- Relabel grouped features with coaching prompt
- Detection scoring on 2048 k=32
- Deploy puzzle SAE (Queue item 2 still waiting)

## 2026-04-13 (session 3) — Blunder SAE experiment

- Starting blunder-trained SAE experiment (Queue item 3 from plan.md)
- Scripts: `cache_blunder_activations.py` (two-phase: download+filter → batch encode), `train_blunder_sae.py`
- Fixed original caching script: wrong BASE path, no batching, print bug at 5K boundary
- Rewrote as two-phase pipeline: Phase 1 = CPU-only HuggingFace streaming, Phase 2 = batched GPU encoding
- Notebook: chess-poc (ml.g6.16xlarge, L4 GPU, 242GB RAM)
- Phase 1: 200K blunders from 1.24M positions (7.2M rows) at 247/sec, 16.1% hit rate, 809s total
- Phase 2: 400K forward passes (blunder + best), batch_size=64, ~20 min per pass
  - Blunder encoding: 200K in 1208s
  - Best encoding: 200K in 1220s
  - Cache saved: 60GB (`blunder_acts_200k.pt`)
- Training BTK 2048 k=64 + aux on 15.4M blunder activations (200K × 77 tokens), 5 epochs, 306s
  - ep0: mse=0.151, dead=1 → ep4: mse=0.136, dead=3
  - Final eval: dead=89 (4.3%), alive=1959, L0=64, FVU=0.129, c_dec=0.034
  - **Structural metrics pass** — comparable to puzzle SAE (10% dead, FVU=0.082, c_dec=0.036)
  - Fewer dead, more alive, higher FVU (blunders harder to reconstruct but features are well-separated)
- Weights saved to S3: `s3://chess-stage-a-140023406996/sae-weights/sae_btk_blunder_2048_k64_aux.pt`
- Profiling: 2,048 alive, mean fire rate 31.1% (higher than puzzle SAE — blunders more diverse)
- Profiles saved: `s3://chess-stage-a-140023406996/sae-eval/profiles_btk_blunder_2048_k64.json`
- Fixed IAM: added PassRole permission to ChessResearchSageMakerRole for BedrockBatchInferenceRole
- Sonnet+thinking labeling batch submitted: `wtewr9qxt9gy` (2,048 features), combined batch `63ouxzbuzjh2` (5,900)
- **High fire rate (31% mean for 2048, 20% for 4096) is too high** — target is <5%
- Root cause: trained on all 77 tokens (64 board + extras), but production uses only hidden[77] (move token)
  - Board tokens = "what position is this" → fires broadly across position types
  - Move token = "what kind of move is this" → should fire selectively on move patterns
- Previous "move-token-only" attempt used wrong token (index 76 = last FEN token, not index 77 = move token)
- Built `cache_move_token.py`: extracts only hidden[77] from encoder output, ~400MB cache
- Also trained 4096 k=64 on all tokens: dead=836 (20%), alive=3260, FVU=0.125
- Correct move-token pipeline: cache_move_token.py extracts hidden[77] from encoder output
  - Move-token cache: 804 MB (vs 60GB for all-token cache)
  - Training instant: 8-14 seconds for 200K activations
- **Move-token results:**
  - 2048 k=64: dead=9, alive=2039, FVU=0.093, **fire rate mean=3.15%, median=2.00%** ✅
  - 4096 k=64: dead=26, alive=4070, FVU=0.085, profiling in progress
  - Move-token fixed the fire rate problem (was 31% all-token → 3.15% move-token)
- **Move-token profiling results:**
  - MT 2048: 2,033 alive, 15 dead, **fire rate mean=3.15%, median=2.00%** ✅
  - MT 4096: 4,027 alive, 69 dead, **fire rate mean=1.59%, median=0.84%** ✅
  - Both within <5% target. 4096 is more selective (sub-1% median)
- Uploaded to S3: `sae_btk_blunder_mt_2048_k64_aux.pt`, `sae_btk_blunder_mt_4096_k64_aux.pt`
- Move-token labeling batch submitted: `mjgqyjem1w28` (6,060 features)
- Also running all-token Sonnet labeling batches: `wtewr9qxt9gy` (2048), `63ouxzbuzjh2` (both) — for comparison
- Updated /organized skill with S3, git commit, naming conventions

## 2026-04-12 (session 2) — Sonnet labels, detection comparison, k=32 aux, 4096 sweep

- Sonnet+thinking labeling completed (`pztzjp2jzh8v`): 1,872/1,961 parsed, 30.6% poly flagged
- Detection scoring with Sonnet labels (`ac6bc19768ax`): BA 0.632 (+0.013 vs Haiku labels), 325 STRONG
- Full 5-condition detection comparison: enrichment >> judge quality >> label quality
- 4096 k=64 + aux completed: 3,017 active, 1,079 dead (26%), c_dec=0.035
- 4096 k=128 + aux crashed epoch 3 (GPU OOM on eval)
- 2048 k=32 + aux: 184 dead (9%), 1,864 active — aux fixes k=32 same as k=64 (57% → 9%)
- 4096 k=32 + aux: reran solo (earlier crash was memory pressure from sequential sweep). 2,908 active, 1,188 dead (29%), FVU=0.126, c_dec=0.041
- Organized research scripts into chess-deck-research GitHub repo, proper git workflow
- Scripts committed: eval_sae_checkpoint.py, parse_batch_results.py, profile_sae.py, sweep_k32_aux.py, profile_and_label_all.py
- Profiling 3 variants running (2048 k=32, 4096 k=32, 4096 k=64) — all from git repo scripts
- Next: upload profiles → Sonnet+thinking labeling batch → enriched detection scoring → compare all 4
- Created /organized skill: "never run code on remote that isn't a committed script"
- Consolidated drift-nudge hook to invoke /organized every 50 tool calls
- Notebook resized to g6.16xlarge (256GB) for 30GB cache loading, don't downsize yet
- Key insight: dead features aren't bad, active count is what matters
- 4096 profiling fixed: encoder weight key auto-detection (linear/w vs query/kernel), batch_size=32
- 4096 k=32 profiled: 3,287 alive, 809 dead, fire rate mean=10.35% median=2.00%
- 4096 k=64 profiling in progress (16K/50K)
- 2048 k=32 labeling batch submitted (`9tve7y1jz72h`), Sonnet+thinking, stuck in Scheduled >1hr
- Fixed batch_label_and_score.py: max_tokens must be > thinking.budget_tokens
- All 4 variants labeled with Sonnet+thinking: k=32 poly=3.5%, k=64 poly=30.6%
- Detection scoring complete: 2048 k=64 wins (BA=0.632, 659 HOLDS, 325 STRONG)
- k=32 less polysemantic but worse on detection — poly ≠ quality
- 4096 worse per-feature than 2048 — extra dict capacity doesn't help
- All results in chess-deck-research/output/COMPARISON.md

## 2026-04-12 (session 1) — k=64 + aux loss, enrichment pipeline, polysemantic validation

- Trained BTK 2048 k=64 with auxiliary dead-feature loss on 200K puzzles. Dead: 57% → 10%.
- Added aux loss to all 3 training scripts (train_tactics_sae, train_encoder_sae, train_and_profile_all)
- Built FEN enrichment pipeline (enrich_fens.py): Stockfish engine pool (8 parallel) + python-chess tactical annotations (forks, pins, back rank, overloaded, skewers, discovered attacks, promotion threats, eval delta)
- Enrichment cache: 17,923 unique FENs enriched, cached to disk
- Detection scoring 3-way comparison: Haiku+raw (0.571), Sonnet+raw (0.577), Haiku+enriched (0.619). Enrichment wins, judge doesn't matter.
- Updated batch_label_and_score.py: prefill forces array output (fixed 92% parse failure), enrichment wired to both labeling and scoring, persistent output to research/output/
- Polysemantic validation: 19/20 features flagged by diversity metric are actually monosemantic. The metric is invalid — measures generality not polysemanticity.
- 10-feature comparison: Sonnet+thinking > two-pass > Haiku 1-pass for label specificity ("Royal fork" vs "Winning captures")
- Confirmed Bedrock Batch supports thinking (needs ≥100 records)
- Submitted Sonnet+thinking labeling on all 1,961 features with polysemantic audit (job pztzjp2jzh8v)
- Production SAE Lambda updated: filters by coaching_useful flag, threshold raised to BA ≥ 0.6, fire_rate ≤ 3.0. 218 → will update after new labels.
- Built cache_activations.py and train_from_cache.py for fast SAE iteration (skip re-encoding)
- All baseline data saved to research/output/k64_baseline/ with README

## 2026-04-11 — Detection Scoring + K-Sweep + "Sparse but Wrong"

- Built T3b detection scoring framework (`research/scripts/detection_scoring.py`), adapted from Sandstone's evaluation methodology
- Scored all 395 production labels: mean BA=0.650, 35% FAILED (near-random)
- Fire rate vs BA: Spearman r=-0.166 — common features are polysemantic and unlabelable
- Tried contrastive relabeling on 169 WEAK features: +0.005 BA — negligible. Problem is features not labels.
- Found "Sparse but Wrong" paper (Chanin & Garriga-Alonso 2025): c_dec proxy for optimal L0
- Ran k-sweep (8,16,32,64,128,256) on chess-poc: c_dec monotonically decreasing, k=32 confirmed too sparse (57% dead features)
- Training 4096×{128,256} SAEs on chess-poc — in progress
- Will profile all 4 variants, label via Bedrock Batch, detection score, pick champion
- detection_accuracy baked into production labels.json for frontend filtering
- Key insight: the "k=32 is ideal" belief was wrong. Higher k = more alive features = more specific = more labelable
- Trained 4096×{128,256} on chess-poc, profiled all 4 variants
- Created BedrockBatchInferenceRole IAM role for Bedrock Batch
- Submitted labeling batch (10,255 features, Haiku) — completed successfully
- Submitted detection scoring batch — completed but 90% parse failures (Haiku doesn't follow "return ONLY list" instructions)
- Serial scoring on btk_2048_k128: BA=0.600 (n=13, below production 0.650)
- **Result: new variants did NOT beat production on T3b.** But comparison is confounded — production used Sonnet labeling, new used Haiku. Need Sonnet labeling on new SAEs for fair comparison.
- Also confounded by 50K vs 150K training data.
- Next: relabel k=128/256 variants with Sonnet (same quality as production), retrain at 200K, re-score

---

## 2026-04-09 — SAE Feature Relabeling Sprint

**Problem:** Practice page showed "B+K checkmate in 154 games" — completely wrong. Top FENs were generic middlegame positions. Labels were hallucinated from 5 examples.

**Per-feature normalization:** Each feature's strength / its own historical max, threshold 0.2. Replaces per-moment normalization. `featureMaxStrengths.json` generated from DDB scan (128 features, max range 0.5-25.9).

**Relabeling pipeline:**
1. Downloaded canonical profiles from SAIS chess-poc: 395 features × 20 FEN examples
2. Sonnet 4 pass 1: all 395 features → 394 high confidence but 181 got "Creating Multiple Simultaneous Threats" (lazy)
3. Sonnet 4 pass 2: differentiation prompt on 230+ duplicates — "what makes THIS feature different?"

**Practice page fixes:** Hero → top 10 worst features (2×5 grid), removed blue accent, per-feature norm, small category merging, redistributed SAE colors.

---

## 2026-04-08 — Corpus baselines + Practice page

**Pipeline built and run:**
- Sampled 5,000 Lichess rapid games (1K per rating band: 1400-1600, 1600-1800, 1800-2000, 2000-2200, 2200+) from June 2016 monthly dump
- Stockfish depth 14 analysis on ml.c5.9xlarge (36 vCPU, ~2h, zero failures)
- SAE feature extraction on ml.g5.2xlarge (A10G, ~15 min with per-token)
- Three SAE runs: (1) Maia SAE mean-pooled (wrong SAE), (2) Puzzle SAE mean-pooled (wrong pooling), (3) Puzzle SAE per-token (still wrong — used all 77 tokens, not move token)
- Discovered production Lambda uses **move token (hidden[77])**, not mean-pool or per-token-all

**Critical findings:**
- Production pipeline: FEN + UCI move → 79 tokens → encoder → hidden[77] (move token) → SAE → top 5 features
- Corpus script was using mean-pooled 77 tokens → completely different activation distribution → non-comparable features
- DeepMind 270M checkpoint loading: tensorstore pip wheel doesn't include OCDBT driver. Fix: `zarr` driver over `ocdbt` kvstore, save as npz. Should have cached npz the first time it ever loaded.
- Per-token fires ~127 features per position (vs ~5 for move token). Categories saturate at 100% of games when using per-token.

**Practice page shipped:**
- Merged /patterns + /drill into single /practice page with "Drill" nav tab
- Hero card + 2x2 category grid + sidebar (accuracy/rating trends, radar, drill scores)
- Rating band toggle: My Games | 1400 | 1600 | 1800 | 2000 | 2200+
- PF-ICF ranking: categories sorted by multiplier (your % / baseline %)
- Library cards redesigned: pokemon card ratio, SAE feature moments, cburnett pieces

**What's still wrong:**
- Corpus features don't match user features (per-token-all vs move-token). Need re-run with FEN+move→hidden[77] to match Lambda.
- 6 categories missing from baseline (captures, deflection, opening, piece_activity, quiet_moves, zwischenzug)

## 2026-04-04 (Bridge experiments + SAE deep dive)

- DeepMind 270M loaded on GPU (360 pos/s). Discovered: model is BIDIRECTIONAL (use_causal_mask=False), our reimplementation was wrong (causal mask bug). All prior SAE/probe results used corrupted activations.
- Model expects 79 tokens (FEN + move + return_bucket), not 77. It's an action-value evaluator, not a move predictor.
- SAE experiments: trained on puzzles + general positions. Key finding — SAE is wrong tool for tactics, raw activations beat SAE on every theme.
- Bridge experiments v2-v4: move prediction plateaus at ~1.3 regardless of encoder config (causal/bidirectional, 1/77/79 tokens). Bottleneck is task difficulty.
- Win probability 3-class: loss 0.59 at step 360, below random 1.10. **Bridge works for evaluation signal.** Phased curriculum validated.
- MATE dataset discovered: 592K positions with English strategy/tactics annotations. Training data for phases 3-5.
- MATE model discovered: LLaMA-3.1-8B fine-tuned on MATE, MIT license. But it's a reader, not writer.
- CCC paper (NAACL 2025): 0.60 correctness with concept extraction → GPT-4. Our Path B baseline to beat.
- Hybrid architecture proposed: bridge + diff vector concept probes. Probes as guardrails for LLM.
- Lab infrastructure: scientist/engineer modes, file-watch hooks, agent definitions, loop-based monitoring.
- Compressed plan.md from 202 → 64 lines. Archived 8 stale docs.

## 2026-04-01 (SAE sweep + rating gradient session)

- Maia SAE 2048/k=32/200K/50ep completed. Coherence: 0.4% @1.2 (useless). Structural metrics healthy. → DECISION: coherence metric broken for high-dim hidden states, use concept-correlation instead.
- Concept-correlation labeling: 799/2048 features labeled with 54 chess concepts. 27% with |corr| ≥ 0.1. Top: fianchetto (0.48), queenside castling (0.44), open files (0.42). → see findings.md § Concept Labels
- Rating gradient analysis: 5K positions × 4 ratings (1100/1400/1700/1900). 295 features increase, 387 decrease with rating. Higher-rated Maia activates more features (698→748 at >1% fire rate).
- Feature investigation: top unlabeled gradient features reveal perception shift. 1100 Maia → uncertainty/crisis/material fixation. 1900 Maia → coordination/targeted danger/piece activity. → see findings.md § Rating Gradient
- Debated puzzles vs general positions for SAE training. DECISION: general positions (matches Maia's training distribution, covers all position types). Puzzles as diagnostic.
- k-sweep complete: k=16/32/64/128, all zero dead. k=128 best for concept interpretability (60% labeled vs 38% at k=32). Contradicts Sandstone k=32 finding.
- Encoder SAE v2 trained (dict=2048, k=32, 27K positions). Still zero concept correlations → BUT test likely invalid (FEN alignment broken, 27K/200K extraction with no FEN file saved).
- Steelmanned encoder finding: position alignment, mean-pooling spatial destruction, limited concept list. Downgraded from CONFIRMED to INVALIDATED. → see findings.md § Encoder SAE
- k=128 full concept labels: 1278/2048 features labeled (62.4%). piece_count dominates (382 features). Strong king safety, pawn structure, piece activity coverage.
- k=128 rating gradient: 940 increasing + 789 decreasing = 84% of features show rating gradient (vs 33% at k=32). Features are more specialized.
- DeepMind searchless_chess 270M model downloaded (JAX/Orbax format). adamkarvonen model was wrong (512-dim, not 1024). Encoder re-test blocked on JAX setup.
- Final coaching feature analysis: 60 features with both concept labels AND rating gradients. Clear pattern: 1100→1900 = counting→understanding.
- All results saved locally to research/sae/: k128 checkpoint, concept labels, gradients, sweep results.
- DECISION: k=128 on 20K positions is the production SAE. Better than k=32 on 200K.
- Puzzle diagnostic complete: SAE correctly differentiates tactical (check, forced) from positional (opening, pawn shield). Validates general-position training decision. → see findings.md § Puzzle vs General
- JAX + orbax installed. DeepMind 270M checkpoint downloaded but loading needs searchless_chess codebase. Deprioritized — Maia results are strong enough.
- All results saved locally to research/sae/.
- REVISED: k=32 is best for coaching (not k=128). Sam's insight: tags should fire 1-5% of positions. k=32 = 1.6% (good), k=128 = 6.2% (too broad).
- Rating Progression Guide written → docs/2026-04-02 Rating Progression Guide (SAE-derived).md
- Key coaching insight: 1400→1900 = stop over-focusing on center pawns, start evaluating king safety through pawn shields, spot passed pawns early, understand rook activity beyond open files.
- Autonomous pipeline queued: 200K extraction → opening concepts → FEN extraction → k=128 200K training.
- 552 production labels created (38 unique concepts) → research/sae/maia_2048_k32_concept_labels.json
- Sonnet interpretation: 551 features labeled via Bedrock ($1-2, 174s). 540 unique labels. Rating-aware framing: "Learn..." for increasing, "Beginners..." for decreasing. → research/sae/maia_2048_k32_final_labels.json
- Neutral labels created: 197 rating-aware labels ("Beginners...") rewritten to neutral position descriptions ("Kingside castling with standard pawn shield"). Rating-aware preserved in `rating_aware_labels` field.
- position_features.py updated: new SAE path (2048 k=32), new labels, removed threshold (all 32 topk features are meaningful).
- Tests updated for new paths. MCP checkpoint updated.
- All code changes ready for commit.
- Fargate worker wired: Dockerfile adds torch+maia2, copies SAE checkpoint+labels, server_worker.py computes `sae_features` per moment. Graceful degradation if torch unavailable. ~10 lines changed.
- Tag→SAE correlation: 27 tags tested on 10K moments. `undeveloped_pieces` r=0.30 (strongest). F1438 is a "missed tactic" multi-tag detector (quiet_when_winning + missed_check + missed_overloaded). → see findings.md § Tag → SAE
- Top FENs extracted: 2041/2048 features with ≥3 top-activating positions.
- Key unlabeled features (F1438/F1555/F1281/F886) interpreted with Sonnet + actual FENs. All now labeled.
- Opening concept correlation: only 2/10K position matches (dataset is mid/endgame, not openings). Not useful.
- 200K activations cached at 4 ratings. Notebook idle, all work complete.
- Final label count: 554→561 features labeled after full tag correlation (64 tags including computed positional + opening detection).
- KEY: SAE detects specific openings — Sicilian (r=0.38), French (r=0.38), Kings Pawn (r=0.43). Also bad_bishop (0.18), passive_rook (0.34), weakened_pawn_shield (0.33). undeveloped_pieces jumped to r=0.65 with more examples.
- Mechanical labeling: extended concept list to ~100 concepts (endgame types, pawn structures, piece coordination, strategic themes). 1885/2041 features labeled (92%). Remaining 156 are compositional/unique patterns.
- Haiku-refined all 2041 features using top 5 FENs + concept guidance ($1 total, 370s). 2008+ labeled, 1818 unique. 100% coverage.
- Final labels: specific chess concepts — "French Defense with closed center", "Rook endgame with outside passed pawn", "Tactical melee with multiple forcing moves".
- MCP engine updated: `get_sae_features` now returns labeled tuples (fid, strength, label). Labels loaded from production file.
- Two new MCP tools: `check_game_stockfish` (pure eval) and `check_game_full` (eval + SAE labels). Both accept SAN moves directly.
- Standalone game analyzer: `research/sae/analyze_game_sae.py` — tested on Ruy Lopez and Sicilian games.
- Analyzed Sam's game (chess.com/game/live/166695486394): 17...Qf6 was the critical miss (41% of 1900s find Rxc2). 21...Kh8 understandable (89% play it). Post-move-22 mistakes are engine-only.
- Maia rating comparison integrated into `check_game_full` MCP tool — for each mistake, shows "would a 1100/1400/1700/1900 find the right move?"
- `check_game_full` returns: eval, best move, SAE features, Maia rating comparison, coaching verdict per mistake.
- 8 categories + 695 chip labels (2-3 words) for UI display ($0.30 Haiku, 218s). Endgame 26%, Tactics 19%, Piece Play 12%, Opening 11%, Rooks 11%, Strategy 9%, King Safety 8%, Pawn Structure 4%.
- position_features.py returns chip + category per feature. Data structure: `{label, chip, category, strength, feature_id}`.
- Frontend wired: `sae_features` added to Moment type, passed through both `buildMomentPayload` and both `streamCoachOverview` call sites. Backward-compatible (empty array for pre-SAE games).
- Full end-to-end pipeline: Fargate→DynamoDB→Frontend→Backend prompts→LLM. All code ready.

## 2026-04-01 (research session — lab manager)

- Deep research on chess LLM landscape: 50+ papers, 5 competing implementations. Wrote comprehensive fusion research doc.
- Steelmanned against own thesis (CLIP gap is insurmountable) — evidence from multimodal literature says it's bridgeable. Changed position from 20/80 against to 60/40 for.
- Ablation results came in: encoder IS the signal (constant output without, varying with). 54.5% vs 50% = ~4.5% delta. Architecture validated, data is bottleneck.
- Key discovery: FEN tokenizer expands to 64 individual board square tokens. The mean-pooled contrastive training DESTROYED this spatial structure. The translator was only trained on eval direction (one scalar) via mean pooling.
- Wrote board reading test (`research/encoder/scripts/test_board_reading.py`). This is the blocking experiment — determines whether translator carries spatial info or just a faint eval residual.
- Brainstormed 6 data paths + curriculum approach. Sam proposed game-level eval ("find 6 critical moments") as the real benchmark for chess understanding.
- SAE vs translator debate: SAE decomposes encoder knowledge into interpretable features for any LLM. Translator preserves holism but LLM can barely read it. Both have merit — board reading test determines which path.
- Reorganized lab structure: merged chess-research lab, consolidated docs, renamed findings→learnings, updated lab skill, created workspace/lab/ symlinks.
- Docs written: What We're Building, What We Don't Know, Translator Signal Analysis, Data Paths Brainstorm, Board Reading Test spec.

## 2026-04-01 (building session)

- LoRA v1 training completed (25K steps, 434 min, loss 0.018). Output: adapter_model.safetensors 617MB + proj_norm.pt
- Eval: 100% degenerate output (`!!!...`). Every position, same output. → see learnings.md § F10
- Debug: logits all NaN, argmax(NaN)→token 0 (`!`). Text-only (no chess) also broken — LoRA weights corrupted.
- Root cause: LayerNorm gives per-token norm sqrt(3584)=59.8, Qwen expects 0.94. 60x mismatch. nan_to_num masked the NaN losses.
- v2 training launched (10K, scale_factor=0.0157). Step 0: chess_norm=0.94, text_norm=0.96. Loss: 2.97→1.39 at step 400. NaN: 3/450.
- Loss plateaued ~1.47-1.58 from step 400-950. Expected with 10K examples (overparameterized).
- Caught critical bug: eval_lora.py used `scale_factor` in generate() but it was local to main(). Would have crashed auto-eval. Fixed.
- Auto-eval watcher staged in screen. Background poll running locally.
- All docs updated: design doc, projection signal analysis, research README, STATUS.md, plan.md.
- LoRA v2 training complete: 5000 steps, 86.5 min, final loss 1.2035, 36 NaN (0.7%).
- Loss trajectory: 2.97→1.39(400)→1.55(1000, plateau)→1.30(3600)→1.20(5000). Cosine LR recovered from plateau.
- Eval running (200 positions). Partial results at 80 positions: eval_direction=61%, format=100%, move_legal=6%, best_move=0%.
- Model outputs structured text ("Eval: +1.1. Best: d7d6") but defaults to common moves (d7d6, d2d4). Eval direction above random.
- Ablation partial results (140/200): eval_direction oscillating 45-54%, centering at ~50%. Normal was 54.5%.
- Ablation output is CONSTANT across all positions — same text every time. Encoder IS the only position signal.
- Delta: ~4.5%. Real signal, but p≈0.10 and 0% move accuracy = not useful for coaching.
- DECISION: Park encoder projection. Pivot to Maia (L2). Notebook stopped. → docs/2026-04-01 Encoder Experiment Retrospective.md
- Read Sam's brainstorm doc → revised decision: architecture works, DATA is the bottleneck. Multi-task (Path D) with existing data.
- Extracted 21,341 multi-task moments from DynamoDB (4,073 games). 27 tags, 6 fields per moment. → research/data/multitask_moments.jsonl
- New plan: Maia (product) + multi-task encoder training (research) in parallel.
- Board reading test: 0/100. Model outputs training format regardless of prompt. Can't read pieces.
- CRITICAL: Eval direction bias check (E15). Training data is 55.3% positive. Always-positive gets 55.3%. Our model gets 54.5% — BELOW trivial baseline. The "4.5% delta vs ablation" was comparing against a parser artifact (ablation had no eval field → default 0 → ~50%).
- Revised understanding: encoder transmits position-varying signal (proven) but that signal doesn't beat trivial baselines (also proven). The architecture works for transmission, fails for useful prediction.
- Wrote 6 analysis docs capturing the full reasoning chain.
- Two cheap probes queued: per-token projected embeddings (spatial info?) and LLM hidden states (where does signal die?).
- Notebook still running ($1.40/hr). Should stop if probes aren't run today.
- Per-token probe results (raw encoder): occupied 68.6% (baseline 64.2%), piece type 65.1% (baseline 64.2%), color 66.0%. Piece type is barely above baseline — encoder doesn't have per-square info after attention layers.
- REFUTED: per-token alignment path. Encoder computes board-level features, not square-level. The transformer attention distributed spatial info across all tokens.
- Consolidated docs: 20 → 10. Archived 10 to archive/docs-2026-04-01/.
- Projected representation: occupied 66.6% (vs raw 68.6%). Piece type still computing when stopped.
- Partial projected result: 66.6% ≈ 68.6% raw = projection preserved what little per-token info exists.
- DECISION: Per-token alignment eliminated. Mean-pooled with richer signals + SAE are the remaining encoder paths.
- Notebook stopped. Next encoder experiment: multi-task training on 21K moments (data already extracted).
- Pivoted to SAE path. Ran 5K positions through existing Maia SAE. 1020/1024 features active.
- Interpreted 20 more features (total 46/1024). Features are meaningful: "king under attack," "centralized knight," "undefended pieces."
- Proof of concept: SAE features on 4 test positions match human chess understanding. Complex middlegame → knight/king attack features. Fork position → "converting advantage requires precision."
- Wrote SAE vs Projection comparison doc, SAE plan, research/sae/ directory with README + labels JSON.
- SAE sweep: 512/1024/2048/4096 on 20K Maia activations. 4096 wins: 84.8% explained variance, zero dead features, orthogonal decoder columns. All trained in <20s on CPU.
- 4096 SAE differentiates position types (endgame/opening/middlegame get different top features). Needs interpretation (4096 new feature IDs, no labels yet).
- Structural metrics all healthy: no dead features, near-zero decoder cosine sim, constant 6.25% fire rate across all sizes.
- Encoder SAE trained on notebook: 4096 (MSE 0.053) and 8192. Float32 fix required (half precision NaN on some positions).
- Both Maia SAE (local) and encoder SAE (SAIS) checkpoints downloaded locally to research/sae/.
- Notebook stopped. Total SAE work: 5 Maia sizes + 2 encoder sizes trained, all zero dead features.
- KEY RESULT: Feature overlap test on Maia 4096 SAE. K+P endgame vs tactical = 0/50 overlap. Endgame vs exposed king = 0/50. Features are highly position-specific and monosemantic.
- Profiled top 30 features across 3K positions. Clean phase split: opening features (eval ~0) vs late middlegame features (eval ±1). 4047/4096 features active.
- 35 features labeled for Maia 4096 SAE. Integration PoC: SAE correctly identifies "opening mistake — fundamental principles violated" for a position where player took pawn instead of recapturing knight.
- DeepMind encoder SAE: 4096 (MSE 0.053) and 8192 trained on notebook, checkpoints downloaded.
- Pipeline proven: Maia → hidden state → SAE → feature labels → coaching prompt enhancement. No LLM training needed.
- 500 features interpreted by Claude Haiku ($2, 700s). 233 unique labels. Top concepts: rook on 7th rank, king in center, centralized king in endgame.
- Maia SAE heavily focused on king safety (~100 features detect king position variants). Makes sense for 1800 Elo human move prediction.
- Started wiring SAE into Fargate worker Dockerfile, then reverted — premature.
- A/B TEST DONE: 2 positions, coaching with vs without SAE features. SAE improves framing — connects moves to position type principles. Rook endgame: "rook on the seventh is fundamental" (with SAE) vs generic "prioritize rook activity" (without).
- Prompt changes shipped: `build_moment_prompt` and `build_overview_prompt` now accept `sae_features` field. 3 lines each, backward-compatible.
- Euclidean distance in SAE space: 1.01-1.03x separation (useless — sparse vectors make everything equidistant). Jaccard on active feature sets: endgame↔opening = 2.43x separation. Adjacent phases: ~1.1x. Saved Jaccard insight as cross-project memory.
- Built `backend/shared/position_features.py` — clean module: `get_position_feature_labels(fen)` → ["Rook on seventh rank", ...]. Importable by worker, Lambda, or MCP. Tested, works.
- 100% label coverage on test positions. Rook endgame → "Rook on seventh rank". Opening → "Undeveloped pieces, king not castled".
- Tag↔SAE correlation: `undeveloped_pieces` tag → SAE "Undeveloped pieces, blocked" (10% co-occurrence). Tags and SAE capture different levels — tags say WHAT went wrong, SAE says WHAT KIND of position. Complementary.
- Verified prompts.py diff: Feigned Discovery rewrite + SAE feature support. 89 lines, clean. Not committed. Ready to commit+deploy next session.
- Process feedback written to `.lab/meta/`. MEMORY.md updated.
- Encoder SAE features interpreted: extracted 5K position FENs on GPU, downloaded locally, interpreted via Bedrock ($2). 500 features labeled. Heavy on king safety + "requires immediate defense" — GM-level urgency concepts.
- **1000 total features labeled** across both SAEs for $4 total. Maia (human modeling) + Encoder (deep analysis) = complementary coaching context.
- Unique concept analysis: encoder has 172 labels with no Maia match (urgency concepts: "requires immediate defense", "vulnerability", "coordination imbalance"). Maia has 160 labels with no encoder match (spatial: "rook on fifth rank", "pieces on starting squares", "king not castled"). Encoder is EVALUATIVE, Maia is DESCRIPTIVE.
- Gitignore updated: all .pt checkpoints, maia2_models/, large JSONL files excluded. Only text files would be committed.
- research/sae/README.md written: full documentation of files, results, integration, regeneration.
- KEY: Maia SAE vs Encoder SAE decoder comparison — max cosine similarity 0.113. ZERO features overlap. 4096/4096 unique in each. The two models encode completely different information. Fully complementary — using both gives 8192 unique features.

## 2026-03-31

- (Prior session) Contrastive Phase 0 trained (200K, 3 epochs). Preserved 92% of encoder signal. → learnings.md § F3b
- LoRA v1 launched on 50K mixed data, batch 2, 1.0 step/s
- FSDP crash root cause found: LayerNorm dtype mismatch → learnings.md § F9

## 2026-04-03 (SAE v3 Tag Analysis)

- Pulled 502 cabbagelover5566 games to local JSON (12.4 MB → 26.6 MB with SAE features)
- Computed SAE features (all 32 per moment) on 3,012 moments, 45 seconds on CPU, 0 errors
- Raw feature analysis: top features are too generic ("Bishop pair" 143%, "Knight outpost" 94%). Useless as tags.
- Blunder enrichment analysis: "Undefended piece" 11x enriched at blunders, "Knight fork" 5x, "Unsupported pawn" 4.6x. These are real mistake signals.
- PF-ICF scoring: ICF range 0.27–8.01. Generic features (Bishop pair ICF=0.27) properly downweighted. Rare features (Rook endgame maneuvering ICF=8.01) amplified.
- Combined PF-ICF × enrichment scoring produces actionable per-game narratives:
  - Game 1 (Caro-Kann, 13 mistakes): "Underdeveloped position" in 6/13 mistakes (4.5x enriched). Development → pawn problems → endgame collapse.
  - Game 2 (Sicilian, close loss): "Back Rank Weakness" in 4/6 mistakes. Collapsed after good opening.
  - Game 3 (messy win): "Rapid development" in 7/10 mistakes (14.8x enriched). Pushed pawns instead of developing.
- Key finding: enrichment ratio with min-count threshold (n≥10) eliminates small-sample noise (was getting 99x artifacts)
- Verdict: SAE features ARE the tags when properly filtered. PF-ICF × enrichment is the filter.
- Best-move position comparison attempted but needs init_models() call — deferred.
- Built `player_profile.py` — full cross-game analysis: weaknesses, strengths, opening-specific, phase-specific, time pressure, best-move comparison
- Player profile results (cabbagelover5566):
  - #1 weakness: "Unclear coordination" (47.6x enriched, 122 games) — 16.5% of mistakes vs 0.3% of good moves
  - #2: "Underdeveloped pieces" (20.5x, 88 games)
  - #3: "Piece sacrifice" (23.9x, 95 games) — positions demanding sacrifices overwhelm this player
  - #4: "Rapid development" (14.8x, 54 games) — behind in development
  - Strengths: King opposition (6.8x at good moves), Centralized kings, Rook on seventh — classic endgame competence
  - Opening patterns: Caro-Kann → coordination problems, Vienna → development speed, Sicilian → underdevelopment
  - Phase: Opening = underdevelopment, Middlegame = coordination + sacrifice themes, Endgame = rook play
- Best-move comparison: top 50 worst blunders compared current vs best-move position. "Knight outpost" (40%), "Rook endgame" (34%) appear after best move but not current — these are missed improvements.
- DECISION: PF-ICF × enrichment is the v3 tag system. Enrichment ratios are validated (raw counts confirm 47.6x for top weakness). Ready to wire into frontend.
- Scripts: `lab/chess/scripts/analyze_sae_features.py`, `lab/chess/scripts/game_deep_dive.py`, `lab/chess/scripts/player_profile.py`
- Analysis doc: `lab/chess/docs/2026-04-03 SAE Feature Analysis.md`

## 2026-04-03 (Design Language Implementation Sprint)

**CSS-only design language pass across all pages. No structural/component changes.**

### Global changes
- **Fonts:** Added Lora to Google Fonts import in index.html. Added `--font-display: 'Lora'` to theme.css. Changed `--font-sans` from Inter to DM Sans in both theme.css and theme.ts.
- **Nav:** Active tab changed from bottom-border underline to subtle pill highlight (background tint). Brand text now uses Lora serif.

### Per-page changes
- **Patterns (weaknesses.css):** Tag rows: removed card backgrounds, use border-bottom. Severity bars: 3px→2px. Section accents: 3px→2px. Chips: card-like→simple pills. Cost bars: 4px→3px. Positive card: thinner accent. All JetBrains Mono→tabular-nums.
- **Drill (drill.css):** Mode card active state: removed double-ring box-shadow. Category chips: tighter padding, smaller font. Active chip uses accent-text color.
- **Library (library.css):** Grade uses Lora 18px/600. Removed 3px focus box-shadow rings on search/select inputs.
- **Landing (landing.css):** Headline uses Lora 400 weight instead of DM Sans 700. Removed gradient text clip on highlight (now just accent color). Removed focus ring on import bar.
- **Players (players.css):** Search title uses Lora. Removed focus ring.
- **Review/Board (board.css, nav.css):** No structural changes — board CSS was already clean. Nav changes affect Review's appearance globally.

### Code audit findings
- 11 `any` types across 5 files (existing code, not new). Not changed during design pass.
- No TypeScript build available in this environment — changes are CSS-only and low-risk.

### What's NOT done (needs Sam's environment)
- Build verification (npm start / npm run build)
- Playwright testing (node not available)
- Players page: existing page got CSS pass but doesn't yet show patterns+openings combo view for showcase players

---

## 2026-05-29 — Maia 3 taxonomy rebuild (chip-first → title-first)

**Trigger:** the "other claude" session was hitting Bedrock 503s mid-labeling. Diagnosed: 503s were the model backend, not the pipeline. The real work was the Maia3 2048 k=32 v2 coaching taxonomy.

**Diagnosis (long, several wrong turns):**
- Old 13-category taxonomy had junk drawers (Missed Tactics = 372 features, every piece type).
- Decoder geometry: v2 features near-orthogonal (mean cosine 0.000) → genuinely distinct, not duplicates.
- Found the labels' `description` field is accurate (verified against the board) but the 2-4 word `chip` was generic on ~400 features. Categories were built from the lossy chips → junk drawers.
- **Root cause: chip-first pipeline.** Compression (chip) happened before categorization, so lossiness propagated down.
- **Detour that cost time:** the canonical 19K Opus Pass-1 English analyses are complete only on chess-poc (`all_positions_labeled_opus.json`, 19,342). The S3 `..._final.json` is TRUNCATED to 10,648. I burned time computing on the wrong cache + the truncated file before reading S3_INVENTORY.md (which Sam rightly pointed at). Lesson logged in knowledge.md + inventory.

**Fix — rebuilt TITLE→CATEGORIZE→CHIP:**
- Deterministic per-feature structural fingerprint + description verification (`scripts/sae/taxonomy/`, TDD, 7 tests).
- Reused the validated 20-category vocab from `chess_blunder_taxonomy_v2` (coaching vocab is checkpoint-independent).
- Assigned 1,996 features + wrote specific category-aware chips via Sonnet 4.6 on the research account (NOT the Claude Code backend — sidesteps the 503s; resumable, throttle-tolerant, 0 errors on full run).
- Targeted chip regen for generic/vague ones.

**Result:** generic chips 398→0, no junk drawer (largest 20%), 20/20 categories used, 0 unassigned. Every category's structural signature matches its definition (greedy_captures 88% cap, checks_lose_tempo 73% chk, king_walks 98% king, slow_play 88% quiet, pawn cats 99-100% pawn). QA gate PASSED. Ship artifact: `output/taxonomy_v2/taxonomy_v2.json` + `REBUILD_REPORT.md`.

**Known limit:** 4 low-pop categories (hangs/undefended/back_rank/fork, 4-7 each) — features route to sharper abandons_defense/lands_badly; kept as checkpoint-stable vocab, not mis-routing. ~2% of chips name no specific square (genuine quiet "slow play" features).

---

## 2026-05-30 — taxonomy provenance bug found; redo planned on flat k=32

**Context:** Sam asked to (a) put the taxonomy in an HTML atlas matching the persona-atlas style, (b) add fire rate per feature summed per category, (c) add semantic sub-clusters inside each category — none of which the 2026-05-29 `taxonomy_v2` had.

**Built:** `scripts/sae/taxonomy/build_atlas.py` → `chess_taxonomy_atlas.html` (persona-atlas style: warm paper palette, Fraunces + IBM Plex, sidebar + expandable card grid, category→feature). Sam confirmed the *look* is right.

**Two problems surfaced (both make the existing taxonomy suspect):**
1. **Categorization was top-down (wrong).** I'd assigned each feature independently into 20 pre-baked categories → magnet effect (Slow Play Punished 408, Pieces Left Undefended 4). Sam: "that doesn't look like good categories." Correct method is the persona approach: bge-m3 semantic clustering FIRST, categories emerge bottom-up, one agent regroups within each. Saved as `docs/knowledge/taxonomy-method-persona.md`.
2. **PROVENANCE BUG.** Tried to add fire rates → needed the checkpoint the 2007 labels came from. Verified the v2 cache is correct (`cache_v2[137471]` == profile feat3 ex0 Bxf7+). But NO checkpoint reproduces the profile: flat k=32 (l2_200ep/v2/base) AND H1 perlevel matryoshka all give feature 3 top-firings ≠ Bxf7+ (0/10). The label scripts (`label_v2_features.py`, `extract_v2_labels.py`) use a matryoshka forward pass (prefixes + k_per_level), but even the matryoshka checkpoints I have don't match. **Source model unknown → taxonomy_v2 per-feature categories are unverified.**

**Decision:** Sam wants the FLAT k=32 model (`maia3_sae_diff_2048_k32_l2_200ep.pt`). Don't reverse-engineer the old profile — regenerate from scratch on flat k=32: fresh profile + fire rates over v2 cache, join 19K Opus English, bge-m3 bottom-up cluster → emergent categories → sub-clusters → atlas. One known model end-to-end = reproducible. Full plan in plan.md "Current State (2026-05-30)".

**Process lesson (Sam flagged I was disorganized):** I thrashed across 5 checkpoints × 3 normalizations chasing the provenance without writing down what was verified vs assumed. Should have pinned "what generated this artifact" by reading the generator script FIRST (it names cache + norm + model-type), not by trial-and-error forward passes. Ran /organize-research to capture ground truth before continuing.

## 2026-05-30 — Maia3 space investigation

**Goal:** Understand what the Maia3 layer-7 representation encodes and whether it can produce "missed fork" SAE features.

**What we found:**
- Current v2 SAE: diff is  (NOT blunder-best). Best move never encoded. Explains why all features are "what you played" not "what you missed."
- Option A (repr_best - repr_blunder): tested on v1. cp_loss probe r≈0.07 (weak, expected). Tactical clustering real: capture gap=0.12, fork gap=0.04. An SAE on this WILL produce missed-fork features.
- Value head encodes eval well (r=0.94 for win-prob), but only 3D — too small for SAE.
- Mean pooling over 64 squares is correct for the diff.
- No dominant single-token eval position.

**Decision:** Build Option-A cache using v1 data (has best_uci + player elos), retrain SAE. This gives tactically-organized features.

**Scripts written on notebook:** layer7_probe.py, token_pos_probe.py, value_head_probe.py, pooling_test2.py, tests_clean.py

## 2026-05-31 — SAE architecture experiments + overnight labeling

**Key findings from model probing:**
- Layer 2 better for positional mistakes (hanging piece, king safety, passed pawn): gap 0.038-0.049
- Layer 7 better for tactical timing (fork, tempo_loss): gap 0.047-0.055
- board_diff (mean64(after_best - after_blunder)) better than single-square for forks
- 79M model NOT better than existing ONNX for SAE purposes — per-sq cosine 0.92-0.998
- Policy projections (proj_sq_to) don't add fork signal over raw h[sq]
- Mistake-type separation overall: L2=0.038, L7=0.028 mean gap across 6 taxonomy categories

**Three SAEs trained/building:**
- option_a (done): h[best_to] - h[blunder_to], layer 7 ONNX, 512-dim
- board_diff (done): mean64(h_after_best - h_after_blunder), layer 7 ONNX, 512-dim
- l2l7_concat (building, ~3hr): concat(L2_mean64_diff, L7_mean64_diff), 79M, 2048-dim

**Overnight pipeline running on chess-poc:**
- Labeling all 3 SAEs with Sonnet overnight
- Handles unlabeled positions (v1 data doesn't overlap opus) via fresh Sonnet labeling pass
- Outputs: option_a_profiles.json, board_diff_profiles.json, l2l7_profiles.json + labels

## 2026-05-31 (later) — CORRECTION: option_a / board_diff / l2l7 invalidated (v1 data)

The three new-architecture SAEs above (option_a, board_diff, l2l7) were all trained on the
**v1 blunder cache** (`maia3_blunder_diff.pt`), which has the documented Black-to-move
label-inversion bug. Every result from them — the board_diff "leading candidate" call, the f46
"Recapture leaves piece hanging" eval, the >50%-coherent expectation, the L2/L7 probe gaps —
is therefore unverified and must be re-derived on the v2 cache (`maia3_blunder_diff_v2.pt`)
before being trusted.

Cleanup done this session:
- Deleted from git: `output/new_saes/{option_a,board_diff,l2l7}_labels.json` (tracked).
- Deleted (gitignored/untracked): `output/new_saes/*_profiles.json`; removed empty `output/new_saes/`.
- Deleted (untracked v1-based eval artifacts): `output/{eval10_out.log, eval10_parsed.json,
  eval10_payload.json, full_eval_out.log, ksweep_eval.html, option_a_ksweep_labels.json,
  sae_eval.html, feature_explorer.html, explorer_v2.html}`.
- SAE weights + intermediate caches already removed from S3 and the chess-poc notebook.
- Annotated all 6 build/train scripts in `scripts/sae/new_sae_architecture/` with a v1 warning.
- Corrected `output/S3_INVENTORY.md`, `plan.md`, `knowledge.md`.

The legit shipped v2 SAE (`maia3_sae_diff_v2_2048_k32_l2.pt`) is untouched — still valid.

## 2026-06-02 (overnight auto-run) — blob-concentration experiments

Triggered by Sam's f101 catch: the k=16 SAE concentrates activation in ~32 broad "blob"
features (fire >10% corpus, mislabeled with specific names) that drown out specific
mistake-features. Two experiments to find the cause.

**Exp1 k-sweep (k=4,8,12,16,24,32, fixed 168k):** blob count monotonic in k —
12→89 across k=8→32. k=4 collapses (1800 dead). k=8 is blob-minimizing-while-alive
(12 blobs, spec-feature act 0.30, 0 dead; cost: FVU 0.38, 452 useful features).

**Exp2 corpus-size (k=16, 42k/84k/126k/168k subsamples):** weak effect. blobs 36/41/36/32 —
basically flat across 4× data. Specific-feature act improves only 0.19→0.25.

**Verdict: blobs are a k problem, not a data problem.** k dominates by ~7×. For coaching
specificity, k=8 > k=16. The 1M Lichess build is still worth it for *coverage* (rare mistakes)
but won't fix blob concentration.

**Method note:** added `blob_metric.py` (calibrate threshold, count blobs, spec-feature
activation, FVU) as a reusable per-SAE diagnostic. Also caught + fixed: my test-position
analysis earlier in the session ran at elo=1800 while corpus diffs use real-player elos;
elo turned out to be a near-scalar (feature identity stable across elo), so conclusions held.

**Left open for Sam:** are blobs actually bad, or useful coarse signal? Not auto-decided.

## 2026-06-02 — Normalization + dual-axis coherence: the "it's noise" reversal

Started from Sam's gut "these features seem more trash than before BatchTopK / activations are 0.1 not 0.8."
Ran /systematic-debugging. Chain of corrections, each from a Sam catch:

1. **"BatchTopK regression" was FALSE.** Old "good" SAE is also BatchTopK, identical norm + activation
   dist (max 0.92). The "0.8→0.1" was MY >10% fire-rate display filter hiding the strong (blob) features.
2. **"33% can't be hung queens"** → top-60 characterization is unrepresentative (a 33%-fire feature hits
   55k positions). Measured across activation bands: f101 = real graded concept (75% at peak→10% at base),
   f1487 = flat noise. Don't characterize high-fire features by top-N.
3. **"f712 is coherent"** → it was; I'd over-generalized from one bad feature. Piece-signature (python-chess,
   100% coverage) revealed real structure the weak opus-motif join (30% coverage) missed.
4. **"you're not capturing best move"** → THE big one. Diff = L7[best]−L7[blunder]. My probes only measured
   the BLUNDER move. Dual-axis (real SEE + maia_best): best-move axis has 2.5× more coherent features (458 vs 184).
   The model mostly encodes "what you should've played" (missed-good-move mistakes) — invisible to blunder-only probing.
5. **"it's L2+z-score vs z-score"** → trained z-score-only variants. z-score-only nearly TRIPLES coherent
   features (990 vs 350). L2 projects to unit sphere, erasing magnitude=severity. Dominant lever, bigger than k.
6. **Multi-axis brainstorm (Sam):** tried phase/direction/severity/trajectory/refutation-motif as coherence
   axes. Base-rate correction showed these are LEAKY (features concentrate via corpus base rate, not mistake
   structure). Only piece/hang(SEE)/best-move are trustworthy. Refutation-motif moot (0 features; Maia never
   computes refutations — Sam flagged this correctly).

**Outcome:** chosen model `btk_2048_k16_zscore.pt` (990 coherent, 48%). Used real SEE everywhere (replaced
crude attacker>defender head-count Sam flagged). Meta-lesson: a coherence probe on a feature defined over a
DIFFERENCE must measure both sides — one-sided probing = guaranteed false negatives, cost most of the session.

Also did k-sweep (k4 dead, k8/12/16/24/32) + corpus-size sweep (42-168k, weak) earlier — both in
`blob_experiments_report.md`. Model is UNLABELED (old labels are L2-model, indices differ); relabel with
both-axes prompt is the next task.

---

## 2026-06-02 (cont.) — k-sweep settled at k=6 (3 methods), labeling pipeline built, sparse probing

**k decision, three independent angles all point to k=6** (was tentatively "k=16"):
- Trained full z-score-only sweep k4/6/8/10/12/16/32 + dict=1024 at k4/6/8.
- **Mass band (0.1–10%):** k6 = 61.5%, most among fully-alive models; k4 higher (67%) but 951 dead.
  Blobs grow monotonically with k (4→68). `band_mass_sweep.py`.
- **Raw Gini U-shaped, min at k6** — the only non-monotone structural metric (real interior optimum).
  Threshold-Gini was an artifact (lower threshold at high k counts more tiny acts). `gini_sweep.py`, `fvu_rawgini_sweep.py`.
- **0% decoder twins at every k** (`distinct_vocab_sweep.py`) — high-k extra features are distinct, not dupes.
- **Dict-size:** d1024_k4 recovers k4's 66.7% band-mass with 0 dead but only 452 feats (`dict_size_compare.py`).
  Dead-count = dict_size artifact, not quality. 2048 gives room to explore.

**Labeling pipeline (new skill `label-sae-features`):** enrich_gap.py (Stockfish, 22.5k positions,
cache 55k→77.8k) → label_positions_btk.py (Opus 4.6 per-position motif/tags, +6,149 for k4, total 54,763)
→ fuse_feature_names.py (Opus motif + SEE facts → name). k4: 1075/1097 named, 22 diffuse.
Fork/back-rank/king-safety come from Opus (SEE can't); hang specifics from SEE. Blobs over-claim (flagged).
**Scope note:** briefly over-expanded the Opus gap to 4-model union (23.6k); caught + reverted to k4-only (6k)
before any Opus calls wasted. Enrichment (22.5k) kept — reusable across all models.

**SAE-eval literature** (workflow, 6/10 readers salvaged after 4 stragglers stalled on GitHub fetches):
- Loss-recovered/KL fidelity CANNOT be computed on our diff x=L7[best]−L7[blunder] (no forward pass). 5/6 agree.
- **Sparse probing + feature splitting fit cleanly AND aren't forced monotone** — the metrics that can pick k.
- FVU/explained-variance is the field's sanctioned fidelity fallback when patching unavailable (we did this right).
- Detection-score (auto-interp, held-out) > our circular top-10 audit. Skip the LLM judge, use SEE labels.

**Sparse probing** (`compute_see_labels.py` → 168k SEE concepts cached; `sparse_probe.py`):
- Sped up 10× via vectorized f_classif ranking (vs KNN mutual_info); bal_acc identical, validated `--validate-rank`.
- **Cleanly isolated:** best_check 0.89@1, hang_queen 0.81, hang_major 0.78, hang_rook 0.72 (all via f952/f1372).
- **Smeared (not nameable):** endgame +0.04, severe +0.06, hang_knight +0.06 @1 — confirms leaky descriptors.
- **Splitting signal:** hang_queen@1 drops 0.81→0.70 at k16 (top feature f952→f926); lower k keeps concepts
  in single features. Independent confirmation that k4/k6 > k16 for interpretability.

Artifacts: `output/fused_names_k4_slim.json`, `output/eval/sparse_probe_results.json`;
full versions S3 `sae/labels/fused_names_k4.json`, `sae/eval/sparse_probe_results.json`.
Models `btk_2048_k{4,6,8,10,12,16,32}_nol2.pt` + `btk_1024_k{4,6,8}_nol2.pt` on notebook.

---

## 2026-06-03 — feature labeling method corrected (SEE-on-both-moves), d1024_k4 labeled

**The pivot:** my `fuse_feature_names.py` (aggregate per-position SEE stats over top-10 → vote on a
name) was the WRONG method. It fragments a concept and got DIRECTION backwards: f127 (Sam flagged)
came out "hangs a piece; missed a capture" but is actually MISSED HANGING PIECE — best move wins a
free enemy piece in 91% of its top-500, player played a non-capture. My own-hang metric (player's OWN
piece after the blunder) is the noisy axis; I led with it and buried the real signal. Also: SEE can't
see 2 moves ahead, so a check-that-then-wins (Be3+) read as "no enemy hang" — but Opus eyeballing the
board just sees it.

**Correct method (`label_features_see.py`):** Opus reads each feature's top-N boards HOLISTICALLY +
is handed a SEE-on-both-moves aggregate over **top-500** (not top-10 — stats must be robust) as raw
data. Aggregate = best_wins_material_pct (missed-winning) vs blunder_hangs_own_pct (hung-own) +
piece-class dist + best_is_check/capture. Two-step: `compute_feature_see_stats.py` (top-500, 16-proc,
106k unique positions) → `label_features_see.py` (Opus 4.6, 12 boards + aggregate, conc 20).

**d1024_k4 labeled: 1020/1024**, 629 distinct chips, cleanly disambiguated:
Missed Winning Check 35 · Missed Hanging Piece 26 · Hung Own Piece 25 · Greedy Capture Hangs Piece 20 ·
Missed Hanging Queen 13 · Hung Own Queen/Major 20 · Missed Knight Fork 8. Opus names mechanisms SEE
can't: f198 "Ignored Pawn Attacks Knight", f341 "Greedy Capture Allows Mate". f127 → "Missed Win, Hung
Piece". Validated against the specific features Sam inspected.

**Categorization direction (next task):** natural taxonomy = mechanism (SEE, objective: missed 218 /
hung 108 / greedy-both 213 / other 481) × tactical motif (Opus: fork/pin/back-rank...). Piece+severity
= filters. Bottom-up cluster the chips; do NOT reuse suspect taxonomy_v2 20-buckets.

Artifacts: `output/feature_labels_see_d1024_k4.json`, `output/see_stats_d1024_k4.json` (git + S3
sae/labels/). Method scripts: compute_feature_see_stats.py, label_features_see.py, render_feature_html.py.
Superseded (kept): fuse_feature_names.py, heuristic_name_all.py.

---

## 2026-06-03 — integrated labeling (SEE+trajectory+Opus), 11-bucket taxonomy, audit

**The f91 lesson drove a methodology overhaul.** f91 (queen-takes-queen) was mislabeled "Greedy Queen
Capture" because SEE scores the recapture as -9 "hung own piece" — but it's a TRADE (you took a queen too).
SEE structurally mis-reads trades as material loss and is blind to positional/trajectory mistakes. Fixes:
- **net-material** (capture value − recapture): QxQ = net 0 = trade, not -9 loss. material_kind: trade/loses/hangs/safe.
- **eval trajectory** (winning/drawn/losing, player POV, cp units, ±150 draw zone): what the mistake cost.
- **per-feature normalized cohorts** (≥0.7max, ≥0.8max): features are pure at their activation peak (f91 QxQ
  74%→89%→98% as you tighten), noisy in the tail. Divide-by-max for the THRESHOLD (Sam's "0.8+ is pure"
  model); fixed top-N was diluting. (Tested elbow/p99/activation-weighting — all worse; simple normalized
  threshold won.)
- SEE stats now comprehensive: moved/captured piece, Maia-move piece+captures, played-check, phase,
  net-material, trajectory. 10/11 buckets derivable from stats alone (only Ignored-Tension wasn't → folded).

**Integrated labeler** (`label_features_integrated.py`): Opus reads top-12 boards + all 3 signals → name.
f91 → "Premature Queen Trade" (was "Greedy Queen Capture"). All 1020 relabeled.

**11-bucket taxonomy** (was an earlier 11 flat → refined): bottom-up from mistake_type spine, 5-way consensus
method. NEW buckets the old taxonomy couldn't house: **Premature Trade** (66, the f91 family, was buried in
Greedy) and **King Safety** (88, positional, SEE-blind). Greedy Capture 169→29 (trades+endgame rerouted).
Audited (`audit_buckets.py` objective cross-check + Opus semantic grade): 1.0% flagged, 17 reassignments
applied, all remaining flags verified correct. Ignored Tension (14) folded into Missed Tactic. Sub-buckets
by piece/phase with fire-rate coverage. Tree: `render_taxonomy_tree.py` → atlas html (gitignored).

**Decision:** apply to cabbagelover's 1,209 blunders next (coaching payoff + the only real k4-vs-k6 criterion).
Defer k6 retrain until we see whether k4's leak report is good enough.

Scripts added: compute_feature_see_stats.py, label_features_integrated.py, audit_buckets.py,
subbucket_and_rollup.py, render_taxonomy_tree.py. (Superseded but kept: fuse_feature_names.py, label_features_see.py.)

---

## 2026-06-05 — v7 pipeline applied to d2048_k4 (the k4-vs-k6 comparison, finally apples-to-apples)

Ran the full v7 peak+median labeling pipeline on `btk_2048_k4_nol2.pt` — same L7 cache
(`maia3_l7only_v2_dedup.pt`), same recipe, same opus-4-8 xhigh engine as k6. Differs only in k (4 vs 6).
Verified inputs clean before launch (right cache, W_enc [1024,2048], d_input matches).

**Result — k6 wins on every quality metric.** The hypothesis that k4 might be *less* over-specialized was
wrong; it's marginally *more* polysemantic.

| metric | d2048_k6 | d2048_k4 |
|---|---|---|
| live / labeled | 2033 | 1148 |
| median consistency | 63 | 60 |
| mean consistency | 65 | 62 |
| ≥70 (clean) | 35% | 22% |
| confidence=high | 38% | 30% |
| blobs (>5% fire) | 15 | 11 |

**Why:** fewer live features (1148 vs 2033) covering the same corpus → each does more work → more mixed
boards per feature. Sparser per-position activation (k=4) does NOT buy cleaner features when you also have
fewer total features. The blobs are the SAME feature indices in both dicts (f1487, f1313, f952, f1372,
f2018, f1965, f290, f1329, f1165) firing on the same coarse material-lost patterns — k4 doesn't escape
the coarse-detector problem either.

**What v7 DID fix: the naming.** The earlier k4 frustration was bad labels, and that's gone — top k4
features read clean (f647 "Missed mate, grabbed material (M1)", f1946 "Hangs queen (usually to bishop)",
f148 "Allowed back-rank mate (no luft)", f1475 "Premature queen trade (squanders win)"). So v7 naming
generalizes across dictionaries; k4 is just not a better dictionary than k6.

Taxonomy: 1148 assigned to the 12 v3 categories (7 unassignable), 163 clusters, 11 blobs folded as
`⚠ Coarse detectors`. Atlas rendered + visually verified (home / bucket / expanded feature all clean,
boards draw with played/best arrows). **k6 remains the dictionary of record.**

Scripts added (were ad-hoc for k6, now committed for reproducibility): `build_leaf.py` (clusters → atlas
leaf + blob fold), `profiles_to_atlas.py` (peak/median → atlas profiles + best_uci_map). `render_atlas_v3.py`
got a `--dict-label` arg so the title isn't hardcoded to k6.

## 2026-06-07/08 (overnight) — rule-based mistake tagger: cook port finished, validated against SAE

**The pivot is real and it works.** Abandoned the SAE for per-position label *assignment* (it's
polysemantic — f55 cons 50, f0 split). Built a deterministic rule tagger in `scripts/04_tagger/`.
Tagging blunders with a known vocabulary ("Missed Fork", "Allowed Mate", "Hung Material") is supervised
classification, not unsupervised discovery — rules win. The SAE's lasting value: it seeded the label
vocabulary AND it's now the ground-truth regression/validation set.

**Architecture (see `scripts/04_tagger/README.md`):** L0 `mistake.py` (data contract) · L1
`predicates.py` (position/material) · L2 `motifs.py`+`chesslib_util.py` (tactics/mates ported from
lichess-puzzler cook.py, pov-explicit) · L3 `maia_rarity.py` (numeric). Orchestrator `tagger.py`.

**The key idea — three directions from one detector set:** Missed X (best line, pov=mover) / Allowed X
([played]+refutation, pov=opponent — the puzzle shape) / Failed X (played move, single-move only).

**THE BUG CLASS this whole module kills (pov parity):** cook hardcodes `mainline[1::2]` (solver moves)
because a puzzle always starts with the opponent's setup blunder. That parity is right for ALLOWED,
WRONG for MISSED. The old `cook_adapter.py` overrode pov + used a parity-union hack → **Sacrifice 46%**.
Port replaces `mainline[1::2]`→`pov_nodes(nodes,pov)`, `mainline[::2]`→`opp_nodes(nodes,pov)`, pov
explicit. Verified: `node.turn()` = color to move AFTER the node, so pov moves = `turn()!=pov`. cook
geometry copied verbatim (proven). Same bug found+fixed in 3 places:
- `sacrifice_line` raw `diffs[1::2]` sampled wrong side in MISSED → Missed Sacrifice 24.6%→12%.
- `mate_in_line` used `len//2` (assumes node0=opp) → count pov's own moves → parity-robust.
- `hung_material` (L1) measured GROSS mover loss, ignored recaptures → over-claimed ~2x (30% of fires
  claimed 5+pts when cp_loss justified <40% of that). Fixed to NET material_diff swing →
  **Hung Material 65%→41.6%, cp/material agreement ratio 0.56→0.91.**

**Validation — "not completely wrong" gate PASSED.** `validate_vs_sae.py` cross-checks detectors against
Sam's hand-confirmed SAE gold (`relabel_v9_d64_k1.json` + `all_feat_boards_d64_k1.json`). On the SAE's
OWN confirmed positions: f54 fork/pin **87%** agreement (0% on a non-tactical control), f3 hanging 67%,
f17 free-capture 60%, f47 mate **9/9**, f59 allowed-mate 1/1. <100% expected (SAE polysemantic). The rule
detectors agree with the hand-confirmed ground truth AND have clean specificity. `regression.py` 16/16
(single-move from gold f54, line/mate, mate-suppression).

**Coaching-quality fixes beyond cook:** mate-suppression (a forced mate outranks lesser tactics in the
same direction — "you missed mate in 3", not "you missed a fork"). Endgame-type + pawn-structure +
backward-pawn predicates added (wishlist tags). Maia rarity (L3) verified end-to-end (rare/common
blunder, skill-gap move) but not run on full corpus (slow ONNX; it's a per-game report-time annotation).

**Full corpus (19,362 blunders) → 104 distinct tags, fire rates all sane:** Hung Material 41.6%,
Missed Sacrifice 15.6%, Allowed Mate 6.4%, Allowed Sacrifice 4.8%, Missed Fork 4.1%, named mates firing
(back-rank, smothered, anastasia, etc.), endgame types (Rook 5.6%, Pawn 3.6%, Bishop/Queen ~1%).
Evaluation atlas `output/tag_atlas.html` (98 tags, example boards + SAN lines + cp_loss + freq) built +
visually verified via Playwright. `output/mistake_tags.json` (14MB) gitignored — regenerable in 12min.

**Open for Sam to judge in the atlas:** Hung Material still 41.6% (high but now net-calibrated; ~20%
remaining are positional-compensation cases, arguably correct) · Missed Sacrifice 15.6% vs Allowed 4.8%
(best lines longer → more material-investment chances; "Missed Sacrifice" vs "Missed Combination"?) ·
skewer geometry still a known TODO (rare ~1%) · interference/clearance/backward-pawn lower-confidence.

NOTE: Sam's hand-edits to `relabel_v9_d64_k1.json` (the per-feature chip corrections + `review:confirmed`
flags) are uncommitted in the working tree — left untouched, the validation script only READS that file.
