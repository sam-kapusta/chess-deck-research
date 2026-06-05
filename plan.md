# Chess Encoder

> **CORRECTION (2026-05-31): the option_a / board_diff / l2l7 architecture search below is INVALID.**
> All three SAEs were trained on the v1 blunder cache (`maia3_blunder_diff.pt`, Black-to-move
> label-inversion bug). The weights, caches, labels, profiles, and eval artifacts have been
> deleted from S3, the notebook, and git. No conclusion below (board_diff "leading candidate",
> f46 "Recapture leaves piece hanging", >50%-coherent expectations) holds. Build/train scripts
> kept under `scripts/sae/new_sae_architecture/` but must be repointed to the v2 cache before
> any rerun. Next step if revisiting: rebuild all three constructions on v2 data.

## CURRENT STATE (2026-06-04) — d2048_k6 fully relabeled with all-fields method

**k6 is the working model now** (`btk_2048_k6_nol2.pt`, S3 `sae/weights/`). All 2035/2047 live
features relabeled by `scripts/03_feature_labeling/relabel_all_fields.py`.

**Why we relabeled (the methodology fix):** the old `label_features_integrated.py` fed Opus only
~2.5 of the 7 per-position fields — it DROPPED `best_moves_analysis` (what the player should have
played) and `move_intent`. Without the best move you can't tell "hung a piece" from "missed
winning a piece", so SEE's single-ply `blunder_hangs_own` made it label missed-X features as
passive/hung. The fix feeds all fields + a one-line SEE floor (the long "distrust SEE" caveat
over-steered and made labels flip between runs — one neutral line is stable).

**Result (`output/relabel_allfields_d2048_k6.json`):**
- 2035 labeled (12 insufficient_boards). 1910/2035 chips changed vs old integrated labels (94%).
- `missed_win` is the LARGEST category — 1020/2035 (50%). The direction-blindness was
  dictionary-wide, not a few features.
- spread: missed_win 1020 · hung_own 478 · endgame 216 · greedy 164 · positional 87 · trade 70
- consistency: median 90, mean 87. 95 features ≤70 flagged for review (~4.7%). The
  `consistency` field is the review signal — genuinely-mixed features (f745-type) come back ~80,
  and over-fit names ("Missed Nxd4") cluster in the flagged set.

## RE-BUCKET DONE (2026-06-04) — v3 12-bucket taxonomy, bottom-up, validated 3×

Rebuilt the taxonomy from scratch on the new labels. Old buckets were built on the OLD (wrong-
direction) labels — stale. Method (bottom-up, not top-down — the taxonomy_v2 lesson):
- Tried mechanical clustering (Titan embeddings, agglomerative). FAILED: the prose labels all
  share "Player consistently..." scaffold + chess vocab, so everything is cosine-near everything.
  No flat distance cut works (0.45→156 clusters, 0.55→one 958-blob, 0.75→total collapse).
  Chip-only embedding separates better but still 150+ clusters. Bottom-up clustering alone can't
  find top-level structure here — recorded as a dead end.
- What worked: sample 200 features (chip+label+SEE self-inflicted/omission tell), ask Opus for the
  natural top-level set, reuse old 11 as starting hypothesis (not forced). Ran on **3 disjoint
  200-samples** — all converged on the same ~11-12 buckets within a few % each. Stable taxonomy.

**v3 taxonomy (`output/buckets_v3_d2048_k6.json`, 12 buckets):** organized by error CHARACTER —
- self-inflicted (played move loses): Left Piece Hanging, Abandoning Defensive Duty, Greedy
  Capture, Premature Trade, Unsound Aggression, Pointless Check, King Safety Error
- omission (played move safe, missed a win): Missed Hanging Piece, Missed Tactic, Missed Check
  or Mate, Passive Play
- phase: Endgame Technique

**Assignment (`output/feature_buckets_v3_d2048_k6.json`, all 2035, only 5 unassignable):**
Missed Tactic 371(18%) · Left Piece Hanging 328(16%) · Missed Hanging 287(14%) · Endgame 210(10%)
· Greedy 173(9%) · Missed Check/Mate 164(8%) · Pointless Check 111 · Premature Trade 102 · Passive
89 · King Safety 79 · Abandoning Defense 59 · Unsound Aggression 57. **No catch-all** — biggest
bucket 18%, omissions properly split by WHAT was missed (the original "missed_win=50%" problem is
gone). fire% >> feat% in Left-Hanging (91%) and Missed-Tactic (106%) = those buckets hold the blobs.

## DEBIAS + SUB-BUCKET DONE (2026-06-04 cont'd)

**Found the relabel was biased.** v1 relabel prompt said "prefer Missed X if a capture was
available" → 275/2035 features direction-conflicted (chip said "Missed" but the played move
hangs own material). Audited f19/f745 with Stockfish (`audit_direction_stockfish.py`): the real
hang-vs-miss tell is PLAYED-move-captures vs BEST-move-captures, NOT SEE hang% (and NOT the
drop-vs-gap heuristic, which is tautological garbage — see [[project_direction_arbiter_is_board_not_see]]).

**Fix = v2 neutral relabel** (`relabel_all_fields_v2_neutral.py`): same prompt/format as v1, the
biasing sentence replaced with "decide direction from evidence; high loses-own = Hangs X even if a
better move existed; name dominant, lower consistency when dual." Re-ran all 2035
(`relabel_v2_neutral_d2048_k6.json`, 2027 labeled). Direction shift: missed_win 1020→905,
hung_own 478→707 (~200 features corrected). consistency median 85 (vs 90 — lower is honest;
dual features now score ~80 not falsely-confident). 130 flagged ≤70.

**Re-assigned to v3 buckets** (`feature_buckets_v3_v2labels_d2048_k6.json`, 3 unassignable):
dictionary now ~56% self-inflicted / 33% omission / 10% endgame. Left Piece Hanging largest (433,
21%, 128% fire — blob-heavy).

**Sub-bucketed** (`subbucket_v3.py` — MECHANICAL, no LLM; `feature_leaf_v3_d2048_k6.json`):
material buckets split by piece, tactical by theme keyword. Clean splits everywhere except Missed
Tactic's "General/Forcing" sub (220, 95% fire) — labels too vague to subdivide, holds the blobs,
surfaced honestly. **Browsable tree: `output/atlas/taxonomy_v3_d2048_k6.html`** (char-group →
bucket → sub → feature, blob fire% flagged red, boards clickable to chess.com).

## BLOB REDUNDANCY RESOLVED (2026-06-04 cont'd) — they're distinct, not duplicates

Worried the blobs (esp. 92 "hangs own queen" features) were the SAE smearing one concept across
many redundant features → splittable. Tested 3 ways (scripts: `decoder_overlap.py`,
`firing_overlap.py`):
- **Decoder cosine** (W_dec rows): mean 0.05, max 0.59 vs random-pair p99=0.27 → near-orthogonal.
- **Top-10 board Jaccard**: 0.0009 → disjoint.
- **Corpus-wide firing Jaccard** (encode all 168K, fire = act≥0.5·max): mean 0.001, median 0.000,
  1 pair >0.3 of 4186 → disjoint. (`output/firing_overlap_qh.json`)

**Verdict: the 92 queen-hang features are GENUINELY DISTINCT** — each fires on a different position
type (to-bishop / giving-check / promoted / opening / ...). NOT redundant, NOT splittable. They
only LOOK redundant because they collapse to the same 2-word chip. f952 (17% fire) is not a bad
blob — it's the most GENERAL queen-hang detector, with 91 finer siblings = a coarse→specific
hierarchy (good structure). The "high overlap-coefficient" pairs are tiny specific features being
subsets of f952's broad range (Jaccard still ~0), exactly what that hierarchy looks like.

**So the blob problem is NAMING, not the features or redundancy.** Implication for next steps:
don't merge/filter blobs — instead (a) give same-concept features mechanism-specific chips, and
(b) the coarse→specific hierarchy is a feature, not a bug — surface it (general detector at top of
a sub-group, specific siblings under it).

## v3 5-WORD LABELS LOCKED (2026-06-04 cont'd) — the label baseline

Labels improved monotonically: integrated "Piece Left Hanging" → v2 "Hangs own queen" → **v3
"Hangs queen attacking enemy queen"** (the f882 worked example — player attacks the ENEMY queen
but leaves their OWN hanging; mutual-queen exchange they lose). v3 = neutral direction + 5-word
mechanism chips, fed SEE + move_intent + best_moves + blunder_summary. **v3 is the baseline.**

- `output/relabel_v3_5word_d2048_k6.json` (2006 labeled): chip diversity 57%→78% unique vs v2;
  worst duplicate 52→26. Mechanism now in the chip (to-bishop / giving-check / open-file / etc).
- `output/feature_buckets_v3_5word_d2048_k6.json` — categories (3 unassignable).
- `output/feature_leaf_v3_5word_d2048_k6.json` — sub-clusters (by piece / by theme).
- **Atlas: `output/atlas/atlas_v3_5word_d2048_k6.html`** (SPA, 1.6MB, browsable, boards verified).

**v4 was a REGRESSION — removed.** Tried adding refutation_analysis + result-framing; it pushed
f882 back to generic "Leaves own piece undefended" by over-weighting the refutation and dropping
the move_intent signal ("intending to attack the queen") that carries the specificity. Lesson:
move_intent + blunder_summary read together already carry the pattern; don't over-engineer.

**Worked-example lesson (f882):** the pattern WAS in the prose all along (move_intent: "attack the
queen", blunder_summary: "leaves own queen undefended"). Claude reading 10 boards/summaries
together can get there; the failure mode is ME mis-reading individual board tactics as different
mistakes when they share one structure. The board-geometry "both queens attacked" signal is real
but a too-specific fix — v3's existing inputs already suffice.

## LLM CLUSTERING DONE (2026-06-04 cont'd) — 2005 features → 174 coaching clusters

Within each category, Opus groups features (chip+label) into natural coaching clusters. Decoder
and chip-embedding clustering both FAILED (muddy "Hangs"-blob groups — geometry ≠ coaching theme);
LLM clustering reading labels holistically is the method that works. `cluster_llm.py` (coarse,
10-20/category, merges near-dups). Output: `feature_clusters_llm_d2048_k6.json`,
`feature_leaf_llm_d2048_k6.json`. **Atlas: `output/atlas/atlas_v3_llm_d2048_k6.html`.**

**Cleanup applied:**
- Cross-category moves (4 clusters): RULE = Endgame Technique is mistakes that can ONLY occur in an
  endgame (king activation, opposition, pawn breaks). A fork/hang/missed-capture that happens in an
  endgame goes with its MISTAKE-TYPE, not Endgame. Moved "Missed capture in endgame", "Greedy loses
  endgame advantage", etc. out of Endgame.
- Per-cluster audit (`audit_clusters.py`): placed all 47 orphans, moved 54 misfit members to better
  clusters. 7 targetless flags (features fitting no existing cluster) left in place. 0 catch-alls.
- Integrity verified: 2005 features, 0 dupes, each clustered once. 8-23 clusters/category.

Cluster counts: Missed Tactic 20, Left Hanging 23, Greedy 23, Missed Hanging 17, Missed Check/Mate
14, Endgame 13, Abandoning Defense 13, Pointless Check 12, Passive 11, Premature Trade 10, Unsound
Aggression 10, King Safety 8.

**NEXT:**
1. Optional: the 7 targetless-flagged features + any low-consistency review.
2. Then: apply taxonomy to cabbagelover5566's 1,209 blunders (the coaching payoff).

## (superseded) earlier re-bucket on v1 labels
The first v3 assignment used the BIASED v1 labels (`feature_buckets_v3_d2048_k6.json`). Superseded
by the v2-label assignment above. Taxonomy buckets themselves unchanged (validated 3×).

Pipeline now lives in `scripts/03_feature_labeling/` (Personas NN_stage convention) with a README.

## DONE (2026-06-03) — k4-vs-k6 head-to-head + k6 gap labeling
Gap labeling completed (8,193 k6 positions Opus-labeled so k6 isn't motif-handicapped). k6 won on
vocabulary at equal cleanliness; chosen as working model. `see_stats_d2048_k6.json`,
`d2048_k6_profiles.json`, gap-filled `all_positions_labeled_opus.json` (62,956 positions) all on
notebook. Spec: `docs/superpowers/specs/2026-06-03-k4-vs-k6-interpretability-headtohead-design.md`

## Current State (2026-06-03 end) — d1024_k4 fully labeled + 11-bucket taxonomy, audited

**Working model for the taxonomy:** `btk_1024_k4_nol2.pt` (S3 `sae/weights/`). Chosen over k6 for this
pass because it's fully alive (0 dead), cleanest concentration, 1024 useful features. k6 remains the
structural sweet-spot candidate (see below) but we labeled d1024_k4 first — apply-to-games will tell us
whether k6's extra vocab is needed before retraining the taxonomy on it.

**LABELING — final method (`label_features_integrated.py`):** Opus names each feature from THREE
integrated signals, not SEE alone (SEE mis-reads trades as material loss and is blind to positional/
trajectory mistakes — the f91 lesson). Inputs:
1. SEE descriptive stats (`compute_feature_see_stats.py`, per-feature **normalized cohorts** ≥0.7max &
   ≥0.8max — features are pure at their activation peak, noisy in the tail): moved/captured piece,
   **net-material kind** (trade vs loses vs hangs — fixes the trade-as-loss bug), played-check, phase.
2. **Eval trajectory** (winning/drawn/losing, player POV, ±150cp draw zone): what the mistake cost.
3. Opus per-position **tactical_motif + tags** (queen_trade_error, king_safety…) — the positional layer.
- All 1020 live features labeled → `output/feature_labels_integrated_d1024_k4.json` (+S3). f91 went from
  wrong "Greedy Queen Capture" → correct **"Premature Queen Trade"** once trade-material + trajectory + motif fed in.
- Stats: `output/see_stats_d1024_k4.json` (+S3). Profiles: S3 `d1024_k4_profiles.json`.

**TAXONOMY — 11 buckets (`buckets_v2_d1024_k4.json`), all 1020 assigned + audited (1.0% flagged, all verified):**
Left Piece Hanging 219 · Endgame Technique 173 · Missed Tactic 138 · King Safety 88 · Missed Check/Mate 79 ·
Missed Hanging Piece 78 · Premature Trade 66 · Passive Play 65 · Pointless Check 58 · Greedy Capture 29 ·
Unsound Aggression 27. Built bottom-up (mistake_type spine), audited via `audit_buckets.py` (objective
mechanism cross-check) + Opus semantic grade; 17 reassignments applied; Ignored Tension folded into Missed
Tactic. **Sub-buckets** by piece (Left Hanging → Hung Queen/Rook/…) / phase (Endgame → King/Pawn/Rook),
with fire-rate coverage per sub. Assignments: `feature_buckets_v2_d1024_k4.json`, leaf:
`feature_leaf_v2_d1024_k4.json`. Browsable tree: `output/atlas/taxonomy_tree_v2_d1024_k4.html` (gitignored,
regen via `render_taxonomy_tree.py`).

**NEXT (priority):**
1. **Apply taxonomy to cabbagelover5566's games** — his 1,209 blunders (502 games) → encode via Maia3-L7 →
   d1024_k4 → map to the 11 buckets → leak report. Validates the taxonomy + comparable to the old
   2026-05-27 profile (`cabbagelover_profile.json`, used the OLD SAE: "Pawn Captures That Open King" 2.3×,
   "Trading Away Dominant Pieces" 1.5×). This is the coaching payoff AND the only criterion for k4-vs-k6.
2. **Then** consider k6: relabel + re-bucket `btk_2048_k6_nol2.pt` with the same pipeline, compare leak reports.
3. Detection-score (auto-interp held-out) for label quality; feature-splitting metric.

---
## k-sweep / structural analysis (2026-06-02) — k=6 is the structural sweet spot

**Leading model:** `btk_2048_k6_nol2.pt` (notebook; z-score-only). Superseded the earlier
"k=16 chosen" call after a full k-sweep (k4/6/8/10/12/16/32) + dict-size + sparse-probing analysis.

**Why k=6 (corroborated three independent ways):**
1. **Mass concentration** — k6 holds the most activation signal (61.5%) in the useful 0.1–10%
   fire band among fully-alive models (0 dead), with the fewest blobs. k4 has more band-mass
   (67%) but 951 dead features; everything above k6 bleeds mass into blobs.
2. **Raw Gini** (threshold-free) is U-shaped with its MINIMUM at k6 — the only non-monotone
   structural metric, a real interior optimum. Below k6: dead features hoard mass. Above k6:
   blobs hoard mass.
3. **Sparse probing** (SEE concepts) — bal_acc@1 for cleanly-isolated concepts (hang_queen 0.81,
   hang_major 0.78, best_check 0.89) is FLAT or DECREASING with k; at k16 concepts start
   SPLITTING (hang_queen@1 drops 0.81→0.70, top feature changes f952→f926). Lower k keeps
   concepts in single nameable features. k4≈k6 on @1; k6 edges it on smeared concepts.

**Decided earlier (still holds):**
- DROP L2 — z-score-only nearly triples coherent features. Magnitude = severity, L2 erases it.
- Coherence/signature must measure BOTH moves (blunder + best); best-move axis dominant.
- Position-descriptor axes (phase/dir/severity/traj) are leaky base-rate effects — confirmed by
  sparse probing (endgame/severe barely beat base rate @1, +0.04/+0.06 lift).
- dict=1024 recovers k4's concentration without dead waste but caps vocab (452 feats); 2048
  gives room to explore. Dead-count is a dict_size artifact, not a quality signal.
- 0% decoder twins at every k — extra features at high k are distinct directions, not duplicates.

**Labeling pipeline built this session** (skill: `label-sae-features`):
- `enrich_gap.py` (Stockfish) → `label_positions_btk.py` (Opus per-position motif/tags) →
  feature-level labeler. Enrichment cache 55k→77.8k. Opus position labels 48.6k→54.8k.

**LABELING METHOD — corrected mid-session (this is the one that works):**
- ❌ `fuse_feature_names.py` (per-position SEE-stat aggregation → name) was WRONG. It fragments the
  concept across top-10 and got direction backwards: f127 came out "hangs a piece; missed a capture"
  when it's actually MISSED HANGING PIECE (player ignored a free enemy piece). Per-position voting on
  my own-hang metric buried the real signal. Kept in repo but superseded.
- ✅ `label_features_see.py` is correct: **Opus reads each feature's top-N boards HOLISTICALLY**, and
  is handed a **SEE-on-both-moves aggregate over top-500** as raw data. The aggregate disambiguates
  direction: `best_wins_material_pct` (missed winning) vs `blunder_hangs_own_pct` (hung own). f127 →
  91% best-wins-material → "Missed Win, Hung Piece". Two-step: `compute_feature_see_stats.py`
  (top-500 SEE-both, 16-proc) → `label_features_see.py` (Opus 4.6, eyeballs 12 boards + aggregate).
- **d1024_k4 fully labeled: 1020/1024**, 629 distinct chips. Clean disambiguation: Missed Hanging
  Piece (26), Hung Own Piece (25), Missed Winning Check (35), Greedy Capture Hangs Piece (20),
  Missed Knight Fork (8)... Opus names mechanisms SEE can't (f198 "Ignored Pawn Attacks Knight").
- Results: `output/feature_labels_see_d1024_k4.json` + `output/see_stats_d1024_k4.json` (git);
  also S3 `sae/labels/`. Sparse-probe `output/eval/sparse_probe_results.json`.

**CATEGORIZATION (next real task) — natural taxonomy is 2-axis:**
- PRIMARY = **mistake mechanism** (objective, from SEE stats, no LLM): Missed-winning (218 feats,
  bestwins≥70 & ownhang<50), Hung-own (108, ownhang≥70 & bestwins<50), Both/greedy (213, both≥60),
  Other/allowed (481 — need motif axis to organize).
- SECONDARY = **tactical motif** (Opus): fork / pin / skewer / back-rank / hanging-piece / promotion.
- Piece (queen/rook/minor/pawn) + severity = FILTERS, not categories.
- Do BOTTOM-UP clustering of the 1020 SEE-grounded chips, then name clusters — do NOT force into the
  old `taxonomy_v2.json` 20 buckets (suspect: top-down, unknown source model — see umbrella memory).

**NEXT (in priority order):**
1. **Categorize the 1020 features** — mechanism×motif taxonomy (above). Bottom-up cluster the chips.
2. **Label + name k6** (the leading model) via `label_features_see.py` — build k6 profiles +
   `compute_feature_see_stats.py --model k6`, top-10 already Opus-covered from union enrichment.
3. **Feature-splitting metric** (lit review's #1 unrun fit) — k16 extra features new vs fragments.
4. Audit the SEE labels (skeptical reviewer vs top-10) — earlier fused-label audit was 31% clean.
5. Detection-score (auto-interp held-out) as the non-circular label-quality metric.
6. Ship-vs-expand decision (1M Lichess corpus optional, for coverage not blob-fix).

**Experiment scripts added (git):** blob_metric[_norm].py, dual_coherence.py, dual_2x2.py,
multiaxis[_lift].py, see_coherence.py, blob_activation_decay.py, make_subsamples.py, calibrate_threshold.py.

## Current State (2026-06-01 end) — BatchTopK k=16 labeling overnight (SUPERSEDED)

**Active overnight:** Pass-1 Opus labeling gap positions for k=16 v2 profiles (13,208 positions, ETA ~2.5h from ~22:00). Pass-2 (feature chips) chains after. Outputs: `all_positions_labeled_opus.json` (growing to ~47k), `feature_labels_btk_2048_k16_v2.json`.

**What's done this session:**
- BatchTopK SAE trained (matches SandstonePersonas exactly): 2048/k16 v2, 200 epochs. Weights in S3.
- Calibrated inference threshold θ=0.0806 (k-th largest method). All eval scripts updated to use threshold not BatchTopK.
- Feature stats computed with extended fields: piece type (blunder + best move), is_capture, is_check, piece_left_hanging. Both k=16 and k=32.
- Atlas (`l7only_atlas.html`) working with clickable boards + chess.com links.
- corpus builder (`cache_real_game_blunders.py`) identified — streams `Lichess/standard-chess-games`, pulls 200k blunders with elo + eval trajectory. Plan to run 1M position build.

**Key decisions made:**
- k=16 over k=32 for coaching precision (Noyan/Jonathan recommendation, paper confirms k=16 best for downstream CE on Gemma 2B)
- BatchTopK at inference is wrong — must use calibrated threshold (per arXiv:2412.06410 §3 + inference_example.py)
- AuxK not firing is correct behavior for 168k corpus (features don't die, they're just rare)
- Inference threshold method: mean of k-th largest activation per position (not min-positive, which collapses to 0 on small corpora)

**Next session:**
1. Check overnight results — download `feature_labels_btk_2048_k16_v2.json`, coherence stats
2. Run T1 with threshold inference on k=16 v2
3. Re-run feature stats with threshold inference (already uploaded, pending run)
4. Update atlas with stats panel + k=16 labels
5. Launch 1M Lichess corpus build (`cache_real_game_blunders.py --n-positions 1000000`)
6. Commit `cache_real_game_blunders.py` to git (currently only on notebook via S3_INVENTORY §6)

## Current State (2026-05-31 end) — v2 REBUILD running (superseded by 2026-06-01)

**Active work:** rebuilding the 4 SAE constructions on v2 (corrected) data + Maia-best moves.

**Key unblock this session:** the 3 constructions need a *best move* per position; v2 cache
dropped best_uci and the only 200k best_uci (v1 metadata) is bug-affected. **Decision: best =
Maia3 policy argmax @ elo 2600** (not Stockfish) — the SAE reads Maia's activations, so Maia's
own human-best is the consistent, coaching-relevant target. Building `maia_best_200k.json`.

**Maia-best extraction gotcha (cost a debug cycle):** must use the maia3 package primitives —
`get_all_possible_moves()` = **4352**-move vocab (NOT the 1968-move `move_to_action.json`, which
is the DeepMind-270m model), `tokenize_board` (mirrors board for black-to-move),
`get_legal_moves_mask`, and `mirror_move` on the chosen move for black. Wrong vocab → 3%
Stockfish agreement (chance). Correct → 50% agreement + passes start/hanging-Q/M1 sanity.

**Unattended pipeline (2 screens on chess-poc):**
- `maiabest`: `build_maia_best.py` → `maia_best_200k.json` (~110 pos/s, ~30min)
- `pipeline`: `wait_and_run.sh` → `run_all_v2.sh`: build option_a + board_diff + l2l7 caches →
  slice l7only → train 4 SAEs @ **k=16** → `eval_v2_html.py` → `eval_v2.html` + results JSON

**4 constructions (all v2 + maia_best@2600, conditioned at player's real elo):**
- option_a: `h[best_to]-h[blunder_to]` before-board (ONNX, 512d)
- board_diff: `mean64(h_after_best - h_after_blunder)` (ONNX, 512d)
- l2l7: `concat(L2_mean64_diff, L7_mean64_diff)` (79M PyTorch, 2048d)
- l7only: L7 half of l2l7 (1024d)

**Scripts (in /tmp locally + ~/SageMaker on chess-poc; commit to scripts/sae/ before trusting results):**
`build_maia_best.py`, `build_option_a_v2.py`, `build_board_diff_v2.py`, `build_l2l7_v2.py`,
`train_sae_v2.py` (generic), `eval_v2_html.py`, `run_all_v2.sh`, `wait_and_run.sh`.

**Decision after eval:** if one SAE clearly fires the right features on the 10 test positions →
start labeling it. Drop degenerate positions (maia_best==blunder, ~14%).

 — SAE Feature Pipeline

## Current State (2026-05-30)

### Taxonomy design — LOCKED via /grill-with-docs (2026-05-30)

Decisions (don't re-litigate):
1. **Shape:** 2-level, **category → cluster → feature**. Fresh artifact, replaces the old prod taxonomy (`realgames_2048_k64_v1` domain/subcategory schema is stale — not bound to it).
2. **Top-level axis = mistake-type ("what kind of mistake can a player make"), FLAT.** No "missed-win vs losing-blunder" halves wrapper at the top (considered, rejected — Sam thinks in mistake-types, not in that split).
3. **Data-driven, not hand-imposed.** The ~11 seed categories below are the *target shape*; the real categories come from reading each feature. Merge/drop/rename by what the data supports. Anything under ~2% folds into a neighbor.
4. **Offensive-miss mistakes are first-class top-level categories**, not sub-flavors of a "slow play" blob. The big bge-m3 "Slow Play Punished / Autopilot" mass (~61% of features) is NOT one junk-drawer category — it's several real "you had a line and played a nothing-move" categories collapsed under one bad name. This was THE key correction (Sam: "the entire Slow Play Punished category is what I've been talking about where you missed a good move").
5. **CRITICAL METHOD NOTE:** keyword classification is UNUSABLE for assigning these categories — 77% of features match 3+ category keywords, and commission-vs-offensive-miss are keyword-entangled (same feature counts as "hung a piece" or "missed a capture" depending on which keyword you check first). Counts MUST come from reading each feature's description (LLM/agent judgment of the PRIMARY mistake), not regex. Every keyword-based count this session was an artifact — ignore them.

**Seed category vocabulary (~11, data may revise):**
- Commission: Hung a Piece · Walked Into a Tactic (got forked/pinned) · Greedy Capture · Exposed Your King · Bad Trade/Simplification · Abandoned a Defender
- Offensive miss: Misplayed an Attack (had attack/initiative, failed to convert to mate OR winning material) · Missed a Capture · Missed a Tactic
- Other: Endgame Error · Missed a Defensive Resource


### Sub-cluster level = COACHING TOPICS, not fine slivers (decision 2026-05-30, revises earlier)

When asked "would you use these 12 endgame clusters to coach?", answer was NO — semantic
clustering fragments ONE skill into many (e.g. 5 near-dup "king race" clusters). A coach
organizes by SKILL TO LEARN, few meaningful topics. So:

- **Structure: category -> coaching TOPIC -> feature** (2-level). The cluster level = ~3-6
  coaching topics per category (what a coach actually teaches), NOT ~12-15 fine slivers.
- **SUPERSEDES the earlier "sub-clusters in 2-15 range" goal** — that target pulled the wrong
  way (kept slivers apart; coaching wants them merged). Topic size can be large (60+ feats) if
  it's one real skill.
- Method per category: read fine clusters -> drop misfits (boundary rule) -> CONSOLIDATE into
  the few coaching topics a teacher would name -> verify coherence.
- **Endgame Error done (template):** 3 topics — King & Pawn Technique (65), Passed Pawns &
  Promotion (36), Endgame Piece Activity (32). See output/taxonomy_v2/endgame_final.json.


**Model is pinned:** `maia3_sae_diff_v2_2048_k32_l2` (flat k=32, v2 corrected data). Verified labels align + fire rates in `output/taxonomy_v2/firerate_flat_v2_k32.npy`. See below for the earlier provenance hunt.

**Scheme exploration done:** `output/taxonomy_v2/TOP_LEVEL_SCHEMES.md` + `schemes_atlas.html` (the latter's category assignments are keyword-based → stale; structure/UI is the keeper). `chess_taxonomy_atlas.html` is STALE (old top-down 20-cat) — regenerate after assignment.

**Next:** robust reading-based assignment of 1996 features to the seed categories (small agent batches ≤10 to avoid the StructuredOutput stall seen at batch=50), then sub-cluster within each, then atlas. Then 3 QC passes (misfit reconciliation, member verification, coherence bar).

### ⚠️ Taxonomy needs a clean redo on the FLAT k=32 model (earlier 2026-05-30 notes)

Sam wants the taxonomy built on the **flat k=32 SAE** (`maia3_sae_diff_2048_k32_l2_200ep.pt` — 200ep, the "2007 labeled" champion per S3_INVENTORY), with **semantic sub-clusters inside each category** and **fire rates** (per feature, summed per cluster + category). Two problems with the existing `taxonomy_v2.json` must be fixed:

1. **Categorization was done top-down (WRONG ORDER).** Each feature was independently dropped into one of 20 pre-baked categories → magnet effect: Slow Play Punished got 408, Pieces Left Undefended got 4. The persona-atlas method (see `docs/knowledge/taxonomy-method-persona.md` — paste from Sam) is the correct one: **cluster FIRST on label-text semantics (bge-m3), let categories emerge bottom-up, one agent regroups within each.** No imposed buckets.
2. **No sub-clusters, no fire rates.** Both were asked for from the start; the flat→atlas only had category→feature.

### ⚠️⚠️ PROVENANCE BUG — taxonomy_v2 labels' source model is UNKNOWN (verified 2026-05-30)

`l2_feature_profiles_v2.json` (the profile the 2007 Opus labels were built from) was **NOT reproduced by ANY checkpoint I tested**: not flat k=32 (l2_200ep / v2 / base), not the H1 perlevel matryoshka. Verified via: forward-pass v2 cache (`maia3_blunder_diff_v2.pt`, idx 137471 == profile feat3 ex0 Bxf7+ ✓ so the *cache* is right), check whether feature 3's top firings = the profile's Bxf7+ set. ALL candidates gave 0/10.

**Implication:** the labels in `taxonomy_v2.json` are bound to a profile of unknown model origin → treat `taxonomy_v2.json` as **suspect, not a foundation.** Don't cite its per-feature category as ground truth.

**Clean path (next session):** regenerate from scratch on the flat k=32 model — fresh profile (`extract` top-20 per feature over v2 cache, flat top-k=32, z-score→L2 norm), join the 19K Opus English by `fen|uci`, then bge-m3 cluster → emergent categories → fire rates. One known model end-to-end = reproducible.

**Established facts (verified this session, trustworthy):**
- v2 corrected cache: `chess-poc:~/SageMaker/chess-stage-a/cache/maia3_blunder_diff_v2.pt` (200K×512, has `metadata` with fen/blunder_uci/cp_loss, NO stored mean/std — compute z-score then L2 per `label_v2_features.py`).
- Normalization that label scripts use: `x=(raw-mean)/std; x=x/||x||`.
- bge-m3 + sklearn available locally → semantic clustering runs locally, no Bedrock.
- 19K Opus English: `chess-poc:~/SageMaker/all_positions_labeled_opus.json` (19,342, keyed fen|uci). S3 `_final.json` truncated to 10,648 — DON'T use.

**Matryoshka SAE:** Per-level-k, dict=[32,256,2048]=2336, k_per_level=[3,8,16]. Zero dead, FVU=0.209. See `docs/knowledge/matryoshka-sae.md`. (Separate track; Sam wants flat k=32 for the taxonomy right now.)

**Next steps (taxonomy redo, in order):**
1. Generate fresh profile + per-feature fire rate from FLAT k=32 (`l2_200ep`) over v2 cache. Verify feat-N top firings look sane.
2. Embed each feature's label-text (chip+description, or fresh Opus labels) with bge-m3.
3. Agglomerative cluster → ~280 sub-clusters; name each from members.
4. Group sub-clusters → emergent categories (one agent per group, holistic, "name the type of mistake"). NOT independent per-feature assignment.
5. Fire rate summed per sub-cluster + per category (reach % + sum-rate %).
6. QC: misfit reconciliation + member verification + coherence bar (≥ the persona method's 0.593-style cutoff).
7. Rebuild atlas: category → sub-cluster → feature, fire rates at each level.

---

## Previous State (2026-05-23)

**Production SAE:** `puzzle_2048_k32_v1` — filtering by `coaching_useful` flag + `detection_accuracy >= 0.6`. 218 features served.

**Puzzle SAE champion:** BTK 2048 k=64 + aux, BA=0.632 — ready to deploy (Queue item 2).

**Blunder SAE winner: MT 2048 k=32** — 1,080 unique coaching labels, 65% label uniqueness, 1.56% median fire rate.

**NEW — Maia 3 SAE v2 (2026-05-24): RETRAINING on correct data.**
- v1 had a critical bug: ~50% of training positions (Black-to-move) had inverted blunder/best labels from the Lichess eval dataset sort. See knowledge.md § "Gemini labeling" for full details.
- v2 uses real-game blunder data (`blunder_positions.json`, 200K positions, already cached) which correctly identifies blunders for both colors.
- **Status:** Maia 3 activation extraction RUNNING on chess-poc (PID 22253, 16/s, ETA ~3h from 21:30 UTC May 24). Output: `maia3_blunder_diff_v2.pt`.
- **Next after activations:** Run `bash /home/ec2-user/SageMaker/run_pipeline.sh` to chain train + profile. Then Gemini CLI labeling locally.

**OLD Maia 3 SAE v1 (2026-05-22):** BTK 2048 k=32, diff pooling, L2 normalized. INVALIDATED by data bug.
- 2007 features labeled (Sonnet 4.6 + thinking). Labels unreliable due to mixed blunder/best-move training.
- Structural analysis showed: 41 hub features, 91% classified into 15 structural categories.
- Geometric python-chess labeling attempted — dead end (too shallow, mostly noise).
- Gemini batch ($0.05 actual cost, April promotional?) produced 5,851 position labels that cover all 2048 features.
- Cost audit: published rates would be ~$67-200 for similar batch. Batch API may have been free during preview.

Full sweep (9 variants trained, 5 labeled):

| Config | Alive | FVU | FR Med | Quality | Unique Labels | Verdict |
|--------|-------|-----|--------|---------|---------------|---------|
| 1024 k=16 | 1,023 | 0.155 | 1.56% | — | — | Too coarse |
| 1024 k=32 | 1,016 | 0.127 | 3.12% | — | — | Too coarse, 71% of 2048 missed |
| **2048 k=32** | **2,031** | **0.115** | **0.87%** | **1,670** | **1,080** | **WINNER** |
| 2048 k=16 | 2,040 | 0.144 | 0.78% | — | — | Unlabeled |
| 2048 k=64 | 2,033 | 0.093 | 2.00% | 2,984 | — | More redundant |
| 4096 k=32 | 4,009 | 0.107 | 0.35% | 3,447 | 1,914 | +834 unique but 44% redundant |
| 4096 k=64 | 4,027 | 0.085 | 0.84% | 2,984 | — | Diminishing returns |
| 4096 k=128 | 4,092 | 0.066 | 2.09% | 2,711 | — | Too many broad features |
| 8192 k=32 | 8,024 | 0.101 | — | — | — | Research only |

**Key findings:**
- Move-token-only (hidden[77]) fixed fire rates: 20-31% all-token → 0.8-3.1% move-token
- 60% high-confidence labels across all variants (up from 27% in old blunder SAE)
- Pairwise Jaccard 0.12-0.19 across variants — SAEs find different decompositions
- Within categories: features are unique (Jaccard <0.5), but labels are the bottleneck (40% get generic names)
- 1024 too small (misses 71% of 2048 coverage), 4096 diminishing returns (44% redundant)
- Top blunder categories: hanging pieces (20%), endgame technique (17%), passed pawns (11%), deflection (12%)

**Repo structure:** Everything in chess-deck-research now. See README.md.

## Beliefs
- [CONFIRMED] BTK is the only viable SAE architecture
- [CONFIRMED] Move-token (hidden[77]) >> mean-pooled or per-token-all
- [CONFIRMED] Puzzle-trained wins over blunder-trained (72% vs 27%)
- [CONFIRMED] Features are subtypes not duplicates (Jaccard ~0)
- [CONFIRMED] Versioned architecture works — swap SAE = new version dir
- [OVERTURNED] ~~k=32 gives right specificity~~ → k=32 too sparse without aux (57% dead). With aux, k=32 works fine (9% dead, 1,864 active).
- [CONFIRMED] Aux loss reduces dead features dramatically (57% → 10% at k=64)
- [CONFIRMED] FEN enrichment improves detection scoring (+0.048 mean BA, +141 STRONG features)
- [CONFIRMED] Judge quality (Haiku vs Sonnet) doesn't matter for detection scoring (+0.006, negligible)
- [CONFIRMED] Enrichment matters more than judge model for detection scoring
- [OVERTURNED] ~~Phase/piece diversity measures polysemanticity~~ → 95% false positive rate. Measures generality not polysemanticity.
- [OVERTURNED] ~~Dead features are bad~~ → Dead = unused capacity. Optimize for active count, not dead %.
- [MEASURED] 2048 k=64 + aux: Sonnet labels + enriched → mean BA 0.632, HOLDS 659, STRONG 325
- [CONFIRMED] Sonnet+thinking labels > Haiku labels: +0.013 BA, +36 STRONG, -67 FAIL
- [CONFIRMED] Aux loss fixes k=32 too: 57% dead → 9% dead (same effect as k=64)
- [MEASURED] 2048 k=32 + aux: 1,864 active, FVU=0.128, c_dec=0.045
- [MEASURED] 4096 k=64 + aux: 3,017 active, FVU=0.092, c_dec=0.035
- [CONFIRMED] k=32 + aux labels much less polysemantic than k=64: 3.5% vs 30.6%.
- [CONFIRMED] But k=64 wins on detection scoring despite higher poly rate (0.632 vs 0.557 BA). Poly ≠ quality.
- [CONFIRMED] 2048 >> 4096 per-feature detection quality. Extra dict capacity doesn't help.
- [MEASURED] Final: 2048 k=64 + aux = BA 0.632, 659 HOLDS, 325 STRONG. Winner.
- [UNTESTED] SAE feature diffs improve coaching output (A/B test needed)
- [CONFIRMED] Move-token-only >> all-token for blunder SAE (fire rate 2% vs 31%)
- [CONFIRMED] Blunder move tokens produce viable SAE structure (0.4-2.1% dead, FVU 0.066-0.115)
- [MEASURED] Pre-topk: 318 features naturally activate, top-64 = 60% energy, top-128 = ~75%
- [OVERTURNED] ~~Puzzles >> blunders for SAE training~~ Previous test was k=32 no-aux all-token. Move-token + aux changes the picture. Labeling pending.
- [UNTESTED] Blunder SAE features are interpretable (labeling will determine)

## Queue

### 0. Sonnet+thinking labeling (DONE)
- 1,872/1,961 parsed. 1,139 mono+high-confidence. 30.6% poly (correlated with uncertainty).
- Detection scoring: mean BA 0.632 (+0.013 vs Haiku labels), 325 STRONG (+36), 293 FAIL (-67)
- Sonnet labels measurably better than Haiku labels. Use Sonnet+thinking going forward.

### 1. Pick k and dict_size (DONE — 2048 k=64 wins)
All 4 variants profiled, labeled (Sonnet+thinking), and scored (Haiku + enriched):

| Config | Mean BA | HOLDS | STRONG | FAIL |
|--------|---------|-------|--------|------|
| **2048 k=64** | **0.632** | **659** | **325** | **293** |
| 4096 k=64 | 0.566 | 566 | 159 | 824 |
| 4096 k=32 | 0.563 | 537 | 155 | 854 |
| 2048 k=32 | 0.557 | 284 | 70 | 515 |

**Winner: 2048 k=64 + aux.** Best on every detection metric. Deploy this.

### 2. Deploy 2048 k=64 + aux as production SAE (NEXT)
Full plan: `lab/chess/website/plans/2026-04-12-deploy-sae-k64.md`
1. Convert weights .pt → .npz (Lambda uses numpy, not PyTorch)
2. Build labels.json (merge Sonnet labels + detection scores + profiles, set coaching_useful)
3. Create version dir `puzzle_2048_k64_v1/` with config.json, labels.json, sae_weights.npz
4. Read k from config.json instead of hardcoding 32 in app.py
5. Update active_version.json, run relabel.py
6. Smoke test locally
7. Deploy CDK

### 3. Blunder-trained SAE experiment (IN PROGRESS — labeling)
- **Hypothesis:** Move-token SAE on blunder moves clusters "what kind of mistake" patterns
- **Data:** 200K blunders (≥200cp loss) from Lichess eval dataset, move-token cache (804MB)
- **Scripts:** `cache_move_token.py`, `train_blunder_sae.py`, `profile_sae.py --move-token-only`
- **All 5 weights on S3** — see `output/S3_INVENTORY.md`
- Pipeline:
  1. ✅ 200K blunders collected from HuggingFace (16.1% hit rate, 13min)
  2. ✅ Move-token cache built (hidden[77] only, 804MB)
  3. ✅ 5 variants trained (2048×{k32,k64} + 4096×{k32,k64,k128}), 8-14s each
  4. ✅ All profiled — fire rates 0.35-3.15% median (all under 5% target)
  5. ✅ Labeling complete: all 3 batches (k=32, k=64, k=128). ~60% high confidence across all.
  6. ✅ Pairwise Jaccard: SAEs find different features (mean best 0.12-0.19)
  7. ✅ Quality filter: 2048 k=32 = 1,670 passing, 1,080 unique labels (65%)
  8. ✅ Within-category analysis: features are unique (Jaccard <0.5) but labels are bottleneck (40% generic)
  9. ✅ Dict size sweep: 1024 too coarse (misses 71%), 4096 diminishing returns (44% redundant)
  10. 🔄 **NEXT: Cluster fire patterns into 20-30 coaching categories**
  11. Relabel with coaching taxonomy (short_label, coaching_advice, theme/subtopic)
  12. Detection scoring on 2048 k=32
- **Winner: 2048 k=32** — best balance of unique labels and quality
- **Presentation problem:** 1,080 features → need 20-30 coaching categories → 5-6 player-facing themes
- **Approach:** Cluster fire patterns (not labels) via cosine similarity + hierarchical clustering. Then name clusters with Sonnet. Categories should map to Heisman's mistake taxonomy.
- See `output/blunder_sae_reasoning.md` for full design rationale.

### 4. Coaching taxonomy
- Cluster 2048 k=32 fire patterns into ~25 coaching subtopics
- Map subtopics to ~6 player-facing themes (piece safety, tactical awareness, endgame play, etc.)
- Relabel features with fixed taxonomy + short labels + coaching advice
- Taxonomy must be stable across SAE architectures (it's about chess, not features)

### 5. Deploy puzzle + blunder SAEs
- Puzzle SAE: 2048 k=64, BA=0.632 — deploy plan at `lab/chess/website/plans/2026-04-12-deploy-sae-k64.md`
- Blunder SAE: 2048 k=32 — deploy alongside after coaching taxonomy is set

### 6. Coaching A/B test
- 50 blunders. Coaching with vs without SAE feature context. Sam rates.

## Constraints
- chess-poc: ml.g6.16xlarge (L4 + 256GB RAM), account 140023406996
- Bedrock calls: account 140023406996, default profile
- Bedrock Batch supports thinking (tested 2026-04-12, needs ≥100 records)
- Opus doesn't support Bedrock Batch

## Pipeline (repeatable)
1. Cache activations (cache_activations.py — run once per 200K puzzles, ~15 min)
2. Train SAE (sweep scripts in chess-deck-research/scripts/sae/)
3. Eval structural metrics (eval_sae_checkpoint.py — dead, L0, FVU, c_dec)
4. Profile (profile_sae.py — top-20 examples per feature, ~5 min per SAE)
5. Enrich FENs (enrich_fens.py — Stockfish + python-chess, cached)
6. Label (batch_label_and_score.py label — Sonnet+thinking via Bedrock Batch)
7. Detection score (batch_label_and_score.py score — Haiku + enriched FENs)
8. Auto-flag coaching_useful (BA ≥ 0.6, FR ≤ 3.0, not polysemantic)

All scripts in `chess-deck-research` repo. Run on notebook via git pull.

## After Maia3 space investigation (May 2026)

### Findings (see knowledge.md for full detail)
- Current v2 SAE: diff is before-after-blunder, NOT blunder-best. Features describe what you played.
- Option-A (repr_best - repr_blunder): tactical clustering is real. Fork gap=0.04, capture gap=0.12.
- An SAE trained on Option-A WILL produce missed-fork/missed-capture/missed-quiet-tactic features.
- cp_loss doesn't cluster in this space — expected, not a bug. Severity = statistic per feature.

### Decision pending from Sam
Build Option-A cache (~3.5hr on chess-poc) and retrain SAE, OR continue finishing the
taxonomy categorization on the current v2 SAE first.

### If rebuilding Option-A cache
Script needed: encode v1 positions (200k, have best_uci + player elos), run Maia3 ONNX
on both resulting boards at player elo, diff mean64, save. Then retrain BatchTopK SAE.

## Current work (2026-05-31) — INVALID (v1-corrupted, see correction at top of file)

### SAE architecture search

> All three SAEs below were trained on the v1 cache (label-inversion bug) and have been deleted.
> Findings unverified. To revisit, rebuild on the v2 cache.

**Goal:** Find SAE where >50% of features have coherent coaching labels.
**Done criterion:** Feature descriptions match what a coach would say ("you hung your bishop," not "insufficient data").

**Three new SAEs built and trained:** (DELETED — v1-corrupted)
- `maia3_option_a_2048_k32.pt` — h[best_to] - h[blunder_to], layer-7 ONNX, 512-dim
- `maia3_board_diff_2048_k32.pt` — mean64(after_best - after_blunder), layer-7 ONNX, 512-dim
- `maia3_l2l7_2048_k32.pt` — concat(L2_mean64_diff, L7_mean64_diff), 79M PyTorch, 2048-dim

**Probe findings:** (derived on v1 data — needs re-verification on v2)
- board_diff best overall for mistake-type separation (mean gap 0.038 across 6 taxonomy categories)
- L2 better for positional mistakes, L7 better for tactical timing — l2l7 captures both
- v2 SAE (current): h[to_sq]-h[from_sq] of blunder — encodes what you played, not what you missed

**In progress:**
- Encoding 18k Opus positions through all 3 SAEs (chess-poc, ~80min remaining)
- After: Sonnet labeling pass using Opus descriptions as source
- After: Eval on 4 real positions from cabbagelover games

**Test positions (output/test_positions.json):**
- bishop_f5: 9...Bf5 → hung piece (463cp)
- queen_h4: 17.dxe4 → left piece undefended (453cp)
- knight_e4_trap: 5...Nxe4 → walked into tactic (331cp)
- qd5_king_exposed: 8...Qd5 → misplayed attack (425cp)

**Decision tree:**
- If l2l7 hits >50% coherent labels AND fires right features on test positions → use l2l7
- If none hit 50% → rebuild on v2 data (same constructions, v2 positions that match the 18k Opus labels)

## OVERNIGHT AUTO-RUN (2026-06-02) — blob-concentration experiments

**Mission:** Two questions, gather data + write report. No deploy/ship.
1. What k minimizes blob concentration while keeping features alive?
2. Does corpus size reduce blobs at fixed k=16?

**Blob metric** (per SAE): calibrate threshold (mean k-th-largest), then on 20k corpus sample:
n_blob = #features firing >10%; pct_top_is_blob; specific-top activation p50.

### Exp 1 — k-sweep [QUEUE]
- [x] k=4 trained — 1794 DEAD (disqualified), FVU 0.46
- [x] k=8 trained — 0 dead, FVU 0.37
- [x] k=16 trained — 32 blobs, 77% top-is-blob, spec 0.25, FVU 0.29
- [x] k=32 trained — 89 blobs, 92% top-is-blob, spec 0.17, FVU 0.22
- [ ] k=12 train + blob analysis
- [ ] k=24 train + blob analysis
- [ ] blob analysis on k=4, k=8 (trained, not yet analyzed)
- [ ] assemble k-sweep curve: k vs (n_blob, pct_top_blob, spec_act, n_dead, FVU)

### Exp 2 — corpus-size sweep at fixed k=16 [QUEUE]
- [ ] subsample dedup cache to 42k / 84k / 126k / 168k
- [ ] train k=16 on each (same hyperparams, 200ep)
- [ ] blob analysis on each → does n_blob grow as data shrinks?
- [ ] verdict: if monotonic increase with less data → more data would help; if flat → architecture not data

### Report
- [ ] write output/blob_experiments_report.md with both curves + verdict

## OVERNIGHT AUTO-RUN RESULTS (2026-06-02 ~02:00) — DONE

Both experiments complete. Full report: `output/blob_experiments_report.md`.
- **Exp1 (k-sweep):** blob concentration monotonic in k. k=4 dead(1800), k=8 sweet spot
  (12 blobs, 0 dead, spec 0.30), k=16 (32 blobs), k=32 (89 blobs). k is THE lever.
- **Exp2 (corpus-size):** weak. blobs ~32-41 across 42k→168k (4× data). NOT the driver.
- **Verdict:** blobs are a k problem, not a data problem. For specificity → use k=8 not k=16.
  1M corpus still worth it for coverage, won't fix blobs.

**OPEN QUESTION for Sam (do not auto-decide):** Are blobs actually BAD? Never verified the
high-freq features are incoherent — they might be useful coarse signal in a coarse→specific
hierarchy. If coherent, the k=8 recommendation flips (keep blobs, label them honestly by fire
rate, filter at display). Needs Sam's call + the f897-style board check on f101/f1487 at correct threshold.

**Weights produced (notebook, NOT in S3):** btk_2048_k{4,8,12,24}_v2_weights.pt,
btk_2048_k16_{42,84,126}k.pt. Upload survivors to S3 if any get adopted.

**Still running (independent screen):** k=32 Pass-2 + k=16 straggler-fill labeling.

## 2026-06-02 late — audit + two bugs found

**L0 / threshold-elo bug (Sam caught):** test positions fire ~20 features at k=8 (should be 8),
~36 at k=16 (should be 16). Cause: threshold calibrated on corpus (encoded at real player elo ~1500)
but test positions encoded at elo 2600 → larger activation magnitude → too many features clear the
fixed threshold. Implication: ALL test-position diagnostics this session (atlas, k8-vs-k16 HTMLs)
OVER-COUNTED fired features. Fix for any coaching inference: threshold must match encoding elo
(encode test positions at corpus-elo, or recalibrate per-elo). The label AUDIT is unaffected — it
reads each feature's top-10 CORPUS positions, not test firings.

**Correct audit method (Sam's framing):** don't enrich the whole corpus (113k unenriched — too big,
not needed). Audit each feature against its OWN top-10 corpus positions + deep signature stats.
Elo-safe (corpus positions at real elo), threshold-free (top-k by activation). Running now on the 54
labeled features. Next: if constrained labels grade well, scale to top-1k features (label + audit).

**best_uci is in cache metadata** (100% coverage) — use it for best-move, not the enrichment cache
(only 27% coverage). Fixed audit_data2.py to pull from metadata.
