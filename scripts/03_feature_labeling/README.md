# 03 — Feature Labeling

Name each live SAE feature for the ONE recurring chess mistake it fires on, ground the name
in objective signals, verify coherence, and organize into a browsable mistake taxonomy.

Inputs come from stage 02 (per-position Opus analyses) and the SAE profiler (`*_profiles.json`,
top-activating positions per feature). All scripts are standalone CLIs — run on `chess-poc`
(inputs live there) or locally against files in `output/`.

## The lesson that shaped this stage

SEE (static exchange eval) is **single-ply** — it sees the immediate capture/recapture and
nothing else. It is blind to trapped pieces, 2-move tactics, and zwischenzugs. Treat every
SEE-derived field (`best_wins_material`, `best_is_capture`, `material_kind`) as an objective
*floor*, not ground truth.

The original labeler fed Opus only ~2.5 of the 7 per-position fields it had — it dropped
`best_moves_analysis` (what the player should have played) and `move_intent`. Without the best
move, you cannot tell "hung a piece" from "missed winning a piece". Result: half the dictionary
was labeled in the wrong *direction* (passive/hung instead of missed-win). The fix — feed all
fields — is `relabel_all_fields.py`. On the d2048_k6 dictionary it flipped 1900/2035 labels;
`missed_win` turned out to be the single largest category (1020 of 2035).

## CURRENT pipeline (d2048_k6) — run in this order

All on chess-poc except render. Model for labeling = **opus-4-8 with adaptive thinking + xhigh
effort** (set `EFFORT=xhigh`; matches the interactive Claude Code config).

| # | Script | What it does | Out |
|---|--------|--------------|-----|
| 1 | `compute_feature_see_stats.py` | Per-feature SEE signature (normalized cohorts ≥0.7/0.8·max): moved/captured piece, net-material, eval trajectory, phase. Objective floor. | `see_stats_{model}.json` |
| 2 | `build_peak_median_profiles.py` | Encode corpus → per feature, 10 PEAK + 10 MEDIAN (p40-60) Opus-covered boards. The median is what stops piece over-specification. | `peak_median_profiles_{model}.json` |
| 3 | **`relabel_v7_peakmedian.py`** | **CURRENT labeler.** Labels each feature from peak+median boards + the prior label as head-start. Chip form "Core mistake (often queen / major piece)"; label narrates top→median broadening. Emits `confidence` + `review`. `EFFORT=xhigh`. | `relabel_v7_{model}.json` |
| 4 | `assign_v3.py` | Assign labeled features to the 12-category v3 taxonomy (`buckets_v3_*.json`). Allows `unassignable`. NO rules block — buckets + evidence + self-inflicted/omission axis. | `feature_buckets_{...}.json` |
| 5 | `cluster_llm.py` | Within each category, Opus groups features into ~10-20 natural coaching clusters (merges near-dups). | `feature_clusters_{...}.json` |
| 6 | `build_leaf.py` | Flatten clusters → per-feature `{bucket, sub}` leaf the atlas reads. FOLDS >5% blob features into one `⚠ Coarse detectors` sub per bucket (they're broad material-lost detectors, not specific tactics). | `feature_leaf_{...}.json` |
| 7 | `profiles_to_atlas.py` | Adapt peak+median profiles → the `{examples, fire_rate}` profiles + `best_uci_map` the atlas needs (peak first, extracts inline best move for the green arrow). | `atlas_profiles_{...}.json` + `best_uci_map_{...}.json` |
| 8 | `audit_clusters.py` | Per-category audit: place orphans + flag features whose label contradicts their category/cluster. | `cluster_audit_{...}.json` |
| 9 | `render_atlas_v3.py` (local) | SPA atlas: category → cluster → feature, boards client-side from FEN, chess.com links. `--dict-label` tags the title. ~1-2MB. | `atlas/atlas_*.html` |

**Dictionaries built with this pipeline:** `d2048_k6` (2033 features, median consistency 63) and
`d2048_k4` (1148 features, median consistency 60). k6 is cleaner on every quality metric — fewer
features (1148 vs 2033) covering the same corpus makes k4 marginally *more* polysemantic, not less.
The `buckets_v3_d2048_k6.json` category definitions are dictionary-independent and reused for both.

## Labeler version lineage (each fixed a validated defect — see plan.md / log.md)

`relabel_all_fields.py` (v1, biased) → `relabel_all_fields_v2_neutral.py` (debiased) →
`relabel_v3_5word_mech.py` (5-word mechanism chips) → (v4 refutation+result-framing, REJECTED) →
`relabel_v5_refutation_conf.py` (refutation fed + confidence + opus-4-8 xhigh) →
`relabel_v6_secondpass.py` (re-run flagged features with prior guess shown) →
**`relabel_v7_peakmedian.py` (current — peak+median boards fix piece over-specification).**

**THE LESSON:** labeling from top-10 (p99 peak) over-specifies the piece (peak boards are
piece-homogeneous). A feature's true identity is at its MEDIAN activation, which is broader.
v7 feeds both bands so "Hangs queen to knight fork" (peak only) becomes "Hangs to knight fork
(often queen)" (peak+median). Same root cause as the earlier missed-vs-hung and knight-75% bugs.

## Game application

`encode_game_blunders.py` — encode a real game's blunders through Maia3 → L7-diff → k6 SAE →
taxonomy (replicates the corpus build exactly). Read per-position diagnoses at the CLUSTER level,
not the feature chip (a feature labeled "queen fork" can fire on a rook fork — out-of-distribution
structural match). See `docs`/`/analyze-game` skill.

## Diagnostics / one-offs

`deep_signature.py`, `coherence_depth.py`, `decoder_overlap.py`, `firing_overlap.py` (redundancy
tests — proved blob features are distinct, not duplicates), `audit_direction_stockfish.py`
(hang-vs-miss from played-vs-best capture), `render_feature_list.py`. The v1–v6 labelers,
`assign_to_buckets.py`, `subbucket_and_rollup.py`, `cluster_taxonomy.py` (embedding clustering —
dead end), `emergent_categories.py` are kept for provenance; not the current path.
