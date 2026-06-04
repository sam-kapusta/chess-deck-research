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

## Run order

| # | Script | What it does | Out |
|---|--------|--------------|-----|
| 1 | `compute_feature_see_stats.py` | Per-feature SEE signature over top-N activating positions, in normalized cohorts (≥0.7·max core, ≥0.8·max peak): moved/captured piece, net-material kind (trade/loses/hangs/safe), missed-material, eval trajectory, phase. The objective floor. | `see_stats_{model}.json` |
| 2 | `deep_signature.py` | Optional deeper multi-granularity signature (hang granularity, best-move type concentration). Diagnostic. | `deep_sig_{model}.json` |
| 3 | `relabel_all_fields.py` | **The canonical labeler.** Opus names each feature from its top-10 positions with ALL per-position fields (blunder_summary + best_moves_analysis + intent) + a one-line SEE floor. Single call per feature; emits `consistency` 0-100 — features ≤70 are flagged for review. | `relabel_allfields_{model}.json` |
| 4 | `coherence_depth.py` | Objective (no-LLM) check: does a feature's dominant axis hold from peak down to 0.7·max, or is it peak-only? Verdicts: holds_to_0.7 / holds_to_0.8 / peak_only. | `coherence_depth_{model}.json` |
| 5 | `assign_to_buckets.py` | Assign labeled features to the 11-bucket mistake taxonomy. Allows `unassignable` (recorded, not force-fit). | `feature_buckets_{model}.json` |
| 6 | `subbucket_and_rollup.py` | Split buckets into sub-categories, roll up by fire-rate coverage. | `feature_leaf_{model}.json` |
| 7 | `audit_buckets.py` | Objective audit: does a feature's SEE signature contradict its assigned bucket? Flags violations. | stdout / audit JSON |

## Rendering (eyeball checks)

| Script | What |
|--------|------|
| `render_feature_list.py` | Rich HTML for an arbitrary feature list — SEE signature (both cohorts), coherence verdict, top-N boards clickable to chess.com, optional per-board Opus analysis. `--filter hardcut` isolates the clear blobs. |
| `render_taxonomy_tree.py` | The full taxonomy as a collapsible bucket → sub → feature tree, each feature with example boards (played = red arrow, Maia top = green). |

## `consistency` is the review signal

`relabel_all_fields.py` emits `consistency` per feature (how many of the 10 boards fit the named
mistake). A genuinely mixed feature comes back ~80, not 100. Features ≤70 are flagged — review
those rather than re-running with consensus voting. Watch for labels that over-fit a specific
move (e.g. "Missed Nxd4"); those won't generalize and belong in the flagged set.

## Predecessors (superseded)

`label_features_integrated.py` is the previous labeler — kept for provenance of the
`feature_labels_integrated_*.json` files, but superseded by `relabel_all_fields.py`. Older
one-off labelers (gemini, pass2, btk, constrained, synthesize) remain in `scripts/labeling/`.
