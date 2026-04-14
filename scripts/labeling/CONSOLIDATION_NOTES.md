# Labeling Script Consolidation Notes

## What to fold into label.py from old scripts

### Must have (from auditor analysis):

1. **Better labeling prompt** (from `label_sae_features.py`):
   - Good/bad example in prompt
   - Game count stats from baselines ("fires in 41% of games = too broad")
   - "Could a student practice this?" test for coaching_useful
   - Longer category descriptions

2. **Detection scoring** (from `detection_scoring.py`):
   - BA computation (TPR + TNR / 2)
   - Negative sampling from FEN pool
   - HOLDS/WEAK/FAILED thresholds
   - `print_report()` function

3. **FEN enrichment** (from `enrich_fens.py`):
   - python-chess tactical analysis (8 patterns, no engine needed)
   - Stockfish engine pool (persistent, queue-based)
   - Disk cache (MD5 hash of FEN)
   - Two-layer: instant python-chess + parallel Stockfish

4. **Robust parsing** (from `batch_label_and_score.py`):
   - Gzip output handling
   - modelOutput as string or dict
   - Multi-format detection response parsing
   - Assistant prefill `"["` for arrays

5. **Thread-local Bedrock clients** (from `label_sae_features.py`):
   - One client per thread
   - Exponential backoff retries

6. **Merge strategy** (from `label_sae_features.py`):
   - Preserve examples/stats when relabeling
   - Only overwrite label fields

7. **Progress reporting** (from both):
   - Live category distribution
   - Incremental save every 100 features

### Nice to have:
- Per-token square mapping (only for per-token SAEs)
- ASCII board in prompt
- temperature=0 for deterministic labels

## Scripts to archive

Move to `scripts/archive/` (not delete — preserve git history):
- `data_prep/label_sae_features.py`
- `encoder/label_features_v2.py`
- `evaluation/batch_label_and_score.py`
- `evaluation/exp35_label_512_k8.py`
- `evaluation/label_blunder_coaching.py`
- `evaluation/label_breakdown.py`
- `evaluation/llm_classify_borderline.py`
- `evaluation/parse_batch_results.py`
- `evaluation/relabel_weak_features.py`
- `evaluation/analyze_game_saes.py`
- `sae/interpret_encoder_sae.py`
- `sae/label_by_concepts.py`
- `sae/profile_and_label_all.py`
- `sae/profile_sae.py`
- `encoder/profile_k16_k64.py`
- `sae/analyze_game_sae.py`

## Scripts to keep (different purpose):
- `evaluation/detection_scoring.py` — fold into label.py as `detect` subcommand
- `evaluation/enrich_fens.py` — fold into label.py as enrichment option
- `evaluation/classify_coaching_quality.py` — keep separate
- `evaluation/quality_filter.py` — keep separate
