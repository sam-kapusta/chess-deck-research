# SAE Research Scripts

Canonical scripts for the encoder SAE pipeline. **Do not create new scripts in /tmp — check here first.**

## Pipeline (run in order on SageMaker notebook)

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `cache_encoder_activations.py` | Game analysis JSONs | `encoder_activation_cache.pt` |
| 2 | `train_sweep.py` | Cached activations | SAE checkpoints (`.pt`) |
| 3 | `sweep_sae_agreement.py` | Checkpoints + cache | `feature_agreement.json` |
| 4 | `t1_t2_eval.py` | Checkpoints + cache | T1/T2 metrics per checkpoint |
| 5 | `coherence_sweep.py` | Checkpoints + cache | Coherence scores |
| 6 | `temporal_polysemantic.py` | Checkpoints + cache | Temporal + polysemanticity scores |
| 7 | `lichess_rich_profiler.py` | SAE checkpoint + Lichess dataset | `*_profiles.json` (with FENs) |
| 8 | `meanpool_profiler.py` | SAE checkpoint + Lichess dataset | `*_meanpool_profiles.json` |
| 9 | `label_sae_features.py` | Any `*_profiles.json` | `*_labels.json` |

## Labeling consistency

`label_sae_features.py` is the **canonical labeling script**. All feature labels must come from this script. If you change the prompt or parsing logic, re-label everything.

Key implementation detail: Sonnet wraps output fields in markdown bold (`**LABEL:**`). The parser strips `**` before matching. Do not remove this.

## Other scripts (pre-existing)

| Script | Purpose |
|--------|---------|
| `classify_coaching_quality.py` | Rate coaching output quality |
| `llm_classify_borderline.py` | LLM classification of borderline moves |
| `prepare_training_data.py` | Prepare SAE training data |
| `scrape_lichess_studies.py` | Scrape Lichess studies |
| `upload_to_sais.sh` | Upload files to SAIS notebook |
