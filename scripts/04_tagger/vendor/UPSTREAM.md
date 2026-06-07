# Vendored from lichess-puzzler

Source: https://github.com/ornicar/lichess-puzzler  (tagger/)
Commit: c188837cd2411d5c17d4f33c59ac38a8722d694f
Pulled: 2026-06-07
Files: cook.py, util.py, model.py, zugzwang.py

**DO NOT EDIT.** Pure rule-based python-chess tactic/mate tagger. We use it unchanged via
../cook_adapter.py. To update: re-pull and re-run the regression tests in ../tests/.
Built for python-chess 1.3; we run 1.11 (imports + validated on our data, watch attack/pin API).
