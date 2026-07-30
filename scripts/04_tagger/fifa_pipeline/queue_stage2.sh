#!/bin/bash
# Wait for the corpus analysis to finish, then run the FIFA stage-2 re-detect.
# Both are Stockfish-bound on the same 64-core box (load was ~120 with the corpus job alone),
# so running them concurrently would just make each slower. Serialize instead.
cd ~/SageMaker/fifa_blitz || exit 1

echo "$(date -u +%H:%M:%S) waiting for analyze_corpus to finish..."
while pgrep -f analyze_corpus.py > /dev/null; do sleep 60; done
echo "$(date -u +%H:%M:%S) corpus analysis done; starting stage 2 re-detect"

# args: <in> <out_enrich> <out_sweep> <nproc>. All four are positional with defaults, so a
# placeholder in slot 3 would silently create a junk file named after it.
python3 -u redetect_positions_d16.py \
  fifa_blunders_all.json \
  fifa_enrich.json \
  fifa_sweep.json \
  60 > redetect.log 2>&1
echo "$(date -u +%H:%M:%S) stage 2 exit=$?"
