#!/bin/bash
# Wait for maia_best build to finish (DONE marker), then run full v2 pipeline.
cd /home/ec2-user/SageMaker
echo "=== waiting for maia_best_200k.json DONE marker ==="
while ! grep -q "^DONE " maia_best.log 2>/dev/null; do sleep 30; done
echo "=== maia_best done, starting pipeline $(date) ==="
bash run_all_v2.sh 2>&1
echo "=== PIPELINE COMPLETE $(date) ==="
