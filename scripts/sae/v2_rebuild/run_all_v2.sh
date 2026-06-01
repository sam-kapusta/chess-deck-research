#!/bin/bash
# Chains the full v2 SAE pipeline after maia_best_200k.json exists.
# Caches -> l7 slice -> train 4 SAEs (k=16). Logs each stage.
set -e
cd /home/ec2-user/SageMaker
SAE=chess-stage-a/output/maia3_sae
CACHE=chess-stage-a/cache
log(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }

log "STAGE 1: build option_a cache"
python3 build_option_a_v2.py 2>&1 | tail -40

log "STAGE 2: build board_diff cache"
python3 build_board_diff_v2.py 2>&1 | tail -40

log "STAGE 3: build l2l7 cache (GPU)"
python3 build_l2l7_v2.py 2>&1 | tail -40

log "STAGE 4: slice l7only from l2l7 (second 1024 dims)"
python3 -c "
import torch,numpy as np
d=torch.load('$CACHE/maia3_l2l7_concat_v2.pt',map_location='cpu',weights_only=False)
A=d['activations'].numpy()[:,1024:]  # L7 half
torch.save({'activations':torch.tensor(A),'mean':A.mean(0),'std':A.std(0),'metadata':d['metadata'],
  'config':{'construction':'L7_mean64_diff only (sliced from l2l7)','source':'v2+maia_best@2600','n':len(A)}},'$CACHE/maia3_l7only_v2.pt')
print('l7only sliced',A.shape)
"

log "STAGE 5: train 4 SAEs at k=16"
python3 train_sae_v2.py $CACHE/maia3_option_a_diff_v2.pt $SAE/maia3_option_a_v2_2048_k16.pt 16 2>&1 | tail -12
python3 train_sae_v2.py $CACHE/maia3_board_diff_v2.pt   $SAE/maia3_board_diff_v2_2048_k16.pt 16 2>&1 | tail -12
python3 train_sae_v2.py $CACHE/maia3_l2l7_concat_v2.pt   $SAE/maia3_l2l7_v2_2048_k16.pt 16 2>&1 | tail -12
python3 train_sae_v2.py $CACHE/maia3_l7only_v2.pt        $SAE/maia3_l7only_v2_2048_k16.pt 16 2>&1 | tail -12

log "STAGE 6: eval 4 SAEs on test positions -> HTML"
python3 eval_v2_html.py 2>&1 | tail -20

log "ALL DONE"
ls -la $SAE/*_v2_2048_k16.pt
ls -la eval_v2.html eval_v2_results.json
