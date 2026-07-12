cd /home/ec2-user/SageMaker
CACHE=chess-stage-a/cache/maia3_l7only_v2_dedup.pt
OUT=/home/ec2-user/SageMaker/jr_canon_out; mkdir -p $OUT
RES=$OUT/canon_results.jsonl; : > $RES
PY=~/anaconda3/envs/pytorch_p310/bin/python
for D in 512 256; do
  $PY train_jr_canonical.py --cache $CACHE --dict-size $D --target-l0 8 --epochs 60 \
    --tag jr${D}_k8 -o $OUT/jr${D}_k8.pt --results $RES 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'ep60|ep50|DONE|target_l0'
done
echo CANON_ALL_DONE
