cd /home/ec2-user/SageMaker
CACHE=chess-stage-a/cache/maia3_l7only_v2_dedup.pt
OUT=/home/ec2-user/SageMaker/jr_sweep_out
mkdir -p $OUT
RES=$OUT/jr_sweep_results.jsonl
: > $RES
# Winning region from the diagnostic: init_threshold near the pre-act median (0.4) puts ~1300 features
# in the 1-5% band. Sweep thr around there (death boundary was ~0.5) x two L0 penalties, bw=0.2,
# 60 epochs (theta climbs slowly, more epochs = cleaner separation). dict=2048 to match k6/v7 lineage.
for THR in 0.30 0.40 0.50; do
  for L0 in 0.02 0.06; do
    TAG="jr_thr${THR}_l0${L0}"
    echo "### $TAG"
    python3 train_jr_sweep.py --cache $CACHE --dict-size 2048 --epochs 60 \
      --bandwidth 0.2 --init-threshold $THR --l0-coeff $L0 \
      --tag $TAG -o $OUT/${TAG}.pt --results $RES 2>&1 | grep -E 'ep60|DONE'
  done
done
echo "SWEEP_ALL_DONE"
