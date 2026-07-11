cd /home/ec2-user/SageMaker
CACHE=chess-stage-a/cache/maia3_l7only_v2_dedup.pt
# Hypothesis: bw=0.02 is 28x smaller than the pre-act median (0.565) -> starved threshold gradient.
# Test bandwidth (window width) x init_threshold (start position) at enough epochs for theta to move.
for BW in 0.02 0.2 0.5; do
  for THR in 0.06 0.4; do
    python3 train_jr_sweep.py --cache $CACHE --epochs 40 --l0-coeff 0.02 \
      --bandwidth $BW --init-threshold $THR --tag "diag_bw${BW}_thr${THR}" \
      -o /tmp/diag_bw${BW}_thr${THR}.pt --results /tmp/jr_diag.jsonl 2>&1 | grep -E 'ep40|DONE'
  done
done
echo "ALL_DIAG_DONE"
