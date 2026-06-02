"""Is the 'blob' problem real, or a threshold artifact of k?

The >10%-fire-rate blob definition is NOT comparable across k: total fire mass = k, so
mean fire rate = k/2048. At k8 that's 0.39%, at k32 1.56% — so the absolute >10% line is
25x the mean at k8 but only 6.4x at k32. Higher k crosses it mechanically. So blob-mass
rising with k may be an artifact, not worse concentration.

Threshold-free test: Gini coefficient of the per-feature activation-mass distribution
(0 = mass spread evenly, 1 = all mass in one feature). And the rank-based Lorenz points:
share of total mass held by the top 1%, 5%, 10% of features BY MASS. These are
k-independent. If Gini rises with k, concentration genuinely worsens. If flat, the blob
curve was a mirage and higher k is strictly better (more distinct vocab, same real spread).

Run on chess-poc from ~/SageMaker:  python gini_sweep.py
"""
import torch, numpy as np, torch.nn.functional as F, os
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)

def gini(v):
    v = np.sort(v[v >= 0]); n = len(v)
    if n == 0 or v.sum() == 0: return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)

print(f"{'k':>4s} | {'gini(mass)':>10s} {'gini(fire)':>10s} | {'top1%':>6s} {'top5%':>6s} {'top10%':>6s} | {'mean_fire':>9s} {'>10x_mean':>9s}")
print('-' * 86)
for tag, kk in [('k4',4),('k6',6),('k8',8),('k10',10),('k12',12),('k16',16),('k32',32)]:
    wp = BASE + f'/output/maia3_sae/btk_2048_{tag}_nol2.pt'
    if not os.path.exists(wp):
        print(f"{tag}: not trained"); continue
    sd = torch.load(wp, map_location='cpu', weights_only=False)['state_dict']
    kth = []
    for i in range(0, 40000, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
        kth.append(torch.topk(z, kk, 1).values[:, -1].numpy())
    th = float(np.concatenate(kth).mean())
    mass = np.zeros(2048); fire = np.zeros(2048)
    for i in range(0, N, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
        za = z * (z > th); mass += za.sum(0); fire += (za > 0).sum(0)
    fire /= N; g = mass.sum()
    ms = np.sort(mass)[::-1]; cms = np.cumsum(ms) / g
    top1 = 100 * cms[int(0.01 * 2048)]; top5 = 100 * cms[int(0.05 * 2048)]; top10 = 100 * cms[int(0.10 * 2048)]
    mean_fire = fire.mean()
    # relative-blob: features firing >10x the mean fire rate (k-normalized blob def)
    relblob_mass = 100 * mass[fire > 10 * mean_fire].sum() / g
    print(f"{tag:>4s} | {gini(mass):>10.3f} {gini(fire):>10.3f} | {top1:>5.0f}% {top5:>5.0f}% {top10:>5.0f}% | {mean_fire*100:>8.2f}% {relblob_mass:>8.0f}%")
