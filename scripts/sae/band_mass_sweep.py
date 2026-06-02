"""Where does the model's activation MASS live, as a function of k?

The blob-mass table (mass>5%, mass>10%) is monotonic in k and so can't show a sweet
spot. The quantity that *can* peak is the share of total activation mass that lands in
the USEFUL band (features firing 0.1%-5% of the time) — specific-but-not-rare features.
Low k starves it (half the dict dead, mass crammed into survivors+blobs); high k starves
it too (blobs eat an ever-larger share). This script measures that band directly, plus a
full fire-rate histogram, all on z-score-only models.

Run: python band_mass_sweep.py   (on chess-poc, from ~/SageMaker)
"""
import torch, numpy as np, torch.nn.functional as F, os
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)

# fire-rate bins (fraction of corpus a feature fires on) and labels
EDGES = [0.0, 1e-9, 0.001, 0.01, 0.05, 0.10, 1.01]
BINLBL = ['dead', '<0.1%', '0.1-1%', '1-5%', '5-10%', '>10%']
print(f"{'k':>4s} | " + " ".join(f"{l:>7s}" for l in BINLBL) + " | counts: " + " ".join(f"{l:>5s}" for l in BINLBL))
print(f"{'':>4s} | " + " (% of total activation mass per bin)" )
print('-' * 110)

rows = []
for tag, kk in [('k4', 4), ('k8', 8), ('k12', 12), ('k16', 16), ('k32', 32)]:
    wp = BASE + f'/output/maia3_sae/btk_2048_{tag}_nol2.pt'
    if not os.path.exists(wp):
        print(f"{tag}: not trained"); continue
    sd = torch.load(wp, map_location='cpu', weights_only=False)['state_dict']
    # calibrate eval threshold = mean of k-th largest activation per position (not BatchTopK at eval)
    kth = []
    for i in range(0, 40000, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
        kth.append(torch.topk(z, kk, 1).values[:, -1].numpy())
    th = float(np.concatenate(kth).mean())
    mass = np.zeros(2048); fire = np.zeros(2048)
    for i in range(0, N, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
        za = z * (z > th)
        mass += za.sum(0); fire += (za > 0).sum(0)
    fire /= N; g = mass.sum()
    # mass + count per bin
    massbin, cntbin = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (fire >= lo) & (fire < hi) if lo > 0 else (fire == 0)
        if lo == 1e-9: sel = (fire >= lo) & (fire < hi)
        massbin.append(100 * mass[sel].sum() / g); cntbin.append(int(sel.sum()))
    band_mass = massbin[2] + massbin[3]          # 0.1-1% + 1-5%  = useful band
    band_cnt = cntbin[2] + cntbin[3]
    nondead = 2048 - cntbin[0]
    rows.append((tag, band_mass, band_cnt, nondead, massbin[5]))
    print(f"{tag:>4s} | " + " ".join(f"{m:6.1f}%" for m in massbin) +
          " | " + " ".join(f"{c:5d}" for c in cntbin))

print('-' * 110)
print(f"\n{'k':>4s}  {'BAND mass(0.1-5%)':>17s}  {'band feats':>10s}  {'non-dead':>8s}  {'blob mass(>10%)':>15s}")
for tag, bm, bc, nd, blob in rows:
    print(f"{tag:>4s}  {bm:16.1f}%  {bc:10d}  {nd:8d}  {blob:14.1f}%")
# sweet spot = max band mass subject to non-dead being healthy
best = max(rows, key=lambda r: r[1])
print(f"\nmax useful-band mass: {best[0]} ({best[1]:.1f}% of activation in 0.1-5% features, {best[2]} such features, {best[3]} non-dead)")
