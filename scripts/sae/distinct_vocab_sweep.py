"""Distinct interpretable vocabulary vs k — the metric that actually matters.

Dead-count is a dict_size artifact, not a quality signal (a dead feature never fires,
costs nothing, isn't a blob). What matters for a coaching vocabulary:
  - how many DISTINCT features (decoder twins = fake vocabulary, must dedup)
  - that are SPECIFIC-not-rare (the 0.1-5% band)
  - and carry the model's activation mass (vs leaking into blobs)

This measures, for each z-score model, restricted to BAND features (fire 0.1-5%):
  - n_band                : raw count in the band
  - n_distinct@0.9/0.8/0.7: count after greedy dedup of decoder twins at cosine thresh
  - twin_frac             : fraction of band features that are near-duplicates (cos>0.9)
  - band_mass             : % of total activation mass the band carries
So we can see if k16's extra band features are REAL distinct directions or duplicates.

Run on chess-poc from ~/SageMaker:  python distinct_vocab_sweep.py
"""
import torch, numpy as np, torch.nn.functional as F, os
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)

def greedy_distinct(W, idx, thresh):
    """Greedy dedup: among features `idx`, count clusters where any pair with cos>thresh
    is merged. Returns number of distinct clusters (representatives)."""
    if len(idx) == 0: return 0
    V = W[idx]                                   # [m, d]
    V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
    S = V @ V.T                                  # cosine matrix [m, m]
    m = len(idx); kept = np.ones(m, bool)
    S = S.numpy(); np.fill_diagonal(S, 0)
    for i in range(m):
        if not kept[i]: continue
        # absorb later features too similar to i
        dup = (S[i] > thresh) & kept
        dup[:i+1] = False
        kept[dup] = False
    return int(kept.sum())

print(f"{'k':>4s} | {'n_band':>6s} {'dist@.9':>7s} {'dist@.8':>7s} {'dist@.7':>7s} {'twin%@.9':>8s} | {'band_mass':>9s} {'blob_mass':>9s}")
print('-' * 88)
for tag, kk in [('k4',4),('k6',6),('k8',8),('k10',10),('k12',12),('k16',16),('k32',32)]:
    wp = BASE + f'/output/maia3_sae/btk_2048_{tag}_nol2.pt'
    if not os.path.exists(wp):
        print(f"{tag:>4s} | not trained yet"); continue
    sd = torch.load(wp, map_location='cpu', weights_only=False)['state_dict']
    # decoder: find the [2048, 1024] weight (unit-norm rows = feature directions)
    W_dec = sd.get('W_dec')
    if W_dec is None:  # fall back to encoder direction if needed
        W_dec = sd['W_enc'].T
    if W_dec.shape[0] != 2048:
        W_dec = W_dec.T
    # calibrate threshold + fire rates + mass
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
    band = np.where((fire >= 0.001) & (fire < 0.05))[0]
    band_mass = 100 * mass[band].sum() / g
    blob_mass = 100 * mass[fire > 0.10].sum() / g
    d9 = greedy_distinct(W_dec, band, 0.9)
    d8 = greedy_distinct(W_dec, band, 0.8)
    d7 = greedy_distinct(W_dec, band, 0.7)
    twin = 100 * (len(band) - d9) / max(1, len(band))
    print(f"{tag:>4s} | {len(band):>6d} {d9:>7d} {d8:>7d} {d7:>7d} {twin:>7.0f}% | {band_mass:>8.1f}% {blob_mass:>8.1f}%")
