"""Does shrinking dict_size recover low-k band-concentration WITHOUT the dead-slot waste,
or does it just recreate blobs?

Sam's hypothesis: k4/dict2048 had 67% of mass in the useful 0.1-10% band but 951 dead
slots. Shrink to dict1024 and maybe you keep the concentration with no waste.
Risk: mean fire rate = k/dict, so halving dict DOUBLES every feature's fire rate — the
deleted specialists' mass may resurface as blobs.

This prints, for each (dict, k), the same band/blob/dead/Gini picture so 1024 and 2048
sit side by side. Band = 0.1-10% (what Sam wants maximized among ALIVE dictionaries).

Run on chess-poc from ~/SageMaker:  python dict_size_compare.py
"""
import torch, numpy as np, torch.nn.functional as F, os
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)

def gini(v):
    v = np.sort(v[v >= 0]); n = len(v)
    if n == 0 or v.sum() == 0: return 0.0
    cum = np.cumsum(v); return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)

print(f"{'model':>12s} | {'dead':>5s} {'<0.1%':>5s} {'band0.1-10%':>11s} {'blob>10%':>8s} | {'rawGini':>7s} {'FVU':>6s} {'band_feats':>10s}")
print('-' * 88)
MODELS = []
for dct in (2048, 1024):
    for kk in (4, 6, 8):
        MODELS.append((f'btk_{dct}_k{kk}_nol2', dct, kk))

for name, dct, kk in MODELS:
    wp = BASE + f'/output/maia3_sae/{name}.pt'
    if not os.path.exists(wp):
        print(f"{name:>12s} | not trained"); continue
    sd = torch.load(wp, map_location='cpu', weights_only=False)['state_dict']
    D = sd['W_enc'].shape[1]
    kth = []
    for i in range(0, 40000, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
        kth.append(torch.topk(z, kk, 1).values[:, -1].numpy())
    th = float(np.concatenate(kth).mean())
    mass = np.zeros(D); fire = np.zeros(D); rawmass = np.zeros(D); sse = 0.0; sst = 0.0; xm = x.mean(0)
    for i in range(0, N, 8192):
        xb = x[i:i+8192]; z = F.relu((xb - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
        zt = torch.topk(z, kk, 1); zk = torch.zeros_like(z).scatter_(1, zt.indices, zt.values)
        recon = zk @ sd['W_dec'] + sd['b_dec']
        sse += ((xb - recon) ** 2).sum().item(); sst += ((xb - xm) ** 2).sum().item()
        zn = z.numpy(); rawmass += zn.sum(0); za = zn * (zn > th); mass += za.sum(0); fire += (za > 0).sum(0)
    fire /= N; g = mass.sum()
    dead = int((fire == 0).sum()); nd = 100 * mass[(fire > 0) & (fire < 0.001)].sum() / g
    band = (fire >= 0.001) & (fire < 0.10); band_mass = 100 * mass[band].sum() / g
    blob = 100 * mass[fire >= 0.10].sum() / g
    fvu = sse / sst
    label = f"d{dct}_k{kk}"
    print(f"{label:>12s} | {dead:>5d} {nd:>4.0f}% {band_mass:>10.1f}% {blob:>7.1f}% | {gini(rawmass):>7.3f} {fvu:>6.3f} {int(band.sum()):>10d}")
