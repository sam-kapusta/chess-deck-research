#!/usr/bin/env python3
"""Quick sweep: 4096 k=64 and k=128 with aux loss. Uses cached activations."""
import numpy as np, torch, time, sys, os
import torch.nn as nn, torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
CACHE = BASE + "/cache/puzzle_acts_200k.pt"
OUTPUT = BASE + "/output/k_sweep"
os.makedirs(OUTPUT, exist_ok=True)

print("Loading cache...")
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
flat = cache["token_acts"].float().reshape(-1, 1024)
print(f"Loaded: {flat.shape}")

class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k; self.dd = dd
    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros_like(z); a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a

def train_sae(flat, dict_size, k, epochs=5, batch_size=2048, lr=1e-3):
    n = flat.shape[0]
    sae = SAE(1024, dict_size, k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    steps_since_fired = torch.zeros(dict_size, device="cuda")

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_mse = 0; total_aux = 0; nb = 0
        for i in range(0, n, batch_size):
            batch = flat[perm[i:i+batch_size]].cuda()
            recon, acts = sae(batch)
            mse = F.mse_loss(recon, batch)
            fired = (acts > 0).any(dim=0)
            steps_since_fired[fired] = 0
            steps_since_fired[~fired] += 1
            dead = steps_since_fired > 50
            aux = torch.tensor(0.0, device="cuda")
            if dead.sum() > 0:
                res = (batch - recon).detach()
                de = sae.encoder.weight[dead] @ res.T
                da = F.relu(de).T
                dr = da @ sae.decoder.weight[:, dead].T
                aux = F.mse_loss(dr, res)
            loss = mse + (1/32)*aux
            opt.zero_grad(); loss.backward(); opt.step()
            total_mse += mse.item(); total_aux += aux.item(); nb += 1
        nd = (steps_since_fired > 50).sum().item()
        print(f"  ep{ep} mse={total_mse/nb:.6f} aux={total_aux/nb:.6f} dead={nd}/{dict_size}")
        sys.stdout.flush()

    with torch.no_grad():
        sample = flat[:min(50000, n)].cuda()
        _, sa = sae(sample)
        dead_n = ((sa > 0).sum(dim=0) == 0).sum().item()
        l0 = (sa > 0).float().sum(dim=-1).mean().item()
        recon, _ = sae(sample)
        fvu = F.mse_loss(recon, sample).item() / sample.var().item()
        dec_w = sae.decoder.weight.data
        dec_n = F.normalize(dec_w, dim=0)
        cos = dec_n.T @ dec_n
        mask = ~torch.eye(dict_size, dtype=torch.bool, device=cos.device)
        c_dec = cos[mask].abs().mean().item()

    elapsed = time.time() - t0

    # Save weights
    out_path = f"{OUTPUT}/sae_btk_{dict_size}_k{k}_aux.pt"
    torch.save({
        "encoder_weight": sae.encoder.weight.data.cpu(),
        "encoder_bias": sae.encoder.bias.data.cpu(),
        "decoder_weight": sae.decoder.weight.data.cpu(),
        "pre_bias": sae.pre_bias.data.cpu(),
        "k": k, "dict_size": dict_size,
        "mean": cache["mean"], "std": cache["std"],
    }, out_path)

    return {"dict": dict_size, "k": k, "dead": dead_n, "active": dict_size-dead_n,
            "l0": round(l0,1), "fvu": round(fvu,4), "c_dec": round(c_dec,6), "time": round(elapsed)}

configs = [(4096, 64), (4096, 128)]
results = []
for ds, k in configs:
    print(f"\nTraining dict={ds} k={k}...")
    sys.stdout.flush()
    r = train_sae(flat, ds, k)
    results.append(r)
    print(f"  -> dead={r['dead']}/{ds} active={r['active']} L0={r['l0']} FVU={r['fvu']} c_dec={r['c_dec']} ({r['time']}s)")
    sys.stdout.flush()

print("\n" + "="*60)
print("SWEEP RESULTS")
print(f"{'dict':>5} {'k':>4}  {'dead':>6}  {'active':>6}  {'L0':>5}  {'FVU':>6}  {'c_dec':>8}")
for r in results:
    print(f"{r['dict']:>5} {r['k']:>4}  {r['dead']:>6}  {r['active']:>6}  {r['l0']:>5}  {r['fvu']:>.4f}  {r['c_dec']:>.6f}")

# Also print 2048 k=64 baseline for comparison
print("\nBaseline (2048 k=64 aux): dead=213 active=1835 L0=64.0 FVU=82.14 c_dec=0.035895")
