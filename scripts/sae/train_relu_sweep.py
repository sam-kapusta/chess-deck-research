#!/usr/bin/env python3
"""Sweep plain ReLU+L1 SAEs to map the L0-vs-reconstruction frontier. (Sam, 2026-06-17)

Question being settled: "is L0~32 high? is that a dict-size thing? would 256/512 be lower?"
Answer can't come from one config — L0 and reconstruction trade off and the L1 penalty just picks a
point on that curve. So sweep dict x penalty and report the FRONTIER: for each (dict, l1) -> achieved
L0, reconstruction quality (FVU = fraction of variance unexplained), dead %, blob %.

Classic Anthropic ReLU SAE (Bricken et al. 2023):
  f(x) = ReLU(W_enc (x - b_dec) + b_enc)
  x_hat = f W_dec + b_dec
  loss = ||x - x_hat||^2 / ||x - mean||^2 (normalized)  +  l1 * sum_i |f_i| * ||W_dec_i||
The L1 term is weighted by decoder column norm (resolves the shrinkage/scale degeneracy without
forcing unit-norm, the standard fix). L0 (active count) is an OUTCOME of l1, not set directly — that's
the whole point of the sweep vs TopK/JumpReLU where you dial it.

Usage (chess-poc GPU):
  python3 train_relu_sweep.py --cache <blunder_diff.pt> \
     --dicts 256,512,1024,2048,4096 --l1s 0.5,1,2,4,8 --epochs 40 \
     --out /home/ec2-user/SageMaker/relu_sweep.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUSAE(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, dict_size)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, l1):
        acts = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        x_hat = acts @ self.W_dec + self.b_dec
        # reconstruction = plain MSE (per-element mean) for the gradient — do NOT normalize it down,
        # or the L1 term dominates and zeroes every feature (the smoke-test bug: FVU=1.0, L0=0.5).
        mse = (x_hat - x).pow(2).mean()
        # L1 weighted by decoder column norm (scale-aware shrinkage fix)
        dec_norms = self.W_dec.norm(dim=-1)
        l1_pen = l1 * (acts * dec_norms).sum(-1).mean()
        # FVU reported separately for the frontier (fraction of variance unexplained)
        fvu = (x_hat - x).pow(2).sum(-1).mean() / ((x - x.mean(0)).pow(2).sum(-1).mean() + 1e-8)
        return mse + l1_pen, x_hat, acts, fvu


def train_one(data, input_dim, dict_size, l1, epochs, bs, lr, device):
    sae = ReLUSAE(input_dim, dict_size).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            batch = data[perm[i:i+bs]].to(device)
            loss, _, _, _ = sae(batch, l1)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    sae.eval()
    with torch.no_grad():
        s = data[torch.randperm(n)[:16384]].to(device)
        _, _, acts, l2 = sae(s, l1)
        L0 = (acts > 1e-6).float().sum(-1).mean().item()
        fire = (acts > 1e-6).float().mean(0)
        dead = (fire == 0).float().mean().item()
        blobs = (fire > 0.10).float().mean().item()
        fvu = l2.item()   # normalized reconstruction ~ fraction variance unexplained
    return {"dict": dict_size, "l1": l1, "L0": round(L0, 1), "fvu": round(fvu, 4),
            "dead_frac": round(dead, 3), "blob_frac": round(blobs, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dicts", default="256,512,1024,2048,4096")
    ap.add_argument("--l1s", default="0.5,1,2,4,8")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    data = cache["activations"].float() if "activations" in cache else cache["blunder_mt"].float()
    cm, cs = cache.get("mean"), cache.get("std")
    if cm is not None:
        mean = torch.tensor(np.array(cm), dtype=torch.float32)
        std = torch.tensor(np.array(cs), dtype=torch.float32).clamp(min=1e-6)
    else:
        mean = data.mean(0); std = data.std(0).clamp(min=1e-6)
    data = (data - mean) / std
    n, input_dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dicts = [int(x) for x in args.dicts.split(",")]
    l1s = [float(x) for x in args.l1s.split(",")]
    print(f"{n} positions, {input_dim}-dim, device={device}; {len(dicts)}x{len(l1s)} grid", flush=True)

    t0 = time.time()
    rows = []
    for ds in dicts:
        for l1 in l1s:
            r = train_one(data, input_dim, ds, l1, args.epochs, args.batch_size, args.lr, device)
            rows.append(r)
            print(f"  dict={ds:<5} l1={l1:<5} L0={r['L0']:<6} FVU={r['fvu']:<7} "
                  f"dead={r['dead_frac']:<5} blobs={r['blob_frac']} ({(time.time()-t0)/60:.1f}min)", flush=True)
        json.dump(rows, open(args.out, "w"), indent=2)   # checkpoint per dict

    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)
    # frontier summary: for each dict, the l1 giving L0 closest to {16,32,64}
    print("\n=== frontier: FVU at matched L0 (lower FVU = better recon at that sparsity) ===", flush=True)
    for ds in dicts:
        sub = [r for r in rows if r["dict"] == ds]
        line = " ".join(f"L0={r['L0']}|FVU={r['fvu']}" for r in sorted(sub, key=lambda r: r["L0"]))
        print(f"  dict={ds:<5} {line}", flush=True)


if __name__ == "__main__":
    main()
