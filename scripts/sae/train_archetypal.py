#!/usr/bin/env python3
"""Train Relaxed-Archetypal JumpReLU SAE (RA-JumpSAE) on Maia blunder-diff activations + test stability.

Why: our plain JumpReLU SAE is seed-UNSTABLE (MMCS~0.41, only ~6% of features reproduce >0.7 across
seeds) — the known SAE failure (Fel et al., ICML 2025, "Archetypal SAE"). Archetypal SAEs constrain
dictionary atoms to the convex hull of the data, which the paper shows dramatically improves
cross-seed stability. We use the relaxed-archetypal JUMPRELU variant (RAJumpSAE) — archetypal
stability + JumpReLU's learned per-feature threshold (no fixed-k long tail).

This script: train 2 seeds, then measure decoder MMCS between them. Success = MMCS and the >0.7
reproducible fraction jump well above our plain-JumpReLU baseline (0.41 / ~6%).

Uses the official `overcomplete` library (Fel et al.). RAJumpSAE(input_shape, nb_concepts, points,
bandwidth, delta); forward(x)->(z_pre, z, x_hat); criterion = mse_l1(x,x_hat,z_pre,z,dict,penalty).

Run (chess-poc GPU):
  python3 train_archetypal.py --cache <blunder_diff.pt> --nb-concepts 2048 \
     --n-points 4096 --penalty 0.04 --epochs 40 --out /home/ec2-user/SageMaker/ra_jumpsae
"""
import argparse, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from overcomplete.sae import RAJumpSAE, train_sae
from overcomplete.sae.losses import mse_l1


def load(cache):
    c = torch.load(cache, map_location="cpu", weights_only=False)
    data = c["activations"].float() if "activations" in c else c["blunder_mt"].float()
    cm, cs = c.get("mean"), c.get("std")
    if cm is not None:
        mean = torch.tensor(np.array(cm), dtype=torch.float32)
        std = torch.tensor(np.array(cs), dtype=torch.float32).clamp(min=1e-6)
    else:
        mean = data.mean(0); std = data.std(0).clamp(min=1e-6)
    return (data - mean) / std


def train_one(data, nb_concepts, n_points, penalty, bandwidth, epochs, bs, lr, top_k, device, seed):
    torch.manual_seed(seed)
    n, dim = data.shape
    # archetype candidate points: random sample of the data (their hull constrains the dictionary)
    idx = torch.randperm(n)[:n_points]
    points = data[idx].to(device)
    model = RAJumpSAE(dim, nb_concepts, points, bandwidth=bandwidth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(data), batch_size=bs, shuffle=True, drop_last=True)
    crit = lambda x, x_hat, zp, z, D: mse_l1(x, x_hat, zp, z, D, penalty=penalty)
    train_sae(model, loader, crit, opt, nb_epochs=epochs, device=device, monitoring=0)
    return model


@torch.no_grad()
def stats(model, data, device):
    model.eval()
    feats = []
    for i in range(0, len(data), 16384):
        zp, z, xh = model(data[i:i+16384].to(device))
        feats.append((z > 0).float().mean(0).cpu())
    fire = torch.stack(feats).mean(0)
    live = fire > 0
    D = model.get_dictionary().detach()           # (nb_concepts, dim)
    D = D / D.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # reconstruction on a sample
    s = data[torch.randperm(len(data))[:16384]].to(device)
    _, z, xh = model(s)
    L0 = (z > 0).float().sum(-1).mean().item()
    fvu = ((xh - s).pow(2).sum(-1).mean() / ((s - s.mean(0)).pow(2).sum(-1).mean() + 1e-8)).item()
    return D[live.to(D.device)], round(L0, 1), round(fvu, 4), int(live.sum())


def mmcs(A, B):
    best = (A @ B.T).max(dim=1).values
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nb-concepts", type=int, default=2048)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--penalty", type=float, default=0.04)
    ap.add_argument("--bandwidth", type=float, default=0.001)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load(args.cache)
    n, dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} pos, {dim}-dim, device={device}; RA-JumpSAE concepts={args.nb_concepts} "
          f"points={args.n_points} penalty={args.penalty}", flush=True)

    t0 = time.time()
    mA = train_one(data, args.nb_concepts, args.n_points, args.penalty, args.bandwidth,
                   args.epochs, args.batch_size, args.lr, args.top_k, device, seed=0)
    DA, L0a, fvua, nla = stats(mA, data, device)
    print(f"  seed0: L0={L0a} FVU={fvua} live={nla} ({(time.time()-t0)/60:.1f}min)", flush=True)
    torch.save({"state_dict": mA.state_dict(),
                "config": {"dim": dim, "nb_concepts": args.nb_concepts, "penalty": args.penalty}},
               args.out + "_seed0.pt")

    mB = train_one(data, args.nb_concepts, args.n_points, args.penalty, args.bandwidth,
                   args.epochs, args.batch_size, args.lr, args.top_k, device, seed=1)
    DB, L0b, fvub, nlb = stats(mB, data, device)
    print(f"  seed1: L0={L0b} FVU={fvub} live={nlb} ({(time.time()-t0)/60:.1f}min)", flush=True)

    best = mmcs(DA.to(device), DB.to(device))
    out = {"nb_concepts": args.nb_concepts, "penalty": args.penalty,
           "seed0": {"L0": L0a, "fvu": fvua, "live": nla},
           "seed1": {"L0": L0b, "fvu": fvub, "live": nlb},
           "MMCS": round(float(best.mean()), 3),
           "frac>0.5": round(float((best > 0.5).mean()), 3),
           "frac>0.7": round(float((best > 0.7).mean()), 3),
           "frac>0.8": round(float((best > 0.8).mean()), 3),
           "frac>0.9": round(float((best > 0.9).mean()), 3)}
    json.dump(out, open(args.out + "_stability.json", "w"), indent=2)
    print("\n=== RA-JumpSAE seed stability (vs plain JumpReLU baseline MMCS=0.41, >0.7=6%) ===", flush=True)
    print(f"  MMCS={out['MMCS']}  >0.5={out['frac>0.5']}  >0.7={out['frac>0.7']}  "
          f">0.8={out['frac>0.8']}  >0.9={out['frac>0.9']}", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f}min -> {args.out}_*", flush=True)


if __name__ == "__main__":
    main()
