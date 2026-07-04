#!/usr/bin/env python3
"""Map the L0-vs-reconstruction frontier across dict sizes, using the PROVEN JumpReLU trainer.

Settles: "is L0~32 high? is it a dict-size artifact? would 256/512 give lower L0?"
Answer needs the FRONTIER, not one point: L0 and reconstruction trade off; l0_coeff picks the point.
So sweep dict x l0_coeff and report, per config: achieved L0, FVU (fraction variance unexplained),
dead%, blob%. Then compare dicts AT MATCHED reconstruction (same FVU) — that's the honest comparison.

Uses JumpReLUSAE (stable: 0 dead, converges) — NOT plain ReLU+L1 (collapses to FVU=1.0).

Usage (chess-poc GPU):
  python3 jumprelu_frontier_sweep.py --cache <blunder_diff.pt> \
     --dicts 256,512,1024,2048,4096 --l0s 0.01,0.02,0.04,0.08 --epochs 40 \
     --out /home/ec2-user/SageMaker/jumprelu_frontier.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_jumprelu import JumpReLUSAE, load_data


def train_eval(data, input_dim, dict_size, l0_coeff, epochs, bs, lr, device):
    sae = JumpReLUSAE(input_dim, dict_size, l0_coeff=l0_coeff).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        sae.train()
        for i in range(0, n, bs):
            batch = data[perm[i:i+bs]].to(device)
            loss, _, _, _, _, _ = sae(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            sae.make_decoder_weights_unit_norm()
    sae.eval()
    with torch.no_grad():
        s = data[torch.randperm(n)[:16384]].to(device)
        _, x_hat, acts, _, _, _ = sae(s)
        L0 = (acts > 0).float().sum(-1).mean().item()
        fire = (acts > 0).float().mean(0)                       # per-feature fire rate
        dead = (fire == 0).float().mean().item()
        blobs = (fire > 0.10).float().mean().item()
        fvu = ((x_hat - s).pow(2).sum(-1).mean() / ((s - s.mean(0)).pow(2).sum(-1).mean() + 1e-8)).item()

        # --- LABEL-FREE INTERPRETABILITY METRICS (FVU is not the goal; these are) ---
        # (1) Feature SPLITTING/redundancy: for each LIVE feature, max cosine to any OTHER feature's
        #     decoder vector. High = near-duplicate features (one concept smeared across many). This is
        #     the failure FVU hides — a bigger dict can "reconstruct better" purely by splitting concepts.
        live = fire > 0
        Wd = sae.W_dec[live]                                     # (n_live, input_dim), already unit-norm
        sim = Wd @ Wd.T                                          # cosine (rows are unit-norm)
        sim.fill_diagonal_(-1.0)
        max_cos = sim.max(dim=1).values                          # nearest neighbor cosine per feature
        dup_frac = (max_cos > 0.9).float().mean().item()         # % features with a near-twin (>0.9)
        median_maxcos = max_cos.median().item()
        # (2) Fire-rate INEQUALITY (the long tail, quantified): Gini of fire rates + top-5% share.
        fr = fire[live].sort().values
        nlive = fr.numel()
        idx = torch.arange(1, nlive + 1, device=fr.device, dtype=fr.dtype)
        gini = ((2 * idx - nlive - 1) * fr).sum().item() / (nlive * fr.sum().item() + 1e-9)
        k5 = max(1, nlive // 20)
        top5_share = fr[-k5:].sum().item() / (fr.sum().item() + 1e-9)
    return {"dict": dict_size, "l0_coeff": l0_coeff, "L0": round(L0, 1), "fvu": round(fvu, 4),
            "dead_frac": round(dead, 3), "blob_frac": round(blobs, 3),
            "dup_frac": round(dup_frac, 3), "median_maxcos": round(median_maxcos, 3),
            "fire_gini": round(gini, 3), "top5pct_share": round(top5_share, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dicts", default="256,512,1024,2048,4096")
    ap.add_argument("--l0s", default="0.01,0.02,0.04,0.08")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data, _, _ = load_data(args.cache)
    n, input_dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dicts = [int(x) for x in args.dicts.split(",")]
    l0s = [float(x) for x in args.l0s.split(",")]
    print(f"{n} positions, {input_dim}-dim, device={device}; {len(dicts)}x{len(l0s)} grid", flush=True)

    t0 = time.time()
    rows = []
    for ds in dicts:
        for c in l0s:
            r = train_eval(data, input_dim, ds, c, args.epochs, args.batch_size, args.lr, device)
            rows.append(r)
            print(f"  dict={ds:<5} l0c={c:<6} L0={r['L0']:<6} FVU={r['fvu']:<7} "
                  f"dead={r['dead_frac']:<5} blobs={r['blob_frac']:<6} "
                  f"dup%={r['dup_frac']:<6} gini={r['fire_gini']:<6} top5={r['top5pct_share']} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)
    print("\n=== frontier per dict (sorted by L0): does bigger dict give better FVU at same L0? ===", flush=True)
    for ds in dicts:
        sub = sorted([r for r in rows if r["dict"] == ds], key=lambda r: r["L0"])
        print(f"  dict={ds:<5} " + "  ".join(f"L0={r['L0']}/FVU={r['fvu']}/blob={r['blob_frac']}" for r in sub), flush=True)


if __name__ == "__main__":
    main()
