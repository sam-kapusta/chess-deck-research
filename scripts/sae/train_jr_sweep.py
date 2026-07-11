#!/usr/bin/env python3
"""JumpReLU SAE sweep on the Maia3 l7only best-blunder diff cache.

Goal (Sam, 2026-07-11): good models with FIRE RATES in the 1-5% band (optimal), L0 flexible
(4/6/8/16 all fine). Built on the proven canonical JumpReLU (Rajamanoharan et al. / SAELens port) that
already produced real models on this notebook — NOT a from-memory reconstruction of the tanh scheme.

Key de-risking (measured on this exact cache, probe_scale.py):
  normed input std≈1.0; ReLU pre-act nonzero median=0.565, p90=1.63.
  The first pass DIED at init_threshold=0.5 (== the median → half the features start below θ → dead).
  Healthy family (Sam's jr_A/B/C): init_threshold=0.06, bandwidth=0.02 (θ sits BELOW the small end so
  features start alive and θ learns UPWARD). This script scale-matches by default.

Sweep the L0 penalty (l0_coeff) to walk the sparsity/fire-rate frontier. Softer penalty = more features
active = lower fire rate per feature (Sam: jr_B softest penalty was healthiest).

Per run, reports the metric that matters: how many features fire in [1%,5%], plus dead%, FVU, L0, θ.
Appends one JSON line to --results so the sweep can be ranked unattended.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors; b = ctx.bandwidth
        thr_grad = torch.sum(-(1.0 / b) * rectangle((x - threshold) / b) * grad_output, dim=0)
        return None, thr_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x * (x > threshold)).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors; b = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * grad_output
        thr_grad = torch.sum(-(threshold / b) * rectangle((x - threshold) / b) * grad_output, dim=0)
        return x_grad, thr_grad, None


class JumpReLUSAE(nn.Module):
    def __init__(self, input_dim, dict_size, l0_coeff=4e-3, bandwidth=0.02,
                 init_threshold=0.06, k_aux=256, aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, dict_size)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.log_theta = nn.Parameter(torch.full((dict_size,), float(np.log(init_threshold))))
        self.dict_size = dict_size; self.l0_coeff = l0_coeff; self.bandwidth = bandwidth
        self.k_aux = k_aux; self.aux_alpha = aux_alpha; self.n_batches_to_dead = n_batches_to_dead
        self.register_buffer("num_batches_not_active", torch.zeros(dict_size))

    @property
    def threshold(self):
        return torch.exp(self.log_theta)

    def forward(self, x):
        theta = self.threshold
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        acts = JumpReLU.apply(pre, theta, self.bandwidth)
        x_hat = acts @ self.W_dec + self.b_dec
        l2 = (x_hat.float() - x.float()).pow(2).mean()
        if self.training:
            l0 = Step.apply(pre, theta, self.bandwidth).sum(dim=-1).mean()
            l0_pen = self.l0_coeff * l0
            active = (acts > 0).any(dim=0)
            self.num_batches_not_active[active] = 0
            self.num_batches_not_active[~active] += 1
        else:
            l0_pen = torch.tensor(0.0, device=x.device)
        aux = torch.tensor(0.0, device=x.device)
        if self.training and self.k_aux > 0:
            dead = self.num_batches_not_active >= self.n_batches_to_dead
            if dead.sum() > 0:
                err = (x - x_hat).detach(); dpre = pre[:, dead]
                ka = min(self.k_aux, int(dead.sum()))
                tk = torch.topk(dpre, k=ka, dim=-1)
                dacts = torch.zeros_like(dpre).scatter(-1, tk.indices, tk.values)
                err_hat = dacts @ self.W_dec[dead]
                aux = self.aux_alpha * (err_hat.float() - err.float()).pow(2).mean()
        loss = l2 + l0_pen + aux
        return loss, x_hat, acts, l2, l0_pen, aux

    @torch.no_grad()
    def make_decoder_weights_unit_norm(self):
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)


def load_data(cache_path):
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    data = c["activations"].float() if "activations" in c else c[list(c.keys())[0]].float()
    cm, cs = c.get("mean"), c.get("std")
    if cm is not None:
        mean = torch.tensor(np.array(cm), dtype=torch.float32)
        std = torch.tensor(np.array(cs), dtype=torch.float32).clamp(min=1e-6)
    else:
        mean = data.mean(0); std = data.std(0).clamp(min=1e-6)
    return (data - mean) / std, mean, std


@torch.no_grad()
def evaluate(sae, data, device, var_total):
    """FVU (against total variance), fire-rate distribution, dead%, θ stats. On a fixed eval sample."""
    sae.eval()
    n = len(data)
    idx = torch.arange(0, n, max(1, n // 32768))  # ~32k stratified sample
    fires = torch.zeros(sae.dict_size, device=device)
    sse = 0.0; count = 0
    for i in range(0, len(idx), 8192):
        b = data[idx[i:i+8192]].to(device)
        _, x_hat, acts, _, _, _ = sae(b)
        fires += (acts > 0).float().sum(dim=0)
        sse += (x_hat.float() - b.float()).pow(2).sum().item()
        count += b.numel()
    fire_rate = (fires / len(idx)).cpu()
    fvu = sse / (var_total * count)   # var_total = per-elem variance of normed data (~1.0)
    band = lambda lo, hi: int(((fire_rate >= lo) & (fire_rate < hi)).sum())
    mean_l0 = float(fire_rate.sum())  # sum of per-feature fire rates = mean active features per position
    return {
        "fvu": round(fvu, 4),
        "mean_L0": round(mean_l0, 2),
        "dead_pct": round(100.0 * float((fire_rate == 0).sum()) / sae.dict_size, 1),
        "fire_lt1pct": band(0.0001, 0.01),     # (0,1%) excludes dead
        "fire_1_5pct": band(0.01, 0.05),       # THE target band
        "fire_5_10pct": band(0.05, 0.10),
        "fire_gt10pct": int((fire_rate >= 0.10).sum()),  # blobs
        "theta_median": round(float(sae.threshold.median()), 4),
        "theta_p90": round(float(torch.quantile(sae.threshold, 0.9)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dict-size", type=int, default=2048)
    ap.add_argument("--l0-coeff", type=float, default=4e-3)
    ap.add_argument("--bandwidth", type=float, default=0.02)
    ap.add_argument("--init-threshold", type=float, default=0.06)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="jr")
    ap.add_argument("--results", default="jr_sweep_results.jsonl")
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    data, mean, std = load_data(args.cache)
    n, input_dim = data.shape
    var_total = float(data.float().var())   # ≈1.0 for z-scored; FVU denominator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] {n} pos, {input_dim}d, var={var_total:.3f}, device={device} | "
          f"l0={args.l0_coeff} thr={args.init_threshold} bw={args.bandwidth}", flush=True)

    sae = JumpReLUSAE(input_dim, args.dict_size, l0_coeff=args.l0_coeff,
                      bandwidth=args.bandwidth, init_threshold=args.init_threshold).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n); sae.train()
        eL2 = eL0 = eAux = nb = 0
        for i in range(0, n, args.batch_size):
            batch = data[perm[i:i+args.batch_size]].to(device)
            loss, _, acts, l2, l0p, aux = sae(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            sae.make_decoder_weights_unit_norm()
            eL2 += l2.item(); eL0 += l0p.item(); eAux += aux.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            m = evaluate(sae, data, device, var_total)
            print(f"  [{args.tag}] ep{ep+1}/{args.epochs} l2={eL2/nb:.5f} l0pen={eL0/nb:.4f} "
                  f"| FVU={m['fvu']} dead={m['dead_pct']}% fire<1%={m['fire_lt1pct']} "
                  f"fire1-5%={m['fire_1_5pct']} fire5-10%={m['fire_5_10pct']} blobs={m['fire_gt10pct']} "
                  f"θmed={m['theta_median']}", flush=True)

    m = evaluate(sae, data, device, var_total)
    torch.save({"state_dict": sae.state_dict(), "mean": mean, "std": std,
                "config": {"input_dim": input_dim, "dict_size": args.dict_size,
                           "l0_coeff": args.l0_coeff, "bandwidth": args.bandwidth,
                           "init_threshold": args.init_threshold, "arch": "jumprelu"}}, args.output)
    rec = {"tag": args.tag, "output": os.path.basename(args.output), "l0_coeff": args.l0_coeff,
           "bandwidth": args.bandwidth, "init_threshold": args.init_threshold,
           "dict_size": args.dict_size, "epochs": args.epochs,
           "minutes": round((time.time() - t0) / 60, 1), **m}
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{args.tag}] DONE {rec['minutes']}min -> {args.output} | {json.dumps(m)}", flush=True)


if __name__ == "__main__":
    main()
