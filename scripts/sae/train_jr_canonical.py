#!/usr/bin/env python3
"""Canonical JumpReLU SAE (ported faithfully from SandstonePersonas samtkap/sae/model.py, commit
d6d2229) — target_l0 quadratic sparsity + AuxK dead-feature revival + separate high-LR threshold group.

Differs from the earlier train_jr_sweep.py (which used a linear l0_coeff penalty + single LR):
  - L0 loss = l0_alpha * (L0 − target_l0)^2   → PULLS L0 toward target_l0 from both sides (SETS it).
  - JumpReLUSAEAuxK: dead features (>= n_batches_to_dead) reconstruct the residual (revival).
  - log_threshold gets its OWN param group at ~33x LR — the piece missing before, which is why θ
    "couldn't climb" and l0_coeff looked inert. The reference's own comment (model.py L414-418)
    documents this exact chess-deck lesson. With the high-LR group θ trains properly and target_l0
    becomes the real sparsity knob (no hand-init hack needed).
  - init_threshold ~0.5, bandwidth ~0.1 (reference defaults for standardized embeddings).

Usage (chess-poc, pytorch_p310):
  python3 train_jr_canonical.py --cache <l7only.pt> --dict-size 512 --target-l0 8 --epochs 60 \
      --tag jr512_k8 -o out/jr512_k8.pt --results out/canon_results.jsonl
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x * (x > threshold)).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        x, threshold = ctx.saved_tensors; b = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * g
        thr_grad = torch.sum(-(threshold / b) * rectangle((x - threshold) / b) * g, dim=0)
        return x_grad, thr_grad, None


class _StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        x, threshold = ctx.saved_tensors; b = ctx.bandwidth
        thr_grad = torch.sum(-(1.0 / b) * rectangle((x - threshold) / b) * g, dim=0)
        return None, thr_grad, None


class JumpReLUSAEAuxK(nn.Module):
    """Canonical: target_l0 quadratic + AuxK revival. Mirrors the reference class of the same name."""
    def __init__(self, emb_size, d_hidden, target_l0=8.0, l0_alpha=0.1, bandwidth=0.1,
                 init_threshold=0.5, k_aux=None, aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(emb_size, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(emb_size))
        self.log_threshold = nn.Parameter(torch.full((d_hidden,), float(np.log(init_threshold))))
        self.d_hidden = d_hidden
        self.target_l0 = target_l0; self.l0_alpha = l0_alpha; self.bandwidth = bandwidth
        self.k_aux = k_aux if k_aux is not None else max(64, d_hidden // 8)
        self.aux_alpha = aux_alpha; self.n_batches_to_dead = n_batches_to_dead
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    @property
    def threshold(self):
        return self.log_threshold.exp()

    def threshold_param_group(self, base_lr, mult=33.0):
        """Reference exposes this so the trainer gives log_threshold its own ~33x LR."""
        other = [p for n, p in self.named_parameters() if n != "log_threshold"]
        return [{"params": other, "lr": base_lr},
                {"params": [self.log_threshold], "lr": base_lr * mult}]

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        theta = self.threshold
        acts = _JumpReLU.apply(z, theta, self.bandwidth)
        if self.training:
            active = (acts > 0).any(dim=0)
            self.num_batches_not_active[active] = 0
            self.num_batches_not_active[~active] += 1
        x_hat = acts @ self.W_dec + self.b_dec
        l2 = (x_hat.float() - x.float()).pow(2).sum(-1).mean()
        l0 = _StepSTE.apply(z, theta, self.bandwidth).sum(-1).mean()
        l0_loss = self.l0_alpha * (l0 - self.target_l0).pow(2)
        aux = torch.tensor(0.0, device=x.device)
        if self.training and self.k_aux > 0:
            dead = self.num_batches_not_active >= self.n_batches_to_dead
            if dead.sum() > 0:
                err = (x - x_hat).detach()
                dpre = F.relu(z[:, dead])
                ka = min(self.k_aux, int(dead.sum()))
                tk = torch.topk(dpre, k=ka, dim=-1)
                dacts = torch.zeros_like(dpre).scatter(-1, tk.indices, tk.values)
                err_hat = dacts @ self.W_dec[dead]
                aux = self.aux_alpha * (err_hat.float() - err.float()).pow(2).mean()
        loss = l2 + l0_loss + aux
        return loss, x_hat, acts, l2, l0_loss, aux

    @torch.no_grad()
    def make_decoder_unit_norm_grad_proj(self):
        Wn = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if self.W_dec.grad is not None:
            proj = (self.W_dec.grad * Wn).sum(-1, keepdim=True) * Wn
            self.W_dec.grad -= proj
        self.W_dec.data = Wn


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
    sae.eval()
    n = len(data); idx = torch.arange(0, n, max(1, n // 32768))
    fires = torch.zeros(sae.d_hidden, device=device); sse = 0.0; cnt = 0
    for i in range(0, len(idx), 8192):
        b = data[idx[i:i+8192]].to(device)
        _, x_hat, acts, _, _, _ = sae(b)
        fires += (acts > 0).float().sum(0); sse += (x_hat.float()-b.float()).pow(2).sum().item(); cnt += b.numel()
    fr = (fires/len(idx)).cpu(); active = fr[fr>0]
    band = lambda lo,hi: int(((fr>=lo)&(fr<hi)).sum())
    return {"fvu": round(sse/(var_total*cnt),4), "mean_L0": round(float(fr.sum()),2),
            "dead_pct": round(100.0*float((fr==0).sum())/sae.d_hidden,1),
            "fire_lt1pct": band(1e-4,0.01), "fire_1_5pct": band(0.01,0.05),
            "fire_5_10pct": band(0.05,0.10), "fire_gt10pct": int((fr>=0.10).sum()),
            "fire_median_pct": round(float(active.median())*100,2) if len(active) else 0.0,
            "theta_median": round(float(sae.threshold.median()),4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dict-size", type=int, default=512)
    ap.add_argument("--target-l0", type=float, default=8.0)
    ap.add_argument("--l0-alpha", type=float, default=0.1)
    ap.add_argument("--bandwidth", type=float, default=0.1)
    ap.add_argument("--init-threshold", type=float, default=0.5)
    ap.add_argument("--thr-lr-mult", type=float, default=33.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="jr")
    ap.add_argument("--results", default="canon_results.jsonl")
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    data, mean, std = load_data(args.cache)
    n, emb = data.shape; var_total = float(data.float().var())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] {n} pos {emb}d dict={args.dict_size} target_l0={args.target_l0} "
          f"thr_lr_mult={args.thr_lr_mult} dev={dev}", flush=True)
    sae = JumpReLUSAEAuxK(emb, args.dict_size, target_l0=args.target_l0, l0_alpha=args.l0_alpha,
                          bandwidth=args.bandwidth, init_threshold=args.init_threshold).to(dev)
    opt = torch.optim.Adam(sae.threshold_param_group(args.lr, args.thr_lr_mult))

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n); sae.train(); eL2=eL0=eAux=nb=0
        for i in range(0, n, args.batch_size):
            b = data[perm[i:i+args.batch_size]].to(dev)
            loss, _, _, l2, l0l, aux = sae(b)
            opt.zero_grad(); loss.backward(); sae.make_decoder_unit_norm_grad_proj(); opt.step()
            eL2+=l2.item(); eL0+=l0l.item(); eAux+=aux.item(); nb+=1
        if (ep+1)%10==0 or ep==args.epochs-1:
            m = evaluate(sae, data, dev, var_total)
            print(f"  [{args.tag}] ep{ep+1} l2={eL2/nb:.4f} l0loss={eL0/nb:.4f} | FVU={m['fvu']} "
                  f"L0={m['mean_L0']} dead={m['dead_pct']}% fireMed={m['fire_median_pct']}% "
                  f"1-5%={m['fire_1_5pct']} blobs={m['fire_gt10pct']} θ={m['theta_median']}", flush=True)
    m = evaluate(sae, data, dev, var_total)
    torch.save({"state_dict": sae.state_dict(), "mean": mean, "std": std,
                "config": {"emb_size": emb, "dict_size": args.dict_size, "target_l0": args.target_l0,
                           "l0_alpha": args.l0_alpha, "bandwidth": args.bandwidth,
                           "init_threshold": args.init_threshold, "arch": "jumprelu_auxk_canonical"}}, args.output)
    rec = {"tag": args.tag, "dict_size": args.dict_size, "target_l0": args.target_l0,
           "minutes": round((time.time()-t0)/60,1), **m}
    with open(args.results, "a") as f: f.write(json.dumps(rec)+"\n")
    print(f"[{args.tag}] DONE {rec['minutes']}min -> {args.output} | {json.dumps(m)}", flush=True)


if __name__ == "__main__":
    main()
