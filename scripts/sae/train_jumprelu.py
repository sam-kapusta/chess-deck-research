#!/usr/bin/env python3
"""Train a JumpReLU SAE on Maia3 blunder-diff activations.

Why: the current BatchTopK SAE has the long-tail problem Sam dislikes — TopK forces a fixed budget
per batch, so a few "blob" features hog the top-k slots and 1000+ latents go dead. JumpReLU
(Rajamanoharan et al., arXiv:2407.14435) replaces the top-k with a LEARNABLE per-feature threshold:
a feature fires only when its pre-activation exceeds its OWN threshold theta_i. No fixed budget ->
no forced long tail; the sparsity penalty (L0) lets each feature claim exactly the activation mass
it earns. Dead latents are resurrected via an AuxK term (same as the BatchTopK trainer).

Architecture (per the paper):
  f(x) = JumpReLU_theta( (x - b_dec) W_enc + b_enc )      JumpReLU_theta(z) = z * H(z - theta)
  x_hat = f(x) W_dec + b_dec
  loss  = ||x - x_hat||^2  +  lambda * L0(f(x))  +  aux
L0 is non-differentiable -> straight-through estimator on the Heaviside gate with a kernel bandwidth
(paper Eq. 9-10). theta is trained in log-space (always positive). Decoder rows kept unit-norm.

Mirrors train_batchtopk.py's harness (cache load, mean/std, decoder unit-norm, AuxK, S3 upload).

Usage (chess-poc GPU):
  python3 train_jumprelu.py --cache <blunder_diff.pt> --dict-size 2048 --l0-coeff 4e-3 \
     --epochs 30 -o sae_jumprelu_blunderdiff_2048.pt
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- canonical JumpReLU autograd (Rajamanoharan et al. 2024; ported from SAELens jumprelu_sae.py) ---
def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class Step(torch.autograd.Function):
    """Heaviside H(x-theta): forward = mask; grad ONLY to threshold via rectangle kernel. Used for L0."""
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        b = ctx.bandwidth
        thr_grad = torch.sum(-(1.0 / b) * rectangle((x - threshold) / b) * grad_output, dim=0)
        return None, thr_grad, None   # no grad to x (pure threshold learner)


class JumpReLU(torch.autograd.Function):
    """acts = x * H(x-theta). grad to x = straight-through (1 where active); grad to theta = kernel."""
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x * (x > threshold)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        b = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * grad_output
        thr_grad = torch.sum(-(threshold / b) * rectangle((x - threshold) / b) * grad_output, dim=0)
        return x_grad, thr_grad, None


class JumpReLUSAE(nn.Module):
    def __init__(self, input_dim, dict_size, l0_coeff=4e-3, bandwidth=0.1,
                 init_threshold=0.5, k_aux=256, aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, dict_size)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        # threshold in log-space (theta = exp(log_theta) > 0). Init IN the activation range so the
        # rectangle kernel actually sees activations near it (the bug before: init 0.001 << act scale
        # 0.6 -> kernel window empty -> threshold never trained).
        self.log_theta = nn.Parameter(torch.full((dict_size,), float(np.log(init_threshold))))
        self.dict_size = dict_size
        self.l0_coeff = l0_coeff
        self.bandwidth = bandwidth
        self.k_aux = k_aux
        self.aux_alpha = aux_alpha
        self.n_batches_to_dead = n_batches_to_dead
        self.register_buffer("num_batches_not_active", torch.zeros(dict_size))

    @property
    def threshold(self):
        return torch.exp(self.log_theta)

    def forward(self, x):
        theta = self.threshold
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        pre = F.relu(pre)                      # JumpReLU operates on ReLU pre-acts
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

        # AuxK resurrection of persistently dead features (same idea as BatchTopK trainer)
        aux = torch.tensor(0.0, device=x.device)
        if self.training and self.k_aux > 0:
            dead = self.num_batches_not_active >= self.n_batches_to_dead
            if dead.sum() > 0:
                err = (x - x_hat).detach()
                dpre = pre[:, dead]
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
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "activations" in cache:
        data = cache["activations"].float()
    elif "blunder_mt" in cache:
        data = cache["blunder_mt"].float()
    else:
        raise ValueError(f"unknown cache keys: {list(cache.keys())}")
    cm, cs = cache.get("mean"), cache.get("std")
    if cm is not None:
        mean = torch.tensor(np.array(cm), dtype=torch.float32)
        std = torch.tensor(np.array(cs), dtype=torch.float32).clamp(min=1e-6)
    else:
        mean = data.mean(dim=0); std = data.std(dim=0).clamp(min=1e-6)
    return (data - mean) / std, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dict-size", type=int, default=2048)
    ap.add_argument("--l0-coeff", type=float, default=4e-3)
    ap.add_argument("--bandwidth", type=float, default=0.1)   # kernel width; must span the act scale
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    data, mean, std = load_data(args.cache)
    n, input_dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} positions, {input_dim}-dim, device={device}", flush=True)

    sae = JumpReLUSAE(input_dim, args.dict_size, l0_coeff=args.l0_coeff, bandwidth=args.bandwidth).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    print(f"dict={args.dict_size} l0_coeff={args.l0_coeff} epochs={args.epochs}", flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        eL2 = eL0 = eAux = nb = 0
        sae.train()
        for i in range(0, n, args.batch_size):
            batch = data[perm[i:i+args.batch_size]].to(device)
            loss, _, acts, l2, l0p, aux = sae(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            sae.make_decoder_weights_unit_norm()
            eL2 += l2.item(); eL0 += l0p.item(); eAux += aux.item(); nb += 1
        # eval-mode L0 (real active count) + dead count
        sae.eval()
        with torch.no_grad():
            sample = data[torch.randperm(n)[:8192]].to(device)
            _, _, ea, _, _, _ = sae(sample)
            real_l0 = (ea > 0).float().sum(dim=-1).mean().item()
            # feature fire rates -> long-tail check
            fire_rate = (ea > 0).float().mean(dim=0)
            dead = (fire_rate == 0).sum().item()
            hog = (fire_rate > 0.10).sum().item()   # features firing >10% = blobs (the long tail)
        print(f"  ep{ep+1}/{args.epochs} l2={eL2/nb:.5f} l0pen={eL0/nb:.4f} aux={eAux/nb:.5f} "
              f"| eval_L0={real_l0:.1f} dead={dead} blobs>10%={hog}", flush=True)

    torch.save({"state_dict": sae.state_dict(), "mean": mean, "std": std,
                "config": {"input_dim": input_dim, "dict_size": args.dict_size,
                           "l0_coeff": args.l0_coeff, "arch": "jumprelu"}}, args.output)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
