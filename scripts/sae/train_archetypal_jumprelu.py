#!/usr/bin/env python3
"""Archetypal JumpReLU SAE = OUR working JumpReLU + overcomplete's RelaxedArchetypalDictionary.

ROOT CAUSE (systematic-debugging, 2026-06-17): overcomplete's RA-JumpSAE collapses (FVU->1.0, L0->0)
because its sparsity is L1-on-MAGNITUDE (mse_l1) which shrinks the encoder pre-codes (enc_wnorm 26->5.5,
pre_code 0.46->0.013 over epochs) while its threshold is near-frozen (bandwidth=1e-3, pseudo-grad too
weak to adapt). Slow encoder shrinkage -> features fall below threshold -> gate off -> dead. NOT a
threshold-runaway. The fix is to keep the sparsity mechanism that DIDN'T collapse (ours) and graft on
only the part we want from overcomplete (the archetypal decoder = the seed-stability constraint).

OURS (FVU=0.20, trains stably): L0-COUNT penalty via Step (no magnitude shrinkage) + bandwidth=0.1
(threshold learns) + AuxK (revives dead features). We keep all of that and replace ONLY the free
decoder W_dec with RelaxedArchetypalDictionary (atoms = convex combos of data points -> seed-stable).

Run (chess-poc GPU), 2 seeds + MMCS, vs plain-JumpReLU baseline MMCS=0.41/>0.7=6%:
  python3 train_archetypal_jumprelu.py --cache <blunder_diff.pt> --dict 2048 --n-points 4096 \
     --l0-coeff 0.02 --epochs 40 --out /home/ec2-user/SageMaker/arch_jumprelu
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_jumprelu import JumpReLU, Step, rectangle, load_data
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary


class ArchetypalJumpReLUSAE(nn.Module):
    """Our JumpReLU encoder + L0 loss + AuxK, but decoder = archetypal dictionary (D = W@C + Relax)."""
    def __init__(self, input_dim, dict_size, points, l0_coeff=0.02, bandwidth=0.3,
                 init_threshold=0.5, delta=1.0, k_aux=256, aux_alpha=1/32, n_batches_to_dead=5):
        super().__init__()
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, dict_size)))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.log_theta = nn.Parameter(torch.full((dict_size,), float(np.log(init_threshold))))
        # archetypal decoder: atoms constrained to conv(points) + bounded relaxation (seed-stability)
        self.dict = RelaxedArchetypalDictionary(input_dim, dict_size, points, delta=delta,
                                                use_multiplier=True, device=points.device)
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

    def get_dictionary(self):
        return self.dict.get_dictionary()        # (dict_size, input_dim)

    def forward(self, x):
        theta = self.threshold
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        acts = JumpReLU.apply(pre, theta, self.bandwidth)
        D = self.dict.get_dictionary()
        x_hat = acts @ D + self.b_dec
        l2 = (x_hat.float() - x.float()).pow(2).mean()
        if self.training:
            l0 = Step.apply(pre, theta, self.bandwidth).sum(dim=-1).mean()
            l0_pen = self.l0_coeff * l0
            active = (acts > 0).any(dim=0)
            self.num_batches_not_active[active] = 0
            self.num_batches_not_active[~active] += 1
            aux = torch.tensor(0.0, device=x.device)
            if self.k_aux > 0:
                dead = self.num_batches_not_active >= self.n_batches_to_dead
                if dead.sum() > 0:
                    err = (x - x_hat).detach()
                    dpre = pre[:, dead]
                    ka = min(self.k_aux, int(dead.sum()))
                    tk = torch.topk(dpre, k=ka, dim=-1)
                    dacts = torch.zeros_like(dpre).scatter(-1, tk.indices, tk.values)
                    err_hat = dacts @ D[dead]
                    aux = self.aux_alpha * (err_hat.float() - err.float()).pow(2).mean()
            loss = l2 + l0_pen + aux
        else:
            loss = l2
        return loss, x_hat, acts


def train_one(data, input_dim, dict_size, points, l0_coeff, epochs, bs, lr, device, seed):
    # NOTE: `points` (archetype candidate set C) is passed in and SHARED across seeds — archetypal
    # stability assumes a FIXED candidate set; sampling it per-seed was a confound in the first run.
    torch.manual_seed(seed)
    n = data.shape[0]
    sae = ArchetypalJumpReLUSAE(input_dim, dict_size, points, l0_coeff=l0_coeff).to(device)
    # ROOT-CAUSE FIX (systematic-debugging): the threshold's pseudo-gradient is too weak to climb to
    # the activation scale at the shared LR, so L0 stayed stuck at 200+ (sparsity pressure escaped into
    # decoder-shrinking instead of gating). Give log_theta its OWN high LR so it can actually gate ->
    # L0 drops to ~16. (Plain JumpReLU avoided this via unit-norm decoder forcing the threshold to move.)
    theta_p = [sae.log_theta]
    other_p = [p for nm, p in sae.named_parameters() if nm != "log_theta"]
    opt = torch.optim.Adam([{"params": other_p, "lr": lr}, {"params": theta_p, "lr": 1e-2}])
    for ep in range(epochs):
        perm = torch.randperm(n); sae.train()
        eL0 = eF = nb = 0
        for i in range(0, n, bs):
            loss, x_hat, acts = sae(data[perm[i:i+bs]].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in (0, 4, 19, epochs-1):
            sae.eval()
            with torch.no_grad():
                s = data[torch.randperm(n)[:8192]].to(device)
                _, xh, z = sae(s)
                L0 = (z > 0).float().sum(-1).mean().item()
                fvu = ((xh-s).pow(2).sum(-1).mean()/((s-s.mean(0)).pow(2).sum(-1).mean()+1e-8)).item()
            print(f"    seed{seed} ep{ep+1}: L0={L0:.1f} FVU={fvu:.3f}", flush=True)
    return sae


@torch.no_grad()
def live_dec(sae, data, device):
    sae.eval()
    n = data.shape[0]
    fr = torch.zeros(sae.dict_size, device=device)
    for i in range(0, n, 16384):
        _, _, z = sae(data[i:i+16384].to(device)); fr += (z > 0).float().sum(0)
    live = fr > 0
    D = sae.get_dictionary().detach().to(device)
    D = D / D.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return D[live], int(live.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--l0-coeff", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data, _, _ = load_data(args.cache)
    n, input_dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{n} pos, {input_dim}-dim, device={device}; ARCH-JumpReLU dict={args.dict} pts={args.n_points}", flush=True)

    t0 = time.time()
    # SHARED archetype candidate set C across both seeds (fixes the per-seed-sampling confound).
    torch.manual_seed(12345)
    points = data[torch.randperm(n)[:args.n_points]].to(device)
    A = train_one(data, input_dim, args.dict, points, args.l0_coeff, args.epochs, args.batch_size, args.lr, device, 0)
    DA, nla = live_dec(A, data, device)
    print(f"  seed0 done live={nla} ({(time.time()-t0)/60:.1f}min)", flush=True)
    B = train_one(data, input_dim, args.dict, points, args.l0_coeff, args.epochs, args.batch_size, args.lr, device, 1)
    DB, nlb = live_dec(B, data, device)
    print(f"  seed1 done live={nlb} ({(time.time()-t0)/60:.1f}min)", flush=True)

    best = (DA @ DB.T).max(dim=1).values.cpu().numpy()
    out = {"dict": args.dict, "live0": nla, "live1": nlb,
           "MMCS": round(float(best.mean()), 3),
           "frac>0.5": round(float((best > 0.5).mean()), 3),
           "frac>0.7": round(float((best > 0.7).mean()), 3),
           "frac>0.8": round(float((best > 0.8).mean()), 3),
           "frac>0.9": round(float((best > 0.9).mean()), 3)}
    json.dump(out, open(args.out + "_stability.json", "w"), indent=2)
    torch.save({"state_dict": A.state_dict()}, args.out + "_seed0.pt")
    print("\n=== ARCHETYPAL JumpReLU seed stability (vs plain JumpReLU MMCS=0.41 >0.7=6%) ===", flush=True)
    print(f"  MMCS={out['MMCS']} >0.5={out['frac>0.5']} >0.7={out['frac>0.7']} >0.8={out['frac>0.8']} >0.9={out['frac>0.9']}", flush=True)
    print(f"DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
