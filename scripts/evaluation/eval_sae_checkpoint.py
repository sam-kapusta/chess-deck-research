#!/usr/bin/env python3
"""Evaluate a trained SAE checkpoint: dead features, L0, FVU, c_dec.

Usage (on SAIS):
    python3 eval_sae_checkpoint.py /path/to/sae_btk_4096_k32_aux.pt
    python3 eval_sae_checkpoint.py /path/to/sae_btk_4096_k32_aux.pt --n-sample 50000
"""
import argparse, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
CACHE = BASE + "/cache/puzzle_acts_200k.pt"


class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k
        self.dd = dd

    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros_like(z)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to SAE .pt file")
    parser.add_argument("--n-sample", type=int, default=50000)
    parser.add_argument("--output", help="Save results JSON to this path")
    args = parser.parse_args()

    print(f"Loading cache: {CACHE}")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    flat = cache["token_acts"].float().reshape(-1, 1024)
    print(f"  Flat shape: {flat.shape}")

    print(f"Loading checkpoint: {args.checkpoint}")
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dict_size = weights["dict_size"]
    k = weights["k"]
    print(f"  dict_size={dict_size}, k={k}")

    sae = SAE(1024, dict_size, k)
    sae.encoder.weight.data = weights["encoder_weight"]
    sae.encoder.bias.data = weights["encoder_bias"]
    sae.decoder.weight.data = weights["decoder_weight"]
    sae.pre_bias.data = weights["pre_bias"]
    sae = sae.cuda().eval()

    n = min(args.n_sample, flat.shape[0])
    print(f"Evaluating on {n} samples...")

    with torch.no_grad():
        sample = flat[:n].cuda()
        _, sa = sae(sample)
        dead_n = ((sa > 0).sum(dim=0) == 0).sum().item()
        l0 = (sa > 0).float().sum(dim=-1).mean().item()
        recon, _ = sae(sample)
        fvu = F.mse_loss(recon, sample).item() / sample.var().item()
        dec_n = F.normalize(sae.decoder.weight.data, dim=0)
        cos = dec_n.T @ dec_n
        mask = ~torch.eye(dict_size, dtype=torch.bool, device=cos.device)
        c_dec = cos[mask].abs().mean().item()

    result = {
        "dict": dict_size,
        "k": k,
        "dead": dead_n,
        "active": dict_size - dead_n,
        "l0": round(l0, 1),
        "fvu": round(fvu, 4),
        "c_dec": round(c_dec, 6),
        "n_sample": n,
        "checkpoint": os.path.basename(args.checkpoint),
    }

    print(f"\nRESULTS:")
    print(f"  dead={dead_n}/{dict_size} active={dict_size-dead_n}")
    print(f"  L0={l0:.1f} FVU={fvu:.4f} c_dec={c_dec:.6f}")

    out_path = args.output
    if not out_path:
        # Default: same dir as checkpoint, replace sae_ with eval_
        base = os.path.basename(args.checkpoint).replace("sae_", "eval_").replace(
            ".pt", ".json"
        )
        out_path = os.path.join(os.path.dirname(args.checkpoint), base)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
