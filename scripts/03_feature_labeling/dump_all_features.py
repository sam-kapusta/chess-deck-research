#!/usr/bin/env python3
"""Encode the corpus ONCE and dump sampled boards for EVERY feature across activation bands.
Output feeds a local predicate analyzer (no torch needed downstream).

  cd ~/SageMaker && python3 dump_all_features.py \
    --weights chess-stage-a/output/maia3_sae/btk_64_k1_nol2.pt --n 15 --out all_feat_boards_d64_k1.json
"""
import argparse, json, torch, numpy as np, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--cache", default="/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
ap.add_argument("--n", type=int, default=15, help="boards per band")
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = torch.load(a.weights, map_location="cpu", weights_only=False)
sd = ck["state_dict"]; K = a.k or int((ck.get("config") or {}).get("k", 6))
c = torch.load(a.cache, map_location="cpu", weights_only=False)
raw = c["activations"].float(); meta = c["metadata"]
x = (raw - raw.mean(0)) / raw.std(0).clamp(min=1e-6)
We, be, bd = sd["W_enc"], sd["b_enc"], sd["b_dec"]; N, D = x.shape[0], We.shape[1]

# full gated activation matrix (only top-K per row kept)
ACT = np.zeros((N, D), np.float32)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - bd) @ We + be)
    tv, ti = z.topk(K, dim=1)
    m = torch.zeros_like(z).scatter_(1, ti, 1.0)
    ACT[i:i+8192] = (z * m).numpy()
print(f"encoded {N}x{D} (k={K})", flush=True)


def rec(idx):
    mm = meta[idx]
    return {"fen": mm["fen"], "uci": mm.get("blunder_uci") or mm.get("uci", ""),
            "best": mm.get("best_uci", ""), "act": round(float(ACT[idx, fid]), 2)}


out = {}
for fid in range(D):
    fa = ACT[:, fid]
    fired = np.where(fa > 0)[0]
    if len(fired) == 0:
        continue
    fired = fired[np.argsort(-fa[fired])]
    nf = len(fired)
    bands = {}
    for name, lo, hi in [("top", 0, .25), ("upper", .25, .5), ("mid", .5, .75), ("low", .75, 1.0)]:
        idxs = fired[int(nf*lo):int(nf*hi)]
        step = max(1, len(idxs) // a.n)
        bands[name] = [rec(int(i)) for i in idxs[::step][:a.n]]
    out[str(fid)] = {"n_fired": int(nf), "bands": bands}

json.dump(out, open(a.out, "w"))
print(f"wrote {a.out} | {len(out)} features", flush=True)
