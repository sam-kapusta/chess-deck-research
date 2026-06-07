#!/usr/bin/env python3
"""Dump MANY boards for one SAE feature across its activation range (top / upper / mid / low), so a
feature can be inspected on far more than the 10+10 the profiler keeps.

Encodes the full l7only cache through the SAE (z-score, BatchTopK at the model's k), ranks every
position by the chosen feature's activation, and samples N boards from each quartile band. Maps each
to its Opus analysis if available.

Run on chess-poc:
  cd ~/SageMaker && python3 dump_feature_boards.py \
    --weights chess-stage-a/output/maia3_sae/btk_64_k1_nol2.pt --feat 15 --n 30 \
    --positions all_positions_labeled_opus.json --out f15_boards.json
"""
import argparse, json, torch, numpy as np, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--cache", default="/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
ap.add_argument("--feat", type=int, required=True)
ap.add_argument("--n", type=int, default=30, help="boards per band")
ap.add_argument("--positions", default="")
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = torch.load(a.weights, map_location="cpu", weights_only=False)
sd = ck["state_dict"]; K = a.k or int((ck.get("config") or {}).get("k", 6))
c = torch.load(a.cache, map_location="cpu", weights_only=False)
raw = c["activations"].float(); meta = c["metadata"]
x = (raw - raw.mean(0)) / raw.std(0).clamp(min=1e-6)
We, be, bd = sd["W_enc"], sd["b_enc"], sd["b_dec"]; N, D = x.shape[0], We.shape[1]

# activation of THIS feature at every position, gated by BatchTopK (only counts when it's in the top-K)
fa = np.zeros(N)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - bd) @ We + be)
    tv, ti = z.topk(K, dim=1)
    intop = (ti == a.feat).any(dim=1)
    val = z[:, a.feat] * intop
    fa[i:i+8192] = val.numpy()

fired = np.where(fa > 0)[0]
fired = fired[np.argsort(-fa[fired])]  # high to low
nf = len(fired)
print(f"feature {a.feat}: fires on {nf} positions ({100*nf/N:.2f}%)", flush=True)

op = json.load(open(a.positions)) if a.positions else {}
def rec(idx):
    m = meta[idx]; fen = m["fen"]; uci = m.get("blunder_uci") or m.get("uci", "")
    best = m.get("best_uci", "")
    o = op.get(fen + "|" + uci)
    if isinstance(o, dict): o = o.get("analysis", o)
    out = {"fen": fen, "uci": uci, "best": best, "act": round(float(fa[idx]), 2)}
    if isinstance(o, dict):
        out["blunder_summary"] = (o.get("blunder_summary") or "")[:200]
        out["best_moves_analysis"] = (o.get("best_moves_analysis") or "")[:160]
    return out

# bands across the fired range
bands = {}
if nf:
    q = lambda lo, hi: fired[int(nf*lo):int(nf*hi)]
    for name, lo, hi in [("top", 0, .25), ("upper", .25, .5), ("mid", .5, .75), ("low", .75, 1.0)]:
        idxs = q(lo, hi)
        step = max(1, len(idxs) // a.n)
        bands[name] = [rec(int(i)) for i in idxs[::step][:a.n]]

json.dump({"feat": a.feat, "n_fired": int(nf), "bands": bands}, open(a.out, "w"))
print(f"wrote {a.out} | " + " ".join(f"{k}:{len(v)}" for k, v in bands.items()), flush=True)
