#!/usr/bin/env python3
"""Build per-feature PEAK + MEDIAN board profiles for relabeling.

Why: labeling from top-10 (the p99 peak) over-specifies the piece — the peak boards are
piece-homogeneous (e.g. f103's peak is queen knight-forks), but at the MEDIAN activation the same
feature fires on rook/king forks too. Labeling from peak alone produced "Hangs queen to knight
fork" when the feature is really "Knight fork (major piece)". This profiler samples BOTH bands so
the relabeler sees the true spread.

For each live feature: encode the corpus through the SAE (BatchTopK k=6), take the firing
positions, and sample up to N_PEAK Opus-covered positions from the top of the activation range and
N_MED Opus-covered positions from the median band (p40-60). Writes a profile {fid: {peak:[...],
median:[...]}} of (fen, uci, act).

Run on chess-poc.
  python3 build_peak_median_profiles.py --out peak_median_profiles_d2048_k6.json [--only 103,1536]
"""
import argparse, json
import numpy as np, torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
ap.add_argument("--weights", default="/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k6_nol2.pt")
ap.add_argument("--positions", default="/home/ec2-user/SageMaker/all_positions_labeled_opus.json")
ap.add_argument("--out", required=True)
ap.add_argument("--n-peak", type=int, default=10)
ap.add_argument("--n-med", type=int, default=10)
ap.add_argument("--only", default="", help="comma fids to limit (testing)")
a = ap.parse_args()
ONLY = set(int(x) for x in a.only.split(",") if x.strip())

cache = torch.load(a.cache, map_location="cpu", weights_only=False)
craw = cache["activations"].float(); meta = cache["metadata"]
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd
sd = torch.load(a.weights, map_location="cpu", weights_only=False)["state_dict"]
op = json.load(open(a.positions))
We, be, bd = sd["W_enc"], sd["b_enc"], sd["b_dec"]
N, D = x.shape[0], We.shape[1]
print(f"corpus {N} x {x.shape[1]} | encoding (BatchTopK k=6) ...", flush=True)

# full activation matrix after BatchTopK gating (only top-6 per row kept)
ACT = np.zeros((N, D), np.float32)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - bd) @ We + be)
    topv, topi = z.topk(6, dim=1)
    mask = torch.zeros_like(z); mask.scatter_(1, topi, 1.0)
    ACT[i:i+8192] = (z * mask).numpy()
print("encoded.", flush=True)

# precompute opus-coverage key per position
keys = [ (m.get("fen", "") + "|" + m.get("blunder_uci", "")) for m in meta ]
covered = np.array([k in op for k in keys])

fids = range(D) if not ONLY else sorted(ONLY)
out = {}
for fid in fids:
    col = ACT[:, fid]
    fired = np.where(col > 0)[0]
    if len(fired) < 5: continue
    fa = col[fired]
    # peak: highest-activation, opus-covered
    order = fired[np.argsort(-fa)]
    peak = []
    for i in order:
        if covered[i]:
            peak.append(i)
            if len(peak) >= a.n_peak: break
    # median band p40-60, opus-covered, sampled spread across the band
    lo, hi = np.percentile(fa, 40), np.percentile(fa, 60)
    band = fired[(col[fired] >= lo) & (col[fired] <= hi)]
    band_cov = [i for i in band if covered[i]]
    # even spread across the band
    med = []
    if band_cov:
        step = max(1, len(band_cov) // a.n_med)
        med = band_cov[::step][:a.n_med]
    if len(peak) < 3 and len(med) < 3: continue
    out[str(fid)] = {
        "peak": [{"fen": meta[i].get("fen"), "uci": meta[i].get("blunder_uci"),
                  "best": meta[i].get("best_uci"), "act": round(float(col[i]), 2)} for i in peak],
        "median": [{"fen": meta[i].get("fen"), "uci": meta[i].get("blunder_uci"),
                    "best": meta[i].get("best_uci"), "act": round(float(col[i]), 2)} for i in med],
    }
    if len(out) % 200 == 0: print(f"  {len(out)} features profiled", flush=True)

json.dump(out, open(a.out, "w"))
np_peak = np.mean([len(v["peak"]) for v in out.values()])
np_med = np.mean([len(v["median"]) for v in out.values()])
print(f"\nwrote {a.out} | {len(out)} features | avg peak {np_peak:.1f} median {np_med:.1f}", flush=True)
