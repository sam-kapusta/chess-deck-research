#!/usr/bin/env python3
"""Comprehensive SAE evaluation battery — label-free + tagger-based. (Sam, 2026-06-17)

FVU/dead/blob/dup@0.9 only said "is this SAE pathological?". This adds the metrics that say
"is it a GOOD dictionary?" — across 6 families. Everything here is label-free EXCEPT tag-purity,
which uses the rule tagger (no LLM). Runs per (dict, l0_coeff); trains then evaluates.

Families & metrics:
  RECON   fvu (var unexplained) · recon_cos (DIRECTIONAL fidelity — separate from magnitude)
  SPARSE  L0_mean · L0_median · L0_p90
  HEALTH  dead% · blob%(>10%) · fire_gini · act_density (mean |act| when active)
  GEOM    decoder max-cos distribution at 0.5/0.7/0.85/0.9 (splitting at REALISTIC thresholds,
          not just near-identical) · mean_pairwise_cos · participation_ratio (effective #
          distinct decoder directions = (Σλ)²/Σλ² of the Gram spectrum, normalized by n_live) ·
          enc_dec_align (mean cos between a feature's encoder row and its decoder col)
  PURITY  tag_purity (for each live feature, top-N positions: max share of any single tag) — the
          monosemanticity proxy. mean + fraction of features with purity>0.5. (tagger, no LLM.)

Usage (chess-poc GPU):
  python3 sae_metric_battery.py --cache <blunder_diff.pt> --dicts 512,1024,2048,4096 \
     --l0 0.02 --epochs 40 --purity-topn 30 --purity-feats 300 \
     --out /home/ec2-user/SageMaker/sae_metric_battery.json
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")   # tagger.py, mistake.py live here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_jumprelu import JumpReLUSAE, load_data
import chess
import tagger as T
from mistake import Mistake


def tag_pos(meta_row):
    try:
        fen = meta_row["fen"]
        m = Mistake(fen_before=fen, played_uci=meta_row["blunder_uci"], best_uci="",
                    best_line_san=[], refutation_san=[], eval_before=None, eval_after=None,
                    cp_loss=int(meta_row.get("cp_loss", 0) or 0), mover=(fen.split()[1] == "w"))
        return set(t["label"] for t in T.tag_mistake_full(m, with_maia=False)["tags"]
                   if t.get("direction") != "info")
    except Exception:
        return set()


def evaluate(sae, data, meta, device, purity_topn, purity_feats):
    sae.eval()
    n = data.shape[0]
    with torch.no_grad():
        # encode a big sample for recon/sparsity/health
        s_idx = torch.randperm(n)[:16384]
        s = data[s_idx].to(device)
        _, x_hat, acts, _, _, _ = sae(s)

        # RECON
        fvu = ((x_hat - s).pow(2).sum(-1).mean() / ((s - s.mean(0)).pow(2).sum(-1).mean() + 1e-8)).item()
        recon_cos = torch.nn.functional.cosine_similarity(x_hat, s, dim=-1).mean().item()

        # SPARSE
        l0_per = (acts > 0).float().sum(-1)
        L0_mean, L0_med, L0_p90 = l0_per.mean().item(), l0_per.median().item(), torch.quantile(l0_per, 0.9).item()

        # HEALTH
        fire = (acts > 0).float().mean(0)
        dead = (fire == 0).float().mean().item()
        blobs = (fire > 0.10).float().mean().item()
        live = fire > 0
        frl = fire[live].sort().values
        nlive = frl.numel()
        ii = torch.arange(1, nlive + 1, device=frl.device, dtype=frl.dtype)
        gini = ((2 * ii - nlive - 1) * frl).sum().item() / (nlive * frl.sum().item() + 1e-9)
        nz = acts[acts > 0]
        act_density = nz.mean().item() if nz.numel() else 0.0

        # GEOM (decoder is unit-norm)
        Wd = sae.W_dec[live]
        sim = Wd @ Wd.T
        sim.fill_diagonal_(-1.0)
        max_cos = sim.max(dim=1).values
        cos_ge = {f"dup>{t}": (max_cos > t).float().mean().item() for t in (0.5, 0.7, 0.85, 0.9)}
        mean_pair = sim[sim > -1].mean().item()
        # participation ratio of decoder direction spread (effective # distinct directions / n_live)
        gram = Wd @ Wd.T
        evals = torch.linalg.eigvalsh(gram).clamp(min=0)
        pr = (evals.sum() ** 2 / (evals.pow(2).sum() + 1e-9)).item()
        pr_frac = pr / nlive
        # encoder-decoder alignment: cos between W_enc[:,j] and W_dec[j,:]
        We = sae.W_enc[:, live]                       # (input, n_live)
        enc_dec = torch.nn.functional.cosine_similarity(We.T, Wd, dim=-1).mean().item()

    # PURITY (tagger) — sample features, tag their top positions on the FULL set
    with torch.no_grad():
        feats_all = []
        for i in range(0, n, 8192):
            _, _, a, _, _, _ = sae(data[i:i+8192].to(device))
            feats_all.append(a.cpu())
        Fmat = torch.cat(feats_all, 0).numpy()
    live_idx = np.where((Fmat > 0).mean(0) > 0)[0]
    # sample live, non-blob features
    fr_all = (Fmat > 0).mean(0)
    cand = [f for f in live_idx if fr_all[f] <= 0.10]
    cand = cand[:purity_feats]
    tagcache = {}
    purities = []
    for f in cand:
        top = np.argsort(-Fmat[:, f])[:purity_topn]
        top = [int(i) for i in top if Fmat[i, f] > 0]
        if len(top) < 10:
            continue
        cnt = Counter()
        for i in top:
            if i not in tagcache:
                tagcache[i] = tag_pos(meta[i])
            for t in tagcache[i]:
                cnt[t] += 1
        if cnt:
            purities.append(cnt.most_common(1)[0][1] / len(top))   # share of dominant tag
        else:
            purities.append(0.0)   # no tag at all = uncovered (could be missing concept OR untaggable)
    purities = np.array(purities) if purities else np.array([0.0])
    return {
        "fvu": round(fvu, 4), "recon_cos": round(recon_cos, 4),
        "L0_mean": round(L0_mean, 1), "L0_median": round(L0_med, 1), "L0_p90": round(L0_p90, 1),
        "dead": round(dead, 3), "blob": round(blobs, 3), "gini": round(gini, 3),
        "act_density": round(act_density, 3),
        **{k: round(v, 3) for k, v in cos_ge.items()},
        "mean_pair_cos": round(mean_pair, 4), "particip_ratio_frac": round(pr_frac, 3),
        "enc_dec_align": round(enc_dec, 3),
        "tag_purity_mean": round(float(purities.mean()), 3),
        "tag_purity_gt50pct": round(float((purities > 0.5).mean()), 3),
        "n_purity_feats": len(purities),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dicts", default="512,1024,2048,4096")
    ap.add_argument("--l0", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--purity-topn", type=int, default=30)
    ap.add_argument("--purity-feats", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data, _, _ = load_data(args.cache)
    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta = d["metadata"]
    n, input_dim = data.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dicts = [int(x) for x in args.dicts.split(",")]
    print(f"{n} pos, {input_dim}-dim, device={device}; dicts={dicts} l0={args.l0}", flush=True)

    def train(ds, seed):
        torch.manual_seed(seed)
        sae = JumpReLUSAE(input_dim, ds, l0_coeff=args.l0).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            perm = torch.randperm(n); sae.train()
            for i in range(0, n, args.batch_size):
                loss, _, _, _, _, _ = sae(data[perm[i:i+args.batch_size]].to(device))
                opt.zero_grad(); loss.backward(); opt.step(); sae.make_decoder_weights_unit_norm()
        return sae

    def live_dec(sae):
        with torch.no_grad():
            feats = []
            for i in range(0, n, 16384):
                _, _, a, _, _, _ = sae(data[i:i+16384].to(device))
                feats.append((a > 0).float().mean(0).cpu())
            fire = torch.stack(feats).mean(0) if len(feats) > 1 else feats[0]
            live = fire > 0
            return sae.W_dec[live].detach()              # unit-norm rows

    def mmcs(A, B):
        """For each row of A, max cosine to any row of B; report mean (MMCS) + frac matched >0.8."""
        S = A @ B.T                                       # both unit-norm
        best = S.max(dim=1).values
        return round(best.mean().item(), 3), round((best > 0.8).float().mean().item(), 3)

    t0 = time.time()
    rows = []
    saes = {}   # dict -> trained SAE (seed 0)
    for ds in dicts:
        sae = train(ds, seed=0)
        saes[ds] = sae
        m = evaluate(sae, data, meta, device, args.purity_topn, args.purity_feats)
        m["dict"] = ds
        rows.append(m)
        print(f"  dict={ds}: " + " ".join(f"{k}={v}" for k, v in m.items() if k != "dict")
              + f"  ({(time.time()-t0)/60:.1f}min)", flush=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    # CROSS-SAE decoder cosine (stability + version comparison)
    print("\n=== cross-SAE decoder MMCS (feature stability/overlap) ===", flush=True)
    cross = {}
    # (a) seed stability: retrain the largest dict with a different seed, match A<->B
    big = max(dicts)
    sae_b = train(big, seed=1)
    A, B = live_dec(saes[big]), live_dec(sae_b)
    mean_ab, frac_ab = mmcs(A, B)
    cross[f"seed_stability_dict{big}"] = {"mmcs": mean_ab, "frac_matched>0.8": frac_ab,
                                          "n_A": A.shape[0], "n_B": B.shape[0]}
    print(f"  seed stability dict={big}: MMCS={mean_ab} frac>0.8={frac_ab} (A={A.shape[0]} B={B.shape[0]})", flush=True)
    # (b) version overlap: do smaller-dict features reappear in the bigger dict?
    for ds in dicts:
        if ds == big: continue
        A = live_dec(saes[ds])                            # smaller dict features
        m_, f_ = mmcs(A, live_dec(saes[big]))             # best match into the big dict
        cross[f"dict{ds}_into_dict{big}"] = {"mmcs": m_, "frac_matched>0.8": f_, "n_small": A.shape[0]}
        print(f"  dict={ds} -> dict={big}: MMCS={m_} frac>0.8={f_} (n={A.shape[0]})", flush=True)

    json.dump({"per_sae": rows, "cross_sae": cross}, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
