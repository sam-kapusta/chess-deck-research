#!/usr/bin/env python3
"""What does Maia REPRESENT that our tagger has NO tag for? (Sam, 2026-06-17)

Reverse of the probe. Instead of "for each tag, does Maia have it?", ask "what does Maia separate
that we don't name?" Cluster Maia's blunder-diff activations (best - blunder, layer 7 — "what Maia
sees differently between the right move and the mistake"), then find clusters that are:
  - INTERNALLY COHERENT (tight in activation space), AND
  - NOT explained by any existing tag (low tag coverage / no dominant tag).
Those are candidate mistake concepts Maia captures that our rule-based tagger misses → eyeball + name.

Method:
  1. KMeans on the 200k blunder-diff activations (512-d).
  2. Tag each position with the shipped tagger (best_uci absent in this cache, so MISSED-line motifs
     won't fire — but Bad Capture / Hung / Failed / direction-agnostic tags do; enough to measure
     "is this cluster already covered by SOME tag").
  3. Per cluster: tightness (mean cosine to centroid), tag coverage (% positions with >=1 tag),
     dominant tag (most common label) + its share. A cluster that is TIGHT but has LOW coverage or
     NO dominant tag = a concept we don't capture.
  4. Dump example FENs per interesting cluster for eyeballing.

Run on chess-poc:
  /home/ec2-user/anaconda3/envs/pytorch_p310/bin/python maia_untagged_concepts.py \
     --acts /home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff_v2.pt \
     --k 60 --out /home/ec2-user/SageMaker/maia_untagged_concepts.json
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/ec2-user/SageMaker")

import numpy as np
import torch
import chess
import tagger as T
from mistake import Mistake
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


def tag_pos(fen, blunder_uci, cp_loss):
    """No best line in this cache → only played-move/material tags fire. Enough to ask 'is this
    cluster covered by ANY tag we have'."""
    try:
        m = Mistake(fen_before=fen, played_uci=blunder_uci, best_uci="",
                    best_line_san=[], refutation_san=[], eval_before=None, eval_after=None,
                    cp_loss=int(cp_loss or 0), mover=(fen.split()[1] == "w"))
        res = T.tag_mistake_full(m, with_maia=False)
        return [t["label"] for t in res["tags"] if t.get("direction") != "info"]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    d = torch.load(args.acts, map_location="cpu", weights_only=False)
    X = d["activations"].float().numpy()
    meta = d["metadata"]
    N = len(meta)
    print(f"acts {X.shape}, meta {N}", flush=True)

    Xn = normalize(X)   # cosine geometry
    km = MiniBatchKMeans(n_clusters=args.k, random_state=0, batch_size=4096, n_init=3, max_iter=200)
    labels = km.fit_predict(Xn)
    print(f"clustered into {args.k} ({(time.time()-t0)/60:.1f}min)", flush=True)

    # tag every position once
    pos_tags = [None] * N
    for i in range(N):
        pos_tags[i] = tag_pos(meta[i]["fen"], meta[i]["blunder_uci"], meta[i].get("cp_loss", 0))
        if (i + 1) % 25000 == 0:
            print(f"  tagged {i+1}/{N} ({(time.time()-t0)/60:.1f}min)", flush=True)

    cents = normalize(km.cluster_centers_)
    out = []
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        if len(idx) < 50:
            continue
        # tightness = mean cosine of members to centroid
        tight = float((Xn[idx] @ cents[c]).mean())
        # tag coverage
        tagged = sum(1 for i in idx if pos_tags[i])
        coverage = tagged / len(idx)
        tagcount = Counter(t for i in idx for t in set(pos_tags[i]))
        dom_tag, dom_n = (tagcount.most_common(1)[0] if tagcount else ("(none)", 0))
        dom_share = dom_n / len(idx)
        # example fens (closest to centroid)
        sims = Xn[idx] @ cents[c]
        order = idx[np.argsort(-sims)]
        examples = [{"fen": meta[i]["fen"], "uci": meta[i]["blunder_uci"], "cp": meta[i].get("cp_loss")}
                    for i in order[:6]]
        out.append({
            "cluster": c, "size": int(len(idx)), "tightness": round(tight, 3),
            "tag_coverage": round(coverage, 3), "dom_tag": dom_tag, "dom_share": round(dom_share, 3),
            "top_tags": tagcount.most_common(5), "examples": examples,
        })

    # rank: most interesting = tight AND under-explained (low dom_share)
    out.sort(key=lambda c: (c["dom_share"], -c["tightness"]))
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)
    print("\n=== clusters Maia separates that our tags DON'T explain (low dom_share, tight) ===", flush=True)
    for c in out[:20]:
        print(f"  cl{c['cluster']:<3} size={c['size']:<5} tight={c['tightness']:.2f} "
              f"cover={c['tag_coverage']:.2f} dom='{c['dom_tag']}'({c['dom_share']:.2f}) "
              f"top={[t for t,_ in c['top_tags'][:3]]}", flush=True)


if __name__ == "__main__":
    main()
