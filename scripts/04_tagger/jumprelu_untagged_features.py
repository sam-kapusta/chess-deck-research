#!/usr/bin/env python3
"""Which JumpReLU features encode concepts our tagger has NO tag for? (Sam, 2026-06-17)

Now that we have a clean JumpReLU SAE (L0~32, 0 dead, few blobs) on Maia blunder-diff activations,
find features that are MONOSEMANTIC (coherent top positions) but NOT explained by any existing tag —
those are mistake concepts Maia represents that our rule-based tagger misses.

Per feature:
  - top-N positions by activation
  - of those, what fraction carry ANY tag, and the dominant tag + its share
  - fire rate (skip blobs >10% and dead/too-rare)
A feature whose top positions are mostly UNTAGGED (low coverage) or have NO dominant tag (incoherent
wrt our taxonomy) but ARE coherent to the eye = a candidate missing concept -> dump boards to name.

Tags here come from the played-move detectors only (this cache has no best line), so "untagged" means
"not even a played-move/material tag" — a conservative bar. We surface those for human eyeballing.

Run on chess-poc:
  /home/ec2-user/anaconda3/envs/pytorch_p310/bin/python jumprelu_untagged_features.py \
     --sae /home/ec2-user/SageMaker/sae_jumprelu_blunderdiff_2048.pt \
     --acts /home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff_v2.pt \
     --topn 40 --out /home/ec2-user/SageMaker/jumprelu_untagged_features.json
"""
import argparse, json, os, sys, time
from collections import Counter

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/ec2-user/SageMaker")

import numpy as np
import torch
import chess
import tagger as T
from mistake import Mistake
from train_jumprelu import JumpReLUSAE  # reuse the arch  (scripts/sae on path via run dir)


def tag_pos(fen, uci, cp):
    try:
        m = Mistake(fen_before=fen, played_uci=uci, best_uci="", best_line_san=[], refutation_san=[],
                    eval_before=None, eval_after=None, cp_loss=int(cp or 0), mover=(fen.split()[1] == "w"))
        return [t["label"] for t in T.tag_mistake_full(m, with_maia=False)["tags"] if t.get("direction") != "info"]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--topn", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sae = JumpReLUSAE(cfg["input_dim"], cfg["dict_size"], l0_coeff=cfg.get("l0_coeff", 4e-3))
    sae.load_state_dict(ck["state_dict"]); sae.eval()

    d = torch.load(args.acts, map_location="cpu", weights_only=False)
    X = d["activations"].float()
    meta = d["metadata"]
    mean = torch.tensor(np.array(ck["mean"]), dtype=torch.float32)
    std = torch.tensor(np.array(ck["std"]), dtype=torch.float32).clamp(min=1e-6)
    Xs = (X - mean) / std
    N = Xs.shape[0]
    print(f"acts {tuple(Xs.shape)}, meta {N}", flush=True)

    # encode all positions -> feature activations (batched, eval mode = hard threshold)
    feats = []
    with torch.no_grad():
        for i in range(0, N, 8192):
            _, _, a, _, _, _ = sae(Xs[i:i+8192])
            feats.append(a)
    F = torch.cat(feats, 0).numpy()    # (N, dict_size)
    fire_rate = (F > 0).mean(axis=0)
    print(f"encoded; mean L0={ (F>0).sum(1).mean():.1f}", flush=True)

    # tag the union of all features' top positions (lazily, cache by index)
    tag_cache = {}
    def tags_for(i):
        if i not in tag_cache:
            tag_cache[i] = set(tag_pos(meta[i]["fen"], meta[i]["blunder_uci"], meta[i].get("cp_loss", 0)))
        return tag_cache[i]

    results = []
    n_done = 0
    for f in range(cfg["dict_size"]):
        fr = float(fire_rate[f])
        if fr < 0.001 or fr > 0.10:    # skip dead/too-rare and blobs
            continue
        n_done += 1
        if n_done % 100 == 0:
            print(f"  feature {n_done} ({(time.time()-t0)/60:.1f}min, {len(tag_cache)} tagged)", flush=True)
        top = np.argsort(-F[:, f])[:args.topn]
        top = [int(i) for i in top if F[i, f] > 0]
        if len(top) < 10:
            continue
        tagcount = Counter(t for i in top for t in tags_for(i))
        covered = sum(1 for i in top if tags_for(i))
        coverage = covered / len(top)
        dom, dom_n = (tagcount.most_common(1)[0] if tagcount else ("(none)", 0))
        results.append({
            "feature": f, "fire_rate": round(fr, 4), "n_top": len(top),
            "tag_coverage": round(coverage, 3), "dom_tag": dom, "dom_share": round(dom_n/len(top), 3),
            "top_tags": tagcount.most_common(4),
            "examples": [{"fen": meta[i]["fen"], "uci": meta[i]["blunder_uci"], "cp": meta[i].get("cp_loss")} for i in top[:6]],
        })

    # most interesting = low coverage (untagged) first, then low dom_share (incoherent wrt tags)
    results.sort(key=lambda r: (r["tag_coverage"], r["dom_share"]))
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min, {len(results)} features -> {args.out}", flush=True)
    print("\n=== features Maia has that our tags DON'T explain (low coverage, top of list) ===", flush=True)
    for r in results[:25]:
        print(f"  f{r['feature']:<4} fire={r['fire_rate']:.3f} cover={r['tag_coverage']:.2f} "
              f"dom='{r['dom_tag']}'({r['dom_share']:.2f}) top={[t for t,_ in r['top_tags'][:3]]}", flush=True)


if __name__ == "__main__":
    main()
