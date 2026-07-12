#!/usr/bin/env python3
"""Label JumpReLU SAE features by TAGGER VOTE (Sam's hint: use the tagger as the labeling seed).

For each feature: encode the whole l7-diff cache, take the top-N positions by activation, run the
rule-based tagger on each (fen + blunder_uci + best_uci, giving best_line_san=[best_uci_san] so the
1-ply MISSED predicate detectors + FAILED motifs fire), and name the feature by its DOMINANT non-info
tagger label. Deterministic, free, grounded — replaces the Opus-reads-20-boards pipeline for the
labelable subset. Features with no dominant tag (< min_frac agreement) are left "unlabeled" (honest —
these are the mechanism-ceiling / polysemantic ones).

Coverage caveat: the cache has best_uci (move 1) but NOT the full best-line PV, so multi-ply MOTIF
tags (Missed Fork-in-3) can't fire — only 1-ply MISSED predicates (Missed Free Piece, capture/
exchange, hung material, greedy, endgame technique) + FAILED motifs. That's still a rich label set.

Usage:
  pytorch_p310/bin/python label_via_tagger.py --weights jr_sweep_out/jr_thr0.40_l00.02.pt \
      --cache chess-stage-a/cache/maia3_l7only_v2_dedup.pt --topn 200 -o feature_labels_jr_thr0.40.json
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
sys.path.insert(0, "/home/ec2-user/SageMaker")
import chess
from mistake import Mistake
from tagger import tag_mistake_full, categorize
from train_jr_canonical import JumpReLUSAEAuxK, load_data


def build_mistake(md):
    """One cache metadata entry -> Mistake. best_line_san = [best move SAN] (1 ply; no deeper PV)."""
    fen = md["fen"]; blunder = md["blunder_uci"]; best = md.get("best_uci") or ""
    b = chess.Board(fen)
    try:
        bmv = chess.Move.from_uci(blunder)
        played_san = b.san(bmv) if bmv in b.legal_moves else blunder
    except Exception:
        return None
    best_san = ""; best_line = []
    if best:
        try:
            bm = chess.Move.from_uci(best)
            if bm in b.legal_moves:
                best_san = b.san(bm); best_line = [best_san]
        except Exception:
            pass
    return Mistake(fen_before=fen, played_uci=blunder, best_uci=best,
                   best_line_san=best_line, refutation_san=[],
                   eval_before=None, eval_after=None, cp_loss=int(md.get("cp_loss") or 0),
                   mover=b.turn, played_san=played_san, best_san=best_san)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--min-frac", type=float, default=0.25)  # dominant label must be >=25% of top-N
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    data, mean, std = load_data(args.cache)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta = cache["metadata"]
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    c = ck["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAEAuxK(c["emb_size"], c["dict_size"], target_l0=c.get("target_l0",8),
                          l0_alpha=c.get("l0_alpha",4.0), bandwidth=c["bandwidth"],
                          init_threshold=c["init_threshold"]).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    n, D = data.shape
    print(f"encoding {n} positions through dict={c['dict_size']}...", flush=True)

    # Encode all positions -> per-feature activation columns. Keep on GPU, take top-N indices per feature.
    acts_cols = torch.zeros(c["dict_size"], n, dtype=torch.float32)  # feature-major (for topk per feat)
    with torch.no_grad():
        for i in range(0, n, 16384):
            b = data[i:i+16384].to(dev)
            pre = torch.relu((b - sae.b_dec) @ sae.W_enc + sae.b_enc)
            a = (pre * (pre > sae.threshold)).cpu()
            acts_cols[:, i:i+a.shape[0]] = a.T
    print("encoded; voting...", flush=True)

    labels = {}
    t0 = time.time()
    for f in range(c["dict_size"]):
        col = acts_cols[f]
        nz = int((col > 0).sum())
        if nz == 0:
            labels[str(f)] = {"label": None, "reason": "dead", "fire_rate": 0.0}
            continue
        k = min(args.topn, nz)
        top_idx = torch.topk(col, k).indices.tolist()
        votes = Counter()
        for idx in top_idx:
            m = build_mistake(meta[idx])
            if m is None:
                continue
            try:
                tags = tag_mistake_full(m, with_maia=False)["tags"]
            except Exception:
                continue
            for t in tags:
                if t["direction"] != "info":
                    votes[t["label"]] += 1
        fire_rate = round(nz / n, 4)
        if not votes:
            labels[str(f)] = {"label": None, "reason": "no_explain_tags", "fire_rate": fire_rate}
            continue
        top_label, cnt = votes.most_common(1)[0]
        frac = cnt / k
        labels[str(f)] = {
            "label": top_label if frac >= args.min_frac else None,
            "confidence": round(frac, 3),
            "category": categorize(top_label),
            "fire_rate": fire_rate,
            "top_votes": dict(votes.most_common(5)),
            "n_positions": k,
        }
        if (f + 1) % 256 == 0:
            print(f"  {f+1}/{c['dict_size']} ({(time.time()-t0)/60:.1f}min)", flush=True)

    named = sum(1 for v in labels.values() if v.get("label"))
    with open(args.output, "w") as fh:
        json.dump({"weights": os.path.basename(args.weights), "topn": args.topn,
                   "min_frac": args.min_frac, "n_features": c["dict_size"],
                   "n_labeled": named, "labels": labels}, fh, indent=1)
    print(f"\nDONE {(time.time()-t0)/60:.1f}min -> {args.output} | {named}/{c['dict_size']} labeled "
          f"({100*named/c['dict_size']:.0f}%)", flush=True)
    # category distribution of labeled features
    cats = Counter(v["category"] for v in labels.values() if v.get("label"))
    print("category distribution:", dict(cats.most_common()), flush=True)


if __name__ == "__main__":
    main()
