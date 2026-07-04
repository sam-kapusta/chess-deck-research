#!/usr/bin/env python3
"""Does Maia 3 internally REPRESENT our mistake-tag patterns? — linear probe, NO SAE. (Sam, 2026-06-17)

Question: not "does Maia's policy predict the move" (behavioral) and not "which SAE feature fires"
(SAE is noisy — long-tail BatchTopK, blobs, mislabels). Instead: is each mistake concept LINEARLY
DECODABLE from Maia's raw layer-7 activations? That's the clean test of "is the concept in Maia at all",
independent of any dictionary.

Method: for each tag, train a logistic-regression probe on the cached 1024-dim layer-7 activations:
  positives = positions where the tag fires; negatives = balanced random sample where it doesn't.
  Report 5-fold cross-validated ROC-AUC.

  AUC ~0.5  -> Maia does NOT linearly separate this concept (no SAE will recover it; it's a SF/rule
               concept, not something Maia represents). STOP chasing a better SAE for it.
  AUC high  -> Maia DOES represent it linearly. If the SAE feature is messy, that's the SAE's fault
               (JumpReLU/Matryoshka could help) — the concept is genuinely there.

Activations: maia3_l7only_v2_dedup.pt (168,132 x 1024, keyed by fen+blunder_uci). No GPU — pure CPU
logistic regression on cached vectors. Tags: the shipped (fixed) tagger run inline on each position.

Run on chess-poc:
  /home/ec2-user/anaconda3/envs/pytorch_p310/bin/python maia_linear_probe.py \
     --acts /home/ec2-user/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
     --min-fires 200 --neg-per-pos 1 --out /home/ec2-user/SageMaker/maia_linear_probe.json
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/ec2-user/SageMaker")

import numpy as np
import torch
import chess
import tagger as T
from mistake import Mistake
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# deterministic negative sampling (no RNG that breaks resume / reproducibility)
RNG = np.random.RandomState(0)


def tag_position(fen, blunder_uci, best_uci, cp_loss):
    """Tag a single cached position with the shipped tagger. We have fen + blunder + best + cp_loss
    but NOT the full best line / refutation — so MISSED-line motifs that need the PV won't fire here.
    That's fine: the probe asks whether Maia represents whatever tag DOES fire on this position; we
    only probe tags with enough fires. We pass best_uci as a 1-ply best line so material/exchange
    predicates (which key off the best move) work; played-move detectors work directly."""
    try:
        b = chess.Board(fen)
        best_san = b.san(chess.Move.from_uci(best_uci)) if best_uci else ""
    except Exception:
        best_san = ""
    m = Mistake(
        fen_before=fen, played_uci=blunder_uci, best_uci=best_uci or "",
        best_line_san=[best_san] if best_san else [], refutation_san=[],
        eval_before=None, eval_after=None, cp_loss=int(cp_loss or 0),
        mover=(fen.split()[1] == "w"),
    )
    try:
        res = T.tag_mistake_full(m, with_maia=False)
    except Exception:
        return []
    return [t["label"] for t in res["tags"] if t.get("direction") != "info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--min-fires", type=int, default=200)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--max-per-class", type=int, default=2000)  # cap for probe speed
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    d = torch.load(args.acts, map_location="cpu", weights_only=False)
    X = d["activations"].float().numpy()           # (N, 1024)
    meta = d["metadata"]                            # list of {fen, blunder_uci, best_uci, cp_loss, is_white}
    N = len(meta)
    print(f"activations {X.shape}, metadata {N}", flush=True)

    # tag every cached position, collect fire-index sets per label
    fires = defaultdict(list)   # label -> [row idx]
    for i, mrow in enumerate(meta):
        labs = tag_position(mrow["fen"], mrow["blunder_uci"], mrow.get("best_uci", ""), mrow.get("cp_loss", 0))
        for lab in set(labs):
            fires[lab].append(i)
        if (i + 1) % 20000 == 0:
            print(f"  tagged {i+1}/{N} ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"tagged all; {len(fires)} labels", flush=True)

    # standardize activations once (use stored mean/std if present, else fit)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    all_idx = np.arange(N)
    out = {}
    for lab in sorted(fires, key=lambda k: -len(fires[k])):
        pos = np.array(fires[lab])
        if len(pos) < args.min_fires:
            continue
        pos_set = set(pos.tolist())
        # balanced negatives: random rows where the tag does NOT fire
        neg_pool = np.array([i for i in all_idx if i not in pos_set])
        n_pos = min(len(pos), args.max_per_class)
        n_neg = min(len(neg_pool), n_pos * args.neg_per_pos)
        pos_s = pos if len(pos) <= n_pos else RNG.choice(pos, n_pos, replace=False)
        neg_s = RNG.choice(neg_pool, n_neg, replace=False)
        idx = np.concatenate([pos_s, neg_s])
        y = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        try:
            auc = cross_val_score(clf, Xs[idx], y, cv=5, scoring="roc_auc")
        except Exception as e:
            print(f"  {lab}: probe ERR {e}", flush=True)
            continue
        out[lab] = {
            "n_fires": int(len(pos)),
            "auc_mean": round(float(auc.mean()), 3),
            "auc_std": round(float(auc.std()), 3),
        }
        print(f"  AUC {auc.mean():.3f}±{auc.std():.3f}  n={len(pos):<5} {lab}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)
    print("\n=== ranked: does Maia linearly represent it? ===", flush=True)
    for lab, v in sorted(out.items(), key=lambda kv: -kv[1]["auc_mean"]):
        verdict = "REPRESENTED" if v["auc_mean"] >= 0.75 else ("weak" if v["auc_mean"] >= 0.6 else "NOT (≈chance)")
        print(f"  {v['auc_mean']:.3f}  n={v['n_fires']:<5} {lab:<30} {verdict}", flush=True)


if __name__ == "__main__":
    main()
