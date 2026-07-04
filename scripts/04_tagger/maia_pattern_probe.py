#!/usr/bin/env python3
"""Can Maia's POLICY head detect our mistake-tag patterns? (Sam, 2026-06-17)

Hypothesis: a tag is "Maia-detectable" if the blunder move is MORE popular among weak players
than strong players in Maia's policy — i.e. Maia predicts humans grow out of it. That mirrors the
cohort drop_ratio, but measured by Maia's human-move-prediction instead of the blunder sweep.

  maia_pop_drop(tag) = mean P(blunder move | Elo 1100) / mean P(blunder move | Elo 1900)
                       over that tag's blunders (Maia policy `played_prob`).

  >1  -> Maia "sees" the pattern: weak players play this move, strong don't (a knowledge/popularity
         blind spot Maia models). Maia could flag it without a cohort.
  ~1  -> Maia is BLIND: the move is equally (un)popular at all Elos. The mistake is about calculation
         depth, not a human-predictable pattern — Maia's policy can't surface it.

Per-position tags come from running the shipped tagger inline (CPU). Maia policy from maia3_engine.

Run on chess-poc:
  /home/ec2-user/anaconda3/envs/pytorch_p310/bin/python maia_pattern_probe.py \
     --blunders /home/ec2-user/SageMaker/sweep_blunders_d16.json \
     --cache /home/ec2-user/SageMaker/position_enrichment_cache_d16.json \
     --per-tag 150 --elo-low 1100 --elo-high 1900 \
     --out /home/ec2-user/SageMaker/maia_pattern_probe.json
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/ec2-user/SageMaker")

import maia3_engine as M
import tagger as T
# Reuse the EXACT join run_rating_bands uses — it parses top_3_best / top_3_refutations into SANs
# (line_to_sans) and flips eval POV. Rolling my own dropped the best line, so only played-move tags
# fired (8 labels, no "Missed X"). Import it so the probe sees the same tags as the band stats.
from run_rating_bands import build_mistake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blunders", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--per-tag", type=int, default=150)
    ap.add_argument("--elo-low", type=int, default=1100)
    ap.add_argument("--elo-high", type=int, default=1900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blunders = json.load(open(args.blunders))
    cache = json.load(open(args.cache))
    print(f"blunders {len(blunders)}, cache {len(cache)}", flush=True)

    # 1) tag every blunder (inline), bucket positions by tag label
    by_tag = defaultdict(list)   # label -> [(fen, uci), ...]
    tagged = 0
    for row in blunders:
        ce = cache.get(row["fen"] + "|" + row["uci"])
        if ce is None:
            continue
        try:
            res = T.tag_mistake_full(build_mistake(row, ce), with_maia=False)
        except Exception:
            continue
        tagged += 1
        seen = set()
        for tg in res["tags"]:
            if tg.get("direction") == "info":
                continue
            lab = tg["label"]
            if lab in seen:
                continue
            seen.add(lab)
            by_tag[lab].append((row["fen"], row["uci"]))
    print(f"tagged {tagged} positions, {len(by_tag)} distinct labels", flush=True)

    # 2) for a stratified sample per tag, query Maia played_prob at low & high Elo
    out = {}
    t0 = time.time()
    for lab in sorted(by_tag, key=lambda k: -len(by_tag[k])):
        positions = by_tag[lab]
        if len(positions) < 30:   # need a minimum sample for a stable mean
            continue
        sample = positions[: args.per_tag]   # deterministic (no RNG in workflow-safe code)
        lows, highs, ok = [], [], 0
        for fen, uci in sample:
            try:
                rl = M.analyze(fen, args.elo_low, top_k=1, played_uci=uci)
                rh = M.analyze(fen, args.elo_high, top_k=1, played_uci=uci)
            except Exception:
                continue
            pl, ph = rl.get("played_prob"), rh.get("played_prob")
            if pl is None or ph is None:
                continue
            lows.append(pl); highs.append(ph); ok += 1
        if ok < 20:
            continue
        mean_low = sum(lows) / len(lows)
        mean_high = sum(highs) / len(highs)
        pop_drop = mean_low / mean_high if mean_high > 1e-9 else None
        out[lab] = {
            "n_total": len(positions), "n_sampled": ok,
            "maia_pop_low": round(mean_low, 4), "maia_pop_high": round(mean_high, 4),
            "maia_pop_drop": round(pop_drop, 3) if pop_drop else None,
        }
        print(f"  {lab:<28} n={ok:<4} pop@{args.elo_low}={mean_low:.3f} "
              f"pop@{args.elo_high}={mean_high:.3f} drop={pop_drop:.2f}x", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
