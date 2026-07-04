#!/usr/bin/env python3
"""GH #26 — per-PHASE drop_ratio for every tag (exposure-clean denominator).

The bug we kept hitting: dividing a tag's fires by TOTAL moves flattens endgame-heavy tags (experts
reach more endgames -> bigger denominator -> looks flat), while dividing by ENDGAME moves inflates
middlegame-heavy tags. Neither is right for a mixed tag.

The fix (this script): denominate each fire by the moves of the PHASE it occurred in, then sum.

  rate(tag, band) = Σ_phase  fires(tag, band, phase) / moves(band, phase)
  drop_ratio(tag) = rate(tag, LOW_band) / rate(tag, HIGH_band)

Each phase term is fires-per-move-IN-THAT-PHASE, so band-to-band exposure shifts cancel. This is the
honest skill axis for every tag at once (general + phase-specific), with the FIXED tagger
(#27 gate, exchange rename, pawn-break/prophylaxis gates).

Phase per fire: from the enrichment cache `phase` field. Phase move counts: sweep_denominators.json
(opening/middle/endgame per band). Tags: the shipped tagger run inline.

Run on chess-poc:
  /home/ec2-user/anaconda3/envs/pytorch_p310/bin/python per_phase_drop_ratio.py \
     --blunders /home/ec2-user/SageMaker/sweep_blunders_d16.json \
     --cache /home/ec2-user/SageMaker/position_enrichment_cache_d16.json \
     --denoms /home/ec2-user/SageMaker/sweep_denominators.json \
     --out /home/ec2-user/SageMaker/per_phase_drop_ratio.json
"""
import argparse, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/home/ec2-user/SageMaker")

import tagger as T
from run_rating_bands import build_mistake   # exact join (parses top_3_best/refutations, flips POV)

# cache phase strings -> denominator phase keys
PHASE_KEY = {"opening": "opening", "middlegame": "middle", "middle": "middle", "endgame": "endgame"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blunders", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--denoms", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blunders = json.load(open(args.blunders))
    cache = json.load(open(args.cache))
    den = json.load(open(args.denoms))
    bands = sorted(den.keys())
    LOW, HIGH = bands[0], bands[-1]
    print(f"blunders {len(blunders)}, cache {len(cache)}, bands {bands}", flush=True)

    # fires[label][band][phase] = count
    fires = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    tagged = 0
    t0 = time.time()
    for i, row in enumerate(blunders):
        ce = cache.get(row["fen"] + "|" + row["uci"])
        if ce is None:
            continue
        band = row["band"]
        phase = PHASE_KEY.get((ce.get("phase") or "").lower())
        if phase is None:
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
            fires[lab][band][phase] += 1
        if tagged % 5000 == 0:
            print(f"  tagged {tagged} ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"tagged {tagged} positions, {len(fires)} labels", flush=True)

    def phase_rate(lab, band):
        r = 0.0
        for ph in ("opening", "middle", "endgame"):
            f = fires[lab][band][ph]
            mv = den[band][ph]
            if mv:
                r += f / mv
        return r

    out = {}
    for lab in fires:
        total = sum(fires[lab][b][p] for b in bands for p in ("opening", "middle", "endgame"))
        if total < 150:
            continue
        rlow, rhigh = phase_rate(lab, LOW), phase_rate(lab, HIGH)
        dr = rlow / rhigh if rhigh > 1e-12 else None
        # phase mix (where the tag fires, summed over bands) — context for the denominator
        pe = {ph: sum(fires[lab][b][ph] for b in bands) for ph in ("opening", "middle", "endgame")}
        out[lab] = {
            "n_total": total,
            "phase_mix": {k: round(v / total, 2) for k, v in pe.items()},
            "rate_low_per1000": round(rlow * 1000, 3),
            "rate_high_per1000": round(rhigh * 1000, 3),
            "drop_ratio_perphase": round(dr, 3) if dr else None,
        }

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min -> {args.out}", flush=True)
    for lab, v in sorted(out.items(), key=lambda kv: (kv[1]["drop_ratio_perphase"] or 0)):
        mix = v["phase_mix"]
        print(f"  {v['drop_ratio_perphase'] or 0:5.2f}x  n={v['n_total']:<5} "
              f"O/M/E {mix['opening']:.0%}/{mix['middle']:.0%}/{mix['endgame']:.0%}  {lab}", flush=True)


if __name__ == "__main__":
    main()
