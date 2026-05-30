"""Recompute Mechanism + Thinking-Error scheme spreads from AGENT assignments
(not keyword proxies). Reads the assign-schemes workflow result, joins fire rates,
reports true counts + fire share, and writes scheme_assignments.json for the atlas.

Usage:
    python3 scripts/sae/taxonomy/recompute_schemes.py \
        --assignments output/taxonomy_v2/scheme_assignments_raw.json \
        --features output/taxonomy_v2/cluster_input.json \
        --out output/taxonomy_v2/scheme_assignments.json
"""
import argparse
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", required=True, help="list of {feature_id,mechanism,thinking_error}")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.load(open(args.assignments))
    asg = raw.get("assignments", raw)
    rows = json.load(open(args.features))

    by_fid = {a["feature_id"]: a for a in asg}
    fr = {int(f): rows[f]["fire_rate"] for f in rows}
    total_fire = sum(fr.values())
    n = len(rows)

    out_feats = {}
    for f in rows:
        fid = int(f)
        a = by_fid.get(fid, {})
        out_feats[f] = {
            "feature_id": fid,
            "chip": rows[f]["chip"],
            "fire": round(rows[f]["fire_rate"] * 100, 3),
            "mechanism": a.get("mechanism", "UNASSIGNED"),
            "thinking_error": a.get("thinking_error", "UNASSIGNED"),
        }

    json.dump({"features": out_feats}, open(args.out, "w"))

    for scheme in ("mechanism", "thinking_error"):
        cnt = Counter(out_feats[f][scheme] for f in out_feats)
        fire = defaultdict(float)
        for f in out_feats:
            fire[out_feats[f][scheme]] += fr[int(f)]
        print(f"\n=== {scheme.upper()} (agent-assigned, {len(by_fid)} features) ===")
        print(f"{'category':<28}{'feats':>6}{'%feat':>6}{'%fire':>6}")
        for c, k in cnt.most_common():
            print(f"{c:<28}{k:>6}{k/n*100:>5.0f}%{fire[c]/total_fire*100:>5.0f}%")
        sizes = list(cnt.values())
        print(f"  largest {max(sizes)/n*100:.0f}%  smallest {min(sizes)/n*100:.0f}%  unassigned {cnt.get('UNASSIGNED',0)}")


if __name__ == "__main__":
    main()
