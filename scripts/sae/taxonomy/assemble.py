"""Merge evidence + workflow assignments (category+chip) + vocab into taxonomy_v2.json.

The assignment file is produced by the relabel workflow and has shape:
  {"assignments": {"<fid>": {"feature_id", "category", "chip", "confidence", "corrected"}}, ...}
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    evidence = json.load(open(args.evidence))
    raw = json.load(open(args.assignments))
    assignments = raw.get("assignments", raw)  # tolerate either shape
    vocab = json.load(open(args.vocab))
    names = {c["id"]: c["name"] for c in vocab["categories"]}

    # normalize assignment keys to str(feature_id)
    asg = {}
    for k, v in assignments.items():
        asg[str(v.get("feature_id", k))] = v

    features = {}
    by_cat = defaultdict(list)
    missing = []
    for fid, e in evidence.items():
        a = asg.get(fid)
        if not a:
            missing.append(fid)
            cat = None
            chip = ""
            conf = 0
            corrected = False
        else:
            cat = a.get("category")
            chip = a.get("chip", "")
            conf = a.get("confidence", 0)
            corrected = a.get("corrected", False)
        features[fid] = {
            "feature_id": int(fid),
            "chip": chip,
            "title": e["label"],          # specific one-sentence title (was `label`)
            "description": e["description"],
            "category": cat,
            "category_name": names.get(cat),
            "confidence": conf,
            "corrected": corrected,
            "fingerprint": e["fingerprint"],
            "verification": e["verification"]["verdict"],
            "old_chip": e["old_chip"],
        }
        by_cat[cat].append(int(fid))

    out = {
        "meta": {
            "sae": "maia3_sae_diff_2048_k32 (labels: l2 0-2007)",
            "n_features": len(features),
            "n_categories": len(vocab["categories"]),
            "n_missing_assignment": len(missing),
            "source": "rebuild 2026-05-29 (title->categorize->chip, Sonnet 4.6 on research account)",
            "vocab_source": vocab.get("source", ""),
        },
        "categories": vocab["categories"],
        "category_index": {str(k): sorted(v) for k, v in by_cat.items()},
        "features": features,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Assembled {len(features)} features into {len([k for k in by_cat if k])} categories -> {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} features had no assignment: {missing[:10]}")


if __name__ == "__main__":
    main()
