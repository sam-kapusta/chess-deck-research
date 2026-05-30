"""Assemble the v3 taxonomy from: clustering (clusters.json), per-feature labels
+ fire rates (cluster_input.json), and the naming workflow output (categories
with names + sub-cluster names + misfits).

Produces taxonomy_v3.json: category -> sub-cluster -> feature, with fire rates
summed at each level (reach proxy = sum of member fire rates; we don't have the
co-fire matrix joined here so we report sum-rate, and note reach needs the
activation matrix).

Usage:
    python3 scripts/sae/taxonomy/assemble_v3.py \
        --clusters output/taxonomy_v2/clusters.json \
        --features output/taxonomy_v2/cluster_input.json \
        --naming output/taxonomy_v2/naming_result.json \
        --out output/taxonomy_v2/taxonomy_v3.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--naming", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clusters = json.load(open(args.clusters))
    feats = json.load(open(args.features))
    naming = json.load(open(args.naming))  # {"categories":[{coarse_id,category_name,...}]}

    cats_in = naming["categories"] if "categories" in naming else naming
    by_coarse = {c["coarse_id"]: c for c in cats_in}

    # misfits: feature_id -> reason (re-routed in QC later; here we just mark them)
    misfit_ids = {}
    for c in cats_in:
        for m in c.get("misfits", []):
            misfit_ids[m["feature_id"]] = {"from_coarse": c["coarse_id"], "reason": m.get("reason", "")}

    def frate(fid):
        return feats[str(fid)]["fire_rate"]

    categories = []
    for coarse_id, cat in sorted(by_coarse.items()):
        subs = []
        cat_feats = 0
        cat_sum_rate = 0.0
        for sc in cat.get("subclusters", []):
            members = [fid for fid in sc.get("feature_ids", []) if fid not in misfit_ids]
            if not members:
                continue
            sub_rate = sum(frate(fid) for fid in members)
            cat_feats += len(members)
            cat_sum_rate += sub_rate
            subs.append({
                "fine_id": sc["fine_id"],
                "name": sc["name"],
                "n_features": len(members),
                "sum_fire_rate_pct": round(sub_rate * 100, 3),
                "features": sorted([
                    {
                        "feature_id": fid,
                        "chip": feats[str(fid)]["chip"],
                        "fire_rate_pct": round(frate(fid) * 100, 3),
                        "description": feats[str(fid)]["description"],
                        "title": feats[str(fid)].get("title", ""),
                        "dom_piece": feats[str(fid)].get("dom_piece", ""),
                    } for fid in members
                ], key=lambda x: -x["fire_rate_pct"]),
            })
        subs.sort(key=lambda s: -s["n_features"])
        categories.append({
            "coarse_id": coarse_id,
            "name": cat["category_name"],
            "definition": cat.get("category_definition", ""),
            "n_features": cat_feats,
            "n_subclusters": len(subs),
            "sum_fire_rate_pct": round(cat_sum_rate * 100, 3),
            "subclusters": subs,
        })
    categories.sort(key=lambda c: -c["n_features"])

    total_feats = sum(c["n_features"] for c in categories)
    out = {
        "meta": {
            "sae": "maia3_sae_diff_v2_2048_k32_l2 (flat k=32, v2 corrected data)",
            "method": "bge-m3 concept clustering (ward) -> LLM holistic naming -> QC",
            "n_features": total_feats,
            "n_categories": len(categories),
            "n_misfits_pending_qc": len(misfit_ids),
            "fire_rate_source": "firerate_flat_v2_k32.npy",
        },
        "categories": categories,
        "misfits": [{"feature_id": fid, **info} for fid, info in misfit_ids.items()],
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"taxonomy_v3: {len(categories)} categories, {total_feats} features, "
          f"{len(misfit_ids)} misfits pending QC -> {args.out}")
    print(f"\n{'category':<32} {'feats':>5} {'subs':>4} {'sum_fire%':>9}")
    for c in categories:
        print(f"{c['name']:<32} {c['n_features']:>5} {c['n_subclusters']:>4} {c['sum_fire_rate_pct']:>9.2f}")


if __name__ == "__main__":
    main()
