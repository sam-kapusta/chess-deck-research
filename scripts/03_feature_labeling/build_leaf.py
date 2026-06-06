#!/usr/bin/env python3
"""Build the atlas leaf file (feature -> {bucket, bucket_name, sub}) from the cluster file.

The atlas (render_atlas_v3.py) reads a flat leaf map: each feature -> the bucket it belongs to
and the sub-group (cluster) name shown under that bucket. cluster_llm.py emits the clusters; this
script flattens them, and FOLDS the >5% "blob" features into one coarse sub-group per bucket.

WHY THE FOLD: features firing on >5% of all positions are coarse "material lost" detectors, not
specific tactics (f1487 fires 25-34%). Their cluster_llm name pretends a specificity they don't
have. We collapse them all into "⚠ Coarse detectors (>5% fire — broad, not specific)" inside their
assigned bucket; render_atlas_v3.py sorts any ⚠ sub last. Matches the k6 v7 finalize convention.

Run locally:
  python3 build_leaf.py --clusters output/feature_clusters_v7_d2048_k4.json \
    --assign output/feature_buckets_v7_d2048_k4.json --stats output/see_stats_d2048_k4.json \
    --buckets output/buckets_v3_d2048_k6.json --out output/feature_leaf_v7_d2048_k4.json
"""
import argparse, json

COARSE_SUB = "⚠ Coarse detectors (>5% fire — broad, not specific)"
BLOB_THRESHOLD = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--clusters", required=True, help="cluster_llm.py output (bucket -> [clusters])")
ap.add_argument("--assign", required=True, help="feature_buckets (for unassignable + bucket truth)")
ap.add_argument("--stats", required=True, help="see_stats (fire_rate -> blob fold)")
ap.add_argument("--buckets", required=True, help="buckets_v3 (id -> display name)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

clusters = json.load(open(a.clusters))
assign = json.load(open(a.assign))
assign = assign.get("assignments", assign)  # assign_v3 wraps the map under "assignments"
st = json.load(open(a.stats))
buckets = json.load(open(a.buckets))
BNAME = {b["id"]: b["name"] for b in buckets}


def fr(f):
    return (st.get("f" + f) or st.get(f) or {}).get("fire_rate", 0)


leaf, n_coarse = {}, 0
for bid, cls in clusters.items():
    for c in cls:
        for f in c["features"]:
            if fr(f) >= BLOB_THRESHOLD:
                sub = COARSE_SUB
                n_coarse += 1
            else:
                sub = c["name"]
            leaf[f] = {"bucket": bid, "bucket_name": BNAME.get(bid, bid), "sub": sub}

# carry through unassignable so the atlas can report the count
nun = 0
for f, v in assign.items():
    b = v.get("bucket") if isinstance(v, dict) else v
    if b == "unassignable" and f not in leaf:
        leaf[f] = {"bucket": "unassignable", "bucket_name": "Unassignable", "sub": "unassignable"}
        nun += 1

json.dump(leaf, open(a.out, "w"), indent=1)
print(f"wrote {a.out} — {len(leaf)} features | {n_coarse} folded as coarse | {nun} unassignable")
