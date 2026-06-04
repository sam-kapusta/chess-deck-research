#!/usr/bin/env python3
"""Bottom-up taxonomy: embed each feature's label, cluster mechanically, let groups emerge.

WHY BOTTOM-UP (not the old 11 buckets):
  The previous taxonomy was top-down — we wrote 11 bucket names and assigned features into them,
  which can only ever confirm the structure we'd already decided on. This clusters the actual
  feature LABELS by semantic similarity and lets the groups form themselves. No preset buckets,
  no rules, no LLM deciding the structure here. (Naming the emergent clusters is a separate,
  later step — this script only produces and reports the grouping for human judgment.)

METHOD:
  - Embed "chip. label" per feature with Titan v2 (1024-d, cosine space).
  - Agglomerative clustering, cosine metric, average linkage. We cut the dendrogram at a
    DISTANCE threshold (not a fixed k) so the cluster count emerges from the data.
  - Report each cluster: size, fire-rate coverage, sample chips, and the high-fire (blob)
    members — flagged, NOT dropped. No silent truncation.

Run locally (Titan = Bedrock, needs default creds):
  AWS_PROFILE=default python3 cluster_taxonomy.py \
    --labels output/relabel_allfields_d2048_k6.json --stats output/see_stats_d2048_k6.json \
    --threshold 0.45 --out output/taxonomy_clusters_d2048_k6.json
"""
import argparse, json, time, boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threshold", type=float, default=0.45, help="cosine-distance cut (lower = more, tighter clusters)")
ap.add_argument("--text-mode", default="both", choices=["both", "chip", "label"],
    help="what to embed: chip+label ('both'), chip only, or label only. chip strips the shared prose scaffold.")
ap.add_argument("--emb-cache", default="", help="embedding cache path (default derived from text-mode)")
a = ap.parse_args()
if not a.emb_cache:
    a.emb_cache = f"/tmp/label_embeddings_{a.text_mode}.json"

lab = json.load(open(a.labels))
st = json.load(open(a.stats))
def S(f): return st.get("f" + f) or st.get(f) or {}
def fire(f): return S(f).get("fire_rate", 0)

feats = [(f, v) for f, v in lab.items() if isinstance(v, dict) and "error" not in v and v.get("chip")]
feats.sort(key=lambda kv: int(kv[0]))
fids = [f for f, _ in feats]
if a.text_mode == "chip":
    texts = [v.get("chip", "") for _, v in feats]
elif a.text_mode == "label":
    texts = [v.get("label", "") for _, v in feats]
else:
    texts = [f"{v.get('chip','')}. {v.get('label','')}" for _, v in feats]

# --- embed (cached) ---
try:
    cache = json.load(open(a.emb_cache))
except Exception:
    cache = {}
client = boto3.client("bedrock-runtime", region_name="us-east-1",
    config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 4}))
def embed(i_text):
    i, text = i_text
    if str(i) in cache:
        return i, cache[str(i)]
    r = client.invoke_model(modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}))
    return i, json.loads(r["body"].read())["embedding"]

todo = [(i, t) for i, t in enumerate(texts) if str(i) not in cache]
print(f"embedding {len(todo)} labels ({len(cache)} cached) ...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as pool:
    for fu in as_completed([pool.submit(embed, it) for it in todo]):
        i, v = fu.result(); cache[str(i)] = v
json.dump(cache, open(a.emb_cache, "w"))
X = np.array([cache[str(i)] for i in range(len(texts))], dtype=np.float32)
print(f"embedded {len(X)} in {time.time()-t0:.0f}s, dim {X.shape[1]}", flush=True)

# --- cluster (cosine distance, average linkage, distance-threshold cut) ---
cl = AgglomerativeClustering(n_clusters=None, distance_threshold=a.threshold,
                             metric="cosine", linkage="average")
labels = cl.fit_predict(X)
nC = len(set(labels))
print(f"\nthreshold {a.threshold} -> {nC} clusters\n", flush=True)

# --- report (size, fire coverage, sample chips, blob members) ---
from collections import defaultdict
groups = defaultdict(list)
for fid, c in zip(fids, labels):
    groups[int(c)].append(fid)
order = sorted(groups, key=lambda c: -sum(fire(f) for f in groups[c]))  # by fire coverage

out = {"threshold": a.threshold, "n_clusters": nC, "clusters": []}
for c in order:
    members = sorted(groups[c], key=lambda f: -fire(f))
    cov = sum(fire(f) for f in members)
    chips = [lab[f]["chip"] for f in members]
    blobs = [f for f in members if fire(f) >= 0.01]
    # modal mistake_type in the cluster
    from collections import Counter
    mt = Counter(lab[f].get("mistake_type", "?") for f in members)
    out["clusters"].append({
        "id": c, "n": len(members), "fire_coverage": round(cov, 4),
        "modal_type": mt.most_common(1)[0][0],
        "type_mix": dict(mt.most_common()),
        "members": members, "chips": chips,
        "blobs": [{"f": f, "fire": round(fire(f), 4), "chip": lab[f]["chip"]} for f in blobs],
    })

json.dump(out, open(a.out, "w"), indent=1)

# console summary
print(f"{'#':>3} {'n':>4} {'fire%':>6} {'type':>11}  sample chips")
for cl_ in out["clusters"]:
    sample = ", ".join(cl_["chips"][:5])
    blobmark = f"  [{len(cl_['blobs'])} blob]" if cl_["blobs"] else ""
    print(f"{cl_['id']:>3} {cl_['n']:>4} {cl_['fire_coverage']*100:>5.1f}% {cl_['modal_type']:>11}  {sample[:80]}{blobmark}")
singletons = sum(1 for c in out["clusters"] if c["n"] == 1)
print(f"\n{nC} clusters · {singletons} singletons · wrote {a.out}")
