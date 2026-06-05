#!/usr/bin/env python3
"""LLM clustering: within each category, Opus groups features into natural sub-clusters.

Decoder geometry and chip-embedding clustering both produce muddy "Hangs"-blob groups (tried,
rejected) — the coaching theme lives in the feature's chip+label, read holistically. So we let Opus
read all of a category's features (id, chip, one-line label) and form the natural groups itself,
naming each. Targets ~2-12 features/cluster but the model decides what's natural (no hard cap).

Every feature must be assigned to exactly one cluster. For big categories the feature list is sent
whole (≤~400 features fits one context) so the model sees them all together.

Usage:
  python3 cluster_llm.py --labels output/relabel_v3_5word_d2048_k6.json \
    --assign output/feature_buckets_v3_5word_d2048_k6.json --buckets output/buckets_v3_d2048_k6.json \
    --stats output/see_stats_d2048_k6.json --out output/feature_clusters_llm_d2048_k6.json [--only left_hanging]
"""
import argparse, json
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--assign", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
a = ap.parse_args()

lab = json.load(open(a.labels))
asg = json.load(open(a.assign))["assignments"]
buckets = json.load(open(a.buckets))
st = json.load(open(a.stats))
name = {b["id"]: b["name"] for b in buckets}
desc = {b["id"]: b["desc"] for b in buckets}
def fr(f): return (st.get("f" + f) or st.get(f) or {}).get("fire_rate", 0)

client = boto3.client("bedrock-runtime", region_name="us-east-1",
    config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 4}))

bycat = defaultdict(list)
for f, bid in asg.items():
    if bid != "unassignable": bycat[bid].append(f)


def _first_json(txt):
    """Parse the first balanced {...} object, ignoring any trailing text."""
    st = txt.find("{")
    depth = 0; instr = False; esc = False
    for i in range(st, len(txt)):
        ch = txt[i]
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
        else:
            if ch == '"': instr = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(txt[st:i+1])
    return json.loads(txt[st:])


def cluster_category(bid, fids):
    fids = sorted(fids, key=lambda f: -fr(f))
    body = "\n".join(f"f{f}: \"{lab[f]['chip']}\" — {lab[f].get('label','')[:140]}" for f in fids)
    prompt = f"""You are organizing {len(fids)} chess-mistake features that all belong to the category
"{name[bid]}" ({desc[bid]}). Each feature detects one recurring mistake; below is its id, short chip
name, and a one-line description.

Group them into a SMALL number of natural coaching clusters — the kinds of mistakes a coach would
teach together. Target roughly 10-20 clusters TOTAL for this category (fewer if it's small). Each
cluster should be a broad, recognizable theme, NOT a hyper-specific variant.

CRITICAL — merge aggressively:
- Do NOT create near-duplicate clusters. "Hangs piece with aggressive move", "Hangs piece on active
  square", and "Hangs piece via routine move" are the SAME coaching idea — merge into one.
- Do NOT append "(general)" or split a theme into general + specific versions. One cluster per theme.
- Group by the broad mechanism (e.g. "Hangs queen to a simple capture", "Hangs a piece to a fork",
  "Hangs a piece in the opening", "Hangs queen while attacking the enemy queen"). A cluster of 20-40
  features is fine if they share one theme. Avoid singletons unless a feature is truly unique.

Name each cluster with a clear, broad coaching label. EVERY feature id must appear in exactly one
cluster. Do not drop or duplicate any.

FEATURES:
{body}

Return JSON only:
{{"clusters":[{{"name":"<specific cluster name>","features":["<id>","<id>",...]}}, ...]}}"""
    r = client.invoke_model(modelId="us.anthropic.claude-opus-4-6-v1", body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}]}))
    txt = "".join(b.get("text", "") for b in json.loads(r["body"].read())["content"] if b.get("type") == "text")
    obj = _first_json(txt)
    # normalize ids, validate coverage
    seen = set(); clusters = []
    for c in obj["clusters"]:
        fs = [str(x).lstrip("f") for x in c["features"] if str(x).lstrip("f") in [str(i) for i in fids]]
        fs = [f for f in fs if f not in seen]
        for f in fs: seen.add(f)
        if fs: clusters.append({"name": c["name"], "n": len(fs),
                                "fire": round(sum(fr(f) for f in fs) * 100, 2), "features": fs})
    missing = [f for f in fids if f not in seen]
    if missing:  # park any dropped features in a catch-all so nothing is lost
        clusters.append({"name": "Other / unclustered", "n": len(missing),
                         "fire": round(sum(fr(f) for f in missing) * 100, 2), "features": missing})
    clusters.sort(key=lambda c: -c["fire"])
    return bid, clusters


cats = [(bid, fids) for bid, fids in bycat.items() if not a.only or bid == a.only]
out = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    for fu in as_completed([pool.submit(cluster_category, bid, fids) for bid, fids in cats]):
        try:
            bid, clusters = fu.result(); out[bid] = clusters
        except Exception as ex:
            print(f"  {ex}")

for bid in [b["id"] for b in buckets if b["id"] in out]:
    clusters = out[bid]; sizes = [c["n"] for c in clusters]
    print(f"\n{name[bid]}: {sum(sizes)} feats -> {len(clusters)} clusters (min {min(sizes)} med {sorted(sizes)[len(sizes)//2]} max {max(sizes)})")
    for c in clusters[:40]:
        print(f"    [{c['n']:>2}] {c['fire']:>5.1f}%  {c['name']}")

json.dump(out, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
