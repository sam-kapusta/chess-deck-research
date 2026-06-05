#!/usr/bin/env python3
"""Per-category cluster audit: place unclustered features + flag members that don't fit.

After cluster_llm.py forms clusters, two residual issues:
  1. Some features land in an "Other / unclustered" catch-all (the model skipped them).
  2. Some clustered features don't actually fit the cluster they're in.

This runs ONE Opus pass per category: shows the existing clusters (name + members' chips) and the
unclustered features, and asks the model to (a) assign each unclustered feature to the best existing
cluster, and (b) flag any current member that belongs in a DIFFERENT cluster (within this category),
naming the better one. Output is a set of moves applied by apply_audit (separate), so nothing is
changed without review.

Usage:
  python3 audit_clusters.py --labels output/relabel_v3_5word_d2048_k6.json \
    --clusters output/feature_clusters_llm_d2048_k6.json --buckets output/buckets_v3_d2048_k6.json \
    --stats output/see_stats_d2048_k6.json --out output/cluster_audit_d2048_k6.json [--only missed_tactic]
"""
import argparse, json
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--clusters", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
a = ap.parse_args()

lab = json.load(open(a.labels))
clusters = json.load(open(a.clusters))
buckets = json.load(open(a.buckets))
st = json.load(open(a.stats))
name = {b["id"]: b["name"] for b in buckets}
def fr(f): return (st.get("f" + f) or st.get(f) or {}).get("fire_rate", 0)

client = boto3.client("bedrock-runtime", region_name="us-east-1",
    config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 4}))


def _first_json(txt):
    s = txt.find("{"); depth = 0; instr = False; esc = False
    for i in range(s, len(txt)):
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
                if depth == 0: return json.loads(txt[s:i+1])
    return json.loads(txt[s:])


def audit_category(bid):
    cls = clusters[bid]
    named = [c for c in cls if "unclustered" not in c["name"].lower()]
    orphans = [f for c in cls if "unclustered" in c["name"].lower() for f in c["features"]]
    if not named: return bid, {"place": {}, "flag": []}
    # cluster list with members
    clist = []
    for ci, c in enumerate(named):
        mem = ", ".join(f"f{f}(\"{lab[f]['chip']}\")" for f in c["features"][:14])
        clist.append(f"C{ci}: {c['name']}\n     members: {mem}")
    orph = "\n".join(f"f{f}: \"{lab[f]['chip']}\" — {lab[f].get('label','')[:120]}" for f in orphans)
    prompt = f"""Category "{name[bid]}". Below are its existing clusters (C0..) with members, then a list
of UNCLUSTERED features that need a home.

TWO tasks:
1. PLACE each unclustered feature into the best existing cluster (by id C0, C1, ...). If a feature
   truly fits none, mark it "NEW" and give a short cluster name.
2. FLAG any CURRENT member that clearly belongs in a different existing cluster in THIS list (a
   feature whose mistake doesn't match its cluster's theme). Only flag clear mismatches, not
   borderline ones. Give the feature id and the better cluster id.

EXISTING CLUSTERS:
{chr(10).join(clist)}

UNCLUSTERED FEATURES TO PLACE:
{orph if orphans else "(none)"}

Return JSON only:
{{"place":{{"<feat_id>":"C<n> or NEW:<name>", ...}},
"flag":[{{"feat":"<id>","from":"C<n>","to":"C<n>","why":"<short>"}}, ...]}}"""
    r = client.invoke_model(modelId="us.anthropic.claude-opus-4-6-v1", body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 6000,
        "messages": [{"role": "user", "content": prompt}]}))
    txt = "".join(b.get("text", "") for b in json.loads(r["body"].read())["content"] if b.get("type") == "text")
    res = _first_json(txt)
    res["_clusternames"] = {f"C{i}": c["name"] for i, c in enumerate(named)}
    res["_orphans"] = orphans
    return bid, res


cats = [b["id"] for b in buckets if b["id"] in clusters and (not a.only or b["id"] == a.only)]
out = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    for fu in as_completed([pool.submit(audit_category, bid) for bid in cats]):
        try:
            bid, res = fu.result(); out[bid] = res
        except Exception as ex:
            print(f"  err: {str(ex)[:100]}")

# report
nplace = nflag = 0
for bid in cats:
    if bid not in out: continue
    r = out[bid]; cn = r.get("_clusternames", {})
    place = r.get("place", {}); flag = r.get("flag", [])
    nplace += len(place); nflag += len(flag)
    if place or flag:
        print(f"\n{name[bid]}: placed {len(place)} orphans, flagged {len(flag)} misfits")
        for fid, dest in list(place.items())[:8]:
            dn = cn.get(dest, dest)
            print(f"    place f{fid} -> {dn}")
        for fl in flag[:8]:
            print(f"    FLAG f{fl.get('feat')} : {cn.get(fl.get('from'),'?')} -> {cn.get(fl.get('to'),'?')} ({fl.get('why','')[:50]})")
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nTOTAL: {nplace} orphans placed, {nflag} misfits flagged -> {a.out}")
