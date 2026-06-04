#!/usr/bin/env python3
"""Assign every labeled feature to the v3 12-bucket taxonomy. No rules, no steer.

The v3 taxonomy (buckets_v3_d2048_k6.json) was derived bottom-up from three disjoint 200-feature
samples that independently converged (see log.md). This script assigns ALL features into it.

Unlike the old assign_to_buckets.py, there is NO hardcoded RULES block mapping SEE signals to
buckets. The model gets the bucket definitions (each tagged self-inflicted / omission / phase) and
each feature's evidence — chip, label, and the objective self-inflicted-vs-omission SEE tell
(played-loses-material % vs best-wins-material %) — and decides. 'unassignable' is allowed and
recorded, never force-fit. The self-inflicted vs omission axis is the primary guard against the
"missed win" catch-all: a high played-loses-material feature is self-inflicted even if a better
move existed.

Run locally (Bedrock, default creds):
  AWS_PROFILE=default python3 assign_v3.py --labels output/relabel_allfields_d2048_k6.json \
    --stats output/see_stats_d2048_k6.json --buckets output/buckets_v3_d2048_k6.json \
    --out output/feature_buckets_v3_d2048_k6.json
"""
import argparse, json, time, boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=40)
a = ap.parse_args()

lab = json.load(open(a.labels))
st = json.load(open(a.stats))
buckets = json.load(open(a.buckets))
def S(f): return st.get("f" + f) or st.get(f) or {}

valid = {b["id"] for b in buckets} | {"unassignable"}
name = {b["id"]: b["name"] for b in buckets}; name["unassignable"] = "UNASSIGNABLE"
blist = "\n".join(f"  {b['id']} [{b['char']}]: {b['name']} — {b['desc']}" for b in buckets)

feats = [(f, v) for f, v in lab.items() if isinstance(v, dict) and "error" not in v and v.get("chip")]
feats.sort(key=lambda kv: int(kv[0]))

def line(f, v):
    s = S(f)
    return (f"f{f}: \"{v['chip']}\" — {v.get('label','')[:130]} "
            f"[played-loses-material {s.get('blunder_hangs_own_pct',0)*100:.0f}%, "
            f"best-wins-material {s.get('best_wins_material_pct',0)*100:.0f}%, "
            f"material={list((s.get('material_kind_pct') or {}).items())[:2]}, "
            f"phase={list((s.get('phase_pct') or {}).items())[:1]}]")

client = boto3.client("bedrock-runtime", region_name="us-east-1",
    config=Config(read_timeout=200, connect_timeout=10, retries={"max_attempts": 4}))

def run(ch):
    body = "\n".join(line(f, v) for f, v in ch)
    prompt = f"""Assign each chess-mistake feature to exactly ONE bucket id from the taxonomy below.

Decide from the chip, the label, and the objective signals. The buckets are tagged [self-inflicted]
(the move PLAYED loses material — high played-loses-material), [omission] (the move played is safe but
a better move was missed — high best-wins-material, low played-loses), or [phase] (endgame technique).
Use the self-inflicted-vs-omission character to place the feature: a feature whose move usually loses
its own material is self-inflicted even if a stronger move also existed — do NOT default everything to
a "missed" bucket. If a feature genuinely fits NO bucket, answer "unassignable" (do not force-fit).

TAXONOMY:
{blist}

FEATURES:
{body}

Return JSON only: {{"f<id>":"<bucket_id or unassignable>", ...}} for EVERY feature listed."""
    r = client.invoke_model(modelId="us.anthropic.claude-opus-4-6-v1", body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]}))
    txt = "".join(b.get("text", "") for b in json.loads(r["body"].read())["content"] if b.get("type") == "text")
    s, e = txt.find("{"), txt.rfind("}") + 1
    return json.loads(txt[s:e])

chunks = [feats[i:i+a.chunk] for i in range(0, len(feats), a.chunk)]
print(f"assigning {len(feats)} features in {len(chunks)} chunks ...", flush=True)
asg = {}; t0 = time.time()
with ThreadPoolExecutor(max_workers=12) as pool:
    for fu in as_completed([pool.submit(run, ch) for ch in chunks]):
        try:
            asg.update({k.lstrip("f"): v for k, v in fu.result().items() if v in valid})
        except Exception as ex:
            print("  chunk err", str(ex)[:80])

c = Counter(asg.values())
fire = Counter()
for f, bid in asg.items(): fire[bid] += S(f).get("fire_rate", 0)
out = {"buckets": [b["id"] for b in buckets], "assignments": asg,
       "counts": {name[k]: v for k, v in c.items()}}
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nassigned {len(asg)}/{len(feats)} in {(time.time()-t0)/60:.1f}min -> {a.out}\n")
print(f"{'bucket':28s} {'n':>5} {'%feat':>6} {'fire%':>7}")
order = {b["id"]: i for i, b in enumerate(buckets)}; order["unassignable"] = 99
for bid in sorted(c, key=lambda k: order.get(k, 98)):
    print(f"  {name[bid]:26s} {c[bid]:>5} {c[bid]/len(asg)*100:>5.0f}% {fire[bid]*100:>6.0f}%")
