"""Assign a model's integrated-labeled features to the EXISTING 11-bucket taxonomy.
Allows an 'unassignable' verdict (feature fits no bucket) — recorded, not force-fit.
Used for the k4-vs-k6 fixed-taxonomy head-to-head.

Usage (on chess-poc):
  AWS_PROFILE=default python assign_to_buckets.py --labels feature_labels_integrated_d2048_k6.json \
    --stats see_stats_d2048_k6.json --buckets buckets_v2_d1024_k4.json --out feature_buckets_k6_into_v2.json
"""
import json, argparse, boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True); ap.add_argument('--stats', required=True)
ap.add_argument('--buckets', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
integ = json.load(open(a.labels)); st = json.load(open(a.stats)); buckets = json.load(open(a.buckets))
valid = {b['id'] for b in buckets} | {'unassignable'}
blist = "\n".join(f"  {b['id']}: {b['name']} — {b['desc']}" for b in buckets)
feats = [(fid, v) for fid, v in integ.items() if v.get('chip')]
def line(fid, v):
    s = st.get('f'+fid) or st.get(fid) or {}
    return (f"f{fid}: chip='{v['chip']}' type={v.get('mistake_type','?')} | {v.get('label','')[:90]} "
            f"|| material={list((s.get('material_kind_pct') or {}).items())[:2]} "
            f"traj={[k for k in (s.get('trajectory_pct') or {}) if k!='?->?'][:2]} "
            f"phase={list((s.get('phase_pct') or {}).items())[:1]} missed={s.get('best_wins_material_pct',0)}")
client = boto3.client('bedrock-runtime', region_name='us-east-1', config=Config(read_timeout=200, connect_timeout=10, retries={'max_attempts':3}))
CH = 40
chunks = [feats[i:i+CH] for i in range(0, len(feats), CH)]
def run(ch):
    body = "\n".join(line(f, v) for f, v in ch)
    prompt = f"""Assign each chess-mistake feature to exactly ONE bucket id, using chip, mistake_type, label,
and SEE signals (material: trade vs loses vs hangs; traj; phase; missed-material).
RULES: material=trade + eval drops => premature_trade. material=loses+capture => greedy_capture.
material=hangs (non-capture drops piece) => left_hanging. positional+king => king_safety.
positional+aimless => passive_play. endgame => endgame_technique. best-is-check+missed => missed_check_mate.
If a feature genuinely fits NO bucket, answer "unassignable" (do not force-fit).

BUCKETS:
{blist}

FEATURES:
{body}

Return JSON only: {{"f<id>":"<bucket_id or unassignable>", ...}} every feature."""
    r = client.invoke_model(modelId='us.anthropic.claude-opus-4-6-v1', body=json.dumps({
        'anthropic_version':'bedrock-2023-05-31','max_tokens':4000,'messages':[{'role':'user','content':prompt}]}))
    txt = ''.join(b.get('text','') for b in json.loads(r['body'].read())['content'] if b.get('type')=='text')
    s, e = txt.find('{'), txt.rfind('}')+1
    return json.loads(txt[s:e])
asg = {}
with ThreadPoolExecutor(max_workers=12) as pool:
    for fu in as_completed([pool.submit(run, ch) for ch in chunks]):
        try: asg.update({k.lstrip('f'): v for k, v in fu.result().items() if v in valid})
        except Exception as ex: print('err', str(ex)[:70])
name = {b['id']: b['name'] for b in buckets}; name['unassignable'] = 'UNASSIGNABLE'
c = Counter(asg.values())
json.dump({'assignments': asg, 'counts': {name[k]: v for k, v in c.items()}}, open(a.out, 'w'), indent=1)
print(f"assigned {len(asg)}/{len(feats)}")
for bid, n in c.most_common(): print(f"  {n:>4}  {name[bid]}")
