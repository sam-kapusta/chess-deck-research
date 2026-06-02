"""Constrained labeler: uses ONLY concentrated facts (>=70%) from deep_signature.
Strict rules prevent the over-specification bug (16/30 queen -> 'queen feature')."""
import json,time,boto3,argparse
from botocore.config import Config
from botocore.exceptions import ClientError,ReadTimeoutError
from concurrent.futures import ThreadPoolExecutor,as_completed
B='/home/ec2-user/SageMaker'
MODEL='us.anthropic.claude-opus-4-6-v1'
client=boto3.client('bedrock-runtime',region_name='us-east-1',config=Config(read_timeout=120,connect_timeout=10,retries={'max_attempts':0}))
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args()
sig=json.load(open(f'{B}/deep_sig_{a.model}.json'))
LABELS={'hang_class':'piece left hanging (value class)','hang_exact':'exact piece hanging',
        'anyhang':'whether a piece hangs','best_type':'character of the correct move',
        'best_class':'correct move piece (class)','best_piece':'correct move piece',
        'blunder_piece':'piece the player moved','iscap':'was blunder a capture','phase':'game phase',
        'traj':'eval trajectory'}
def factsheet(d):
    lines=[]
    for ax,fk in [('anyhang','anyhang'),('hang_class','hang_class'),('hang_exact','hang_exact'),
                  ('best_type','best_type'),('best_class','best_class'),('best_piece','best_piece'),
                  ('blunder_piece','blunder_piece'),('traj','traj'),('phase','phase'),('iscap','iscap')]:
        fc=d['facts'].get(fk)
        if not fc:continue
        if fc.get('value') is not None:
            lines.append(f"  [{int(fc['pct']*100)}%] {LABELS[ax]} = {fc['value']}")
        # else: not concentrated, omit (this is the key fix — don't show non-concentrated as a fact)
    return '\n'.join(lines) if lines else '  (no axis concentrated >=70% — feature is diffuse)'
def build(fid,d):
    return f"""Label this SAE chess-mistake feature. Below are ONLY the properties that are
CONCENTRATED (>=70%) across {d['n_sampled']} sampled positions where it fires (fires on {d['fire_rate']*100:.0f}% of corpus).

CONCENTRATED FACTS (these are measured — trust them, ignore anything not listed):
{factsheet(d)}

Example positions: {' | '.join(d['examples'][:6])}

RULES:
- Name a specific piece ONLY if 'exact piece' is listed as concentrated. If only the value-CLASS is
  concentrated (e.g. 'major piece'), say "major piece" / "minor piece" — do NOT name a specific piece.
- If only 'whether a piece hangs = hangs' is concentrated (no class), say "a piece" generically.
- Lead the chip with the MOST concentrated fact. Best-move facts describe what the player MISSED.
- Be concrete but do not over-claim beyond the listed facts.

JSON: {{"chip":"<3-6 words, the dominant pattern>","played":"<what went wrong>","missed":"<what the correct move was>","confidence":<0-100, lower if few facts concentrated>}}"""
def is_diffuse(d):
    """No axis concentrated >=70% → feature has no clear pattern, refuse to label specifically."""
    return not any(fc.get('value') is not None for fc in d['facts'].values())
def call(fid,d):
    if is_diffuse(d):
        # do NOT ask Opus to invent a chip from a weak plurality (the f2018 bug)
        return fid,{'chip':'diffuse — no clear pattern','played':'no concentrated pattern across positions',
                    'missed':'','confidence':0,'diffuse':True}
    for att in range(3):
        try:
            r=client.invoke_model(modelId=MODEL,body=json.dumps({'anthropic_version':'bedrock-2023-05-31','max_tokens':1200,'messages':[{'role':'user','content':build(fid,d)}]}))
            t=json.loads(r['body'].read())['content'][0]['text'].strip()
            if t.startswith('```'):t=t.split('```')[1].lstrip('json').strip()
            return fid,json.loads(t[t.find('{'):t.rfind('}')+1])
        except (ReadTimeoutError,ClientError):time.sleep(2**att)
        except Exception as ex:return fid,{'error':str(ex)[:80]}
    return fid,{'error':'retries'}
res={}
with ThreadPoolExecutor(max_workers=10) as p:
    for fu in as_completed({p.submit(call,f,d):f for f,d in sig.items()}):
        f,r=fu.result();res[f]=r
out={f:{'fire_rate':d['fire_rate'],'facts':d['facts'],**res.get(f,{})} for f,d in sig.items()}
json.dump(out,open(f'{B}/test_labels_v2_{a.model}.json','w'),indent=1)
print(f"{a.model}: relabeled {sum(1 for v in res.values() if 'error' not in v)}/{len(sig)}")
