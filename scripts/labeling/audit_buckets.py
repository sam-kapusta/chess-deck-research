"""Audit a feature->bucket assignment objectively: does each feature's SEE signature
contradict the bucket it's filed under? Flags violations for semantic review.

Rules are deliberately conservative and exempt known-correct edge cases so the audit
itself doesn't false-positive:
  - a forced MATE need not be a check (smothered/quiet mate) -> 'mate' chips exempt from the check rule
  - promotion / passed-pawn are endgame concepts even in late middlegame -> exempt from the phase rule

Usage: python audit_buckets.py --assign output/feature_buckets_v2_d1024_k4.json \
         --labels output/feature_labels_integrated_d1024_k4.json --stats output/see_stats_d1024_k4.json \
         --out output/audit_v2_flagged.json
"""
import json, argparse
from collections import Counter, defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('--assign', required=True); ap.add_argument('--labels', required=True)
ap.add_argument('--stats', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
asg = json.load(open(a.assign))['assignments']
integ = json.load(open(a.labels)); st = json.load(open(a.stats))

def sig(fid):
    s = st.get('f'+fid) or st.get(fid) or {}
    mk = s.get('material_kind_pct', {}) or {}
    return {'trade': mk.get('trade', 0), 'hangs': mk.get('hangs', 0), 'loses': mk.get('loses', 0),
            'missed': s.get('best_wins_material_pct', 0), 'check': s.get('best_is_check_pct', 0),
            'pchk': s.get('played_is_check_pct', 0), 'phase': (s.get('phase_pct', {}) or {}),
            'type': integ.get(fid, {}).get('mistake_type', '?'), 'chip': integ.get(fid, {}).get('chip', '')}

ENDGAME_OK = ('promot', 'passed pawn', 'pawn endgame', 'king and pawn', 'endgame')
def check(bid, g):
    ph_end = g['phase'].get('endgame', 0); chip = g['chip'].lower()
    if bid == 'premature_trade' and g['trade'] < 0.4:
        return f"trade-bucket but trade-material only {g['trade']:.0%}"
    if bid == 'greedy_capture' and g['loses'] < 0.3 and g['trade'] > 0.5:
        return f"greedy but trade-material {g['trade']:.0%}"
    if bid == 'left_hanging' and g['hangs'] < 0.3 and g['trade'] > 0.5:
        return f"left-hanging but trade-material {g['trade']:.0%}"
    if bid == 'missed_hanging' and g['missed'] < 0.4:
        return f"missed-hanging but missed-material only {g['missed']:.0%}"
    if bid == 'missed_check_mate' and g['check'] < 0.4 and 'mate' not in chip:
        return f"missed-check but best-is-check only {g['check']:.0%}"
    if bid == 'endgame_technique' and ph_end < 0.4 and not any(w in chip for w in ENDGAME_OK):
        return f"endgame but endgame-phase only {ph_end:.0%}"
    if bid == 'pointless_check' and g['pchk'] < 0.4:
        return f"pointless-check but played-check only {g['pchk']:.0%}"
    return None

viol = []; bybk = Counter()
for fid, bid in asg.items():
    bybk[bid] += 1
    r = check(bid, sig(fid))
    if r: viol.append({'fid': fid, 'bucket': bid, 'chip': sig(fid)['chip'], 'reason': r})
print(f"=== objective audit: {len(viol)}/{len(asg)} flagged ({100*len(viol)/len(asg):.1f}%) ===")
byb = defaultdict(list)
for v in viol: byb[v['bucket']].append(v)
for bid in sorted(byb, key=lambda b: -len(byb[b])):
    print(f"\n{bid} ({len(byb[bid])} of {bybk[bid]}):")
    for v in byb[bid][:10]: print(f"   f{v['fid']}: '{v['chip']}' — {v['reason']}")
json.dump(viol, open(a.out, 'w'), indent=1)
