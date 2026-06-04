import json
from collections import defaultdict,Counter
asg=json.load(open('output/feature_buckets_v2_d1024_k4.json'))['assignments']
integ=json.load(open('output/feature_labels_integrated_d1024_k4.json'))
st=json.load(open('output/see_stats_d1024_k4.json'))
def s(f): return st.get('f'+f) or st.get(f) or {}
def fr(f): return s(f).get('fire_rate',0)
def dom(f,key,d=None):
    dd=s(f).get(key,{}) or {}
    return max(dd,key=dd.get) if dd else (d or '?')
# sub-rules per big bucket: function(fid)->sub label
PIECE={'queen':'Queen','rook':'Rook','bishop':'Bishop','knight':'Knight','minor':'Minor','pawn':'Pawn','king':'King','none':'Other','?':'Other'}
def sub_hanging(f): return 'Hung '+PIECE.get(dom(f,'moved_piece_pct'),'Piece')
def sub_endgame(f):
    p=dom(f,'moved_piece_pct')
    if p=='king': return 'King Technique'
    if p=='pawn': return 'Pawn Technique'
    if p in('rook',): return 'Rook Endgame'
    return 'Other Endgame'
def sub_piece(f): return PIECE.get(dom(f,'moved_piece_pct'),'Piece')
RULES={'left_hanging':sub_hanging,'endgame_technique':sub_endgame,'missed_tactic':sub_piece,
       'missed_hanging':lambda f:'Missed '+PIECE.get(dom(f,'best_captured_piece_pct'),'Capture'),
       'missed_check_mate':sub_piece,'king_safety':sub_piece,'pointless_check':sub_piece,
       'premature_trade':lambda f:PIECE.get(dom(f,'moved_piece_pct'),'Piece')+' Trade',
       'passive_play':sub_piece,'greedy_capture':sub_piece,'unsound_aggression':sub_piece}
buckets={b['id']:b['name'] for b in json.load(open('output/buckets_v2_d1024_k4.json'))}
out={}
for fid,bid in asg.items():
    sub=RULES.get(bid,lambda f:'-')(fid)
    out[fid]={'bucket':bid,'bucket_name':buckets[bid],'sub':sub}
json.dump(out,open('output/feature_leaf_v2_d1024_k4.json','w'),indent=1)
# rollup: bucket -> sub -> (count, total fire)
roll=defaultdict(lambda:defaultdict(lambda:[0,0.0]))
for fid,bid in asg.items():
    sub=out[fid]['sub']; roll[bid][sub][0]+=1; roll[bid][sub][1]+=fr(fid)
for bid in ['left_hanging','endgame_technique','missed_tactic']:
    print(f'\\n{buckets[bid]}:')
    for sub,(n,tf) in sorted(roll[bid].items(),key=lambda kv:-kv[1][1]):
        print(f'   {sub:18s} {n:>3} feats  {tf*100:5.1f}% coverage')
