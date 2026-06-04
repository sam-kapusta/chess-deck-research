"""Render an ARBITRARY list of feature ids (one model) to a rich HTML — full per-feature SEE
signature (both cohorts, distributions, trajectory, material) + coherence verdict + top-N boards
(chess.com clickable). Used to eyeball a filtered subset (e.g. the hard-cut blobs).

Usage (local): python render_feature_list.py --model k4 --dict 1024 --feats f87,f564,... \
   [--filter hardcut] --boards 10 --out output/atlas/k4_cut.html
--filter hardcut = peak-only AND pct_0.7<0.35 AND fire>=0.001 (the clear blobs).
"""
import json, argparse, chess, chess.svg
from urllib.parse import quote as _q
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--dict', type=int, required=True)
ap.add_argument('--feats', default=''); ap.add_argument('--filter', default='')
ap.add_argument('--boards', type=int, default=10); ap.add_argument('--out', required=True)
ap.add_argument('--title', default='')
a = ap.parse_args()
M = f"d{a.dict}_{a.model}"
lab = json.load(open(f'output/feature_labels_integrated_{M}.json'))
st = json.load(open(f'output/see_stats_{M}.json'))
coh = json.load(open(f'output/coherence_depth_{M}.json'))
prof = json.load(open(f'/tmp/{M}_profiles.json'))
best = json.load(open('/tmp/best_uci_map.json'))
def S(f): return st.get('f'+f) or st.get(f) or {}
def CO(f): return coh.get('f'+f) or coh.get(f) or {}
def fire(f): s = S(f); return s.get('fire_rate', 0)
# select features
if a.filter == 'hardcut':
    feats = [f.lstrip('f') for f in coh if CO(f.lstrip('f')).get('verdict')=='peak_only'
             and CO(f.lstrip('f')).get('pct_0.7',1)<0.35 and fire(f.lstrip('f'))>=0.001]
else:
    feats = [x.strip().lstrip('f') for x in a.feats.split(',') if x.strip()]
feats.sort(key=lambda f: -fire(f))
def esc(s): return (str(s) or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def top2(d): return ', '.join(f"{k} {v*100:.0f}%" for k,v in list(d.items())[:3]) if d else '—'
def topn(d):
    if not d: return '—'
    t=sum(d.values()) or 1; return ', '.join(f"{k} {v/t*100:.0f}%" for k,v in list(d.items())[:3])
def sig(s, tag):
    if not s: return f"<i>{tag}: none</i>"
    return (f"<u>{tag} (n={s.get('n','?')})</u> moved {top2(s.get('moved_piece_pct',{}))} · "
            f"captured {top2(s.get('captured_piece_pct',{}))} · played-check {s.get('played_is_check_pct',0)*100:.0f}% · "
            f"Maia {top2(s.get('best_piece_pct',{}))} captures {top2(s.get('best_captured_piece_pct',{}))} · "
            f"material {top2(s.get('material_kind_pct',{}))} · missed-mat {s.get('best_wins_material_pct',0)*100:.0f}% · "
            f"phase {top2(s.get('phase_pct',{}))}<br>&nbsp;&nbsp;trajectory {top2({k:v for k,v in (s.get('trajectory_pct',{}) or {}).items() if k!='?->?'})}")
def board(fen, uci):
    try:
        b=chess.Board(fen); ar=[]; lm=None; cap=''
        try:
            mv=chess.Move.from_uci(uci); lm=mv; ar.append(chess.svg.Arrow(mv.from_square,mv.to_square,color='#cc2b2b'))
            cap=b.san(mv)+(' ×'+chess.piece_name(b.piece_at(mv.to_square).piece_type) if b.is_capture(mv) and b.piece_at(mv.to_square) else ' (quiet)')
        except: cap=uci
        bu=best.get(fen+'|'+uci,'')
        if bu and len(bu)>=4:
            try:
                bm=chess.Move.from_uci(bu); ar.append(chess.svg.Arrow(bm.from_square,bm.to_square,color='#2b8a3e'))
                cap+=f" · Maia {b.san(bm)}"
            except: pass
        svg=chess.svg.board(b,size=190,arrows=ar,lastmove=lm,orientation=chess.WHITE if b.turn else chess.BLACK)
        url='https://www.chess.com/analysis?fen='+_q(fen,safe='')
        return f'<a class=cell href="{url}" target=_blank style=text-decoration:none;color:inherit>{svg}<div class=cap>{esc(cap)} ↗</div></a>'
    except: return '<div class=cell>bad</div>'
CSS="body{font-family:sans-serif;background:#0f1115;color:#e6e6e6;margin:0}.feat{padding:14px 18px;border-bottom:2px solid #2a2f3a}.fn{font-size:15px;font-weight:600}.fl{font-size:12px;color:#8b95a3;margin:2px 0}.mech{font-size:11px;color:#aeb6c2;line-height:1.5;margin:4px 0}.mech u{color:#c9a227}.boards{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}.cell{width:190px}.cap{font-size:10px;color:#9aa4b2;margin-top:2px}h1{padding:14px 18px;margin:0;font-size:16px}"
parts=[f"<!doctype html><meta charset=utf-8><style>{CSS}</style><h1>{esc(a.title or (M+' — '+str(len(feats))+' features'))} · boards link to chess.com ↗</h1>"]
for f in feats:
    av=lab.get(f) or {}; s=S(f); c=CO(f)
    cv=f"coherence {c.get('peak_pct','?')}→{c.get('pct_0.8','?')}→{c.get('pct_0.7','?')} [{c.get('verdict','?')}]" if c else ''
    parts.append(f"<div class=feat><div class=fn>f{f} — {esc(av.get('chip','?'))} <span style=color:#6ea8fe;font-weight:400>fires {fire(f)*100:.1f}% · {cv}</span></div>"
        f"<div class=fl>{esc(av.get('label',''))}</div>"
        f"<div class=mech>{sig(s,'≥0.7·max')}<br>{sig(s.get('at_0.8'),'≥0.8·max')}</div><div class=boards>")
    for ex in prof.get(f,{}).get('examples',[])[:a.boards]:
        parts.append(board(ex['fen'],ex['uci']))
    parts.append("</div></div>")
open(a.out,'w').write('\n'.join(parts))
print(f"wrote {a.out} ({len(feats)} features)")
