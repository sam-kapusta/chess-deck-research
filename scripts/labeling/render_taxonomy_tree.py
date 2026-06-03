"""Render the 2-level taxonomy as a browsable collapsible HTML tree.
Bucket -> sub-category -> features, each feature shows its label + N example boards
(played move red arrow, best move green). Self-contained inline SVG.

Usage (local): python render_taxonomy_tree.py --boards 3 --out output/atlas/taxonomy_tree_d1024_k4.html
"""
import json, argparse, chess, chess.svg
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('--boards', type=int, default=3)
ap.add_argument('--out', required=True)
a = ap.parse_args()

d = json.load(open('output/feature_labels_see_d1024_k4.json'))
st = json.load(open('output/see_stats_d1024_k4.json'))
clean = json.load(open('output/taxonomy_clean_names_d1024_k4.json'))['buckets']
leaf = json.load(open('output/feature_leaf_assignments_d1024_k4.json'))
prof = json.load(open('/tmp/d1024_k4_profiles.json'))
try:
    best_map = json.load(open('/tmp/best_uci_map.json'))   # 'fen|blunder_uci' -> best_uci
except Exception:
    best_map = {}

# tree: bucket_name -> sub -> [feature ids]
tree = defaultdict(lambda: defaultdict(list))
for bare, v in leaf.items():
    tree[v['bucket_name']][v['sub']].append(bare)
bucket_order = [b['name'] for b in clean]
# sub order from clean taxonomy
sub_order = {b['name']: [sg['name'] for sg in b.get('subgroups', [])] for b in clean}

def esc(s): return (str(s) or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def board_svg(fen, uci, best_uci):
    try:
        b = chess.Board(fen); arrows = []; lastmove = None
        try:
            mv = chess.Move.from_uci(uci); lastmove = mv
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color='#cc2b2b'))
        except Exception: pass
        if best_uci and len(best_uci) >= 4:
            try:
                bm = chess.Move.from_uci(best_uci)
                arrows.append(chess.svg.Arrow(bm.from_square, bm.to_square, color='#2b8a3e'))
            except Exception: pass
        return chess.svg.board(b, size=210, arrows=arrows, lastmove=lastmove,
                               orientation=chess.WHITE if b.turn == chess.WHITE else chess.BLACK)
    except Exception:
        return '<div>bad fen</div>'

# best_uci lookup from profiles? profiles only have fen,uci(blunder). best from feature labels? use cache-less: skip best arrow if unknown
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#161922;padding:12px 20px;border-bottom:1px solid #2a2f3a;z-index:20}
header h1{margin:0;font-size:17px} header .sub{color:#9aa4b2;font-size:12px;margin-top:3px}
details{margin:0} summary{cursor:pointer;list-style:none;user-select:none}
summary::-webkit-details-marker{display:none}
.bucket>summary{padding:11px 18px;background:#1b2030;border-bottom:1px solid #2a2f3a;font-size:16px;font-weight:600;color:#fff}
.bucket>summary:hover{background:#212840}
.bucket .cnt{color:#6ea8fe;font-weight:400;font-size:13px}
.sub{margin:0 0 0 14px}
.sub>summary{padding:8px 16px;color:#cdd6e0;font-size:14px;border-bottom:1px solid #1c2027}
.sub>summary:hover{background:#181c26} .sub .cnt{color:#7d8896;font-size:12px}
.feat{margin:6px 0 10px 30px;padding:8px 12px;border-left:2px solid #2a3550;background:#13161e}
.feat .fn{font-size:13px;color:#e6e6e6;margin-bottom:2px}
.feat .fl{font-size:11px;color:#8b95a3;margin-bottom:6px}
.feat .mech{font-size:10px;color:#c9a227}
.boards{display:flex;gap:8px;flex-wrap:wrap;margin-top:5px}
.legend{font-size:11px}.legend .r{color:#ff8a8a}.legend .g{color:#7ee2a0}
"""
parts = [f"<!doctype html><meta charset=utf-8><title>Taxonomy — d1024_k4</title><style>{CSS}</style>"]
nfeat = sum(len(f) for bk in tree.values() for f in bk.values())
parts.append(f"<header><h1>d1024_k4 mistake taxonomy — {len(tree)} buckets · {nfeat} features</h1>"
             f"<div class=sub>click to expand · <span class=legend><span class=r>red = played (blunder)</span> "
             f"<span class=g>green = Maia top move (elo 2600, not necessarily engine-best)</span></span></div></header>")

for bk in bucket_order:
    if bk not in tree: continue
    subs = tree[bk]; total = sum(len(v) for v in subs.values())
    parts.append(f"<details class=bucket><summary>{esc(bk)} <span class=cnt>{total} features</span></summary>")
    ordered_subs = [s for s in sub_order.get(bk, []) if s in subs] + [s for s in subs if s not in sub_order.get(bk, [])]
    for sub in ordered_subs:
        fids = sorted(subs[sub], key=lambda f: -(st.get('f'+f, st.get(f, {})).get('fire_rate', 0)))
        parts.append(f"<details class=sub><summary>{esc(sub)} <span class=cnt>{len(fids)}</span></summary>")
        for f in fids:
            a_ = d[f]['analysis']; s = st.get('f'+f) or st.get(f) or {}
            parts.append(f"<div class=feat><div class=fn>f{f} — {esc(a_['chip'])}</div>"
                         f"<div class=fl>{esc(a_.get('label',''))}</div>"
                         f"<div class=mech>bw {s.get('best_wins_material_pct',0)} · oh {s.get('blunder_hangs_own_pct',0)} · chk {s.get('best_is_check_pct',0)} · fires {s.get('fire_rate',0)*100:.1f}%</div>"
                         f"<div class=boards>")
            for ex in prof.get(f, {}).get('examples', [])[:a.boards]:
                best = best_map.get(ex['fen'] + '|' + ex['uci'], '')
                parts.append(board_svg(ex['fen'], ex['uci'], best))
            parts.append("</div></div>")
        parts.append("</details>")
    parts.append("</details>")
open(a.out, 'w').write('\n'.join(parts))
print(f"wrote {a.out} — {len(tree)} buckets, {nfeat} features")
