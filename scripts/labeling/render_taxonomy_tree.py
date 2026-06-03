"""Render the 2-level taxonomy as a browsable collapsible HTML tree.
Bucket -> sub-category -> features, each feature shows its label + N example boards
(played move red arrow, best move green). Self-contained inline SVG.

Usage (local): python render_taxonomy_tree.py --boards 3 --out output/atlas/taxonomy_tree_d1024_k4.html
"""
import json, argparse, chess, chess.svg
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('--boards', type=int, default=10)
ap.add_argument('--out', required=True)
a = ap.parse_args()

d = json.load(open('output/feature_labels_integrated_d1024_k4.json'))   # integrated labels (v2)
# normalize: integrated file keys are bare ids with chip/label at top level; wrap as {'analysis': ...}
d = {k: ({'analysis': v} if 'chip' in v else v) for k, v in d.items()}
st = json.load(open('output/see_stats_d1024_k4.json'))
clean = [{'name': b['name'], 'subgroups': []} for b in json.load(open('output/buckets_v2_d1024_k4.json'))]
leaf = json.load(open('output/feature_leaf_v2_d1024_k4.json'))
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

def board_cell(fen, uci, best_uci):
    """SVG board + a per-board caption: played move (what it captures), Maia move (type)."""
    try:
        b = chess.Board(fen); arrows = []; lastmove = None; cap_txt = ''; best_txt = ''
        try:
            mv = chess.Move.from_uci(uci); lastmove = mv
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color='#cc2b2b'))
            psan = b.san(mv)
            if b.is_capture(mv):
                tgt = b.piece_at(mv.to_square)
                cap_txt = f"{psan} ×{chess.piece_name(tgt.piece_type) if tgt else 'pawn'}"
            else:
                cap_txt = f"{psan} (quiet)"
        except Exception: psan = uci; cap_txt = uci
        if best_uci and len(best_uci) >= 4:
            try:
                bm = chess.Move.from_uci(best_uci)
                arrows.append(chess.svg.Arrow(bm.from_square, bm.to_square, color='#2b8a3e'))
                kind = 'capture' if b.is_capture(bm) else 'check' if b.gives_check(bm) else 'quiet'
                best_txt = f"Maia: {b.san(bm)} ({kind})"
            except Exception: pass
        svg = chess.svg.board(b, size=200, arrows=arrows, lastmove=lastmove,
                              orientation=chess.WHITE if b.turn == chess.WHITE else chess.BLACK)
        cap = f"<div class=bc><span class=r>{esc(cap_txt)}</span>" + (f" · <span class=g>{esc(best_txt)}</span>" if best_txt else "") + "</div>"
        return f"<div class=cell>{svg}{cap}</div>"
    except Exception:
        return '<div class=cell>bad fen</div>'

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
.feat .mech{font-size:11px;color:#aeb6c2;line-height:1.5;margin-bottom:4px}.feat .mech b{color:#c9a227;font-weight:600}
.boards{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}
.cell{width:200px}
.bc{font-size:10px;color:#9aa4b2;margin-top:2px;line-height:1.3}
.legend{font-size:11px}.legend .r{color:#ff8a8a}.legend .g{color:#7ee2a0}
.r{color:#ff8a8a}.g{color:#7ee2a0}
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
            def top2(dist):
                if not dist: return '—'
                return ', '.join(f"{k} {v*100:.0f}%" for k, v in list(dist.items())[:3])
            def topn(dist):  # for raw-COUNT dists (own_hang_piece_dist) — show as % of total
                if not dist: return '—'
                tot = sum(dist.values()) or 1
                return ', '.join(f"{k} {v/tot*100:.0f}%" for k, v in list(dist.items())[:3])
            def sig_line(sg, tag):
                if not sg: return f"<i>{tag}: no positions</i>"
                return (f"<u>{tag} (n={sg.get('n','?')})</u> "
                        f"moved {top2(sg.get('moved_piece_pct',{}))} · captured {top2(sg.get('captured_piece_pct',{}))} · "
                        f"played-check {sg.get('played_is_check_pct',0)*100:.0f}% · "
                        f"Maia {top2(sg.get('best_piece_pct',{}))} captures {top2(sg.get('best_captured_piece_pct',{}))} · "
                        f"material {top2(sg.get('material_kind_pct',{}))} (net {sg.get('net_material_median','?')}) · "
                        f"missed-material {sg.get('best_wins_material_pct',0)*100:.0f}% · phase {top2(sg.get('phase_pct',{}))}<br>"
                        f"&nbsp;&nbsp;<b>trajectory:</b> {top2({k:v for k,v in sg.get('trajectory_pct',{}).items() if k!='?->?'})} · eval-drop {sg.get('eval_drop_median','?')}cp")
            mech = (sig_line(s, '≥0.7·max') + "<br>" + sig_line(s.get('at_0.8'), '≥0.8·max') + "<br>"
                    f"<b>fires on</b> {int(s.get('fire_rate',0)*168132):,} of 168,132 positions ({s.get('fire_rate',0)*100:.1f}%) · max-act {s.get('max_act','?')}")
            parts.append(f"<div class=feat><div class=fn>f{f} — {esc(a_['chip'])}</div>"
                         f"<div class=fl>{esc(a_.get('label',''))}</div>"
                         f"<div class=mech>{mech}</div>"
                         f"<div class=boards>")
            for ex in prof.get(f, {}).get('examples', [])[:a.boards]:
                best = best_map.get(ex['fen'] + '|' + ex['uci'], '')
                parts.append(board_cell(ex['fen'], ex['uci'], best))
            parts.append("</div></div>")
        parts.append("</details>")
    parts.append("</details>")
open(a.out, 'w').write('\n'.join(parts))
print(f"wrote {a.out} — {len(tree)} buckets, {nfeat} features")
