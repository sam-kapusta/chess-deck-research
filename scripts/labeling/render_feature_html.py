"""Render fused feature names as a browsable HTML: each feature = its name + motif
distribution + its top-10 positions as boards (played move highlighted, best move arrowed).

Self-contained single HTML (inline SVG boards via python-chess). Sorted by fire rate;
diffuse features pushed to the end. Open in a browser.

Usage (local):
  python render_feature_html.py --labels /tmp/fused_names_d1024_k4.json --title "d1024_k4" --out feature_atlas_d1024_k4.html
"""
import json, argparse, chess, chess.svg
ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--title', default='SAE features')
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0, help='0 = all features')
a = ap.parse_args()
d = json.load(open(a.labels))

def board_svg(fen, played_san, best_san):
    try:
        b = chess.Board(fen)
        arrows = []; lastmove = None
        # played move: red square highlight via lastmove (rendered orange by python-chess)
        try:
            mv = b.parse_san(played_san); lastmove = mv
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color='#cc2b2b'))
        except Exception: pass
        if best_san:
            try:
                bm = b.parse_san(best_san)
                arrows.append(chess.svg.Arrow(bm.from_square, bm.to_square, color='#2b8a3e'))
            except Exception: pass
        flip = not b.turn  # show from side-to-move's perspective: white to move -> white bottom
        return chess.svg.board(b, size=300, arrows=arrows, lastmove=lastmove,
                               orientation=chess.WHITE if b.turn == chess.WHITE else chess.BLACK)
    except Exception as e:
        return f'<div class="err">bad FEN: {fen}<br>{e}</div>'

# order: named by fire desc, then diffuse
items = sorted(d.items(), key=lambda kv: (kv[1]['status'] == 'diffuse', -kv[1]['fire_rate']))
if a.limit: items = items[:a.limit]

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#161922;padding:14px 20px;border-bottom:1px solid #2a2f3a;z-index:10}
header h1{margin:0;font-size:18px}
header .sub{color:#9aa4b2;font-size:13px;margin-top:4px}
#nav{padding:8px 20px;background:#11141c;border-bottom:1px solid #2a2f3a;font-size:12px;line-height:1.9}
#nav a{color:#6ea8fe;text-decoration:none;margin-right:10px}
.feat{padding:18px 20px;border-bottom:1px solid #222732}
.feat.diffuse{opacity:.55}
.fname{font-size:16px;font-weight:600;color:#fff}
.meta{color:#9aa4b2;font-size:12px;margin:4px 0 2px}
.motifs{font-size:11px;color:#7d8896;margin-bottom:10px}
.motifs b{color:#b9c2cf}
.boards{display:flex;flex-wrap:wrap;gap:14px}
.cell{width:300px;background:#161922;border:1px solid #2a2f3a;border-radius:8px;padding:8px}
.cell .mv{font-size:12px;color:#cfd6df;margin-bottom:4px}
.cell .mv .pl{color:#ff8a8a}.cell .mv .bs{color:#7ee2a0}
.cell .tac{font-size:11px;color:#c9a227;margin-top:6px}
.cell .sum{font-size:11px;color:#8b95a3;margin-top:4px;line-height:1.4}
.legend{font-size:11px;color:#7d8896}
.legend .pl{color:#ff8a8a}.legend .bs{color:#7ee2a0}
"""

def esc(s): return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

parts = [f"<!doctype html><meta charset=utf-8><title>{esc(a.title)}</title><style>{CSS}</style>"]
named = sum(1 for _, v in items if v['status'] == 'named')
parts.append(f"<header><h1>{esc(a.title)} — feature atlas</h1>"
             f"<div class=sub>{len(items)} features · {named} named · {len(items)-named} diffuse · "
             f"<span class=legend><span class=pl>red arrow = played (blunder)</span> · "
             f"<span class=bs>green arrow = best move</span></span></div></header>")
# nav: jump links every 50
parts.append("<div id=nav>")
for i in range(0, len(items), 50):
    parts.append(f"<a href='#f{i}'>{i}</a>")
parts.append("</div>")

for i, (fid, v) in enumerate(items):
    anchor = f"<a id='f{i}'></a>" if i % 50 == 0 else ""
    md = v.get('motif_dist', {})
    motline = ' · '.join(f"<b>{esc(m)}</b> {n}" for m, n in list(md.items())[:6])
    facts = v.get('see_facts', {})
    factline = ' · '.join(f"{esc(k)}={esc(str(fc.get('value')))}({int(fc.get('pct',0)*100)}%)"
                          for k, fc in facts.items() if fc.get('value') is not None)
    cls = 'feat diffuse' if v['status'] == 'diffuse' else 'feat'
    parts.append(f"{anchor}<div class='{cls}'>")
    parts.append(f"<div class=fname>{esc(fid)} — {esc(v['name'])}</div>")
    parts.append(f"<div class=meta>fires {v['fire_rate']*100:.1f}% · top motif {esc(v.get('motif_top','?'))} "
                 f"({int(v.get('motif_pct',0)*100)}%) · opus cov {v.get('opus_coverage','?')}/50</div>")
    parts.append(f"<div class=motifs>motifs: {motline}<br>SEE: {factline or '(none concentrated)'}</div>")
    parts.append("<div class=boards>")
    for p in v['top10']:
        svg = board_svg(p['fen'], p.get('played',''), p.get('best',''))
        mv = f"<span class=pl>{esc(p.get('played','?'))}</span>"
        if p.get('best'): mv += f" → <span class=bs>{esc(p['best'])}</span>"
        hang = f" · hang {esc(p['hang'])}" if p.get('hang') and p['hang'] != 'none' else ""
        cp = f" · cp {p['cp_loss']}" if p.get('cp_loss') is not None else ""
        tac = f"<div class=tac>{esc(p['motif'])}</div>" if p.get('motif') else ""
        summ = f"<div class=sum>{esc(p.get('summary',''))}</div>" if p.get('summary') else ""
        parts.append(f"<div class=cell><div class=mv>{mv}{hang}{cp}</div>{svg}{tac}{summ}</div>")
    parts.append("</div></div>")

open(a.out, 'w').write('\n'.join(parts))
print(f"wrote {a.out} ({len(items)} features)")
