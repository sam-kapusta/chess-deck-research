#!/usr/bin/env python3
"""Standalone board viewer for a handful of SAE features — top row + median row, with move arrows.
Used to eyeball whether a low-consistency feature is clean-broad or muddy-broad.

  python3 render_feature_boards.py --profiles output/atlas_profiles_d64_k1.json \
    --best output/best_uci_map_d64_k1.json --labels output/relabel_v7_d64_k1.json \
    --stats output/see_stats_d64_k1.json --feats 48,3,47,42,31 --out output/feat_boards_d64.html
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--profiles", required=True)
ap.add_argument("--best", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--feats", required=True, help="comma fids")
ap.add_argument("--n", type=int, default=8, help="boards per band")
ap.add_argument("--out", required=True)
a = ap.parse_args()

prof = json.load(open(a.profiles)); best = json.load(open(a.best))
lab = json.load(open(a.labels)); st = json.load(open(a.stats))
feats = [x.strip() for x in a.feats.split(",")]


def S(f): return st.get("f" + f) or st.get(f) or {}
def fr(f): return S(f).get("fire_rate", 0)


cards = []
for f in feats:
    v = lab.get(f, {}); p = prof.get(f, {})
    peak = [{"fen": e["fen"], "u": e["uci"], "b": best.get(e["fen"] + "|" + e["uci"], "")} for e in p.get("peak", [])[:a.n]]
    med = [{"fen": e["fen"], "u": e["uci"], "b": best.get(e["fen"] + "|" + e["uci"], "")} for e in p.get("median", [])[:a.n]]
    cards.append({"id": int(f), "chip": v.get("chip", ""), "label": v.get("label", ""),
                  "cons": v.get("consistency", 0), "fire": round(fr(f) * 100, 2),
                  "peak": peak, "median": med})

HTML = r"""<!DOCTYPE html><html><head><meta charset=UTF-8>
<title>d64_k1 feature boards</title>
<style>
body{background:#faf8f4;color:#1b1b22;font-family:'IBM Plex Sans',system-ui,sans-serif;margin:0;padding:24px 32px}
h1{font-family:Georgia,serif;font-size:22px;margin:0 0 18px}
.feat{background:#fff;border:1px solid #e8e3d9;border-radius:12px;padding:18px 20px;margin-bottom:22px}
.feat h2{font-family:Georgia,serif;font-size:18px;margin:0 0 4px}
.feat .meta{font-family:ui-monospace,monospace;font-size:12px;color:#8a8a96;margin-bottom:6px}
.feat .desc{font-size:13px;color:#54555f;line-height:1.5;margin-bottom:14px;max-width:900px}
.bandlab{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#9a6b2f;margin:12px 0 6px}
.boards{display:flex;gap:8px;flex-wrap:wrap}
.bd{width:128px}.bd .cap{font-size:9px;color:#8a8a96;font-family:ui-monospace;margin-top:2px}
.legend{font-family:ui-monospace;font-size:11px;color:#8a8a96;margin-bottom:14px}
.r{color:#c0392b}.g{color:#1f8a4c}
</style></head><body>
<h1>d64_k1 — feature boards (top + median)</h1>
<div class=legend><span class=r>▶ red = move played</span> &nbsp; <span class=g>▶ green = best move</span></div>
<div id=app></div>
<script>
const DATA=__DATA__;
const P={k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟',K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙'};
const SQ=128/8;let _m=0;
function sqxy(s,fl){let f=s.charCodeAt(0)-97,r=8-(+s[1]);if(fl){f=7-f;r=7-r}return[f*SQ+SQ/2,r*SQ+SQ/2]}
function board(o){const rows=o.fen.split(' ')[0].split('/');const fl=o.fen.split(' ')[1]==='b';
  let c='';for(let dr=0;dr<8;dr++)for(let df=0;df<8;df++){c+=`<rect x=${df*SQ} y=${dr*SQ} width=${SQ} height=${SQ} fill="${(dr+df)%2?'#b88a5a':'#efe2cf'}"/>`}
  for(let r=0;r<8;r++){let f=0;for(const ch of rows[r]){if(/\d/.test(ch)){f+=+ch;continue}const dr=fl?7-r:r,df=fl?7-f:f;
    c+=`<text x=${df*SQ+SQ/2} y=${dr*SQ+SQ/2+1} font-size=${SQ*0.82} text-anchor=middle dominant-baseline=central>${P[ch]||''}</text>`;f++}}
  function arr(u,col,id){if(!u||u.length<4)return'';let[x1,y1]=sqxy(u.slice(0,2),fl),[x2,y2]=sqxy(u.slice(2,4),fl);
    const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L;x1+=ux*SQ*.3;y1+=uy*SQ*.3;x2-=ux*SQ*.34;y2-=uy*SQ*.34;
    return `<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke=${col} stroke-width=5 stroke-linecap=round opacity=.9 marker-end="url(#${id})"/>`}
  const i=_m++,rc='r'+i,gc='g'+i;
  const d=`<defs><marker id=${rc} markerWidth=3 markerHeight=3 refX=2 refY=1.5 orient=auto><path d="M0,0 L3,1.5 L0,3 z" fill=#c0392b /></marker><marker id=${gc} markerWidth=3 markerHeight=3 refX=2 refY=1.5 orient=auto><path d="M0,0 L3,1.5 L0,3 z" fill=#1f8a4c /></marker></defs>`;
  return `<svg width=128 height=128 viewBox="0 0 128 128" style="border-radius:4px;display:block">${d}${c}${arr(o.b,'#1f8a4c',gc)}${arr(o.u,'#c0392b',rc)}</svg>`}
function cell(o){const url='https://www.chess.com/analysis?fen='+encodeURIComponent(o.fen);
  return `<div class=bd><a href="${url}" target=_blank>${board(o)}</a></div>`}
let h='';for(const f of DATA){
  h+=`<div class=feat><h2>f${f.id} — ${f.chip}</h2><div class=meta>fires ${f.fire}% · consistency ${f.cons}</div><div class=desc>${f.label}</div>
  <div class=bandlab>Top activating (peak)</div><div class=boards>${f.peak.map(cell).join('')}</div>
  <div class=bandlab>Median activating (typical)</div><div class=boards>${f.median.map(cell).join('')}</div></div>`}
document.getElementById('app').innerHTML=h;
</script></body></html>"""
import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write(HTML.replace("__DATA__", json.dumps(cards, separators=(",", ":"))))
print(f"wrote {a.out} — {len(cards)} features")
