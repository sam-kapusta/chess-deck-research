#!/usr/bin/env python3
"""Render the v3 taxonomy as a navigable SPA atlas (warm-paper editorial style).

Matches the look of the good prior atlas (output/taxonomy_v2/chess_taxonomy_atlas.html):
Fraunces display + IBM Plex body/mono, paper palette, left sidebar with colored bucket dots,
breadcrumb nav, card grid, live search. Three-level taxonomy: character group (self-inflicted /
omission / endgame) -> bucket -> sub-bucket -> feature.

KEY: a compact DATA JSON is embedded (chip/label/stats + top-N FENs per feature — NO inline SVG).
Boards are drawn CLIENT-SIDE from FEN on feature-expand. This keeps the file ~2-3MB (the old
inline-SVG tree was 176MB and choked the browser).

Run locally:
  python3 render_atlas_v3.py --boards 6 --out output/atlas/atlas_v3_d2048_k6.html
"""
import argparse, json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--boards", type=int, default=6)
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="output/relabel_v2_neutral_d2048_k6.json")
ap.add_argument("--leaf", default="output/feature_leaf_v3_d2048_k6.json")
ap.add_argument("--buckets", default="output/buckets_v3_d2048_k6.json")
ap.add_argument("--stats", default="output/see_stats_d2048_k6.json")
ap.add_argument("--profiles", default="/tmp/d2048_k6_profiles.json")
ap.add_argument("--best", default="/tmp/best_uci_map.json")
a = ap.parse_args()

lab = json.load(open(a.labels))
leaf = json.load(open(a.leaf))
buckets = json.load(open(a.buckets))
st = json.load(open(a.stats))
prof = json.load(open(a.profiles))
try: best_map = json.load(open(a.best))
except Exception: best_map = {}

def S(f): return st.get("f" + f) or st.get(f) or {}
def fr(f): return S(f).get("fire_rate", 0)

# warm palette per character group; one hue per bucket within
CHAR_ORDER = ["self-inflicted", "omission", "phase"]
CHAR_LABEL = {"self-inflicted": "Self-Inflicted", "omission": "Omission", "phase": "Endgame"}
CHAR_BLURB = {
    "self-inflicted": "The move you played is the blunder — it loses your own material.",
    "omission": "Your move was materially safe, but you missed a stronger one.",
    "phase": "Endgame-specific technique errors.",
}
# bucket colors (warm editorial spectrum, grouped by character)
BCOLOR = {
    "left_hanging": "#c0492f", "abandoned_defense": "#b5532b", "greedy_capture": "#a8612f",
    "premature_trade": "#9a6b2f", "unsound_aggression": "#b04a3a", "pointless_check": "#8a5a3a",
    "king_safety": "#c43d4f",
    "missed_hanging": "#3f7d6e", "missed_tactic": "#2f6b7d", "missed_check_mate": "#456b9a",
    "passive_play": "#5a7d5a",
    "endgame_technique": "#7a5a8a",
}
BNAME = {b["id"]: b["name"] for b in buckets}
BDESC = {b["id"]: b["desc"] for b in buckets}
CHAR = {b["id"]: b["char"] for b in buckets}
border = {b["id"]: i for i, b in enumerate(buckets)}

# build tree: char -> bucket -> sub -> [fids]
tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f, v in leaf.items():
    if v["bucket"] == "unassignable": continue
    tree[CHAR.get(v["bucket"], "phase")][v["bucket"]][v["sub"]].append(f)

def feat_obj(f):
    v = lab[f]; s = S(f)
    boards = [{"fen": ex["fen"], "u": ex["uci"], "b": best_map.get(ex["fen"] + "|" + ex["uci"], "")}
              for ex in prof.get(f, {}).get("examples", [])[:a.boards]]
    return {
        "id": int(f), "chip": v.get("chip", ""), "label": v.get("label", ""),
        "cons": v.get("consistency", 0), "fire": round(fr(f) * 100, 2),
        "mixed": bool(v.get("mixed")),
        "loses": round(s.get("blunder_hangs_own_pct", 0) * 100),
        "wins": round(s.get("best_wins_material_pct", 0) * 100),
        "boards": boards,
    }

# assemble DATA: groups[].buckets[].subs[].features[]
groups = []
for ch in CHAR_ORDER:
    if ch not in tree: continue
    bks = []
    for bid in sorted(tree[ch], key=lambda k: border.get(k, 99)):
        subs_d = tree[ch][bid]
        subs = []
        # sort by fire-coverage, but push "coarse detectors" (⚠) clusters to the bottom
        for sub in sorted(subs_d, key=lambda s: (s.startswith("⚠"), -sum(fr(x) for x in subs_d[s]))):
            fids = sorted(subs_d[sub], key=lambda x: -fr(x))
            subs.append({"name": sub, "n": len(fids),
                         "fire": round(sum(fr(x) for x in fids) * 100, 1),
                         "features": [feat_obj(x) for x in fids]})
        nb = sum(len(s["features"]) for s in subs)
        bks.append({"id": bid, "name": BNAME[bid], "desc": BDESC[bid], "color": BCOLOR.get(bid, "#9a6b2f"),
                    "n": nb, "fire": round(sum(fr(x) for s in subs_d.values() for x in s) * 100), "subs": subs})
    groups.append({"char": ch, "label": CHAR_LABEL[ch], "blurb": CHAR_BLURB[ch],
                   "n": sum(b["n"] for b in bks), "buckets": bks})

nfeat = sum(g["n"] for g in groups)
nun = sum(1 for v in leaf.values() if v["bucket"] == "unassignable")
DATA = {"nfeat": nfeat, "nun": nun, "nbuckets": len(buckets), "groups": groups}

HTML = r"""<!DOCTYPE html><html lang=en><head><meta charset=UTF-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Chess Mistake Taxonomy — d2048_k6</title>
<link rel=preconnect href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel=stylesheet>
<style>
:root{--bg:#faf8f4;--panel:#fff;--ink:#1b1b22;--ink2:#54555f;--muted:#8a8a96;--line:#e8e3d9;--line2:#d8d2c4;--accent:#9a6b2f;--chip:#f3efe7;--red:#c0492f;--grn:#2f7d5e}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;display:flex;flex-direction:column;overflow:hidden}
#top{height:56px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:18px;padding:0 24px;background:var(--panel);flex-shrink:0;z-index:10}
#top .logo{font-family:'Fraunces',serif;font-size:18px;font-weight:600;letter-spacing:.01em}
#top .logo span{color:var(--accent)}
#crumbs{font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:8px}
#crumbs b{cursor:pointer;font-weight:500;color:var(--ink2)}#crumbs b:hover{color:var(--accent)}
#crumbs .sep{color:var(--line2)}
#stat{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);display:flex;gap:18px}
#stat b{color:var(--accent);font-weight:500}
#app{flex:1;display:flex;overflow:hidden}
#side{width:248px;border-right:1px solid var(--line);overflow-y:auto;background:var(--panel);flex-shrink:0;padding:8px 0}
.shdr{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:12px 18px 5px}
.sb{width:100%;border:none;background:none;display:flex;align-items:center;gap:10px;padding:7px 18px;cursor:pointer;font-family:'IBM Plex Sans';font-size:12.5px;color:var(--ink2);text-align:left;border-left:3px solid transparent}
.sb:hover{background:var(--chip)}.sb.on{background:var(--chip);color:var(--ink);font-weight:600;border-left-color:var(--c)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--c);flex-shrink:0}
.sb .lbl{flex:1;line-height:1.2}.sb .n{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted)}
#main{flex:1;overflow-y:auto;padding:26px 32px 80px}
.head{display:flex;align-items:center;gap:13px;margin-bottom:6px}
.head .gdot{width:13px;height:13px;border-radius:50%;flex-shrink:0}
.head h1{font-family:'Fraunces',serif;font-size:27px;font-weight:600;letter-spacing:-.01em}
.sub{font-size:13px;color:var(--ink2);margin-bottom:20px;line-height:1.55;max-width:780px}
.sub b{color:var(--accent);font-family:'IBM Plex Mono';font-weight:500}
#search{width:100%;max-width:440px;background:var(--panel);border:1px solid var(--line2);border-radius:7px;padding:9px 13px;font-family:'IBM Plex Sans';font-size:13px;color:var(--ink);margin-bottom:22px;outline:none;display:block}
#search:focus{border-color:var(--accent)}#search::placeholder{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}
.glcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:17px 18px 15px;cursor:pointer;position:relative;overflow:hidden;transition:transform .12s,box-shadow .12s,border-color .12s}
.glcard::after{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--c),transparent 62%);opacity:.05;transition:opacity .2s}
.glcard:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(70,55,30,.1);border-color:var(--line2)}.glcard:hover::after{opacity:.12}
.glcard .gdot{width:12px;height:12px;border-radius:50%;background:var(--c);margin-bottom:12px}
.glcard h3{font-family:'Fraunces',serif;font-size:16px;font-weight:600;margin-bottom:7px;line-height:1.2;position:relative}
.glcard .cdef{font-size:11.5px;color:var(--ink2);line-height:1.45;margin-bottom:11px;position:relative;min-height:33px}
.glcard .meta{font-family:'IBM Plex Mono';font-size:10.5px;color:var(--muted);position:relative}.glcard .meta b{color:var(--accent);font-weight:500}
.subhd{font-family:'Fraunces',serif;font-size:14px;color:var(--ink2);margin:22px 0 9px;display:flex;align-items:baseline;gap:9px;border-bottom:1px solid var(--line);padding-bottom:5px}
.subhd .sn{font-family:'IBM Plex Mono';font-size:10px;color:var(--muted);font-weight:400}
.gc{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px 16px;cursor:pointer;position:relative;overflow:hidden;transition:transform .12s,box-shadow .12s,border-color .12s}
.gc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--c);opacity:0;transition:opacity .12s}
.gc:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(70,55,30,.08);border-color:var(--line2)}
.gc:hover::before,.gc.open::before{opacity:1}.gc.open{border-color:var(--line2);grid-column:1/-1}
.gc h4{font-family:'Fraunces',serif;font-size:14.5px;font-weight:600;line-height:1.25;margin-bottom:7px}
.gc .stats{display:flex;gap:12px;font-family:'IBM Plex Mono';font-size:10px;color:var(--muted);flex-wrap:wrap;align-items:center}
.gc .stats b{color:var(--accent);font-weight:500}
.gc .stats .blob{color:var(--red);font-weight:600}
.gc .stats .mix b{color:#b5852b}
.mixbadge{font-family:'IBM Plex Mono';font-size:9px;font-weight:500;color:#8a6d3b;background:#f0e6d2;border:1px solid #e0d2b4;border-radius:3px;padding:1px 5px;vertical-align:middle;letter-spacing:.03em}
.gc .det{display:none;margin-top:13px;padding-top:11px;border-top:1px solid var(--line)}.gc.open .det{display:block}
.gc .lab{font-family:'IBM Plex Mono';font-size:9.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.gc .desc{font-size:12px;color:var(--ink2);line-height:1.5;margin-bottom:12px}
.boards{display:flex;gap:9px;flex-wrap:wrap}
.bd{width:148px}.bd .cap{font-size:9.5px;color:var(--muted);margin-top:3px;font-family:'IBM Plex Mono'}
.bd a{text-decoration:none}
.bdg{display:inline-block;font-family:'IBM Plex Mono';font-size:9px;padding:1px 5px;border-radius:3px;background:var(--chip);color:var(--ink2)}
#side::-webkit-scrollbar,#main::-webkit-scrollbar{width:9px}#side::-webkit-scrollbar-thumb,#main::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
.fade{animation:f .22s ease both}@keyframes f{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.fid{font-family:'IBM Plex Mono';font-size:10px;color:var(--muted)}
.legend{font-family:'IBM Plex Mono';font-size:10px;color:var(--muted);margin-top:4px}.legend .r{color:var(--red)}.legend .g{color:var(--grn)}
</style></head><body>
<div id=top>
  <div class=logo>Chess Mistake <span>Taxonomy</span></div>
  <div id=crumbs></div>
  <div id=stat></div>
</div>
<div id=app>
  <div id=side></div>
  <div id=main></div>
</div>
<script>
const DATA=__DATA__;
const BUCKET={};DATA.groups.forEach(g=>g.buckets.forEach(b=>{b._char=g.char;BUCKET[b.id]=b}));
const PIECES={k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟',K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙'};
const SQ=148/8;
// FEN -> inline SVG board. arrows: played(red) from uci, best(green) from b.
function sqxy(sq,flip){let f=sq.charCodeAt(0)-97,r=8-(+sq[1]);if(flip){f=7-f;r=7-r}return [f*SQ+SQ/2,r*SQ+SQ/2];}
function board(o){
  const rows=o.fen.split(' ')[0].split('/');const wtm=o.fen.split(' ')[1]!=='b';const flip=!wtm;
  // 1) draw ALL 64 squares as the board background
  let cells='';
  for(let dr=0;dr<8;dr++)for(let df=0;df<8;df++){
    const dark=(dr+df)%2===1;
    cells+=`<rect x="${df*SQ}" y="${dr*SQ}" width="${SQ}" height="${SQ}" fill="${dark?'#b88a5a':'#efe2cf'}"/>`;
  }
  // 2) overlay pieces
  for(let r=0;r<8;r++){let f=0;for(const ch of rows[r]){
    if(/\d/.test(ch)){f+=+ch;continue;}
    const dr=flip?7-r:r, df=flip?7-f:f;const x=df*SQ,y=dr*SQ;
    cells+=`<text x="${x+SQ/2}" y="${y+SQ/2+0.5}" font-size="${SQ*0.82}" text-anchor="middle" dominant-baseline="central">${PIECES[ch]||''}</text>`;
    f++;
  }}
  // arrows
  function arrow(uci,col){if(!uci||uci.length<4)return'';const[a,b]=[uci.slice(0,2),uci.slice(2,4)];
    const[x1,y1]=sqxy(a,flip),[x2,y2]=sqxy(b,flip);
    return `<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke="${col}" stroke-width=4 stroke-linecap=round opacity=.85 marker-end="url(#ar${col.slice(1)})"/>`;}
  const defs=`<defs><marker id=ar${o._rc} markerWidth=4 markerHeight=4 refX=2 refY=2 orient=auto><path d="M0,0 L4,2 L0,4 z" fill="${o._red}"/></marker><marker id=ar${o._gc} markerWidth=4 markerHeight=4 refX=2 refY=2 orient=auto><path d="M0,0 L4,2 L0,4 z" fill="${o._grn}"/></marker></defs>`;
  return `<svg width=148 height=148 viewBox="0 0 148 148" style="border-radius:5px;display:block">${defs}${cells}${arrow(o.u,o._red)}${arrow(o.b,o._grn)}</svg>`;
}
function boardCell(o){o._red='#c0392b';o._grn='#1f8a4c';o._rc='R';o._gc='G';
  const url='https://www.chess.com/analysis?fen='+encodeURIComponent(o.fen);
  return `<div class=bd><a href="${url}" target=_blank>${board(o)}</a></div>`;}

function setOn(id){document.querySelectorAll('.sb').forEach(b=>b.classList.remove('on'));const e=document.getElementById('sb-'+id);if(e)e.classList.add('on');}
function crumbs(items){document.getElementById('crumbs').innerHTML=items.map((it,i)=>(i>0?'<span class=sep>›</span>':'')+`<b ${it.fn?`onclick="${it.fn}"`:''}>${it.label}</b>`).join('');}
function stat(t){document.getElementById('stat').innerHTML=t;}

function sidebar(){
  let h='';
  DATA.groups.forEach(g=>{
    h+=`<div class=shdr>${g.label} · ${g.n}</div>`;
    g.buckets.forEach(b=>{h+=`<button class=sb id=sb-${b.id} style="--c:${b.color}" onclick="showBucket('${b.id}')"><span class=dot></span><span class=lbl>${b.name}</span><span class=n>${b.n}</span></button>`;});
  });
  document.getElementById('side').innerHTML=h;
}

function home(){
  setOn('');crumbs([{label:'All categories'}]);
  stat(`<span><b>${DATA.nfeat.toLocaleString()}</b> features</span><span><b>${DATA.nbuckets}</b> buckets</span><span><b>${DATA.nun}</b> unassigned</span>`);
  let h=`<div class="head fade"><h1>Chess Mistake Taxonomy</h1></div>
  <div class="sub fade"><b>${DATA.nfeat.toLocaleString()}</b> SAE features (d2048_k6) across <b>${DATA.nbuckets}</b> coaching buckets, grouped by error character. Each feature detects one recurring kind of blunder; the bucket is what a coach would name it. Click a bucket, then a feature to see its boards.</div>`;
  DATA.groups.forEach(g=>{
    h+=`<div class=subhd>${g.label} <span class=sn>${g.blurb} · ${g.n} features</span></div><div class="grid fade">`;
    g.buckets.forEach(b=>{h+=`<div class=glcard style="--c:${b.color}" onclick="showBucket('${b.id}')">
      <div class=gdot></div><h3>${b.name}</h3><div class=cdef>${b.desc}</div>
      <div class=meta><b>${b.n}</b> features · ${b.fire}% fire</div></div>`;});
    h+=`</div>`;
  });
  document.getElementById('main').innerHTML=h;
}

const REG={};
function fcard(f,color){const k='f'+f.id;REG[k]=f;
  const blob=f.fire>=1;
  return `<div class=gc style="--c:${color}" onclick="toggle(this,'${k}')">
    <h4>${esc(f.chip)}${f.mixed?' <span class=mixbadge title="genuinely mixed — top boards share no single mistake">mixed</span>':''}</h4>
    <div class=stats><span class=fid>#${f.id}</span>
      <span class="${blob?'blob':''}">fires <b class="${blob?'blob':''}">${f.fire}%</b>${blob?' ◉':''}</span>
      <span class="${f.mixed?'mix':''}">cons <b>${f.cons}</b></span>
      <span>loses-own <b>${f.loses}%</b></span>
      <span>best-wins <b>${f.wins}%</b></span>
    </div><div class=det></div></div>`;
}
function toggle(el,k){const f=REG[k];const was=el.classList.contains('open');
  document.querySelectorAll('.gc.open').forEach(e=>{e.classList.remove('open');e.querySelector('.det').innerHTML='';});
  if(!was){el.classList.add('open');
    el.querySelector('.det').innerHTML=`<div class=lab>What this feature detects</div><div class=desc>${esc(f.label)}</div>
    <div class=lab>Top activating positions</div><div class=boards>${f.boards.map(boardCell).join('')}</div>
    <div class=legend><span class=r>▶ red = move played (blunder)</span> &nbsp; <span class=g>▶ green = Maia best move</span> · click a board for chess.com</div>`;
    el.scrollIntoView({behavior:'smooth',block:'nearest'});}
}
const esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

function showBucket(id){const b=BUCKET[id];if(!b)return;setOn(id);
  crumbs([{label:'All categories',fn:'home()'},{label:b.name}]);
  stat(`<span><b>${b.n}</b> features</span><span><b>${b.fire}%</b> fire</span>`);
  let h=`<div class="head fade"><div class=gdot style="background:${b.color}"></div><h1>${b.name}</h1></div>
  <div class="sub fade">${b.desc}<br><b>${b.n}</b> features in ${b.subs.length} sub-groups</div>
  <input id=search placeholder="Filter features in this bucket…" oninput="filt('${id}',this.value)">`;
  h+=`<div id=body>`+b.subs.map(s=>`<div class=subhd>${esc(s.name)} <span class=sn>${s.n} · ${s.fire}% fire</span></div>
    <div class="grid fade">${s.features.map(f=>fcard(f,b.color)).join('')}</div>`).join('')+`</div>`;
  document.getElementById('main').innerHTML=h;
}
function filt(id,q){const b=BUCKET[id];q=q.toLowerCase();
  let h='';b.subs.forEach(s=>{const ms=s.features.filter(f=>f.chip.toLowerCase().includes(q)||f.label.toLowerCase().includes(q));
    if(ms.length)h+=`<div class=subhd>${esc(s.name)} <span class=sn>${ms.length}</span></div><div class=grid>${ms.map(f=>fcard(f,b.color)).join('')}</div>`;});
  document.getElementById('body').innerHTML=h||'<div class=sub>no matches</div>';
}
sidebar();home();
</script></body></html>"""

import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write(HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))))
sz = os.path.getsize(a.out) / 1e6
print(f"wrote {a.out} — {nfeat} features, {len(buckets)} buckets, {sz:.1f}MB")
