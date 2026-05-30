"""Render taxonomy_v2.json as an interactive HTML atlas, styled to match the
Sandstone Persona Atlas (warm paper palette, Fraunces + IBM Plex, sidebar +
card grid that expands to feature detail).

Category -> feature (2 levels; persona atlas is 3, we collapse the middle).

Usage:
    python3 scripts/sae/taxonomy/build_atlas.py \
        --taxonomy output/taxonomy_v2/taxonomy_v2.json \
        --out output/taxonomy_v2/chess_taxonomy_atlas.html
"""
import argparse
import json
import html

# 20 distinct, warm-leaning colors (persona-atlas palette family)
PALETTE = [
    "#e8852f", "#5b7fc4", "#4a9d6f", "#c79a3e", "#dd6f4a", "#7a8c4e",
    "#b3589e", "#cc6f8e", "#3fa9a0", "#cc7a38", "#9a6fc0", "#6fae54",
    "#5aa6cc", "#c4823f", "#94785a", "#6aa84f", "#7a85a3", "#d65f9c",
    "#9a9488", "#8c6fae",
]


def esc(s):
    return html.escape(str(s or ""), quote=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = json.load(open(args.taxonomy))
    feats = t["features"]
    cats = t["categories"]
    meta = t["meta"]

    # group features by category, preserve vocab order, sort within by confidence desc
    by_cat = {c["id"]: [] for c in cats}
    for fid, f in feats.items():
        if f["category"] in by_cat:
            by_cat[f["category"]].append(f)
    for cid in by_cat:
        by_cat[cid].sort(key=lambda f: -(f.get("confidence") or 0))

    # ordered category list (by feature count desc), with color
    ordered = sorted(cats, key=lambda c: -len(by_cat[c["id"]]))
    colors = {}
    for i, c in enumerate(ordered):
        colors[c["id"]] = PALETTE[i % len(PALETTE)]

    # build compact DATA object for the JS
    data = {
        "sae": meta.get("sae", ""),
        "n_features": meta.get("n_features", len(feats)),
        "categories": [],
    }
    for c in ordered:
        cid = c["id"]
        members = by_cat[cid]
        data["categories"].append({
            "id": cid,
            "name": c["name"],
            "definition": c.get("definition", ""),
            "color": colors[cid],
            "n": len(members),
            "features": [{
                "id": f["feature_id"],
                "chip": f["chip"],
                "title": f.get("title", ""),
                "desc": f.get("description", ""),
                "conf": f.get("confidence", 0),
                "piece": (f.get("fingerprint") or {}).get("dom_piece", ""),
                "piece_frac": round(((f.get("fingerprint") or {}).get("dom_frac") or 0) * 100),
                "cap": round(((f.get("fingerprint") or {}).get("cap_rate") or 0) * 100),
                "chk": round(((f.get("fingerprint") or {}).get("check_rate") or 0) * 100),
                "old": f.get("old_chip", ""),
            } for f in members],
        })

    data_json = json.dumps(data, separators=(",", ":"))

    html_doc = TEMPLATE.replace("__DATA__", data_json).replace(
        "__SAE__", esc(meta.get("sae", "maia3 v2"))
    ).replace("__NFEAT__", str(meta.get("n_features", len(feats)))).replace(
        "__NCAT__", str(len(cats))
    )
    open(args.out, "w").write(html_doc)
    print(f"Wrote {args.out} ({len(html_doc)//1024} KB, {len(feats)} features, {len(cats)} categories)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess Blunder Taxonomy — __SAE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#faf8f4; --panel:#ffffff; --ink:#1b1b22; --ink2:#54555f; --muted:#8a8a96;
  --line:#e8e3d9; --line2:#d8d2c4; --accent:#9a6b2f; --chip:#f3efe7;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;display:flex;flex-direction:column;overflow:hidden}

#top{height:56px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:18px;padding:0 24px;background:var(--panel);flex-shrink:0}
#top .logo{font-family:'Fraunces',serif;font-size:18px;font-weight:600;color:var(--ink);letter-spacing:.01em}
#top .logo span{color:var(--accent)}
#crumbs{font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:8px}
#crumbs b{cursor:pointer;font-weight:500;color:var(--ink2)}
#crumbs b:hover{color:var(--accent)}
#crumbs .sep{color:var(--line2)}
#stat{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);display:flex;gap:18px}
#stat b{color:var(--accent);font-weight:500}

#app{flex:1;display:flex;overflow:hidden}
#side{width:262px;border-right:1px solid var(--line);overflow-y:auto;background:var(--panel);flex-shrink:0;padding:10px 0}
.sb{width:100%;border:none;background:none;display:flex;align-items:center;gap:10px;padding:8px 18px;cursor:pointer;font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;color:var(--ink2);text-align:left;border-left:3px solid transparent;transition:background .1s}
.sb:hover{background:var(--chip)}
.sb.on{background:var(--chip);color:var(--ink);font-weight:600;border-left-color:var(--c)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--c);flex-shrink:0}
.sb .lbl{flex:1;line-height:1.25}
.sb .n{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted)}

#main{flex:1;overflow-y:auto;padding:26px 32px 60px}
.head{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.head .gdot{width:13px;height:13px;border-radius:50%;flex-shrink:0}
.head h1{font-family:'Fraunces',serif;font-size:26px;font-weight:600;letter-spacing:-.01em}
.sub{font-size:13px;color:var(--ink2);margin-bottom:18px;line-height:1.5;max-width:760px}
.sub b{color:var(--accent);font-family:'IBM Plex Mono',monospace;font-weight:500}

#search{width:100%;max-width:440px;background:var(--panel);border:1px solid var(--line2);border-radius:7px;padding:9px 13px;font-family:'IBM Plex Sans';font-size:13px;color:var(--ink);margin-bottom:22px;outline:none;display:block}
#search:focus{border-color:var(--accent)}
#search::placeholder{color:var(--muted)}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}

/* category home cards */
.glcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 18px 16px;cursor:pointer;position:relative;overflow:hidden;transition:transform .12s,box-shadow .12s,border-color .12s}
.glcard::after{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--c),transparent 65%);opacity:.05;transition:opacity .2s}
.glcard:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(70,55,30,.1);border-color:var(--line2)}
.glcard:hover::after{opacity:.11}
.glcard .gdot{width:12px;height:12px;border-radius:50%;background:var(--c);margin-bottom:13px}
.glcard h3{font-family:'Fraunces',serif;font-size:16px;font-weight:600;color:var(--ink);margin-bottom:7px;line-height:1.2;position:relative}
.glcard .cdef{font-size:11.5px;color:var(--ink2);line-height:1.45;margin-bottom:11px;position:relative}
.glcard .meta{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--muted);position:relative}
.glcard .meta b{color:var(--accent);font-weight:500}

/* feature cards */
.gc{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px 17px;cursor:pointer;position:relative;overflow:hidden;transition:transform .12s,box-shadow .12s,border-color .12s}
.gc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--c);opacity:0;transition:opacity .12s}
.gc:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(70,55,30,.08);border-color:var(--line2)}
.gc:hover::before,.gc.open::before{opacity:1}
.gc.open{border-color:var(--line2)}
.gc h4{font-family:'Fraunces',serif;font-size:15px;font-weight:600;color:var(--ink);line-height:1.25;margin-bottom:7px}
.gc .stats{display:flex;gap:13px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);flex-wrap:wrap}
.gc .stats b{color:var(--accent);font-weight:500}
.gc .det{display:none;margin-top:13px;padding-top:11px;border-top:1px solid var(--line)}
.gc.open .det{display:block}
.gc .det .lab{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.gc .det .desc{font-size:12px;color:var(--ink2);line-height:1.5;margin-bottom:11px}
.gc .det .old{font-size:11px;color:var(--muted);font-style:italic}
.gc .det .old s{color:#b08;opacity:.7}

#side::-webkit-scrollbar,#main::-webkit-scrollbar{width:8px}
#side::-webkit-scrollbar-thumb,#main::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
.fade{animation:f .22s ease both}
@keyframes f{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.fid{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted)}
</style></head><body>
<div id="top">
  <div class="logo">Chess Blunder <span>Taxonomy</span></div>
  <div id="crumbs"><b onclick="home()">All categories</b></div>
  <div id="stat"><span><b id="sc">__NCAT__</b> categories</span><span><b id="sf">__NFEAT__</b> features</span><span>__SAE__</span></div>
</div>
<div id="app">
  <div id="side"></div>
  <div id="main"></div>
</div>
<script>
const DATA=__DATA__;
const col=id=>{const c=DATA.categories.find(x=>x.id===id);return c?c.color:'#888';};
document.getElementById('sf').textContent=DATA.n_features.toLocaleString();

// sidebar
const side=document.getElementById('side');
const hb=document.createElement('button');
hb.className='sb on';hb.id='sb-home';hb.style.setProperty('--c','#9a6b2f');
hb.innerHTML=`<div class="dot"></div><span class="lbl">All categories</span>`;
hb.onclick=home;side.appendChild(hb);
DATA.categories.forEach(c=>{
  const b=document.createElement('button');b.className='sb';b.id='sb-'+c.id;b.style.setProperty('--c',c.color);
  b.innerHTML=`<div class="dot"></div><span class="lbl">${c.name}</span><span class="n">${c.n}</span>`;
  b.onclick=()=>showCat(c.id);side.appendChild(b);
});
function setOn(id){document.querySelectorAll('.sb').forEach(b=>b.classList.remove('on'));const e=document.getElementById(id);if(e)e.classList.add('on');}
function crumbs(items){document.getElementById('crumbs').innerHTML=items.map((it,i)=>(i>0?'<span class="sep">›</span>':'')+`<b ${it.fn?`onclick="${it.fn}"`:''}>${it.label}</b>`).join('');}
const escq=s=>(s||'').replace(/'/g,"\\'");

function home(){
  setOn('sb-home');crumbs([{label:'All categories'}]);
  const m=document.getElementById('main');
  m.innerHTML=`<div class="head fade"><h1>Chess Blunder Taxonomy</h1></div>
  <div class="sub fade"><b>${DATA.n_features.toLocaleString()}</b> SAE features across <b>${DATA.categories.length}</b> coaching categories · <b>__SAE__</b>. Each feature detects one kind of mistake; categories are what a coach would name. Click a category, then a feature to see its full pattern.</div>
  <input id="search" placeholder="Search chips and descriptions…" oninput="searchAll(this.value)">
  <div class="grid fade" id="grid">${DATA.categories.map(c=>
    `<div class="glcard" style="--c:${c.color}" onclick="showCat('${c.id}')">
      <div class="gdot"></div><h3>${c.name}</h3>
      <div class="cdef">${c.definition||''}</div>
      <div class="meta"><b>${c.n}</b> features &nbsp;·&nbsp; ${(c.n/DATA.n_features*100).toFixed(0)}% of total</div></div>`).join('')}</div>`;
}

function showCat(id){
  const c=DATA.categories.find(x=>x.id===id);if(!c)return;
  setOn('sb-'+id);crumbs([{label:'All categories',fn:'home()'},{label:c.name}]);
  const m=document.getElementById('main');
  m.innerHTML=`<div class="head fade"><div class="gdot" style="background:${c.color}"></div><h1>${c.name}</h1></div>
  <div class="sub fade">${c.definition||''}<br><b>${c.n}</b> features</div>
  <input id="search" placeholder="Filter features in this category…" oninput="filterCat('${id}',this.value)">
  <div class="grid fade" id="grid">${c.features.map((f,i)=>card(f,c.color,i)).join('')}</div>`;
}

const REG={};
function card(f,cc,i){
  const k='k'+i+'_'+(f.id);REG[k]=f;
  return `<div class="gc" style="--c:${cc}" onclick="toggle(this,'${k}')">
    <h4>${f.chip}</h4>
    <div class="stats">
      <span class="fid">#${f.id}</span>
      <span><b>${f.piece}</b> ${f.piece_frac}%</span>
      ${f.cap>=40?`<span>cap <b>${f.cap}%</b></span>`:''}
      ${f.chk>=40?`<span>chk <b>${f.chk}%</b></span>`:''}
      <span>conf <b>${f.conf}</b></span>
    </div>
    <div class="det"></div></div>`;
}
function toggle(el,k){
  const f=REG[k];const was=el.classList.contains('open');
  el.parentElement.querySelectorAll('.gc.open').forEach(e=>e.classList.remove('open'));
  if(!was){el.classList.add('open');
    el.querySelector('.det').innerHTML=
      `<div class="lab">Pattern</div><div class="desc">${f.desc}</div>`+
      (f.title?`<div class="lab">One-line</div><div class="desc">${f.title}</div>`:'')+
      (f.old?`<div class="old">was: <s>${f.old}</s></div>`:'');
    el.scrollIntoView({behavior:'smooth',block:'nearest'});}
}
function filterCat(id,q){
  const c=DATA.categories.find(x=>x.id===id);q=q.toLowerCase();
  const ms=c.features.filter(f=>f.chip.toLowerCase().includes(q)||(f.desc||'').toLowerCase().includes(q));
  document.getElementById('grid').innerHTML=ms.map((f,i)=>card(f,c.color,i)).join('')||'<div class="sub">no matches</div>';
}
function searchAll(q){
  const grid=document.getElementById('grid');if(!grid)return;
  if(!q){home();return;}q=q.toLowerCase();
  const res=[];DATA.categories.forEach(c=>c.features.forEach(f=>{
    if(f.chip.toLowerCase().includes(q)||(f.desc||'').toLowerCase().includes(q))res.push([f,c.color]);}));
  grid.className='grid fade';
  grid.innerHTML=res.slice(0,90).map(([f,cc],i)=>card(f,cc,i)).join('')||'<div class="sub">no matches</div>';
}
home();
</script></body></html>
"""

if __name__ == "__main__":
    main()
