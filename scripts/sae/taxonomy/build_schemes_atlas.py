"""Render a SCHEME-COMPARISON atlas: one page, toggle between 5 top-level schemes,
each shows its categories (count + fire% + example chips) as cards. Persona-atlas
styled. Lets Sam visually compare how each axis carves the 1996 features.
"""
import argparse
import json
import html


def esc(s):
    return html.escape(str(s or ""), quote=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.load(open(args.data))
    blob = json.dumps(data, separators=(",", ":"))
    doc = TEMPLATE.replace("__DATA__", blob)
    open(args.out, "w").write(doc)
    print(f"wrote {args.out} ({len(doc)//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess Taxonomy — Scheme Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#faf8f4;--panel:#fff;--ink:#1b1b22;--ink2:#54555f;--muted:#8a8a96;--line:#e8e3d9;--line2:#d8d2c4;--accent:#9a6b2f;--chip:#f3efe7}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;display:flex;flex-direction:column;overflow:hidden}
#top{height:56px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:18px;padding:0 24px;background:var(--panel);flex-shrink:0}
#top .logo{font-family:'Fraunces',serif;font-size:18px;font-weight:600}
#top .logo span{color:var(--accent)}
#stat{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted)}
#tabs{display:flex;gap:6px;padding:14px 24px 0;background:var(--bg);flex-shrink:0;flex-wrap:wrap}
.tab{border:1px solid var(--line2);background:var(--panel);border-radius:7px 7px 0 0;padding:9px 15px;cursor:pointer;font-size:13px;color:var(--ink2);font-family:'IBM Plex Sans'}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
.tab .sub{font-family:'IBM Plex Mono';font-size:10px;opacity:.7;margin-left:6px}
#main{flex:1;overflow-y:auto;padding:22px 24px 60px}
.schemehead{font-family:'Fraunces',serif;font-size:20px;margin-bottom:4px}
.schemesub{font-size:12.5px;color:var(--ink2);margin-bottom:18px;max-width:720px;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.gc{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px 17px;cursor:pointer;position:relative;overflow:hidden;transition:transform .12s,box-shadow .12s}
.gc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--c)}
.gc:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(70,55,30,.08)}
.gc h4{font-family:'Fraunces',serif;font-size:15px;font-weight:600;margin-bottom:6px}
.gc .bar{height:6px;background:var(--chip);border-radius:3px;overflow:hidden;margin:7px 0}
.gc .bar i{display:block;height:100%;background:var(--c)}
.gc .stats{display:flex;gap:14px;font-family:'IBM Plex Mono';font-size:10px;color:var(--muted)}
.gc .stats b{color:var(--accent)}
.gc .ex{margin-top:9px;padding-top:8px;border-top:1px solid var(--line);font-size:11px;color:var(--ink2);line-height:1.5;display:none}
.gc.open .ex{display:block}
.gc .ex div{padding:1px 0}
#main::-webkit-scrollbar{width:8px}#main::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
</style></head><body>
<div id="top"><div class="logo">Chess Taxonomy <span>· Scheme Comparison</span></div>
<div id="stat">1,996 features · maia3 flat k=32 v2</div></div>
<div id="tabs" id="tabs"></div>
<div id="main"></div>
<script>
const DATA=__DATA__;
const SCHEMES=["Mechanism","Thinking Error","Player Theme","Piece","Semantic cluster"];
const DESC={
 "Mechanism":"What concretely went wrong on the board (priority-ordered). Most actionable for drills/diagnosis.",
 "Thinking Error":"WHY the brain failed — the cognitive habit. Most coaching-resonant; balanced spread.",
 "Player Theme":"Broad player-facing buckets — a UI roll-up of the mechanism axis.",
 "Piece":"Which piece you mishandled. Most evenly spread; best as a secondary drill axis.",
 "Semantic cluster":"Raw bge-m3 emergent clusters (before naming) — the unsupervised baseline."
};
const COLORS=["#e8852f","#5b7fc4","#4a9d6f","#c79a3e","#dd6f4a","#7a8c4e","#b3589e","#cc6f8e","#3fa9a0","#cc7a38","#9a6fc0","#6fae54","#5aa6cc","#c4823f","#94785a","#6aa84f","#7a85a3","#d65f9c","#9a9488","#8c6fae"];
let cur="Mechanism";
const tabs=document.getElementById('tabs');
SCHEMES.forEach(s=>{
  const cats={};DATA.features.forEach(f=>{const c=f.schemes[s];cats[c]=(cats[c]||0)+1;});
  const b=document.createElement('div');b.className='tab'+(s==cur?' on':'');
  b.innerHTML=`${s}<span class="sub">${Object.keys(cats).length}</span>`;
  b.onclick=()=>{cur=s;render();};tabs.appendChild(b);
});
function render(){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('on',SCHEMES[i]==cur));
  const agg={};let totFire=0;
  DATA.features.forEach(f=>{const c=f.schemes[cur];if(!agg[c])agg[c]={n:0,fire:0,ex:[]};agg[c].n++;agg[c].fire+=f.fire;totFire+=f.fire;if(agg[c].ex.length<6)agg[c].ex.push(f);});
  const order=Object.keys(agg).sort((a,b)=>agg[b].n-agg[a].n);
  const maxn=Math.max(...order.map(c=>agg[c].n));
  const m=document.getElementById('main');
  m.innerHTML=`<div class="schemehead">${cur}</div><div class="schemesub">${DESC[cur]} — ${order.length} categories, largest ${(maxn/DATA.features.length*100).toFixed(0)}% of features.</div>
  <div class="grid">${order.map((c,i)=>{
    const a=agg[c];const col=COLORS[i%COLORS.length];
    return `<div class="gc" style="--c:${col}" onclick="this.classList.toggle('open')">
      <h4>${c}</h4>
      <div class="bar"><i style="width:${a.n/maxn*100}%"></i></div>
      <div class="stats"><span><b>${a.n}</b> feats (${(a.n/DATA.features.length*100).toFixed(0)}%)</span><span><b>${(a.fire/totFire*100).toFixed(0)}%</b> of fire</span></div>
      <div class="ex">${a.ex.map(f=>`<div>• ${f.chip} <span style="color:var(--muted);font-family:IBM Plex Mono;font-size:9px">${f.fire}%</span></div>`).join('')}</div>
    </div>`;}).join('')}</div>`;
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
