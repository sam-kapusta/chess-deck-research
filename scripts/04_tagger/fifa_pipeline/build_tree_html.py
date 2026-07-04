"""Build a standalone HTML report: Group -> Cluster -> Feature, with per-band fire RATES at every
level (sparkline + numbers), volume, and the beginner/master discrimination RATIO.

Usage: python build_tree_html.py <fifaSkillRatings.json> <out.html>
Opens with no server — pure HTML/CSS/JS, data inlined.
"""
import sys, json

BANDS = ["600-800","800-1000","1000-1200","1200-1400","1400-1600","1600-1800",
         "1800-2000","2000-2200","2200-2400","2400-2600","2600-2800"]
BAND_SHORT = ["6","8","10","12","14","16","18","20","22","24","26"]  # x100

def ratio(rates):
    nz = [r for r in rates if r]
    return (nz[0]/nz[-1]) if nz and nz[-1] else (float('inf') if nz else 0)

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/Users/samtkap/workspace/chess-deck/src/chess-deck-code/frontend/src/data/fifaSkillRatings.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fifa_tree.html"
    d = json.load(open(src))
    feats = d["features"]
    band_n = d["_band_n"]
    endmoves = d["_endmoves"]

    # ---- assemble a nested model: group -> clusters -> features, each with per-band rates ----
    groups = {}
    # 5 skill groups have group-level rates in d['bands']; Opening groups don't (their group rate is the
    # volume-weighted mean of their families' per-band rates, filled in after clusters are attached).
    for g in d["_groups"]:
        grates = [d["bands"].get(b, {}).get(g) for b in BANDS]
        groups[g] = {"name": g, "rates": grates, "clusters": [], "opening": g.startswith("Openings")}

    for c in d["clusters"]:
        g = c["group"]
        if g not in groups:
            continue
        crates = []
        cfires = []
        for bb in c.get("by_band", []):
            crates.append(bb.get("rate"))
            cfires.append(bb.get("fires", 0))
        # pad if missing
        while len(crates) < 11: crates.append(None); cfires.append(0)
        feat_rows = []
        for fl in c["features"]:
            fe = feats.get(fl)
            if not fe:
                feat_rows.append({"name": fl, "rates": [None]*11, "fires": [0]*11, "total": 0, "missing": True})
                continue
            frates = [fe["by_band"].get(b, {}).get("rate") for b in BANDS]
            ffires = [fe["by_band"].get(b, {}).get("fires", 0) for b in BANDS]
            feat_rows.append({"name": fl, "rates": frates, "fires": ffires,
                              "total": fe.get("total_fires", 0), "missing": False})
        feat_rows.sort(key=lambda r: -r["total"])
        groups[g]["clusters"].append({
            "name": c["name"], "rates": crates, "fires": cfires,
            "total": sum(cfires), "spotlight": c.get("spotlight", True),
            "features": feat_rows,
        })
    for g in groups.values():
        if g.get("opening"):
            # group-level rate = volume-weighted mean of family rates per band (Σ blunders / Σ moves);
            # fires = Σ moves. Sort families by volume (repertoire view, most-played first).
            grates = []
            for i in range(11):
                num = sum((c["rates"][i] or 0) * (c["fires"][i] or 0) for c in g["clusters"])
                den = sum((c["fires"][i] or 0) for c in g["clusters"])
                grates.append((num / den) if den else None)
            g["rates"] = grates
            g["clusters"].sort(key=lambda c: -c["total"])
        else:
            g["clusters"].sort(key=lambda c: -ratio(c["rates"]))

    model = list(groups.values())
    payload = json.dumps(model)

    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>FIFA tagger tree</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--dim:#8b949e;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:24px}
h1{font-size:18px;font-weight:600;margin:0 0 4px}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px;max-width:1100px}
.row{display:flex;align-items:center;padding:4px 8px;border-radius:6px}
.row:hover{background:#1c2230}
.caret{width:12px;color:var(--dim);cursor:pointer;user-select:none;flex:0 0 12px}
.nm{flex:0 0 290px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer}
.group>.row>.nm{font-weight:700;font-size:15px}
.cluster>.row>.nm{font-weight:600}
.feature>.row>.nm{color:var(--dim);font-weight:400}
.ratio{flex:0 0 56px;text-align:right;font-weight:600;padding-right:14px}
.vol{flex:0 0 64px;text-align:right;color:var(--dim);font-size:11px;padding-right:14px}
.cell{flex:0 0 42px;text-align:right;font-size:11px;font-variant-numeric:tabular-nums}
.children{margin-left:18px;border-left:1px solid var(--border);padding-left:4px}
.hidden{display:none}
.tag{font-size:9px;padding:1px 5px;border-radius:3px;background:#30363d;color:#8b949e;margin-left:6px;vertical-align:middle}
.miss{color:#f85149}
.hdr{display:flex;padding:0 8px 6px;color:var(--dim);font-size:10px;border-bottom:1px solid var(--border);margin-bottom:6px;position:sticky;top:0;background:var(--bg)}
.hdr .nm{flex:0 0 290px}.hdr .ratio{flex:0 0 56px;text-align:right;padding-right:14px}.hdr .vol{flex:0 0 64px;text-align:right;padding-right:14px}.hdr .cell{flex:0 0 42px;text-align:right}
.controls{margin-bottom:12px}
button{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:5px;padding:4px 10px;font:inherit;cursor:pointer;margin-right:6px}
button:hover{border-color:#58a6ff}
.legend{color:var(--dim);font-size:11px;margin-top:16px;border-top:1px solid var(--border);padding-top:10px;max-width:1100px}
</style></head><body>
<h1>FIFA tagger &mdash; Group &rarr; Cluster &rarr; Feature</h1>
<div class="sub">Each number = blunder-fire RATE per 1000 moves at that rating band (columns 600&hellip;2800).
Cell shading: brighter = higher rate. A row of numbers that falls left&rarr;right = beginners err more (good drill signal).
Ratio = beginner&divide;master. Fires = total over the 200k-per-band corpus.</div>
<div class="controls">
<button onclick="setAll(false)">Collapse all</button>
<button onclick="setAll(true)">Expand all</button>
<button onclick="expandTo(1)">Groups+Clusters</button>
</div>
<div class="hdr" id="hdr"></div>
<div id="tree"></div>
<div class="legend">
ratio color: <span style="color:#3fb950">&ge;10&times;</span> &nbsp; <span style="color:#6fad4f">6&ndash;10&times;</span> &nbsp;
<span style="color:#d29922">3&ndash;6&times;</span> &nbsp; <span style="color:#e88c30">&lt;3&times;</span> &nbsp; <span class="miss">inverted/none</span>.
&nbsp; <span class="tag">score-only</span> = feeds the group bar, not shown as a drill card.
&nbsp; Group/Cluster rates use their own denominators (Endgame = endgame moves); feature rates use total moves.
</div>
<script>
const DATA = __PAYLOAD__;
const BANDS = __BANDS__;
const COLS = BANDS.map(b=>b.split('-')[0]);  // 600,800,...
function rat(rates){const nz=rates.filter(r=>r);if(!nz.length)return 0;if(!nz[nz.length-1])return Infinity;return nz[0]/nz[nz.length-1];}
function ratColor(r){if(r===Infinity||r===0||isNaN(r))return '#f85149';if(r>=10)return '#3fb950';if(r>=6)return '#6fad4f';if(r>=3)return '#d29922';if(r>=1)return '#e88c30';return '#f85149';}
function fmtRatio(r){if(r===Infinity)return '∞';if(r===0||isNaN(r))return '–';return r.toFixed(1)+'×';}
function fmtRate(v){if(v==null)return '·';v=v*1000;if(v===0)return '0';if(v<1)return v.toFixed(2);if(v<10)return v.toFixed(1);return Math.round(v).toString();}
function cells(rates){
  const vals=rates.map(r=>r==null?null:r*1000);
  const mx=Math.max(...vals.filter(v=>v!=null),0.0001);
  return rates.map((r,i)=>{
    const v=vals[i];
    let bg='transparent', fg='#6b7280';
    if(v!=null){const t=Math.min(v/mx,1); bg='rgba(88,166,255,'+(0.06+0.34*t).toFixed(2)+')'; fg= t>0.55?'#e6edf3':'#9aa4b2';}
    return '<span class="cell" style="background:'+bg+';color:'+fg+'">'+fmtRate(r)+'</span>';
  }).join('');
}
function totalFires(node){return node.fires?node.fires.reduce((a,b)=>a+b,0):(node.total||0);}
function row(node,cls,hasChildren){
  const r=rat(node.rates);const vol=cls==='feature'?node.total:totalFires(node);
  const tag=node.spotlight===false?'<span class="tag">score-only</span>':'';
  const miss=node.missing?'<span class="tag miss">no data</span>':'';
  const caret=hasChildren?'<span class="caret">▸</span>':'<span class="caret"></span>';
  return '<div class="row">'+caret+
    '<span class="nm">'+node.name+tag+miss+'</span>'+
    '<span class="ratio" style="color:'+ratColor(r)+'">'+fmtRatio(r)+'</span>'+
    '<span class="vol">'+(vol?vol.toLocaleString():'–')+'</span>'+
    cells(node.rates)+'</div>';
}
document.getElementById('hdr').innerHTML='<span class="nm">name</span><span class="ratio">ratio</span><span class="vol">fires</span>'+COLS.map(c=>'<span class="cell">'+c+'</span>').join('');
function render(){
  const tree=document.getElementById('tree');let html='';
  for(const g of DATA){
    html+='<div class="group" data-lvl="0">'+row(g,'group',true);
    html+='<div class="children hidden">';
    for(const c of g.clusters){
      html+='<div class="cluster" data-lvl="1">'+row(c,'cluster',c.features.length>0);
      html+='<div class="children hidden">';
      for(const f of c.features){html+='<div class="feature" data-lvl="2">'+row(f,'feature',false)+'</div>';}
      html+='</div></div>';
    }
    html+='</div></div>';
  }
  tree.innerHTML=html;
  tree.querySelectorAll('.row').forEach(r=>{
    const car=r.querySelector('.caret');const kids=r.parentElement.querySelector(':scope > .children');
    if(!kids)return;
    const toggle=()=>{kids.classList.toggle('hidden');car.textContent=kids.classList.contains('hidden')?'▸':'▾';};
    car.onclick=toggle;r.querySelector('.nm').onclick=toggle;
  });
}
function setAll(open){document.querySelectorAll('.children').forEach(c=>{c.classList.toggle('hidden',!open);});
  document.querySelectorAll('.caret').forEach(c=>{if(c.parentElement.parentElement.querySelector(':scope>.children'))c.textContent=open?'▾':'▸';});}
function expandTo(lvl){document.querySelectorAll('.children').forEach(c=>{
  const owner=c.parentElement;const l=parseInt(owner.dataset.lvl);c.classList.toggle('hidden',l>=lvl);});
  document.querySelectorAll('.caret').forEach(c=>{const owner=c.parentElement.parentElement;const kids=owner.querySelector(':scope>.children');
   if(kids)c.textContent=kids.classList.contains('hidden')?'▸':'▾';});}
render();expandTo(1);
</script></body></html>"""
    html = html.replace("__PAYLOAD__", payload).replace("__BANDS__", json.dumps(BANDS))
    open(out, "w").write(html)
    print(f"wrote {out} ({len(model)} groups, {sum(len(g['clusters']) for g in model)} clusters)")

if __name__ == "__main__":
    main()
