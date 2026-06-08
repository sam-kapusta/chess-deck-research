#!/usr/bin/env python3
"""Build an evaluation atlas for the rule-based mistake tagger.

Reads mistake_tags.json (output of run_corpus.py) and emits a self-contained HTML page:
each tag -> frequency, then a grid of example boards (inline SVG, best=green / played=red /
refutation=blue arrows) with the line + evidence + cp_loss. This is what Sam scrolls through to
judge whether each tag is CRISP (fires on the right positions) or noisy.

  python3 build_atlas.py --tags output/mistake_tags.json --out output/tag_atlas.html [--per-tag 12]

Board SVG renderer is the same convention as render_atlas_v3.py (green best / red played / blue refutation).
"""
import argparse, json, os, html, chess
from collections import Counter, defaultdict


def line_to_san(fen, evidence):
    """Evidence like 'line=g6h5 h6h7 g8f8' -> readable SAN 'g5 hxg6 Kf8' for the atlas caption."""
    if not evidence or "line=" not in evidence:
        return evidence or ""
    ucis = evidence.split("line=", 1)[1].split()
    try:
        b = chess.Board(fen)
    except Exception:
        return evidence
    sans = []
    for u in ucis:
        try:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                break
            sans.append(b.san(mv)); b.push(mv)
        except Exception:
            break
    return ("line: " + " ".join(sans)) if sans else evidence

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="output/mistake_tags.json")
ap.add_argument("--out", default="output/tag_atlas.html")
ap.add_argument("--per-tag", type=int, default=12)
ap.add_argument("--min-count", type=int, default=2)
a = ap.parse_args()

data = json.load(open(a.tags))
N = len(data)

# index positions by tag label
by_tag = defaultdict(list)
for o in data:
    for t in o["tags"]:
        by_tag[t["label"]].append((o, t))

counts = Counter({lab: len(v) for lab, v in by_tag.items()})

# group tags by the category the tagger assigns (rough — re-derive here to avoid importing tagger)
def categorize(label):
    l = label.lower()
    if any(w in l for w in ["mate", "check", "fork", "pin", "skewer", "discovered", "deflection",
                            "attraction", "clearance", "interference", "zwischenzug", "overload",
                            "x-ray", "sacrifice", "double check", "f2/f7", "trapped piece", "capture of defender"]):
        return "Tactical"
    if any(w in l for w in ["capture", "exchange", "hung", "bad capture", "material", "wrong piece"]):
        return "Material"
    if any(w in l for w in ["king", "castl", "attack"]):
        return "King Safety"
    if "endgame" in l or "zugzwang" in l or "promotion" in l:
        return "Endgame"
    if any(w in l for w in ["pawn", "tempo", "development", "advanced", "doubled", "isolated", "backward"]):
        return "Positional"
    if any(w in l for w in ["blunder while", "only move", "move order", "opening", "middlegame", "endgame"]):
        return "Meta / Phase"
    return "Other"

cat_tags = defaultdict(list)
for lab, n in counts.most_common():
    if n >= a.min_count:
        cat_tags[categorize(lab)].append((lab, n))

# build the JSON the page renders: for each tag, sample positions across the cp_loss range.
# NB cp_loss > ~5000 is a forced-mate SENTINEL (not a real centipawn loss); sorting by raw cp_loss
# floods every tag with the same mate positions. We clamp the sentinel for sorting and sample a
# spread (clearest worst-but-real + an even spread across the rest) so Sam sees the tag's true range.
MATE_SENTINEL = 5000

def _eval_key(o):
    cp = o.get("cp_loss") or 0
    return min(cp, MATE_SENTINEL)  # clamp so mate positions don't all sort to the very top

def sample(lab):
    items = by_tag[lab]
    # Show the TYPICAL case, not mate outliers: prefer non-sentinel positions (cp_loss <= MATE_SENTINEL)
    # for most slots, and include a couple of forced-mate examples only if that's all there is.
    non_mate = [it for it in items if (it[0].get("cp_loss") or 0) <= MATE_SENTINEL]
    mate = [it for it in items if (it[0].get("cp_loss") or 0) > MATE_SENTINEL]
    pool = non_mate if len(non_mate) >= a.per_tag else (non_mate + mate)
    pool = sorted(pool, key=lambda x: -_eval_key(x[0]))
    n = len(pool)
    if n <= a.per_tag:
        chosen = pool
    else:
        # half from the clearest (high real cp_loss), half evenly spread across the remainder
        half = a.per_tag // 2
        top = pool[:half]
        step = max(1, (n - half) // (a.per_tag - half))
        spread = pool[half::step][:a.per_tag - half]
        chosen = top + spread
    out = []
    for o, t in chosen[:a.per_tag]:
        # refutation uci0 / best uci0 for arrows
        out.append({
            "fen": o["fen"], "u": o["played"], "b": o.get("best", ""),
            "ps": o.get("played_san", ""), "bs": o.get("best_san", ""),
            "cp": o.get("cp_loss", 0), "ev": line_to_san(o["fen"], t.get("evidence", "")),
            "dir": t.get("direction", ""),
        })
    return out

PAGE = {
    "n": N,
    "cats": [{"name": c, "tags": [{"label": lab, "n": n, "pct": round(100*n/N, 1), "ex": sample(lab)}
                                  for lab, n in tags]}
             for c, tags in cat_tags.items()],
}

HTML = """<!doctype html><html><head><meta charset=utf-8><title>Mistake Tag Atlas</title>
<style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#13151a;color:#e8e6e1}
#top{position:sticky;top:0;background:#1a1d24;border-bottom:1px solid #2a2e38;padding:14px 22px;z-index:10}
.logo{font-size:19px;font-weight:700;letter-spacing:-.3px}.logo span{color:#d99}
#stat{color:#8a909c;font-size:13px;margin-top:3px}
#app{display:flex}#side{width:260px;flex:none;border-right:1px solid #2a2e38;height:calc(100vh - 60px);overflow:auto;position:sticky;top:60px}
#main{flex:1;padding:20px 26px;min-width:0}
.shdr{padding:12px 16px 5px;color:#6cf;font-size:11px;text-transform:uppercase;letter-spacing:.8px;font-weight:700}
.sb{display:flex;align-items:center;gap:8px;width:100%;border:0;background:none;color:#cfd2d8;padding:6px 16px;cursor:pointer;text-align:left;font-size:13px}
.sb:hover{background:#222630}.sb .n{margin-left:auto;color:#787e8a;font-variant-numeric:tabular-nums}
.sb.on{background:#2a3140;color:#fff}
h1{font-size:24px;margin:0 0 4px}.sub{color:#9aa0ac;margin-bottom:20px;max-width:760px}
.taghdr{display:flex;align-items:baseline;gap:12px;margin:26px 0 4px;padding-top:8px;border-top:1px solid #232730}
.taghdr h2{font-size:19px;margin:0}.taghdr .freq{color:#d99;font-weight:700}.taghdr .dir{font-size:11px;padding:2px 8px;border-radius:10px;background:#2a3140;color:#9cf}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:14px;margin:12px 0 8px}
.bd{background:#1a1d24;border:1px solid #262a34;border-radius:8px;padding:8px}
.bd a{text-decoration:none}
.cap{font-size:11px;color:#9aa0ac;margin-top:5px;line-height:1.35}
.cap b{color:#e8e6e1}.cap .cp{color:#d99}
.legend{font-size:12px;color:#8a909c;margin:2px 0 16px}
.legend i{font-style:normal;font-weight:700}.lg{color:#1f8a4c}.lr{color:#c0392b}.lb{color:#2563c9}
</style></head><body>
<div id=top><div class=logo>Mistake Tag <span>Atlas</span></div><div id=stat></div></div>
<div id=app><div id=side></div><div id=main></div></div>
<script>
const DATA=__DATA__;
const PIECES={k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟',K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙'};
const SQ=148/8;
function sqxy(sq,flip){let f=sq.charCodeAt(0)-97,r=8-(+sq[1]);if(flip){f=7-f;r=7-r}return [f*SQ+SQ/2,r*SQ+SQ/2];}
function board(o){
  const rows=o.fen.split(' ')[0].split('/');const wtm=o.fen.split(' ')[1]!=='b';const flip=!wtm;
  let cells='';
  for(let dr=0;dr<8;dr++)for(let df=0;df<8;df++){const dark=(dr+df)%2===1;
    cells+=`<rect x="${df*SQ}" y="${dr*SQ}" width="${SQ}" height="${SQ}" fill="${dark?'#b88a5a':'#efe2cf'}"/>`;}
  for(let r=0;r<8;r++){let f=0;for(const ch of rows[r]){
    if(/\\d/.test(ch)){f+=+ch;continue;}
    const dr=flip?7-r:r, df=flip?7-f:f;const x=df*SQ,y=dr*SQ;
    cells+=`<text x="${x+SQ/2}" y="${y+SQ/2+0.5}" font-size="${SQ*0.82}" text-anchor="middle" dominant-baseline="central">${PIECES[ch]||''}</text>`;f++;}}
  function arrow(uci,col,mid){if(!uci||uci.length<4)return'';const a=uci.slice(0,2),b=uci.slice(2,4);
    let[x1,y1]=sqxy(a,flip),[x2,y2]=sqxy(b,flip);const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len;
    x1+=ux*SQ*0.30;y1+=uy*SQ*0.30;x2-=ux*SQ*0.34;y2-=uy*SQ*0.34;
    return `<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke="${col}" stroke-width=5.5 stroke-linecap=round opacity=.9 marker-end="url(#${mid})"/>`;}
  const mk=(id,c)=>`<marker id=${id} markerWidth=3.2 markerHeight=3.2 refX=2.2 refY=1.6 orient=auto><path d="M0,0 L3.2,1.6 L0,3.2 z" fill="${c}"/></marker>`;
  const id=o._id;const rc='r'+id,gc='g'+id;
  const defs=`<defs>${mk(rc,'#c0392b')}${mk(gc,'#1f8a4c')}</defs>`;
  return `<svg width=148 height=148 viewBox="0 0 148 148" style="border-radius:5px;display:block">${defs}${cells}${arrow(o.b,'#1f8a4c',gc)}${arrow(o.u,'#c0392b',rc)}</svg>`;}
let _bid=0;
function bcell(o){o._id=_bid++;
  const url='https://www.chess.com/analysis?fen='+encodeURIComponent(o.fen);
  const cap=`<div class=cap>played <b>${o.ps}</b> · best <b>${o.bs}</b> · <span class=cp>-${o.cp}cp</span><br>${o.ev||''}</div>`;
  return `<div class=bd><a href="${url}" target=_blank>${board(o)}</a>${cap}</div>`;}
function stat(){document.getElementById('stat').innerHTML=`<b>${DATA.n.toLocaleString()}</b> positions · <b>${DATA.cats.reduce((s,c)=>s+c.tags.length,0)}</b> tags across <b>${DATA.cats.length}</b> categories`;}
function sidebar(){let h='';DATA.cats.forEach((c,ci)=>{h+=`<div class=shdr>${c.name}</div>`;
  c.tags.forEach((t,ti)=>{h+=`<button class=sb onclick="show(${ci},${ti})"><span class=lbl>${t.label}</span><span class=n>${t.n}</span></button>`;});});
  document.getElementById('side').innerHTML=h;}
function show(ci,ti){
  document.querySelectorAll('.sb').forEach(b=>b.classList.remove('on'));
  const tag=DATA.cats[ci].tags[ti];
  let h=`<div class=taghdr><h2>${tag.label}</h2><span class=freq>${tag.n} (${tag.pct}%)</span></div>`;
  h+=`<div class=legend>arrows: <i class=lg>best move</i> · <i class=lr>played move</i></div>`;
  h+=`<div class=grid>${tag.ex.map(bcell).join('')}</div>`;
  document.getElementById('main').innerHTML=h;window.scrollTo(0,0);}
function home(){let h=`<h1>Mistake Tag Atlas</h1><div class=sub>Every tag the rule-based tagger produced across ${DATA.n.toLocaleString()} blunder positions. Click a tag to see example boards — green is the best move, red is what was played. Judge each tag for CRISPNESS: do the boards all show the same kind of mistake?</div>`;
  DATA.cats.forEach((c,ci)=>{h+=`<div class=taghdr><h2>${c.name}</h2><span class=freq>${c.tags.length} tags</span></div><div style="columns:3;gap:18px">`;
    c.tags.forEach((t,ti)=>{h+=`<div style="break-inside:avoid;padding:3px 0"><a href="#" onclick="show(${ci},${ti});return false" style="color:#9cf;text-decoration:none">${t.label}</a> <span style="color:#787e8a">${t.n} (${t.pct}%)</span></div>`;});
    h+='</div>';});
  document.getElementById('main').innerHTML=h;}
stat();sidebar();home();
</script></body></html>"""

out_html = HTML.replace("__DATA__", json.dumps(PAGE))
open(a.out, "w").write(out_html)
print(f"wrote {a.out} | {N} positions, {sum(len(c['tags']) for c in PAGE['cats'])} tags, {len(PAGE['cats'])} categories")
print("\nTag counts by category:")
for c in PAGE["cats"]:
    print(f"  {c['name']}: {len(c['tags'])} tags")
