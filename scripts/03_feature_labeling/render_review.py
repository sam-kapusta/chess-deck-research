#!/usr/bin/env python3
"""Interactive d64_k1 feature review page: collapsible per-feature cards, each expanding to 10 top +
10 median boards (red=played, green=best arrows) PLUS the formatted SEE stats. Features grouped by
review provenance (confirmed / reconfirm / predicate-unseen / structural / sequence). Expand/collapse
all. Click a board for chess.com.

  python3 render_review.py --out output/atlas/d64_review.html
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--boards", default="output/all_feat_boards_d64_k1.json")
ap.add_argument("--labels", default="output/relabel_v9_d64_k1.json")
ap.add_argument("--preds", default="output/predicates_d64_k1.json")
ap.add_argument("--stats", default="output/see_stats_d64_k1.json")
ap.add_argument("--profiles", default="output/peak_median_profiles_d64_k1.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

A = json.load(open(a.boards)); L = json.load(open(a.labels))
P = json.load(open(a.preds)); ST = json.load(open(a.stats))
PROF = json.load(open(a.profiles))
def S(f): return ST.get("f" + f) or ST.get(f) or {}
def fr(f): return S(f).get("fire_rate", 0) * 100


def tier(fid):
    rv = L[fid].get("review")
    if rv == "confirmed": return 0
    if rv == "reconfirm": return 1
    if rv == "predicate_unseen": return 2
    p = P[fid]["preds"]; g = p.get; chip = L[fid]["chip"].lower()
    if g("best_check", 0) >= .8 or g("best_attacks_queen", 0) >= .7 or (g("best_capture", 0) >= .85 and g("played_capture", 0) < .4) or g("played_attacks_queen", 0) >= .8 or g("played_capture", 0) >= .85:
        return 3
    if any(w in chip for w in ["zwischen", "discover", "deflect", "overload", "decoy", "interpos", "removal", "trap", "fork", "simplif", "zugzwang", "tempo", "breakthrough"]):
        return 5
    return 4


TN = {0: "Confirmed by you", 1: "Discussed - reconfirm", 2: "Predicate-applied - NOT seen by you",
      3: "Predicate-confident - quick confirm", 4: "Structural - needs your read", 5: "Sequence-concept - needs your read"}
TC = {0: "#2f7d5e", 1: "#4a7da0", 2: "#c43d4f", 3: "#9a6b2f", 4: "#c0492f", 5: "#7a5a8a"}


def topd(d, k=4):
    if not isinstance(d, dict) or not d: return "—"
    return ", ".join(f"{kk} {vv*100:.0f}%" for kk, vv in sorted(d.items(), key=lambda x: -x[1])[:k])


def stat_html(f):
    s = S(f)
    def pc(k): return f"{s.get(k,0)*100:.0f}%"
    nn = s.get("n", 0)
    return (
        "<table class=st>"
        f"<tr><td>fire</td><td>{fr(f):.2f}%</td><td>stats over</td><td>n={nn} high-act boards (≥0.7·max) · phase {topd(s.get('phase_pct',{}))}</td></tr>"
        f"<tr><td class=h>PLAYED</td><td colspan=3>piece {topd(s.get('moved_piece_pct',{}))} · capture {pc('played_capture_pct')} · check {pc('played_is_check_pct')} · hung-own {pc('blunder_hangs_own_pct')} · outcome {topd(s.get('material_kind_pct',{}))}</td></tr>"
        f"<tr><td class=h>BEST</td><td colspan=3>capture {pc('best_is_capture_pct')} · check {pc('best_is_check_pct')} · wins-mat {pc('best_wins_material_pct')} · piece {topd(s.get('best_piece_pct',{}))} · takes {topd(s.get('best_captured_piece_pct',{}))}</td></tr>"
        f"<tr><td>eval swing</td><td>{s.get('eval_drop_median',0):.0f}cp</td><td>trajectory</td><td>{topd({k:v for k,v in (s.get('trajectory_pct',{}) or {}).items() if k!='?->?'})}</td></tr>"
        "</table>")


feats = sorted([f for f in L if "error" not in L[f]], key=lambda f: (tier(f), -fr(f)))
DATA = []
for fid in feats:
    v = L[fid]
    bands = {}
    for b in ("peak", "median"):
        src = PROF.get(fid, {}).get(b, [])[:10]
        bands[b] = [{"fen": e["fen"], "u": e["uci"], "b": e.get("best", "")} for e in src]
    DATA.append({"id": fid, "chip": v["chip"], "cons": v.get("consistency", 0), "fire": round(fr(fid), 2),
                 "tier": tier(fid), "stat": stat_html(fid), "peak": bands["peak"], "median": bands["median"]})

HTML = r"""<!DOCTYPE html><html><head><meta charset=UTF-8><title>d64_k1 review</title><style>
body{background:#faf8f4;font-family:system-ui,sans-serif;margin:0;padding:18px 26px;color:#1b1b22}
h1{font-family:Georgia,serif;margin:0 0 4px}
#bar{position:sticky;top:0;background:#faf8f4;padding:8px 0;border-bottom:1px solid #e8e3d9;z-index:5;font-size:13px}
#bar button{font:inherit;background:#fff;border:1px solid #d8d2c4;border-radius:6px;padding:5px 11px;cursor:pointer;margin-right:6px}
.leg{font-family:monospace;font-size:11px;color:#888;margin-left:6px}.r{color:#c0392b;font-weight:600}.g{color:#1f8a4c;font-weight:600}
.tierhd{font-family:Georgia,serif;font-size:18px;margin:22px 0 8px;padding-bottom:4px;border-bottom:2px solid}
.f{background:#fff;border:1px solid #e8e3d9;border-left:5px solid;border-radius:9px;margin-bottom:9px;overflow:hidden}
.fh{padding:11px 15px;cursor:pointer;display:flex;align-items:center;gap:12px;user-select:none}
.fh:hover{background:#faf7f1}
.fh .nm{font-family:Georgia,serif;font-size:15.5px;font-weight:600;flex:1}
.fh .mt{font-family:monospace;font-size:11px;color:#999}
.fh .ar{font-family:monospace;color:#bbb;font-size:13px}
.body{display:none;padding:0 15px 14px}
.f.open .body{display:block}
.st{font-family:monospace;font-size:11px;border-collapse:collapse;margin:4px 0 12px;width:100%}
.st td{padding:2px 8px 2px 0;vertical-align:top;color:#54555f}.st td.h{color:#9a6b2f;font-weight:600}
.bl{font-size:9px;font-family:monospace;color:#aaa;text-transform:uppercase;letter-spacing:.05em;margin:6px 0 3px}
.bd{display:inline-block;width:96px;margin:2px;vertical-align:top}.cap{font-size:8px;color:#999;font-family:monospace;text-align:center}
</style></head><body>
<h1>d64_k1 — feature review</h1>
<div id=bar><button onclick="all(1)">Expand all</button><button onclick="all(0)">Collapse all</button>
<span class=leg><span class=r>▶ red = played</span> &nbsp; <span class=g>▶ green = best</span> · click a header to expand · click a board → chess.com</span></div>
<div id=app></div>
<script>
const DATA=__DATA__, TN=__TN__, TC=__TC__;
const P={k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟',K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙'};const SQ=96/8;let _m=0;
function bd(o){const rows=o.fen.split(' ')[0].split('/');const flip=o.fen.split(' ')[1]==='b';let c='';
 for(let dr=0;dr<8;dr++)for(let df=0;df<8;df++)c+=`<rect x=${df*SQ} y=${dr*SQ} width=${SQ} height=${SQ} fill="${(dr+df)%2?'#b88a5a':'#efe2cf'}"/>`;
 for(let r=0;r<8;r++){let f=0;for(const ch of rows[r]){if(/\d/.test(ch)){f+=+ch;continue}const dr=flip?7-r:r,df=flip?7-f:f;
  c+=`<text x=${df*SQ+SQ/2} y=${dr*SQ+SQ/2+1} font-size=${SQ*0.8} text-anchor=middle dominant-baseline=central>${P[ch]||''}</text>`;f++}}
 function ar(u,col,id){if(!u||u.length<4)return'';function xy(s){let f=s.charCodeAt(0)-97,r=8-(+s[1]);if(flip){f=7-f;r=7-r}return[f*SQ+SQ/2,r*SQ+SQ/2]}
  let[x1,y1]=xy(u.slice(0,2)),[x2,y2]=xy(u.slice(2,4));const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L;
  x1+=ux*SQ*.3;y1+=uy*SQ*.3;x2-=ux*SQ*.34;y2-=uy*SQ*.34;
  return `<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke="${col}" stroke-width=5 stroke-linecap=round opacity=.9 marker-end="url(#${id})"/>`}
 const i=_m++,rc='r'+i,gc='g'+i;
 const d=`<defs><marker id=${gc} markerWidth=3 markerHeight=3 refX=2 refY=1.5 orient=auto><path d="M0,0 L3,1.5 L0,3 z" fill=#1f8a4c /></marker><marker id=${rc} markerWidth=3 markerHeight=3 refX=2 refY=1.5 orient=auto><path d="M0,0 L3,1.5 L0,3 z" fill=#c0392b /></marker></defs>`;
 return `<svg width=96 height=96 viewBox="0 0 96 96" style="border-radius:4px">${d}${c}${ar(o.b,'#1f8a4c',gc)}${ar(o.u,'#c0392b',rc)}</svg>`}
function cells(arr){return arr.map(o=>{const url='https://www.chess.com/analysis?fen='+encodeURIComponent(o.fen);
 return `<div class=bd><a href="${url}" target=_blank>${bd(o)}</a><div class=cap><span class=r>${o.u}</span>/<span class=g>${o.b||''}</span></div></div>`}).join('')}
function render(){let h='',last=-1;
 DATA.forEach((f,idx)=>{if(f.tier!==last){h+=`<div class=tierhd style="color:${TC[f.tier]};border-color:${TC[f.tier]}">${TN[f.tier]}</div>`;last=f.tier}
  h+=`<div class=f id=f${idx} style="border-left-color:${TC[f.tier]}"><div class=fh onclick="document.getElementById('f${idx}').classList.toggle('open')">
   <span class=ar>▸</span><span class=nm>f${f.id} — ${f.chip}</span><span class=mt>fires ${f.fire}% · cons ${f.cons}</span></div>
   <div class=body>${f.stat}<div class=bl>Top 10 (peak)</div><div>${cells(f.peak)}</div><div class=bl>Median 10 (typical)</div><div>${cells(f.median)}</div></div></div>`});
 document.getElementById('app').innerHTML=h}
function all(open){document.querySelectorAll('.f').forEach(e=>e.classList.toggle('open',!!open))}
render();
</script></body></html>"""
import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
out = (HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":")))
           .replace("__TN__", json.dumps(TN)).replace("__TC__", json.dumps(TC)))
open(a.out, "w").write(out)
print(f"wrote {a.out} — {len(DATA)} features, {os.path.getsize(a.out)/1e6:.1f}MB")
