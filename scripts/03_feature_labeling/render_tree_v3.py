"""Render the v3 taxonomy as a browsable collapsible HTML tree.
Character group (self-inflicted / omission / phase) -> bucket -> sub-bucket -> features.
Each feature: chip + label, SEE signature, fire rate, top-N boards (played=red, Maia best=green,
clickable to chess.com). Sub-buckets ordered by fire coverage.

Run locally:
  python3 render_tree_v3.py --boards 3 --out output/atlas/taxonomy_v3_d2048_k6.html
"""
import json, argparse, chess, chess.svg
from collections import defaultdict
from urllib.parse import quote as _q

ap = argparse.ArgumentParser()
ap.add_argument("--boards", type=int, default=3)
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
def esc(s): return (str(s) or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def top2(d): return ", ".join(f"{k} {v*100:.0f}%" for k, v in list(d.items())[:3]) if d else "—"

CHAR = {b["id"]: b["char"] for b in buckets}
BNAME = {b["id"]: b["name"] for b in buckets}
BDESC = {b["id"]: b["desc"] for b in buckets}
CHAR_ORDER = ["self-inflicted", "omission", "phase"]
CHAR_LABEL = {"self-inflicted": "SELF-INFLICTED — the move you played loses",
              "omission": "OMISSION — your move was safe but you missed a better one",
              "phase": "ENDGAME — phase-specific technique"}
border = {b["id"]: i for i, b in enumerate(buckets)}

# tree: char -> bucket -> sub -> [fids]
tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f, v in leaf.items():
    bid = v["bucket"]
    if bid == "unassignable": continue
    tree[CHAR.get(bid, "phase")][bid][v["sub"]].append(f)

def board_cell(fen, uci):
    try:
        b = chess.Board(fen); ar = []; lm = None; cap = ""
        try:
            mv = chess.Move.from_uci(uci); lm = mv
            ar.append(chess.svg.Arrow(mv.from_square, mv.to_square, color="#cc2b2b"))
            cap = b.san(mv) + (" ×" + chess.piece_name(b.piece_at(mv.to_square).piece_type)
                               if b.is_capture(mv) and b.piece_at(mv.to_square) else " (quiet)")
        except Exception: cap = uci
        bu = best_map.get(fen + "|" + uci, "")
        if bu and len(bu) >= 4:
            try:
                bm = chess.Move.from_uci(bu)
                ar.append(chess.svg.Arrow(bm.from_square, bm.to_square, color="#2b8a3e"))
                cap += f" · Maia {b.san(bm)}"
            except Exception: pass
        svg = chess.svg.board(b, size=190, arrows=ar, lastmove=lm,
                              orientation=chess.WHITE if b.turn else chess.BLACK)
        url = "https://www.chess.com/analysis?fen=" + _q(fen, safe="")
        return f'<a class=cell href="{url}" target=_blank style=text-decoration:none;color:inherit>{svg}<div class=bc>{esc(cap)} ↗</div></a>'
    except Exception:
        return "<div class=cell>bad</div>"

def sig(f):
    s = S(f)
    return (f"loses-own {s.get('blunder_hangs_own_pct',0)*100:.0f}% · best-wins {s.get('best_wins_material_pct',0)*100:.0f}% · "
            f"material {top2(s.get('material_kind_pct',{}))} · phase {top2(s.get('phase_pct',{}))}")

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#161922;padding:12px 20px;border-bottom:1px solid #2a2f3a;z-index:20}
header h1{margin:0;font-size:16px}header .sub{color:#9aa4b2;font-size:12px;margin-top:3px}
details{margin:0}summary{cursor:pointer;list-style:none;user-select:none}summary::-webkit-details-marker{display:none}
.char>summary{padding:12px 18px;background:#0b0d12;border-bottom:2px solid #2a2f3a;font-size:13px;font-weight:700;letter-spacing:.5px;color:#c9a227;text-transform:uppercase}
.bucket{margin-left:8px}.bucket>summary{padding:10px 18px;background:#1b2030;border-bottom:1px solid #2a2f3a;font-size:15px;font-weight:600;color:#fff}
.bucket>summary:hover{background:#212840}.bucket .cnt{color:#6ea8fe;font-weight:400;font-size:12px}
.bdesc{font-size:11px;color:#8b95a3;font-weight:400;margin-top:2px}
.sub{margin-left:18px}.sub>summary{padding:7px 16px;color:#cdd6e0;font-size:13px;border-bottom:1px solid #1c2027}
.sub>summary:hover{background:#181c26}.sub .cnt{color:#7d8896;font-size:11px}
.feat{margin:6px 0 10px 28px;padding:7px 12px;border-left:2px solid #2a3550;background:#13161e}
.feat .fn{font-size:13px;color:#e6e6e6}.feat .fl{font-size:11px;color:#8b95a3;margin:2px 0}
.feat .mech{font-size:10.5px;color:#aeb6c2;margin:3px 0}.blob{color:#ff8a8a;font-weight:600}
.boards{display:flex;gap:8px;flex-wrap:wrap;margin-top:5px}.cell{width:190px}
.bc{font-size:10px;color:#9aa4b2;margin-top:2px}
"""
parts = [f"<!doctype html><meta charset=utf-8><title>Taxonomy v3 — d2048_k6</title><style>{CSS}</style>"]
nfeat = sum(len(fl) for ch in tree.values() for bk in ch.values() for fl in bk.values())
nun = sum(1 for v in leaf.values() if v["bucket"] == "unassignable")
parts.append(f"<header><h1>Mistake taxonomy v3 — d2048_k6 · {len(buckets)} buckets · {nfeat} features · {nun} unassignable</h1>"
             f"<div class=sub>3 character groups → bucket → sub-bucket → feature · "
             f"<span style=color:#ff8a8a>red = played (blunder)</span> "
             f"<span style=color:#7ee2a0>green = Maia top move</span> · "
             f"<span class=blob>red fire% = blob (high-frequency, low-specificity)</span> · boards → chess.com ↗</div></header>")

def cover(fids): return sum(fr(f) for f in fids)
for ch in CHAR_ORDER:
    if ch not in tree: continue
    chbuckets = tree[ch]
    nfeat_ch = sum(len(fl) for bk in chbuckets.values() for fl in bk.values())
    parts.append(f"<details class=char open><summary>{CHAR_LABEL[ch]} · {nfeat_ch} features</summary>")
    for bid in sorted(chbuckets, key=lambda k: border.get(k, 99)):
        subs = chbuckets[bid]
        tot = sum(len(v) for v in subs.values())
        allf = [f for v in subs.values() for f in v]
        parts.append(f"<details class=bucket><summary>{esc(BNAME[bid])} "
                     f"<span class=cnt>{tot} features · {cover(allf)*100:.0f}% fire</span>"
                     f"<div class=bdesc>{esc(BDESC[bid])}</div></summary>")
        for sub in sorted(subs, key=lambda s: -cover(subs[s])):
            fids = sorted(subs[sub], key=lambda f: -fr(f))
            parts.append(f"<details class=sub><summary>{esc(sub)} <span class=cnt>{len(fids)} · {cover(fids)*100:.1f}% fire</span></summary>")
            for f in fids:
                v = lab[f]; s = S(f); blob = fr(f) >= 0.01
                fire_html = f"<span class={'blob' if blob else ''}>{fr(f)*100:.1f}%</span>"
                parts.append(f"<div class=feat><div class=fn>f{f} — {esc(v['chip'])} "
                             f"<span style=color:#6ea8fe>fires {fire_html} · cons {v.get('consistency','?')}</span></div>"
                             f"<div class=fl>{esc(v.get('label',''))}</div>"
                             f"<div class=mech>{sig(f)}</div><div class=boards>")
                for ex in prof.get(f, {}).get("examples", [])[:a.boards]:
                    parts.append(board_cell(ex["fen"], ex["uci"]))
                parts.append("</div></div>")
            parts.append("</details>")
        parts.append("</details>")
    parts.append("</details>")
import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(parts))
print(f"wrote {a.out} — {len(buckets)} buckets, {nfeat} features")
