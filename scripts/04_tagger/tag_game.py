#!/usr/bin/env python3
"""Tag a single analyzed game with the rule-based mistake tagger.

Consumes the JSON that `analyze_cli.py` (chess-deck-code, the /analyze-game skill) produces — the
shape with a `deep` list of mistakes (fen, uci, best_san, cp_loss, top_lines, refutation). Runs the
tagger (motifs + predicates) on each mistake and emits:
  - <out>.json  : the game with per-mistake `tags` + `categories` added
  - <out>.html  : a per-game review page (board per mistake, best/played arrows, tags listed)

This keeps the validated tagger in the research package; the product CLI stays decoupled (the skill
orchestrates both steps, neither package imports the other — respects the one-way handoff).

  python3 tag_game.py /tmp/game_analysis.json --elo 1800 [--out output/game_tagged]
"""
import argparse, json, os, sys, html, chess
sys.path.insert(0, os.path.dirname(__file__))
from mistake import Mistake
from tagger import tag_mistake_full, categorize


def _real_best(e, b):
    """The deep best move = top_lines[0].moves[0], NOT the shallow `best_san` field. The shallow
    best_san can be STALE (equal to the played move) while the deep MultiPV has the true best — e.g.
    40.Bd2: best_san='Bd2' but top_lines[0]='cxb4'. Always trust the deep PV. Returns (uci, san)."""
    tl = e.get("top_lines") or []
    if tl and tl[0].get("moves"):
        san0 = tl[0]["moves"][0]
        try:
            return b.parse_san(san0).uci(), san0
        except Exception:
            pass
    # fallback to the shallow field only if no deep line
    bs = e.get("best_san", "")
    try:
        return b.parse_san(bs).uci(), bs
    except Exception:
        return "", bs


def deep_entry_to_mistake(e, player_elo, oppo_elo):
    """Map one analyze_cli `deep` entry -> Mistake. analyze_cli stores `refutation` (singular dict)
    and `cp_before/after`; we normalize to the Mistake contract. Best move = deep PV's first move."""
    fen = e["fen"]; uci = e["uci"]
    b = chess.Board(fen)
    best_uci, best_san = _real_best(e, b)
    best_line = (e.get("top_lines") or [{}])[0].get("moves", [])
    refut = (e.get("refutation") or {}).get("moves", [])
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=best_uci,
        best_line_san=best_line, refutation_san=refut,
        eval_before=e.get("cp_before"), eval_after=e.get("cp_after"),
        cp_loss=int(e.get("cp_loss", 0) or 0), mover=b.turn,
        player_elo=player_elo, oppo_elo=oppo_elo,
        played_san=e.get("san", ""), best_san=best_san,
    )


def _eval_cp(s):
    """Deep eval string -> mover-relative cp. '#'/'mate' -> large signed sentinel."""
    if s is None:
        return None
    s = str(s)
    if "#" in s or "mate" in s.lower():
        sign = -1 if "-" in s else 1
        return sign * 100000
    try:
        return int(float(s))
    except ValueError:
        return None


def _only_move(e):
    """True if there is exactly one good move: a big eval gap between the best line and the 2nd-best.
    Uses deep MultiPV evals (white-POV). Threshold 150cp = clearly only one move holds the position."""
    tl = e.get("top_lines") or []
    if len(tl) < 2:
        return False
    e0, e1 = _eval_cp(tl[0].get("eval")), _eval_cp(tl[1].get("eval"))
    if e0 is None or e1 is None:
        return False
    return abs(e0 - e1) >= 150


def _played_is_best(e):
    """Deep analysis says the played move IS the best move (the deep PV's first move == played uci).
    Uses the deep PV, not the stale best_san field. These are shallow-scan false positives."""
    try:
        b = chess.Board(e["fen"])
        best_uci, _ = _real_best(e, b)
        return best_uci == e["uci"]
    except Exception:
        return False


def tag_game(analysis, player_elo=1800, oppo_elo=1800, with_maia=False, player_side=None):
    """Add tags to each deep mistake. If player_side ('white'/'black') is given, only that side's
    mistakes are tagged with player_elo (the coached player); the opponent's keep oppo_elo.

    Two deep-analysis-aware adjustments (the shallow scan flags moves the deep pass later clears):
      - played == best  -> NOT a mistake. Tag only 'Best Move (deep)' + phase context, skip motifs.
      - only one good move (>=150cp gap to 2nd best) -> add 'Only Move'."""
    tagged = []
    for e in analysis.get("deep", []):
        side = e.get("side")
        pe = player_elo if (player_side is None or side == player_side) else oppo_elo
        m = deep_entry_to_mistake(e, pe, oppo_elo)
        if _played_is_best(e):
            # deep analysis revised the verdict: the played move was best. Don't tag a mistake.
            phase_state = [t for t in tag_mistake_full(m, with_maia=False)["tags"] if t["layer"] == "position" and t["direction"] == "info"]
            tags = [{"label": "Best Move (deep analysis)", "direction": "info", "evidence": "shallow flagged, deep cleared", "layer": "info"}] + phase_state
            tagged.append({**e, "tags": tags, "categories": [], "maia": {}})
            continue
        res = tag_mistake_full(m, with_maia=with_maia)
        tags = res["tags"]
        if _only_move(e):
            tags = tags + [{"label": "Only Move", "direction": "info",
                            "evidence": "best move >=150cp better than 2nd best", "layer": "position"}]
        tagged.append({**e, "tags": tags, "categories": res["categories"], "maia": res["maia"]})
    return {**analysis, "deep": tagged}


# ---------- per-game HTML review (reuses the atlas board renderer) ----------
_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Game Tags — __TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#13151a;color:#e8e6e1}
#top{position:sticky;top:0;background:#1a1d24;border-bottom:1px solid #2a2e38;padding:14px 22px;z-index:10}
.logo{font-size:19px;font-weight:700}.logo span{color:#d99}#stat{color:#8a909c;font-size:13px;margin-top:3px}
#main{padding:20px 26px;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
.card{background:#1a1d24;border:1px solid #262a34;border-radius:10px;padding:12px}
.mv{font-size:15px;font-weight:700;margin-bottom:2px}.mv .cp{color:#d99;font-weight:600;font-size:13px}
.sub{color:#9aa0ac;font-size:12px;margin-bottom:8px}
.bd a{text-decoration:none}
.tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:5px}
.tag{font-size:11px;padding:3px 8px;border-radius:11px;background:#2a3140;color:#cfe}
.tag.tactic{background:#2d2540;color:#d9c3ff}.tag.position{background:#243140;color:#bcd}
.tag.info{background:#222;color:#888}
.legend{grid-column:1/-1;font-size:12px;color:#8a909c;margin:-6px 0 4px}
.legend i{font-style:normal;font-weight:700}.lg{color:#1f8a4c}.lr{color:#c0392b}
.gem{margin-top:10px;background:#0e1014;border:1px solid #262a34;border-radius:6px;padding:8px;position:relative}
.gem pre{margin:0;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:#aeb6c2;white-space:pre-wrap;word-break:break-all}
.gem button{position:absolute;top:6px;right:6px;font-size:10px;padding:2px 7px;border:1px solid #3a4150;border-radius:5px;background:#222630;color:#cfd2d8;cursor:pointer}
.gem button:active{background:#2a3140}
</style></head><body>
<div id=top><div class=logo>Game Tags · <span>__TITLE__</span></div><div id=stat>__STAT__</div></div>
<div id=main>
<div class=legend>arrows: <i class=lg>best move</i> · <i class=lr>played move</i></div>
</div>
<script>
const PIECES={k:'\\u265a',q:'\\u265b',r:'\\u265c',b:'\\u265d',n:'\\u265e',p:'\\u265f',K:'\\u2654',Q:'\\u2655',R:'\\u2656',B:'\\u2657',N:'\\u2658',P:'\\u2659'};
const SQ=148/8;
function sqxy(sq,flip){let f=sq.charCodeAt(0)-97,r=8-(+sq[1]);if(flip){f=7-f;r=7-r}return [f*SQ+SQ/2,r*SQ+SQ/2];}
function board(o,id){
  const rows=o.fen.split(' ')[0].split('/');const wtm=o.fen.split(' ')[1]!=='b';const flip=!wtm;
  let cells='';
  for(let dr=0;dr<8;dr++)for(let df=0;df<8;df++){const dark=(dr+df)%2===1;
    cells+=`<rect x="${df*SQ}" y="${dr*SQ}" width="${SQ}" height="${SQ}" fill="${dark?'#b88a5a':'#efe2cf'}"/>`;}
  for(let r=0;r<8;r++){let f=0;for(const ch of rows[r]){
    if(/\\d/.test(ch)){f+=+ch;continue;}
    const dr=flip?7-r:r, df=flip?7-f:f;
    cells+=`<text x="${df*SQ+SQ/2}" y="${dr*SQ+SQ/2+0.5}" font-size="${SQ*0.82}" text-anchor="middle" dominant-baseline="central">${PIECES[ch]||''}</text>`;f++;}}
  function arrow(uci,col,mid){if(!uci||uci.length<4)return'';const a=uci.slice(0,2),b=uci.slice(2,4);
    let[x1,y1]=sqxy(a,flip),[x2,y2]=sqxy(b,flip);const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len;
    x1+=ux*SQ*0.30;y1+=uy*SQ*0.30;x2-=ux*SQ*0.34;y2-=uy*SQ*0.34;
    return `<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke="${col}" stroke-width=5.5 stroke-linecap=round opacity=.9 marker-end="url(#${mid})"/>`;}
  const mk=(i,c)=>`<marker id=${i} markerWidth=3.2 markerHeight=3.2 refX=2.2 refY=1.6 orient=auto><path d="M0,0 L3.2,1.6 L0,3.2 z" fill="${c}"/></marker>`;
  const defs=`<defs>${mk('r'+id,'#c0392b')}${mk('g'+id,'#1f8a4c')}</defs>`;
  return `<svg width=148 height=148 viewBox="0 0 148 148" style="border-radius:5px;display:block">${defs}${cells}${arrow(o.b,'#1f8a4c','g'+id)}${arrow(o.u,'#c0392b','r'+id)}</svg>`;}
const CARDS=__DATA__;
function gemText(o){
  const tags=o.tags.map(t=>t.label).join(', ');
  return `FEN: ${o.fen}
Side to move: ${o.fen.split(' ')[1]==='w'?'White':'Black'}
Played move: ${o.ps} (${o.u})  [lost ${o.cp}cp]
Best move: ${o.bs} (${o.b})
Best line: ${o.bl||'n/a'}
Refutation of played: ${o.rf||'n/a'}
Our tags: ${tags}`;
}
document.getElementById('main').insertAdjacentHTML('beforeend', CARDS.map((o,i)=>{
  const url='https://www.chess.com/analysis?fen='+encodeURIComponent(o.fen);
  const tags=o.tags.map(t=>`<span class="tag ${t.layer}">${t.label}</span>`).join('');
  const gt=gemText(o).replace(/&/g,'&amp;').replace(/</g,'&lt;');
  return `<div class=card><div class=mv>${o.mv} <span class=cp>-${o.cp}cp</span></div>
    <div class=sub>played ${o.ps} · best ${o.bs}</div>
    <div class=bd><a href="${url}" target=_blank>${board(o,i)}</a></div>
    <div class=tags>${tags}</div>
    <div class=gem><button onclick="navigator.clipboard.writeText(this.nextElementSibling.textContent);this.textContent='copied'">copy</button><pre>${gt}</pre></div></div>`;
}).join(''));
</script></body></html>"""


def build_review_html(tagged, title):
    cards = []
    deep = sorted(tagged.get("deep", []), key=lambda e: e.get("ply", 0))  # game order
    for e in deep:
        b = chess.Board(e["fen"])
        best_uci = ""
        if e.get("best_san"):
            try:
                best_uci = b.parse_san(e["best_san"]).uci()
            except Exception:
                best_uci = ""
        num = e.get("move_num", "")
        dots = "" if e.get("side") == "white" else "..."
        best_line = " ".join((e.get("top_lines") or [{}])[0].get("moves", []))
        refutation = " ".join((e.get("refutation") or {}).get("moves", []))
        cards.append({
            "fen": e["fen"], "u": e["uci"], "b": best_uci,
            "mv": f"{num}{dots} {e.get('san','')}", "ps": e.get("san", ""), "bs": e.get("best_san", ""),
            "cp": e.get("cp_loss", 0), "bl": best_line, "rf": refutation,
            "tags": [{"label": t["label"], "layer": t.get("layer", "info")} for t in e["tags"]],
        })
    n_tags = sum(len(c["tags"]) for c in cards)
    stat = f"{len(cards)} mistakes · {n_tags} tags"
    return (_PAGE.replace("__TITLE__", html.escape(title)).replace("__STAT__", stat)
            .replace("__DATA__", json.dumps(cards)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", help="game_analysis.json from analyze_cli")
    ap.add_argument("--out", default="", help="output base path (default: alongside input, _tagged)")
    ap.add_argument("--elo", type=int, default=1800, help="player Elo for Maia rarity")
    ap.add_argument("--player-side", choices=["white", "black"], default=None,
                    help="only the coached player's side gets player Elo")
    ap.add_argument("--maia", action="store_true", help="include Maia rarity (slow ONNX)")
    a = ap.parse_args()

    analysis = json.load(open(a.analysis))
    tagged = tag_game(analysis, player_elo=a.elo, oppo_elo=a.elo,
                      with_maia=a.maia, player_side=a.player_side)

    base = a.out or os.path.splitext(a.analysis)[0] + "_tagged"
    json.dump(tagged, open(base + ".json", "w"), indent=2)
    title = os.path.basename(analysis.get("meta", {}).get("source", a.analysis))
    open(base + ".html", "w").write(build_review_html(tagged, title))

    # stdout summary
    print(f"Tagged {len(tagged['deep'])} mistakes -> {base}.json / {base}.html\n")
    for e in tagged["deep"]:
        num = e.get("move_num", ""); dots = "" if e.get("side") == "white" else "..."
        labels = [t["label"] for t in e["tags"] if t.get("layer") != "info" or "Blunder" in t["label"]]
        print(f"  {num}{dots} {e.get('san',''):7} (best {e.get('best_san','')}, -{e.get('cp_loss',0)}cp)")
        print(f"      {', '.join(labels)}")


if __name__ == "__main__":
    main()
