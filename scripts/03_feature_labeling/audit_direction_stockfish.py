#!/usr/bin/env python3
"""Audit a feature's direction (self-inflicted HANG vs OMISSION/MISS) with Stockfish ground truth.

WHY (and a lesson recorded the hard way):
  SEE's `blunder_hangs_own_pct` is NOT a reliable hang-vs-miss arbiter (it's single-ply). To check
  whether a feature is mislabeled in direction, go to the boards with a real engine.

  A FAILED first attempt used `drop = eval_before - eval_after_played` vs `gap = eval_bestline -
  eval_after_played` and called it a HANG when drop≈gap. That is meaningless: eval_before already
  assumes best play, so eval_before == eval_bestline and drop==gap ALWAYS — for hangs and misses
  alike. It produced confident wrong verdicts (called f19 and f745 hangs; both were misreads).

  The signal that ACTUALLY distinguishes the two, read straight off the boards:
    - OMISSION ("Missed X"): the player's PLAYED move is quiet / non-winning, while Stockfish's BEST
      move is a CAPTURE the player declined. (f19: played quiet queen moves; SF best = QxQueen on
      9/10 → missed queen trade.)
    - SELF-INFLICTED ("Hangs X"): the recurring motif is the player's OWN piece going en prise;
      best moves are about rescuing/avoiding rather than grabbing a free enemy piece.
  So compare what the PLAYED move captures vs what the BEST move captures, per board, and tally.

Run locally (needs stockfish on PATH + python-chess). Inputs: /tmp/audit_boards.json
({"f<id>":[[fen,uci],...]}) and optional /tmp/audit_opus.json for the analysis text.
"""
import json, chess, chess.engine, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--boards", default="/tmp/audit_boards.json")
ap.add_argument("--opus", default="/tmp/audit_opus.json")
ap.add_argument("--sf", default="/opt/homebrew/bin/stockfish")
ap.add_argument("--depth", type=int, default=18)
ap.add_argument("--out", default="/tmp/audit_sf_results.json")
a = ap.parse_args()

boards = json.load(open(a.boards))
try: op = json.load(open(a.opus))
except Exception: op = {}
eng = chess.engine.SimpleEngine.popen_uci(a.sf)
LIMIT = chess.engine.Limit(depth=a.depth)


def cp(score, pov):
    s = score.pov(pov)
    if s.is_mate(): return 10000 if s.mate() > 0 else -10000
    return s.score()


def captured_piece(b, mv):
    """Name of the piece a move captures, or None for a quiet move."""
    if b.is_en_passant(mv): return "pawn"
    pc = b.piece_at(mv.to_square)
    return chess.piece_name(pc.piece_type) if (b.is_capture(mv) and pc) else None


def analyze(fen, uci):
    b = chess.Board(fen); pov = b.turn
    info = eng.analyse(b, LIMIT, multipv=1)
    best = info[0]["pv"][0]; eval_before = cp(info[0]["score"], pov)
    mv = chess.Move.from_uci(uci)
    b2 = b.copy(); b2.push(mv)
    eval_after = cp(eng.analyse(b2, LIMIT)["score"], pov)
    return dict(pov="W" if pov else "B", played=b.san(mv), best=b.san(best),
                played_captures=captured_piece(b, mv), best_captures=captured_piece(b, best),
                eval_before=eval_before, eval_after=eval_after,
                drop=eval_before - eval_after)


def verdict(r):
    """MISS if the best move captures something the played move didn't grab (player declined a
    win). HANG if the played move itself loses eval and isn't declining an enemy capture."""
    if r.get("best_captures") and not r.get("played_captures"):
        return f"MISS — declined {r['best_captures']} capture ({r['best']})"
    if r["drop"] >= 200:
        return "HANG — own move loses eval, no enemy capture declined"
    return "~equal / minor"


out = {}
for f, blist in boards.items():
    if f == "keys": continue
    rows = []
    for fen, uci in blist:
        try:
            r = analyze(fen, uci); r["fen"] = fen; r["uci"] = uci
            oa = op.get(fen + "|" + uci, {}).get("analysis") if isinstance(op.get(fen + "|" + uci), dict) else {}
            r["motif"] = oa.get("tactical_motif", "") if isinstance(oa, dict) else ""
            r["verdict"] = verdict(r)
            rows.append(r)
        except Exception as ex:
            rows.append({"fen": fen, "uci": uci, "err": str(ex)[:60]})
    out[f] = rows
eng.quit()
json.dump(out, open(a.out, "w"), indent=1)

for f, rows in out.items():
    miss = sum(1 for r in rows if "err" not in r and r["verdict"].startswith("MISS"))
    hang = sum(1 for r in rows if "err" not in r and r["verdict"].startswith("HANG"))
    print(f"\n=== {f}: MISS(omission) {miss}/{len(rows)} · HANG(self-inflicted) {hang}/{len(rows)} ===")
    for r in rows:
        if "err" in r: print("  err", r["err"]); continue
        print(f"  {r['pov']} played {r['played']:9s}(cap {str(r['played_captures']):6s}) | "
              f"SF best {r['best']:9s}(cap {str(r['best_captures']):6s}) | drop {r['drop']:>5} | {r['verdict']}")
