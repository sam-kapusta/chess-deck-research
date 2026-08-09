"""Position-level depth-16 re-detect for the rapid pull (no PGNs needed — analyze each blunder FEN).
For each pulled blunder: SF d16 on the FEN (best move + eval_before, PV) and on the post-played board
(eval_after, refutation PV). Emits the enrichment-record format run_rating_bands/build_mistake consume.
Multiprocess over positions. Input: proof_3band.json {band: [ {fen, blunder_uci, ...} ]}."""
import os, sys, json, time, io, chess, chess.engine, chess.pgn
from multiprocessing import Pool

SF = "/home/ec2-user/SageMaker/stockfish_compiled"
DEPTH = 16
IN  = sys.argv[1] if len(sys.argv)>1 else "proof_3band.json"
OUT_ENRICH = sys.argv[2] if len(sys.argv)>2 else "proof_3band_enrich.json"
OUT_SWEEP  = sys.argv[3] if len(sys.argv)>3 else "proof_3band_sweep.json"
NPROC = int(sys.argv[4]) if len(sys.argv)>4 else 16

# Engine-line cap (plies) for BOTH the best line and the refutation. Must match TAGGER_LINE_PLIES in
# chess-deck-code (backend/mcp/analysis.py + frontend/src/utils/batchAnalysis.ts) or this corpus cannot
# reproduce production tag output — which is exactly what happened at 6: a production Allowed Sacrifice
# false positive needed a ~10-ply refutation, so a 56,950-position audit of this corpus could not see it.
# Design: chess-deck-code/docs/superpowers/specs/2026-08-09-tagger-line-length-contract-design.md
TAGGER_LINE_PLIES = 8

def line_sans(board, pv):
    out=[]; b=board.copy()
    for mv in pv[:TAGGER_LINE_PLIES]:
        try: out.append(b.san(mv)); b.push(mv)
        except Exception: break
    return " ".join(out)

def analyse_one(args):
    band, fen, uci = args
    try:
        eng = analyse_one.eng
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves: return None
        mover_white = b.turn == chess.WHITE
        info_b = eng.analyse(b, chess.engine.Limit(depth=DEPTH))
        sb = info_b["score"].white(); pv_b = info_b.get("pv", [])
        best_uci = pv_b[0].uci() if pv_b else ""
        best_san = b.san(pv_b[0]) if pv_b else ""
        best_line = line_sans(b, pv_b)
        ab = b.copy(); ab.push(mv)
        info_a = eng.analyse(ab, chess.engine.Limit(depth=DEPTH))
        sa = info_a["score"].white(); pv_a = info_a.get("pv", [])
        refut_line = line_sans(ab, pv_a)
        eb = sb.score(mate_score=10000); ea = sa.score(mate_score=10000)
        cp_loss = max(int((eb-ea) if mover_white else (ea-eb)), 0)
        played_san = b.san(mv)
        rec = {"fen":fen,"uci":uci,"played_san":played_san,"best_san":best_san,"best_uci":best_uci,
               "eval_before":str(sb),"eval_after":str(sa),"cp_loss":cp_loss,
               "side":"White" if mover_white else "Black","phase":None,
               "top_3_best":[{"line":best_line,"eval":str(sb)}],
               "top_3_refutations":[{"line":refut_line,"eval":str(sa)}],
               "position_features":[]}
        row = {"band":band,"game_id":"","outcome":"","cp_loss":cp_loss,"fen":fen,"uci":uci}
        return (f"{fen}|{uci}", rec, row)
    except Exception as e:
        return None

def init():
    analyse_one.eng = chess.engine.SimpleEngine.popen_uci(SF)
    try: analyse_one.eng.configure({"Threads":1,"Hash":64})
    except Exception: pass

def main():
    data = json.load(open(IN))
    tasks=[]
    for band, rows in data.items():
        for r in rows:
            tasks.append((band, r["fen"], r.get("blunder_uci") or r.get("uci")))
    print(f"{len(tasks)} positions to re-detect at d{DEPTH}, {NPROC} procs", flush=True)
    enrich={}; sweep=[]; t0=time.time(); n=0
    with Pool(NPROC, initializer=init) as p:
        for res in p.imap_unordered(analyse_one, tasks, chunksize=8):
            n+=1
            if res:
                k,rec,row=res; enrich[k]=rec; sweep.append(row)
            if n%500==0:
                print(f"  {n}/{len(tasks)} | {len(enrich)} kept | {(time.time()-t0)/60:.1f}min", flush=True)
    json.dump(enrich, open(OUT_ENRICH,"w")); json.dump(sweep, open(OUT_SWEEP,"w"))
    print(f"DONE {len(enrich)} enriched ({(time.time()-t0)/60:.1f}min) -> {OUT_ENRICH}", flush=True)

if __name__=="__main__":
    main()

