#!/usr/bin/env python3
"""Encode the corpus ONCE; for each requested feature, write a full-prose dossier text file
(all opus-covered positions across activation bands) ready to paste into an LLM to find the pattern.

  cd ~/SageMaker && python3 dump_all_dossiers.py --fids 0,1,5,... --per-feat 40 --outdir f_dossiers
"""
import argparse, json, os, chess, torch, numpy as np, torch.nn.functional as F

ap=argparse.ArgumentParser()
ap.add_argument("--weights", default="chess-stage-a/output/maia3_sae/btk_64_k1_nol2.pt")
ap.add_argument("--cache", default="chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
ap.add_argument("--positions", default="all_positions_labeled_opus.json")
ap.add_argument("--fids", required=True)
ap.add_argument("--per-feat", type=int, default=40, help="max opus-covered positions per feature")
ap.add_argument("--outdir", default="f_dossiers")
a=ap.parse_args()
FIDS=[int(x) for x in a.fids.split(",")]
os.makedirs(a.outdir, exist_ok=True)

ck=torch.load(a.weights,map_location="cpu",weights_only=False); sd=ck["state_dict"]; K=int((ck.get("config") or {}).get("k",1))
c=torch.load(a.cache,map_location="cpu",weights_only=False); raw=c["activations"].float(); meta=c["metadata"]
x=(raw-raw.mean(0))/raw.std(0).clamp(min=1e-6); We,be,bd=sd["W_enc"],sd["b_enc"],sd["b_dec"]; N,D=x.shape[0],We.shape[1]
opus=json.load(open(a.positions))

# activation matrix for the requested features
ACT={f:np.zeros(N) for f in FIDS}
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-bd)@We+be); tv,ti=z.topk(K,dim=1); mask=torch.zeros_like(z).scatter_(1,ti,1.0)
    g=(z*mask).numpy()
    for f in FIDS: ACT[f][i:i+8192]=g[:,f]
print("encoded",flush=True)

HEADER="""Sparse-autoencoder chess feature firing on these mistakes by one ~1500 player (many games). Bands: TOP=strongest/most homogeneous (weight heaviest), LOW=weak/likely noise. Find the ONE recurring mistake-pattern, give a 2-5 word coaching label, note any greed-vs-passive sub-split, flag off-pattern noise. Not "blunder"/"lost material" — the specific mechanism. PLAYED=the mistake, BEST=engine move.
======================================================================
"""
def band(rank,n):
    q=rank/max(1,n)
    return "top" if q<.25 else "upper" if q<.5 else "mid" if q<.75 else "low"

for f in FIDS:
    fa=ACT[f]; fired=np.where(fa>0)[0]; fired=fired[np.argsort(-fa[fired])]; nf=len(fired)
    # take positions WITH opus, capped, spread across the ranking
    recs=[]
    for rank,idx in enumerate(fired):
        m=meta[idx]; fen=m["fen"]; uci=m.get("blunder_uci") or m.get("uci",""); best=m.get("best_uci","")
        o=opus.get(fen+"|"+uci)
        if isinstance(o,dict): o=o.get("analysis",o)
        if not isinstance(o,dict): continue
        b=chess.Board(fen)
        try: psan=b.san(chess.Move.from_uci(uci))
        except: psan=uci
        bsan=best
        try: bsan=b.san(chess.Move.from_uci(best))
        except: pass
        recs.append((band(rank,nf), round(float(fa[idx]),1), psan, bsan, o.get("tactical_motif"),
                     o.get("blunder_summary") or "", o.get("best_moves_analysis") or ""))
        if len(recs)>=a.per_feat: break
    lines=[HEADER]
    for i,(bn,act,p,bm,mo,ws,ba) in enumerate(recs,1):
        lines.append(f"{i}.[{bn} act{act}] {p}>{bm} ({mo}): {ws} | BEST: {ba[:200]}")
    open(f"{a.outdir}/f{f}.txt","w").write("\n".join(lines))
    print(f"f{f}: {len(recs)} positions -> {a.outdir}/f{f}.txt", flush=True)
print("DONE",flush=True)
