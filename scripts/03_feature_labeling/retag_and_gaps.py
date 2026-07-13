#!/usr/bin/env python3
"""Re-tag the 60k analyzed positions with the FULL Stockfish lines (pv_uci + refutation_uci +
eval_after from sf_batch_60k), aggregate per SAE feature, and find the DELIVERABLE:

  SAE 'good' features (Opus verdict) where the FULL-strength tagger STILL fires no explain tag,
  but Opus named a concept  ==>  MISSING TAGGER DETECTORS.

This is the point of the SAE-as-discovery-tool: the SAE clusters positions by a mistake concept;
if the tagger can't name that cluster even with full lines, that concept is a gap in the detector set.

Steps:
 1. load SF lines (keyed fen|blunder) + the Opus feature labels + tagger corpus.
 2. run full tagger per position (win_drop gate uses real eval_before/after; motifs use pv+refutation).
 3. encode the 60k through the SAE; per feature take top-N firing → aggregate full-tagger labels (vote).
 4. compare to the Opus verdict/label: report features good@Opus with empty/weak full-tagger vote.
"""
import argparse, json, os, sys, torch
from collections import Counter
sys.path.insert(0, "/home/ec2-user/SageMaker")
sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
import chess
from mistake import Mistake
from tagger import tag_mistake_full, categorize, family_of
from train_jr_canonical import JumpReLUSAEAuxK, load_data

BASE = "/home/ec2-user/SageMaker"


def full_tags(sf, classification=None):
    """Run the full tagger on one position given the SF-lines record."""
    fen = sf["key"].split("|")[0]; blunder = sf["key"].split("|")[1]
    b = chess.Board(fen)
    pv = sf.get("pv_uci") or []
    best = pv[0] if pv else sf.get("best_uci", "")
    # SAN lines
    def ucis_to_san(board, ucis):
        out=[]; bb=board.copy()
        for u in ucis:
            try:
                mv=chess.Move.from_uci(u)
                if mv not in bb.legal_moves: break
                out.append(bb.san(mv)); bb.push(mv)
            except Exception: break
        return out
    best_line_san = ucis_to_san(b, pv)
    after = b.copy()
    try: after.push(chess.Move.from_uci(blunder))
    except Exception: return []
    refut_san = ucis_to_san(after, sf.get("refutation_uci") or [])
    eb = sf.get("eval_before"); ea = sf.get("eval_after")
    m = Mistake(fen_before=fen, played_uci=blunder, best_uci=best,
                best_line_san=best_line_san, refutation_san=refut_san,
                eval_before=eb, eval_after=ea, cp_loss=abs((eb or 0)-(ea or 0)),
                mover=b.turn, played_san=sf.get("san",""), best_san=sf.get("bestMoveSan",""))
    try:
        return [t for t in tag_mistake_full(m, with_maia=False)["tags"] if t["direction"]!="info"]
    except Exception:
        return []


def _tag_one(kv):
    """Module-level worker (picklable for Pool): (key, sf_record) -> (key, [labels])."""
    k, r = kv
    return k, [t["label"] for t in full_tags(r)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sf", default=f"{BASE}/jr_canon_out/sf_lines_60k.jsonl")
    ap.add_argument("--weights", default=f"{BASE}/jr_canon_out/jr512_k8_final.pt")
    ap.add_argument("--opus_labels", default=f"{BASE}/jr_canon_out/labels_decile_jr512.json")
    ap.add_argument("--cache", default=f"{BASE}/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--out", default=f"{BASE}/jr_canon_out/retag_full.json")
    args=ap.parse_args()

    sf={r["key"]:r for r in (json.loads(l) for l in open(args.sf) if l.strip()) if "key" in r}
    print(f"SF lines: {len(sf)}", flush=True)
    # precompute full tags per position — PARALLEL (the heavy step: full tagger over 60k). Cache to
    # disk so stage-2 (encode+aggregate) can be re-run without re-tagging.
    ptf=args.out.replace(".json","_postags.json")
    if os.path.exists(ptf):
        pos_tags=json.load(open(ptf)); print(f"loaded cached pos_tags: {len(pos_tags)}",flush=True)
    else:
        from multiprocessing import Pool
        items=list(sf.items())
        with Pool(40) as pool:
            pos_tags=dict(pool.map(_tag_one, items, chunksize=200))
        json.dump(pos_tags, open(ptf,"w"))
        print(f"tagged {len(pos_tags)} positions (parallel) -> {ptf}",flush=True)
    ntagged=sum(1 for v in pos_tags.values() if v)
    print(f"positions with >=1 full-tagger explain tag: {ntagged}/{len(sf)} ({100*ntagged//len(sf)}%)", flush=True)

    # encode the analyzed positions
    data,_,_=load_data(args.cache)
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); meta=cache["metadata"]
    all_keys=[f"{m['fen']}|{m['blunder_uci']}" for m in meta]
    cov=[i for i,k in enumerate(all_keys) if k in sf]
    keys=[all_keys[i] for i in cov]
    ck=torch.load(args.weights,map_location="cpu",weights_only=False); c=ck["config"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=JumpReLUSAEAuxK(c["emb_size"],c["dict_size"],target_l0=c.get("target_l0",8),
        l0_alpha=c.get("l0_alpha",4),bandwidth=c["bandwidth"],init_threshold=c["init_threshold"]).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    D=c["dict_size"]; n=len(keys); cov_t=torch.tensor(cov)
    acts=torch.zeros(D,n)
    with torch.no_grad():
        for i in range(0,len(cov),16384):
            sel=cov_t[i:i+16384]; b=data[sel].to(dev)
            z=(b-sae.b_dec)@sae.W_enc+sae.b_enc; a=(z*(z>sae.threshold)).cpu()
            acts[:,i:i+a.shape[0]]=a.T
    print(f"encoded {n} positions", flush=True)

    opus={r["feature_idx"]:r for r in json.load(open(args.opus_labels))["labels"]}
    out={}
    for f in range(D):
        col=acts[f]; nz=int((col>0).sum())
        if nz==0: continue
        top=torch.topk(col,min(args.topn,nz)).indices.tolist()
        votes=Counter()          # raw piece-specific labels (what the product shows as chips)
        fam_votes=Counter()      # rolled up to concept parents (Missed Free Material, Hung Material, Fork, ...)
        covered=0
        for idx in top:
            tags=pos_tags.get(keys[idx],[])
            if tags: covered+=1
            # board for this position — lets family_of activate the position-gated Pawn Endgame
            # Technique family (fragments only roll up in a K+P ending, #50).
            try:
                pos_board=chess.Board(keys[idx].split("|")[0])
            except Exception:
                pos_board=None
            # per-position: a label and its family each count once (dedupe within the position so a
            # single move firing Missed Free Rook + Missed Free Pawn contributes ONE to the family).
            fams_here=set()
            for t in tags:
                votes[t]+=1
                fams_here.add(family_of(t, pos_board))
            for fam in fams_here: fam_votes[fam]+=1
        top_lab=votes.most_common(1)[0] if votes else (None,0)
        top_fam=fam_votes.most_common(1)[0] if fam_votes else (None,0)
        out[f]={"tagger_top":top_lab[0],"tagger_conf":round(top_lab[1]/len(top),3),
                "tagger_covered_frac":round(covered/len(top),3),
                "tagger_votes":dict(votes.most_common(5)),
                # family rollup: the honest "what concept does this feature encode" signal —
                # piece-specific variants no longer fragment the dominant concept below the top-5.
                "tagger_top_family":top_fam[0],"tagger_family_conf":round(top_fam[1]/len(top),3),
                "tagger_family_votes":dict(fam_votes.most_common(6)),
                "opus_verdict":opus.get(f,{}).get("verdict"),
                "opus_label":opus.get(f,{}).get("good_label")}
    json.dump(out, open(args.out,"w"), indent=1)

    # DELIVERABLE: good@Opus, tagger fires nothing / very weak
    good=[f for f in out if out[f]["opus_verdict"]=="good"]
    gap=[f for f in good if out[f]["tagger_conf"]<0.15 or out[f]["tagger_top"] is None]
    print(f"\n=== {len(good)} good features; {len(gap)} where FULL tagger still finds ~nothing (MISSING DETECTORS) ===",flush=True)
    for f in gap[:30]:
        print(f"  f{f:>3}  OPUS: {out[f]['opus_label']:<34} tagger_top={out[f]['tagger_top']} (conf {out[f]['tagger_conf']}, cov {out[f]['tagger_covered_frac']})",flush=True)
    # cluster the missing concepts by opus label word
    from collections import Counter as C2
    import re
    words=C2()
    for f in gap:
        for w in re.findall(r'[A-Za-z]+', out[f]['opus_label'] or ''):
            if w.lower() not in {'missed','the','a','of','to','left','piece'}: words[w.lower()]+=1
    print(f"\nmissing-concept word freq: {dict(words.most_common(15))}",flush=True)


if __name__=="__main__": main()
