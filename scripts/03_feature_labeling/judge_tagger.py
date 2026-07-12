#!/usr/bin/env python3
"""Board-grounded three-way judge: for each SAE feature the RULE TAGGER labeled confidently, show a
judge the ACTUAL top-firing positions (FEN + played move + Stockfish best line + refutation + the
per-position analysis) and BOTH candidate labels, and have it rule:
  - tagger_accurate : the tagger label correctly + fully names what these positions share
  - tagger_shallow  : tagger label is right but GENERIC; a deeper/more-specific concept is present
  - tagger_wrong    : tagger label names the wrong concept (incl. direction flips allowed<->missed)
plus its OWN concept name (so it's not forced to pick between the two given labels) + which is the
DEEPER true concept + reasoning citing the boards. Judge is blind to which label came from where
beyond being told 'tagger' vs 'other' — and told BOTH may be wrong.

This corrects the earlier token-overlap hack AND the Opus-is-ground-truth assumption: the judge reads
boards, and can reject BOTH labels.

Run: python3 judge_tagger.py --min_conf 0.30 --out judge_tagger.json   (all confident good feats)
     ... --only 21,303,102   (smoke)
"""
import argparse, json, os, re, sys, time, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,"/home/ec2-user/SageMaker")
import chess, boto3
from train_jr_canonical import JumpReLUSAEAuxK, load_data
BASE="/home/ec2-user/SageMaker"
MODEL_ID="us.anthropic.claude-opus-4-8"

SYS="""You audit a rule-based chess mistake TAGGER using an SAE (a model that clusters blunder positions
by a shared latent concept). You are shown, for one SAE feature, its TOP-FIRING positions — the
positions where this latent is strongest, so they share whatever concept the feature encodes. For each
you get: FEN, the move PLAYED (a blunder), Stockfish's BEST line, the REFUTATION line (how the blunder
is punished), and a one-line summary of what went wrong.

You are also given TWO candidate labels for the feature:
  - TAGGER label: what the rule-based tagger assigned (may be right, generic, or wrong).
  - ALT label: an independent model's label (also may be right, generic, or wrong).

Judge what the positions ACTUALLY share, from the boards — do NOT assume either label is correct.

Output STRICT JSON:
- "true_concept": 2-6 words, YOUR name for what these positions actually share (the deepest accurate
  concept). Name the specific mistake (piece, motif, mechanism) the boards support.
- "tagger_verdict": one of
    "accurate" — the TAGGER label correctly and fully names the shared concept.
    "shallow"  — the TAGGER label is not wrong but GENERIC; the positions share a more SPECIFIC concept
                 it misses (e.g. tagger "Missed Mate" but it's specifically a back-rank mate; tagger
                 "Missed Fork" but specifically a knight fork; tagger "Advanced Pawn" but a PASSED pawn).
    "wrong"    — the TAGGER label names the WRONG concept, including a DIRECTION FLIP (tagger says
                 "Allowed X" when the player MISSED X, or vice versa) or a category error (tagger
                 "Greedy Capture" when the move is actually an unsound SACRIFICE).
- "tagger_direction_flip": true/false — is the error specifically allowed<->missed (or attacker/victim) inverted?
- "alt_better": true/false — is the ALT label closer to true_concept than the tagger label?
- "reasoning": 2-3 sentences citing what you saw across the boards (which motif, which direction, at
  how many of the shown positions) that justifies the verdict.
Return ONLY the JSON."""

def line_san(fen, ucis, after=None):
    b=chess.Board(fen) if after is None else after
    out=[]
    for u in ucis or []:
        try:
            mv=chess.Move.from_uci(u)
            if mv not in b.legal_moves: break
            out.append(b.san(mv)); b.push(mv)
        except Exception: break
    return " ".join(out)

def build_prompt(fi, positions, tagger_label, alt_label):
    lines=[f"SAE FEATURE {fi}",
           f"TAGGER label: {tagger_label}", f"ALT label: {alt_label}","",
           "TOP-FIRING POSITIONS (each is a blunder these positions share a concept):"]
    for j,p in enumerate(positions,1):
        b=chess.Board(p["fen"])
        try: played=b.san(chess.Move.from_uci(p["blunder"]))
        except: played=p["blunder"]
        best=line_san(p["fen"], p.get("pv_uci"))
        after=b.copy()
        try: after.push(chess.Move.from_uci(p["blunder"]))
        except: pass
        refut=line_san(None, p.get("refutation_uci"), after=after)
        lines.append(f"  [{j}] FEN {p['fen']}")
        lines.append(f"      played (blunder): {played}   best: {best or '?'}   refutation: {refut or '?'}")
        if p.get("summary"): lines.append(f"      summary: {p['summary'][:200]}")
    lines.append("\nWhat do these positions actually share? Judge the tagger label. Return the strict JSON.")
    return "\n".join(lines)

def call(bedrock, prompt):
    body={"anthropic_version":"bedrock-2023-05-31","max_tokens":1500,
          "thinking":{"type":"adaptive"},"output_config":{"effort":"medium"},
          "system":SYS,"messages":[{"role":"user","content":prompt}]}
    for a in range(4):
        try:
            r=bedrock.invoke_model(modelId=MODEL_ID,body=json.dumps(body))
            blk=json.loads(r["body"].read())["content"]
            txt=next((b["text"] for b in blk if b.get("type")=="text"),"")
            return json.loads(re.search(r"\{.*\}",txt,re.S).group(0))
        except Exception as e:
            if a==3: return {"_error":str(e)[:150]}
            time.sleep(2*(a+1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--retag", default=f"{BASE}/jr_canon_out/retag_full.json")
    ap.add_argument("--sf", default=f"{BASE}/jr_canon_out/sf_lines_60k.jsonl")
    ap.add_argument("--opus", default=f"{BASE}/all_positions_labeled_opus.json")
    ap.add_argument("--weights", default=f"{BASE}/jr_canon_out/jr512_k8_final.pt")
    ap.add_argument("--cache", default=f"{BASE}/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
    ap.add_argument("--min_conf", type=float, default=0.30)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=f"{BASE}/jr_canon_out/judge_tagger.json")
    args=ap.parse_args()

    R=json.load(open(args.retag))
    sf={r["key"]:r for r in (json.loads(l) for l in open(args.sf) if l.strip()) if "key" in r}
    opus=json.load(open(args.opus))
    # which features to judge: good@opus + tagger fired >= min_conf
    if args.only:
        feats=[int(x) for x in args.only.split(",")]
    else:
        feats=[int(f) for f,v in R.items() if v.get("opus_verdict")=="good" and v.get("tagger_conf",0)>=args.min_conf]
    print(f"judging {len(feats)} features",flush=True)

    data,_,_=load_data(args.cache)
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); meta=cache["metadata"]
    all_keys=[f"{m['fen']}|{m['blunder_uci']}" for m in meta]
    cov=[i for i,k in enumerate(all_keys) if k in sf]; keys=[all_keys[i] for i in cov]
    ck=torch.load(args.weights,map_location="cpu",weights_only=False); c=ck["config"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=JumpReLUSAEAuxK(c["emb_size"],c["dict_size"],target_l0=c.get("target_l0",8),l0_alpha=c.get("l0_alpha",4),bandwidth=c["bandwidth"],init_threshold=c["init_threshold"]).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    D=c["dict_size"]; cov_t=torch.tensor(cov)
    want=set(feats)
    acts={f:torch.zeros(len(cov)) for f in feats}
    with torch.no_grad():
        for i in range(0,len(cov),16384):
            sel=cov_t[i:i+16384]; b=data[sel].to(dev)
            z=(b-sae.b_dec)@sae.W_enc+sae.b_enc; a=(z*(z>sae.threshold)).cpu()
            for f in feats: acts[f][i:i+a.shape[0]]=a[:,f]
    print("encoded",flush=True)

    def positions_for(fi):
        top=torch.topk(acts[fi],args.topn).indices.tolist()
        ps=[]
        for idx in top:
            k=keys[idx]; s=sf.get(k,{}); fen,bl=k.split("|")
            a=opus.get(k,{}).get("analysis",{})
            ps.append({"fen":fen,"blunder":bl,"pv_uci":s.get("pv_uci"),"refutation_uci":s.get("refutation_uci"),
                       "summary":a.get("blunder_summary","")})
        return ps

    bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")
    def work(fi):
        v=R[str(fi)]
        out=call(bedrock, build_prompt(fi, positions_for(fi), v["tagger_top"], v["opus_label"]))
        out["feature_idx"]=fi; out["tagger_label"]=v["tagger_top"]; out["tagger_conf"]=v["tagger_conf"]
        out["opus_label"]=v["opus_label"]
        return out
    res=[]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for fut in as_completed({ex.submit(work,f):f for f in feats}):
            res.append(fut.result())
    if args.only:
        for r in res: print(json.dumps(r,indent=1))
        return
    res.sort(key=lambda r:r["feature_idx"])
    json.dump(res,open(args.out,"w"),indent=1)
    from collections import Counter
    vc=Counter(r.get("tagger_verdict","_err") for r in res)
    flips=sum(1 for r in res if r.get("tagger_direction_flip"))
    print(f"DONE {len(res)} -> {args.out} | verdicts={dict(vc)} | direction_flips={flips}",flush=True)

if __name__=="__main__": main()
