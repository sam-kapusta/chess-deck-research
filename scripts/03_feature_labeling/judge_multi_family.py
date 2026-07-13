#!/usr/bin/env python3
"""Multi-tag judge: shows the feature's FULL tagger tag distribution (top-5 with counts) + positions,
and asks whether the true concept is COVERED by ANY of the tags (recall), not just top-1 match.
Quick & dirty adaptation of judge_tagger.py for the re-measurement Sam asked for."""
import argparse, json, os, re, time, sys, torch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,"/home/ec2-user/SageMaker")
import chess, boto3
from train_jr_canonical import JumpReLUSAEAuxK, load_data
BASE="/home/ec2-user/SageMaker"
MODEL_ID="us.anthropic.claude-opus-4-8"
SYS="""You audit a chess-mistake tagger. For one SAE feature (a cluster of blunder positions) you see:
  - TOP-FIRING POSITIONS: FEN + played (blunder) + best line + refutation + summary
  - TAGGER TAGS: the FULL list of tags the rule-based tagger emits on these positions, with vote counts.
    Multiple tags fire PER POSITION (a blunder can be a hung piece AND an unsound sacrifice simultaneously).
  - ALT label: an independent model's name for the feature.

Judge from the boards what the positions ACTUALLY share. Then:
  "true_concept": 2-6 words naming the real shared mistake concept.
  "tagger_coverage": one of
    "covered"      — the true concept is NAMED (right + deep enough) by at LEAST ONE of the tagger's tags
                     OR by one of the CONCEPT FAMILIES (a family like "Hung Material" covers "hanging piece
                     left en prise"; "Missed Free Material" covers "missed capturing a free piece"). ANY
                     tag or family that appears counts — not just the highest-voted one.
    "shallow_only" — a RELATED but generic version of the concept is in the tags, but the SPECIFIC
                     concept is not named (e.g. tags have "Missed Fork" but the concept is "Knight Fork";
                     tags have "Allowed Mate" but the concept is "Back-Rank Mate").
    "not_covered"  — NONE of the tagger's tags correctly names the concept (they fire on co-occurring
                     side-effects, not the core concept).
  "covering_tag": which specific tagger tag covers the concept (null if not_covered).
  "alt_better": true/false.
  "reasoning": 2-3 sentences.
Return ONLY the JSON."""

def line_san(fen,ucis,after=None):
    b=chess.Board(fen) if after is None else after; o=[]
    for u in ucis or []:
        try:
            mv=chess.Move.from_uci(u)
            if mv not in b.legal_moves:break
            o.append(b.san(mv));b.push(mv)
        except:break
    return " ".join(o)

def build_prompt(fi,positions,tagger_votes,alt_label,family_votes=None):
    lines=[f"SAE FEATURE {fi}",
           f"TAGGER TAGS (piece-specific labels, vote count across top-firing positions, highest first):",
           "  "+", ".join(f"{t} ({c})" for t,c in tagger_votes.items())]
    if family_votes:
        lines.append("TAGGER CONCEPT FAMILIES (piece-specific tags rolled up to their parent concept; "
                     "'Missed Free Material' = any Missed Free Q/R/B/N/P, 'Hung Material' = any Hung piece):")
        lines.append("  "+", ".join(f"{t} ({c})" for t,c in family_votes.items()))
    lines += [f"ALT label: {alt_label}","","TOP-FIRING POSITIONS:"]
    for j,p in enumerate(positions,1):
        b=chess.Board(p["fen"])
        try:played=b.san(chess.Move.from_uci(p["blunder"]))
        except:played=p["blunder"]
        best=line_san(p["fen"],p.get("pv_uci"))
        after=b.copy()
        try:after.push(chess.Move.from_uci(p["blunder"]))
        except:pass
        refut=line_san(None,p.get("refutation_uci"),after=after)
        lines.append(f"  [{j}] FEN {p['fen']}")
        lines.append(f"      played: {played}   best: {best or '?'}   refutation: {refut or '?'}")
        if p.get("summary"):lines.append(f"      summary: {p['summary'][:200]}")
    lines.append("\nJudge the positions. Is the true concept COVERED BY at least one of the tagger tags? Return strict JSON.")
    return "\n".join(lines)

def call(bedrock,prompt):
    body={"anthropic_version":"bedrock-2023-05-31","max_tokens":1200,
          "thinking":{"type":"adaptive"},"output_config":{"effort":"medium"},
          "system":SYS,"messages":[{"role":"user","content":prompt}]}
    for a in range(4):
        try:
            r=bedrock.invoke_model(modelId=MODEL_ID,body=json.dumps(body))
            blk=json.loads(r["body"].read())["content"]
            txt=next((b["text"] for b in blk if b.get("type")=="text"),"")
            return json.loads(re.search(r"\{.*\}",txt,re.S).group(0))
        except Exception as e:
            if a==3:return {"_error":str(e)[:150]}
            time.sleep(2*(a+1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--retag",default=f"{BASE}/jr_canon_out/retag_full.json")
    ap.add_argument("--sf",default=f"{BASE}/jr_canon_out/sf_lines_60k.jsonl")
    ap.add_argument("--opus",default=f"{BASE}/all_positions_labeled_opus.json")
    ap.add_argument("--opus_labels",default=f"{BASE}/jr_canon_out/labels_decile_jr512.json")
    ap.add_argument("--weights",default=f"{BASE}/jr_canon_out/jr512_k8_final.pt")
    ap.add_argument("--cache",default=f"{BASE}/chess-stage-a/cache/maia3_l7only_v2_dedup.pt")
    ap.add_argument("--min_conf",type=float,default=0.20)
    ap.add_argument("--topn",type=int,default=8)
    ap.add_argument("--only",default="")
    ap.add_argument("--out",default=f"{BASE}/jr_canon_out/judge_multi_family.json")
    args=ap.parse_args()
    R=json.load(open(args.retag))
    sf={r["key"]:r for r in (json.loads(l) for l in open(args.sf) if l.strip()) if "key" in r}
    opus=json.load(open(args.opus))
    opus_labels={r["feature_idx"]:r for r in json.load(open(args.opus_labels))["labels"]}
    if args.only: feats=[int(x) for x in args.only.split(",")]
    else: feats=[int(f) for f,v in R.items() if v.get("opus_verdict")=="good" and v.get("tagger_top")]
    print(f"judging {len(feats)} features (multi-tag)",flush=True)
    data,_,_=load_data(args.cache)
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); meta=cache["metadata"]
    ak=[f"{m['fen']}|{m['blunder_uci']}" for m in meta]; cov=[i for i,k in enumerate(ak) if k in sf]; keys=[ak[i] for i in cov]
    ck=torch.load(args.weights,map_location="cpu",weights_only=False); c=ck["config"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=JumpReLUSAEAuxK(c["emb_size"],c["dict_size"],target_l0=c.get("target_l0",8),l0_alpha=c.get("l0_alpha",4),bandwidth=c["bandwidth"],init_threshold=c["init_threshold"]).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    acts={f:torch.zeros(len(cov)) for f in feats}
    with torch.no_grad():
        for i in range(0,len(cov),16384):
            b=data[torch.tensor(cov[i:i+16384])].to(dev); z=(b-sae.b_dec)@sae.W_enc+sae.b_enc; a=(z*(z>sae.threshold)).cpu()
            for f in feats: acts[f][i:i+a.shape[0]]=a[:,f]
    print("encoded",flush=True)
    bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")
    def positions_for(fi):
        top=torch.topk(acts[fi],args.topn).indices.tolist()
        return [{"fen":keys[i].split("|")[0],"blunder":keys[i].split("|")[1],
                 "pv_uci":sf.get(keys[i],{}).get("pv_uci"),"refutation_uci":sf.get(keys[i],{}).get("refutation_uci"),
                 "summary":opus.get(keys[i],{}).get("analysis",{}).get("blunder_summary","")} for i in top]
    def work(fi):
        v=R[str(fi)]
        out=call(bedrock,build_prompt(fi,positions_for(fi),v.get("tagger_votes",{}),opus_labels.get(fi,{}).get("good_label",""),v.get("tagger_family_votes",{})))
        out["feature_idx"]=fi; out["tagger_votes"]=v.get("tagger_votes",{}); out["tagger_family_votes"]=v.get("tagger_family_votes",{}); out["opus_label"]=opus_labels.get(fi,{}).get("good_label","")
        return out
    res=[]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for fut in as_completed({ex.submit(work,f):f for f in feats}):res.append(fut.result())
    if args.only:
        for r in res:print(json.dumps(r,indent=1))
        return
    res.sort(key=lambda r:r["feature_idx"])
    json.dump(res,open(args.out,"w"),indent=1)
    vc=Counter(r.get("tagger_coverage","_err") for r in res)
    print(f"DONE {len(res)} | coverage={dict(vc)}",flush=True)

if __name__=="__main__":main()
