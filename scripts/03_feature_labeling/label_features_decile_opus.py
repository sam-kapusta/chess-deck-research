#!/usr/bin/env python3
"""Chess SAE feature labeling — decile + Opus, ported from SandstonePersonas opus_label_audit_2.py.

Same MACHINERY as the persona labeler: activation deciles (TIP=top-100, D10..D1), orthogonal
STRENGTH verdict (good/diffuse/polysemantic/noise/too_broad/wrong) + REACH (good_until_decile,
broad_until_decile), Opus 4.8 medium/adaptive, threaded, JSONL checkpoint. Domain reframed to chess
MISTAKE CONCEPTS: per band we aggregate the existing Opus per-position analyses (tactical_motif +
tags + blunder_summary snippets) as the signal — the chess analog of persona category/title lift.

Coverage caveat: only ~35% of cache positions have an Opus analysis, so each band's signal is over
the covered subset; per-band coverage count is shown so Opus can weight it (mirrors persona n=100).

Run (chess-poc, pytorch_p310 for encode; Bedrock for Opus):
  python3 label_features_decile_opus.py --weights jr_canon_out/jr512_k8_final.pt \
      --cache chess-stage-a/cache/maia3_l7only_v2_dedup.pt --opus all_positions_labeled_opus.json \
      --random 20 --seed 7 --out labels_decile_pilot.json     # pilot
  ... (no --random) → all features
"""
import argparse, json, os, re, time, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, torch, boto3
sys.path.insert(0, "/home/ec2-user/SageMaker")
from train_jr_canonical import JumpReLUSAEAuxK, load_data

MODEL_ID = "us.anthropic.claude-opus-4-8"
N_THREADS = 16
N_DECILES = 10
TIP_N = 100

SYS = """You label interpretability features for an SAE (sparse autoencoder) trained on CHESS MISTAKES.
Each feature fires on chess positions where a player blundered; the training signal is the Maia
layer-7 activation DIFF between the engine's best move and the played (blunder) move — so a feature
captures a recurring KIND of mistake / missed idea.

Activation strength orders the firing positions into bands: TIP = the top-100 strongest activators,
then D10 (next strongest 10%) down to D1 (weakest positives near the noise floor). For each band we
aggregate independent per-position analyses (from a strong chess model) into:
  - MOTIF distribution: the single tactical_motif per position (hanging_piece, fork, king_safety,
    tempo_loss, pin, back_rank, passed_pawn, trapped_piece, discovered_attack, overloaded_defender,
    pawn_endgame, rook_endgame, promotion_error, positional_mistake, skewer, missed_tactic, ...),
    shown as "motif COV%(Nc)" where COV% is the share of that band's ANALYZED positions with it.
  - TAG distribution: finer free tags (knight_fork, exposed_king, back_rank, undefended_piece, ...).
  - SUMMARY SAMPLES: a few one-line blunder_summary snippets from the band's top positions.
COVERAGE% is REACH within a band. Only ~35% of positions carry an analysis, so each band also shows
COV (n analyzed / n in band) — trust a band's signal in proportion to how many analyses back it.

READ THE MOTIF/TAG COVERAGE FIRST. A feature genuinely encodes a concept when one motif (or a tight
cluster of related motifs/tags) DOMINATES a substantial share of the TIP + top deciles. A motif at
~background rate is not discriminating; a motif at 2 positions is the noise floor. The concept is
whatever recurs sharply at the TIP and top deciles.

The science questions your output must answer (same as any SAE autointerp):
  1. Which features encode a TRULY sharp mistake concept at the top activators?
  2. How DEEP does it hold — down to which decile before it broadens/decays?
  3. Which are good only at the very TIP then go diffuse immediately?
  4. Which are POLYSEMANTIC — 2+ unrelated mistake themes, no single concept?
  5. Which are NOISE — no coherent theme even at the strongest activators?

DECAY IS NORMAL: a precise concept ("Hanging Rook to Queen") broadens as activation drops
("hanging piece" then "material loss"). Record the broadening in broad_label; judge the PRECISE
label at the TOP bands (TIP, D10-D8). Do NOT punish normal decay.

Output STRICT JSON:
1. "verdict" — quality of the best precise concept, judged at the TOP bands:
   - "good"        : a sharp, nameable mistake concept dominates the top activators (use
                     good_until_decile for reach; good@D10 and good@D2 are both "good").
   - "diffuse"     : a real theme at the tip but only a minority share it and it fades immediately.
   - "too_broad"   : label vaguer than the top-band signal supports (could be sharpened).
   - "wrong"       : label contradicts the top-band signal.
   - "polysemantic": fires on 2+ UNRELATED mistake themes; name them in reasoning.
   - "noise"       : no coherent theme even at the top activators.
2. "good_until_decile" — deepest decile (10..1) the PRECISE concept still dominates a substantial
   share. D10=holds only at top 10%; D1=holds almost all the way. null if no clean precise zone.
3. "broad_until_decile" — deepest decile the feature is still BROADLY related before coverage
   collapses to noise. Must be <= good_until_decile. null if even the tip is noise.
4. "good_label" + "broad_label" — 2-5 words, Title Case, name the MISTAKE CONCEPT (e.g. "Hanging Rook
   to Queen", "Missed Back-Rank Mate", "Premature Queen Trade", "Overloaded Defender", "King-Safety
   Pawn Push"). good_label = precise concept at the top; broad_label = what it decays into (set equal
   if it never broadens). Name the ERROR/idea, not a generic "Mistake"/"Bad Move".
5. "description" — 1-2 sentence behavioral description of the mistake at the top activators, grounded
   in the actual motif/tag/summary signal.
6. "reasoning" — 2-4 sentence evidence trail citing ACTUAL numbers: which motif dominates at TIP and
   at what coverage, which tags carry it, where the precise concept gives way (-> good_until_decile),
   where it fades to noise (-> broad_until_decile), why diffuse/polysemantic if so.
7. "tag" — compact "<verdict>@D<good_until_decile>", e.g. "good@D10", "diffuse@D10", "polysemantic@-".
Return ONLY the JSON object."""


def band_signal(keys_in_band, opus, n_in_band, n_summ=3):
    """Aggregate the Opus analyses for the positions in one band -> a compact text block."""
    analyzed = [opus[k] for k in keys_in_band if k in opus]
    n_an = len(analyzed)
    if n_an == 0:
        return f"    (0 of {n_in_band} analyzed — no signal)"
    def A(e): return e.get("analysis", e)
    mot = Counter(str(A(e).get("tactical_motif","")).strip().lower() for e in analyzed if A(e).get("tactical_motif"))
    tags = Counter(t.strip().lower() for e in analyzed for t in (A(e).get("tags") or []))
    pct = lambda c: f"{round(100*c/n_an)}%({c}c)"
    mot_s = "  ".join(f"{m} {pct(c)}" for m,c in mot.most_common(6))
    tag_s = "  ".join(f"{t} {pct(c)}" for t,c in tags.most_common(8))
    summ = [str(A(e).get("blunder_summary","")).strip()[:160] for e in analyzed[:n_summ] if A(e).get("blunder_summary")]
    lines = [f"    coverage: {n_an}/{n_in_band} analyzed",
             f"    motif: {mot_s or '-'}",
             f"    tags:  {tag_s or '-'}"]
    for s in summ: lines.append(f"    summary: {s}")
    return "\n".join(lines)


def build_prompt(fi, bands, prev_label):
    blocks = []
    for name, keys, n_in in bands:
        blocks.append(f"  {name}:\n{band_signal(keys, OPUS, n_in)}")
    prev = f'\nPREVIOUS (tagger-vote) LABEL, reference only: {prev_label}' if prev_label else ''
    return f"""FEATURE {fi}{prev}

PER-BAND SIGNAL (TIP = top-100 strongest activators; then D10 = next strongest 10% -> D1 = weakest).
Each band: coverage (analyzed/total), then the MOTIF and TAG distributions and a few blunder summaries.
{chr(10).join(blocks)}

Label this feature's mistake concept. Read motif/tag COVERAGE at the TIP and top deciles first. Where
does the precise concept hold (-> good_until_decile), what does it broaden into (broad_label), where
does coverage collapse (-> broad_until_decile)? Return the strict JSON object."""


def call_opus(bedrock, prompt, effort="medium"):
    body = {"anthropic_version":"bedrock-2023-05-31","max_tokens":4000,
            "thinking":{"type":"adaptive"},"output_config":{"effort":effort},
            "system":SYS,"messages":[{"role":"user","content":prompt}]}
    for a in range(4):
        try:
            r=bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
            blocks=json.loads(r["body"].read())["content"]
            txt=next((b["text"] for b in blocks if b.get("type")=="text"),"")
            m=re.search(r"\{.*\}",txt,re.S); return json.loads(m.group(0))
        except Exception as e:
            if a==3: return {"_error":str(e)[:200]}
            time.sleep(2*(a+1))


OPUS = {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",required=True); ap.add_argument("--cache",required=True)
    ap.add_argument("--opus",required=True); ap.add_argument("--prev",default="")
    ap.add_argument("--random",type=int,default=0); ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--only",default=""); ap.add_argument("--effort",default="medium")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    global OPUS
    OPUS=json.load(open(args.opus)); print(f"opus analyses: {len(OPUS)}",flush=True)
    prev_labels={}
    if args.prev and os.path.exists(args.prev):
        pj=json.load(open(args.prev)).get("labels",{}); prev_labels={int(k):v.get("label") for k,v in pj.items()}

    data,_,_=load_data(args.cache)
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); meta=cache["metadata"]
    all_keys=[f"{m['fen']}|{m['blunder_uci']}" for m in meta]
    # RESTRICT to positions that HAVE an Opus analysis, then decile ONLY those (Sam) — so every band
    # is fully backed by analyses instead of ~35% coverage that thins in the low deciles.
    covered_idx=[i for i,k in enumerate(all_keys) if k in OPUS]
    keys=[all_keys[i] for i in covered_idx]
    ck=torch.load(args.weights,map_location="cpu",weights_only=False); c=ck["config"]
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=JumpReLUSAEAuxK(c["emb_size"],c["dict_size"],target_l0=c.get("target_l0",8),
        l0_alpha=c.get("l0_alpha",4),bandwidth=c["bandwidth"],init_threshold=c["init_threshold"]).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    D=c["dict_size"]; n=len(keys)
    print(f"encoding {len(data)} positions; deciling over {n} WITH-analysis positions, dict={D}...",flush=True)
    cov_t=torch.tensor(covered_idx)
    acts=torch.zeros(D,n,dtype=torch.float32)   # columns = analyzed positions only
    with torch.no_grad():
        for i in range(0,len(covered_idx),16384):
            sel=cov_t[i:i+16384]
            b=data[sel].to(dev)
            z=(b-sae.b_dec)@sae.W_enc+sae.b_enc; a=(z*(z>sae.threshold)).cpu()
            acts[:,i:i+a.shape[0]]=a.T
    print(f"encoded {n} analyzed positions.",flush=True)

    feats=list(range(D))
    if args.only: feats=[int(x) for x in args.only.split(",")]
    elif args.random:
        import random; feats=sorted(random.Random(args.seed).sample(feats,min(args.random,D)))
    print(f"labeling {len(feats)} features",flush=True)

    def bands_for(fi):
        col=acts[fi]; nz=int((col>0).sum())
        if nz==0: return None
        order=torch.argsort(col,descending=True)[:nz].tolist()
        out=[]
        tip=order[:min(TIP_N,nz)]
        out.append(("TIP (top-100)",[keys[i] for i in tip],len(tip)))
        # deciles over the firing positions
        for d in range(10,0,-1):
            lo=int((10-d)/10*nz); hi=int((10-d+1)/10*nz)
            idx=order[lo:hi]
            if idx: out.append((f"D{d} (top {(11-d)*10}%)",[keys[i] for i in idx],len(idx)))
        return out

    bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")
    ckpt=args.out.replace(".json",".jsonl")
    done=set()
    if not args.only and os.path.exists(ckpt):
        done={json.loads(l)["feature_idx"] for l in open(ckpt) if l.strip()}
    todo=[fi for fi in feats if fi not in done]
    ckf=None if args.only else open(ckpt,"a")

    def work(fi):
        b=bands_for(fi)
        if b is None: return {"feature_idx":fi,"verdict":"noise","tag":"noise@-","good_label":None,"reasoning":"dead (no firing among analyzed positions)"}
        fr=int((acts[fi]>0).sum())/n   # fire rate among ANALYZED positions (deciling universe)
        out=call_opus(bedrock,build_prompt(fi,b,prev_labels.get(fi)),args.effort)
        out["feature_idx"]=fi; out["fire_rate"]=round(fr,4)
        gd=out.get("good_until_decile")
        out["good_pct"]=(11-gd)*10 if isinstance(gd,int) and 1<=gd<=10 else None
        return out

    res=[]; nd=0
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futs={ex.submit(work,fi):fi for fi in todo}
        for fut in as_completed(futs):
            r=fut.result(); res.append(r)
            if ckf: ckf.write(json.dumps(r)+"\n"); ckf.flush()
            nd+=1
            if nd%20==0: print(f"  {nd}/{len(todo)}",flush=True)
    if ckf: ckf.close()
    if args.only:
        for r in res: print(json.dumps(r,indent=1))
        return
    final=[json.loads(l) for l in open(ckpt) if l.strip()]; final.sort(key=lambda r:r["feature_idx"])
    json.dump({"weights":os.path.basename(args.weights),"n_features":D,"labels":final},open(args.out,"w"),indent=1)
    verdicts=Counter(r.get("verdict","_err") for r in final)
    print(f"DONE {len(final)} -> {args.out} | verdicts={dict(verdicts)}",flush=True)

if __name__=="__main__": main()
