#!/usr/bin/env python3
"""Label v2 Matryoshka features using matched Opus analyses.

v2 data matches the 19.6K Opus analyses directly by FEN+UCI key (no idx
indirection needed — the analyses were generated FROM v2 positions).

Extracts top-activating positions per feature, pulls their analyses,
then synthesizes a feature label with Sonnet.

Usage (on chess-poc):
    python3 scripts/sae/label_v2_features.py \
        --model chess-stage-a/output/maia3_sae_v2/matryoshka_v2_H1_p32_288_2336_k3_8_16.pt \
        --n-features 32 --output h1_top32_labels.json
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
V2_PATH = BASE + "/cache/maia3_blunder_diff_v2.pt"
OPUS_PATH = "/home/ec2-user/SageMaker/all_positions_labeled_opus.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    MODEL_ID = "us.anthropic.claude-sonnet-4-6"

    # Load v2 data
    print("Loading v2 data...")
    data = torch.load(V2_PATH, map_location="cpu", weights_only=False)
    raw = data["activations"].float()
    meta = data["metadata"]
    fens = [m.get("fen", "") for m in meta]

    mean = raw.mean(dim=0)
    std = raw.std(dim=0).clamp(min=1e-6)
    x = (raw - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw

    # Load opus, build fen|uci -> analysis
    print("Loading analyses...")
    with open(OPUS_PATH) as f:
        opus = json.load(f)

    # Load model
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    We = sd.get("W_enc", sd.get("We"))
    be = sd.get("b_enc", sd.get("be"))
    bd = sd.get("b_dec", sd.get("bd"))
    config = ckpt["config"]
    prefixes = config["prefixes"]
    k_per_level = config.get("k_per_level", [config.get("k", 16)] * len(prefixes))

    # Forward pass
    print("Forward pass...")
    z = F.relu((x - bd) @ We + be)
    n = x.shape[0]
    acts = torch.zeros_like(z)
    for i, (ps, k) in enumerate(zip(prefixes, k_per_level)):
        s = 0 if i == 0 else prefixes[i - 1]
        g = z[:, s:ps]
        fg = g.reshape(-1)
        gk = min(k * n, fg.numel())
        tv, ti = torch.topk(fg, k=gk)
        ga = torch.zeros_like(fg)
        ga[ti] = tv
        acts[:, s:ps] = ga.reshape(n, ps - s)
    feat_max = acts.max(dim=0).values.clamp(min=1e-8)
    acts_norm = acts / feat_max.unsqueeze(0)

    def get_analysis_brief(idx):
        key = fens[idx] + "|" + meta[idx]["blunder_uci"]
        if key not in opus:
            return None
        a = opus[key]
        inner = a["analysis"] if "analysis" in a and isinstance(a["analysis"], dict) else a
        return {
            "intent": inner.get("move_intent", "")[:140],
            "refutation": inner.get("refutation_analysis", "")[:140],
            "summary": inner.get("blunder_summary", "")[:140],
            "motif": inner.get("tactical_motif", ""),
        }

    def label_feature(fi):
        feat_acts = acts_norm[:, fi]
        # Sample across activation range: half from top, half from mid
        above = torch.where(feat_acts > args.threshold)[0]
        sorted_above = above[torch.argsort(feat_acts[above], descending=True)]

        examples = []
        # Top half
        for idx in sorted_above.tolist():
            brief = get_analysis_brief(idx)
            if brief and len(examples) < args.n_examples // 2:
                examples.append((float(feat_acts[idx]), meta[idx]["blunder_uci"], brief))
        # Mid sample (every Nth from the rest)
        rest = sorted_above[len(sorted_above)//3:].tolist()
        step = max(1, len(rest) // (args.n_examples // 2))
        for idx in rest[::step]:
            brief = get_analysis_brief(idx)
            if brief and len(examples) < args.n_examples:
                examples.append((float(feat_acts[idx]), meta[idx]["blunder_uci"], brief))

        if len(examples) < 5:
            return {"chip": "INSUFFICIENT", "n_examples": len(examples), "confidence": 0}

        txt = ""
        for act, uci, b in examples:
            txt += f"(act={act:.2f}) move={uci}: {b['summary']} [{b['motif']}]\n"

        prompt = f"""These chess blunders all activate the same SAE feature. Find the SHARED pattern — what category of mistake is this? Keep it BROAD (this is a top-level category, not a specific tactic).

{txt}

Respond ONLY with JSON:
{{"chip": "2-4 word category", "label": "one sentence", "confidence": 0-100}}"""

        resp = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            })
        )
        result = json.loads(resp["body"].read())
        text = result["content"][0]["text"]
        st, en = text.find("{"), text.rfind("}") + 1
        label = json.loads(text[st:en])
        label["n_examples"] = len(examples)
        return label

    results = {}
    for fi in range(args.n_features):
        try:
            label = label_feature(fi)
            results[fi] = label
            print(f"F{fi:3d}: {label.get('chip','?'):35s} (conf={label.get('confidence','?')}, n={label.get('n_examples','?')})")
        except Exception as e:
            results[fi] = {"chip": "ERROR", "label": str(e)[:80], "confidence": 0}
            print(f"F{fi:3d}: ERROR {str(e)[:60]}")
        sys.stdout.flush()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
