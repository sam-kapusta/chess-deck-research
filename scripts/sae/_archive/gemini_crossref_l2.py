"""Run 5568 Gemini-analyzed positions through the L2 SAE and cross-reference."""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from maia3_activations import extract_activations, pool_activations

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
SAE_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_l2_200ep.pt"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"
LABELS_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_labels_sonnet.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_l2_crossref.json"

# Load Gemini positions
print("Loading Gemini analyses...")
with open(GEMINI_PATH) as f:
    gemini_data = json.load(f)

positions = []
seen = set()
for fid, feat in gemini_data.items():
    for ex in feat.get("examples", []):
        fen = ex.get("fen", "")
        uci = ex.get("uci", "")
        if fen and uci and fen not in seen:
            seen.add(fen)
            positions.append({
                "fen": fen,
                "uci": uci,
                "gemini_intent": ex.get("gemini_intent", ""),
                "gemini_blunder": ex.get("gemini_blunder", ""),
                "gemini_failure": ex.get("gemini_failure", ""),
            })
print(f"  {len(positions)} unique Gemini positions")

# Extract activations
fens = [p["fen"] for p in positions]
ucis = [p["uci"] for p in positions]
rng = np.random.default_rng(42)
elos = rng.integers(600, 2601, size=len(fens)).tolist()

print("Extracting Maia 3 activations...")
raw, mirrored = extract_activations(fens, elo_self=elos, elo_oppo=elos)
pooled = pool_activations(raw, "diff", ucis, mirrored)
print(f"  Shape: {pooled.shape}, NaN: {np.isnan(pooled).any()}")

# Load L2 SAE
print("Loading L2 SAE...")
sae_data = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
state = sae_data["state_dict"]
k = sae_data["config"]["k"]

# Apply L2 normalization (matching training)
train_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)
train_x = train_data["activations"].float()
mean = train_x.mean(dim=0)
std = train_x.std(dim=0).clamp(min=1e-6)
del train_x

x = torch.from_numpy(pooled).float()
x = (x - mean) / std
norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
x = x / norms

# Forward pass
z = (x - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

# Per-feature: normalize by feature max (from training profiles)
with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json") as f:
    profiles = json.load(f)

feat_maxes = torch.zeros(2048)
for fid, prof in profiles.items():
    examples = prof.get("examples", [])
    if examples:
        feat_maxes[int(fid)] = examples[0].get("activation", examples[0].get("strength", 0.01))
feat_maxes = feat_maxes.clamp(min=0.01)

# Normalize to relevance
acts_normed = acts / feat_maxes.unsqueeze(0)

print(f"\nActivation stats:")
print(f"  Features with any Gemini position > 0.5 relevance: {(acts_normed.max(dim=0).values > 0.5).sum().item()}")
print(f"  Features with any Gemini position > 0.7 relevance: {(acts_normed.max(dim=0).values > 0.7).sum().item()}")

# Load labels for cross-reference
with open(LABELS_PATH) as f:
    labels = json.load(f)

# For each feature, collect Gemini positions that fire strongly
results = {}
for feat_id in range(2048):
    feat_acts = acts_normed[:, feat_id]
    strong_mask = feat_acts > 0.5  # Above 50% of feature's max
    n_strong = strong_mask.sum().item()
    if n_strong >= 3:
        strong_idx = strong_mask.nonzero(as_tuple=True)[0]
        sorted_idx = strong_idx[feat_acts[strong_idx].argsort(descending=True)]
        examples = []
        for idx in sorted_idx[:10]:
            idx = idx.item()
            p = positions[idx]
            examples.append({
                "relevance": round(feat_acts[idx].item(), 3),
                "fen": p["fen"],
                "uci": p["uci"],
                "gemini_intent": p["gemini_intent"][:200],
                "gemini_blunder": p["gemini_blunder"][:200],
            })
        fid_str = str(feat_id)
        label_info = labels.get(fid_str, {})
        results[fid_str] = {
            "n_strong": n_strong,
            "sonnet_label": label_info.get("specific_label", "unlabeled"),
            "sonnet_confidence": label_info.get("confidence_score", 0),
            "examples": examples,
        }

print(f"\nFeatures with 3+ Gemini positions > 0.5 relevance: {len(results)}")
low_conf_with_gemini = sum(1 for fid, r in results.items()
                           if labels.get(fid, {}).get("confidence_score", 1) < 0.7)
print(f"Of those, low-confidence features: {low_conf_with_gemini}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
