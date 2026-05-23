"""Analyze Gemini validation results — check activation strengths."""
import json

path = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_validation.json"
with open(path) as f:
    data = json.load(f)

n_pos = data["n_positions"]
n_active = data["n_features_active"]
print(f"Active features on Gemini positions: {n_active}")
print(f"Total positions: {n_pos}")

# Check activation strengths
all_strengths = []
for fid, feat in data["features"].items():
    for ex in feat["examples"]:
        all_strengths.append(ex["strength"])

print(f"\nActivation strength distribution (top-100 features, up to 10 examples each):")
print(f"  Total examples: {len(all_strengths)}")
print(f"  Max: {max(all_strengths):.3f}")
print(f"  Mean: {sum(all_strengths)/len(all_strengths):.3f}")
print(f"  > 0.5: {sum(1 for s in all_strengths if s > 0.5)}")
print(f"  > 0.8: {sum(1 for s in all_strengths if s > 0.8)}")
print(f"  > 1.0: {sum(1 for s in all_strengths if s > 1.0)}")

# Per-feature: how many have 10+ examples > 0.8?
features_with_strong = 0
for fid, feat in data["features"].items():
    strong = [ex for ex in feat["examples"] if ex["strength"] > 0.8]
    if len(strong) >= 10:
        features_with_strong += 1

print(f"\n  Features with 10+ examples > 0.8: {features_with_strong}")

# Lower threshold
for thresh in [0.1, 0.2, 0.3, 0.5]:
    count = 0
    for fid, feat in data["features"].items():
        strong = [ex for ex in feat["examples"] if ex["strength"] > thresh]
        if len(strong) >= 10:
            count += 1
    print(f"  Features with 10+ examples > {thresh}: {count}")

# The problem: our validation only saved top-10 examples per feature
# But we have fire_count — how many positions total
print("\n\nFire count distribution (features sorted by fire_count):")
fire_counts = [(fid, feat["fire_count"], feat["fire_rate"]) for fid, feat in data["features"].items()]
fire_counts.sort(key=lambda x: -x[1])

print(f"  Hub features (>5% fire rate): {sum(1 for _,_,fr in fire_counts if fr > 0.05)}")
print(f"  Good range (1-5%): {sum(1 for _,_,fr in fire_counts if 0.01 <= fr <= 0.05)}")
print(f"  Sparse (<1%): {sum(1 for _,_,fr in fire_counts if fr < 0.01)}")

# Show features with highest max strength
print("\n\nFeatures with strongest activations:")
feat_max = []
for fid, feat in data["features"].items():
    strengths = [ex["strength"] for ex in feat["examples"]]
    if strengths:
        feat_max.append((fid, max(strengths), feat["fire_count"]))
feat_max.sort(key=lambda x: -x[1])

for fid, ms, fc in feat_max[:15]:
    feat = data["features"][fid]
    top_ex = feat["examples"][0]
    print(f"  Feature {fid}: max={ms:.3f}, fires={fc}, intent: {top_ex['gemini_intent'][:80]}")
