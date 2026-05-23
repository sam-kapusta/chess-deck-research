"""Check how many features have Gemini-analyzed positions from the raw SAE validation."""
import json

with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/gemini_raw_validation.json") as f:
    validation = json.load(f)

features = validation.get("features", {})
print(f"Features with 3+ Gemini positions (activation > 1.0): {len(features)}")

n_5 = sum(1 for f in features.values() if f.get("n_strong", 0) >= 5)
n_10 = sum(1 for f in features.values() if f.get("n_strong", 0) >= 10)
n_20 = sum(1 for f in features.values() if f.get("n_strong", 0) >= 20)
print(f"  With 5+ strong positions: {n_5}")
print(f"  With 10+ strong positions: {n_10}")
print(f"  With 20+ strong positions: {n_20}")

# But wait — this was the RAW SAE. We need to redo for L2.
# Actually the key question: can we just use these Gemini texts
# to re-label features that had low confidence?
# Let's check overlap with our low-confidence features

with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_labels_sonnet.json") as f:
    labels = json.load(f)

low_conf_ids = set(fid for fid, v in labels.items() if v.get("confidence_score", 1.0) < 0.7)
print(f"\nLow confidence features: {len(low_conf_ids)}")

# How many low-conf features have Gemini data from the raw validation?
low_conf_with_gemini = set(fid for fid in features.keys() if fid in low_conf_ids)
print(f"Low-conf features with Gemini data (raw SAE): {len(low_conf_with_gemini)}")

# Show some examples
print("\nSample low-conf features with Gemini positions:")
count = 0
for fid in sorted(low_conf_with_gemini, key=lambda x: -features[x]["n_strong"])[:10]:
    feat = features[fid]
    label = labels[fid]["specific_label"]
    conf = labels[fid]["confidence_score"]
    print(f"\n  Feature {fid}: '{label}' (conf={conf}, {feat['n_strong']} Gemini positions)")
    for ex in feat["examples"][:2]:
        intent = ex.get("gemini_intent", "")[:100]
        print(f"    [{ex['strength']:.2f}] {ex['uci']}  Intent: {intent}")
