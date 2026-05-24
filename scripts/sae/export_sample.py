"""Export sample features with Stockfish data to JSON."""
import json, random

with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json") as f:
    profiles = json.load(f)
with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/stockfish_data.json") as f:
    sf_data = json.load(f)
with open("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_labels_sonnet.json") as f:
    labels = json.load(f)

# Get features with confidence around 0.8
candidates = [(fid, lab) for fid, lab in labels.items() if 0.75 <= lab.get("confidence_score", 0) <= 0.85]
random.seed(42)
random.shuffle(candidates)
picks = candidates[:5]

output = {}
for fid, label in picks:
    prof = profiles.get(fid, {})
    examples = prof.get("examples", [])[:10]
    positions = []
    for ex in examples:
        key = ex.get("fen", "") + "|" + ex.get("uci", "")
        sf = sf_data.get(key, {})
        positions.append({"fen": ex.get("fen"), "uci": ex.get("uci"), "relevance": ex.get("relevance", 0), **sf})
    output[fid] = {"feature_id": int(fid), "label": label, "positions": positions}

with open("/home/ec2-user/SageMaker/features_08_sample.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved 5 features:")
for fid, label in picks:
    l = label["specific_label"]
    c = label["confidence_score"]
    print(f"  F{fid}: {l} (conf={c})")
