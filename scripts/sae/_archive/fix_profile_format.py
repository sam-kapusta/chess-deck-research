"""Fix profile format to match label.py expectations."""
import json

path = "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/l2_feature_profiles.json"

with open(path) as f:
    profiles = json.load(f)

for fid, p in profiles.items():
    if "top_examples" in p:
        p["examples"] = p.pop("top_examples")
    for ex in p.get("examples", []):
        if "activation" in ex and "strength" not in ex:
            ex["strength"] = ex["activation"]

with open(path, "w") as f:
    json.dump(profiles, f)

n_with_5 = sum(1 for p in profiles.values() if len(p.get("examples", [])) >= 5)
n_with_20 = sum(1 for p in profiles.values() if len(p.get("examples", [])) >= 20)
print(f"Fixed. Features with 5+ examples: {n_with_5}, with 20: {n_with_20}")
