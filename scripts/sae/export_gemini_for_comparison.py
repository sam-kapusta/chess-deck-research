"""Export 10 Gemini-labeled positions + run Sonnet on same ones for comparison."""
import json

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
OUTPUT_PATH = "/home/ec2-user/SageMaker/gemini_vs_sonnet_10.json"

with open(GEMINI_PATH) as f:
    gemini_profiles = json.load(f)

samples = []
for fid, feat in gemini_profiles.items():
    for ex in feat.get("examples", []):
        if ex.get("gemini_intent") and ex.get("uci") and len(samples) < 10:
            samples.append({
                "fen": ex["fen"],
                "uci": ex["uci"],
                "gemini_intent": ex.get("gemini_intent", ""),
                "gemini_blunder": ex.get("gemini_blunder", ""),
                "gemini_failure": ex.get("gemini_failure", ""),
            })
    if len(samples) >= 10:
        break

with open(OUTPUT_PATH, "w") as f:
    json.dump(samples, f, indent=2)
print(f"Saved {len(samples)} positions to {OUTPUT_PATH}")
