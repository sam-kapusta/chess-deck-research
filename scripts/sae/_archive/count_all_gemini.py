"""Count total unique Gemini-analyzed positions across all runs."""
import json

paths = [
    "/home/ec2-user/SageMaker/chess-stage-a/cache/1024_k32_feature_profiles_gemini.json",
    "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json",
    "/home/ec2-user/SageMaker/chess-stage-a/cache/maia2_2048_k64_feature_profiles_gemini.json",
]

all_positions = {}  # fen -> analysis

for path in paths:
    with open(path) as f:
        data = json.load(f)
    file_count = 0
    for fid, feat in data.items():
        for ex in feat.get("examples", []):
            fen = ex.get("fen", "")
            intent = ex.get("gemini_intent", "")
            if fen and intent:
                file_count += 1
                if fen not in all_positions:
                    all_positions[fen] = {
                        "fen": fen,
                        "uci": ex.get("uci", ""),
                        "gemini_intent": intent,
                        "gemini_blunder": ex.get("gemini_blunder", ""),
                        "gemini_failure": ex.get("gemini_failure", ""),
                    }
    name = path.split("/")[-1]
    print(f"{name}: {file_count} analyzed examples")

print(f"\nTotal unique FENs with Gemini analysis: {len(all_positions)}")

# Save combined
output_path = "/home/ec2-user/SageMaker/chess-stage-a/cache/all_gemini_positions.json"
with open(output_path, "w") as f:
    json.dump(list(all_positions.values()), f)
print(f"Saved combined to {output_path}")
