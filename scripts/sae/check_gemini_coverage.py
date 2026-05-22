"""Check Gemini position analysis coverage for Maia 3 SAE features."""
import json
import torch

GEMINI_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/2048_k64_feature_profiles_gemini.json"
ACT_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"

# Load Gemini analyses — keyed by encoder SAE feature, but positions have FENs
print("Loading Gemini position analyses...")
with open(GEMINI_PATH) as f:
    gemini_data = json.load(f)

# Build a lookup: FEN → gemini analysis text
fen_to_analysis = {}
for fid, feat in gemini_data.items():
    for ex in feat.get("examples", []):
        fen = ex.get("fen", "")
        intent = ex.get("gemini_intent", "")
        blunder = ex.get("gemini_blunder", "")
        failure = ex.get("gemini_failure", "")
        if fen and (intent or blunder or failure):
            fen_to_analysis[fen] = {
                "intent": intent,
                "blunder": blunder,
                "failure": failure,
                "full_text": f"{intent} {blunder} {failure}".strip(),
            }

print(f"  Unique FENs with Gemini analysis: {len(fen_to_analysis)}")

# Load our Maia 3 SAE activations metadata
print("Loading Maia 3 activation metadata...")
act_data = torch.load(ACT_PATH, map_location="cpu", weights_only=False)
our_metadata = act_data["metadata"]
our_fens = set(m["fen"] for m in our_metadata)

# Check overlap
overlap = our_fens & set(fen_to_analysis.keys())
print(f"  Our positions: {len(our_fens)}")
print(f"  Overlap with Gemini: {len(overlap)} ({100*len(overlap)/len(our_fens):.1f}%)")

# Sample some analyses
print("\nSample Gemini analyses (from overlapping positions):")
for i, fen in enumerate(list(overlap)[:5]):
    a = fen_to_analysis[fen]
    print(f"\n  Position {i+1}: {fen[:50]}...")
    print(f"    Intent: {a['intent'][:100]}...")
    print(f"    Blunder: {a['blunder'][:100]}...")
