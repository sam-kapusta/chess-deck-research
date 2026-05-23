"""Quick check: are Gemini positions blunders or puzzles?"""
import json

with open("/home/ec2-user/SageMaker/chess-stage-a/cache/all_gemini_positions.json") as f:
    data = json.load(f)

print(f"Total: {len(data)} positions")
print()

for i, p in enumerate(data[:5]):
    fen = p["fen"]
    uci = p["uci"]
    intent = p["gemini_intent"][:200]
    blunder = p["gemini_blunder"][:200]
    print(f"Position {i+1}:")
    print(f"  FEN: {fen[:60]}")
    print(f"  UCI: {uci}")
    print(f"  Intent (why they played it): {intent}")
    print(f"  Blunder (what went wrong): {blunder}")
    print()
