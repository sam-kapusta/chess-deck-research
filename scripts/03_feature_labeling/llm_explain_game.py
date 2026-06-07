#!/usr/bin/env python3
"""LLM-only per-move mistake explainer — the SAE-free baseline.

The SAE cannot say WHY a move is bad mechanically (mean64 pooling deleted localization). An LLM
reasoning over the board + engine lines CAN. This is the head-to-head: feed each mistake to Haiku
AND Sonnet, get a coaching explanation, measure real token cost per game.

Inputs: an analyze_cli.py game json (--top must be high enough that EVERY mistake gets the deep
pass — refutation + MultiPV; --top 16 covers a full game). We feed the LLM everything we have:
  - an ASCII board (LLMs reason far better with a drawn board than raw FEN)
  - clean evals before/after (pawns), the engine's best move
  - ALL 3 MultiPV best lines (not just the top) — alternatives matter for coaching
  - the refutation line (what actually punishes the played move) — the v5 lesson: the explanation
    is only as good as the evidence, and the mechanism lives in the refutation
  - MAIA move probabilities at the player's Elo AND a stronger Elo — "a 1500 finds Kh4 31% of the
    time, a 2000 finds it 55%": tells the student whether this is a known blind spot or a rare miss.
    Runs locally via maia3_engine.py (ONNX) in the code package — no notebook needed.

Run locally (research account / Bedrock + the code package's maia3_engine on PYTHONPATH):
  python3 llm_explain_game.py --game /tmp/game_169764992210_full.json --player white \
    --player-elo 1518 --strong-elo 2200 --models haiku,sonnet \
    --out output/llm_explain_169764992210.json
"""
import argparse, json, time, sys, chess, boto3
from botocore.config import Config

# maia3_engine lives in the code package (ONNX move-probability model)
sys.path.insert(0, "/Users/samtkap/workspace/chess-deck/src/chess-deck-code/backend/mcp")
try:
    import maia3_engine as MAIA
except Exception as e:
    MAIA = None
    print(f"[warn] maia3_engine unavailable ({e}) — running without Maia move probabilities", flush=True)

REGION = "us-east-1"
MODELS = {
    "haiku":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "opus":   "us.anthropic.claude-opus-4-8",
}
# Bedrock on-demand $/1M tokens (us-east-1, 2026-06). in/out.
PRICE = {"haiku": (0.80, 4.0), "sonnet": (3.0, 15.0), "opus": (15.0, 75.0)}

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 2}))

SYSTEM = (
    "You are a chess coach reviewing a student's game. For each mistake you are given the board, "
    "Stockfish's evaluation and best lines, the engine's refutation of the move played, and Maia "
    "(human-move model) probabilities showing how often players at the student's level vs a stronger "
    "level choose each move. Write a coaching note that:\n"
    "1. Says concretely WHY the move is bad — name the squares, pieces, and the tactical or "
    "positional idea, grounded in the refutation line you're given. Do NOT invent tactics: if the "
    "engine shows no forcing punishment, the mistake is positional (a wasted tempo, a weakened "
    "square, a misplaced piece) — say that, do not claim a piece hangs when it doesn't.\n"
    "2. Gives the better move and the idea behind it, citing the engine's line.\n"
    "3. Uses the Maia numbers to calibrate: if stronger players find the right move much more often "
    "than the student's level does, say so ('this is a move stronger players spot but is a coin-flip "
    "at your level') — that tells the student whether it's a trainable blind spot or a rare slip.\n"
    "Two-to-four sentences, plain language, speak to the student ('you'). Be specific, not generic.")


def ascii_board(fen):
    return chess.Board(fen).unicode(borders=True, empty_square=".")


def line_san(fen, ucis):
    """Render an engine line (list of SAN or UCI) as SAN from the position, best-effort."""
    b = chess.Board(fen)
    out = []
    for mv in ucis:
        try:
            m = b.parse_san(mv)
        except Exception:
            try:
                m = chess.Move.from_uci(mv)
            except Exception:
                break
        if m not in b.legal_moves:
            break
        out.append(b.san(m)); b.push(m)
    return " ".join(out)


def fmt_eval(cp_str):
    """Stockfish white-POV eval string -> readable pawns/mate."""
    s = str(cp_str)
    if s.startswith("#") or "mate" in s.lower():
        return f"mate {s.lstrip('#')}"
    try:
        return f"{int(s)/100:+.1f}"
    except Exception:
        return s


def maia_block(fen, played_uci, player_elo, strong_elo):
    """Maia move probabilities at the player's Elo and a stronger Elo."""
    if MAIA is None:
        return ""
    try:
        lo = MAIA.analyze(fen, player_elo, player_elo, top_k=5, played_uci=played_uci)
        hi = MAIA.analyze(fen, strong_elo, strong_elo, top_k=5, played_uci=played_uci)
    except Exception:
        return ""
    def tm(r):
        return ", ".join(f"{s} {p*100:.0f}%" for s, p in r.get("top_moves", []))
    return (
        f"\nMaia (how humans actually move here):\n"
        f"  at your level (~{player_elo}): {tm(lo)}   [you played: {lo.get('played_prob',0)*100:.0f}% likely]\n"
        f"  at ~{strong_elo}: {tm(hi)}   [the move you played: {hi.get('played_prob',0)*100:.0f}% likely]")


def build_prompt(m, deep, player_elo, strong_elo, sae_feats=None):
    fen = m["fen"]
    b = chess.Board(fen)
    played_san = m.get("san") or b.san(chess.Move.from_uci(m["uci"]))
    stm = "White" if b.turn == chess.WHITE else "Black"
    best_san = m.get("best_san", "?")
    d = deep.get(m["ply"]) if "ply" in m else None
    parts = [
        f"Move {m['move_num']}. {stm} to play.",
        f"\n{ascii_board(fen)}\n",
        f"You played: {played_san}",
    ]
    if d and d.get("deep_eval_before") is not None:
        parts.append(f"Evaluation: {fmt_eval(d['deep_eval_before'])} before your move "
                     f"-> {fmt_eval(d['deep_eval_after'])} after (you lost {m['cp_loss']} centipawns).")
    else:
        parts.append(f"You lost {m['cp_loss']} centipawns.")
    parts.append(f"Engine's best move: {best_san}")
    if d and d.get("top_lines"):
        # ALL MultiPV lines, not just the top — alternatives are coaching-relevant
        lines = []
        for i, ln in enumerate(d["top_lines"][:3], 1):
            lines.append(f"  {i}. ({fmt_eval(ln['eval'])}) {line_san(fen, ln['moves'])}")
        parts.append("Engine's top lines (best alternatives):\n" + "\n".join(lines))
    if d and d.get("refutation"):
        # refutation starts with the played move; render what follows it
        parts.append(f"What punishes your move (engine refutation, {fmt_eval(d['refutation']['eval'])}): "
                     f"{line_san(fen, d['refutation']['moves'])}")
    mb = maia_block(fen, m["uci"], player_elo, strong_elo)
    if mb:
        parts.append(mb)
    if sae_feats:
        # SAE-augmented arm: inject the mistake-pattern detector's read as a HINT, not gospel
        feats = "\n".join(f"  - {f['category']}: {f['chip']}" for f in sae_feats[:4])
        parts.append("\nA mistake-pattern detector (trained on this player's games) flagged this "
                     "move as fitting these recurring patterns — use them only if they match what "
                     "you see on the board, ignore them otherwise:\n" + feats)
    parts.append("\nWrite the coaching note.")
    return "\n".join(parts)


def call(model_id, prompt):
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 400,
            "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}
    r = client.invoke_model(modelId=model_id, body=json.dumps(body))
    resp = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    u = resp.get("usage", {})
    return txt.strip(), u.get("input_tokens", 0), u.get("output_tokens", 0)


ap = argparse.ArgumentParser()
ap.add_argument("--game", required=True, help="analyze_cli.py game json")
ap.add_argument("--player", default="white", choices=["white", "black"])
ap.add_argument("--player-elo", type=int, default=1500, help="student's Elo (for Maia probabilities)")
ap.add_argument("--strong-elo", type=int, default=2200, help="reference strong Elo (for Maia comparison)")
ap.add_argument("--models", default="haiku,sonnet")
ap.add_argument("--classes", default="blunder,mistake")
ap.add_argument("--sae-features", default="", help="v7 k6 game output (game_v7_*.json) — adds an SAE-hint arm per model")
ap.add_argument("--out", required=True)
a = ap.parse_args()

g = json.load(open(a.game))
deep = {d["ply"]: d for d in g.get("deep", [])}
classes = set(a.classes.split(","))
models = [x.strip() for x in a.models.split(",")]

# SAE features keyed by move_num → list of {category, chip}; enables the "+sae" arm
sae_by_move = {}
if a.sae_features:
    for o in json.load(open(a.sae_features)):
        sae_by_move[o["move_num"]] = o.get("features", [])

# build the mistakes list (player's moves only)
mistakes = []
for m in g["moves"]:
    if m["side"] != a.player or m.get("classification") not in classes:
        continue
    b = chess.Board(m["fen"])
    # resolve a real best move from the deep top line (best_san is a shallow artifact — can equal played)
    d = deep.get(m["ply"])
    best_san = m.get("best_san", "")
    if d and d.get("top_lines") and d["top_lines"][0].get("moves"):
        best_san = d["top_lines"][0]["moves"][0]
    if best_san == m.get("san"):
        continue  # best == played → nothing to explain
    m = {**m, "best_san": best_san}
    mistakes.append(m)

# arms: each model runs plain; if SAE features given, also a "+sae" arm
arms = []
for mo in models:
    arms.append((mo, mo, False))
    if sae_by_move:
        arms.append((mo + "+sae", mo, True))

print(f"{len(mistakes)} {a.player} mistakes | arms: {[a0 for a0,_,_ in arms]}", flush=True)
results, cost = [], {a0: [0, 0] for a0, _, _ in arms}
for m in mistakes:
    feats = sae_by_move.get(m["move_num"]) if sae_by_move else None
    row = {"move_num": m["move_num"], "san": m["san"], "best_san": m["best_san"],
           "cp_loss": m["cp_loss"], "explanations": {}}
    for arm, mo, use_sae in arms:
        prompt = build_prompt(m, deep, a.player_elo, a.strong_elo, feats if use_sae else None)
        txt, ti, to = call(MODELS[mo], prompt)
        row["explanations"][arm] = txt
        cost[arm][0] += ti; cost[arm][1] += to
    results.append(row)
    print(f"  move {m['move_num']}. {m['san']} done", flush=True)

json.dump({"game": a.game, "player": a.player, "results": results}, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}\n")
print(f"{'arm':>11} {'in tok':>8} {'out tok':>8} {'$/game':>9}   $/1000 games")
print("-" * 54)
for arm, mo, _ in arms:
    ti, to = cost[arm]
    pin, pout = PRICE[mo]
    dollars = ti / 1e6 * pin + to / 1e6 * pout
    print(f"{arm:>11} {ti:>8} {to:>8} {dollars:>9.4f}   ${dollars*1000:>7.2f}")
