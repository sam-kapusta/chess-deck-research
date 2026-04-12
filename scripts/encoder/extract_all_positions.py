#!/usr/bin/env python3
"""Extract ALL training positions from local + DynamoDB games.

Handles both formats:
- Local MCP games (chesscom-*.json): moments[] + all_moves[]
- DynamoDB games (chesscom:*.json): allMoves[] with bestMoveSan + eval

Output: research/data/stage_a_all_positions.jsonl
"""
import json
import glob
import random
import chess
from pathlib import Path
from collections import Counter

GAMES_DIR = Path.home() / ".chess-coach" / "games"
OUTPUT = Path("research/data/stage_a_all_positions.jsonl")


def get_phase(fen):
    try:
        board = chess.Board(fen)
        pieces = len(board.piece_map())
        if pieces > 24: return "opening"
        elif pieces > 10: return "middlegame"
        return "endgame"
    except:
        return "unknown"


def extract_dynamo_game(game_data):
    """Extract from DynamoDB format (allMoves with bestMoveSan)."""
    positions = []
    game_id = game_data.get("gameOriginId", "")
    opening = game_data.get("opening", "")

    for mv in game_data.get("allMoves", []):
        fen = mv.get("fen_before", "")
        if not fen:
            continue

        best_move = mv.get("bestMoveSan", mv.get("bestMove", ""))
        ev = mv.get("eval", 0)
        if isinstance(ev, dict):
            ev = ev.get("value", 0)
        ev = float(ev) if ev else 0

        pos = {
            "fen": fen,
            "eval": round(ev, 2),
            "best_move": best_move,
            "played_move": mv.get("san", ""),
            "classification": mv.get("classification", ""),
            "alternatives": [],
            "pv_line": "",
            "phase": get_phase(fen),
            "is_key_moment": mv.get("classification", "") in ("blunder", "mistake"),
            "game_id": game_id,
            "opening": opening,
            "source": "dynamo",
        }
        positions.append(pos)

    return positions


def extract_local_game(game_data):
    """Extract from local MCP format (moments + all_moves)."""
    positions = []
    game_id = game_data.get("game_id", "")
    opening = game_data.get("opening", "")

    # From moments (have bestMove + alternatives)
    for m in game_data.get("moments", []):
        fen = m.get("fen", "")
        if not fen:
            continue
        pos = {
            "fen": fen,
            "eval": float(m.get("eval", 0)),
            "best_move": m.get("bestMove", ""),
            "played_move": m.get("san", ""),
            "classification": m.get("classification", ""),
            "alternatives": [{"move": a.get("san", ""), "eval": float(a.get("eval", 0))} for a in m.get("alternatives", [])],
            "pv_line": "",
            "phase": get_phase(fen),
            "is_key_moment": True,
            "game_id": game_id,
            "opening": opening,
            "source": "local_moment",
        }
        positions.append(pos)

    return positions


def main():
    # Find all game files
    local_files = sorted(glob.glob(str(GAMES_DIR / "chesscom-*.json")))
    dynamo_files = sorted(glob.glob(str(GAMES_DIR / "chesscom:*.json")))
    print(f"Local games: {len(local_files)}")
    print(f"DynamoDB games: {len(dynamo_files)}")

    all_positions = []

    # Extract from DynamoDB games (have bestMoveSan on every move)
    for i, gf in enumerate(dynamo_files):
        try:
            game = json.loads(Path(gf).read_text())
            positions = extract_dynamo_game(game)
            all_positions.extend(positions)
        except:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  DynamoDB: {i+1}/{len(dynamo_files)} ({len(all_positions)} positions)")

    dynamo_count = len(all_positions)
    print(f"DynamoDB positions: {dynamo_count}")

    # Extract from local games (moments only — have bestMove)
    for gf in local_files:
        try:
            game = json.loads(Path(gf).read_text())
            positions = extract_local_game(game)
            all_positions.extend(positions)
        except:
            pass

    local_count = len(all_positions) - dynamo_count
    print(f"Local positions: {local_count}")
    print(f"Total: {len(all_positions)}")

    # Filter: must have best_move and fen
    with_best = [p for p in all_positions if p.get("best_move") and p.get("fen")]
    print(f"With best_move: {len(with_best)}")

    # Deduplicate by FEN
    seen = set()
    unique = []
    for p in with_best:
        fen_key = p["fen"].split(" ")[0]
        if fen_key not in seen:
            seen.add(fen_key)
            unique.append(p)

    print(f"After FEN dedup: {len(unique)}")
    print(f"Phase: {Counter(p['phase'] for p in unique)}")
    print(f"Classification: {Counter(p['classification'] for p in unique if p['classification'])}")
    print(f"Key moments: {sum(1 for p in unique if p['is_key_moment'])}")

    # Save
    random.seed(42)
    random.shuffle(unique)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in unique:
            f.write(json.dumps(p) + "\n")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
