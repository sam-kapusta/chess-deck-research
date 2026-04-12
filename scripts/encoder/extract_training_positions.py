#!/usr/bin/env python3
"""Extract training positions with Stockfish labels from analyzed games.

Sources:
- moments (key positions): fen, eval, bestMove, classification, alternatives, tags
- all_moves (every position): fen (derived from PGN), eval, classification

Output: research/data/stage_a_positions.jsonl

Usage:
  python extract_training_positions.py
  python extract_training_positions.py --moments-only
"""
import json
import glob
import random
import chess
import chess.pgn
import io
from pathlib import Path
from collections import Counter

GAMES_DIR = Path.home() / ".chess-coach" / "games"
OUTPUT = Path("research/data/stage_a_positions.jsonl")


def derive_fen_from_pgn(pgn_text, target_ply):
    """Get FEN at a specific ply from PGN."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return None
        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            if i + 1 == target_ply:
                return board.fen()
    except:
        pass
    return None


def get_phase(fen):
    """Classify position as opening/middlegame/endgame."""
    try:
        board = chess.Board(fen)
        pieces = len(board.piece_map())
        if pieces > 24:
            return "opening"
        elif pieces > 10:
            return "middlegame"
        return "endgame"
    except:
        return "unknown"


def extract_from_game(game_data):
    """Extract all usable positions from one game."""
    positions = []
    pgn = game_data.get("pgn", "")
    game_id = game_data.get("game_id", "")
    opening = game_data.get("opening", "")
    player_color = game_data.get("player", {}).get("color", "")

    # Extract from moments (key positions with bestMove + alternatives)
    for m in game_data.get("moments", []):
        fen = m.get("fen", "")
        if not fen:
            continue

        pos = {
            "fen": fen,
            "eval": m.get("eval", 0),
            "best_move": m.get("bestMove", ""),
            "played_move": m.get("san", ""),
            "classification": m.get("classification", ""),
            "alternatives": [{"move": a.get("san", ""), "eval": a.get("eval", 0)} for a in m.get("alternatives", [])],
            "tags": m.get("tags", []),
            "ply": m.get("ply", 0),
            "phase": get_phase(fen),
            "is_key_moment": True,
            "game_id": game_id,
            "opening": opening,
            "source": "moment",
        }
        positions.append(pos)

    # Extract from all_moves (every position with eval)
    all_moves = game_data.get("all_moves", [])
    for mv in all_moves:
        ply = mv.get("ply", 0)
        ev = mv.get("eval", 0)
        san = mv.get("san", "")
        cls = mv.get("classification", "")

        # Derive FEN from PGN
        fen = derive_fen_from_pgn(pgn, ply)
        if not fen:
            continue

        # Skip if we already have this position from moments
        if any(p["fen"] == fen for p in positions):
            continue

        pos = {
            "fen": fen,
            "eval": ev,
            "best_move": "",  # all_moves doesn't have bestMove — need Stockfish
            "played_move": san,
            "classification": cls,
            "alternatives": [],
            "tags": [],
            "ply": ply,
            "phase": get_phase(fen),
            "is_key_moment": cls in ("blunder", "mistake"),
            "game_id": game_id,
            "opening": opening,
            "source": "all_moves",
        }
        positions.append(pos)

    return positions


def main():
    game_files = sorted(glob.glob(str(GAMES_DIR / "chesscom-*.json")))
    print(f"Found {len(game_files)} games")

    all_positions = []
    failed = 0

    for i, gf in enumerate(game_files):
        try:
            game = json.loads(Path(gf).read_text())
            positions = extract_from_game(game)
            all_positions.extend(positions)
        except Exception as e:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(game_files)} games ({len(all_positions)} positions)")

    print(f"\nProcessed {len(game_files)} games ({failed} failed)")
    print(f"Total positions: {len(all_positions)}")

    # Stats
    from_moments = [p for p in all_positions if p["source"] == "moment"]
    from_moves = [p for p in all_positions if p["source"] == "all_moves"]
    key = [p for p in all_positions if p["is_key_moment"]]
    with_best = [p for p in all_positions if p["best_move"]]

    print(f"\nFrom moments (have bestMove): {len(from_moments)}")
    print(f"From all_moves (no bestMove): {len(from_moves)}")
    print(f"Key moments: {len(key)}")
    print(f"With best_move: {len(with_best)}")
    print(f"Phase: {Counter(p['phase'] for p in all_positions)}")
    print(f"Classification: {Counter(p['classification'] for p in all_positions if p['classification'])}")

    # Deduplicate by FEN
    seen_fens = set()
    unique = []
    for p in all_positions:
        fen_key = p["fen"].split(" ")[0]  # Position only
        if fen_key not in seen_fens:
            seen_fens.add(fen_key)
            unique.append(p)

    print(f"\nAfter FEN dedup: {len(unique)} (removed {len(all_positions) - len(unique)})")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    random.shuffle(unique)
    with open(OUTPUT, "w") as f:
        for p in unique:
            f.write(json.dumps(p) + "\n")

    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
