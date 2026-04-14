#!/usr/bin/env python3
"""Analyze a full game: Stockfish eval + Maia SAE features for every move.

Usage:
    python3 analyze_game_sae.py "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6" --output game.json
    python3 analyze_game_sae.py --pgn game.pgn --output game.json

Produces a JSON file with per-move: eval, best move, classification, SAE features, position type.
Then prints a coaching summary.
"""
import argparse, json, sys, os
import chess
import chess.engine
import chess.pgn
import io

# Add backend to path for position_features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'shared'))

SAE_AVAILABLE = False
try:
    from position_features import init_models, get_position_features
    SAE_AVAILABLE = init_models()
    if SAE_AVAILABLE:
        print("SAE loaded", flush=True)
except:
    print("SAE not available — will skip position features", flush=True)

STOCKFISH = "/opt/homebrew/bin/stockfish"


def analyze_game(moves_san, player_color="white", player_elo=1800, depth=16):
    """Analyze every move of a game."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 4, "Hash": 256})

    board = chess.Board()
    results = []

    for i, san in enumerate(moves_san):
        ply = i + 1
        is_white = (ply % 2 == 1)
        move_num = (ply + 1) // 2

        # Pre-move analysis
        fen_before = board.fen()
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)

        # Eval
        score = info[0]['score'].white()
        if score.is_mate():
            eval_cp = 10000 * (1 if score.mate() > 0 else -1)
        else:
            eval_cp = score.score() or 0

        # Best move
        best_move = info[0].get('pv', [None])[0]
        best_san = board.san(best_move) if best_move else None

        # SAE features
        sae_features = []
        if SAE_AVAILABLE:
            try:
                feats = get_position_features(fen_before, player_elo=player_elo, max_features=5)
                sae_features = [f['label'] for f in feats]
            except:
                pass

        # Make the move
        try:
            move = board.parse_san(san)
        except:
            print(f"  Invalid move at ply {ply}: {san}")
            break

        # Post-move eval
        board.push(move)
        post_info = engine.analyse(board, chess.engine.Limit(depth=depth))
        post_score = post_info['score'].white()
        if post_score.is_mate():
            eval_after = 10000 * (1 if post_score.mate() > 0 else -1)
        else:
            eval_after = post_score.score() or 0

        # Classification (based on eval loss)
        if is_white:
            eval_loss = eval_cp - eval_after
        else:
            eval_loss = eval_after - eval_cp

        if eval_loss > 300:
            cls = "blunder"
        elif eval_loss > 100:
            cls = "mistake"
        elif eval_loss > 50:
            cls = "inaccuracy"
        elif eval_loss < -50:
            cls = "great"
        else:
            cls = "normal"

        label = f"{move_num}. {san}" if is_white else f"{move_num}...{san}"

        results.append({
            "ply": ply,
            "move": san,
            "label": label,
            "is_white": is_white,
            "fen_before": fen_before,
            "eval_before": round(eval_cp / 100, 2),
            "eval_after": round(eval_after / 100, 2),
            "eval_loss": round(eval_loss / 100, 2),
            "best_move": best_san,
            "played_best": (san == best_san),
            "classification": cls,
            "sae_features": sae_features,
        })

        if (ply) % 10 == 0:
            print(f"  Analyzed {ply} moves...", flush=True)

    engine.quit()
    return results


def print_summary(results, player_color="white"):
    """Print a coaching-oriented game summary."""
    is_player = lambda r: (r['is_white'] and player_color == 'white') or (not r['is_white'] and player_color == 'black')

    player_moves = [r for r in results if is_player(r)]
    blunders = [r for r in player_moves if r['classification'] == 'blunder']
    mistakes = [r for r in player_moves if r['classification'] == 'mistake']
    inaccuracies = [r for r in player_moves if r['classification'] == 'inaccuracy']
    great = [r for r in player_moves if r['classification'] == 'great']

    print("\n" + "=" * 60)
    print("GAME SUMMARY")
    print("=" * 60)
    print(f"Total moves: {len(results)} ({len(player_moves)} by {player_color})")
    print(f"Blunders: {len(blunders)}, Mistakes: {len(mistakes)}, Inaccuracies: {len(inaccuracies)}, Great: {len(great)}")

    # Accuracy
    best_count = sum(1 for r in player_moves if r['played_best'])
    print(f"Best moves played: {best_count}/{len(player_moves)} ({100*best_count/max(len(player_moves),1):.0f}%)")

    # Key moments with SAE context
    print("\nKEY MOMENTS:")
    for r in player_moves:
        if r['classification'] in ('blunder', 'mistake'):
            sae_str = ', '.join(r['sae_features'][:2]) if r['sae_features'] else 'no SAE data'
            print(f"  {r['label']} ({r['classification']}) — eval {r['eval_before']:+.1f} → {r['eval_after']:+.1f}")
            print(f"    Best was: {r['best_move']}")
            print(f"    Position type: {sae_str}")
            print()

    # SAE feature summary across all moves
    from collections import Counter
    all_features = Counter()
    for r in player_moves:
        for f in r['sae_features']:
            all_features[f] += 1

    if all_features:
        print("POSITION TYPES THROUGHOUT THE GAME:")
        for feat, count in all_features.most_common(10):
            pct = 100 * count / len(player_moves)
            print(f"  {pct:4.0f}% of moves: {feat}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('moves', nargs='?', help='Moves in SAN (e.g., "1.e4 e5 2.Nf3 Nc6")')
    parser.add_argument('--pgn', help='PGN file path')
    parser.add_argument('--color', default='white', choices=['white', 'black'])
    parser.add_argument('--elo', type=int, default=1800)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--output', default='game_analysis.json')
    args = parser.parse_args()

    if args.pgn:
        with open(args.pgn) as f:
            game = chess.pgn.read_game(f)
        moves = [move.san() for move in game.mainline()]
    elif args.moves:
        # Parse "1.e4 e5 2.Nf3 Nc6" format
        import re
        clean = re.sub(r'\d+\.+', '', args.moves).split()
        moves = [m for m in clean if m and m != '..']
    else:
        print("Provide moves or --pgn")
        return

    print(f"Analyzing {len(moves)} moves (depth {args.depth})...", flush=True)
    results = analyze_game(moves, args.color, args.elo, args.depth)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    print_summary(results, args.color)


if __name__ == '__main__':
    main()
