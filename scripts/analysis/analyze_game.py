#!/usr/bin/env python3
"""Analyze a chess game with Stockfish.

Usage:
    analyze_game.py <url-or-pgn-path> [--depth 18] [--output game_analysis.json]

Supports:
    - Chess.com game URLs (live or daily)
    - Lichess game URLs
    - Local PGN files

Output: JSON array of moves with evals, classifications, best moves.
"""
import argparse
import chess
import chess.engine
import json
import os
import re
import sys
import urllib.request


def fetch_pgn_chesscom(url):
    """Fetch PGN from Chess.com game URL."""
    # Extract game ID
    match = re.search(r'/game/(?:live|daily)/(\d+)', url)
    if not match:
        raise ValueError(f"Can't extract game ID from: {url}")
    game_id = match.group(1)

    # Try to find in player's archive
    # First get player from the callback API
    api_url = f'https://www.chess.com/callback/live/game/{game_id}'
    req = urllib.request.Request(api_url, headers={'User-Agent': 'ChessDeck/1.0'})
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        players = data.get('players', {})
        white = players.get('top', {}).get('username') or players.get('bottom', {}).get('username')
        if not white:
            # Try different structure
            game = data.get('game', {})
            white = game.get('pgnHeaders', {}).get('White', '')
    except:
        white = None

    if not white:
        raise ValueError(f"Could not determine player for game {game_id}")

    # Search archives for this game
    archives_url = f'https://api.chess.com/pub/player/{white.lower()}/games/archives'
    req = urllib.request.Request(archives_url, headers={'User-Agent': 'ChessDeck/1.0'})
    resp = urllib.request.urlopen(req)
    archives = json.loads(resp.read())['archives']

    for archive_url in reversed(archives[-6:]):  # Check last 6 months
        req = urllib.request.Request(archive_url, headers={'User-Agent': 'ChessDeck/1.0'})
        resp = urllib.request.urlopen(req)
        games = json.loads(resp.read())['games']
        for g in games:
            if game_id in g.get('url', ''):
                return g['pgn']

    raise ValueError(f"Game {game_id} not found in {white}'s archives")


def fetch_pgn_lichess(url):
    """Fetch PGN from Lichess game URL."""
    match = re.search(r'lichess\.org/(\w+)', url)
    if not match:
        raise ValueError(f"Can't extract game ID from: {url}")
    game_id = match.group(1)[:8]  # Lichess IDs are 8 chars

    api_url = f'https://lichess.org/game/export/{game_id}?clocks=false&evals=false'
    req = urllib.request.Request(api_url, headers={'Accept': 'application/x-chess-pgn'})
    resp = urllib.request.urlopen(req)
    return resp.read().decode()


def fetch_pgn(source):
    """Fetch PGN from URL or file."""
    if os.path.exists(source):
        with open(source) as f:
            return f.read()
    elif 'chess.com' in source:
        return fetch_pgn_chesscom(source)
    elif 'lichess.org' in source:
        return fetch_pgn_lichess(source)
    else:
        raise ValueError(f"Unknown source: {source}. Provide a Chess.com URL, Lichess URL, or PGN file path.")


def parse_moves(pgn_text):
    """Extract SAN moves from PGN text."""
    # Split header from moves
    parts = re.split(r'\n\n', pgn_text, maxsplit=1)
    move_text = parts[1] if len(parts) > 1 else parts[0]

    # Strip comments, clock annotations, NAGs
    move_text = re.sub(r'\{[^}]*\}', '', move_text)
    move_text = re.sub(r'\$\d+', '', move_text)
    move_text = re.sub(r'\d+\.\.\.', '', move_text)

    tokens = move_text.split()
    moves = [t for t in tokens
             if not t[0].isdigit() and t not in ('0-1', '1-0', '1/2-1/2', '*', '')]
    return moves


def find_stockfish():
    """Find Stockfish binary."""
    candidates = [
        '/opt/homebrew/bin/stockfish',
        '/usr/local/bin/stockfish',
        '/usr/bin/stockfish',
        'stockfish',
    ]
    for path in candidates:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.quit()
            return path
        except:
            continue
    raise RuntimeError("Stockfish not found. Install with: brew install stockfish")


def analyze(pgn_text, depth=18, stockfish_path=None):
    """Analyze all moves in a PGN. Returns list of move dicts."""
    san_moves = parse_moves(pgn_text)
    if not san_moves:
        raise ValueError("No moves found in PGN")

    sf_path = stockfish_path or find_stockfish()
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 4, "Hash": 256})

    board = chess.Board()
    positions = []

    for i, san in enumerate(san_moves):
        fen_before = board.fen()
        try:
            move = board.parse_san(san)
        except Exception as e:
            print(f"Failed to parse move '{san}' at ply {i+1}: {e}", file=sys.stderr)
            break

        uci = move.uci()
        ply = i + 1
        is_white = (ply % 2 == 1)

        # Eval BEFORE the move
        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_before = info_before["score"].white()
        cp_before_white = score_before.score(mate_score=10000) or 0

        best_move = info_before.get("pv", [None])[0]
        best_uci = best_move.uci() if best_move else uci

        # From mover's perspective
        cp_before_mover = cp_before_white if is_white else -cp_before_white

        # Push the move
        board.push(move)

        # Eval AFTER the move
        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_after = info_after["score"].white()
        cp_after_white = score_after.score(mate_score=10000) or 0

        # From mover's perspective
        cp_after_mover = cp_after_white if is_white else -cp_after_white

        # cp_loss = how much the mover's eval dropped
        cp_loss = max(0, cp_before_mover - cp_after_mover)

        # Classify
        if cp_loss >= 200:
            classification = 'blunder'
        elif cp_loss >= 100:
            classification = 'mistake'
        elif cp_loss >= 50:
            classification = 'inaccuracy'
        else:
            classification = 'good'

        positions.append({
            'ply': ply,
            'san': san,
            'uci': uci,
            'best_uci': best_uci,
            'fen': fen_before,
            'side': 'white' if is_white else 'black',
            'cp_before_white': cp_before_white,
            'cp_after_white': cp_after_white,
            'cp_before_mover': cp_before_mover,
            'cp_after_mover': cp_after_mover,
            'cp_loss': cp_loss,
            'classification': classification,
        })

    engine.quit()
    return positions


def main():
    parser = argparse.ArgumentParser(description='Analyze a chess game with Stockfish')
    parser.add_argument('source', help='Chess.com URL, Lichess URL, or PGN file path')
    parser.add_argument('--depth', type=int, default=18, help='Stockfish depth (default: 18)')
    parser.add_argument('--output', '-o', default='/tmp/game_analysis.json', help='Output JSON path')
    parser.add_argument('--stockfish', help='Path to Stockfish binary')
    args = parser.parse_args()

    print(f'Fetching PGN from {args.source}...')
    pgn = fetch_pgn(args.source)

    # Save PGN
    pgn_path = args.output.replace('.json', '.pgn')
    with open(pgn_path, 'w') as f:
        f.write(pgn)
    print(f'PGN saved to {pgn_path}')

    print(f'Analyzing at depth {args.depth}...')
    positions = analyze(pgn, depth=args.depth, stockfish_path=args.stockfish)

    with open(args.output, 'w') as f:
        json.dump(positions, f, indent=2)

    # Print summary
    n_moves = len(positions)
    blunders = [p for p in positions if p['classification'] == 'blunder']
    mistakes = [p for p in positions if p['classification'] == 'mistake']
    inaccuracies = [p for p in positions if p['classification'] == 'inaccuracy']

    print(f'\n{n_moves} moves analyzed → {args.output}')
    print(f'  Blunders: {len(blunders)}, Mistakes: {len(mistakes)}, Inaccuracies: {len(inaccuracies)}')
    print()

    for p in positions:
        marker = ''
        if p['classification'] == 'blunder': marker = ' *** BLUNDER ***'
        elif p['classification'] == 'mistake': marker = ' ** MISTAKE **'
        elif p['classification'] == 'inaccuracy': marker = ' * inacc *'

        print(f"  {p['ply']:>2}. {p['san']:<8} ({p['side']}) "
              f"eval={p['cp_before_mover']:>6} → {p['cp_after_mover']:>6}  "
              f"loss={p['cp_loss']:>4}{marker}")
        if p['classification'] in ('blunder', 'mistake'):
            print(f"      played={p['uci']}  best={p['best_uci']}")


if __name__ == '__main__':
    main()
