#!/usr/bin/env python3
"""Cache blunder positions from real Lichess games (not analysis board).

Uses Lichess/standard-chess-games dataset. Filters to games with [%eval] annotations
(~6% of games), finds moves where eval drops ≥200cp, extracts FEN + move.

No Stockfish needed — evals are pre-computed in the PGN.

Usage:
    # Small test
    python3 cache_real_game_blunders.py --n-positions 1000 --max-games 50000

    # Full 200K
    python3 cache_real_game_blunders.py --n-positions 200000
"""
import argparse
import json
import os
import re
import sys
import time

import chess
import chess.pgn

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
POSITIONS_FILE = BASE + '/cache/real_game_blunder_positions.json'


def parse_eval(comment):
    """Extract centipawn eval from PGN comment like '[%eval 2.35]' or '[%eval #-4]'."""
    m = re.search(r'\[%eval\s+([#\-\d.]+)\]', comment)
    if not m:
        return None
    val = m.group(1)
    if val.startswith('#'):
        # Mate score — convert to large cp value
        mate_in = int(val[1:])
        return 10000 * (1 if mate_in > 0 else -1)
    try:
        return int(float(val) * 100)  # convert to centipawns
    except ValueError:
        return None


def extract_blunders_from_game(movetext, min_loss=200, min_elo=1000, max_elo=2200):
    """Parse a PGN movetext with [%eval] annotations, find blunders."""
    # Quick check: does it have eval annotations?
    if '%eval' not in movetext:
        return []

    # Parse with python-chess
    import io
    pgn_str = f'[Event "?"]\n[Result "*"]\n\n{movetext}'
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if not game:
        return []

    blunders = []
    board = game.board()
    prev_eval = None
    ply = 0

    for node in game.mainline():
        move = node.move
        comment = node.comment or ''
        current_eval = parse_eval(comment)

        if prev_eval is not None and current_eval is not None:
            # The move was just played — check if it was a blunder
            # Eval is always from White's perspective
            is_white_move = (ply % 2 == 0)  # ply 0 = white's first move

            if is_white_move:
                # White just moved: if eval dropped, White blundered
                cp_loss = prev_eval - current_eval
            else:
                # Black just moved: if eval rose (from White's view), Black blundered
                cp_loss = current_eval - prev_eval

            if cp_loss >= min_loss:
                # Get the FEN BEFORE the move was played
                fen_before = board.fen()

                # Get the best move (we don't have it — use None, Stockfish would be needed)
                # But we know the played move was bad
                blunders.append({
                    'fen': fen_before,
                    'blunder_uci': move.uci(),
                    'cp_loss': cp_loss,
                    'eval_before': prev_eval,
                    'eval_after': current_eval,
                    'ply': ply,
                    'is_white': is_white_move,
                })

        board.push(move)
        prev_eval = current_eval
        ply += 1

    return blunders


def phase1_collect(n_positions, min_loss, max_games, min_elo, max_elo, positions_file):
    """Stream Lichess games, extract blunder positions."""
    # Resume from cache if exists
    positions = []
    if os.path.exists(positions_file):
        with open(positions_file) as f:
            positions = json.load(f)
        print(f'Loaded {len(positions)} cached positions from {positions_file}')
        if len(positions) >= n_positions:
            return positions[:n_positions]
        print(f'  Need {n_positions - len(positions)} more, resuming...')

    from datasets import load_dataset
    # Load recent years — higher eval annotation rate (8-9% vs 0.4% in 2013)
    data_files = [f'data/year={y}/month={m:02d}/*.parquet'
                  for y in [2025, 2024, 2023] for m in range(12, 0, -1)]
    ds = load_dataset('Lichess/standard-chess-games', split='train',
                      data_files=data_files, streaming=True)

    print(f'Streaming Lichess games, collecting {n_positions} blunders with ≥{min_loss}cp loss...')
    print(f'  Elo range: {min_elo}-{max_elo}')

    n_games = 0
    n_with_eval = 0
    n_found = len(positions)
    t0 = time.time()
    last_print = t0

    for row in ds:
        n_games += 1
        if max_games and n_games > max_games:
            break

        # Filter by Elo
        w_elo = row.get('WhiteElo', 0)
        b_elo = row.get('BlackElo', 0)
        if not w_elo or not b_elo:
            continue
        if w_elo < min_elo or w_elo > max_elo or b_elo < min_elo or b_elo > max_elo:
            continue

        movetext = row.get('movetext', '')
        if '%eval' not in movetext:
            continue

        n_with_eval += 1
        blunders = extract_blunders_from_game(movetext, min_loss=min_loss)

        for b in blunders:
            b['white_elo'] = w_elo
            b['black_elo'] = b_elo
            b['time_control'] = row.get('TimeControl', '')
            positions.append(b)
            n_found += 1

        # Progress
        now = time.time()
        if now - last_print > 10 or n_found >= n_positions:
            elapsed = now - t0
            rate = n_found / elapsed if elapsed > 0 else 0
            eta = (n_positions - n_found) / rate if rate > 0 else 0
            eval_pct = 100 * n_with_eval / n_games if n_games > 0 else 0
            print(f'  {n_found}/{n_positions} blunders | {n_games} games ({eval_pct:.1f}% have eval) | '
                  f'{rate:.0f}/sec | ETA {eta:.0f}s', flush=True)
            last_print = now

            # Save checkpoint every 10K
            if n_found > 0 and n_found % 10000 < len(blunders):
                with open(positions_file, 'w') as f:
                    json.dump(positions, f)

        if n_found >= n_positions:
            break

    # Final save
    with open(positions_file, 'w') as f:
        json.dump(positions[:n_positions], f)
    elapsed = time.time() - t0
    print(f'\nDone: {min(n_found, n_positions)} blunders from {n_games} games '
          f'({n_with_eval} had eval, {100*n_with_eval/n_games:.1f}%) in {elapsed:.0f}s')

    # Print distribution
    import chess as chess_lib
    phase_counts = {'opening': 0, 'middlegame': 0, 'endgame': 0}
    for p in positions[:n_positions]:
        try:
            board = chess_lib.Board(p['fen'])
            pc = len(board.piece_map())
            if pc > 24: phase_counts['opening'] += 1
            elif pc > 12: phase_counts['middlegame'] += 1
            else: phase_counts['endgame'] += 1
        except:
            pass
    total = sum(phase_counts.values())
    print(f'\nPhase distribution:')
    for phase, n in sorted(phase_counts.items(), key=lambda x: -x[1]):
        print(f'  {phase}: {n} ({100*n/total:.0f}%)')

    return positions[:n_positions]


def main():
    parser = argparse.ArgumentParser(description='Cache blunders from real Lichess games')
    parser.add_argument('--n-positions', type=int, default=200000)
    parser.add_argument('--min-loss', type=int, default=200, help='Min cp loss to count as blunder')
    parser.add_argument('--max-games', type=int, default=None, help='Max games to scan (for testing)')
    parser.add_argument('--min-elo', type=int, default=1000)
    parser.add_argument('--max-elo', type=int, default=2200)
    parser.add_argument('--output', default=POSITIONS_FILE)
    args = parser.parse_args()

    positions = phase1_collect(
        args.n_positions, args.min_loss, args.max_games,
        args.min_elo, args.max_elo, args.output,
    )
    print(f'\nSaved {len(positions)} positions to {args.output}')


if __name__ == '__main__':
    main()
