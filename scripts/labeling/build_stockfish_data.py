#!/usr/bin/env python3
"""Step 1: Compute Stockfish data for all unique positions across 512 features.

Reads the full 512-feature labels file, extracts unique FEN+UCI pairs,
runs Stockfish depth 18 on each using multiple parallel engines, and writes stockfish_data.json.

Output keyed by "FEN|UCI" with:
  - fen, uci, best_uci, phase, side_to_move
  - played_san, best_san, is_check, is_capture
  - eval_before, eval_after, cp_loss
  - best_continuation (SAN), refutation_line (SAN)
  - threat (first move of refutation)

Usage:
    python3 build_stockfish_data.py [--positions 20] [--depth 18] [--workers 16]
"""
import argparse
import chess
import chess.engine
import json
import time
import sys
from multiprocessing import Pool, Manager

LABELS_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/labels_512_k8_realgames_v4.json'
OUTPUT_PATH = '/Users/samtkap/workspace/chess-deck/src/chess-deck-research/output/stockfish_data.json'
STOCKFISH = '/opt/homebrew/bin/stockfish'

# Global per-worker engine
_engine = None
_depth = 18


def init_worker(stockfish_path, depth):
    """Each worker process starts its own Stockfish engine."""
    global _engine, _depth
    _depth = depth
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


def cleanup_worker():
    global _engine
    if _engine:
        _engine.quit()


def analyze_one(args):
    """Analyze a single position. Called by pool workers."""
    key, fen, uci, best_uci = args
    global _engine, _depth

    try:
        board = chess.Board(fen)
        side = 'Black' if not board.turn else 'White'
        pc = len(board.piece_map())
        phase = 'endgame' if pc <= 12 else ('middlegame' if pc <= 24 else 'opening')

        played = chess.Move.from_uci(uci)
        played_san = board.san(played)
        is_check = board.gives_check(played)
        is_capture = board.is_capture(played)

        # Eval before + top 3 lines (MultiPV=3)
        results_before = _engine.analyse(board, chess.engine.Limit(depth=_depth), multipv=3)
        r1 = results_before[0]
        eval_before = str(r1['score'].white())
        best_move = r1['pv'][0]
        best_san = board.san(best_move)

        # Extract top 3 lines as SAN
        top_lines = []
        for r in results_before:
            b_copy = board.copy()
            line_san = []
            for m in r['pv'][:8]:
                try:
                    line_san.append(b_copy.san(m))
                    b_copy.push(m)
                except:
                    break
            top_lines.append({
                'moves': line_san,
                'eval': str(r['score'].white()),
            })

        # Eval after played move + refutation (also MultiPV=3)
        board.push(played)
        results_after = _engine.analyse(board, chess.engine.Limit(depth=_depth), multipv=3)
        r2 = results_after[0]
        eval_after = str(r2['score'].white())

        refutation_lines = []
        for r in results_after:
            b_copy = board.copy()
            line_san = []
            for m in r['pv'][:8]:
                try:
                    line_san.append(b_copy.san(m))
                    b_copy.push(m)
                except:
                    break
            refutation_lines.append({
                'moves': line_san,
                'eval': str(r['score'].white()),
            })

        threat = refutation_lines[0]['moves'][0] if refutation_lines and refutation_lines[0]['moves'] else ''
        board.pop()

        # CP loss
        s1 = r1['score'].white().score(mate_score=10000)
        s2 = r2['score'].white().score(mate_score=10000)
        cp_loss = abs(s1 - s2) if s1 is not None and s2 is not None else 0

        return key, {
            'fen': fen,
            'uci': uci,
            'best_uci': best_move.uci(),
            'side_to_move': side,
            'phase': phase,
            'played_san': played_san,
            'best_san': best_san,
            'is_check': is_check,
            'is_capture': is_capture,
            'eval_before': eval_before,
            'eval_after': eval_after,
            'cp_loss': cp_loss,
            'top_lines': top_lines,
            'refutation_lines': refutation_lines,
            'threat': threat,
        }
    except Exception as e:
        return key, {'fen': fen, 'uci': uci, 'error': str(e)[:100]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', type=int, default=20, help='Positions per feature (default 20)')
    parser.add_argument('--depth', type=int, default=18, help='Stockfish depth (default 18)')
    parser.add_argument('--workers', type=int, default=16, help='Parallel Stockfish engines (default 16)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output')
    args = parser.parse_args()

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    # Collect unique positions
    unique = {}
    for fid, feat in labels.items():
        for ex in feat['examples'][:args.positions]:
            key = f"{ex['fen']}|{ex['uci']}"
            if key not in unique:
                unique[key] = (ex['fen'], ex['uci'], ex.get('best_uci', ''))

    print(f"Features: {len(labels)}", flush=True)
    print(f"Unique positions: {len(unique)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    # Resume support
    results = {}
    if args.resume:
        try:
            with open(OUTPUT_PATH) as f:
                results = json.load(f)
            print(f"Resumed: {len(results)} already done", flush=True)
        except:
            pass

    todo = [(k, v[0], v[1], v[2]) for k, v in unique.items() if k not in results]
    print(f"To analyze: {len(todo)}", flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        return

    t0 = time.time()
    errors = 0
    done = 0

    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(STOCKFISH, args.depth)) as pool:

        for key, data in pool.imap_unordered(analyze_one, todo, chunksize=4):
            results[key] = data
            done += 1
            if 'error' in data:
                errors += 1

            if done % 200 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate
                print(f"  {done}/{len(todo)} ({rate:.1f}/s, ETA {eta/60:.1f}min, {errors} errors)", flush=True)
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(results, f)

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(results)} positions in {elapsed:.0f}s ({errors} errors)", flush=True)
    print(f"Saved to {OUTPUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
