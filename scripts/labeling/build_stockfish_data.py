#!/usr/bin/env python3
"""Enrich chess positions with Stockfish analysis for Gemini labeling.

Accepts positions from either:
  - SAE feature profiles (--profiles): JSON with {feature_id: {examples: [{fen, uci}, ...]}}
  - Labels file (--labels): JSON with {feature_id: {examples: [{fen, uci}, ...]}}
  - Raw positions (--positions): JSON list of [{fen, uci}, ...]

Runs Stockfish depth 18 (configurable) with MultiPV=3, outputs keyed by "FEN|UCI":
  - fen, uci, best_uci, phase, side_to_move
  - played_san, best_san, is_check, is_capture
  - eval_before, eval_after, cp_loss
  - top_lines (3 best continuations as SAN)
  - refutation_lines (3 opponent responses as SAN)
  - threat (first move of refutation)

Usage:
    # From SAE profiles (top-10 per feature)
    python3 build_stockfish_data.py --profiles feature_profiles.json --top-n 10 -o stockfish_data.json

    # From raw position list
    python3 build_stockfish_data.py --positions positions.json -o stockfish_data.json

    # Resume interrupted run
    python3 build_stockfish_data.py --profiles profiles.json -o stockfish_data.json --resume
"""
import argparse
import chess
import chess.engine
import json
import os
import time
import sys
from multiprocessing import Pool

STOCKFISH_PATHS = [
    '/usr/games/stockfish',
    '/usr/bin/stockfish',
    '/usr/local/bin/stockfish',
    '/opt/homebrew/bin/stockfish',
    'stockfish',
]

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


def find_stockfish():
    for path in STOCKFISH_PATHS:
        if os.path.exists(path):
            return path
    if os.system('which stockfish >/dev/null 2>&1') == 0:
        return 'stockfish'
    print("ERROR: Stockfish not found. Install it or pass --stockfish path.", file=sys.stderr)
    sys.exit(1)


def load_positions(args):
    """Load positions from whichever source is specified."""
    if args.profiles:
        with open(args.profiles) as f:
            profiles = json.load(f)
        unique = {}
        for fid, prof in profiles.items():
            examples = prof.get('examples', [])[:args.top_n]
            for ex in examples:
                fen = ex.get('fen', '')
                uci = ex.get('uci', '')
                if fen and uci:
                    key = f"{fen}|{uci}"
                    if key not in unique:
                        unique[key] = (fen, uci, ex.get('best_uci', ''))
        print(f"Source: profiles ({len(profiles)} features, top-{args.top_n})")
        return unique

    elif args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        unique = {}
        for fid, feat in labels.items():
            for ex in feat.get('examples', [])[:args.top_n]:
                fen = ex.get('fen', '')
                uci = ex.get('uci', '')
                if fen and uci:
                    key = f"{fen}|{uci}"
                    if key not in unique:
                        unique[key] = (fen, uci, ex.get('best_uci', ''))
        print(f"Source: labels ({len(labels)} features, top-{args.top_n})")
        return unique

    elif args.positions:
        with open(args.positions) as f:
            positions = json.load(f)
        unique = {}
        for p in positions:
            fen = p.get('fen', '')
            uci = p.get('uci', p.get('blunder_uci', ''))
            if fen and uci:
                key = f"{fen}|{uci}"
                if key not in unique:
                    unique[key] = (fen, uci, p.get('best_uci', ''))
        print(f"Source: positions list ({len(positions)} entries)")
        return unique

    else:
        print("ERROR: Provide --profiles, --labels, or --positions", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich chess positions with Stockfish analysis for Gemini labeling.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--profiles', help='SAE feature profiles JSON')
    input_group.add_argument('--labels', help='Feature labels JSON with examples')
    input_group.add_argument('--positions', help='Raw positions JSON list [{fen, uci}, ...]')
    parser.add_argument('--top-n', type=int, default=10, help='Positions per feature (default 10)')
    parser.add_argument('-o', '--output', required=True, help='Output stockfish_data.json path')
    parser.add_argument('--depth', type=int, default=18, help='Stockfish depth (default 18)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel Stockfish engines (default 8)')
    parser.add_argument('--stockfish', help='Path to Stockfish binary')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output')
    args = parser.parse_args()

    sf_path = args.stockfish or find_stockfish()
    print(f"Stockfish: {sf_path}")

    unique = load_positions(args)
    print(f"Unique positions: {len(unique)}")

    # Resume support
    results = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resumed: {len(results)} already done")

    todo = [(k, v[0], v[1], v[2]) for k, v in unique.items() if k not in results]
    print(f"To analyze: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Workers: {args.workers}, Depth: {args.depth}")
    t0 = time.time()
    errors = 0
    done = 0

    with Pool(processes=args.workers,
              initializer=init_worker,
              initargs=(sf_path, args.depth)) as pool:

        for key, data in pool.imap_unordered(analyze_one, todo, chunksize=4):
            results[key] = data
            done += 1
            if 'error' in data:
                errors += 1

            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate
                print(f"  {done}/{len(todo)} ({rate:.1f}/s, ETA {eta/60:.0f}min, {errors} errors)", flush=True)
                with open(args.output, 'w') as f:
                    json.dump(results, f)

    with open(args.output, 'w') as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone. {len(results)} positions in {elapsed:.0f}s ({done/elapsed:.1f}/s, {errors} errors)")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
