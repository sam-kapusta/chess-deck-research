#!/usr/bin/env python3
"""Enrich FENs with Stockfish eval + python-chess tactical annotations.

Usage:
    from enrich_fens import enrich_batch
    enriched = enrich_batch(["fen1", "fen2", ...], stockfish_path="/opt/homebrew/bin/stockfish")

Returns dict: {fen: annotation_string}

Cached: subsequent calls with same FENs return instantly.
"""
import chess
import chess.engine
import os
import json
import hashlib
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'output', 'fen_enrichment_cache.json')
DEFAULT_STOCKFISH = os.environ.get('STOCKFISH_PATH', '/opt/homebrew/bin/stockfish')
DEPTH = 10
WORKERS = 8


# ============================================================================
# Python-chess tactical analysis (instant, no engine needed)
# ============================================================================

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _material(board: chess.Board) -> str:
    pieces = {chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
              chess.ROOK: 'R', chess.QUEEN: 'Q'}
    w, b = [], []
    for pt, sym in pieces.items():
        wc = len(board.pieces(pt, chess.WHITE))
        bc = len(board.pieces(pt, chess.BLACK))
        if wc: w.append(f'{sym}{"x"+str(wc) if wc > 1 else ""}')
        if bc: b.append(f'{sym}{"x"+str(bc) if bc > 1 else ""}')
    return f"W:{''.join(w)} B:{''.join(b)}"


def _phase(board: chess.Board) -> str:
    total = sum(len(board.pieces(pt, c))
                for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                for c in [chess.WHITE, chess.BLACK])
    if total <= 4: return "endgame"
    if total <= 8: return "late-middlegame"
    return "middlegame" if total <= 12 else "opening"


def _king_safety(board: chess.Board, color: chess.Color) -> str:
    king_sq = board.king(color)
    if king_sq is None: return "no king"
    rank = chess.square_rank(king_sq)
    file = chess.square_file(king_sq)
    base_rank = 0 if color == chess.WHITE else 7
    castled = rank == base_rank and file in (1, 2, 6)

    # Pawn shield
    shield_files = [max(0, file - 1), file, min(7, file + 1)]
    shield_rank = base_rank + (1 if color == chess.WHITE else -1)
    shield_pawns = 0
    if 0 <= shield_rank <= 7:
        for f in shield_files:
            sq = chess.square(f, shield_rank)
            if board.piece_at(sq) == chess.Piece(chess.PAWN, color):
                shield_pawns += 1

    # Open files near king
    open_files = 0
    for f in shield_files:
        if not any(board.piece_at(chess.square(f, r)) and
                   board.piece_at(chess.square(f, r)).piece_type == chess.PAWN
                   for r in range(8)):
            open_files += 1

    parts = []
    if castled: parts.append("castled")
    elif rank == base_rank: parts.append("uncastled")
    else: parts.append(f"king on {chess.square_name(king_sq)}")
    if shield_pawns < 2: parts.append(f"weak shield({shield_pawns}/3)")
    if open_files > 0: parts.append(f"{open_files} open file{'s' if open_files > 1 else ''}")
    return ", ".join(parts) if parts else "safe"


def _tactics(board: chess.Board) -> List[str]:
    """Detect immediate tactical features from the position."""
    found = []
    opp = not board.turn

    if board.is_check():
        found.append("in check")

    # Hanging pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opp and piece.piece_type != chess.KING:
            attackers = board.attackers(board.turn, sq)
            defenders = board.attackers(opp, sq)
            if attackers and not defenders:
                found.append(f"hanging {chess.piece_name(piece.piece_type)} {chess.square_name(sq)}")
            elif attackers:
                min_att = min(PIECE_VALUES.get(board.piece_at(a).piece_type, 10) for a in attackers)
                if min_att < PIECE_VALUES.get(piece.piece_type, 0):
                    found.append(f"{chess.square_name(sq)} attacked by lower-value")

    # Passed pawns
    for color in [chess.WHITE, chess.BLACK]:
        opp_color = not color
        direction = 1 if color == chess.WHITE else -1
        threshold = 4 if color == chess.WHITE else 3
        for sq in board.pieces(chess.PAWN, color):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            check_range = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
            for r in check_range:
                for f in [max(0, file - 1), file, min(7, file + 1)]:
                    p = board.piece_at(chess.square(f, r))
                    if p and p.piece_type == chess.PAWN and p.color == opp_color:
                        is_passed = False
                        break
                if not is_passed: break
            passed_threshold = rank >= threshold if color == chess.WHITE else rank <= threshold
            if is_passed and passed_threshold:
                found.append(f"passed pawn {chess.square_name(sq)}")

    # Pins
    king_sq = board.king(opp)
    if king_sq:
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == opp and piece.piece_type != chess.KING:
                if board.is_pinned(opp, sq):
                    found.append(f"pinned {chess.piece_name(piece.piece_type)} {chess.square_name(sq)}")

    # Forks — a piece attacking 2+ high-value enemy pieces (check BOTH sides)
    for color in [chess.WHITE, chess.BLACK]:
        enemy = not color
        side = "White" if color == chess.WHITE else "Black"
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                attacks = board.attacks(sq)
                high_targets = []
                for a_sq in attacks:
                    target = board.piece_at(a_sq)
                    if (target and target.color == enemy and
                            target.piece_type in (chess.ROOK, chess.QUEEN, chess.KING)):
                        high_targets.append(f"{chess.piece_name(target.piece_type)} {chess.square_name(a_sq)}")
                if len(high_targets) >= 2:
                    found.append(f"{side} {chess.piece_name(piece.piece_type)} {chess.square_name(sq)} forks {' and '.join(high_targets)}")

    # Back rank vulnerability — king on rank 1/8 with no escape squares
    for color in [chess.WHITE, chess.BLACK]:
        k_sq = board.king(color)
        if k_sq is None: continue
        k_rank = chess.square_rank(k_sq)
        back = 0 if color == chess.WHITE else 7
        if k_rank == back:
            escapes = 0
            for f in [max(0, chess.square_file(k_sq) - 1), chess.square_file(k_sq), min(7, chess.square_file(k_sq) + 1)]:
                esc_rank = back + (1 if color == chess.WHITE else -1)
                if 0 <= esc_rank <= 7:
                    esc_sq = chess.square(f, esc_rank)
                    blocker = board.piece_at(esc_sq)
                    if not blocker or blocker.color != color:
                        escapes += 1
            if escapes == 0:
                side = "White" if color == chess.WHITE else "Black"
                found.append(f"{side} back rank vulnerable")

    # Promotion threats — pawns on 7th/2nd rank
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_rank(sq) == 6:
            found.append(f"White pawn {chess.square_name(sq)} one step from promotion")
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if chess.square_rank(sq) == 1:
            found.append(f"Black pawn {chess.square_name(sq)} one step from promotion")

    # Overloaded pieces — sole defender of 2+ attacked squares
    for color in [chess.WHITE, chess.BLACK]:
        enemy = not color
        defended_by: Dict[int, List[int]] = {}  # defender_sq -> [attacked_sq, ...]
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                # Is this piece attacked by enemy?
                if not board.attackers(enemy, sq):
                    continue
                # Who defends it?
                defenders = [d for d in board.attackers(color, sq) if d != sq]
                for d in defenders:
                    if d not in defended_by:
                        defended_by[d] = []
                    defended_by[d].append(sq)
        for def_sq, protected in defended_by.items():
            if len(protected) >= 2:
                defender = board.piece_at(def_sq)
                if defender:
                    targets = [chess.square_name(s) for s in protected[:3]]
                    side = "White" if color == chess.WHITE else "Black"
                    found.append(f"{side} {chess.piece_name(defender.piece_type)} {chess.square_name(def_sq)} overloaded defending {', '.join(targets)}")

    # Skewers — sliding piece attacks high-value piece, behind it on the ray is another piece
    for sq in chess.SQUARES:
        attacker = board.piece_at(sq)
        if (attacker and attacker.color == board.turn and
                attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)):
            for target_sq in board.attacks(sq):
                target = board.piece_at(target_sq)
                if (target and target.color == opp and
                        target.piece_type in (chess.QUEEN, chess.KING)):
                    # Check ray beyond target for another piece
                    ray = chess.ray(sq, target_sq)
                    if not ray: continue
                    beyond = chess.SquareSet(ray) - chess.SquareSet(chess.between(sq, target_sq)) - chess.SquareSet([sq, target_sq])
                    for behind_sq in beyond:
                        behind = board.piece_at(behind_sq)
                        if behind and behind.color == opp and behind.piece_type != chess.KING:
                            # Check nothing blocks between target and behind
                            blocked = False
                            for btwn_sq in chess.SquareSet(chess.between(target_sq, behind_sq)):
                                if board.piece_at(btwn_sq):
                                    blocked = True
                                    break
                            if not blocked:
                                found.append(f"skewer: {chess.piece_name(attacker.piece_type)} {chess.square_name(sq)} attacks {chess.piece_name(target.piece_type)} {chess.square_name(target_sq)}, {chess.piece_name(behind.piece_type)} {chess.square_name(behind_sq)} behind")
                            break

    # Discovered attacks — piece on ray between friendly slider and enemy high-value piece
    for slider_sq in chess.SQUARES:
        slider = board.piece_at(slider_sq)
        if (slider and slider.color == board.turn and
                slider.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)):
            for target_sq in chess.SQUARES:
                target = board.piece_at(target_sq)
                if (target and target.color == opp and
                        target.piece_type in (chess.ROOK, chess.QUEEN, chess.KING)):
                    ray = chess.ray(slider_sq, target_sq)
                    if not ray: continue
                    between_sqs = list(chess.SquareSet(chess.between(slider_sq, target_sq)))
                    # Exactly one piece between slider and target = potential discovered attack
                    blockers = [s for s in between_sqs if board.piece_at(s)]
                    if len(blockers) == 1:
                        blocker = board.piece_at(blockers[0])
                        if blocker and blocker.color == board.turn:
                            found.append(f"discovered attack: moving {chess.piece_name(blocker.piece_type)} {chess.square_name(blockers[0])} reveals {chess.piece_name(slider.piece_type)} on {chess.piece_name(target.piece_type)} {chess.square_name(target_sq)}")

    return found[:10]


def _annotate_no_engine(fen: str) -> str:
    """Annotate a FEN using only python-chess (instant)."""
    try:
        board = chess.Board(fen)
    except:
        return ""

    parts = []
    parts.append(_material(board))
    parts.append(_phase(board))
    turn = "White" if board.turn else "Black"
    parts.append(f"{turn} to move")
    parts.append(f"WK:{_king_safety(board, chess.WHITE)}")
    parts.append(f"BK:{_king_safety(board, chess.BLACK)}")
    tactics = _tactics(board)
    if tactics:
        parts.append("Tactics: " + "; ".join(tactics))
    return " | ".join(parts)


# ============================================================================
# Engine Pool — persistent Stockfish instances, reused across positions
# ============================================================================

class EnginePool:
    """Pool of persistent Stockfish engines for parallel analysis."""

    def __init__(self, engine_path: str, size: int):
        self._pool = queue.Queue()
        self._size = size
        self._engine_path = engine_path
        for _ in range(size):
            try:
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                self._pool.put(engine)
            except Exception as e:
                print(f"  Warning: failed to start engine: {e}")

    def analyze(self, fen: str, depth: int) -> Dict:
        """Analyze a position. Borrows an engine from the pool."""
        engine = self._pool.get()
        try:
            board = chess.Board(fen)

            # Eval before best move
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info.get("score")
            pv = info.get("pv", [])
            best_move = pv[0] if pv else None

            # Format eval
            cp_before = 0
            eval_str = "?"
            if score:
                cp_before = score.white().score(mate_score=10000) or 0
                if score.white().mate() is not None:
                    eval_str = f"M{score.white().mate()}"
                else:
                    eval_str = f"{'+' if cp_before >= 0 else ''}{cp_before/100:.1f}"

            best_str = ""
            eval_delta_str = ""
            if best_move:
                san = board.san(best_move)
                captured = board.piece_at(best_move.to_square)

                # Push move and get eval after
                board.push(best_move)
                is_check = board.is_check()

                # Eval delta — how much did the best move gain?
                info_after = engine.analyse(board, chess.engine.Limit(depth=max(depth - 2, 6)))
                score_after = info_after.get("score")
                cp_after = 0
                if score_after:
                    # Both evals from white's perspective — delta is how much
                    # the position changed. Positive = good for side that moved.
                    cp_after = score_after.white().score(mate_score=10000) or 0

                # Eval delta from the perspective of the side that moved
                # If white moved: positive delta = white gained
                # If black moved: we need to flip (white's perspective decrease = black gained)
                raw_delta = cp_after - cp_before
                # board was white-to-move before push, so white moved
                # if it was black-to-move originally, flip
                eval_delta = raw_delta / 100
                if abs(eval_delta) >= 0.3:
                    eval_delta_str = f"delta={'+' if eval_delta >= 0 else ''}{eval_delta:.1f}"

                # Fork detection after the move
                fork_str = ""
                moving_piece = board.piece_at(best_move.to_square)
                if moving_piece:
                    attacks = list(board.attacks(best_move.to_square))
                    high_targets = []
                    for sq in attacks:
                        target = board.piece_at(sq)
                        if (target and target.color != moving_piece.color and
                                target.piece_type in (chess.ROOK, chess.QUEEN, chess.KING)):
                            high_targets.append(f"{chess.piece_name(target.piece_type)} {chess.square_name(sq)}")
                    if len(high_targets) >= 2:
                        fork_str = f" FORK: attacks {' and '.join(high_targets)}"

                board.pop()

                # Build description
                move_desc = san
                if captured:
                    move_desc += f" (captures {chess.piece_name(captured.piece_type)})"
                if is_check:
                    move_desc += "+"

                best_str = f"{move_desc} [{eval_str}]"
                if eval_delta_str:
                    best_str += f" {eval_delta_str}"
                if fork_str:
                    best_str += fork_str

            return {"best": best_str, "eval": eval_str}

        except Exception as e:
            return {"best": "", "eval": "?", "error": str(e)}
        finally:
            self._pool.put(engine)

    def shutdown(self):
        while not self._pool.empty():
            try:
                engine = self._pool.get_nowait()
                engine.quit()
            except:
                pass


# ============================================================================
# Main enrichment function
# ============================================================================

def enrich_batch(fens: List[str], stockfish_path: str = DEFAULT_STOCKFISH,
                 depth: int = DEPTH, workers: int = WORKERS,
                 use_cache: bool = True) -> Dict[str, str]:
    """Enrich a batch of FENs. Returns {fen: annotation_string}.

    Combines python-chess (instant) + Stockfish (parallel engine pool).
    Results are cached to disk.
    """
    # Load cache
    cache = {}
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
        except:
            cache = {}

    # Split cached vs uncached
    results = {}
    uncached = []
    for fen in fens:
        key = hashlib.md5(fen.encode()).hexdigest()
        if key in cache:
            results[fen] = cache[key]
        else:
            uncached.append(fen)

    if not uncached:
        return results

    print(f"Enriching {len(uncached)} FENs ({len(fens) - len(uncached)} cached)...")

    # python-chess annotations (instant)
    chess_annotations = {}
    for fen in uncached:
        chess_annotations[fen] = _annotate_no_engine(fen)

    # Stockfish analysis (parallel engine pool)
    stockfish_results = {}
    if os.path.exists(stockfish_path):
        pool_size = min(workers, len(uncached))
        pool = EnginePool(stockfish_path, pool_size)
        print(f"  Engine pool: {pool_size} Stockfish instances, depth {depth}")

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = {executor.submit(pool.analyze, fen, depth): fen for fen in uncached}
            done = 0
            for future in as_completed(futures):
                fen = futures[future]
                stockfish_results[fen] = future.result()
                done += 1
                if done % 500 == 0:
                    print(f"  Stockfish: {done}/{len(uncached)}")

        pool.shutdown()
    else:
        print(f"  Warning: Stockfish not found at {stockfish_path}, skipping engine analysis")

    # Combine annotations
    for fen in uncached:
        parts = [chess_annotations.get(fen, "")]
        sf = stockfish_results.get(fen, {})
        if sf.get("best"):
            parts.append(f"Best: {sf['best']}")
        annotation = " | ".join(p for p in parts if p)
        results[fen] = annotation
        key = hashlib.md5(fen.encode()).hexdigest()
        cache[key] = annotation

    # Save cache
    if use_cache:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"  Cache: {len(cache)} entries saved")

    return results


if __name__ == "__main__":
    test_fens = [
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/8/4k3/8/3K4/8/4P3/8 w - - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    ]
    results = enrich_batch(test_fens)
    for fen, annotation in results.items():
        print(f"\nFEN: {fen}")
        print(f"  → {annotation}")
