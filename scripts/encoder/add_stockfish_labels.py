#!/usr/bin/env python3
"""Add Stockfish best move + PV to positions that are missing them.

Reads stage_a_positions.jsonl, runs Stockfish on positions without best_move,
writes back with best_move, pv_line, and alternatives filled in.

Usage:
  python add_stockfish_labels.py
  python add_stockfish_labels.py --depth 16 --threads 4
"""
import json
import subprocess
import chess
import time
from pathlib import Path

INPUT = Path("research/data/stage_a_positions.jsonl")
OUTPUT = Path("research/data/stage_a_positions_labeled.jsonl")
STOCKFISH = "/opt/homebrew/bin/stockfish"


class StockfishEngine:
    """Simple Stockfish wrapper for batch analysis."""

    def __init__(self, path=STOCKFISH, depth=8, threads=4, hash_mb=256):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait_for("readyok")
        self.depth = depth

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline().strip()
            if token in line:
                return line

    def analyze(self, fen, num_moves=3):
        """Analyze a position. Returns best_move, eval, pv_line, alternatives."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")

        best_move = None
        eval_cp = None
        pv_line = ""
        lines = []

        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                best_uci = line.split()[1]
                break
            if "score cp" in line and f"depth {self.depth}" in line:
                parts = line.split()
                try:
                    cp_idx = parts.index("cp") + 1
                    eval_cp = int(parts[cp_idx])
                    pv_idx = parts.index("pv") + 1
                    pv_moves = parts[pv_idx:]
                    pv_line = " ".join(pv_moves[:6])  # First 6 moves of PV
                except (ValueError, IndexError):
                    pass
            elif "score mate" in line and f"depth {self.depth}" in line:
                parts = line.split()
                try:
                    mate_idx = parts.index("mate") + 1
                    mate_in = int(parts[mate_idx])
                    eval_cp = 10000 * (1 if mate_in > 0 else -1)
                    pv_idx = parts.index("pv") + 1
                    pv_moves = parts[pv_idx:]
                    pv_line = " ".join(pv_moves[:6])
                except (ValueError, IndexError):
                    pass

        # Convert UCI best move to SAN
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_uci)
            best_san = board.san(move)

            # Convert PV from UCI to SAN
            pv_san = []
            pv_board = board.copy()
            for uci_move in pv_line.split():
                try:
                    m = chess.Move.from_uci(uci_move)
                    pv_san.append(pv_board.san(m))
                    pv_board.push(m)
                except:
                    break
            pv_line_san = " ".join(pv_san)

        except:
            best_san = best_uci
            pv_line_san = pv_line

        eval_pawns = eval_cp / 100.0 if eval_cp is not None else 0

        return {
            "best_move": best_san,
            "eval_sf": round(eval_pawns, 2),
            "pv_line": pv_line_san,
        }

    def close(self):
        self._send("quit")
        self.proc.wait()


def main():
    data = [json.loads(l) for l in INPUT.read_text().strip().split("\n")]
    need_analysis = [p for p in data if not p.get("best_move")]
    have_analysis = [p for p in data if p.get("best_move")]

    print(f"Total positions: {len(data)}")
    print(f"Already have best_move: {len(have_analysis)}")
    print(f"Need Stockfish analysis: {len(need_analysis)}")

    if not need_analysis:
        print("Nothing to analyze!")
        return

    engine = StockfishEngine(depth=16, threads=4)
    t0 = time.time()

    for i, pos in enumerate(need_analysis):
        try:
            result = engine.analyze(pos["fen"])
            pos["best_move"] = result["best_move"]
            pos["pv_line"] = result["pv_line"]
            # Use Stockfish eval if we don't have one, or if ours was from the game (played move eval)
            if not pos.get("eval") or pos["source"] == "all_moves":
                pos["eval_sf"] = result["eval_sf"]
            else:
                pos["eval_sf"] = result["eval_sf"]
        except Exception as e:
            pos["best_move"] = ""
            pos["pv_line"] = ""

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(need_analysis) - i - 1) / rate
            print(f"  {i+1}/{len(need_analysis)} | {rate:.1f} pos/s | ETA: {remaining:.0f}s")

    engine.close()
    elapsed = time.time() - t0
    print(f"\nAnalyzed {len(need_analysis)} positions in {elapsed:.0f}s ({len(need_analysis)/elapsed:.1f} pos/s)")

    # Combine
    all_labeled = have_analysis + need_analysis
    with_best = [p for p in all_labeled if p.get("best_move")]
    print(f"Total with best_move: {len(with_best)}/{len(all_labeled)}")

    # Save
    with open(OUTPUT, "w") as f:
        for p in all_labeled:
            f.write(json.dumps(p) + "\n")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
