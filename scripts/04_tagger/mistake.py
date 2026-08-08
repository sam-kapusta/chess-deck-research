#!/usr/bin/env python3
"""Layer 0 — the Mistake object. The single data contract every tagger layer reads.

Normalizes one blunder from the Stockfish analysis (stockfish_data_v2.json entry, OR an analyze_cli
deep entry) into a uniform dataclass. Pure python; no engine, no torch.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import chess


@dataclass
class Mistake:
    fen_before: str
    played_uci: str
    best_uci: str
    best_line_san: List[str]      # Stockfish PV from fen_before (what you SHOULD play)
    refutation_san: List[str]     # Stockfish punishment of the played move (from fen AFTER played)
    eval_before: Optional[int]    # white-POV centipawns; None if mate
    eval_after: Optional[int]
    cp_loss: int
    mover: bool                   # chess.WHITE / chess.BLACK — whose mistake
    player_elo: int = 1500
    oppo_elo: int = 1500
    played_san: str = ""
    best_san: str = ""
    n_good_moves: Optional[int] = None   # # moves within ~100cp of best at fen_before (MultiPV). None = unknown.
    # Parse caches (not constructor args, not compared, not printed) — see board_before below.
    _board_before_cache: Optional[chess.Board] = field(default=None, repr=False, compare=False)
    _board_after_cache: Optional[chess.Board] = field(default=None, repr=False, compare=False)

    # Cached FEN parses. These were plain @property, so EVERY access reparsed the FEN — measured 574
    # board_before parses for a 6-moment /tag-moments request, the single largest cost in a ~2.5s
    # response (one set_fen is ~200 from_symbol calls). fen_before/played_uci never change on a
    # Mistake, so the parse is pure and cacheable.
    #
    # Each access still returns a COPY: the tagger pushes moves on these boards, so handing out the
    # cached instance would let one caller corrupt every later reader. copy(stack=False) skips
    # duplicating the move history, which the tagger doesn't read on these.
    @property
    def board_before(self) -> chess.Board:
        if self._board_before_cache is None:
            self._board_before_cache = chess.Board(self.fen_before)
        return self._board_before_cache.copy(stack=False)

    @property
    def board_after(self) -> chess.Board:
        if self._board_after_cache is None:
            b = chess.Board(self.fen_before)
            b.push(chess.Move.from_uci(self.played_uci))
            self._board_after_cache = b
        return self._board_after_cache.copy(stack=False)

    @property
    def win_drop(self) -> float:
        """Win% the mover gave up (0-100, mover POV). The shared mistake-severity gate (issue #29),
        replacing the 8 magic cp_loss thresholds. Nonlinear in eval; falls back to cp_loss when
        signed evals are unavailable (mate, cp-only caches). Defined in chesslib_util.win_drop."""
        from chesslib_util import win_drop
        return win_drop(self.eval_before, self.eval_after, self.mover, self.cp_loss)


def _eval_to_cp(s):
    """'+253' -> 253 ; '#+5'/'#-3'/'mate' -> None (mate, handled separately)."""
    if s is None:
        return None
    s = str(s)
    if "#" in s or "mate" in s.lower():
        return None
    try:
        return int(s)
    except ValueError:
        return None


def from_sf_entry(fen, uci, e, player_elo=1500, oppo_elo=1500) -> Optional[Mistake]:
    """Build a Mistake from a stockfish_data_v2 entry `e` (keyed by f'{fen}|{uci}')."""
    if not isinstance(e, dict):
        return None
    b = chess.Board(fen)
    mover = b.turn
    best_uci = e.get("best_uci", "")
    best_line = (e.get("top_lines") or [{}])[0].get("moves", []) if e.get("top_lines") else []
    refut = (e.get("refutation_lines") or [{}])[0].get("moves", []) if e.get("refutation_lines") else []
    played_san = e.get("played_san", "")
    best_san = e.get("best_san", "")
    if not played_san:
        try: played_san = b.san(chess.Move.from_uci(uci))
        except Exception: played_san = uci
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=best_uci,
        best_line_san=best_line, refutation_san=refut,
        eval_before=_eval_to_cp(e.get("eval_before")), eval_after=_eval_to_cp(e.get("eval_after")),
        cp_loss=int(e.get("cp_loss", 0) or 0), mover=mover,
        player_elo=player_elo, oppo_elo=oppo_elo,
        played_san=played_san, best_san=best_san,
    )
