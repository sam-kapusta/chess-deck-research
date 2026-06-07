#!/usr/bin/env python3
"""Layer 3 — Maia rarity annotations (not tags, numeric coaching context).

Uses maia3_engine (ONNX, offline) from the chess-deck-code package. For a mistake:
  - how likely is the played (wrong) move at the player's Elo  -> "common vs rare blunder for your level"
  - how likely at a higher Elo (+400)                          -> "stronger players blunder X% less here"
  - does the BEST move's Maia-prob rise with Elo               -> "skill-gap move" (strong players find it)

Returns a dict of numeric annotations; the renderer/report turns them into prose. Degrades to {} if
maia3_engine is unavailable (so the rest of the tagger never depends on it).
"""
import sys, os

_MAIA = None
def _engine():
    global _MAIA
    if _MAIA is None:
        sys.path.insert(0, "/Users/samtkap/workspace/chess-deck/src/chess-deck-code/backend/mcp")
        try:
            import maia3_engine
            _MAIA = maia3_engine
        except Exception:
            _MAIA = False
    return _MAIA or None


def rarity(m, strong_delta=400):
    eng = _engine()
    if eng is None:
        return {}
    lo = m.player_elo
    hi = m.player_elo + strong_delta
    try:
        r_lo = eng.analyze(m.fen_before, lo, m.oppo_elo, top_k=3, played_uci=m.played_uci)
        r_hi = eng.analyze(m.fen_before, hi, hi, top_k=3, played_uci=m.played_uci)
    except Exception:
        return {}
    # best move's prob at each elo (skill-gap signal)
    best_lo = _prob_of(r_lo, m)
    best_hi = _prob_of(r_hi, m)
    out = {
        "played_prob_at_level": round(r_lo.get("played_prob", 0), 4),
        "played_prob_strong": round(r_hi.get("played_prob", 0), 4),
        "best_prob_at_level": round(best_lo, 4),
        "best_prob_strong": round(best_hi, 4),
        "strong_elo": hi,
    }
    # derived flags
    out["common_blunder"] = out["played_prob_at_level"] >= 0.25       # many at your level play it
    out["rare_blunder"] = out["played_prob_at_level"] < 0.08
    out["skill_gap_move"] = (best_hi - best_lo) >= 0.20               # strong players find the best move much more
    return out


def _prob_of(result, m):
    """Find the best move's probability in a maia result's top_moves (by SAN match)."""
    import chess
    b = m.board_before
    try:
        best_san = b.san(chess.Move.from_uci(m.best_uci)) if m.best_uci else m.best_san
    except Exception:
        best_san = m.best_san
    for san, p in result.get("top_moves", []):
        if san == best_san:
            return p
    return 0.0
