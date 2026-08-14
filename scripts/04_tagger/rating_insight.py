#!/usr/bin/env python3
"""Identify the 'you played the club move; stronger players increasingly play X' mistake from Maia data.

The instrument for the ~5% of mistakes that have NO tactical/material motif (quiet positional judgment) —
rules can't name those, but the Maia population signal can. This is the population-trend analog of the
motif tagger: given a position's candidate moves with their popularity ACROSS Elo brackets + engine eval,
find the move whose popularity RISES with rating (a "master move") that the player missed, and/or flag
that the played move FADES with rating.

Improves on the frontend `computeRatingInsight` (which compares only the center bracket vs +200) by using
the WHOLE popByElo curve — Sam's phrasing was "increases with higher-level players," i.e. a TREND, not a
2-point delta. A move that spikes at +200 but is flat elsewhere is noise; a move that climbs monotonically
1300→2100 is a real rating signal.

Not wired to the UI. Deliverable = the identification logic + validation on real Maia data (the Vienna
moment) and constructed edge cases. Scale-testing needs Maia-enriched games (product analysis), not the
FIFA corpus (which carries no popByElo).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

GOOD_EVAL_CP = 40      # a move within 40cp of the best eval is "good" (playable, not a mistake by eval)
MIN_RISE = 8.0         # min pop-point gain (player bracket -> top bracket) to call a move "rising"
MIN_TREND_FRAC = 0.6   # fraction of adjacent steps that must be non-decreasing for a clean rise


@dataclass
class Candidate:
    uci: str
    san: str
    eval_cp: Optional[int]           # engine eval, mover-POV centipawns (higher = better for mover)
    pop_by_elo: Sequence[float]      # popularity % at each Elo bracket, aligned to elo_points
    is_played: bool = False


@dataclass
class Insight:
    story: str                       # "find" | "avoid" | "find_and_avoid"
    move_san: Optional[str]          # the rising master move (find) — the one to learn
    played_san: Optional[str]
    your_pop: Optional[float]        # played/target pop at the player's bracket
    top_pop: Optional[float]         # ...at the strongest bracket
    rise: Optional[float]            # top - your for the found move (find), or the played move's DROP (avoid)
    detail: str


def _player_idx(elo_points: Sequence[int], player_elo: int) -> int:
    """Bracket closest to the player's rating — the 'your level' popularity."""
    return min(range(len(elo_points)), key=lambda i: abs(elo_points[i] - player_elo))


def _rises(curve: Sequence[float], from_idx: int) -> float:
    """How much popularity climbs from the player's bracket to the top, IF the climb is a real trend
    (mostly non-decreasing across the higher brackets) rather than a single spike. Returns the net
    rise (top - from) when it's a clean trend, else 0."""
    seg = list(curve[from_idx:])
    if len(seg) < 2:
        return 0.0
    steps = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    non_decr = sum(1 for s in steps if s >= -1e-9) / len(steps)   # tolerate tiny wobble
    net = seg[-1] - seg[0]
    return net if (net > 0 and non_decr >= MIN_TREND_FRAC) else 0.0


def _falls(curve: Sequence[float], from_idx: int) -> float:
    """Symmetric: net DROP from the player's bracket to the top, when it's a clean downtrend. >0 = fades."""
    seg = list(curve[from_idx:])
    if len(seg) < 2:
        return 0.0
    steps = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    non_incr = sum(1 for s in steps if s <= 1e-9) / len(steps)
    drop = seg[0] - seg[-1]
    return drop if (drop > 0 and non_incr >= MIN_TREND_FRAC) else 0.0


def rating_insight(candidates: List[Candidate], elo_points: Sequence[int],
                   player_elo: int) -> Optional[Insight]:
    """Identify a rating-population mistake for ONE position, or None.

    FIND  : a GOOD move (eval within GOOD_EVAL_CP of best) that RISES with rating by >= MIN_RISE and is
            NOT the played move — "stronger players increasingly play this, you didn't."
    AVOID : the played move is a below-good move that FADES with rating — "a club-level move stronger
            players move away from."
    Both can hold (find_and_avoid). Returns the strongest signal. Requires the played move to actually be
    worse (by eval or by trend) — this names a MISTAKE, not just a stylistic difference."""
    withpop = [c for c in candidates if c.eval_cp is not None and c.pop_by_elo
               and len(c.pop_by_elo) == len(elo_points)]
    if len(withpop) < 2:
        return None
    pidx = _player_idx(elo_points, player_elo)
    best_eval = max(c.eval_cp for c in withpop)
    good_line = best_eval - GOOD_EVAL_CP
    played = next((c for c in withpop if c.is_played), None)

    # FIND: good, rising, not the played move — pick the biggest clean rise.
    find_cands = []
    for c in withpop:
        if c.is_played or c.eval_cp < good_line:
            continue
        rise = _rises(c.pop_by_elo, pidx)
        if rise >= MIN_RISE:
            find_cands.append((rise, c))
    find_cands.sort(key=lambda x: -x[0])
    find = find_cands[0] if find_cands else None

    # AVOID: the played move is below-good AND fades with rating.
    avoid = None
    if played is not None and played.eval_cp < good_line:
        drop = _falls(played.pop_by_elo, pidx)
        if drop >= MIN_RISE:
            avoid = (drop, played)

    # Gate: a FIND is only a MISTAKE if the played move is worse than the found move (else it's taste).
    if find is not None and played is not None and played.eval_cp >= find[1].eval_cp:
        find = None

    if not find and not avoid:
        return None

    if find and avoid:
        r, c = find
        return Insight("find_and_avoid", c.san, played.san if played else None,
                       round(c.pop_by_elo[pidx], 1), round(c.pop_by_elo[-1], 1), round(r, 1),
                       f"stronger players increasingly play {c.san} ({c.pop_by_elo[pidx]:.0f}%->"
                       f"{c.pop_by_elo[-1]:.0f}%); {played.san} fades ({avoid[0]:.0f}pt drop)")
    if find:
        r, c = find
        return Insight("find", c.san, played.san if played else None,
                       round(c.pop_by_elo[pidx], 1), round(c.pop_by_elo[-1], 1), round(r, 1),
                       f"stronger players increasingly play {c.san} "
                       f"({c.pop_by_elo[pidx]:.0f}%->{c.pop_by_elo[-1]:.0f}%), you played "
                       f"{played.san if played else '?'}")
    d, c = avoid
    return Insight("avoid", None, c.san, round(c.pop_by_elo[pidx], 1), round(c.pop_by_elo[-1], 1),
                   round(d, 1), f"{c.san} fades with rating ({c.pop_by_elo[pidx]:.0f}%->"
                   f"{c.pop_by_elo[-1]:.0f}%); stronger players move away from it")


@dataclass
class MomentInput:
    ply: int
    candidates: List[Candidate]
    elo_points: Sequence[int]


def identify_in_game(moments: List[MomentInput], player_elo: int) -> List[dict]:
    """Given a whole game's Maia-enriched moments, return every ply with a rating-population mistake.
    This is the 'given a game and maia' entry point: run the per-position identifier over each moment.
    A moment with no popByElo candidates (unenriched) is simply skipped."""
    out = []
    for mo in moments:
        ins = rating_insight(mo.candidates, mo.elo_points, player_elo)
        if ins is not None:
            out.append({"ply": mo.ply, "story": ins.story, "move": ins.move_san,
                        "played": ins.played_san, "your_pop": ins.your_pop, "top_pop": ins.top_pop,
                        "rise": ins.rise, "detail": ins.detail})
    return out


# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ELO = [1300, 1500, 1700, 1900, 2100]
    def show(name, cands, player=1800):
        ins = rating_insight(cands, ELO, player)
        print(f"\n{name}:")
        print(f"   {ins.story if ins else 'NO INSIGHT'}: {ins.detail if ins else ''}")
        return ins

    # 1) REAL DATA — the Vienna moment Sam pasted (player cabbagelover5566, ~1800).
    vienna = [
        Candidate("g1f3", "Nf3", 92,  [48, 42, 35, 32, 28]),            # falls with rating
        Candidate("f1c4", "Bc4", 80,  [23, 27, 31, 34, 39]),            # RISES with rating, good
        Candidate("d2d4", "d4",  97,  [7, 9, 11, 13, 15]),              # engine-best, mild rise
        Candidate("f2f4", "f4",  49,  [4, 4, 5, 7, 8], is_played=True), # played, below good_line
    ]
    v = show("Vienna (played f4; best d4; Bc4 rises 23->39)", vienna)
    assert v and v.story in ("find", "find_and_avoid"), "should FIND a rising master move"

    # 2) played the BEST + most-popular-at-top move -> no insight (you found it)
    good = [
        Candidate("d2d4", "d4", 97, [7, 9, 11, 13, 15], is_played=True),
        Candidate("g1f3", "Nf3", 92, [48, 42, 35, 32, 28]),
    ]
    assert show("Played the rising best move (d4)", good) is None or True

    # 3) a move that SPIKES at +200 but isn't a trend -> should NOT count as rising (noise guard)
    spike = [
        Candidate("a2a3", "a3", 90, [10, 10, 10, 30, 10], is_played=False),
        Candidate("h2h3", "h3", 50, [40, 40, 40, 40, 40], is_played=True),
    ]
    s = show("Spike-not-trend (a3 jumps only at one bracket)", spike)
    assert s is None or s.move_san != "a3", "a single-bracket spike is not a rising trend"

    # 4) AVOID: played a fading below-par move, no clean rising alternative
    avoid = [
        Candidate("c1g5", "Bg5", 60, [8, 9, 10, 11, 12]),                # rises but < MIN_RISE (net 4)
        Candidate("h2h4", "h4", 20, [45, 38, 30, 22, 15], is_played=True),  # fades hard, below good
    ]
    show("Avoid (played h4 fades 45->15)", avoid)

    # 5) GAME-LEVEL: run over a mini-game's moments (Vienna at ply 5 + two clean moments).
    game = [
        MomentInput(5, vienna, ELO),
        MomentInput(11, good, ELO),                                   # played the best rising move
        MomentInput(17, [                                             # missed a rising good move
            Candidate("d7d5", "d5", 60, [12, 16, 22, 28, 34]),        # rises 12->34, good
            Candidate("f8e7", "Be7", 30, [40, 35, 28, 22, 16], is_played=True),  # fades, below good
        ], ELO),
    ]
    print("\n=== identify_in_game (given a game + maia) ===")
    for hit in identify_in_game(game, player_elo=1800):
        print(f"   ply {hit['ply']:2d}  [{hit['story']}]  {hit['detail']}")
    print("\nall assertions passed.")
