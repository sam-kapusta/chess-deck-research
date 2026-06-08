#!/usr/bin/env python3
"""Directional validation: do the rule detectors AGREE with the SAE on the SAE's own confirmed
positions? The SAE gold (relabel_v9_d64_k1.json) is hand-confirmed by Sam; all_feat_boards_d64_k1.json
gives each feature's firing positions (fen, played uci, best uci, activation). For features that map
to a motif our detectors implement, we run the detector on those positions and report agreement.

This is the "not completely wrong" gate. We do NOT expect 100% (the SAE is polysemantic — that's why
we're replacing it), but a detector that fires on ~0% of the SAE's confirmed positions for its motif,
or fires far more on a DIFFERENT feature's positions, is a red flag.

  python3 validate_vs_sae.py [--band top] [--min-act 0]

Two checks per mapped feature:
  PRECISION-ish : of the feature's positions, what % does the detector fire on? (agreement)
  CONTRAST      : does it fire MORE on its own feature than on a random control feature? (specificity)
"""
import argparse, json, os, sys, chess
sys.path.insert(0, os.path.dirname(__file__))
import motifs as MO

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GOLD = os.path.join(ROOT, "output", "relabel_v9_d64_k1.json")
BOARDS = os.path.join(ROOT, "output", "all_feat_boards_d64_k1.json")

# Confirmed SAE feature -> (motif test, direction, expected_to_fire).
# motif test runs on (board_before, best_move) for MISSED, or detects on the best line if available.
# We use SINGLE-MOVE detectors here because the SAE board data has fen+best but no PV line.
# expected=True : detector SHOULD fire on most of these (direction agreement)
# expected=None : no single-move detector maps cleanly (skip — needs lines / is a Layer-1 predicate)
FEATURE_MOTIF = {
    "54": ("fork_or_pin", "missed", True),    # Missed Fork/Pin
    "47": ("mate_or_check", "missed", None),  # Missed Check/Mate (needs line)
    "23": ("mate_or_check", "missed", None),  # Missed Check/Mate (needs line)
    "59": ("mate_or_check", "allowed", None), # Allowed Check/Mate (needs refutation line)
    "35": ("zwischenzug", "missed", None),    # Missed Zwischenzug (needs line)
    "3":  ("hanging_best", "missed", True),   # Missed Hanging Piece — best move captures a hanging pc
    "17": ("hanging_best", "missed", True),   # Missed Free Capture (Minor)
    "0":  ("hanging_best", "missed", None),   # Missed Queen Capture / Involved Queen (queen-specific)
}


def best_move(item):
    fen, best = item["fen"], item.get("best", "")
    if not best:
        return None, None
    b = chess.Board(fen)
    try:
        mv = chess.Move.from_uci(best)
    except Exception:
        return b, None
    return b, (mv if mv in b.legal_moves else None)


def test_fork_or_pin(item):
    b, mv = best_move(item)
    if mv is None:
        return None
    return MO.is_fork(b, mv) or MO.is_pin(b, mv)


def test_hanging_best(item):
    """Best move captures a hanging (free) enemy piece."""
    b, mv = best_move(item)
    if mv is None:
        return None
    return MO.is_hanging_piece(b, mv)


TESTS = {"fork_or_pin": test_fork_or_pin, "hanging_best": test_hanging_best}


def run(band="top", min_act=0.0):
    gold = json.load(open(GOLD))
    boards = json.load(open(BOARDS))
    print(f"Directional validation vs SAE gold (band={band}, min_act={min_act})\n")
    print(f"{'feat':>5} {'chip':<34} {'motif':<14} {'fires':>10}  {'control':>10}  verdict")
    print("-" * 92)
    for fk, (motif, direction, expected) in FEATURE_MOTIF.items():
        if motif not in TESTS:
            chip = gold.get(fk, {}).get("chip", "?")
            print(f"{('f'+fk):>5} {chip[:33]:<34} {motif:<14} {'(needs line)':>10}")
            continue
        chip = gold.get(fk, {}).get("chip", "?")
        items = [it for it in boards.get(fk, {}).get("bands", {}).get(band, []) if it.get("act", 0) >= min_act]
        fn = TESTS[motif]
        fired = [fn(it) for it in items]
        fired = [f for f in fired if f is not None]
        rate = sum(fired) / len(fired) if fired else 0.0
        # control: same detector on a DIFFERENT feature's positions (specificity)
        ctrl_fk = "1" if fk != "1" else "6"   # f1 Wasted Tempo / f6 Bad Trade — non-tactical controls
        ctrl_items = boards.get(ctrl_fk, {}).get("bands", {}).get(band, [])
        ctrl = [fn(it) for it in ctrl_items]
        ctrl = [c for c in ctrl if c is not None]
        ctrl_rate = sum(ctrl) / len(ctrl) if ctrl else 0.0
        verdict = "OK" if rate >= 0.30 and rate > ctrl_rate else ("LOW" if rate < 0.15 else "WEAK")
        if expected is None:
            verdict = "info"
        print(f"{('f'+fk):>5} {chip[:33]:<34} {motif:<14} {rate:>9.0%}  {ctrl_rate:>9.0%}  {verdict}")
    print("\nNote: SAE is polysemantic by design, so <100% agreement is expected. The gate is: detector")
    print("fires substantially on its motif's positions AND more than on a non-tactical control feature.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="top")
    ap.add_argument("--min-act", type=float, default=0.0)
    a = ap.parse_args()
    run(a.band, a.min_act)
