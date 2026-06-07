#!/usr/bin/env python3
"""Complete, principled board-predicate vector for SAE chess-mistake features.

DESIGN DISCIPLINE (anti-overfit): the predicate set is derived from the GENERAL space of what is
objectively measurable about a chess mistake — NOT reverse-engineered from any reviewed features.
A mistake = (played move P, best move B, resulting position). Each of the three has the same small
set of mechanisms (capture / check / mate / queen-attack / pawn-push / king-safety / trade / quiet).
We compute ALL of them on every board, aggregate per feature (weighting strong activation bands),
and name by a single generic (direction × mechanism) table applied identically to every feature.

The reviewed/gold features are a HELD-OUT check, never a tuning target. With only ~8 gold points
this cannot 'validate' a tuned classifier; the vector's real use is as evidence for the labeler and
as a flag for labels that contradict the dominant objective signal.

  python3 board_predicates.py --boards output/all_feat_boards_d64_k1.json \
    --labels output/relabel_v9_d64_k1.json --out output/predicates_d64_k1.json
"""
import argparse, json, chess
from collections import Counter

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def king_pressure(board, color):
    """Count enemy attackers on the king's square + its 8 neighbors (a king-safety proxy)."""
    ks = board.king(color)
    if ks is None:
        return 0
    squares = [ks] + [s for s in chess.SQUARES if chess.square_distance(s, ks) == 1]
    return sum(1 for s in squares if board.is_attacked_by(not color, s))


def hangs_material(board, mover):
    """After mover's move (board = post-move, opponent to move): can opponent win material with a
    capture of an under-defended piece? Proxy for 'you hung something'."""
    for mv in board.legal_moves:
        if board.is_capture(mv):
            victim = board.piece_at(mv.to_square)
            if victim is None:  # en passant
                continue
            attacker = board.piece_at(mv.from_square)
            # winning if victim worth more than attacker, OR victim undefended
            if PIECE_VAL[victim.piece_type] > PIECE_VAL.get(attacker.piece_type, 0):
                return True
            if not board.is_attacked_by(mover, mv.to_square):  # victim square not re-defended by mover
                return True
    return False


def attacks_queen(board_after, mover):
    for sq, pc in board_after.piece_map().items():
        if pc.piece_type == chess.QUEEN and pc.color != mover and board_after.is_attacked_by(mover, sq):
            return True
    return False


def board_preds(fen, uci, best):
    b = chess.Board(fen)
    mover = b.turn
    P = {}

    def parse(u):
        if not u or len(u) < 4:
            return None
        try:
            m = chess.Move.from_uci(u)
            return m if m in b.legal_moves else None
        except Exception:
            return None
    pm, bm = parse(uci), parse(best)

    # --- PLAYED move (self-inflicted side) ---
    if pm:
        pt = b.piece_type_at(pm.from_square)
        P["played_capture"] = b.is_capture(pm)
        P["played_check"] = b.gives_check(pm)
        P["played_pawn"] = pt == chess.PAWN
        P["played_king"] = pt == chess.KING
        dr = chess.square_rank(pm.to_square) - chess.square_rank(pm.from_square)
        dr = dr if mover == chess.WHITE else -dr
        P["played_forward"] = dr > 0
        P["played_backward"] = dr < 0
        ba = b.copy(); ba.push(pm)
        P["played_attacks_queen"] = attacks_queen(ba, mover)
        P["allows_check"] = any(ba.gives_check(m) for m in ba.legal_moves)
        P["hangs_own"] = hangs_material(ba, mover)
        # king exposure: did mover's own king pressure go up vs before?
        P["king_exposed"] = king_pressure(ba, mover) > king_pressure(b, mover)
        # sacrifice proxy: non-capture that hangs own material, or capture onto a defended square worth more
        P["played_sacrifice"] = (not b.is_capture(pm)) and P["hangs_own"]

    # --- BEST move (omission side) ---
    if bm:
        P["best_capture"] = b.is_capture(bm)
        P["best_check"] = b.gives_check(bm)
        bt = b.piece_type_at(bm.from_square)
        P["best_pawn"] = bt == chess.PAWN
        bb = b.copy(); bb.push(bm)
        P["best_attacks_queen"] = attacks_queen(bb, mover)
        P["best_quiet"] = (not b.is_capture(bm)) and (not b.gives_check(bm))
    return P


def aggregate(feat, strong=("top", "upper")):
    rows = []
    for bn, boards in feat["bands"].items():
        w = 2 if bn in strong else 1
        for bd in boards:
            try:
                pr = board_preds(bd["fen"], bd["uci"], bd.get("best", ""))
            except Exception:
                continue
            for _ in range(w):
                rows.append(pr)
    n = len(rows) or 1
    keys = ["played_capture", "played_check", "played_pawn", "played_king", "played_forward",
            "played_backward", "played_attacks_queen", "allows_check", "hangs_own", "king_exposed",
            "played_sacrifice", "best_capture", "best_check", "best_pawn", "best_attacks_queen", "best_quiet"]
    return {k: round(sum(1 for r in rows if r.get(k)) / n, 2) for k in keys}


def name(p):
    """Generic (direction × mechanism) table — applied identically to every feature.
    Omission signals live in best_*; self-inflicted in played_*/consequence. Most-specific first."""
    g = p.get
    # ---- omission (you skipped the best move) ----
    if g("best_check", 0) >= 0.8:
        return "Missed Check/Mate"
    if g("best_attacks_queen", 0) >= 0.7 and g("best_capture", 0) < 0.5:
        return "Missed Attack on Queen"
    if g("best_capture", 0) >= 0.8 and g("played_capture", 0) < 0.4:
        return "Missed Capture"
    # ---- self-inflicted ----
    if g("played_attacks_queen", 0) >= 0.8:
        return "Incorrectly Threatened Queen"
    if g("allows_check", 0) >= 0.85 and g("played_capture", 0) < 0.6:
        return "Allowed Check/Mate"
    if g("played_capture", 0) >= 0.8:
        return "Bad Capture"
    if g("played_sacrifice", 0) >= 0.6:
        return "Bad Sacrifice"
    if g("played_pawn", 0) >= 0.7 and g("king_exposed", 0) >= 0.3:
        return "Exposed King (pawn)"
    if g("king_exposed", 0) >= 0.4 or g("allows_check", 0) >= 0.7:
        return "Exposed King"
    if g("best_capture", 0) >= 0.6 and g("played_capture", 0) < 0.4:
        return "Missed Capture (loose)"
    if g("played_pawn", 0) >= 0.6 and g("best_pawn", 0) >= 0.5:
        return "Pawn Push / Break"
    if g("hangs_own", 0) >= 0.6:
        return "Hangs Piece"
    return "Mixed / unclear"


ap = argparse.ArgumentParser()
ap.add_argument("--boards", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
A = json.load(open(a.boards)); L = json.load(open(a.labels))

out = {}
for fid, feat in A.items():
    p = aggregate(feat)
    out[fid] = {"name": name(p), "preds": p,
                "current": L.get(fid, {}).get("chip", ""), "gold": bool(L.get(fid, {}).get("edited"))}
json.dump(out, open(a.out, "w"), indent=1)

print("=== HELD-OUT GOLD CHECK (predicate name vs your hand labels) ===")
hit = 0; tot = 0
for fid, v in sorted(out.items(), key=lambda x: int(x[0])):
    if v["gold"]:
        tot += 1
        # loose match: does the predicate name share the core word?
        ok = any(w in v["name"].lower() for w in v["current"].lower().split() if len(w) > 3)
        hit += ok
        print(f"  {'✓' if ok else '✗'} f{fid:<3} gold='{v['current']}'  predicate='{v['name']}'")
print(f"  -> {hit}/{tot} share a core term\n")
print("=== UNREVIEWED: predicate disagrees with current label (strong signal) ===")
for fid, v in sorted(out.items(), key=lambda x: int(x[0])):
    if not v["gold"] and v["name"] != "Mixed / unclear":
        cur = v["current"].lower()
        if not any(w in cur for w in v["name"].lower().split() if len(w) > 3):
            print(f"  f{fid:<3} current='{v['current']}'  ->  predicate='{v['name']}'")
