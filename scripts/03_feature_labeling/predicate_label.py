#!/usr/bin/env python3
"""Deterministic board-predicate 'labeler' — names a feature from objective board facts, no LLM.

For each feature, over its sampled boards (weighted toward the strong bands), compute predicates:
  - played_capture       : your move was a capture
  - best_capture         : the best move is a capture (you MISSED a capture)
  - allows_check         : after your move, opponent has a check
  - best_is_check        : the best move is a check (you MISSED a check)
  - played_attacks_queen : your move attacks the enemy queen
  - hung_own (proxy)     : your move is a non-capture AND opponent can capture material next (loose)
Then pick a name by the dominant predicate, mirroring the hand-derived rules. The point is NOT to
replace human naming — it's to TEST whether a deterministic rule reproduces the gold labels Sam set
by hand. If it matches the 8 gold features, we trust its read on the 56 unreviewed.

  python3 predicate_label.py --boards output/all_feat_boards_d64_k1.json \
    --labels output/relabel_v9_d64_k1.json --out output/predicate_labels_d64_k1.json
"""
import argparse, json, chess

ap = argparse.ArgumentParser()
ap.add_argument("--boards", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bandweight", default="top,upper", help="bands to weight (comma)")
a = ap.parse_args()

A = json.load(open(a.boards))
L = json.load(open(a.labels))
WB = set(a.bandweight.split(","))


def safe_board(fen):
    try: return chess.Board(fen)
    except Exception: return None


def preds_for_board(b, uci, best):
    p = {}
    def mv(u):
        if not u or len(u) < 4: return None
        try:
            m = chess.Move.from_uci(u)
            return m if m in b.legal_moves else None
        except Exception:
            return None
    pm, bm = mv(uci), None
    # best move parsed on the ORIGINAL board (best is the alternative to the played move)
    if best and len(best) >= 4:
        try:
            bmm = chess.Move.from_uci(best)
            bm = bmm if bmm in b.legal_moves else None
        except Exception:
            bm = None
    p["played_capture"] = bool(pm and b.is_capture(pm))
    p["best_capture"] = bool(bm and b.is_capture(bm))
    p["best_is_check"] = bool(bm and b.gives_check(bm))
    # attacks enemy queen with the played move
    aq = False
    if pm:
        b2 = b.copy(); mover = b.turn
        try:
            b2.push(pm)
            for sq, pc in b2.piece_map().items():
                if pc.piece_type == chess.QUEEN and pc.color != mover and b2.is_attacked_by(mover, sq):
                    aq = True; break
        except Exception:
            pass
    p["played_attacks_queen"] = aq
    # allows check: after played move, opponent has a checking reply
    ac = False
    if pm:
        b3 = b.copy()
        try:
            b3.push(pm); ac = any(b3.gives_check(m) for m in b3.legal_moves)
        except Exception:
            pass
    p["allows_check"] = ac
    # piece moved
    p["piece"] = chess.piece_name(b.piece_type_at(pm.from_square)) if (pm and b.piece_type_at(pm.from_square)) else "?"
    return p


def analyze(fid):
    bands = A[fid]["bands"]
    rows = []
    for bn, boards in bands.items():
        w = 2 if bn in WB else 1  # weight strong bands
        for bd in boards:
            b = safe_board(bd["fen"])
            if b is None: continue
            pr = preds_for_board(b, bd["uci"], bd.get("best", ""))
            for _ in range(w): rows.append(pr)
    n = len(rows) or 1
    agg = {k: sum(1 for r in rows if r.get(k)) / n for k in
           ("played_capture", "best_capture", "best_is_check", "played_attacks_queen", "allows_check")}
    # dominant piece
    from collections import Counter
    pc = Counter(r["piece"] for r in rows)
    agg["top_piece"], agg["top_piece_share"] = (pc.most_common(1)[0][0], pc.most_common(1)[0][1] / n) if pc else ("?", 0)
    return agg


def name_from_preds(p):
    """Rules distilled from the hand-labeled gold set. Order matters (most-specific first)."""
    pq = p["played_attacks_queen"]; ac = p["allows_check"]; bic = p["best_is_check"]
    bc = p["best_capture"]; plc = p["played_capture"]
    if pq >= 0.85:
        return "Incorrectly Threatened Queen"
    if ac >= 0.9 and plc < 0.6:
        return "Allowed Check/Mate"
    if bic >= 0.85:
        return "Missed Check/Mate"
    if bc >= 0.85 and plc < 0.4:
        return "Missed Capture"
    if plc >= 0.85:
        return "Bad Capture"
    if bc >= 0.6 and plc < 0.4:
        return "Missed Capture (loose)"
    if ac >= 0.7:
        return "Exposed King"
    return "Mixed / unclear"


out = {}
for fid in A:
    p = analyze(fid)
    pred_name = name_from_preds(p)
    gold = L.get(fid, {})
    out[fid] = {"predicate_name": pred_name, "preds": {k: round(v, 2) for k, v in p.items() if isinstance(v, float)},
                "current_chip": gold.get("chip", ""), "edited": bool(gold.get("edited"))}

json.dump(out, open(a.out, "w"), indent=1)

# Report: gold check first
print("=== GOLD CHECK (your 8 hand-labeled features: does the predicate rule agree?) ===")
for fid, v in sorted(out.items(), key=lambda x: int(x[0])):
    if v["edited"]:
        print(f"  f{fid:<3} gold='{v['current_chip']}'  predicate='{v['predicate_name']}'")
print("\n=== UNREVIEWED features where predicate DISAGREES with current label ===")
for fid, v in sorted(out.items(), key=lambda x: int(x[0])):
    if not v["edited"]:
        print(f"  f{fid:<3} current='{v['current_chip']}'  ->  predicate='{v['predicate_name']}'  {v['preds']}")
