#!/usr/bin/env python3
"""Dump N examples per tag with INDEPENDENT board facts, for human/model eyeballing.

Sam, 2026-08-08: "i more so wanted you to individually look through like 10-20 examples on each tag and
see if anything obviously wrong." The contradiction-check pass (audit_fifa_corpus.py) only catches gross
violations — it would NOT have caught the move-38 Hung Material bug, where the tag legitimately fired and
only its magnitude was wrong. Reading positions is what has actually found every bug so far.

The facts printed here are recomputed FROM THE BOARD, deliberately not reusing the detector's own
reasoning — otherwise the review is circular (the detector asserting itself). For the key move of the
relevant line we print: what moves, what it captures, what enemy pieces it lands attacking, whether it
checks, and SEE. That's enough to judge most motif claims by hand.

Direction decides which line matters:
  missed  -> the BEST line (what the player could have done)
  allowed/hung -> [played] + REFUTATION (what the opponent does to punish)
  played/info  -> the played move itself

Usage (on chess-poc, from ~/SageMaker/tagger_run):
    python3 sample_tag_examples.py --enrich ../fifa_blitz/fifa_enrich.json --out ../tag_examples.txt \
        --per-tag 12 --scan 20000
    python3 sample_tag_examples.py ... --only "Missed Fork,Allowed Pin"
"""
import argparse
import collections
import os
import json
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistake import Mistake, _eval_to_cp          # noqa: E402
from tagger import tag_mistake_full               # noqa: E402
import chesslib_util as U                         # noqa: E402

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
SYM = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q",
       chess.KING: "K"}
INFO_NOISE = {"Opening", "Middlegame", "Endgame"}


def from_fifa_entry(fen, uci, e):
    b = chess.Board(fen)
    best = e.get("top_3_best") or []
    refut = e.get("top_3_refutations") or []
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=e.get("best_uci", ""),
        best_line_san=((best[0].get("line") or "").split() if best else []),
        refutation_san=((refut[0].get("line") or "").split() if refut else []),
        eval_before=_eval_to_cp(e.get("eval_before")), eval_after=_eval_to_cp(e.get("eval_after")),
        cp_loss=int(e.get("cp_loss", 0) or 0), mover=b.turn,
        played_san=e.get("played_san", ""), best_san=e.get("best_san", ""),
    )


def move_facts(board, mv):
    """What this move DOES, read off the board. No detector logic."""
    piece = board.piece_at(mv.from_square)
    if piece is None:
        return "?"
    bits = [f"{SYM[piece.piece_type]}{chess.square_name(mv.from_square)}->{chess.square_name(mv.to_square)}"]
    victim = board.piece_at(mv.to_square)
    if victim is not None:
        bits.append(f"takes {SYM[victim.piece_type]}({VAL[victim.piece_type]})")
    elif board.is_en_passant(mv):
        bits.append("takes P(1) e.p.")
    after = board.copy()
    after.push(mv)
    if after.is_checkmate():
        bits.append("MATE")
    elif after.is_check():
        bits.append("CHECK")
    # what the moved piece now attacks (valuable targets only) — the fork/skewer/pin evidence
    hits = []
    for sq in after.attacks(mv.to_square):
        t = after.piece_at(sq)
        if t is not None and t.color != piece.color and VAL[t.piece_type] >= 3:
            hits.append(f"{SYM[t.piece_type]}{chess.square_name(sq)}")
        elif t is not None and t.color != piece.color and t.piece_type == chess.KING:
            hits.append(f"K{chess.square_name(sq)}")
    if hits:
        bits.append("hits:" + ",".join(sorted(hits)))
    # is the moved piece itself now attacked / defended (loose-piece + sac evidence)
    atk = len(after.attackers(not piece.color, mv.to_square))
    dfn = len(after.attackers(piece.color, mv.to_square))
    if atk:
        bits.append(f"landing atk={atk} def={dfn}")
    try:
        see = U.see(board, mv) if hasattr(U, "see") else None
        if see is not None:
            bits.append(f"SEE={see}")
    except Exception:
        pass
    return " ".join(bits)


def render(m, tag, key):
    """One example block: the position, the relevant line, and per-ply board facts."""
    out = []
    b = chess.Board(m.fen_before)
    direction = tag["direction"]
    mat = sum(VAL[p.piece_type] * (1 if p.color == m.mover else -1)
              for p in b.piece_map().values())
    out.append(f"  fen      {m.fen_before}")
    out.append(f"  mover    {'White' if m.mover else 'Black'}  material(mover-POV) {mat:+d}  "
               f"eval {m.eval_before}->{m.eval_after}  cp_loss {m.cp_loss}")
    try:
        pm = chess.Move.from_uci(m.played_uci)
        out.append(f"  played   {m.played_san:<8} {move_facts(b, pm)}")
    except Exception:
        out.append(f"  played   {m.played_san}")
    out.append(f"  best     {m.best_san:<8} (uci {m.best_uci})")
    out.append(f"  evidence {tag.get('evidence','')[:120]}")

    if direction == "missed":
        line, start = m.best_line_san, chess.Board(m.fen_before)
        out.append(f"  BEST line (what was missed): {' '.join(line)}")
    else:
        line, start = m.refutation_san, chess.Board(m.fen_before)
        try:
            start.push(chess.Move.from_uci(m.played_uci))
        except Exception:
            pass
        out.append(f"  REFUTATION (the punishment): {' '.join(line)}")
    bb = start
    for i, san in enumerate(line[:8]):
        try:
            mv = bb.parse_san(san)
        except Exception:
            out.append(f"    ply{i+1} {san} <UNPARSEABLE>")
            break
        who = "mover" if bb.turn == m.mover else "opp"
        out.append(f"    ply{i+1} {san:<8} [{who:>5}] {move_facts(bb, mv)}")
        bb.push(mv)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enrich", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-tag", type=int, default=12)
    p.add_argument("--scan", type=int, default=20000, help="positions to scan for examples")
    p.add_argument("--only", default=None, help="comma-separated labels")
    p.add_argument("--stride", type=int, default=1, help="sample every Nth position (spread the draw)")
    args = p.parse_args()

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    data = json.load(open(args.enrich))
    keys = list(data.keys())[:args.scan][::args.stride]

    buckets = collections.defaultdict(list)
    for k in keys:
        if "|" not in k:
            continue
        fen, uci = k.rsplit("|", 1)
        try:
            m = from_fifa_entry(fen, uci, data[k])
            # classification=None on purpose: that makes the tagger apply its REAL entry gate
            # (win_drop >= 10). Forcing "blunder" overrides the gate and surfaces tags on moves that
            # aren't mistakes at all — the first version of this script did that and produced a
            # "Greedy Capture" on a 5-centipawn move, which was my harness, not the tagger.
            tags = [t for t in tag_mistake_full(m, with_maia=False)["tags"]
                    if t["label"] not in INFO_NOISE]
        except Exception:
            continue
        for t in tags:
            lab = t["label"]
            if only and lab not in only:
                continue
            if len(buckets[lab]) < args.per_tag:
                buckets[lab].append(render(m, t, k))

    with open(args.out, "w") as fh:
        for lab in sorted(buckets, key=lambda x: -len(buckets[x])):
            fh.write(f"\n{'='*100}\nTAG: {lab}   (examples: {len(buckets[lab])})\n{'='*100}\n")
            for i, blk in enumerate(buckets[lab], 1):
                fh.write(f"\n--- example {i} ---\n{blk}\n")
    print(f"wrote {args.out}: {len(buckets)} tags, "
          f"{sum(len(v) for v in buckets.values())} examples")


if __name__ == "__main__":
    main()
