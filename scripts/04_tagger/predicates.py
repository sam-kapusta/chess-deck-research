#!/usr/bin/env python3
"""Layer 1 — our deterministic position/material predicates (the tier Lichess's tagger lacks).

Each predicate is a pure function of the Mistake object returning a list of (tag, direction, evidence)
or []. No engine, no torch. direction in {missed, allowed, hung, played, info}.

These are the crisp, high-volume tags: phase, game-state, capture-vs-exchange (by piece), hung
material (from the refutation line, end-of-line delta — the validated metric, NOT one-ply), king
safety, pawn-structure deltas, wrong move-order, only-move, captured-with-wrong-piece.
"""
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE_NAME = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
              chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}


# ---------- helpers ----------
def _material(board, color):
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color)


def _best_move(m):
    b = m.board_before
    for u in (m.best_uci,):
        if u:
            try:
                mv = chess.Move.from_uci(u)
                if mv in b.legal_moves:
                    return mv
            except Exception:
                pass
    # fall back to first move of best_line
    if m.best_line_san:
        try:
            return b.parse_san(m.best_line_san[0])
        except Exception:
            return None
    return None


def _played_move(m):
    try:
        return chess.Move.from_uci(m.played_uci)
    except Exception:
        return None


def _is_defended(board_after_move, sq, by_color):
    return board_after_move.is_attacked_by(by_color, sq)


# ---------- phase / state ----------
def phase(m):
    b = m.board_before
    npieces = len(b.piece_map())
    nonpawn = sum(1 for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    if b.fullmove_number <= 12 and npieces >= 24:
        ph = "Opening"
    elif npieces <= 12 or nonpawn <= 4:
        ph = "Endgame"
    else:
        ph = "Middlegame"
    return [(ph, "info", f"{npieces} pieces, move {b.fullmove_number}")]


def game_state(m):
    if m.eval_before is None:
        return []
    # white-POV cp -> mover-POV
    cp = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    if cp >= 150:
        s = "Winning"
    elif cp <= -150:
        s = "Losing"
    else:
        s = "Equal"
    return [(s, "info", f"{cp:+d}cp before (mover POV)")]


# ---------- material: capture vs exchange, by piece ----------
def capture_or_exchange(m):
    """Best move is a capture: free (undefended) -> Missed Free <Piece> (e.g. "Missed Free Pawn");
    defended/even -> Missed <Piece> Exchange (e.g. "Missed Queen Exchange")."""
    b = m.board_before
    bm = _best_move(m)
    pm = _played_move(m)
    if bm is None or not b.is_capture(bm):
        return []
    # the played move itself being a capture of the same square is handled by captured_wrong_piece
    victim = b.piece_at(bm.to_square)
    if victim is None:  # en passant
        return [("Missed Capture (Pawn)", "missed", "best move = en passant")]
    after = b.copy(); after.push(bm)
    defended = after.is_attacked_by(not m.mover, bm.to_square)
    pname = PIECE_NAME[victim.piece_type]
    if not defended:
        return [(f"Missed Free {pname}", "missed", f"best {m.best_san} takes undefended {pname.lower()}")]
    # defended but you still win material (attacker worth less than victim). Displayed as the same
    # "Missed Free X" tag as the undefended case (Sam: collapse free/winning into one coaching tag);
    # the distinction survives in the evidence string ("wins X for less" vs "takes undefended X").
    attacker = b.piece_at(bm.from_square)
    if attacker and VAL[victim.piece_type] > VAL[attacker.piece_type] + 0.5:
        return [(f"Missed Free {pname}", "missed", f"best {m.best_san} wins {pname.lower()} for less")]
    # Equal-value gate: an "exchange/trade" means like-for-like value. If the attacker is worth MORE
    # than the (defended) victim, capturing it sheds material — that's a SACRIFICE, not an even trade,
    # and sacrifice_line already names it ("Missed Sacrifice"). Without this gate, Q-takes-defended-B
    # mislabels as "Missed Bishop Exchange" — 24% of Exchange fires (310/1274), 207 of which ALSO carry
    # "Missed Sacrifice" (a direct contradiction). Drop the bogus exchange label and let the sac stand.
    # (Sam, ply 50: best Qxe4+ = queen for a defended bishop was tagged "Missed Bishop Exchange".)
    if attacker and VAL[attacker.piece_type] > VAL[victim.piece_type] + 0.5:
        return []
    if victim.piece_type == chess.PAWN:
        return [("Missed Pawn Trade", "missed", f"best {m.best_san} = even pawn trade")]
    return [(f"Missed {pname} Exchange", "missed", f"best {m.best_san} = even trade of {pname.lower()}")]


def capture_direction(m):
    """Behavior-level capture mistake: best is a capture, played is a DIFFERENT capture
      -> 'Wrong Capture' (you captured, but the wrong target).
    The "played a quiet move instead of capturing" case is intentionally NOT tagged here: whenever
    best is a capture, capture_or_exchange already emits the specific piece tag (Missed Free X /
    Missed X Exchange / ...), so a generic "Missed Capture" was always a redundant duplicate of it.
    (Removed per Sam — ply 17 fired both "Missed Free Pawn" and "Missed Capture".)"""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or not b.is_capture(bm):
        return []
    # both captures — wrong one (different target square OR different piece on same square)
    if b.is_capture(pm) and (pm.to_square != bm.to_square or pm.from_square != bm.from_square):
        return [("Wrong Capture", "played", f"captured {m.played_san}; best capture was {m.best_san}")]
    return []


def bad_capture(m):
    """The PLAYED move is a capture that backfires (eval crashed)."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or not b.is_capture(pm) or m.cp_loss < 120:
        return []
    victim = b.piece_at(pm.to_square)
    pname = PIECE_NAME[victim.piece_type] if victim else "Pawn"
    return [(f"Bad Capture", "played", f"played {m.played_san} (took {pname.lower()}) lost {m.cp_loss}cp")]


def _material_diff(board, side):
    return _material(board, side) - _material(board, not side)


def hung_material(m):
    """Played move loses material on NET across the refutation line. Uses the change in material_diff
    (mover minus opponent), so the player's own recaptures in the line are netted out — a player who
    loses a rook but takes a bishop back is down 2, not 5. (The old gross-loss metric over-claimed by
    ~2x: 30% of fires claimed 5+ pts while cp_loss justified far less — line-flow, not a real hang.)

    Split by DEPTH (like fork): IMMEDIATE = >=2 pts already gone after the opponent's FIRST reply
    ("Hung Material" — a one-move oversight, the piece is just taken; 98% of these the 1st reply is a
    capture). DELAYED = the net loss is realized only after a multi-move sequence ("Lost Material to
    Combination" — you allowed a tactic that wins material a few moves deep). Verified clean ~48/52."""
    if not m.refutation_san:
        return []
    # Reference point MUST be board_BEFORE the played move, so that if the played move is itself a
    # capture, the player's own gain is netted in. Measuring from board_after (post-capture) made an
    # EQUAL trade (e.g. Bxc6 bxc6, 3-for-3) read as a 3-pt hang — it counted the recapture loss but
    # not the capture gain. Now an equal trade nets 0 and does NOT fire. (Caught by Sam.)
    b0 = m.board_before
    start_diff = _material_diff(b0, m.mover)
    bb = chess.Board(b0.fen())
    try:
        bb.push(chess.Move.from_uci(m.played_uci))
    except Exception:
        return []
    diffs = [_material_diff(bb, m.mover)]   # diffs[0] = right after the played move (opponent to move)
    first_victim = None                      # the piece the opponent CAPTURES on its first reply
    for i, san in enumerate(m.refutation_san):
        try:
            mv = bb.parse_san(san)
        except Exception:
            break
        if i == 0 and bb.is_capture(mv):
            vic = bb.piece_at(mv.to_square)   # capture target on the post-played board
            if vic is not None:               # None = en passant (a pawn); leave unnamed
                first_victim = vic.piece_type
        bb.push(mv)
        diffs.append(_material_diff(bb, m.mover))
    end_diff = diffs[-1]
    net_lost = start_diff - end_diff   # vs BEFORE the played move — equal trades net 0
    if net_lost < 2:
        return []
    # how much is already gone after the opponent's first reply (ply 1 of the refutation)?
    immediate_lost = start_diff - diffs[1] if len(diffs) > 1 else (start_diff - diffs[0])
    if immediate_lost >= 2:
        # Name the hung piece when the opponent's first reply is a clean capture AND that capture is
        # the dominant loss (the named piece is worth ~the net loss). Else keep generic "Hung Material".
        if first_victim is not None and VAL.get(first_victim, 0) >= net_lost - 1:
            pname = PIECE_NAME[first_victim]
            return [(f"Hung {pname}", "hung",
                     f"opponent's first reply captures your {pname.lower()} ({net_lost} net over line)")]
        return [("Hung Material", "hung",
                 f"opponent's first reply wins {immediate_lost} pts ({net_lost} net over line)")]
    return [("Lost Material to Combination", "hung",
             f"refutation wins {net_lost} pts net over {len(diffs)-1} plies (delayed, not a 1-move hang)")]


# ---------- king safety ----------
def king_in_center(m):
    b = m.board_before
    ph = phase(m)[0][0]
    if ph == "Endgame":
        return []
    ks = b.king(m.mover)
    if ks is None:
        return []
    f = chess.square_file(ks)
    # king still in the center files (d/e) past the opening, hasn't castled
    if f in (3, 4) and not (b.has_castling_rights(m.mover)):
        return [("King in Center", "info", "uncastled king on d/e file in middlegame")]
    return []


def lost_castling(m):
    b = m.board_before
    pm = _played_move(m)
    if pm is None:
        return []
    had = b.has_castling_rights(m.mover)
    if not had:
        return []
    after = b.copy(); after.push(pm)
    if not after.has_castling_rights(m.mover) and not b.is_castling(pm):
        return [("Lost Castling Rights", "played", "played move forfeited castling")]
    return []


def exposed_king_pawn(m):
    """Played move is a pawn move in front of own king's shelter."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or b.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    ks = b.king(m.mover)
    if ks is None:
        return []
    # pawn move within 1 file of the king and on the king's side of the board
    if abs(chess.square_file(pm.from_square) - chess.square_file(ks)) <= 1 and m.cp_loss >= 80:
        # only if it's a structural advance near the king
        return [("Pawn Move Exposed King", "played", "pawn push near own king")]
    return []


# ---------- pawn structure deltas ----------
def _pawn_files(board, color):
    files = {}
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.PAWN and p.color == color:
            files.setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))
    return files


def _doubled_isolated(files, board_files):
    """Return (doubled_file, isolated_file) defects present in `files` (a _pawn_files map)."""
    doubled = next((f for f, ranks in files.items() if len(ranks) >= 2), None)
    return doubled


def pawn_structure(m):
    """A pawn move CREATED a structural weakness that the mistake caused. Two guards make this honest:
      1. Skip CAPTURES — a recapture (gxf3 regaining a piece) that incidentally doubles a pawn is not
         a voluntary structural concession. Only a quiet pawn push can "create" a weakness as the point.
      2. Compare against the BEST line — if the best move leads to the SAME doubled/isolated pawn, the
         defect isn't a consequence of the blunder, so don't tag it. (Caught by Sam on 13.gxf3, where
         the doubled f-pawn appears in the best line too: ...bxc6 gxf3.)"""
    pm = _played_move(m)
    if pm is None or m.board_before.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    before = m.board_before
    if before.is_capture(pm):          # guard 1: recaptures don't "create" a structural concession
        return []
    after = before.copy(); after.push(pm)
    bf = _pawn_files(before, m.mover); af = _pawn_files(after, m.mover)

    # what the BEST move's resulting structure looks like (guard 2)
    best_af = None
    bm = _best_move(m)
    if bm is not None:
        ab = before.copy(); ab.push(bm)
        best_af = _pawn_files(ab, m.mover)

    out = []
    # doubled: a file gained a 2nd+ pawn that the blunder caused AND the best move didn't also cause
    for f, ranks in af.items():
        if len(ranks) >= 2 and len(bf.get(f, [])) < len(ranks):
            if best_af is not None and len(best_af.get(f, [])) >= 2:
                continue   # best move also doubles this file -> not a consequence of the blunder
            out.append(("Created Doubled Pawn", "played", f"file {chr(97+f)} doubled (not in best line)"))
            break
    # isolated: the moved pawn's file has no friendly pawn on adjacent files, and best move avoids it
    tf = chess.square_file(pm.to_square)
    if tf in af and (tf - 1) not in af and (tf + 1) not in af:
        best_isolates = best_af is not None and tf in best_af and (tf - 1) not in best_af and (tf + 1) not in best_af
        if not best_isolates:
            out.append(("Created Isolated Pawn", "played", f"pawn on {chr(97+tf)} isolated (not in best line)"))
    return out


# ---------- move quality meta ----------
def wrong_move_order(m):
    """Played move IS in Stockfish's best line, just not first — a transposition/timing error."""
    if not m.best_line_san:
        return []
    b = m.board_before
    try:
        played_san = b.san(_played_move(m)) if _played_move(m) else m.played_san
    except Exception:
        played_san = m.played_san
    # is the played move the same as a LATER move in the best line (not move 0)?
    if played_san and played_san in m.best_line_san[1:]:
        return [("Wrong Move Order", "played", f"{played_san} is in the best line, played too early")]
    return []


def captured_wrong_piece(m):
    """Played and best move both capture the SAME square, with different pieces."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None:
        return []
    if b.is_capture(pm) and b.is_capture(bm) and pm.to_square == bm.to_square and pm.from_square != bm.from_square:
        return [("Captured With Wrong Piece", "played", f"played {m.played_san}, best {m.best_san} (same square)")]
    return []


# ---------- endgame type (board context, like phase — info tags) ----------
def _only_piece_types_present(board, allowed):
    """True if every non-king piece on the board is in `allowed` (a set of piece types)."""
    for p in board.piece_map().values():
        if p.piece_type == chess.KING:
            continue
        if p.piece_type not in allowed:
            return False
    return True


def endgame_type(m):
    """Name the endgame by surviving material (only fires in the Endgame phase). Mirrors cook's
    piece_endgame / queen_rook_endgame: a 'X endgame' = only kings, pawns, and X-type pieces."""
    if phase(m)[0][0] != "Endgame":
        return []
    b = m.board_before
    has = lambda pt: bool(b.pieces(pt, chess.WHITE) or b.pieces(pt, chess.BLACK))
    P = chess.PAWN
    # pure single-piece endgames (kings + pawns + that one piece type present)
    for pt, name in [(chess.QUEEN, "Queen"), (chess.ROOK, "Rook"),
                     (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight")]:
        if has(pt) and _only_piece_types_present(b, {P, pt}):
            if pt == chess.BISHOP:
                # one bishop each side -> Same/Opposite color (opp-color is famously drawish).
                # Multiple bishops / lopsided (bishop pair vs none) -> bare "Bishop Endgame".
                wb = list(b.pieces(chess.BISHOP, chess.WHITE))
                bb = list(b.pieces(chess.BISHOP, chess.BLACK))
                if len(wb) == 1 and len(bb) == 1:
                    same = (chess.square_rank(wb[0]) + chess.square_file(wb[0])) % 2 == \
                           (chess.square_rank(bb[0]) + chess.square_file(bb[0])) % 2
                    kind = "Same Color" if same else "Opposite Color"
                    return [(f"Bishop Endgame ({kind})", "info",
                             f"one bishop each, {'same' if same else 'opposite'} square color")]
                return [("Bishop Endgame", "info", "only K+P+bishops (multiple/lopsided)")]
            return [(f"{name} Endgame", "info", f"only K+P+{name.lower()}s on board")]
    # pawn endgame: kings + pawns only
    if _only_piece_types_present(b, {P}):
        return [("Pawn Endgame", "info", "only kings and pawns")]
    # queen+rook endgame: exactly one queen, >=1 rook, only Q/R/P/K
    pieces = list(b.piece_map().values())
    nq = sum(1 for p in pieces if p.piece_type == chess.QUEEN)
    if nq == 1 and any(p.piece_type == chess.ROOK for p in pieces) and \
       _only_piece_types_present(b, {P, chess.QUEEN, chess.ROOK}):
        return [("Queen + Rook Endgame", "info", "Q+R+P endgame")]
    return []


def backward_pawn(m):
    """Played move creates/leaves a backward pawn: a pawn behind its neighbors on adjacent files,
    on a half-open file, that can't safely advance. Light heuristic — flag for Sam to judge."""
    pm = _played_move(m)
    if pm is None or m.board_before.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    after = m.board_before.copy(); after.push(pm)
    tf = chess.square_file(pm.to_square); tr = chess.square_rank(pm.to_square)
    files = _pawn_files(after, m.mover)
    # backward: no friendly pawn on adjacent files at or behind this pawn's rank, and the stop
    # square is controlled by an enemy pawn (can't advance). Direction depends on color.
    fwd = 1 if m.mover == chess.WHITE else -1
    neighbors_behind = False
    for nf in (tf - 1, tf + 1):
        for nr in files.get(nf, []):
            # "behind or level" relative to advance direction
            if (m.mover == chess.WHITE and nr <= tr) or (m.mover == chess.BLACK and nr >= tr):
                neighbors_behind = True
    if neighbors_behind:
        return []
    stop = chess.square(tf, tr + fwd) if 0 <= tr + fwd <= 7 else None
    if stop is not None and after.is_attacked_by(not m.mover, stop):
        # only if the attacker on the stop square is a pawn
        for asq in after.attackers(not m.mover, stop):
            if after.piece_type_at(asq) == chess.PAWN:
                return [("Created Backward Pawn", "played", f"pawn on {chr(97+tf)} backward, stop square held")]
    return []


# ---------- registry ----------
ALL_PREDICATES = [
    phase, game_state, capture_or_exchange, capture_direction, bad_capture, hung_material,
    king_in_center, lost_castling, exposed_king_pawn, pawn_structure,
    wrong_move_order, captured_wrong_piece, endgame_type, backward_pawn,
]


def tag_predicates(m):
    out = []
    for fn in ALL_PREDICATES:
        try:
            out.extend(fn(m))
        except Exception:
            pass
    return out
