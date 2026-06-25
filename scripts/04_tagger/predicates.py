#!/usr/bin/env python3
"""Layer 1 — our deterministic position/material predicates (the tier Lichess's tagger lacks).

Each predicate is a pure function of the Mistake object returning a list of (tag, direction, evidence)
or []. No engine, no torch. direction in {missed, allowed, hung, played, info}.

These are the crisp, high-volume tags: phase, game-state, capture-vs-exchange (by piece), hung
material (from the refutation line, end-of-line delta — the validated metric, NOT one-ply), king
safety, pawn-structure deltas, wrong move-order, only-move, captured-with-wrong-piece.
"""
import chess
import chesslib_util as U

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE_NAME = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
              chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}

# The single mistake-severity threshold, in win%-drop (mover POV). Applied ONCE at the tagger entry
# (tagger.tag_mistake_full), NOT per-predicate — predicates are pure pattern detectors; this decides
# whether the position is a real mistake at all. Replaces 8 magic cp_loss thresholds (40/50/60/80/
# 100/120). Win%-drop is prod's native classification currency (classifyMoves.ts: INACCURACY=10,
# MISTAKE=20 win-pts) AND the leak-metrics win%-lost metric — tagger, classifier, and drill metric
# all agree on what "a mistake" is. (GH #29.)
# PROVISIONAL: 10.0 = prod's INACCURACY band (≈108cp at even). Tune on the band corpus (issue #29
# step 5) before shipping — the old gates spanned 3.7–10.9 win-pts, so positional tags need a re-measure.
WIN_DROP_MIN = 10.0


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
    # Pure detector: fires whenever best is a capture. The "is it actually a mistake" gate (win%-drop)
    # is applied ONCE at the tagger entry (tag_mistake_full), so played==best equal trades never reach
    # here. (GH #29 — removed the per-predicate cp_loss/win_drop copy-paste.)
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
    # Name the trade by BOTH pieces, not just the victim. The old "Missed {victim} Exchange" mislabeled
    # bishop-takes-knight as "Missed Knight Exchange" — 64% of minor-exchange fires were attacker≠victim.
    # A B-for-N (or N-for-B) is a distinct decision (bishop pair / good-vs-bad minor), so it gets its own
    # label. NxN -> Knight Exchange, BxB -> Bishop Exchange, BxN/NxB -> Bishop-Knight Exchange. (GH #28, Sam.)
    aname = PIECE_NAME[attacker.piece_type] if attacker else pname
    if {attacker.piece_type, victim.piece_type} == {chess.KNIGHT, chess.BISHOP}:
        return [("Missed Bishop-Knight Exchange", "missed",
                 f"best {m.best_san} = trade {aname.lower()} for {pname.lower()}")]
    return [(f"Missed {pname} Exchange", "missed", f"best {m.best_san} = even trade of {pname.lower()}")]


def greedy_capture(m):
    """The PLAYED move grabs material when the BEST move was QUIET (non-capture, non-check).

    The one real, teachable idea mined from the deleted catch-alls (Bad/Wrong Capture, ply analysis):
    "you took a pawn/piece when a quiet positional move was stronger." Usually a pawn grab (63% of
    coherent fires). The grabbed piece goes in the EVIDENCE string, not the label — one unified tag,
    same convention as the Bishop-Knight Exchange rename (GH #28). Replaces the 5 redundant outcome
    catch-alls that were 86-100% co-fire duplicates and mislabeled missed tactics (GH #29).

    Distinct from capture_or_exchange (best IS a capture you missed) — here best is quiet."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None:
        return []
    if not b.is_capture(pm):                       # the played move must itself be a grab
        return []
    if b.is_capture(bm) or b.gives_check(bm):      # best must be QUIET (not a capture/check)
        return []
    victim = b.piece_at(pm.to_square)
    pname = PIECE_NAME[victim.piece_type].lower() if victim else "pawn"   # None = en passant -> pawn
    return [("Greedy Capture", "played",
             f"grabbed a {pname} ({m.played_san}); best was the quiet {m.best_san}")]


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
    # Delayed (non-immediate) material loss: the old "Lost Material to Combination" catch-all was
    # 93% co-fire-redundant (GH #29) — the multi-move tactic that wins the material is already named
    # by the motif "allowed" detectors (deflection/zwischenzug/...) in tagger.py. Don't double-tag.
    return []


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
    if abs(chess.square_file(pm.from_square) - chess.square_file(ks)) <= 1:
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


def _doubled_files(files):
    """Set of files holding >=2 friendly pawns (doubled), from a _pawn_files map."""
    return {f for f, ranks in files.items() if len(ranks) >= 2}


def _isolated_files(files):
    """Set of files holding a friendly pawn with NO friendly pawn on either adjacent file."""
    present = set(files)
    return {f for f in present if (f - 1) not in present and (f + 1) not in present}


def pawn_structure(m):
    """A pawn move that NEWLY CREATED a structural weakness (doubled or isolated pawn) the best move
    would have avoided. Semantics (2026-06-23 rewrite — the old version had two bugs):
      * A defect is tagged only if it is present AFTER the played move, ABSENT before it, AND absent
        after the BEST move. This "newly created by the blunder, avoidable by the best move" rule is
        what makes it honest — it replaces the old (broken) guards.
      * Bug it fixes #1: doubled pawns are created by CAPTURES (exd5 doubles the d-file), but the old
        code excluded all captures via guard 1 -> "Created Doubled Pawn" was UNREACHABLE (0 fires in
        55k). We no longer blanket-skip captures; the before/after/best comparison handles recaptures
        (if the best move recaptures the same way, the defect is in best_af too -> not tagged).
      * Bug it fixes #2: the old isolated check fired whenever an isolated pawn EXISTED on the to-file
        after the move — even if the pawn was already isolated and merely advanced (98.5% of fires).
        Now we require the isolation to be ABSENT before the move (newly created)."""
    pm = _played_move(m)
    if pm is None or m.board_before.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    before = m.board_before
    after = before.copy(); after.push(pm)
    bf = _pawn_files(before, m.mover); af = _pawn_files(after, m.mover)
    before_dbl, after_dbl = _doubled_files(bf), _doubled_files(af)
    before_iso, after_iso = _isolated_files(bf), _isolated_files(af)

    # best move's resulting structure — a defect the best move ALSO creates isn't blunder-caused
    best_dbl = best_iso = set()
    bm = _best_move(m)
    if bm is not None:
        try:
            ab = before.copy(); ab.push(bm)
            bff = _pawn_files(ab, m.mover)
            best_dbl, best_iso = _doubled_files(bff), _isolated_files(bff)
        except Exception:
            pass

    out = []
    # NEW doubled file: doubled after, not before, and the best move doesn't also double it
    new_dbl = (after_dbl - before_dbl) - best_dbl
    if new_dbl:
        f = sorted(new_dbl)[0]
        out.append(("Created Doubled Pawn", "played", f"move doubled pawns on the {chr(97 + f)}-file (best move avoids it)"))
    # NEW isolated file: isolated after, not before, and the best move doesn't also isolate it
    new_iso = (after_iso - before_iso) - best_iso
    if new_iso:
        f = sorted(new_iso)[0]
        out.append(("Created Isolated Pawn", "played", f"move left an isolated pawn on the {chr(97 + f)}-file (best move avoids it)"))
    return out


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


# ---------- endgame mistake detectors ----------
# All fire on the SAME rule: the best move exhibits the theme AND the played move did not. No causal
# gate ("is this the real mistake?") — fire when the theme is present; noise is pruned by reviewing
# real outputs (Sam's call). These are drill-filter categories: "would someone want to drill positions
# of this type?" — so marking the type whenever present is the goal. (See findings/spec 2026-06-13.)

def _is_endgame(m):
    """Reuse phase()'s endgame determination (npieces<=12 or non-pawns<=4) — single source."""
    return phase(m)[0][0] == "Endgame"


def missed_king_activity(m):
    """Endgame: best move is a non-check king move toward the center OR the enemy pawns, and the played
    move wasn't that. Escaping a check is defense, not activity — excluded."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if b.is_check():
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.KING:
        return []
    toward_center = U.center_distance(bm.to_square) < U.center_distance(bm.from_square)
    toward_pawns = (U.nearest_enemy_pawn_distance(b, bm.to_square, m.mover)
                    < U.nearest_enemy_pawn_distance(b, bm.from_square, m.mover))
    if not (toward_center or toward_pawns):
        return []
    where = "center" if toward_center else "the enemy pawns"
    return [("Missed King Activity", "missed", f"best {m.best_san} activates the king toward {where}")]


def lost_opposition(m):
    """King-and-pawn endgame: best move is a king move that takes DIRECT opposition (kings 2 squares
    apart on the same file or rank), and the played move didn't."""
    b = m.board_before
    if not U.is_pawn_only_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.KING:
        return []
    ek = b.king(not m.mover)
    if ek is None:
        return []
    same_line = (chess.square_file(bm.to_square) == chess.square_file(ek)
                 or chess.square_rank(bm.to_square) == chess.square_rank(ek))
    if chess.square_distance(bm.to_square, ek) == 2 and same_line:
        return [("Lost the Opposition", "missed", f"best {m.best_san} takes the opposition")]
    return []


def missed_passed_pawn(m):
    """Best move is a pawn move that results in a passed pawn (creates a new one or advances an existing
    passer), and the played move wasn't. No phase gate — passers matter before the endgame too."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    after = b.copy(); after.push(bm)
    if U.is_passed_pawn(after, bm.to_square, m.mover):
        return [("Missed Passed Pawn", "missed", f"best {m.best_san} makes/advances a passed pawn")]
    return []


def rook_behind_passer(m):
    """Endgame: best move puts a rook on a file containing a passed pawn (either color), BEHIND that
    pawn (Tarrasch — behind your own to push it, behind the enemy's to stop it), and the played didn't."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    tf = chess.square_file(bm.to_square); tr = chess.square_rank(bm.to_square)
    for sq, pc in b.piece_map().items():
        if pc.piece_type != chess.PAWN or chess.square_file(sq) != tf:
            continue
        if not U.is_passed_pawn(b, sq, pc.color):
            continue
        pr = chess.square_rank(sq)
        behind = (pc.color == chess.WHITE and tr < pr) or (pc.color == chess.BLACK and tr > pr)
        if behind:
            return [("Rook Behind Passer", "missed", f"best {m.best_san} puts the rook behind the passed pawn")]
    return []


# ---------- positional: plan execution ----------

def missed_pawn_break(m):
    """Best move is a pawn advance that creates tension (adjacent to an enemy pawn or opens a file),
    and the played move isn't a pawn advance toward the same goal. Covers central breaks, kingside
    storms, and minority attacks."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    # the played move is also a pawn advance to a nearby file — player tried, just wrong pawn
    if pm and b.piece_type_at(pm.from_square) == chess.PAWN:
        if abs(chess.square_file(pm.to_square) - chess.square_file(bm.to_square)) <= 1:
            return []
    # check: does the pawn advance create tension (enemy pawn adjacent)?
    to_f = chess.square_file(bm.to_square)
    to_r = chess.square_rank(bm.to_square)
    creates_tension = False
    for adj_f in (to_f - 1, to_f, to_f + 1):
        if not (0 <= adj_f <= 7):
            continue
        for adj_r in (to_r - 1, to_r, to_r + 1):
            if not (0 <= adj_r <= 7):
                continue
            pc = b.piece_at(chess.square(adj_f, adj_r))
            if pc and pc.piece_type == chess.PAWN and pc.color != m.mover:
                creates_tension = True
                break
        if creates_tension:
            break
    # A capture only counts as a pawn break if it's pawn-takes-PAWN (or en passant). pawn-takes-PIECE
    # is winning material, not a structural break — it was 53% of capture-fires, mislabeling "grab the
    # hanging bishop" as "Missed Pawn Break". Let capture_or_exchange name those. (GH #28-class fix, Sam.)
    if b.is_capture(bm):
        if b.is_en_passant(bm) or (b.piece_at(bm.to_square) and b.piece_at(bm.to_square).piece_type == chess.PAWN):
            creates_tension = True
    if not creates_tension:
        return []
    # determine the type of break
    opp_king = b.king(not m.mover)
    if opp_king is not None and abs(to_f - chess.square_file(opp_king)) <= 2:
        kind = "kingside" if chess.square_file(opp_king) >= 4 else "queenside"
        return [("Missed Pawn Break", "missed", f"best {m.best_san} = {kind} pawn storm")]
    return [("Missed Pawn Break", "missed", f"best {m.best_san} = pawn break creating tension")]


def missed_tempo_push(m):
    """Best move is a pawn advance (non-capture) that attacks an enemy minor/major piece which was NOT
    attacked before the push — a tempo gain that dislodges the piece. The pawn must be safe to push
    (not just hanging itself). Distinct from Missed Pawn Break (structural tension): this kicks a piece.
    Examples: d5 hitting Nc6, e5 hitting Nf6."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    if b.is_capture(bm):           # a capture is a break/grab, not a tempo push
        return []
    if bm.promotion is not None:   # promotion: the new piece attacks, not the pawn — not a tempo push
        return []
    to_sq = bm.to_square
    after = b.copy(); after.push(bm)
    # the pushed pawn must survive (not a free pawn sac); allow if defended or undefended-but-unattacked
    if after.is_attacked_by(not m.mover, to_sq) and not after.is_attacked_by(m.mover, to_sq):
        return []
    # what does the pawn now attack that it didn't before?
    KICKABLE = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    for sq in after.attacks(to_sq):
        victim = after.piece_at(sq)
        if victim is None or victim.color == m.mover or victim.piece_type not in KICKABLE:
            continue
        # was this enemy piece already attacked by one of our pawns before the push? then no new tempo
        was_attacked = False
        for asq in b.attackers(m.mover, sq):
            if b.piece_type_at(asq) == chess.PAWN:
                was_attacked = True
                break
        if not was_attacked:
            vname = PIECE_NAME[victim.piece_type].lower()
            return [("Missed Tempo Push", "missed",
                     f"best {m.best_san} attacks the {vname} on {chess.square_name(sq)}, gaining tempo")]
    return []


def missed_open_file(m):
    """Best move places a rook on an open or half-open file, and the played move doesn't."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    to_file = chess.square_file(bm.to_square)
    # check if the file is open (no pawns) or half-open (no friendly pawns)
    friendly_pawn_on_file = any(
        p.piece_type == chess.PAWN and p.color == m.mover and chess.square_file(sq) == to_file
        for sq, p in b.piece_map().items()
    )
    enemy_pawn_on_file = any(
        p.piece_type == chess.PAWN and p.color != m.mover and chess.square_file(sq) == to_file
        for sq, p in b.piece_map().items()
    )
    if friendly_pawn_on_file:
        return []  # closed for us
    # rook is already on this file?
    if chess.square_file(bm.from_square) == to_file:
        return []  # just a rook lift, not file occupation
    kind = "open" if not enemy_pawn_on_file else "half-open"
    return [("Missed Open File", "missed", f"best {m.best_san} = rook to {kind} {chr(97+to_file)}-file")]


def premature_trade(m):
    """Played move is a capture that leads to a trade (opponent recaptures), but the engine preferred
    maintaining tension. Only fires when the player had an eval advantage before the trade."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or pm == bm:
        return []
    if not b.is_capture(pm):
        return []
    # player must have been at least slightly better before the trade (mover POV)
    if m.eval_before is None:
        return []
    cp_before = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    if cp_before < 30:
        return []  # not clearly better — trade might be fine
    # check: does the opponent recapture on the same square? (from refutation line)
    if not m.refutation_san or len(m.refutation_san) < 1:
        return []
    after = b.copy(); after.push(pm)
    try:
        recapture = after.parse_san(m.refutation_san[0])
        if recapture.to_square != pm.to_square:
            return []  # opponent's reply isn't a recapture → not a trade
    except Exception:
        return []
    # the best move should NOT be a capture of the same piece (then it's "wrong capture", not premature trade)
    if b.is_capture(bm) and bm.to_square == pm.to_square:
        return []
    victim = b.piece_at(pm.to_square)
    pname = PIECE_NAME[victim.piece_type] if victim else "piece"
    return [("Premature Trade", "played", f"traded {pname.lower()} ({m.played_san}) while ahead; tension was an asset")]


def missed_prophylaxis(m):
    """The opponent's first move in the refutation line is a strong positional threat (pawn break,
    piece to outpost, or attack) that the best move would have prevented. Fires when best move
    directly contests or blocks the threat square."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or pm == bm:
        return []
    # Prophylaxis is QUIET prevention. If the best move is a capture, it's winning material / a tactic,
    # not prophylaxis — that was 50% of fires (best move grabs a piece, mislabeled "Missed Prophylaxis").
    # Let the material/tactic detectors name those. (GH #28-class fix, Sam.)
    if b.is_capture(bm):
        return []
    if not m.refutation_san or len(m.refutation_san) < 1:
        return []
    # the opponent's threat: first move of refutation line
    after_played = b.copy(); after_played.push(pm)
    try:
        threat = after_played.parse_san(m.refutation_san[0])
    except Exception:
        return []
    threat_sq = threat.to_square
    # does the best move directly address the threat square? (moves to it, controls it, or blocks it)
    if bm.to_square == threat_sq:
        return [("Missed Prophylaxis", "missed",
                 f"best {m.best_san} prevents opponent's {m.refutation_san[0]}")]
    # best move controls the threat square
    after_best = b.copy(); after_best.push(bm)
    if after_best.is_attacked_by(m.mover, threat_sq) and not b.is_attacked_by(m.mover, threat_sq):
        return [("Missed Prophylaxis", "missed",
                 f"best {m.best_san} controls the square opponent wants ({m.refutation_san[0]})")]
    return []


def missed_piece_activation(m):
    """Best move repositions a minor piece or rook (not a capture, not a king move) to a square with
    significantly more mobility/influence, and the played move doesn't address the same piece."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    piece_type = b.piece_type_at(bm.from_square)
    if piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        return []
    if b.is_capture(bm):
        return []  # captures are handled by capture predicates
    # the piece currently has low mobility (few legal destinations from its square)
    current_mobility = 0
    for sq in chess.SQUARES:
        test = chess.Move(bm.from_square, sq)
        if test in b.legal_moves:
            current_mobility += 1
    if current_mobility > 4:
        return []  # piece isn't really stuck
    # after the best move, piece has better mobility
    after = b.copy(); after.push(bm)
    new_mobility = 0
    for sq in chess.SQUARES:
        test = chess.Move(bm.to_square, sq)
        if test in after.legal_moves:
            new_mobility += 1
    if new_mobility <= current_mobility:
        return []  # didn't actually improve
    pname = PIECE_NAME[piece_type]
    return [("Missed Piece Activation", "missed",
             f"best {m.best_san} activates the {pname.lower()} ({current_mobility}→{new_mobility} squares)")]


def wrong_pawn_race(m):
    """Endgame with passed pawns on both sides: best move is a king or pawn move that wins the race,
    played move goes in a different direction and loses the race."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or pm == bm:
        return []
    # both sides should have passed pawns or potential passers
    our_passers = [sq for sq, p in b.piece_map().items()
                   if p.piece_type == chess.PAWN and p.color == m.mover and U.is_passed_pawn(b, sq, m.mover)]
    their_passers = [sq for sq, p in b.piece_map().items()
                     if p.piece_type == chess.PAWN and p.color != m.mover and U.is_passed_pawn(b, sq, not m.mover)]
    if not our_passers and not their_passers:
        return []
    # best and played should both be king or pawn moves but in different directions
    bm_type = b.piece_type_at(bm.from_square)
    pm_type = b.piece_type_at(pm.from_square)
    if bm_type not in (chess.KING, chess.PAWN) or pm_type not in (chess.KING, chess.PAWN):
        return []
    # different direction: files diverge or ranks diverge meaningfully
    if abs(chess.square_file(bm.to_square) - chess.square_file(pm.to_square)) < 2:
        if abs(chess.square_rank(bm.to_square) - chess.square_rank(pm.to_square)) < 2:
            return []  # same direction, just a tempo difference — not a "wrong race"
    return [("Wrong Pawn Race", "missed", f"best {m.best_san} wins the race; {m.played_san} loses a tempo")]


# ---------- rook endgame technique ----------

def _is_rook_endgame(board):
    """K+R+P only (each side may have rook(s) and pawns, no other pieces)."""
    for p in board.piece_map().values():
        if p.piece_type not in (chess.KING, chess.ROOK, chess.PAWN):
            return False
    return any(p.piece_type == chess.ROOK for p in board.piece_map().values())


def rook_to_seventh(m):
    """Rook endgame: best move puts a rook on the 7th rank (2nd for Black), and the played didn't.
    The 7th rank rook is one of the most powerful endgame concepts — it cuts the king off and
    attacks pawns from behind."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not _is_rook_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    seventh = 6 if m.mover == chess.WHITE else 1
    if chess.square_rank(bm.to_square) != seventh:
        return []
    if chess.square_rank(bm.from_square) == seventh:
        return []  # rook already on 7th, just sliding along it
    return [("Missed Rook to 7th", "missed", f"best {m.best_san} brings the rook to the 7th rank")]


def rook_cut_off_king(m):
    """Rook endgame: best move places the rook on a file or rank between the enemy king and our
    passed pawn / promotion square, cutting the king off."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not _is_rook_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    ek = b.king(not m.mover)
    if ek is None:
        return []
    ek_file = chess.square_file(ek)
    ek_rank = chess.square_rank(ek)
    to_file = chess.square_file(bm.to_square)
    to_rank = chess.square_rank(bm.to_square)
    our_passers = [sq for sq, p in b.piece_map().items()
                   if p.piece_type == chess.PAWN and p.color == m.mover and U.is_passed_pawn(b, sq, m.mover)]
    if not our_passers:
        return []
    for passer_sq in our_passers:
        pf = chess.square_file(passer_sq)
        # File cut-off: rook lands on a file strictly between king and passer
        if (ek_file < to_file <= pf) or (pf <= to_file < ek_file):
            return [("Missed Rook Cut-Off", "missed",
                     f"best {m.best_san} cuts the enemy king off from the passed pawn")]
        # Rank cut-off: rook on a rank between king and promotion square
        promo_rank = 7 if m.mover == chess.WHITE else 0
        if m.mover == chess.WHITE and ek_rank < to_rank:
            return [("Missed Rook Cut-Off", "missed",
                     f"best {m.best_san} cuts the enemy king off from the promotion square")]
        if m.mover == chess.BLACK and to_rank < ek_rank:
            return [("Missed Rook Cut-Off", "missed",
                     f"best {m.best_san} cuts the enemy king off from the promotion square")]
    return []


def _rook_mobility(board, square, color):
    """Count squares a rook on `square` attacks (regardless of whose turn it is)."""
    return len(board.attacks(square))


def missed_active_rook(m):
    """Rook endgame: best move significantly improves rook activity (mobility increase ≥4 squares),
    and the played move doesn't. Passive rook = endgame death."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not _is_rook_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    before_mobility = _rook_mobility(b, bm.from_square, m.mover)
    after = b.copy(); after.push(bm)
    after_mobility = _rook_mobility(after, bm.to_square, m.mover)
    gain = after_mobility - before_mobility
    if gain < 4:
        return []
    return [("Missed Active Rook", "missed",
             f"best {m.best_san} activates the rook ({before_mobility}→{after_mobility} squares)")]


def rook_endgame_blockade(m):
    """Rook endgame: best move places a piece (king or rook) directly in front of an enemy passed
    pawn, blockading it. The played move doesn't."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not _is_rook_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    # Find enemy passed pawns
    enemy = not m.mover
    fwd = 1 if enemy == chess.WHITE else -1  # enemy's advance direction
    for sq, p in b.piece_map().items():
        if p.piece_type != chess.PAWN or p.color != enemy:
            continue
        if not U.is_passed_pawn(b, sq, enemy):
            continue
        # The square directly in front of this passer (from enemy's perspective)
        block_sq = sq + 8 * fwd
        if not (0 <= block_sq <= 63):
            continue
        if bm.to_square == block_sq:
            piece_name = "king" if b.piece_type_at(bm.from_square) == chess.KING else "rook"
            return [("Missed Blockade", "missed",
                     f"best {m.best_san} blockades the enemy passer with the {piece_name}")]
    return []


def missed_connected_passers(m):
    """Endgame: best move creates or maintains connected passed pawns (passers on adjacent files),
    played move breaks the connection or doesn't create it."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    after_best = b.copy(); after_best.push(bm)
    # Count connected passers after best move
    def connected_passers(board, color):
        passers = sorted(sq for sq, p in board.piece_map().items()
                         if p.piece_type == chess.PAWN and p.color == color
                         and U.is_passed_pawn(board, sq, color))
        connected = 0
        for i in range(len(passers)):
            for j in range(i + 1, len(passers)):
                if abs(chess.square_file(passers[i]) - chess.square_file(passers[j])) == 1:
                    connected += 1
        return connected

    best_connected = connected_passers(after_best, m.mover)
    before_connected = connected_passers(b, m.mover)
    if best_connected <= before_connected:
        return []
    # Check played doesn't achieve the same
    after_played = b.copy(); after_played.push(pm)
    played_connected = connected_passers(after_played, m.mover)
    if played_connected >= best_connected:
        return []
    return [("Missed Connected Passers", "missed",
             f"best {m.best_san} creates connected passed pawns")]


# ---------- general endgame technique ----------

def bad_simplification(m):
    """Endgame: played move is a capture (simplifying) but the best move is NOT a capture.
    The player traded when they shouldn't have — giving up winning chances or activity."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if not b.is_capture(pm):
        return []
    if b.is_capture(bm):
        return []  # best is also a capture — this is about WHICH capture, not whether to capture
    return [("Bad Simplification", "played",
             f"{m.played_san} simplifies but best {m.best_san} keeps the tension")]


def trade_to_simplify(m):
    """Endgame: best move is a capture (simplifying to a won position) but the played move is not.
    The player missed that trading down was winning."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if not b.is_capture(bm):
        return []
    if b.is_capture(pm):
        return []  # player also captured — different tag territory
    return [("Missed Trade to Simplify", "missed",
             f"best {m.best_san} trades down to a simpler position")]


def wrong_king_direction(m):
    """Endgame: both the best and played moves are king moves but to significantly different squares.
    The player moved the king the wrong way — critical in K+P and rook endgames."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.KING:
        return []
    if b.piece_type_at(pm.from_square) != chess.KING:
        return []
    # Must go to meaningfully different squares (not just one square apart)
    if chess.square_distance(bm.to_square, pm.to_square) < 2:
        return []
    return [("Wrong King Direction", "missed",
             f"best {m.best_san} but played {m.played_san} — king went the wrong way")]


def outside_passer(m):
    """Endgame: best move creates or advances a passed pawn on the a/b or g/h files (an outside
    passer — the classic winning technique of drawing the enemy king away from the center)."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    to_f = chess.square_file(bm.to_square)
    if to_f not in (0, 1, 6, 7):  # a, b, g, h files only
        return []
    after = b.copy(); after.push(bm)
    if not U.is_passed_pawn(after, bm.to_square, m.mover):
        return []
    # Exclude if it was already a passed pawn before the move (just advancing an existing one
    # is covered by Missed Passed Pawn)
    if U.is_passed_pawn(b, bm.from_square, m.mover):
        return []
    return [("Missed Outside Passer", "missed",
             f"best {m.best_san} creates an outside passed pawn")]


def rook_to_open_file_endgame(m):
    """Endgame: best move puts a rook on a fully open file (no pawns of either color), and the
    played move doesn't. Rook activity on open files dominates endgames."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not any(p.piece_type == chess.ROOK for p in b.piece_map().values()):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    to_f = chess.square_file(bm.to_square)
    from_f = chess.square_file(bm.from_square)
    if to_f == from_f:
        return []  # staying on same file — not "going to" an open file
    # Check: destination file is fully open (no pawns)
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.PAWN and chess.square_file(sq) == to_f:
            return []
    return [("Missed Rook to Open File", "missed",
             f"best {m.best_san} puts the rook on the open {chr(97 + to_f)}-file")]


def push_to_promote(m):
    """Endgame: best move advances a pawn to the 6th rank or beyond (within 2 ranks of promotion),
    and the played move doesn't advance that pawn. The approach move before queening."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    to_r = chess.square_rank(bm.to_square)
    if m.mover == chess.WHITE and to_r < 5:
        return []  # not yet in the promotion zone (rank 6+)
    if m.mover == chess.BLACK and to_r > 2:
        return []
    # Don't fire if it's an actual promotion (that's Missed Promotion)
    if bm.promotion:
        return []
    return [("Missed Push to Promote", "missed",
             f"best {m.best_san} advances the pawn toward promotion")]


# ---------- opening/middlegame awareness ----------

def _development_count(board, color):
    """Count developed minor pieces (not on back rank) + castled king."""
    back_rank = 0 if color == chess.WHITE else 7
    developed = 0
    for sq, p in board.piece_map().items():
        if p.color != color:
            continue
        if p.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(sq) != back_rank:
                developed += 1
    # Castled counts as +1 development
    king_sq = board.king(color)
    if king_sq is not None:
        kf = chess.square_file(king_sq)
        if color == chess.WHITE and chess.square_rank(king_sq) == 0 and kf in (1, 2, 6):
            developed += 1
        elif color == chess.BLACK and chess.square_rank(king_sq) == 7 and kf in (1, 2, 6):
            developed += 1
    return developed


def pawn_grab_undeveloped(m):
    """Opening/early middlegame: played move is a pawn capture while own pieces are undeveloped
    (fewer than 4 minor pieces developed), and the best move is NOT a capture (i.e. best was
    development/castling/center control). The classic beginner trap — grabbing a pawn while
    the opponent develops with tempo."""
    b = m.board_before
    if b.fullmove_number > 15:
        return []
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    if not b.is_capture(pm):
        return []
    # Must be capturing a pawn (not winning a piece — that's just good)
    captured = b.piece_at(pm.to_square)
    if captured and captured.piece_type != chess.PAWN:
        return []
    if b.is_capture(bm):
        return []  # best is also a capture — not a "grab vs develop" choice
    dev = _development_count(b, m.mover)
    if dev >= 4:
        return []  # already well developed
    return [("Pawn Grab While Undeveloped", "played",
             f"{m.played_san} grabs a pawn but only {dev} pieces developed; best was {m.best_san}")]


def ignored_threat(m):
    """The opponent already had a concrete threat BEFORE our move (an undefended piece under attack),
    and the played move doesn't address it while the best move does. Only fires when the threat
    PRE-EXISTED — not when our move creates a new vulnerability."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    opp = not m.mover
    # Find pre-existing threats: our pieces that are attacked and undefended RIGHT NOW
    pre_threats = []
    for sq, p in b.piece_map().items():
        if p.color == m.mover and p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if b.is_attacked_by(opp, sq) and not b.is_attacked_by(m.mover, sq):
                pre_threats.append(sq)
    if not pre_threats:
        return []
    # Best move addresses the threat (moves the piece, captures attacker, or adds defender)
    best_addresses = False
    after_best = b.copy(); after_best.push(bm)
    for sq in pre_threats:
        if bm.from_square == sq:
            best_addresses = True; break
        if sq in b.attackers(opp, sq) and bm.to_square in b.attackers(opp, sq):
            best_addresses = True; break
        # After best, is the piece still hanging?
        if sq in after_best.piece_map() and after_best.is_attacked_by(m.mover, sq):
            best_addresses = True; break
        if sq not in after_best.piece_map():
            best_addresses = True; break  # piece moved or was traded
    if not best_addresses:
        return []
    # Played move does NOT address the threat
    after_played = b.copy(); after_played.push(pm)
    for sq in pre_threats:
        if pm.from_square == sq:
            return []  # player moved the threatened piece — they noticed
        if sq in after_played.piece_map() and after_played.is_attacked_by(m.mover, sq):
            return []  # player added a defender
    return [("Ignored Threat", "played",
             f"{m.played_san} ignores the hanging piece; {m.best_san} addresses it")]


def premature_attack(m):
    """Opening: played move is an aggressive move (piece toward enemy king / pawn storm on kingside)
    while own development is incomplete (<4 pieces out). The opponent punishes the overextension."""
    b = m.board_before
    if b.fullmove_number > 15:
        return []
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    dev = _development_count(b, m.mover)
    if dev >= 4:
        return []
    # Is the played move "attacking"? Piece moves toward enemy king half, or pawn storms
    piece_type = b.piece_type_at(pm.from_square)
    if piece_type == chess.KING:
        return []  # king moves aren't "attacks"
    enemy_king = b.king(not m.mover)
    if enemy_king is None:
        return []
    ek_file = chess.square_file(enemy_king)
    to_file = chess.square_file(pm.to_square)
    to_rank = chess.square_rank(pm.to_square)
    # Attack = moving toward enemy king's side of the board
    enemy_half_rank = (4, 5, 6, 7) if m.mover == chess.WHITE else (0, 1, 2, 3)
    if to_rank not in enemy_half_rank:
        return []
    # Pawn storms: pawn advancing on kingside when enemy king is kingside
    if piece_type == chess.PAWN:
        if abs(to_file - ek_file) > 2:
            return []  # not near the king
    # Best move should be developmental (back rank piece moving out, or castling)
    best_piece = b.piece_type_at(bm.from_square)
    is_dev_move = False
    if best_piece in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(bm.from_square) == (0 if m.mover == chess.WHITE else 7):
        is_dev_move = True
    if b.is_castling(bm):
        is_dev_move = True
    if not is_dev_move:
        return []
    return [("Premature Attack", "played",
             f"{m.played_san} attacks with only {dev} pieces developed; better to develop with {m.best_san}")]


def missed_defensive_resource(m):
    """Position is under attack (opponent threatens material or mate), a defensive move exists
    (the best move), but the player plays something that doesn't address the threat. Distinct from
    'Ignored Threat' in that here the player IS trying to do something but picks the wrong defense."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    # Must be under threat: opponent attacks one of our pieces right now
    opp = not m.mover
    under_attack = []
    for sq, p in b.piece_map().items():
        if p.color == m.mover and p.piece_type != chess.PAWN and p.piece_type != chess.KING:
            if b.is_attacked_by(opp, sq) and not b.is_attacked_by(m.mover, sq):
                under_attack.append((sq, p))
    if not under_attack and not b.is_check():
        return []
    # Best move is defensive: it moves the attacked piece, blocks, or captures the attacker
    best_defends = False
    if b.is_check():
        best_defends = True  # any legal move in check is "defensive"
    else:
        for sq, p in under_attack:
            if bm.from_square == sq:
                best_defends = True  # moves the threatened piece
                break
            if bm.to_square in b.attackers(opp, sq):
                best_defends = True  # captures an attacker
                break
            # Interposes or adds defender
            if b.is_attacked_by(m.mover, sq) or bm.to_square == sq:
                best_defends = True
                break
    if not best_defends:
        return []
    # Played move does NOT defend
    played_defends = False
    if b.is_check():
        played_defends = True  # must address check
    else:
        for sq, p in under_attack:
            if pm.from_square == sq:
                played_defends = True
                break
    if played_defends:
        return []  # player tried to defend, just picked wrong defense
    return [("Missed Defensive Resource", "missed",
             f"under attack; best {m.best_san} defends but played {m.played_san} ignores")]


def missed_faster_mate(m):
    """Had a forced mate (eval_before is mate) and played a move that's still winning but not
    the fastest mate. Distinct from 'Missed Mate' which fires when you miss mate entirely."""
    if m.eval_before is None or m.eval_after is None:
        return []
    eb_mover = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    ea_mover = m.eval_after if m.mover == chess.WHITE else -m.eval_after
    # Must have mate before (mover has forced mate)
    if eb_mover < 9000:
        return []
    # After played move, still winning big (but not necessarily mate, or longer mate)
    # If eval_after is also mate for mover, it's a "slower mate" (still fine but suboptimal)
    # If eval_after is just +big (not mate), player lost the mate
    # Only fire if NOT already tagged "Missed Mate" (which fires when mate exists but you don't
    # play toward it at all — we want the "played okay but not optimal" case)
    if ea_mover < 500:
        return []  # lost too much — this is "Missed Mate" territory, not "missed faster"
    if ea_mover >= 9000:
        return []  # still mate — just slower; too nitpicky to flag
    # Had mate, played something still winning (+500 to +8999) but not mate
    return [("Missed Faster Mate", "missed",
             f"had forced mate but played {m.played_san} (still winning but slower)")]


# ---------- tactical patterns ----------

def missed_battery(m):
    """Best move aligns two heavy/sliding pieces (Q+R on file, Q+B on diagonal, R+R on file)
    creating a battery that ATTACKS an enemy piece or king. Battery pointing at nothing = no fire."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    mover_piece = b.piece_type_at(bm.from_square)
    if mover_piece not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
        return []
    after = b.copy(); after.push(bm)
    to_sq = bm.to_square
    to_f = chess.square_file(to_sq)
    to_r = chess.square_rank(to_sq)
    opp = not m.mover
    for sq, p in after.piece_map().items():
        if p.color != m.mover or sq == to_sq:
            continue
        if p.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        sq_f = chess.square_file(sq)
        sq_r = chess.square_rank(sq)
        aligned = False
        # Same file
        if sq_f == to_f and p.piece_type in (chess.QUEEN, chess.ROOK) and mover_piece in (chess.QUEEN, chess.ROOK):
            between = chess.SquareSet.between(sq, to_sq)
            if not any(after.piece_at(s) for s in between):
                aligned = True
        # Same rank
        elif sq_r == to_r and p.piece_type in (chess.QUEEN, chess.ROOK) and mover_piece in (chess.QUEEN, chess.ROOK):
            between = chess.SquareSet.between(sq, to_sq)
            if not any(after.piece_at(s) for s in between):
                aligned = True
        # Same diagonal
        elif abs(sq_f - to_f) == abs(sq_r - to_r) and sq != to_sq:
            if p.piece_type in (chess.QUEEN, chess.BISHOP) and mover_piece in (chess.QUEEN, chess.BISHOP):
                between = chess.SquareSet.between(sq, to_sq)
                if not any(after.piece_at(s) for s in between):
                    aligned = True
        if not aligned:
            continue
        # Battery exists — does it attack an enemy piece or king?
        front_attacks = after.attacks(to_sq)
        for target_sq in front_attacks:
            target = after.piece_at(target_sq)
            if target and target.color == opp and target.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
                return [("Missed Battery", "missed", f"best {m.best_san} creates a battery attacking the {chess.piece_name(target.piece_type)}")]
    return []


def missed_overloading(m):
    """Best move attacks a piece that is the sole defender of another piece (overloading the
    defender), and the played move doesn't exploit this. The defender can't protect both."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    opp = not m.mover
    # After best move: does it attack an opponent piece that is defending another piece?
    after = b.copy(); after.push(bm)
    target_sq = bm.to_square
    # What opponent pieces does the moved piece now attack?
    moved_attacks = after.attacks(target_sq)
    for victim_sq in moved_attacks:
        victim = after.piece_at(victim_sq)
        if not victim or victim.color != opp:
            continue
        if victim.piece_type == chess.PAWN:
            continue
        # Is this victim ALSO defending something else valuable?
        victim_defends = after.attacks(victim_sq)  # squares the victim attacks (i.e. defends)
        for defended_sq in victim_defends:
            defended = after.piece_at(defended_sq)
            if not defended or defended.color != opp or defended_sq == victim_sq:
                continue
            if defended.piece_type in (chess.PAWN, chess.KING):
                continue
            # Is the victim the SOLE defender of this piece?
            defenders = after.attackers(opp, defended_sq)
            if len(defenders) == 1 and victim_sq in defenders:
                # The victim is overloaded: attacked by our piece AND sole defender of another
                return [("Missed Overloading", "missed",
                         f"best {m.best_san} overloads the defender")]
    return []


def missed_desperado(m):
    """A piece is about to be captured (attacked and undefended or losing an exchange), and the
    best move uses it to capture something first (desperado — cash in before you lose it).
    The played move doesn't use the doomed piece."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    # Best move must be a capture by a piece that is currently under attack
    if not b.is_capture(bm):
        return []
    from_sq = bm.from_square
    mover_piece = b.piece_type_at(from_sq)
    if mover_piece == chess.KING:
        return []
    opp = not m.mover
    # Is the moving piece currently attacked by opponent?
    if not b.is_attacked_by(opp, from_sq):
        return []
    # Is it undefended or would lose the exchange? (attacked by lower-value piece)
    defenders = b.attackers(m.mover, from_sq)
    attackers = b.attackers(opp, from_sq)
    if not attackers:
        return []
    # Simple heuristic: piece is "doomed" if attacked and either undefended or attacked by lower-value
    piece_val = VAL.get(mover_piece, 0)
    min_attacker_val = min(VAL.get(b.piece_type_at(sq), 9) for sq in attackers)
    undefended = len(defenders) == 0
    losing_exchange = min_attacker_val < piece_val
    if not (undefended or losing_exchange):
        return []
    # Played move must NOT be using this same piece (otherwise player saw the desperado, just chose wrong target)
    if pm.from_square == from_sq:
        return []
    pname = PIECE_NAME.get(mover_piece, "piece")
    return [("Missed Desperado", "missed",
             f"best {m.best_san} cashes in the doomed {pname.lower()} before losing it")]


def missed_doubled_rooks(m):
    """Best move doubles rooks on a file (both rooks on the same open/semi-open file), and the
    played move doesn't. Doubled rooks are a powerful battery for controlling files."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    # After best move, are both rooks on the same file?
    after = b.copy(); after.push(bm)
    to_f = chess.square_file(bm.to_square)
    rooks_on_file = [sq for sq, p in after.piece_map().items()
                     if p.piece_type == chess.ROOK and p.color == m.mover and chess.square_file(sq) == to_f]
    if len(rooks_on_file) < 2:
        return []
    # Were they already doubled before?
    rooks_before = [sq for sq, p in b.piece_map().items()
                    if p.piece_type == chess.ROOK and p.color == m.mover and chess.square_file(sq) == to_f]
    if len(rooks_before) >= 2:
        return []  # already doubled
    return [("Missed Doubled Rooks", "missed",
             f"best {m.best_san} doubles the rooks on the {chr(97+to_f)}-file")]


def allowed_battery(m):
    """Played move allows the opponent to create a battery (Q+R on file, Q+B on diagonal) in the
    refutation that best move would have prevented."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    opp = not m.mover
    after_played = b.copy(); after_played.push(pm)
    # Check if opponent can now create a battery (two sliding pieces aligned with nothing between)
    for mv in after_played.legal_moves:
        piece_type = after_played.piece_type_at(mv.from_square)
        if piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        # After opponent plays this move, do they have aligned pieces?
        test = after_played.copy(); test.push(mv)
        to_sq = mv.to_square
        to_f = chess.square_file(to_sq)
        to_r = chess.square_rank(to_sq)
        for sq, p in test.piece_map().items():
            if p.color != opp or sq == to_sq:
                continue
            if p.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                continue
            sq_f = chess.square_file(sq)
            sq_r = chess.square_rank(sq)
            # Same file
            if sq_f == to_f and p.piece_type in (chess.QUEEN, chess.ROOK) and piece_type in (chess.QUEEN, chess.ROOK):
                between = chess.SquareSet.between(sq, to_sq)
                if not any(test.piece_at(s) for s in between):
                    # Was this battery possible before our move? If yes, not our fault.
                    after_best = b.copy(); after_best.push(bm)
                    if mv in after_best.legal_moves:
                        continue  # opponent could do it regardless
                    return [("Allowed Battery", "allowed", f"{m.played_san} allows opponent to build a battery")]
            # Same diagonal
            if abs(sq_f - to_f) == abs(sq_r - to_r) and sq != to_sq:
                if p.piece_type in (chess.QUEEN, chess.BISHOP) and piece_type in (chess.QUEEN, chess.BISHOP):
                    between = chess.SquareSet.between(sq, to_sq)
                    if not any(test.piece_at(s) for s in between):
                        after_best = b.copy(); after_best.push(bm)
                        if mv in after_best.legal_moves:
                            continue
                        return [("Allowed Battery", "allowed", f"{m.played_san} allows opponent to build a battery")]
    return []


def allowed_overloading(m):
    """Played move leaves one of our pieces overloaded (defending two things), and the opponent
    can exploit it. Best move would have avoided this."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    opp = not m.mover
    after_played = b.copy(); after_played.push(pm)
    # Find our pieces that are now sole defenders of multiple things
    for sq, p in after_played.piece_map().items():
        if p.color != m.mover or p.piece_type in (chess.PAWN, chess.KING):
            continue
        # What does this piece defend?
        defended_pieces = []
        for d_sq in after_played.attacks(sq):
            dp = after_played.piece_at(d_sq)
            if dp and dp.color == m.mover and dp.piece_type not in (chess.PAWN, chess.KING):
                # Is this piece the sole defender?
                defenders = after_played.attackers(m.mover, d_sq)
                if len(defenders) == 1 and sq in defenders:
                    defended_pieces.append(d_sq)
        if len(defended_pieces) < 2:
            continue
        # This piece defends 2+ things alone — is it also attacked?
        if after_played.is_attacked_by(opp, sq):
            # Check best move doesn't have this problem
            after_best = b.copy(); after_best.push(bm)
            best_defenders = after_best.attackers(m.mover, defended_pieces[0]) if defended_pieces[0] in after_best.piece_map() else chess.SquareSet()
            if len(best_defenders) > 1:
                return [("Allowed Overloading", "allowed",
                         f"{m.played_san} leaves a piece overloaded (defending two targets)")]
    return []


def allowed_doubled_rooks(m):
    """Played move allows the opponent to double their rooks on an open file that best move
    would have prevented (e.g. by contesting the file first)."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    opp = not m.mover
    after_played = b.copy(); after_played.push(pm)
    # Does opponent now have (or can immediately achieve) doubled rooks on a file?
    opp_rooks = [(sq, chess.square_file(sq)) for sq, p in after_played.piece_map().items()
                 if p.piece_type == chess.ROOK and p.color == opp]
    if len(opp_rooks) < 2:
        return []
    # Check if opponent can double on next move
    for mv in after_played.legal_moves:
        if after_played.piece_type_at(mv.from_square) != chess.ROOK:
            continue
        to_f = chess.square_file(mv.to_square)
        # Would this put both rooks on the same file?
        other_rook_on_file = any(f == to_f and sq != mv.from_square for sq, f in opp_rooks)
        if not other_rook_on_file:
            continue
        # Was this possible before our move?
        after_best = b.copy(); after_best.push(bm)
        if mv in after_best.legal_moves:
            continue  # could do it regardless
        return [("Allowed Doubled Rooks", "allowed",
                 f"{m.played_san} allows opponent to double rooks on the {chr(97+to_f)}-file")]
    return []


# ---------- registry ----------
ALL_PREDICATES = [
    phase, game_state, capture_or_exchange, greedy_capture, hung_material,
    king_in_center, lost_castling, exposed_king_pawn, pawn_structure,
    endgame_type, backward_pawn,
    missed_king_activity, lost_opposition, missed_passed_pawn, rook_behind_passer,
    rook_to_seventh, rook_cut_off_king, missed_active_rook, rook_endgame_blockade,
    missed_connected_passers,
    bad_simplification, trade_to_simplify, wrong_king_direction, outside_passer,
    rook_to_open_file_endgame, push_to_promote,
    pawn_grab_undeveloped, ignored_threat, premature_attack, missed_defensive_resource,
    missed_faster_mate,
    missed_battery, missed_overloading, missed_desperado, missed_doubled_rooks,
    allowed_battery, allowed_overloading, allowed_doubled_rooks,
    missed_pawn_break, missed_tempo_push, missed_open_file, premature_trade, missed_prophylaxis,
    missed_piece_activation, wrong_pawn_race,
]


def tag_predicates(m):
    out = []
    for fn in ALL_PREDICATES:
        try:
            out.extend(fn(m))
        except Exception:
            pass
    return out
