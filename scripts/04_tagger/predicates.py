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


def _outcome_band(winpct_val):
    """Coarse mover-POV band from a win% (0-100). Winning >=65, Losing <=35, else Even."""
    if winpct_val >= 65:
        return "Winning"
    if winpct_val <= 35:
        return "Losing"
    return "Even"


def conversion_outcome(m):
    """DESCRIPTIVE info tag naming the before->after RESULT band of the move (mover POV): e.g.
    "Winning → Losing", "Winning → Drawn", "Even → Losing". NOT a coaching lesson — it's what the
    SAE's diffuse features actually cluster on (OUTCOME / severity of the swing, not mistake type).
    Lets us NAME those features ("this feature = threw away a won game") even when there's no per-move
    concept to teach. (Sam, 2026-07-14: the ~575 genuinely-diffuse SAE features group by outcome, incl.
    a real 'conversion / squandered a win' theme — capture it descriptively, direction=info.)

    Bands from winpct() (mover POV, ±1200 mate clamp), same currency as win_drop. Only fires when the
    band actually CHANGES (a within-band wobble isn't a conversion event)."""
    if m.eval_before is None or m.eval_after is None:
        return []
    before_cp = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    after_cp = m.eval_after if m.mover == chess.WHITE else -m.eval_after
    b = _outcome_band(U.winpct(max(-1200, min(1200, before_cp))))
    a = _outcome_band(U.winpct(max(-1200, min(1200, after_cp))))
    if b == a:
        return []                                  # no band change -> not a conversion event
    # "Even" as an endpoint reads as "Drawn" when you LAND there (result), "Even" when you start there.
    a_word = "Drawn" if a == "Even" else a
    return [(f"{b} → {a_word}", "info", f"result swing {before_cp:+d}→{after_cp:+d}cp (mover POV)")]


def blunder_severity(m):
    """DESCRIPTIVE info tag: was this a SHARP blunder (one move decisively swings the result) or a
    SLOW BLEED (a small edge given up in a live, balanced position)? The axis the SAE's features
    actually split on (measured: 168 features are mostly big single-move drops, 217 mostly small).

    win%-drop currency (mover POV, same as win_drop). CRITICAL guard (Sam, 2026-07-14): a small
    win%-drop does NOT mean slow bleed — it also happens when the eval is SATURATED (you're +M5 and
    miss the mate: 99%→95% is a tiny drop but a big error). So 'Slow Bleed' requires the position to be
    roughly BALANCED before the move (|win% − 50| < 25); a small drop from a saturated eval is neither
    sharp nor bleed (it's 'inaccuracy while winning/losing' — no severity label)."""
    if m.eval_before is None or m.eval_after is None:
        return []
    drop = U.win_drop(m.eval_before, m.eval_after, m.mover)     # mover-POV win% given up, >=0
    before_cp = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    wp_before = U.winpct(max(-1200, min(1200, before_cp)))
    if drop >= 30:
        return [("Sharp Blunder", "info", f"one move gave up {drop:.0f}% win chance")]
    if drop < 15 and abs(wp_before - 50) < 25:                  # small drop AND not saturated
        return [("Slow Bleed", "info", f"gave up {drop:.0f}% from a balanced position")]
    return []


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
    # #52: greed = grabbing material you KEEP. If SEE says the capture LOSES material (recaptured at a
    # net loss), it's an unsound SACRIFICE, not a grab — do NOT tag Greedy Capture. (Greek-Gift Bxf7+ =
    # bishop-for-pawn, SEE ~-2: sheds material. Was the #45 conflation — 11 confident-wrong SAE features.)
    if U.static_exchange_eval(b, pm) < 0:
        return []
    victim = b.piece_at(pm.to_square)
    pname = PIECE_NAME[victim.piece_type].lower() if victim else "pawn"   # None = en passant -> pawn
    return [("Greedy Capture", "played",
             f"grabbed a {pname} ({m.played_san}); best was the quiet {m.best_san}")]


def unsound_sacrifice(m):
    """The PLAYED move is a material-SHEDDING capture (SEE < 0) into the enemy KING's zone — an
    unsound sacrifice (classic Greek-Gift Bxf7+/Bxh7+, or Bxh3/Bxg6 type). #52b.

    The complement of greedy_capture: greedy = SEE>=0 capture you KEEP; this = SEE<0 capture you SHED
    for an attack that (per the win_drop entry gate) doesn't work. Data-derived from the SAE 'unsound
    sac' feature cluster: 97% of the shedding captures land within 2 squares of the enemy king — the
    king proximity is what separates a real (failed) SACRIFICE from a plain hung piece / bad trade.
    'Unsound' is supplied by the entry gate (a SOUND sac keeps the eval up, so it's not a flagged
    mistake and never reaches here). Fires INSTEAD of the generic 'Hung <piece>' on these (a deeper,
    teachable concept); see _suppress_hung_under_sacrifice in tagger.py."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or not b.is_capture(pm):
        return []
    if U.static_exchange_eval(b, pm) >= 0:          # must SHED material — else it's a grab/trade
        return []
    ks = b.king(not m.mover)                          # enemy king square
    if ks is None or chess.square_distance(pm.to_square, ks) > 2:
        return []                                     # not aimed at the king -> plain hung piece, not a sac
    attacker = b.piece_at(pm.from_square)
    aname = PIECE_NAME[attacker.piece_type] if attacker else "piece"
    return [("Unsound Sacrifice", "played",
             f"sacrificed a {aname.lower()} ({m.played_san}) at the enemy king with no compensation")]


def pointless_check(m):
    """The PLAYED move is a CHECK, the BEST move is QUIET (non-check), and the check was a mistake
    (win_drop entry gate) — an aimless "hope check" that just chases the enemy king to a better square
    (or drops the checking piece) while a quiet improving move existed. #47.

    Data-derived: the SAE surfaced 5 features (f4/f85/f121/f129/f499) that share exactly this — Rc8+ /
    Qb1+ / Qf1+ / Qa8+ where the engine's move is a quiet king/rook/pawn move. The check gains nothing
    (no fork, no material, no mate — those keep the eval up and don't reach here past the gate); it just
    loses a tempo or walks the piece into trouble. Distinct from greedy_capture (played = a grab) and
    unsound_sacrifice (played = a shedding capture at the king). Here the played move is a NON-CAPTURE
    check — a capture that also gives check is a material decision, handled by the capture predicates.

    Excludes: played move that is itself a capture (let the material predicates own it), and any line
    where the best move is also a check (then it's a which-check-is-better calculation, not 'no check
    was called for')."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None:
        return []
    if not b.gives_check(pm):                          # played move must BE a check
        return []
    if b.is_capture(pm):                               # capturing-check = material decision, not a hope check
        return []
    if b.gives_check(bm):                              # best is also a check -> a check WAS called for
        return []
    piece = b.piece_at(pm.from_square)
    pname = PIECE_NAME[piece.piece_type].lower() if piece else "piece"
    return [("Pointless Check", "played",
             f"the check {m.played_san} ({pname}) achieves nothing; best was the quiet {m.best_san}")]


def missed_attacking_check(m):
    """The BEST move is a forcing CHECK (non-mate) the player MISSED — an attacking check that wins
    material or launches a decisive attack on the exposed king. The mirror of pointless_check (there
    the PLAYED move is an aimless check; here the BEST move is a strong check the player didn't play).

    Data-derived: the SAE (jr2048) surfaced ~7 features Opus labels 'Missed Queen Check on Exposed King'
    / 'Missed Forcing Check' (f136/508/704/1034/1530/1831/2019), 83% of whose top positions have best =
    a check. The tagger had NO detector for a missed WINNING check, so it mislabeled them Missed
    Overloading. Classic case: an early Qh4+/Qh5+ hitting the weakened e1-h4 / f7 diagonal.

    Teachable, not naked-rate: it only reaches here past the win_drop entry gate, so the check the
    player missed was worth a mistake-sized swing — it genuinely wins something (material or a decisive
    attack), not a routine check. Guards:
      - best is a CHECK; played is NOT that check.
      - best is NOT mate (Missed Mate owns forced mate) and NOT a capture (Missed Free X / capture_or_
        exchange owns a winning capture-check — this tag is for the QUIET forcing check, the thing no
        material tag sees).
      - best is NOT also matched by the player (bm != pm)."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if not b.gives_check(bm):                          # best must BE a check
        return []
    if b.gives_check(pm):                              # player also checked -> which-check calc, not a miss
        return []
    if b.is_capture(bm):                               # capture-check = material decision (capture_or_exchange)
        return []
    # This is a CO-TAG on the MECHANISM (a forcing quiet check you didn't play). A missed FORKING check
    # or a check that wins material is BOTH that specific tactic AND a missed check — we do NOT suppress
    # on fork/material; those co-fire and (by vote count) usually lead, but the check was still missed.
    # (Sam, 2026-07-13: "it doesn't mean a check wasn't missed.") The ONE exclusion: a check that
    # DELIVERS MATE is a mate, not an "attacking check" — Missed Mate owns it. (If Missed Mate under-
    # fires on mate-in-N, that's a separate label bug; a mate-in-N first move here is still a missed
    # check and correctly co-fires.)
    after = b.copy()
    try:
        after.push(bm)
    except Exception:
        return []
    if after.is_checkmate():                            # the check IS mate -> Missed Mate, not this
        return []
    piece = b.piece_at(bm.from_square)
    pname = PIECE_NAME[piece.piece_type].lower() if piece else "piece"
    return [("Missed Attacking Check", "missed",
             f"best {m.best_san} is a forcing {pname} check; you played {m.played_san}")]


def missed_greek_gift(m):
    """The BEST move is a BISHOP sacrifice with CHECK on a square next to the enemy castled king
    (classic Greek Gift Bxh7+/Bxf7+ and the Bxh2+/Bxf2+ mirror) that the player MISSED. The mirror of
    unsound_sacrifice (the PLAYED bad sac); here it's the missed SOUND sac — it reaches the tagger past
    the win_drop gate, so the sac genuinely works (a bad one wouldn't be the engine's best by a
    mistake-sized margin).

    Data-derived (SAE jr2048 f58/f623/f1786, Opus 'Missed Greek Gift Sacrifice'): best move is a bishop
    CAPTURE giving CHECK, SEE<0 (bishop-for-pawn = a sac), landing adjacent to the enemy king. Follows
    with Ng5+/Ne5 + queen — but we only need the sac signature; the follow-up is implied by the eval."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    pc = b.piece_at(bm.from_square)
    if not pc or pc.piece_type != chess.BISHOP:
        return []
    if not b.is_capture(bm) or not b.gives_check(bm):
        return []
    if U.static_exchange_eval(b, bm) >= 0:            # must SHED material (a sacrifice)
        return []
    ek = b.king(not m.mover)
    if ek is None or chess.square_distance(bm.to_square, ek) > 1:
        return []                                      # bishop must land NEXT to the enemy king
    return [("Missed Greek Gift", "missed",
             f"the bishop sacrifice {m.best_san} cracks the king; you played {m.played_san}")]


def missed_zwischenzug(m):
    """The PLAYED move is a capture, but the BEST line inserts a forcing CHECK first and captures the
    SAME target a move later — a zwischenzug (in-between move) the player skipped by capturing
    immediately. The capture was right; the ORDER was wrong (insert the check, THEN recapture).

    Data-derived (SAE jr2048 f23/f198/f1007/f1959, Opus 'move-order / zwischenzug error'). The tagger
    had no move-order detector — these were mislabeled Hung Material / Missed Desperado. Signature that
    cleanly separated them from plain hangs (f675/f1800 at 1/40): best line ply-1 is a CHECK (not the
    played move), and the played capture's TARGET square is captured later in the best line. Distinct
    from missed_attacking_check (there best is a QUIET/positional check; here the check is a prelude to
    the same capture the player rushed)."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or not b.is_capture(pm):        # player made a capture
        return []
    if not m.best_line_san or len(m.best_line_san) < 3:
        return []
    tgt = pm.to_square
    bb = chess.Board(b.fen())
    first = None
    for j, san in enumerate(m.best_line_san[:6]):
        try:
            mv = bb.parse_san(san)
        except Exception:
            return []
        if j == 0:
            first = mv
            if not bb.gives_check(mv):             # best must INSERT a check first
                return []
            if mv.from_square == pm.from_square and mv.to_square == pm.to_square:
                return []                          # best IS the played move — no zwischenzug
        # the played capture's target gets taken later in the best line (same square, still a capture)
        if j >= 1 and mv.to_square == tgt and bb.is_capture(mv):
            piece = b.piece_at(first.from_square)
            pname = PIECE_NAME[piece.piece_type].lower() if piece else "piece"
            return [("Missed Zwischenzug", "missed",
                     f"insert the in-between {pname} check {m.best_san} first, THEN recapture "
                     f"(you played the immediate {m.played_san})")]
        bb.push(mv)
    return []


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
    b0 = m.board_before
    pm = _played_move(m)
    # SACRIFICE guard: if the PLAYED move is itself a material-SHEDDING capture (SEE < 0), the player
    # CHOSE to give material — that's a sacrifice / bad grab (owned by unsound_sacrifice / greedy_capture),
    # NOT a hang. "Hung" means you left a piece to be taken by a quiet move, not that you initiated a
    # losing exchange. Without this, hung_material out-votes the sac/greed tags and mislabels the concept.
    # (Sam, 2026-07-13: measured 85% of the sac/greed SAE features' hung fires were SEE<0 played captures;
    # only 4% of genuine-hang features were. SEE cleanly separates "I sacrificed" from "I hung".)
    if pm is not None and b0.is_capture(pm) and U.static_exchange_eval(b0, pm) < 0:
        return []
    # Reference point MUST be board_BEFORE the played move, so that if the played move is itself a
    # capture, the player's own gain is netted in. Measuring from board_after (post-capture) made an
    # EQUAL trade (e.g. Bxc6 bxc6, 3-for-3) read as a 3-pt hang — it counted the recapture loss but
    # not the capture gain. Now an equal trade nets 0 and does NOT fire. (Caught by Sam.)
    start_diff = _material_diff(b0, m.mover)
    bb = chess.Board(b0.fen())
    try:
        bb.push(chess.Move.from_uci(m.played_uci))
    except Exception:
        return []
    diffs = [_material_diff(bb, m.mover)]   # diffs[0] = right after the played move (opponent to move)
    first_victim = None                      # the piece the opponent CAPTURES on its first reply
    # Track the WORST point in the line (peak loss) + the biggest piece the opponent captured up to it.
    # A queen hung mid-line for partial compensation (net back to -1) reads as "-1 pawn" if you only look
    # at the end — but the queen genuinely left the board. Peak-loss catches that; end_loss>=1 keeps a
    # full-recovery slosh (equal trade, net 0) from firing. (Sam, 2026-07-12: move 18 Qd3, Ne2+ Rxe2
    # Rxd3 hangs the queen, settles -1; hung_material was silent.)
    peak_victim = None
    opp_promo_gain = 0                       # material the OPPONENT gains by PROMOTING in the line
    opp = not m.mover
    for i, san in enumerate(m.refutation_san):
        try:
            mv = bb.parse_san(san)
        except Exception:
            break
        cap_victim = None
        if bb.is_capture(mv):
            vic = bb.piece_at(mv.to_square)         # None = en passant (pawn)
            cap_victim = vic.piece_type if vic is not None else chess.PAWN
            if i == 0:
                first_victim = cap_victim
        # A promotion by the OPPONENT inflates material_diff by (promoted piece − pawn) without you
        # having HUNG anything — you lost a PAWN RACE. Track that gain so we can tell "hung a piece"
        # (a capture of your material) apart from "let a passer queen" (a different, endgame lesson).
        # (Sam, 2026-07-13: promotion features were mislabeled Hung Material / even Hung Queen.)
        if mv.promotion and bb.turn == opp:
            opp_promo_gain += VAL.get(mv.promotion, 9) - VAL[chess.PAWN]
        bb.push(mv)
        d = _material_diff(bb, m.mover)
        diffs.append(d)
        # A new worst point AND this ply was the opponent capturing one of our pieces → that piece is
        # the peak victim (the thing we hung). Only count the opponent's captures (our own recaptures
        # move the diff back up, not down).
        if cap_victim is not None and d == min(diffs):
            peak_victim = cap_victim
    end_diff = diffs[-1]
    net_lost = start_diff - end_diff        # end-of-line net — equal trades net 0
    peak_lost = start_diff - min(diffs)     # worst point in the line — a mid-line hang shows here
    # PROMOTION-RACE guard: if the opponent's promotion in the line accounts for most of the material
    # swing, this is NOT a hung piece — it's a lost pawn race / botched passed-pawn defense (an ENDGAME
    # technique lesson, tagged by the pawn-endgame fragments, NOT "Hung Material"/"Hung Queen"). Subtract
    # the promotion gain; only fire on what's left, i.e. actual pieces the opponent CAPTURED. (Sam,
    # 2026-07-13: passed-pawn features were being swallowed by Hung Material because a promoted queen
    # reads as +8 material.)
    net_lost -= opp_promo_gain
    peak_lost -= opp_promo_gain
    # Fire when material is lost at the PEAK (>=2) AND is still down at the end (>=1). The end>=1 guard
    # is what stops a full-recovery slosh (peak dips then nets back to 0) from over-claiming.
    if peak_lost < 2 or net_lost < 1:
        return []
    # how much is gone after the opponent's FIRST reply (immediate vs delayed).
    immediate_lost = start_diff - diffs[1] if len(diffs) > 1 else (start_diff - diffs[0])
    # Name the hung piece: prefer the peak victim (the biggest piece the opponent won), else the first-
    # reply victim, when that piece's value ~ the peak loss. Else generic "Hung Material".
    named = peak_victim if peak_victim is not None else first_victim
    if named is not None and VAL.get(named, 0) >= peak_lost - 1:
        pname = PIECE_NAME[named]
        return [(f"Hung {pname}", "hung",
                 f"the refutation wins your {pname.lower()} ({peak_lost} pts at worst, {net_lost} net over line)")]
    if immediate_lost >= 2 or peak_lost >= 2:
        return [("Hung Material", "hung",
                 f"the refutation wins material ({peak_lost} pts at worst, {net_lost} net over line)")]
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
    """The PLAYED move ADVANCES a shelter pawn away from your own CASTLED king, weakening the shelter.

    Old version fired on ANY pawn within 1 file of the king anywhere — 9.8% of the corpus, 34% with the
    king in the CENTRE (no shelter to expose) and 6% captures. Tightened to the real concept (Sam, #50):
      1. King is CASTLED (g/h or a/b/c file, back two ranks, with a shelter pawn) — reuse
         _king_is_castled. A central/uncastled king has no shelter to break.
      2. The move is a NON-CAPTURE pawn PUSH (a capture is a material decision, not a structural weaken).
      3. The pawn starts in the king's shelter (within 1 file, within 2 ranks of the king) and the push
         ADVANCES it toward the enemy (off the back where it was guarding) — i.e. it opens the shelter,
         not a pawn already far up the board nudging further.
    This is a played-direction STATE tag (the resulting weakness), not a "find the move" puzzle."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or b.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    if b.is_capture(pm):                                   # (2) captures aren't shelter pushes
        return []
    if not _king_is_castled(b, m.mover):                   # (1) no castled king -> no shelter to expose
        return []
    ks = b.king(m.mover)
    kf, kr = chess.square_file(ks), chess.square_rank(ks)
    pf = chess.square_file(pm.from_square)
    pr_from, pr_to = chess.square_rank(pm.from_square), chess.square_rank(pm.to_square)
    if abs(pf - kf) > 1 or abs(pr_from - kr) > 2:          # (3a) pawn must be in the king's shelter zone
        return []
    advancing = (pr_to > pr_from) if m.mover == chess.WHITE else (pr_to < pr_from)
    if not advancing:                                       # (3b) must push forward, off its guarding square
        return []
    return [("Pawn Move Exposed King", "played", f"the pawn push {m.played_san} weakens your castled king's shelter")]


def recapture_exposes_king(m):
    """The PLAYED move is a pawn CAPTURE by a shelter pawn in front of your own CASTLED king, opening a
    file/diagonal onto your king — and the best move was NOT that capture (a different recapture or a
    quiet move kept the shelter intact). The classic 'recaptured toward my own king' error, e.g. hxg4 /
    hxg5 with the king on g1/g8 opening the h/g-file.

    Data-derived (SAE jr2048 f24/f896, Opus 'Opening Lines to Own King' / 'Damaging Own King Shelter').
    Distinct from exposed_king_pawn (a PUSH that weakens the shelter) — here it's a CAPTURE that opens a
    line. A played-direction structural-state tag (the resulting weakness), like exposed_king_pawn."""
    b = m.board_before
    pm = _played_move(m)
    if pm is None or b.piece_type_at(pm.from_square) != chess.PAWN:
        return []
    if not b.is_capture(pm):                           # must be a CAPTURE (opens a line), not a push
        return []
    if not _king_is_castled(b, m.mover):               # need a real castled king to expose
        return []
    ks = b.king(m.mover)
    kf, kr = chess.square_file(ks), chess.square_rank(ks)
    # the capturing pawn must be a SHELTER pawn (within 1 file, within 2 ranks of the king)
    if abs(chess.square_file(pm.from_square) - kf) > 1 or abs(chess.square_rank(pm.from_square) - kr) > 2:
        return []
    bm = _best_move(m)
    if bm is not None and bm.from_square == pm.from_square and bm.to_square == pm.to_square:
        return []                                      # best IS the recapture — then it wasn't the error
    return [("Recapture Exposed King", "played",
             f"the pawn recapture {m.played_san} opens a line onto your own king; best was {m.best_san}")]


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


def _opposition_kind(king_sq, enemy_king_sq):
    """Return 'direct' | 'distant' | 'diagonal' if the two kings stand in OPPOSITION, else None.

    Opposition = kings on the same file, rank, OR diagonal with an ODD number of squares between them
    (i.e. an EVEN square-distance of 2/4/6). Direct = adjacent-line distance 2; distant = 4 or 6 on a
    file/rank; diagonal = equal file/rank offset of 2/4/6. The side NOT to move holds it — so when the
    engine's king move REACHES this geometry (and the played move didn't), the mover took the
    opposition. (Generalizes the old direct-only check: SAE endgame features surfaced diagonal cases
    the old code missed. #50.)"""
    fd = abs(chess.square_file(king_sq) - chess.square_file(enemy_king_sq))
    rd = abs(chess.square_rank(king_sq) - chess.square_rank(enemy_king_sq))
    if (fd == 0 or rd == 0) and (fd + rd) in (2, 4, 6):
        return "direct" if fd + rd == 2 else "distant"
    if fd == rd and fd in (2, 4, 6):
        return "diagonal"
    return None


def lost_opposition(m):
    """King-and-pawn endgame: best move is a king move that takes the OPPOSITION (direct, distant, or
    diagonal — see _opposition_kind), and the played move didn't. K+P-only: opposition is a decisive,
    teachable concept precisely because zugzwang rules the pawn ending; with pieces on the board a
    2-square king spacing is coincidence, not a lesson (so we keep the pawn-only gate — deliberately
    NOT loosened; that path is naked-rate, #50 discussion)."""
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
    kind = _opposition_kind(bm.to_square, ek)
    if kind is None:
        return []
    # the played move must NOT already hold the same opposition (else it's not a "lost" opposition —
    # both moves achieve it and the distinction is elsewhere).
    if pm is not None and b.piece_type_at(pm.from_square) == chess.KING and _opposition_kind(pm.to_square, ek):
        return []
    label = "takes the opposition" if kind == "direct" else f"takes the {kind} opposition"
    return [("Lost the Opposition", "missed", f"best {m.best_san} {label}")]


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


# ---------- book-derived endgame technique (Dvoretsky / de la Villa) ----------

def missed_protected_passer(m):
    """Pawn endgame fundamental (Dvoretsky §"The Protected Passed Pawn"): best move is a pawn move
    that creates a PROTECTED passed pawn — a passer defended by another friendly pawn — which the
    played move doesn't. A protected passer is often decisive: the enemy king is tied to it forever."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []

    def protected_passers(board, color):
        out = 0
        for sq, p in board.piece_map().items():
            if p.piece_type != chess.PAWN or p.color != color:
                continue
            if not U.is_passed_pawn(board, sq, color):
                continue
            # Protected = a friendly pawn defends it (attacks its square)
            f, r = chess.square_file(sq), chess.square_rank(sq)
            back = -1 if color == chess.WHITE else 1
            for df in (-1, 1):
                df_f = f + df; df_r = r + back
                if 0 <= df_f <= 7 and 0 <= df_r <= 7:
                    d = board.piece_at(chess.square(df_f, df_r))
                    if d and d.piece_type == chess.PAWN and d.color == color:
                        out += 1; break
        return out

    after_best = b.copy(); after_best.push(bm)
    after_played = b.copy(); after_played.push(pm)
    before_n = protected_passers(b, m.mover)
    best_n = protected_passers(after_best, m.mover)
    played_n = protected_passers(after_played, m.mover)
    if best_n > before_n and played_n < best_n:
        return [("Missed Protected Passer", "missed",
                 f"best {m.best_san} creates a protected passed pawn")]
    return []


def missed_square_rule(m):
    """King-vs-passed-pawn fundamental (Dvoretsky §"The Rule of the Square"): in a position where the
    DEFENDING king must catch an enemy passer, the best move is the king move that steps into the
    pawn's "square" (can catch it) and the played move steps outside it (lets it queen). Easy to label:
    only fires when the side to move has no pawns/pieces racing — a pure king-chases-pawn race."""
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
    # The mover is the DEFENDER chasing an enemy passed pawn: enemy has a passer, mover has no pawn
    # on the passer's path. Find the most advanced enemy passer.
    enemy = not m.mover
    fwd = 1 if enemy == chess.WHITE else -1
    promo_rank = 7 if enemy == chess.WHITE else 0
    target = None
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.PAWN and p.color == enemy and U.is_passed_pawn(b, sq, enemy):
            if target is None or (enemy == chess.WHITE and chess.square_rank(sq) > chess.square_rank(target)) \
                              or (enemy == chess.BLACK and chess.square_rank(sq) < chess.square_rank(target)):
                target = sq
    if target is None:
        return []
    pf, pr = chess.square_file(target), chess.square_rank(target)
    promo_sq = chess.square(pf, promo_rank)
    dist_to_promo = abs(promo_rank - pr)  # pawn's steps to promote

    def king_catches(king_sq):
        # Chebyshev distance from king to the promotion square <= pawn's steps (+1 if enemy not to move)
        kd = chess.square_distance(king_sq, promo_sq)
        # mover is to move, so mover's king effectively gets a tempo: catches if kd <= dist_to_promo
        return kd <= dist_to_promo

    if king_catches(bm.to_square) and not king_catches(pm.to_square):
        return [("Missed Square Rule", "missed",
                 f"best {m.best_san} steps into the pawn's square to catch the passer")]
    return []


def missed_breakthrough(m):
    """Pawn-endgame breakthrough (Dvoretsky §"Breakthrough"): best move is a pawn ADVANCE or pawn
    SAC into the enemy pawn mass that yields a passed pawn the opponent can't stop, which the played
    move doesn't. Heuristic: best is a pawn push to a contact/sac square; after the best line's first
    move + a plausible enemy capture, the mover gets a passed pawn that wasn't passed before."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.PAWN:
        return []
    # Mostly a pawn-ish endgame (few pieces) so "breakthrough" is the point, not tactics
    nonpawn = sum(1 for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    if nonpawn > 2:
        return []
    to_f = chess.square_file(bm.to_square); to_r = chess.square_rank(bm.to_square)
    # Best push must create contact with enemy pawns (adjacent enemy pawn) — the sac trigger
    contact = False
    for af in (to_f - 1, to_f, to_f + 1):
        if not (0 <= af <= 7):
            continue
        for ar in (to_r - 1, to_r, to_r + 1):
            if not (0 <= ar <= 7):
                continue
            pc = b.piece_at(chess.square(af, ar))
            if pc and pc.piece_type == chess.PAWN and pc.color != m.mover:
                contact = True
    if not contact:
        return []
    # After best, mover has (or after a forced enemy capture, will have) a passed pawn that the
    # played move's resulting position lacks. Compare passed-pawn count.
    def passers(board):
        return sum(1 for sq, p in board.piece_map().items()
                   if p.piece_type == chess.PAWN and p.color == m.mover and U.is_passed_pawn(board, sq, m.mover))
    after_best = b.copy(); after_best.push(bm)
    after_played = b.copy(); after_played.push(pm)
    if passers(after_best) > passers(b) and passers(after_played) <= passers(b):
        return [("Missed Breakthrough", "missed",
                 f"best {m.best_san} is a pawn breakthrough creating a passer")]
    return []




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
    """Endgame: best move is an EVEN trade (a capture where you give ~what you get) that simplifies to a
    won position, but the played move is not. The player missed that trading down was winning.

    Must be an EVEN exchange, NOT a free/winning grab: SEE gate (#52). trade_to_simplify was firing on
    'best captures a HANGING piece' — that's winning material (Missed Free X, owned by capture_or_exchange),
    not a simplifying trade. 95% of the confidently-wrong fires had the best capture at SEE>=2 (undefended
    or net-winning target). A trade means SEE~=0: material comes off both sides evenly and the point is the
    RESULTING simpler position, not the material. (Sam, board-grounded SAE audit.)"""
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
    if U.static_exchange_eval(b, bm) >= 2:
        return []  # best capture WINS material -> Missed Free X, not a trade (capture_or_exchange owns it)
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

# A battery is only a threat if its front piece attacks a real enemy target — a minor piece or better,
# or the king. A pawn (or empty) aim doesn't count (that's just two aligned pieces). Shared by both the
# missed and allowed battery detectors so their target definition can't drift.
_BATTERY_TARGETS = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

def _battery_hits_target(board, front_sq, battery_owner):
    """True if the piece on front_sq attacks an enemy (not battery_owner) minor+/king."""
    enemy = not battery_owner
    for t in board.attacks(front_sq):
        p = board.piece_at(t)
        if p and p.color == enemy and p.piece_type in _BATTERY_TARGETS:
            return True
    return False


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
        # Battery exists — does it attack a real enemy target (minor+ or king)?
        if _battery_hits_target(after, to_sq, m.mover):
            return [("Missed Battery", "missed", f"best {m.best_san} creates a battery attacking a piece/king")]
    return []


def missed_overloading(m):
    """Best move attacks a piece that is the SOLE defender of another valuable piece (overloading the
    defender), and the played move doesn't exploit it. The defender can't protect both, so the mover
    wins material.

    TIGHTENED (#57 masker audit): the old geometry-only version fired on ANY best move that happened to
    attack a piece defending something — 9.96% of the corpus, an impossible rate for a real tactic, and
    it MASKED 11 sharper concepts (Hung Material / Missed Attacking Check / fork). A real overload must
    (a) actually WIN material — capturing the overloaded defender must be net-favorable (SEE>=0 on that
    capture), so taking it wins the defended piece; and (b) the defended piece must be worth >= a minor
    (else there's nothing to win). Geometry alone is not an overload."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    # (a) the overload must actually WIN material — verify the best LINE nets >= 2 pawns for the mover.
    # Geometry ("best move attacks a piece that defends something") alone fired on 9.96% of the corpus
    # and masked hung-piece/check features. A real overload converts to material; if the engine's line
    # doesn't win >=2, it's not the overload being exploited. (#57 masker audit.)
    if not m.best_line_san:
        return []
    bb = chess.Board(b.fen()); start = _material_diff(bb, m.mover)
    try:
        for san in m.best_line_san[:6]:
            bb.push(bb.parse_san(san))
    except Exception:
        return []
    if _material_diff(bb, m.mover) - start < 2:
        return []
    opp = not m.mover
    after = b.copy(); after.push(bm)
    target_sq = bm.to_square
    for victim_sq in after.attacks(target_sq):
        victim = after.piece_at(victim_sq)
        if not victim or victim.color != opp or victim.piece_type == chess.PAWN:
            continue
        for defended_sq in after.attacks(victim_sq):
            defended = after.piece_at(defended_sq)
            if not defended or defended.color != opp or defended_sq == victim_sq:
                continue
            if defended.piece_type in (chess.PAWN, chess.KING):
                continue
            if VAL.get(defended.piece_type, 0) < 3:          # (b) defended piece must be >= a minor
                continue
            defenders = after.attackers(opp, defended_sq)
            if len(defenders) == 1 and victim_sq in defenders:   # victim is the SOLE defender
                return [("Missed Overloading", "missed",
                         f"best {m.best_san} overloads the defender of a {PIECE_NAME[defended.piece_type].lower()}")]
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


# ---------- pin exploitation / unpinning (book TOP-TIER, w5zuk548s) ----------
def _ray_pin_on(board, color):
    """Find enemy pieces of `not color` currently pinned (absolute OR relative) by a `color` ray
    piece. Returns list of (pinned_sq, pinned_piece, shield_value). A pin = a `color` ray piece sees
    an enemy piece with a MORE-valuable enemy piece (or king) directly behind it on the same ray, no
    blockers between. Mirrors motifs._pin_target but scans the static board (no move needed)."""
    enemy = not color
    found = []
    DIRS = {chess.ROOK: [(1,0),(-1,0),(0,1),(0,-1)],
            chess.BISHOP: [(1,1),(1,-1),(-1,1),(-1,-1)],
            chess.QUEEN: [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            continue
        fr, ff = chess.square_rank(sq), chess.square_file(sq)
        for dr, df in DIRS[p.piece_type]:
            first = None
            r, f = fr + dr, ff + df
            while 0 <= r <= 7 and 0 <= f <= 7:
                s = chess.square(f, r); q = board.piece_at(s)
                if q is not None:
                    if first is None:
                        if q.color == color or q.piece_type == chess.PAWN:
                            break
                        first = (s, q)
                    else:
                        if (q.color == enemy and
                                U.king_values[q.piece_type] > U.king_values[first[1].piece_type]):
                            found.append((first[0], first[1], U.king_values[q.piece_type]))
                        break
                r += dr; f += df
    return found


def missed_pin_exploitation(m):
    """An enemy piece is pinned and currently HELD (defenders >= attackers, so you can't just take it).
    The best move PILES ON — lands a new attacker (ideally a pawn, which the pinned piece can't trade
    off) on the pinned piece so it falls next move — and the played move doesn't. Classic 'add a second
    attacker to a pinned piece' from every tactics book."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if b.is_capture(bm):
        return []  # a capture is "Missed Pin"/material, not the quiet pile-on prep
    pins = _ray_pin_on(b, m.mover)
    if not pins:
        return []
    opp = not m.mover
    after = b.copy(); after.push(bm)
    for psq, ppiece, _ in pins:
        # held now: at least as many defenders as attackers (can't simply win it today)
        atk = len(b.attackers(m.mover, psq)); dfd = len(b.attackers(opp, psq))
        if atk > dfd:
            continue
        # the best move must ADD a new attacker onto the pinned square
        if psq not in after.attacks(bm.to_square):
            continue
        if b.is_attacked_by(m.mover, psq) and bm.to_square in b.attackers(m.mover, psq):
            continue  # the moved piece already attacked it (didn't add anything new)
        # pin must still stand after the prep (the shield didn't move/get taken)
        if not _ray_pin_on(after, m.mover):
            continue
        adder = b.piece_type_at(bm.from_square)
        extra = " with a pawn" if adder == chess.PAWN else ""
        return [("Missed Pin Exploitation", "missed",
                 f"best {m.best_san} piles onto the pinned {chess.piece_name(ppiece.piece_type)}{extra}")]
    return []


def missed_unpinning_resource(m):
    """One of the MOVER's pieces is pinned (absolute or relative) and the best move BREAKS the pin —
    captures the pinner, steps the king/shield off the ray, interposes, or moves the rear piece — while
    the played move leaves the pin standing. The 'get out of the pin' resource sub-master players sit in."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    pins_before = _ray_pin_on(b, not m.mover)   # enemy ray pieces pinning OUR pieces
    if not pins_before:
        return []
    pinned_sqs = {psq for psq, _, _ in pins_before}
    after_best = b.copy(); after_best.push(bm)
    after_played = b.copy(); after_played.push(pm)
    best_pins = {psq for psq, _, _ in _ray_pin_on(after_best, not m.mover)}
    played_pins = {psq for psq, _, _ in _ray_pin_on(after_played, not m.mover)}
    # best frees at least one piece that played leaves pinned
    freed_by_best = pinned_sqs - best_pins
    if not freed_by_best:
        return []
    still_pinned_after_played = pinned_sqs & played_pins
    if not still_pinned_after_played:
        return []  # the played move also happened to break it — no missed resource
    return [("Missed Unpinning Resource", "missed",
             f"best {m.best_san} breaks the pin; played {m.played_san} sits in it")]


def missed_interposition(m):
    """In check from a single sliding piece (so a block is geometrically possible), the best move
    INTERPOSES a piece on a square between the checker and the king — and the played move runs the
    king (or otherwise doesn't block) when blocking was the right call. The 'block beats flee'
    defensive resource (Active Defense)."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if not b.is_check():
        return []
    checkers = list(b.checkers())
    if len(checkers) != 1:
        return []  # double check can only be answered by a king move — no interposition possible
    csq = checkers[0]
    if b.piece_type_at(csq) not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
        return []  # knight/pawn check can't be blocked
    ksq = b.king(m.mover)
    if ksq is None:
        return []
    between = chess.SquareSet.between(csq, ksq)
    if not between:
        return []  # checker is adjacent — nothing to interpose
    # best move blocks: a non-king piece lands on a between-square (and it's not a capture of the checker)
    if bm.from_square == ksq or bm.to_square not in between:
        return []
    # played move does NOT block (king move, or lands elsewhere)
    if pm.to_square in between and pm.from_square != ksq:
        return []  # played also interposed — no missed resource
    return [("Missed Interposition", "missed",
             f"in check; best {m.best_san} blocks but played {m.played_san} doesn't")]


def _king_ring(board, color):
    """The enemy king square + its 8 neighbours (the squares an attack must penetrate)."""
    ksq = board.king(color)
    if ksq is None:
        return set()
    return {ksq} | {s for s in chess.SQUARES if chess.square_distance(s, ksq) == 1}


def _king_is_castled(board, color):
    """The enemy king sits in a castled position (g/h or a/c file, on its back two ranks) with at
    least one shelter pawn nearby — i.e. there is a real king to attack, not an opening/endgame king."""
    ksq = board.king(color)
    if ksq is None:
        return False
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    home = 0 if color == chess.WHITE else 7
    if abs(kr - home) > 1:
        return False            # king has wandered up the board — not a castled-king attack
    if kf not in (0, 1, 2, 5, 6, 7):
        return False            # central king (incl. uncastled e-file) — not the theme
    # at least one friendly pawn on the three files around the king (a shelter to attack through)
    for f in (kf-1, kf, kf+1):
        if 0 <= f <= 7:
            for r in range(8):
                p = board.piece_at(chess.square(f, r))
                if p and p.color == color and p.piece_type == chess.PAWN:
                    return True
    return False


def missed_remove_the_guard(m):
    """Best move captures (an EVEN trade — defended target) a MINOR piece that DEFENDS the enemy
    CASTLED king's ring, stripping a defender off the king; the played move doesn't. The classic
    'remove the defender of the castled king' (e.g. Bxf6 taking the knight that guards h7/g8). Tightened
    to avoid over-fire: minor-piece victim only (not queen/rook/random material), king must be castled
    (not opening/endgame), and the capture must not be a check (those are forcing tactics, named
    elsewhere)."""
    if _is_endgame(m):
        return []
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    if not b.is_capture(bm):
        return []
    opp = not m.mover
    victim_sq = bm.to_square
    victim = b.piece_at(victim_sq)
    # minor-piece guard only — the textbook defenders (Nf6/Be7/Bg7), not queen/rook/pawn captures.
    if victim is None or victim.color != opp or victim.piece_type not in (chess.KNIGHT, chess.BISHOP):
        return []
    # even trade, not a sac/freebie: the victim is DEFENDED.
    if not b.is_attacked_by(opp, victim_sq):
        return []
    # the enemy king must be a real castled king (there's something to attack).
    if not _king_is_castled(b, opp):
        return []
    # checks are forcing tactics (named by mate/fork/etc.) — not the quiet 'remove the guard' theme.
    after = b.copy(); after.push(bm)
    if after.is_check():
        return []
    ring = _king_ring(b, opp)
    guards = b.attacks(victim_sq) & chess.SquareSet(ring)
    if not guards:
        return []
    # after the capture, those ring squares lose this defender.
    still_guarded = after.attacks(victim_sq) if (after.piece_at(victim_sq) and after.piece_at(victim_sq).color == opp) else chess.SquareSet(0)
    if not (chess.SquareSet(guards) - still_guarded):
        return []
    if pm.to_square == victim_sq:
        return []
    return [("Missed Remove the Guard", "missed",
             f"best {m.best_san} removes a defender of the castled king")]


def allowed_battery(m):
    """Played move allows the opponent to create a battery (Q+R on file, Q+B on diagonal) that
    ATTACKS a real target, IN THE ACTUAL REFUTATION LINE, which the best move would have prevented.

    2026-07-11 (round 2): previously scanned ALL opponent legal moves, so in almost any middlegame
    *some* Q/R/B move aligns on *something* and it fired spuriously (e.g. Rb8 → "Allowed Battery" with
    no real battery in the refutation). Now it only considers the opponent's OWN moves along the stored
    refutation line — the README's "Allowed X = [played]+refutation" definition — so the battery has to
    actually be what punishes the move. Falls back to [] when there's no refutation line."""
    b = m.board_before
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    if not m.refutation_san:
        return []
    opp = not m.mover
    after_played = b.copy(); after_played.push(pm)
    after_best = b.copy(); after_best.push(bm)
    # Walk the refutation line; only test the OPPONENT's moves in it (a battery is the opponent's threat).
    walk = after_played.copy()
    for san in m.refutation_san:
        try:
            mv = walk.parse_san(san)
        except Exception:
            break
        is_opp_move = (walk.turn == opp)
        piece_type = walk.piece_type_at(mv.from_square)
        if not (is_opp_move and piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP)):
            walk.push(mv)
            continue
        # After the opponent plays this refutation move, do they have aligned pieces on a real target?
        test = walk.copy(); test.push(mv)
        walk.push(mv)  # advance the walk regardless, so later plies are reachable
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
            aligned = False
            # Same file / same rank (Q/R only)
            if (sq_f == to_f or sq_r == to_r) and p.piece_type in (chess.QUEEN, chess.ROOK) and piece_type in (chess.QUEEN, chess.ROOK):
                if not any(test.piece_at(s) for s in chess.SquareSet.between(sq, to_sq)):
                    aligned = True
            # Same diagonal (Q/B only)
            elif abs(sq_f - to_f) == abs(sq_r - to_r) and sq != to_sq and p.piece_type in (chess.QUEEN, chess.BISHOP) and piece_type in (chess.QUEEN, chess.BISHOP):
                if not any(test.piece_at(s) for s in chess.SquareSet.between(sq, to_sq)):
                    aligned = True
            if not aligned:
                continue
            # Was this battery possible before our move? If yes, not our fault.
            if mv in after_best.legal_moves:
                continue
            # Battery must attack a real target (minor+ or king) — a pawn/empty aim is not a threat.
            if not _battery_hits_target(test, to_sq, m.mover):
                continue
            return [("Allowed Battery", "allowed", f"{m.played_san} allows opponent to build a battery")]
    return []


def allowed_pawn_capture(m):
    """The played move is QUIET (not a capture) but causes a pawn loss — either immediately (opponent's
    first refutation reply grabs a pawn) OR over the refutation line (net material loss = exactly 1
    pawn). Two conditions, same coaching point: "your move loses a pawn that the best move held."

    Conditions (either triggers the tag):
      A. IMMEDIATE: opponent's first refutation reply is a pawn capture unavailable after best (the
         original detector — e.g. Rb8 lets Bxd5). Fires even if the line later recaptures (allowing the
         grab IS the concession; material may net back to 0 over the full line).
      B. DELAYED: net material loss over the FULL refutation = exactly 1 pawn (for a heavier net loss,
         hung_material fires instead). And the best line does NOT lose the same material. Catches the case
         where the pawn falls after a maneuver (e.g. Nd2 leads to Rxb5 at ply 4, netting -1).

    Guards (apply to both paths):
      - Played must be QUIET (not a capture). Equal trades are hung_material/greedy's job.
      - Opponent's first reply capturing a PIECE (not pawn) → this is a Hung X story, not ours.
      - If hung_material will also fire (net ≥ 2) on this move, we stay silent (it's the bigger story).
    """
    pm = _played_move(m); bm = _best_move(m)
    if pm is None or bm is None or pm == bm:
        return []
    b = m.board_before
    if b.is_capture(pm):            # only QUIET played moves (equal trades handled elsewhere)
        return []
    if not m.refutation_san:
        return []
    after_played = b.copy(); after_played.push(pm)
    after_best = b.copy(); after_best.push(bm)

    # --- PATH A: immediate pawn capture on the first refutation reply ---
    try:
        first = after_played.parse_san(m.refutation_san[0])
        if after_played.is_capture(first):
            victim = after_played.piece_at(first.to_square)
            victim_type = victim.piece_type if victim is not None else chess.PAWN
            if victim_type == chess.PAWN and first not in after_best.legal_moves:
                return [("Allowed Pawn Capture", "allowed",
                         f"{m.played_san} lets the opponent grab a pawn ({m.refutation_san[0]}); best {m.best_san} prevents it")]
    except Exception:
        pass

    # --- PATH B: delayed net-1-pawn loss over the full refutation line ---
    mover = m.mover
    start_diff = _material_diff(b, mover)   # before the played move
    bb = after_played.copy()
    for san in m.refutation_san:
        try:
            bb.push(bb.parse_san(san))
        except Exception:
            break
    end_diff = _material_diff(bb, mover)
    net_lost = start_diff - end_diff
    if net_lost != 1:
        return []   # 0 = no loss, ≥2 = hung_material's job
    # Does the BEST line also lose a pawn? If so, it's not our move's fault.
    bb_best = after_best.copy()
    if m.best_line_san:
        for san in m.best_line_san[1:]:  # skip best-move itself (already pushed into after_best)
            try:
                bb_best.push(bb_best.parse_san(san))
            except Exception:
                break
    best_end_diff = _material_diff(bb_best, mover)
    best_net_lost = start_diff - best_end_diff
    if best_net_lost >= 1:
        return []   # best line also loses material — the pawn was doomed regardless
    # Name the capture that wins the pawn (first refutation capture of a pawn in the line)
    bb2 = after_played.copy()
    grab_san = None
    for san in m.refutation_san:
        try:
            mv = bb2.parse_san(san)
            if bb2.is_capture(mv):
                vic = bb2.piece_at(mv.to_square)
                if vic is None or vic.piece_type == chess.PAWN:
                    grab_san = san; break
            bb2.push(mv)
        except Exception:
            break
    evidence = f"{m.played_san} loses a pawn by force"
    if grab_san:
        evidence += f" ({grab_san})"
    evidence += f"; best {m.best_san} holds"
    return [("Allowed Pawn Capture", "allowed", evidence)]


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


# ---------- minor-piece / queen endgame technique (drill detail for those material clusters) ----------

def _is_minor_endgame(board):
    """K + pawns + only minor pieces (bishops/knights), at least one minor, no rooks/queens."""
    has_minor = False
    for p in board.piece_map().values():
        if p.piece_type in (chess.ROOK, chess.QUEEN):
            return False
        if p.piece_type in (chess.BISHOP, chess.KNIGHT):
            has_minor = True
    return has_minor


def missed_bishop_activity(m):
    """Minor-piece endgame: best move significantly improves a bishop's scope (mobility +≥4 squares),
    played doesn't. A passive bishop is a classic minor-endgame failing. Validated on the eligible
    denominator (positions where a >=4-gain bishop move exists): miss-rate falls 13.8%→3.8% from
    beginner to master, so it's a real skill signal (the earlier raw-count measure was misleading)."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not _is_minor_endgame(b):
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.BISHOP:
        return []
    before = len(b.attacks(bm.from_square))
    after = b.copy(); after.push(bm)
    after_n = len(after.attacks(bm.to_square))
    if after_n - before < 4:
        return []
    return [("Missed Bishop Activity", "missed",
             f"best {m.best_san} activates the bishop ({before}→{after_n} squares)")]


def _activates_piece(m, ptype, gain=4):
    """Shared: best move is a `ptype` move gaining >=`gain` mobility, played didn't. Returns (before,
    after) tuple if it fires, else None. Caller adds the material gate + label."""
    b = m.board_before
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return None
    if b.piece_type_at(bm.from_square) != ptype:
        return None
    before = len(b.attacks(bm.from_square))
    after = b.copy(); after.push(bm)
    after_n = len(after.attacks(bm.to_square))
    if after_n - before < gain:
        return None
    return (before, after_n)


def missed_knight_activity(m):
    """Minor-piece endgame: best move activates a passive knight (mobility +≥4), played doesn't.
    Pairs with Bishop Activity. Validated: eligible miss-rate 11.5%→4.3% beginner→master."""
    if not _is_endgame(m) or not _is_minor_endgame(m.board_before):
        return []
    r = _activates_piece(m, chess.KNIGHT)
    if r is None:
        return []
    return [("Missed Knight Activity", "missed",
             f"best {m.best_san} activates the knight ({r[0]}→{r[1]} squares)")]


def missed_minor_activity(m):
    """Rook + minor endgame: best move activates the MINOR piece (bishop or knight, +≥4 mobility),
    played didn't. Pairs with Rook Activity in the Rook+Minor cluster. Validated: 11.7%→2.4%."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    nr = nq = minors = 0
    for p in b.piece_map().values():
        if p.piece_type == chess.ROOK: nr += 1
        elif p.piece_type == chess.QUEEN: nq += 1
        elif p.piece_type in (chess.BISHOP, chess.KNIGHT): minors += 1
    if nq > 0 or nr == 0 or minors == 0:
        return []
    for pt, name in ((chess.BISHOP, "bishop"), (chess.KNIGHT, "knight")):
        r = _activates_piece(m, pt)
        if r is not None:
            return [("Missed Minor Activity", "missed",
                     f"best {m.best_san} activates the {name} ({r[0]}→{r[1]} squares)")]
    return []


def missed_queen_activity(m):
    """Any queen endgame (Q+P 'Queen' OR Q+pieces 'Heavy'): best move activates/centralizes the queen
    (+≥4 mobility), played put it on a worse square. Captures queen-endgame misplacement broadly (Sam:
    queen mistakes aren't only Q+P). Validated: eligible miss-rate 10.7%→5.7% beginner→master."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    if not any(p.piece_type == chess.QUEEN for p in b.piece_map().values()):
        return []
    r = _activates_piece(m, chess.QUEEN)
    if r is None:
        return []
    return [("Missed Queen Activity", "missed",
             f"best {m.best_san} activates the queen ({r[0]}→{r[1]} squares)")]


def missed_minor_rook_activity(m):
    """Rook + minor endgame: best move activates the rook (mobility +≥4), played doesn't. The same
    'active rook' principle as pure rook endgames, but in R+minor material (where the rook is still
    the workhorse). Distinct cluster (Rook + Minor Endgames)."""
    if not _is_endgame(m):
        return []
    b = m.board_before
    # R+minor material: at least one rook AND at least one minor, no queens
    nr = nb = nn = nq = 0
    for p in b.piece_map().values():
        t = p.piece_type
        if t == chess.ROOK: nr += 1
        elif t == chess.BISHOP: nb += 1
        elif t == chess.KNIGHT: nn += 1
        elif t == chess.QUEEN: nq += 1
    if nq > 0 or nr == 0 or (nb + nn) == 0:
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or bm == pm:
        return []
    if b.piece_type_at(bm.from_square) != chess.ROOK:
        return []
    before = len(b.attacks(bm.from_square))
    after = b.copy(); after.push(bm)
    after_n = len(after.attacks(bm.to_square))
    if after_n - before < 4:
        return []
    return [("Missed Rook Activity (R+Minor)", "missed",
             f"best {m.best_san} activates the rook ({before}→{after_n} squares)")]


def missed_perpetual(m):
    """Queen/heavy endgame defensive resource: the player was losing/worse and the BEST move starts a
    perpetual check (a repeating check that forces a draw), which the played move missed. Detected via
    the best line: first move is a check, and a check recurs later in the PV (a checking sequence).
    Fires only when the mover is worse (a draw is a SAVE, not a concession)."""
    b = m.board_before
    if m.eval_before is None:
        return []
    # Mover must be worse/losing for a perpetual to be a gain (draw saves the game)
    eb_mover = m.eval_before if m.mover == chess.WHITE else -m.eval_before
    if eb_mover > -150:
        return []
    bm = _best_move(m); pm = _played_move(m)
    if bm is None or pm is None or bm == pm:
        return []
    # Need queens on board (perpetual is overwhelmingly a queen resource)
    if not any(p.piece_type == chess.QUEEN for p in b.piece_map().values()):
        return []
    # Best move must give check, and the best line must contain a second check (repeating).
    after_best = b.copy()
    try:
        after_best.push(bm)
    except Exception:
        return []
    if not b.gives_check(bm) and not after_best.is_check():
        return []
    # Walk the best line (SAN) and count checks by the mover.
    checks = 1
    line_board = after_best
    if m.best_line_san:
        try:
            tmp = b.copy(); tmp.push(bm)
            for san in m.best_line_san[1:8]:
                mv = tmp.parse_san(san)
                tmp.push(mv)
                if tmp.is_check():
                    checks += 1
        except Exception:
            pass
    if checks < 2:
        return []
    return [("Missed Perpetual", "missed",
             f"best {m.best_san} starts a perpetual check to save the draw")]


# ---------- registry ----------
ALL_PREDICATES = [
    phase, game_state, conversion_outcome, blunder_severity, capture_or_exchange, greedy_capture, unsound_sacrifice, pointless_check,
    missed_attacking_check, missed_greek_gift, missed_zwischenzug, recapture_exposes_king, hung_material,
    king_in_center, lost_castling, exposed_king_pawn, pawn_structure,
    endgame_type, backward_pawn,
    missed_king_activity, lost_opposition, missed_passed_pawn, rook_behind_passer,
    rook_to_seventh, rook_cut_off_king, missed_active_rook, rook_endgame_blockade,
    missed_connected_passers, missed_protected_passer, missed_square_rule,
    missed_breakthrough,
    bad_simplification, trade_to_simplify, wrong_king_direction, outside_passer,
    rook_to_open_file_endgame, push_to_promote,
    pawn_grab_undeveloped, ignored_threat, premature_attack, missed_defensive_resource,
    missed_faster_mate,
    missed_bishop_activity, missed_knight_activity, missed_minor_activity, missed_queen_activity,
    missed_minor_rook_activity, missed_perpetual,
    missed_battery, missed_overloading, missed_desperado, missed_doubled_rooks,
    missed_pin_exploitation, missed_unpinning_resource, missed_interposition,
    missed_remove_the_guard,
    allowed_battery, allowed_pawn_capture, allowed_overloading, allowed_doubled_rooks,
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
