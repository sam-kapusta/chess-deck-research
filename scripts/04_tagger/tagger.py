#!/usr/bin/env python3
"""The orchestrator — runs all layers on a Mistake and rolls fine tags into high-level categories.

  from tagger import tag_mistake_full
  result = tag_mistake_full(mistake)   # {tags:[...], categories:[...], maia:{...}}

Layer 2 = OWNED motif detectors (motifs.py), driven in three directions:
  MISSED X  : the BEST line contained tactic X that the player failed to play   (pov = mover)
  ALLOWED X : the played move let the OPPONENT execute tactic X in refutation   (pov = opponent)
  FAILED X  : the played move WAS tactic X but it backfired (single-move only)  (pov = mover)
Layer 1 = position predicates (predicates.py). Layer 3 = Maia rarity (numeric, not tags).

No cook_adapter, no vendor import — we own motifs.py. The motif->label map and category map are
config; Sam prunes/edits freely.
"""
import sys, os, re
import chess
sys.path.insert(0, os.path.dirname(__file__))
import motifs as MO
import predicates as PR
# maia_rarity is imported LAZILY inside tag_mistake_full(with_maia=True) — it pulls in the ONNX Maia
# engine, which the product worker doesn't ship. Keeping the import lazy lets the tagger run with
# with_maia=False in environments without the Maia model (e.g. the vendored copy in the ECS worker).

# eval_to_cp maps #N to ±10000; anything above this threshold is treated as a forced mate.
_MATE_SENTINEL = 9000

# direction -> display prefix
DIR_PREFIX = {"missed": "Missed", "allowed": "Allowed", "failed": "Failed",
              "hung": "", "played": "", "info": ""}

# motif key -> human label (combined with the direction prefix)
MOTIF_LABEL = {
    "fork": "Fork", "pin": "Pin", "skewer": "Skewer", "discoveredAttack": "Discovered Attack",
    "deflection": "Deflection", "attraction": "Attraction", "clearance": "Clearance",
    "interference": "Interference", "intermezzo": "Zwischenzug", "overloading": "Overload",
    "xRayAttack": "X-Ray", "trappedPiece": "Trapped Piece", "sacrifice": "Sacrifice",
    "capturingDefender": "Capture of Defender", "hangingPiece": "Hanging Piece",
    "exposedKing": "Exposed King", "kingsideAttack": "Kingside Attack",
    "queensideAttack": "Queenside Attack", "promotion": "Promotion", "underPromotion": "Underpromotion",
    "enPassant": "En Passant", "castling": "Castling", "doubleCheck": "Double Check",
    "attackingF2F7": "f2/f7 Attack", "advancedPawn": "Advanced Pawn", "outpost": "Outpost",
    "mate": "Mate", "anastasiaMate": "Anastasia's Mate", "arabianMate": "Arabian Mate",
    "bodenMate": "Boden's Mate", "doubleBishopMate": "Double Bishop Mate",
    "smotheredMate": "Smothered Mate", "backRankMate": "Back-Rank Mate", "hookMate": "Hook Mate",
    "dovetailMate": "Dovetail Mate", "Fork": "Fork", "Hanging Piece": "Hanging Piece",
    "Pin": "Pin", "Discovered Attack": "Discovered Attack",
}

# Motifs where the FAILED direction (played move itself was the tactic, and it backfired) is sensible.
# Single-move detectors only — sequence motifs need the opponent's cooperation so "failed X" is noise.
# NB "Hanging Piece" is NOT here: capturing a free piece (is_hanging_piece) can't "fail" — you won
# material. It fired "Failed Hanging Piece" on a recapture that was simply sub-optimal (gxf3 took a
# free bishop but Bxc6 first was better). A category error; only real attacking tactics can backfire.
FAILED_OK = {"Fork", "Pin", "Discovered Attack"}

# Mates render with their distance; keep the mate distance string off the label but in evidence.
def _mate_label(key, evidence):
    return MOTIF_LABEL.get(key, key)


def _san_line_to_ucis(start_board: chess.Board, san_line):
    """Convert a SAN move list (from start_board) to UCI strings, stopping at the first unparseable."""
    out = []
    b = start_board.copy()
    for san in san_line:
        try:
            mv = b.parse_san(san)
        except Exception:
            break
        out.append(mv.uci())
        b.push(mv)
    return out


def _best_line_ucis(m):
    """Best line as UCIs from fen_before. Prefer best_uci + best_line_san; ensure move 0 = best_uci."""
    b = m.board_before
    ucis = []
    if m.best_uci:
        try:
            mv = chess.Move.from_uci(m.best_uci)
            if mv in b.legal_moves:
                ucis.append(m.best_uci)
        except Exception:
            pass
    # append the rest of the SAN best line (which usually starts with best move too — dedupe move 0)
    san_ucis = _san_line_to_ucis(b, m.best_line_san)
    if ucis and san_ucis and san_ucis[0] == ucis[0]:
        ucis = san_ucis
    elif not ucis:
        ucis = san_ucis
    else:
        # best_uci leads; continue the line from after best_uci using SAN if it matches
        bb = b.copy(); bb.push(chess.Move.from_uci(ucis[0]))
        ucis += _san_line_to_ucis(bb, m.best_line_san[1:] if m.best_line_san and san_ucis and san_ucis[0]==ucis[0] else m.best_line_san)
    return ucis


def _allowed_line_ucis(m):
    """[played] + refutation, as UCIs from fen_before. Refutation SAN is from fen_after (after played)."""
    b = m.board_before
    try:
        played = chess.Move.from_uci(m.played_uci)
    except Exception:
        return []
    if played not in b.legal_moves:
        return []
    ucis = [m.played_uci]
    after = m.board_after
    ucis += _san_line_to_ucis(after, m.refutation_san)
    return ucis


def _motif_label(key, ev):
    """Map a motif key + evidence to its display label, applying the DEPTH SPLIT for fork:
    depth=0 -> the tactic is the move to play now ('Fork'); depth>0 -> it comes after a setup
    sequence ('Combination → Fork'). Depth is parsed from the 'depth=N ' prefix detect_line adds."""
    if key == "fork" and ev.startswith("depth="):
        try:
            depth = int(ev.split("depth=", 1)[1].split()[0])
        except Exception:
            depth = 0
        return "Fork" if depth == 0 else "Combination → Fork"
    # pin: name what it's against — "Pin (to Queen)" / "Pin (to King)". detect_line prefixes "target=X".
    if key == "pin" and ev.startswith("target="):
        tgt = ev.split("target=", 1)[1].split()[0]
        return f"Pin (to {tgt})"
    # trappedPiece: name the piece — "Trapped Bishop" / "Trapped Knight". detect_line prefixes "piece=X".
    if key == "trappedPiece" and "piece=" in ev:
        piece = ev.split("piece=", 1)[1].split()[0]
        return f"Trapped {piece}"
    return MOTIF_LABEL.get(key, key)


# Motifs whose display label does NOT follow the "Missed X / Allowed X" convention. Each maps a
# motif key -> {direction: explicit label}. exposedKing reads as a positional state, not a sharp
# tactic: your king exposed = "Exposed King"; the enemy king exposed-but-unexploited = "Enemy King
# Exposed". (Sam, 2026-06-14.)
LABEL_OVERRIDE = {
    "exposedKing": {"missed": "Enemy King Exposed", "allowed": "Exposed King"},
}


def _directional_label(key, lab, direction):
    """Apply the Missed/Allowed prefix, unless the motif has an explicit override for this direction."""
    ov = LABEL_OVERRIDE.get(key)
    if ov and direction in ov:
        return ov[direction]
    prefix = {"missed": "Missed", "allowed": "Allowed"}.get(direction, "")
    return f"{prefix} {lab}".strip()


def _motif_tags(m):
    """Layer 2: owned motif detectors across MISSED / ALLOWED / FAILED."""
    out = []  # (label, direction, evidence)
    b = m.board_before
    mover = m.mover
    opp = not mover

    # NOTE: the win%-drop mistake gate is applied ONCE at the tagger entry (tag_mistake_full), not here.
    # _motif_tags only runs on positions already established as real mistakes, so every branch below is
    # unconditional. (GH #29 — was a per-branch cp_loss/win_drop check copy-pasted across detectors.)

    # MISSED: best line, pov = mover.
    best_ucis = _best_line_ucis(m)
    if len(best_ucis) >= 1:
        for key, ev in MO.detect_line(b, best_ucis, mover).items():
            lab = _motif_label(key, ev)
            out.append((_directional_label(key, lab, "missed"), "missed", ev))

    # ALLOWED: [played]+refutation, pov = opponent (== cook's puzzle shape, the validated one)
    allowed_ucis = _allowed_line_ucis(m)
    if len(allowed_ucis) >= 2:   # need the played move + at least one punishment ply
        for key, ev in MO.detect_line(b, allowed_ucis, opp).items():
            lab = _motif_label(key, ev)
            out.append((_directional_label(key, lab, "allowed"), "allowed", ev))

    # FAILED: the played move itself was a (single-move) tactic that backfired
    try:
        played = chess.Move.from_uci(m.played_uci)
        if played in b.legal_moves:
            for key, ev in MO.detect_move(b, played).items():
                if key in FAILED_OK:
                    out.append((f"Failed {key}".strip(), "failed", ev))
    except Exception:
        pass

    # EVAL-BASED MATE FALLBACK: Stockfish often truncates the PV when it sees #N, so
    # mate_in_line() (which requires nodes[-1].is_checkmate()) misses many forced mates.
    # If the eval says "mate" but the PV-based detectors didn't fire, inject the tag.
    existing_labels = {lab for (lab, _, _) in out}
    # Missed Mate: eval_before says mover had a forced mate (mover-POV positive mate)
    if "Missed Mate" not in existing_labels and m.eval_before is not None:
        eb_mover = m.eval_before if m.mover == chess.WHITE else -m.eval_before
        if eb_mover >= _MATE_SENTINEL:
            out.append(("Missed Mate", "missed", "eval: forced mate available (PV truncated)"))
    # Allowed Mate: eval_after says opponent now has a forced mate (mover-POV negative mate)
    if "Allowed Mate" not in existing_labels and m.eval_after is not None:
        ea_mover = m.eval_after if m.mover == chess.WHITE else -m.eval_after
        if ea_mover <= -_MATE_SENTINEL:
            out.append(("Allowed Mate", "allowed", "eval: opponent has forced mate after played move"))

    return _suppress_lesser_under_mate(out)


# Lesser tactical motifs that a forced mate should outrank in the SAME direction. A coach says
# "you missed mate in 3", not "you missed a fork" — the fork is just a step inside the mating net.
# We keep Mate + named mates + Exposed King / attacks (they describe the mating attack), and we never
# touch the OTHER direction or position/material tags.
_MATE_OUTRANKS = {"Fork", "Combination → Fork", "Pin", "Skewer", "Discovered Attack", "Deflection",
                  "Attraction", "Clearance", "Interference", "Zwischenzug", "Overload", "X-Ray",
                  "Capture of Defender", "Hanging Piece", "Sacrifice", "Trapped Piece"}

def _suppress_lesser_under_mate(tags):
    """If a direction produced 'Missed/Allowed Mate', drop the lesser tactical motifs in that
    same direction (config — Sam can disable). Returns the filtered list, order preserved."""
    mate_dirs = {d for (lab, d, ev) in tags if lab in ("Missed Mate", "Allowed Mate")}
    if not mate_dirs:
        return tags
    kept = []
    for (lab, d, ev) in tags:
        if d in mate_dirs:
            # strip the "Missed "/"Allowed "/"Failed " prefix to get the bare motif name
            bare = lab.split(" ", 1)[1] if " " in lab else lab
            # match exact OR parametrized labels like "Pin (to Queen)" -> "Pin"
            base = bare.split(" (", 1)[0]
            if bare in _MATE_OUTRANKS or base in _MATE_OUTRANKS:
                continue
        kept.append((lab, d, ev))
    return kept


def _bare_motif(label):
    """Strip a leading Missed/Allowed/Failed/Hung prefix + a trailing '(...)' qualifier → base motif.
    'Missed Pin (to Queen)' → 'Pin', 'Allowed Battery' → 'Battery'."""
    for pfx in ("Missed ", "Allowed ", "Failed ", "Hung "):
        if label.startswith(pfx):
            label = label[len(pfx):]
            break
    return label.split(" (", 1)[0].strip()


def _collapse_missed_allowed_twins(tags):
    """If the SAME base motif fired both 'Missed X' and 'Allowed X' on this one move, keep only the
    MISSED one and drop the ALLOWED twin. A single move being tagged as both missing motif X and
    allowing motif X is noise — the review is of the player's own move, so 'you missed X' is the honest
    read; the 'allowed' twin double-counts the same geometry. (Sam, 2026-07-11 — the Battery double-fire
    on ply 28.) Operates on (label, direction, evidence, layer) tuples; order preserved."""
    missed_bases = {_bare_motif(t[0]) for t in tags if t[1] == "missed"}
    if not missed_bases:
        return tags
    return [t for t in tags
            if not (t[1] == "allowed" and _bare_motif(t[0]) in missed_bases)]


# Tactical motif substrings — a tag containing one of these IS a tactic. Direction then splits it into
# "Missed Tactic" (find it) vs "Allowed Tactic" (prevent it) — genuinely different drill skills.
_TACTIC_WORDS = ("fork", "pin", "skewer", "discovered", "deflection", "attraction", "clearance",
                 "interference", "zwischenzug", "overload", "x-ray", "trapped", "sacrifice",
                 "double check", "combination")


def _direction_from_label(label):
    """Derive direction from the label prefix when the caller doesn't pass it explicitly
    (the taxonomy builder enumerates labels with their direction baked into the prefix)."""
    for pfx, d in (("Missed ", "missed"), ("Allowed ", "allowed"),
                   ("Hung ", "hung"), ("Failed ", "failed")):
        if label.startswith(pfx):
            return d
    return ""


def categorize(label, direction=None):
    """Map a tag → one of the 10 drill categories (skill-based, direction-aware), plus Meta/Other for
    info/context tags. The 10: Hung Piece, Missed Capture, Missed Tactic, Missed Mate, Allowed Tactic,
    Calculation, Trading, Position, King Safety, Endgame. (Sam, 2026-06-14 — replaces the old
    chess-concept scheme; "hung a piece" and "missed a free piece" are opposite SKILLS, not both
    "Material".)"""
    if direction is None:
        direction = _direction_from_label(label)
    l = label.lower()

    # Phase / game-state context tags (direction == "info") are not drill categories.
    # Bare phase words only — "Rook Endgame" / "Bishop Endgame (...)" are endgame-TYPE tags (below).
    if l in ("opening", "middlegame", "endgame"):
        return "Other"
    if l in ("winning", "losing", "equal") or l.startswith("blunder while"):
        return "Meta"
    if any(w in l for w in ("only move", "multiple good")):
        return "Meta"

    # Mate vision is its own skill when MISSED; allowing mate is King Safety. Word-boundary so
    # "Material"/"checkmate-in-text" don't false-match.
    if re.search(r"\bmate\b", l) or "back-rank" in l:
        return "Missed Mate" if direction == "missed" else "King Safety"

    # Endgame technique + endgame-type context tags ("Rook Endgame", "Pawn Endgame", etc.).
    # BEFORE king/pawn substring branches.
    if "endgame" in l or "activity" in l or any(w in l for w in (
            "opposition", "passed pawn", "passer", "behind passer",
            "promotion", "pawn race", "en passant", "rook to 7th", "rook cut-off",
            "active rook", "blockade", "connected passers", "simplif", "trade to simplify",
            "king direction", "outside passer", "push to promote", "rook to open file",
            "protected passer", "square rule", "breakthrough", "perpetual")):
        return "Endgame"

    # Enemy king exposed / removing the defender of the enemy king = a missed attacking chance
    # (Missed Tactic), NOT your own king safety. Check BEFORE the King Safety branch (catches king/exposed).
    if l == "enemy king exposed" or l == "missed remove the guard":
        return "Missed Tactic"
    # King safety (your king). "Exposed King" (your king exposed), castling, kingside/queenside attack.
    if any(w in l for w in ("exposed king", "kingside attack", "queenside attack", "castl",
                            "f2/f7", "pawn move exposed king", "king in center")):
        return "King Safety"

    # Hung material (you dropped a piece in one move).
    if l.startswith("hung ") or l == "allowed hanging piece":
        return "Hung Piece"

    # Missed free material (opponent gave you something).
    if l.startswith("missed free") or l.startswith("missed winning capture") \
       or l in ("missed hanging piece", "missed capture of defender", "missed capture (pawn)"):
        return "Missed Capture"

    # Trading (exchange decisions).
    if "exchange" in l or l == "missed pawn trade" or l == "premature trade":
        return "Trading"

    # Calculation (saw it, miscounted / wrong execution). Greedy Capture = grabbed material when a
    # quiet move was better (a calculation/judgment error); Failed X = your own tactic backfired.
    # Desperado = had a tactical resource (doomed piece) and didn't cash it in.
    # Pawn Grab While Undeveloped = chose material over development (judgment/calculation error).
    if l in ("greedy capture", "missed desperado", "pawn grab while undeveloped") or l.startswith("failed "):
        return "Calculation"

    # Threat awareness / Active Defense — you ignored a threat or failed to USE a defensive resource
    # (unpin, interpose, remove the attacker, counter-sac, cross-check). These are the "missed defense"
    # half of Defensive Tactics. Checked BEFORE the tactic-words branch so "Missed Unpinning Resource"
    # routes here, not to Offensive via its "pin" substring.
    if l in ("ignored threat", "missed defensive resource") or any(w in l for w in (
            "unpinning", "interposition", "counter-sacrifice", "removing the attacker", "cross-check")):
        return "Allowed Tactic"

    # Premature attack = positional judgment (attacked before developing).
    if l == "premature attack":
        return "Position"

    # Tactical motifs — split by direction. Battery, Overloading, Doubled Rooks are tactical patterns.
    if any(w in l for w in _TACTIC_WORDS) or any(w in l for w in ("battery", "overloading", "doubled rooks")):
        return "Missed Tactic" if direction == "missed" else "Allowed Tactic"
    if "capture of defender" in l:   # Allowed Capture of Defender = a tactic you allowed
        return "Allowed Tactic"

    # Positional (plan execution + structure). Doubled Rooks is a positional setup.
    if any(w in l for w in ("advanced pawn", "isolated pawn", "doubled pawn", "backward pawn",
                            "outpost", "open file", "piece activation", "prophylaxis", "pawn break",
                            "tempo push", "tempo", "development", "doubled rooks")):
        return "Position"

    # Missed Faster Mate = still in the mate category.
    if l == "missed faster mate":
        return "Missed Mate"

    return "Other"


# Move classifications that earn EXPLAIN tags when the caller supplies one. Inaccuracy is INCLUDED
# here (so review cards can explain it) but the caller is responsible for keeping inaccuracies OUT of
# stats/drills — those filter on classification downstream. mistake/blunder are the counting classes.
_EXPLAIN_CLASSIFICATIONS = {"inaccuracy", "mistake", "blunder"}

# GOOD classifications never earn explain (mistake) tags — they're key moments worth SHOWING (a great
# find), but there's nothing to explain as "wrong." A good move must not get Missed/Allowed X, and must
# NOT be rescued by the mate exemption below (a brilliant move that keeps a forced mate allowed/missed
# nothing). 2026-07-11: real-game review showed Missed Mate / Allowed Sacrifice on brilliant moves.
_GOOD_CLASSIFICATIONS = {"brilliant", "great", "excellent", "good", "opening", "best", "book"}


def tag_mistake_full(m, with_maia=True, classification=None):
    fine = []  # (label, direction, evidence, layer)

    # ONE entry gate (GH #29): only EXPLAIN tags (mistake assertions) are gated; INFO/orient tags
    # (phase, game-state, endgame-TYPE) always fire so they can classify the position for drill-bucket
    # filtering. Two ways to pass the gate:
    #   1. classification given (prod knows it authoritatively): explain iff it's inaccuracy/mistake/
    #      blunder. This is how the frontend gets inaccuracies EXPLAINED on review cards without
    #      lowering the win_drop threshold globally. Explicit classification WINS over win_drop.
    #   2. classification=None (research corpus, which never passes it): fall back to win_drop >=
    #      WIN_DROP_MIN (=10 = the mistake/blunder boundary), so research stays mistake/blunder-only.
    # MATE EXEMPTION (both paths): missing/allowing a forced mate is ALWAYS a real mistake regardless
    # of win_drop (which gets squished by the ±1200 clamp when going from mate→still-winning).
    #
    # GOOD-MOVE SUPPRESSION (hard override, wins over everything incl. the mate exemption): a good
    # classification, OR a move that IS the engine's best move (played==best — you can't "miss" or
    # "allow" anything by playing the top move), earns ZERO explain tags. Checked first so a brilliant
    # move in a mating position doesn't get Missed Mate, and so Qh2# (played==best==the mate) is silent.
    played_is_best = bool(m.best_uci) and m.played_uci == m.best_uci
    is_good_move = (classification in _GOOD_CLASSIFICATIONS) or played_is_best

    has_mate_before = (m.eval_before is not None and
                       (m.eval_before if m.mover == chess.WHITE else -m.eval_before) >= _MATE_SENTINEL)
    has_mate_after = (m.eval_after is not None and
                      (m.eval_after if m.mover == chess.WHITE else -m.eval_after) <= -_MATE_SENTINEL)
    if is_good_move:
        is_mistake = False
    elif classification is not None:
        is_mistake = classification in _EXPLAIN_CLASSIFICATIONS or has_mate_before or has_mate_after
    else:
        is_mistake = m.win_drop >= PR.WIN_DROP_MIN or has_mate_before or has_mate_after

    for (label, direction, ev) in _motif_tags(m):
        if not is_mistake:                 # motifs are always explain tags
            continue
        fine.append((label, direction, ev, "tactic"))

    for (label, direction, ev) in PR.tag_predicates(m):
        if direction != "info" and not is_mistake:
            continue                        # gate explain predicates; keep phase/state/endgame-type
        fine.append((label, direction, ev, "position"))

    # dedupe by display label, keep first
    seen = set(); tags = []
    for t in fine:
        if t[0] in seen:
            continue
        seen.add(t[0]); tags.append(t)

    tags = _collapse_missed_allowed_twins(tags)

    cat_set = sorted({categorize(t[0], t[1]) for t in tags})

    maia = {}
    if with_maia:
        try:
            import maia_rarity as MR   # lazy — only load the ONNX Maia engine when actually needed
            maia = MR.rarity(m)
        except Exception:
            maia = {}

    return {
        "tags": [{"label": l, "direction": d, "evidence": e, "layer": ly} for (l, d, e, ly) in tags],
        "categories": cat_set,
        "maia": maia,
    }
