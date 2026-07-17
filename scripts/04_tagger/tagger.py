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
    "exposedKing": "Exposed King", "promotion": "Promotion", "underPromotion": "Underpromotion",
    "enPassant": "En Passant", "castling": "Castling", "doubleCheck": "Double Check",
    "attackingF2F7": "f2/f7 Attack", "advancedPawn": "Advanced Pawn", "outpost": "Outpost",
    "mate": "Mate", "anastasiaMate": "Anastasia's Mate", "arabianMate": "Arabian Mate",
    "bodenMate": "Boden's Mate", "doubleBishopMate": "Double Bishop Mate",
    "smotheredMate": "Smothered Mate", "backRankMate": "Back-Rank Mate", "hookMate": "Hook Mate",
    "dovetailMate": "Dovetail Mate", "Fork": "Fork", "Hanging Piece": "Hanging Piece",
    "Pin": "Pin", "Discovered Attack": "Discovered Attack",
}

# The FAILED direction (played move geometrically creates a fork/pin/discovered-attack pattern) was
# DELETED 2026-07-14. It was a catch-all: `detect_move` sees the PATTERN, not whether the player attempted
# the tactic or it backfired BECAUSE of that pattern. Audit of the 60k corpus: only 20-27% of Failed X
# fires had the moved piece even recaptured; in the most-favorable subset (played a capture AND piece
# recaptured), 69% co-fired a material tag (Hung/Sacrifice/Greedy/Missed Free) that named the real loss
# and only 7/324 were sole-explain. "Failed" asserts intent+causation from pure geometry it can't verify;
# when the move truly loses, a material/sacrifice tag already names it. Deleting left 4% of Failed X fires
# untagged — positions whose only tag was a false "you failed a tactic", worse than silence. Empty set
# keeps the plumbing inert (no branch fires). See tagger_feature_ledger.md.
FAILED_OK = set()

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
    sequence ('Combination → Fork'). Also names the forking PIECE ('Knight Fork') from the
    'forkpiece=X' prefix detect_line adds (#53). Both prefixes may appear, in any order."""
    if key == "fork":
        depth = 0
        if "depth=" in ev:
            try: depth = int(ev.split("depth=", 1)[1].split()[0])
            except Exception: depth = 0
        piece = ""
        if "forkpiece=" in ev:
            piece = ev.split("forkpiece=", 1)[1].split()[0]
        noun = f"{piece} Fork".strip() if piece else "Fork"   # "Knight Fork" / "Fork"
        return noun if depth == 0 else f"Combination → {noun}"
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

# Motifs that only make sense in the MISSED direction — the ALLOWED twin is not a teachable mistake, so
# we never emit it. `castling`: "Allowed Castling" (the opponent castled in the refutation line) fires on
# normal chess — 78% of its cases had the opponent castle 2-4 plies deep, and reading the first-reply
# cases showed the real mistake was always something concrete (hung a piece, missed a tactic) with the
# opponent routinely castling as their reply. Deleted 2026-07-14 (same reasoning as allowed_battery: the
# "allowed" direction of a motif that's only meaningful as your own move). See the feature ledger doc.
_MISSED_ONLY_MOTIFS = {"castling"}


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
            if key in _MISSED_ONLY_MOTIFS:      # e.g. castling — no teachable "Allowed" twin
                continue
            lab = _motif_label(key, ev)
            out.append((_directional_label(key, lab, "allowed"), "allowed", ev))

    # (FAILED direction deleted 2026-07-14 — see FAILED_OK note. It was a geometry catch-all.)

    # EVAL-BASED MATE FALLBACK: Stockfish often truncates the PV when it sees #N, so
    # mate_in_line() (which requires nodes[-1].is_checkmate()) misses many forced mates.
    # If the eval says "mate" but the PV-based detectors didn't fire, inject the tag.
    existing_labels = {lab for (lab, _, _) in out}
    # Missed Mate: eval_before says mover had a forced mate (mover-POV positive mate)
    if "Missed Mate" not in existing_labels and m.eval_before is not None:
        eb_mover = m.eval_before if m.mover == chess.WHITE else -m.eval_before
        if eb_mover >= _MATE_SENTINEL:
            out.append(("Missed Mate", "missed", "eval: forced mate available (PV truncated)"))
    # PV-DEPTH fallback (#56): the eval sentinel misses mate-in-N when Stockfish reported a large-but-
    # sub-sentinel score or the cache lost the mate flag. If the BEST line itself reaches checkmate,
    # the player missed a forced mate — fire it. (SAE jr2048: many 'Missed Mate on Exposed King'
    # features had best-move-is-check on 90%+ but Missed Mate fired on <half via the sentinel alone.)
    if "Missed Mate" not in {lab for (lab, _, _) in out} and m.best_line_san:
        try:
            bb = chess.Board(m.fen_before)
            for san in m.best_line_san:
                bb.push(bb.parse_san(san))
                if bb.is_checkmate():
                    out.append(("Missed Mate", "missed", "best line forces checkmate"))
                    break
        except Exception:
            pass
    # Allowed Mate: eval_after says opponent now has a forced mate (mover-POV negative mate)
    if "Allowed Mate" not in existing_labels and m.eval_after is not None:
        ea_mover = m.eval_after if m.mover == chess.WHITE else -m.eval_after
        if ea_mover <= -_MATE_SENTINEL:
            out.append(("Allowed Mate", "allowed", "eval: opponent has forced mate after played move"))
    # PV-DEPTH fallback for Allowed Mate (#56): if the REFUTATION line (opponent's punishment of the
    # played move, from the board AFTER the played move) reaches checkmate, the player allowed a mate.
    if "Allowed Mate" not in {lab for (lab, _, _) in out} and m.refutation_san:
        try:
            bb = m.board_after()
            for san in m.refutation_san:
                bb.push(bb.parse_san(san))
                if bb.is_checkmate():
                    out.append(("Allowed Mate", "allowed", "refutation line forces checkmate"))
                    break
        except Exception:
            pass

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
            # match exact, parametrized "Pin (to Queen)" -> "Pin", or piece-prefixed "Knight Fork" -> "Fork"
            base = bare.split(" (", 1)[0]
            tail = base.rsplit(" ", 1)[-1] if " " in base else base   # "Knight Fork" -> "Fork"
            if bare in _MATE_OUTRANKS or base in _MATE_OUTRANKS or tail in _MATE_OUTRANKS:
                continue
        kept.append((lab, d, ev))
    return kept


def _bare_motif(label):
    """Strip a leading Missed/Allowed/Failed/Hung prefix + a trailing '(...)' qualifier → base motif.
    'Missed Pin (to Queen)' → 'Pin', 'Allowed Doubled Rooks' → 'Doubled Rooks'."""
    for pfx in ("Missed ", "Allowed ", "Failed ", "Hung "):
        if label.startswith(pfx):
            label = label[len(pfx):]
            break
    return label.split(" (", 1)[0].strip()


# ---------------------------------------------------------------------------
# PARENT → CHILD tag suppression (declarative).
# A "parent" tag is a GENERIC label that a more SPECIFIC "child" tag subsumes. When both fire on one
# move, the parent is noise (the child says the same thing but sharper — names the piece, the target,
# the count). Drop the parent, keep the child. The parent still fires ALONE when no child matched (it
# is then the only signal), so this is suppression-when-redundant, not deletion.
#
# Each row: (parent_label, child_predicate). child_predicate(other_label) -> True if that other tag on
# the SAME move is a specific child of the parent. Add a row here to declare a new relationship — no
# new function needed. Order-preserving; applied once in tag_mistake_full after the twin collapse.
#
# Rows (with the evidence that justifies each):
#   • "Allowed Hanging Piece" → any "Hung <Piece>": same refutation-captures-your-piece fact, but Hung
#     names the piece + net count. Corpus: co-fire ~79%; the ~21% AHP-alone cases (delayed/net-recovered
#     losses hung_material skips) keep it. (Sam, 2026-07-12 — the Bxe7 card buried Greedy Capture.)
_PARENT_CHILD = [
    ("Allowed Hanging Piece", lambda l: l.startswith("Hung ")),
]


def _suppress_parents(tags):
    """Drop any parent tag whose more-specific child also fired on this move (see _PARENT_CHILD).
    Operates on (label, direction, evidence, layer) tuples; order preserved."""
    labels = [t[0] for t in tags]
    drop = set()
    for parent, is_child in _PARENT_CHILD:
        if parent in labels and any(is_child(l) for l in labels if l != parent):
            drop.add(parent)
    return [t for t in tags if t[0] not in drop] if drop else tags


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
    if "→" in l and any(w in l for w in ("winning", "losing", "drawn", "even")):
        return "Meta"                     # conversion_outcome result-band tags (e.g. "Winning → Losing")
    if l in ("sharp blunder", "slow bleed"):
        return "Meta"                     # blunder_severity descriptors
    if l in ("only good move missed", "careless blunder"):
        return "Meta"                     # move_difficulty descriptors
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
            "protected passer", "square rule", "breakthrough", "perpetual", "stalemate")):
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

    # Allowed a pawn grab (quiet move let the opponent snap a pawn the best move prevented). It's a
    # material/tactical concession, not a dropped piece → Allowed Tactic (matches its allowed direction).
    if l == "allowed pawn capture":
        return "Allowed Tactic"

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
    # Unsound Sacrifice = threw material at the king with no compensation (a played-move miscalc). Must
    # be checked BEFORE the tactic-words branch below, or its "sacrifice" substring routes to Allowed Tactic.
    if l in ("greedy capture", "pawn grab while undeveloped", "unsound sacrifice",
             "pointless check", "wrong check") or l.startswith("failed "):
        return "Calculation"

    # Missed Attacking Check = a forcing check on the enemy king you didn't play (offensive tactic).
    if l == "missed attacking check":
        return "Missed Tactic"
    # Missed Zwischenzug = right capture, wrong order (a calculation/move-order error).
    if l == "missed zwischenzug":
        return "Calculation"
    # Missed Greek Gift = a missed bishop sacrifice cracking the king (offensive tactic).
    if l == "missed greek gift":
        return "Missed Tactic"
    # Missed Sacrifice = a broader king-zone sac the player missed (also offensive tactic).
    if l == "missed sacrifice":
        return "Missed Tactic"

    # Threat awareness / Active Defense — you ignored a threat or failed to USE a defensive resource
    # (unpin, interpose, remove the attacker, counter-sac, cross-check). These are the "missed defense"
    # half of Defensive Tactics. Checked BEFORE the tactic-words branch so "Missed Unpinning Resource"
    # routes here, not to Offensive via its "pin" substring.
    if any(w in l for w in (
            "unpinning", "interposition", "counter-sacrifice", "removing the attacker", "cross-check")):
        return "Allowed Tactic"

    # Premature attack / missed development = positional judgment (piece placement in the opening).
    if l in ("premature attack", "missed development"):
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


# ---------------------------------------------------------------------------
# FAMILY roll-up (declarative). Distinct from categorize(): categorize maps a tag to one of the 10
# DRILL SKILLS ("Missed Capture"); family_of maps a tag to its CONCEPT PARENT — the coarser concept the
# piece-specific variants are instances of ("Missed Free Material"). Why both: the product shows the
# SPECIFIC chip ("Missed Free Rook") because that's what teaches; but for AGGREGATION (SAE-feature
# matching, coverage measurement, vote-dilution fixes) the piece split is noise — 5 variants that each
# look minor actually are ONE dominant concept. family_of collapses them so the concept's true weight
# is visible. (Sam, 2026-07-12: the "not_covered" SAE features were firing Missed Free {Q,R,B,N,P} on
# 30-60% of their top positions, but no single piece variant cracked the judge's top-5 view, so the
# concept looked absent. It wasn't — it was fragmented.)
#
# family_of(label) returns the parent concept, or the label unchanged if it heads no family (it IS its
# own concept). The parent is NOT itself an emitted chip — it exists for grouping only.
#
# DIRECTION IS PRESERVED. "Missed X" and "Allowed X" are opposite SKILLS (tactic vision vs defense), so
# they never roll into a common parent. Fork variants roll to the *directional* generic (Missed Knight
# Fork -> Missed Fork; Allowed Queen Fork -> Allowed Fork) which already exists as an emitted label.
# Missed Free Material / Hung Material / Missed Exchange are single-direction concepts by construction.
def _is_fork(l):
    tail = l.split("→")[-1].strip() if "→" in l else l
    return tail.endswith(" fork") or tail == "fork"

_FAMILY = [
    # (parent, predicate on the lowercased label)
    ("Missed Free Material", lambda l: l.startswith("missed free ") or l in (
        "missed capture (pawn)", "missed hanging piece", "missed capture of defender")
        or l.startswith("missed winning capture")),
    ("Hung Material",        lambda l: l.startswith("hung ")),   # Hung Material is also emitted directly
    ("Missed Exchange",      lambda l: "exchange" in l or l == "missed pawn trade"),
    ("Missed Fork",          lambda l: l.startswith("missed ") and _is_fork(l)),
    ("Allowed Fork",         lambda l: l.startswith("allowed ") and _is_fork(l)),
    # King Safety = YOUR move endangered YOUR OWN king (exposed it / let the opponent attack it). A
    # SECOND concept that co-fires with material tags — the SAE surfaced 7 features Opus calls "King
    # Walks Into Danger / King Exposed to Attack" that ALSO hang a piece; each is BOTH "Hung Queen"
    # (material) AND "King Safety" (the attack), fragmented across ~8 own-king tags so neither the
    # material nor the safety concept dominated alone (#50, Sam's multi-tag point). DIRECTION MATTERS:
    # only ALLOWED / self-exposing tags belong — "Enemy King Exposed" and "Missed Kingside Attack" are
    # the OPPOSITE skill (attacking) and must NOT roll in. "Allowed Mate" stays its own concept (a
    # forced-mate lesson, not general king exposure — Sam's call).
    ("King Safety",          lambda l: l in _KING_SAFETY_FRAGMENTS),
]

# Own-king-endangered fragments (see the King Safety family above). Allowed Mate deliberately excluded.
# "Pawn Move Exposed King" is ALSO excluded — it's too trigger-happy (fires on ANY pawn move near a
# king) and was the noise source: on the 7 real king-safety SAE features the family holds at 14-40/200
# WITHOUT it (driven by Exposed King / Allowed Kingside Attack / Allowed Pin-to-King), but on ~13 plain
# HANGING-PIECE features it inflated King Safety to 40-51 that COLLAPSE to 2-18 without it. So it stays
# its own chip but does not define the family. (Sam, 2026-07-13 — validated by decomposing the vote.)
_KING_SAFETY_FRAGMENTS = {
    "exposed king", "king in center", "lost castling rights",
    "allowed kingside attack", "allowed queenside attack", "allowed f2/f7 attack",
    "allowed double check", "allowed pin (to king)", "recapture exposed king",
}

# POSITION-GATED family: the pawn/king endgame technique concept. Unlike the families above (which
# roll SAME-direction piece variants of ONE skill), this groups DIFFERENT tags that are each fragments
# of "mishandled a king-and-pawn endgame" — Wrong Pawn Race, Lost the Opposition, Bad Simplification,
# Missed Prophylaxis, Missed King Activity, etc. They fragment the concept so no single one dominates
# a feature (SAE audit: 29 features Opus-labeled "Pawn Endgame Tempo/Conversion Error" scattered these
# across 6-8 tags, #50). We ONLY roll them up when the POSITION is a K+P (or K+P + at most one heavy
# piece) endgame — because Missed Prophylaxis / Bad Simplification also fire in middlegames, where they
# are NOT this concept. So this family needs the board; static/label-only callers never trigger it
# (correct — it's a position-dependent grouping, not an intrinsic property of the label).
_PAWN_ENDGAME_FRAGMENTS = {
    "wrong pawn race", "lost the opposition", "bad simplification", "missed king activity",
    "wrong king direction", "missed prophylaxis", "missed push to promote", "missed passed pawn",
    "missed advanced pawn", "allowed advanced pawn",
    # promotion-race subset (SAE ground-truth: 20 more jr512 features Opus-labeled "passed-pawn
    # endgame" were dominated by these + swallowed by Hung Material — f327/f399/f441/f277, #50).
    "allowed promotion", "missed promotion", "allowed passed pawn",
}


def _is_pawn_endgame_board(board):
    """K+P endgame, tolerating at most ONE heavy piece per side (covers the near-pawn-endings the SAE
    grouped — e.g. a lone rook that's about to trade). Pure K+P is the core; the one-piece slack keeps
    'pawn endgame technique' from splitting off the rook-endgame conversions it clearly clusters with."""
    import chess as _c
    heavy = 0
    for p in board.piece_map().values():
        if p.piece_type in (_c.KING, _c.PAWN):
            continue
        heavy += 1
        if heavy > 2:  # >1 per side on average — no longer a pawn ending
            return False
    return True


def family_of(label, board=None):
    """Roll a specific tag up to its concept parent for aggregation/matching (see _FAMILY).
    Returns the label itself if it heads no family. NOT used to pick the displayed chip.

    `board` (optional): when supplied AND the position is a pawn endgame, the pawn-endgame technique
    fragments roll up to 'Pawn Endgame Technique'. Omitted → that family never fires (the static
    families are unchanged)."""
    l = label.lower()
    if board is not None and l in _PAWN_ENDGAME_FRAGMENTS and _is_pawn_endgame_board(board):
        return "Pawn Endgame Technique"
    for parent, is_member in _FAMILY:
        try:
            if is_member(l):
                return parent
        except Exception:
            continue
    return label


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
    # NOTE: _suppress_parents / _PARENT_CHILD exists but is NOT called here. Per the 2026-07-12 grill
    # decision (Sam), parent→child suppression is a DISPLAY concern (the frontend's display_game flag),
    # not a data concern — every tag keeps its own stats/drill counts. The function stays in the file
    # for when the frontend display_game flag is wired; it's just not part of the tagging pipeline.

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
