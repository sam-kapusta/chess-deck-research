# Rule-Based Mistake Tagger — Design

**Date:** 2026-06-07
**Status:** approved (Approach A), autonomous build while Sam is away
**Supersedes (for the tagging goal):** the d64_k1 SAE as the per-position assignment engine. The SAE's
gift was discovering the *category vocabulary* (incl. strategic tags like "Deserted the Defender",
"Won't Trade Into Won Endgame"); it is NOT used at runtime here.

## Why this exists

Goal: tag each blunder in a game with crisp, accurate **mistake categories** for a per-game coaching
report. **Tags ARE the output** (accuracy is what matters; prose is secondary).

The SAE approach hit an intrinsic wall: polysemantic features (f55 cons 50, f0's peak/median split).
An SAE is unsupervised *discovery*; we already know the vocabulary, so this is supervised
*classification* — better done with deterministic rules over the board + engine lines.

Key finding (2026-06-07): **Lichess's puzzle tagger (`lichess-puzzler/tagger/cook.py`) is pure
rule-based python-chess** — zero ML, zero engine eval. Every theme incl. named mates (Anastasia,
Arabian, Boden, smothered, back-rank, hook, dovetail) is geometric pattern code. Validated that it
runs on our data: a blunder's refutation line tagged `mateIn1, mate, doubleBishopMate` correctly.
So the hard tier (named mates / advanced tactics) is rule-friendly, not ML. We adapt cook.py.

## The Mistake object (Layer 0 — data contract)

Everything reads this. Produced from the existing Stockfish analysis (`stockfish_data_v2.json` has
`refutation_lines` + `top_lines`; analyze_cli produces the same shape for fresh games).

```
Mistake = {
  fen_before,            # position before the move
  played_uci, played_san,
  best_uci,  best_san,
  best_line:   [san...], # Stockfish PV from fen_before (the line you SHOULD have played)
  refutation:  [san...], # Stockfish's punishment of the played move (from fen AFTER played)
  eval_before, eval_after,   # white-POV cp (or mate)
  cp_loss,
  mover,                 # WHITE/BLACK — whose mistake
  player_elo, oppo_elo,
}
```

## Layered architecture

Four independent tag-producing layers, each reads the Mistake object, each emits `(tag, evidence)`
pairs. A final rollup maps fine tags → high-level categories. **Build order 0→2→1→3** (2 is riskiest,
do it first to de-risk; 1 is ours and easy; 3 is standalone).

### Layer 2 — cook.py tactical/mate motifs (Approach A: vendor + adapt)

- **Vendor** `cook.py`, `util.py`, `model.py`, `zugzwang.py` verbatim under
  `scripts/04_tagger/vendor/lichess_puzzler/` with a `UPSTREAM.md` (commit hash, "do not edit").
- **Adapter** (`cook_adapter.py`, ours) builds three synthetic `Puzzle`s per mistake and runs `cook()`:
  1. **best-line** (pov = mover) → motif here = **"Missed X"**
  2. **refutation** (pov = opponent; from fen AFTER the played move) → motif here = **"Allowed X"**
  3. **played-move as a 1-move line** (pov = mover) + **eval crashed** → motif here = **"Failed X"**
     (the player attempted a tactic that backfired — e.g. unsound sac/check).
- **POV fix:** `model.Puzzle.__post_init__` sets `pov = not game.turn()` (puzzle convention where
  mainline[0] is the opponent's setup move). For our synthetic lines the first move IS the pov side's,
  so the adapter sets `pov` explicitly after construction, overriding the auto value. (This is the bug
  found in validation — fork tagged on the wrong side.)
- **Tag filter:** keep only motif/mate tags; DROP puzzle-meta tags that mean nothing for a single
  mistake: `crushing, advantage, equality, crushing, veryLong, long, short, oneMove, quietMove,
  master*, mateInN granularity` (keep `mate` + named mates, drop the `mateInN` count unless useful).
  The keep-list is a config constant so Sam can add/drop tags trivially.
- **Direction resolver:** for each kept motif, the line it fired on decides Missed/Allowed/Failed.
  If a motif fires on multiple lines, prefer the one matching the eval story (Allowed if eval crashed
  for mover; Missed if best-line is winning).
- **Coarse rollup:** if no specific motif fires but the lines are forcing, emit `Missed Tactic` /
  `Allowed Tactic` so nothing is unlabeled.

cook TagKinds we map (the valuable subset): fork, pin, skewer, discoveredAttack, deflection,
attraction, clearance, interference, intermezzo (=zwischenzug), overloading, xRayAttack, trappedPiece,
sacrifice, capturingDefender, hangingPiece, exposedKing, kingsideAttack, queensideAttack, promotion,
underPromotion, enPassant, castling, doubleCheck, and ALL named mates (anastasia/arabian/boden/
doubleBishop/smothered/backRank/hook/dovetail), plus endgame types (rook/queen/pawn/knight/bishop/
queenRook). zugzwang via `zugzwang.py`.

### Layer 1 — our position predicates (deterministic, the stuff Lichess lacks)

Pure python-chess on the Mistake object. Each is its own small function, independently testable:
- **Phase:** opening / middlegame / endgame (piece count + move number); specific endgame types reuse
  cook's (rook/queen/pawn endgame) or our material check.
- **Game state:** Blunder While Winning / Losing / Drawn (from eval_before bands).
- **Capture vs Exchange (by piece):** best move captures undefended → "Missed Free Capture (Piece)";
  captures defended/even → "Missed Exchange (Piece)". (The defended? + piece-type predicate validated
  this session.)
- **Hung material:** played move loses own material in the refutation line (end-of-line material delta,
  NOT one-ply — the validated metric); name the piece if dominant → "Hung Piece/Queen/Rook…".
- **King safety:** King in Center (king off castled squares, middlegame), Lost Castling Rights (played
  move forfeits castling), Pawn Move Exposed King (pawn move in front of own king's shelter).
- **Pawn structure deltas:** played move creates own Isolated / Doubled / Backward pawn.
- **Wrong Move Order:** played move IS in Stockfish's best line, just played at the wrong time (compare
  played_uci against best_line moves).
- **Only Move vs Multiple Good Moves:** cp-gap between top engine moves (single move within ~50cp = "Only
  Move missed"; several within 50cp = "Multiple Good Moves").
- **Captured With Wrong Piece:** played move and best move both capture the SAME square, different piece.

### Layer 3 — Maia rarity (standalone, engine already built)

Uses `maia3_engine.analyze(fen, elo)` (ONNX, offline, in chess-deck-code/backend/mcp). Emits
**numeric annotations**, not tags:
- played-move probability at player's elo → "N% of players at your level play this"
- played-move probability at a higher elo (e.g. +400) → "higher-rated players blunder X% less here"
- whether the best move's Maia-probability rises sharply with elo → "this is a skill-gap move"

### Category rollup

Fine tags → high-level categories for the report's structure. Initial map (Sam will revise):
- **Tactical (Missed/Allowed/Failed):** all cook motifs + named mates
- **Material:** Free Capture, Exchange, Hung Piece, Bad Capture, Bad Trade
- **King Safety:** King in Center, Lost Castling, Exposed King, Allowed Mate
- **Endgame:** endgame-type tags, Wrong Pawn Push, Losing Opposition, Promotion Blunder, Wrong Move Order in endgame
- **Positional:** Wasted Tempo, Pawn Structure, Bad Development
- **Meta:** Only Move, Multiple Good Moves, Blunder While Winning/Losing/Drawn, Maia rarity

## Output + evaluation artifact

For each tagged blunder: `{move, tags: [(tag, direction, evidence, layer)], category, maia}`.
Run over the corpus (the player's analyzed blunders) and render a **tag-browser atlas**: group by tag,
show example boards + the lines + evidence, so Sam can judge each tag's crispness on return. Mark each
tag's support count. This is the deliverable Sam evaluates.

## What's explicitly OUT of this build

- The SAE at runtime (retired for assignment).
- Layer 4 strategic/LLM tags ("Deserted the Defender" etc.) — those aren't single-line rules; later spec.
- Training/distilling a classifier — only if an offline-fast need arises after rules+coverage are known.
- Aggregate player profiling — consumer is per-game report for now.

## Testing

- **Validation set:** the ~20 hand-confirmed d64_k1 gold features map to expected tags (f59→Allowed
  Check/Mate, f54→Missed Fork, f17→Missed Free Capture, f3→Missed Hanging Piece, f31→Missed Exchange).
  Each layer's output on those positions must match the human label. This is the regression check.
- **Per-predicate unit checks:** hand-built FENs for each Layer-1 predicate.
- **cook adapter:** assert pov fix (fork tags on mover's line, not opponent's) + tag filter (no
  `crushing`/`veryLong` leaking through).

## Risks / known issues

- cook.py tuned for puzzles (forced winning lines). Blunder lines are forcing-for-opponent; mapping is
  clean but POV must be set explicitly (validated bug). Mitigated by the regression set.
- python-chess version drift (cook built for 1.3, we run 1.11) — imports fine, but watch for API
  changes in attack/pin calls. Pin to behavior via the regression tests.
- Many tags will be noisy / Sam will drop them — that's expected and WHY tag selection is a config list,
  not hardcoded. Over-produce, let Sam prune.
