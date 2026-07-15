# Tagger Feature Ledger — every detector, its status, and why

**What this is:** the single running index of every mistake tag the rule tagger can emit (or once could).
One line per concept: what it detects, its corpus fire rate, and — for anything gated or deleted — *why*.
Evergreen (no date prefix); update it whenever a detector is added, gated, or removed. The point is that
a future session never re-litigates a deletion or re-adds a known catch-all without reading the reason.

**Where the truth lives:** detectors are `scripts/04_tagger/predicates.py` (`ALL_PREDICATES`) + the motif
system (`motifs.py` `LINE_DETECTORS`, routed to Missed/Allowed in `tagger.py._motif_tags`). The shipped
label→category→blurb map is `output/mistakeTaxonomy.json` (built by `build_mistake_taxonomy.py`). Rates
are % of the 60k `sf_lines` mistake corpus (chess-poc), measured post-win_drop-gate unless noted.

**Status legend:** ACTIVE = fires as-is · GATED = fires with a concept gate added after audit · DELETED =
removed, do not re-add without a fresh characterization pass · INFO = descriptive (direction=info), not a
mistake assertion.

---

## Deleted / missed-only — READ BEFORE RE-ADDING
These were removed for cause. Re-adding requires a fresh audit that beats the reason below.

| Tag | Status | Rate before | Why removed |
|---|---|---|---|
| **Allowed Battery** | DELETED (2026-07-14) | 9% | Naked-rate catch-all. Q/R/B alignment on an enemy piece is a geometric coincidence in almost any refutation line. Topped 118 SAE features / 81 Opus concepts; 0 sole-explain residue that was actually a battery. See `2026-07-14_battery_catchall_deletion.md`. |
| **Missed Battery (v1)** | DELETED → REBUILT | 4% | v1 used a broken finder (required the back piece to *directly* attack the target — impossible for a battery). Deleting the concept was wrong; **rebuilt** with correct xray geometry (see ACTIVE table). Lesson: validate a finder on a known-positive before proving absence. |
| **Allowed Castling** | DELETED (2026-07-14) | 2.3% | Letting the opponent castle is normal chess, never the teachable mistake. 78% had the opponent castle 2-4 plies deep; first-reply cases all had a concrete real error (hung piece / missed tactic) with castling as routine reply. `tagger._MISSED_ONLY_MOTIFS`. |
| **Ignored Threat** | DELETED (2026-07-14) | 3.6% | Vaguely-defined synonym for the real hierarchy Hung <Piece> → Allowed Hanging Piece. 92% of fires co-fired a sharper material/hang/mate tag that named it better; where it was the sole material signal (162 cases) only 19% actually lost the "threatened" piece — crude is_attacked_by/undefended flagged phantom threats. Redundant when right, wrong when unique. Deleting orphaned 1 corpus position. (Was never in the taxonomy → rendered neutral in prod anyway.) |
| **Missed Desperado** | DELETED (2026-07-15) | 3.1% | Fired when the best move was a capture BY an attacked piece ("cash in a doomed piece"). 90% had best-capture SEE>=0 = a plain WINNING capture, not a salvage — mislabeled. 78% co-fired a sharper tag (Missed Free X / Missed Sacrifice / Mate / Hung); only 15% sole-explain. The genuine SEE<0 salvage cases (~9%) overlap Missed Sacrifice, which names them right. Redundant when right, mislabel when SEE>=0. (Never in taxonomy → neutral in prod.) |
| **Failed Pin / Failed Fork / Failed Discovered Attack** | DELETED (2026-07-14) | 2.2% / 1.0% / 2.7% | The whole FAILED direction (`FAILED_OK=set()`). `detect_move` saw the played move's PATTERN, not whether the player attempted the tactic or it backfired because of it. Only 20-27% of fires had the moved piece even recaptured; in the most-favorable subset (capture + recapture) 69% co-fired a material tag that named the real loss, 7/324 sole-explain. Geometry ≠ intent+causation. Deleting left 4% untagged (whose only tag was a false "you failed a tactic" — worse than silence). |
| (5 outcome catch-alls) | DELETED (GH #29) | — | "Bad/Wrong Capture" etc. were 86-100% co-fire duplicates that mislabeled missed tactics. Replaced by `greedy_capture` (the one real idea mined from them). |

---

## Recently gated (audit outcomes — the gate has a chess reason, never a rate cap)

| Tag | Rate (before→after) | The concept gate |
|---|---|---|
| **Missed Castling** | 3.5% → 0.94% | Fire only if the best MOVE is castling (first pov move of the line), not if castling appears later. `motifs.castling_line`. |
| **Missed Open File** | 6.4% → 4.9% | Best move must be a QUIET rook move (not a capture, not a check) — a capture/check that lands on an open file is a tactic, file incidental. `predicates.missed_open_file`. |
| **Missed Prophylaxis** | 8.2% → ~3% | Best is a quiet non-king move preventing a NON-capture, NON-check plan; excludes check-threats + king moves (those are king-safety, not prophylaxis). |
| **Missed Pawn Break** | (piece-grab bug removed) | Excludes pawn-takes-piece (a material grab, not a structural break). |
| **hung_material** | (promotion + SEE guards) | Subtract opponent-promotion gain (that's a pawn race, not a hang); SEE<0 played-capture = sacrifice not hang. |
| **Missed / Allowed Outpost** | 2.6%→1.15% / 2.3%→1.06% | Fire only if the FIRST pov move quietly establishes the outpost — not if an outpost appears later in the line (51% did) or via a capture (material move, outpost incidental). `motifs.outpost_line`. Both directions kept (allowing the opponent an unchallengeable knight IS a real concession). |
| **missed_overloading** | 9.96% → 3.26% | Best line must net ≥2 material (geometry-only overload was the over-fire). |
| **capture_or_exchange** | (defended-capture branch) | Gate the "sacrifice" exclusion on SEE, not piece values — a defended capture that nets material over the recapture sequence fires Missed Free X even if attacker outvalues victim. |

---

## Active predicates (`ALL_PREDICATES`, 68) — one line each

**Info / descriptive (INFO — direction=info, category Meta, not mistakes):**
- `phase` — Opening/Middlegame/Endgame.
- `game_state` — Winning/Losing/Equal before the move.
- `conversion_outcome` — result-band transition (Winning→Losing, Winning→Drawn, Even→Losing, …).
- `blunder_severity` — Sharp Blunder (win-drop≥30%) vs Slow Bleed (<15% from balanced), saturation-guarded.
- `move_difficulty` — Only Good Move Missed (n_good_moves≤1) vs Careless Blunder (≥4).

**Material:**
- `capture_or_exchange` — best is a capture → Missed Free X (undefended or SEE≥1 win) / Missed X Exchange (SEE 0).
- `greedy_capture` (6%) — you grabbed material (SEE≥0) when the best move was quiet.
- `hung_material` (Hung Rook/Queen/Knight/Bishop/Material, 5-6.5%) — net material lost over the refutation line. Legit-loud; edge #58 (peak-victim even-trade naming) open.
- `allowed_pawn_capture` (11%) — quiet move lets the opponent grab a pawn the best move held. Loudest tag, audited SOLID.

**Checks / sacrifices (the check family):**
- `pointless_check` — you checked; a quiet move was better.
- `missed_attacking_check` (4.8%) — you didn't check; a forcing quiet check was best.
- `wrong_check` (1.4%) — you checked, but a DIFFERENT check was stronger. (Added 2026-07-14; the third case.)
- `unsound_sacrifice` — played a SEE<0 capture at the enemy king with no compensation.
- `missed_greek_gift` — missed a sound bishop sac (Bxh7+/Bxf7+) next to the castled king.
- `missed_zwischenzug` — an in-between check should precede your recapture.
- `recapture_exposes_king` — your pawn recapture opened a line onto your own king.

**Battery / overloading / rooks (rebuilt/gated):**
- `missed_battery` (1.0%) — best QUIETLY builds a stacked xray battery (front+back on one line) on a DEFENDED target. Rebuilt 2026-07-14. See ledger deleted-table for v1.
- `missed_overloading` / `allowed_overloading` — enemy/your piece overloaded (best line nets ≥2).
- `missed_doubled_rooks` / `allowed_doubled_rooks` — doubling rooks on a file.

**King safety (family):**
- `king_in_center`, `lost_castling`, `exposed_king_pawn`, `pawn_structure` (doubled/isolated/backward), `backward_pawn`.

**Defensive draw resources (swindles, from a losing position):**
- `missed_perpetual` — best move starts a perpetual check to force a draw. Endgame category.
- `missed_stalemate` (~0% corpus) — best move forces stalemate (a draw from a lost position). Added 2026-07-14 (Sam's "#68" pick). EXACT board condition (`is_stalemate`), so zero over-fire risk; ~0 corpus fires because forced-stalemate saves are an endgame-swindle phenomenon rare in a middlegame-blunder corpus. Built for correctness/coverage, not frequency; regression uses a synthetic anchor.

**Endgame technique (position-gated family):**
- `missed_king_activity`, `lost_opposition`, `missed_passed_pawn`, `rook_behind_passer`, `rook_to_seventh`,
  `rook_cut_off_king`, `missed_active_rook`, `rook_endgame_blockade`, `missed_connected_passers`,
  `missed_protected_passer`, `missed_square_rule`, `missed_breakthrough`, `wrong_king_direction`,
  `outside_passer`, `rook_to_open_file_endgame`, `push_to_promote`, `missed_perpetual`, `endgame_type`.
- `bad_simplification` / `trade_to_simplify` — traded into a worse/better endgame.

**Plan-execution / positional:**
- `missed_pawn_break`, `missed_tempo_push`, `missed_open_file`, `premature_trade`, `missed_prophylaxis`,
  `missed_piece_activation`, `wrong_pawn_race`, `missed_outpost` (via motif).
- `missed_{bishop,knight,minor,queen,minor_rook}_activity` — a passive piece could be repositioned.

**Calculation / judgment:**
- `pawn_grab_undeveloped`, `premature_attack`, `missed_defensive_resource`, `missed_faster_mate`,
  `missed_pin_exploitation`, `missed_unpinning_resource`, `missed_interposition`, `missed_remove_the_guard`.
- `missed_desperado` — **DELETED 2026-07-15** (see deleted table). NOTE: a shallower 2026-07-14 pass had
  marked this LEAVE (the piece really is doomed 89% of the time). The reversal came from auditing the BEST
  CAPTURE's SEE, not the doomed-ness: 90% of fires have best-capture SEE>=0 = a plain winning capture (not a
  salvage), and 78% co-fire a sharper tag. "Piece is doomed" ≠ "the lesson is desperado" — when the best
  move wins material cleanly it's Missed Free X; when it's a real SEE<0 sac it's Missed Sacrifice. Lesson:
  audit the tag against the SHARPER tag that already names the move, not just its own internal validity.

## Motif detectors (`LINE_DETECTORS`, 25 → Missed/Allowed twins)
`fork, hangingPiece, sacrifice, xRayAttack, discoveredAttack, doubleCheck, trappedPiece, attraction,
deflection, intermezzo, interference, skewer, pin, capturingDefender, exposedKing, attackingF2F7,
kingsideAttack, queensideAttack, clearance, advancedPawn, enPassant, castling*, promotion, underPromotion,
outpost`. Each fires in a direction: MISSED (best line, pov=mover), ALLOWED ([played]+refutation, pov=opp).
The FAILED direction (`Failed X` = played move that geometrically made a tactic) is DELETED
(`FAILED_OK=set()`, 2026-07-14 — it was a geometry catch-all; see deleted table). `castling` is MISSED-ONLY.
`exposedKing` uses explicit labels (Enemy King Exposed / Exposed King). Mate outranks lesser motifs in the
same direction (`_suppress_lesser_under_mate`).

**Audit queue (mid-rate 2-4%, not yet examined):** Ignored Threat (3.6%), Missed Desperado (3.1%),
Failed Discovered Attack (2.7%), Failed Pin (2.2%), Missed/Allowed Outpost (2.6%/2.3%). Same method:
read FENs, check catch-all signature (SAE concepts topped), gate on the concept or delete if no residue.

## Method references
- How to audit: `sae_feature_audit_playbook.md` (delete / gate / leave; the 9 lessons).
- Worked examples: `2026-07-14_battery_catchall_deletion.md`, `2026-07-14_missed_forcing_move_gap.md`.
