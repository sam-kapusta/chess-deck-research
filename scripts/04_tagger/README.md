# Rule-Based Mistake Tagger

Deterministic, supervised mistake tagging for per-game coaching reports. Replaces the SAE for
per-position label *assignment* (the SAE was polysemantic; tags are a known vocabulary, so rules win).
The SAE's lasting value is the category VOCABULARY it discovered + a ground-truth REGRESSION SET.

## Why rules, not the SAE

Tagging a blunder with "Missed Fork" / "Hung Material" / "Allowed Mate" is *supervised classification
with a known vocabulary*, not unsupervised discovery. python-chess + Stockfish lines give exact,
explainable answers. The SAE is kept only to (a) seed the label vocabulary and (b) validate detectors
directionally (`validate_vs_sae.py`).

## Layers

| Layer | File | What | Engine? |
|-------|------|------|---------|
| L0 | `mistake.py` | `Mistake` dataclass — the one data contract (fen_before, played/best uci, best line, refutation, evals, cp_loss, mover). | none |
| L1 | `predicates.py` | Position/material predicates: phase, game-state, capture-vs-exchange (by piece), hung material (net), king safety, pawn structure, endgame type, wrong-move-order. | none |
| L2 | `motifs.py` + `chesslib_util.py` | Tactical/mate motifs ported from lichess-puzzler `cook.py`, **pov-explicit**. Single-move + line detectors + named mates. | none |
| L3 | `maia_rarity.py` | Maia rarity (numeric, not tags): played-move probability at player vs +400 Elo, rare/common blunder, skill-gap move. | Maia3 ONNX (offline) |

`tagger.py` is the orchestrator: `tag_mistake_full(m)` → `{tags, categories, maia}`.

## The three directions (the key design idea)

Every L2 motif is driven in up to three directions by choosing (board, line, pov):

| Direction | board | line | pov | Meaning |
|-----------|-------|------|-----|---------|
| **Missed X** | fen_before | best line | mover | the best line contained tactic X; the player didn't find it |
| **Allowed X** | fen_before | [played]+refutation | opponent | the played move let the OPPONENT execute X |
| **Failed X** | fen_before | played move | mover | the played move WAS tactic X but it backfired (single-move motifs only) |

## Why pov-explicit matters (the bug this whole module exists to kill)

`cook.py` hardcodes `mainline[1::2]` (the solver's moves) because a lichess puzzle ALWAYS starts with
the opponent's setup blunder. That fixed parity is correct for our **Allowed** direction (the puzzle
shape) but WRONG for **Missed** (parity flips). The original `cook_adapter.py` overrode pov and used a
parity-union hack → **Sacrifice fired at 46%**. Driven canonically it's **5%**.

The port replaces every `mainline[1::2]` → `U.pov_nodes(nodes, pov)` and `mainline[::2]` →
`U.opp_nodes(nodes, pov)`, making pov an explicit parameter. Verified fact: `node.turn()` is the color
to move AFTER the node's move, so pov's moves are `[n for n in nodes if n.turn() != pov]`. cook's
geometry is otherwise copied verbatim (it's proven correct on cook's own tests).

Same bug class found + fixed in:
- `sacrifice_line` raw `diffs[1::2]` sampled the wrong side in Missed → Missed Sacrifice 24.6% → 12%.
- `mate_in_line` used `len//2` (assumes node0=opponent) → count pov's own moves instead (parity-robust).
- `hung_material` (L1) measured GROSS loss, ignoring recaptures → over-claimed 2x → use NET material_diff → 65% → 36%.

## Validation

**Before shipping any detector change, run the gate: `python3 tag_harness.py` (see `HARNESS.md` for the
full process + decision rule).** The pieces below are what it orchestrates.

- `regression.py` — 16/16. Single-move (fork/pin/hanging from gold f54), line/mate (back-rank,
  smothered, f59 allowed-mate), mate-suppression. **Run after any detector change.**
- `validate_vs_sae.py` — directional cross-check vs Sam's hand-confirmed SAE gold
  (`relabel_v9_d64_k1.json` + `all_feat_boards_d64_k1.json`). Detectors AGREE with the SAE on its own
  confirmed positions: f54 fork/pin **87%** (0% on control), f3 hanging 67%, f17 60%, f47 mate 9/9.
  This is the "not completely wrong" gate. <100% expected (SAE is polysemantic).

## Run

```bash
# tag the whole Stockfish corpus (L1+L2, fast; --maia adds L3 ONNX, slow)
python3 scripts/04_tagger/run_corpus.py --sf /tmp/stockfish_data_v2.json --out output/mistake_tags.json

# build the evaluation atlas (HTML: every tag -> example boards + SAN lines + freq)
python3 scripts/04_tagger/build_atlas.py --tags output/mistake_tags.json --out output/tag_atlas.html

# regression + SAE directional check
python3 scripts/04_tagger/regression.py
python3 scripts/04_tagger/validate_vs_sae.py --band top
```

## Artifacts

- `output/tag_atlas.html` (224KB, committed) — the evaluation atlas. Open it directly.
- `output/mistake_tags.json` (~14MB, **gitignored** — derived/regenerable) — full per-position tags
  over the 19,362-blunder corpus. Regenerate: `run_corpus.py --sf /tmp/stockfish_data_v2.json
  --out output/mistake_tags.json` (~12 min). If you need it shared, upload to
  `s3://chess-stage-a-140023406996/sae/cache/` (research creds: `ada credentials update
  --account=140023406996 --provider=conduit --role=IibsAdminAccess-DO-NOT-DELETE --once`).

## vendor/

`cook.py`, `util.py`, `model.py`, `zugzwang.py` from lichess-puzzler (commit c188837), DO NOT EDIT —
kept as the reference source we port FROM. Our owned versions are `motifs.py` + `chesslib_util.py`.

## Open / for Sam to judge in the atlas

- **Hung Material 36%** still high — but now net-material-calibrated (cp/material ratio 0.91). The ~20%
  remaining "artifacts" are positional compensation (material down, eval not), which is arguably correct.
- **Missed Sacrifice 12% vs Allowed 5%** — best lines run longer, so more chances for a real material
  investment. Is "Missed Sacrifice" vs "Missed Combination" the crisper label? Judgment call.
- **Fork in mating lines** — mate-suppression now drops "Missed Fork" when the line is a forced mate
  (same direction). Verify in the atlas this reads right.
- **Skewer geometry** — known TODO (`_square_beyond` logic), rare (~1%), no asserted regression case.
- **backward_pawn / interference / clearance** — lower-confidence detectors; eyeball the atlas.
