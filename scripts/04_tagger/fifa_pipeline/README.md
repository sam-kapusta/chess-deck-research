# FIFA skill-card corpus pipeline — full production runbook

Builds the per-band mistake-rate corpus that anchors the Drill-tab FIFA skill card (6 groups,
rate-direct 0-99). Design: `chess-deck-code/docs/plans/2026-06-19-fifa-skill-card-drill-tab.md`.
Findings + gotchas: `chess-deck-research/docs/2026-06-19 FIFA corpus rebuild ....md`. Scoring
philosophy: `chess-deck-code/knowledge/2026-06-16-player-leak-metrics.md` § rate-direct.

**Status (2026-06-19):** chain VALIDATED on a 3-band proof (800-1000, 1600-1800, 2600-2800). The
production run below has NOT been executed yet. Scripts here are the proof versions parameterized for
the proof bands — for production, edit `BANDS`/`TARGET` as noted.

## Runs on
chess-poc (SageMaker, account 140023406996, profile `default`). Env: conda `pytorch_p310` (has
python-chess + datasets 4.0.0). Access via `sais -n chess-poc term` + `screen` (the Jupyter KERNEL
won't start via CLI — use the terminal channel). Stockfish 16.1 at `~/SageMaker/stockfish_compiled`.
64 cores. `export HF_TOKEN=...` first (token in `~/SageMaker/hf_download.py`).

## The 4 stages

### Stage 1 — Pull blunders from HuggingFace (per band, rapid-only)
`pull_3band.py` (proof) → generalize to all 11 bands for production.
- Source: `Lichess/standard-chess-games` parquet, years 2025/2024/2023 (newest first; ~8-9% have
  `[%eval]`). Per-shard `hf_hub_download` + `pyarrow.read_table` (NOT streaming — streaming GIL-crashes).
- Filter: **rapid only** (est game length 480-1500s), `[%eval]` present, ≥200cp blunder.
- **Banding is per-MOVER** (white blunder → WhiteElo band), matching `redetect_sweep_d16.get_band`.
  NOT both-players (that made high bands look empty).
- PRODUCTION bands (set `BANDS`): 600-800, 800-1000, 1000-1200, 1200-1400, 1400-1600, 1600-1800,
  1800-2000, 2000-2200, 2200-2400, 2400-2600, 2600-2800.
- PRODUCTION target (`TARGET`): **9000-10000 for bands ≤2200** (fill in 1 shard). For the scarce
  anchor ends (2600-2800, 600-800), they WON'T reach 9k — run with a time budget (~45min/band) and
  take what exists (~2-4k). LOG per-band n. Mid/low bands hit target in the first shard (~37s);
  2600-2800 accumulates ~40-50/shard at ~40s/shard.
- Run per-band in separate screens to parallelize (each opens its own HF shards).
- Out: `proof_<band>.json` → consolidate to `fifa_blunders_all.json` {band: [{fen, blunder_uci,
  cp_loss, eval_before, eval_after, ply, is_white, white_elo, black_elo, band}]}.

### Stage 2 — Depth-16 Stockfish re-detect (OUR evals, not Lichess's)
`redetect_positions_d16.py <in.json> <out_enrich.json> <out_sweep.json> <nproc>`.
- WHY: Lichess `[%eval]` ≠ our depth-16 Stockfish (the "depth disagreement problem"). The tagger +
  win_drop need OUR eval_before/after + best move. Position-level (analyze each blunder FEN + the
  post-played board) — more efficient than whole-game re-detect, no PGNs needed.
- Stores eval_before/after as **WHITE-POV** (`info.score.white()`). ⚠ Stage 3 must NOT re-flip for black.
- Speed: ~7500 positions/min on 48 procs. Full corpus (~80-100k) ≈ 15-25 min. NOT overnight.
- Out: `fifa_enrich.json` ({fen}|{uci} → record with top_3_best, top_3_refutations, eval_before/after,
  cp_loss, played_san, best_san) + `fifa_sweep.json` (band rows).

### Stage 3 — Tag + aggregate to 6 groups + rates
`tag_aggregate_proof.py <enrich.json> <sweep.json>` (proof: prints rates). PRODUCTION: turn into
`fifa_skill_ratings.py` that WRITES the JSON artifact. Steps:
- Requires the CURRENT tagger in `~/SageMaker/tagger_run/` — SYNC it first (the copy there is stale
  pre-#29). Push: chesslib_util.py, mistake.py, predicates.py, tagger.py, build_mistake_taxonomy.py
  from `scripts/04_tagger/`.
- Build `Mistake` per enrich record (mirror `run_rating_bands.build_mistake` BUT evals are already
  white-POV — skip the mover→white flip). `tag_mistake_full(m, with_maia=False)`.
- 10-cat → 6-group map (`to_group()` in the proof script): Offensive = Missed Tactic + Missed Mate +
  Missed Attacks; Defensive = Allowed Tactic + Allowed-Mate + Allowed Attacks; Calculation = Greedy
  Capture + Failed*; Piece Safety = Hung Piece + Missed Capture; Positional = Position + Trading +
  king-structure (Castling/Exposed King/King in Center); Endgame = endgame mistakes.
- **Rate denominator (LOCKED, #26): Endgame = endgame-MOVES-reached; other 5 = total in-band moves.**
  Proof PROVED this is mandatory: Endgame rises 7→13→28.6% on the all-blunders denom (elite reach
  endgames more), would tell a 2700 they're awful at endgames. Need per-band endgame-move counts —
  either from the PGN cache (count moves where phase==Endgame) or carry a phase tag through Stage 2.
- WIN_DROP_MIN: currently 10.0 provisional (predicates.py). #29 step 5 — tune on this corpus so it
  matches prod's MISTAKE band and keeps positional tags alive. The entry gate already uses it.
- Out: `fifa_skill_ratings.json` = {bands: {band: {group: {rate, fires, n}}}, anchors: {group:
  {beginner_rate, master_rate, n}}, meta: {win_drop_min, generated, per_band_n}}.

### Stage 4 — Rate-direct scores + cabbage + ship
- Per group: score(rate) = 99*(beginner_rate − rate)/(beginner_rate − master_rate), clamp 0-99.
  beginner_rate = lowest band (600-800), master_rate = top band reached (2600-2800). PER SKILL.
- cabbage: pull his games (`pull_cabbage_games.py` / `winpct_cabbage.py` exist), tag, per-group rate,
  score against the SAME anchors. Overall plays-like Elo → the locked rating-anchored curve (overall
  only — it's non-circular).
- Ship `fifaSkillRatings.json` → `frontend/src/data/` (extend ship_tagger.py or copy).

## Validation gates (before trusting the numbers)
1. Per-group rate MONOTONIC-decreasing across bands (proof: 4/6 clean; Endgame fixed by denom;
   **Calculation had a mid-band bump at n=2500 — CONFIRM it smooths at 9-10k, else it's a real
   non-monotonic "dangerous-intermediate" hump that breaks rate→score invertibility for that group**).
2. ZERO stale catch-alls (Bad Capture / Lost Material to Combination / Wrong Move Order / Wrong
   Capture / Captured With Wrong Piece) — proves the current tagger ran.
3. Top band (2600-2800) has the lowest rate per group (valid 99-anchor) — proof: ✓ except Endgame.
4. Log + surface any band with n below ~1.5-2k (anchor ends) — model+label, never silent-extrapolate.

## Proof results (n=2500/band, all-blunders denom — Endgame NOT yet move-denominated)
| group | 800-1000 | 1600-1800 | 2600-2800 |
|---|---|---|---|
| Piece Safety | 25.5% | 18.1% | 11.6% |
| Defensive Tactics | 25.7% | 24.0% | 16.9% |
| Offensive Tactics | 18.5% | 16.4% | 13.9% |
| Positional | 47.8% | 42.7% | 36.2% |
| Calculation | 11.8% | 14.0% | 9.8% (bump) |
| Endgame | 7.0% | 13.2% | 28.6% (denom artifact) |
