# FIFA corpus rebuild — rapid-only bands, HF pull pipeline, chess-poc gotchas (2026-06-19)

> ## 🐞 BUG FOUND in the first full run (2026-06-22) — numerator/denominator sample mismatch
> The rates came out NON-MONOTONIC with the 2600-2800 band cratering to ~⅓ of trend on every group.
> Root cause: **numerator and denominator were two SEPARATE scans with different stopping points.**
> The blunder pull capped at 9000 blunders/band (2707 for the scarce 2600-2800); the denominator pull
> counted a flat 60000 moves/band. So rate = (capped blunders) / (60k moves): the 9000-bands are
> internally consistent (stayed ~comparable) but 2600-2800's numerator alone is ~3.3× under-sampled →
> its rate reads ~3× too low. Tell-tale: 9000/60000 = 15% "blunder rate" is implausible (real ≥200cp
> rate is ~5-10%), proving num/denom are from different-sized scans.
> **FIX: numerator + denominator must come from ONE scan** — count total moves AND group-fires over the
> SAME games per band (no separate denominator pass, no blunder cap that desyncs from the move count).
> Until re-run, `fifa_skill_ratings.json` rates are NOT trustworthy (esp. the top band).

Context: building the Drill-tab FIFA skill card needs per-band mistake rates anchored at true
master/beginner levels. The existing band corpus was inadequate; this rebuild fixes it. Design + plan:
`chess-deck-code/docs/plans/2026-06-19-fifa-skill-card-drill-tab.md`. Scoring design + the rate-direct
decision: `chess-deck-code/knowledge/2026-06-16-player-leak-metrics.md` (§ rate-direct, 2026-06-19).

## Why the existing band corpus had to be rebuilt (two independent flaws)
1. **Stale tagger** — `sweep_blunders_*.json` / `rating_band_tag_stats_*.json` / the shipped
   `ratingBaselines.json` were tagged PRE-#29: they contain the 5 deleted catch-alls (Bad Capture,
   Lost Material to Combination, Wrong Move Order, Wrong Capture, Captured With Wrong Piece), lack
   Greedy Capture, and predate the single entry gate. Any 6-group aggregation off them is wrong.
2. **Mixed time control** — the sweep PGN cache is **59% rapid / 37% blitz / 4% classical** (measured
   from TimeControl tags, 10,462 games). The FIFA curve is chess.com-RAPID-anchored; blitz has
   materially more blunders/move at equal rating, so the mix inflates rates and corrupts the
   rate→score anchoring. **Decision (Sam): rapid-only, re-pull ALL bands.** No partial fix — mixing a
   rapid-only new band with a mixed-TC existing band makes anchors inconsistent.

## Band scheme (Sam-locked 2026-06-19): 11 bands, 600–2800
600-800, 800-1000, 1000-1200, 1200-1400, 1400-1600, 1600-1800, 1800-2000, 2000-2200, 2200-2400,
2400-2600, 2600-2800. Both ends extended (issue #24): 2000 is NOT 99; the anchor ends are where the
"some skills graduate early, some stay bad" signal lives.

### ⚠️ BANDING MUST BE PER-MOVER, not both-players (bug I hit + fixed)
A blunder is bucketed by the **mover's** Elo (white blunder → WhiteElo band), matching the original
`redetect_sweep_d16.get_band`. My first pull required BOTH players in-band — that made 2600-2800 look
nearly empty (~5 games/shard). Per-mover banding (a 2700-vs-2400 game contributes the 2700's blunders
to 2600-2800) raised yield ~4-7×. If a high band looks empty, check this first.

### Per-band TARGET size (Sam-locked 2026-06-19)
Precision is governed by per-GROUP fire count (numerator), ~Poisson 1/√fires, NOT total blunders.
At the measured group shares (Positional 57% … Endgame 9% … of blunders): 2500 blunders/band gives the
big groups ~3% error but Calculation ~5.8% and Endgame ~6.7% — wobbly exactly where the leak signal is,
and rate error compounds through the nonlinear rate→score inversion (worst at the steep anchor ends).
**TARGET: ~9-10k blunders/band for bands ≤2200** (matches the original sweep; Endgame → ~3.5% error),
**anchor ends (2600-2800, 600-800) = max available for a fixed ~45min/band budget** (data-capped, not
a choice — log n, model+label a group's curve point if its fires fall below ~stability). 2500 was only
the fast proof cap. Cost: ~80-100k positions total through depth-16 Stockfish → multi-hour/overnight
re-detect (the long pole; scales linearly — original 56k took hours).

### 2600-2800 rapid volume — sparse but GETTABLE (quantified 2026-06-19)
Mid/low bands (≤2000) hit a 2500 target in the FIRST shard (instant). 2600-2800 rapid accumulates
~30-40 blunders/shard at ~38s/shard, RISING (denser recent months). Reaching ~2500 ≈ 60-90 shards ≈
**30-45 min for that one band**. So it's feasible, just slow — NOT "doesn't exist". Still log per-band
n; if a real run stalls below ~1.5-2k, model+label the 99-anchor rather than ship a noisy rate. NEVER
silently extrapolate.

## The data pipeline (verified 2026-06-19)
- **Source: HuggingFace `Lichess/standard-chess-games`** (parquet by year/month). Re-parameterizable by
  Elo — bands are just Elo filters. The earlier worry "high bands need a special API pull" was WRONG:
  it's a public dataset, filter `WhiteElo`/`BlackElo` + TimeControl + `[%eval]` presence.
- Recent years only (2025/2024/2023): ~8-9% of games have `[%eval]`; older years far less.
- Blunder = ≥200cp eval drop on the mover's move, from the `[%eval]` comments (mover-POV sign:
  white move = prev−cur, black move = cur−prev). Mate evals → ±10000 sentinel.
- Row schema after extract: {fen, blunder_uci, cp_loss, eval_before, eval_after, ply, is_white,
  white_elo, black_elo, band}.
- **Lichess `[%eval]` is NOT our depth-16 Stockfish** — known "depth disagreement problem". The full
  pipeline still needs `redetect_sweep_d16.py` (depth-16 MultiPV=1 Stockfish) on the pulled positions
  before tagging, exactly as the original sweep did. The HF pull only gives candidate blunders +
  Lichess evals; our eval_before/after for win_drop comes from the depth-16 re-detect.

## chess-poc operational gotchas (cost me real time — read before running notebook jobs)
- **The Jupyter KERNEL (`sais exec`) would not start** ("No active kernel sessions", restart didn't
  help). The **terminal channel (`sais term`) works fine.** For long data jobs you want `term` +
  `screen` anyway (kernel subprocesses die on cleanup), so this isn't blocking — just don't wait on
  `exec`.
- **`datasets` (HF) was not installed in ANY conda env.** Install with `--only-binary=:all:` —
  `pip install datasets` tries to build pyarrow from source and FAILS. Binary wheels work
  (datasets 4.0.0). The stack env is **`pytorch_p310`** (has python-chess 1.11.2). Minor: it downgrades
  dill/multiprocess vs pathos — harmless for streaming.
- **DON'T use `conda run -n env python ...` for long/screen jobs** — it buffers stdout and SWALLOWS
  tracebacks (a crash left a 0-byte log + no error). Use `conda activate env && python -u ...` instead.
- **HF streaming (`load_dataset(streaming=True)`) crashed** with `[Errno 9] Bad file descriptor` +
  `Fatal Python error: PyGILState_Release ... finalizing` — a GIL/threading crash in HF's HTTP RETRY
  layer, triggered by **unauthenticated rate-limiting** mid-download. Fixes: (a) set `HF_TOKEN` (a
  working token lives in `~/SageMaker/hf_download.py`); (b) prefer **per-shard `hf_hub_download` +
  `pyarrow.parquet.read_table`** over streaming — download a shard, read it, `os.remove` it (shards are
  large; 932G free on /home/ec2-user/SageMaker but clean up anyway). No mid-stream retry = no crash.
- Sweep files live in `/home/ec2-user/SageMaker/` (NOT git, NOT S3 by default). Original sweep:
  `sweep_blunders_2000.json` (56,110 rows, 6 bands). PGN cache `sweep_pgns_cache.json` (10,462 games).

## Process note (de-risking)
Running the rapid-only rebuild as **3 bands first** (800-1000, 1600-1800, 2600-2800) end-to-end before
scaling to all 11 — validates the pipeline + the 2600-2800 volume question in ~1hr before committing
hours of Stockfish re-detect. Sam's call; matches the "prove it cheap before the expensive run" habit.

## CHAIN VALIDATED on a 2-band sample (2026-06-19) — it works, and it surfaced the #26 issue live
Ran the FULL downstream chain on 2 ready bands (800-1000, 1600-1800, 2500 blunders each, rapid-only):
HF pull → **position-level d16 re-detect** (`redetect_positions_d16.py` — 5000 positions in 48s on 48
procs; analyses each blunder FEN + the post-played board, stores eval_before/after as WHITE-POV) →
**current tagger** (synced the entry-gate + Greedy-Capture tagger into `~/SageMaker/tagger_run/`, which
was STALE pre-#29) → 10-cat → 6-group → rates. Results:

| group | 800-1000 | 1600-1800 | reads correctly? |
|---|---|---|---|
| Piece Safety | 25.9% | 17.9% | ✓ falls with skill |
| Defensive Tactics | 25.7% | 24.4% | ✓ falls slightly |
| Offensive Tactics | 18.3% | 16.4% | ✓ falls slightly |
| Positional | 47.5% | 43.7% | ✓ ~flat (bloated, expected) |
| Calculation | 11.9% | 14.0% | ⚠ rises slightly |
| **Endgame** | **6.9%** | **13.2%** | ⚠ **RISES — the #26 denominator artifact, live** |

- ✅ **Zero stale catch-alls** (the synced current tagger is what ran). ✅ Greedy Capture fires (430).
- **Endgame rising with skill confirms WHY #26 is locked:** rate here = endgame-fires / ALL-blunders,
  but stronger players REACH endgames more (weaker get mated earlier), so more of their blunders are
  endgame ones — not worse endgame skill. `fifa_skill_ratings.py` MUST denominate Endgame on
  endgame-moves-reached, not total blunders/moves. The proof demonstrated the trap on real data.
- Gotcha for the aggregator: my re-detect stores evals WHITE-POV (`info.score.white()`), so do NOT
  re-apply `build_mistake`'s mover→white flip — that double-flips black and corrupts win_drop.

## 3-BAND PROOF incl. the 2600-2800 ANCHOR (2026-06-19) — chain fully validated
Full chain on 800-1000 / 1600-1800 / 2600-2800 (2500 blunders each, rapid-only). Rate = group-fires /
all-blunders-in-band (NOT the final denominator — see Endgame caveat):

| group | 800-1000 | 1600-1800 | 2600-2800 | monotonic? |
|---|---|---|---|---|
| Piece Safety | 25.5% | 18.1% | 11.6% | ✅ clean fall |
| Defensive Tactics | 25.7% | 24.0% | 16.9% | ✅ |
| Offensive Tactics | 18.5% | 16.4% | 13.9% | ✅ |
| Positional | 47.8% | 42.7% | 36.2% | ✅ |
| Calculation | 11.8% | 14.0% | 9.8% | ⚠ mid-bump (likely n-noise, ~6% err @2500) |
| **Endgame** | 7.0% | 13.2% | **28.6%** | ❌ RISES 4× — denominator artifact |

- **4/6 groups cleanly monotonic across 800→2600**, and the 2600-2800 anchor has the LOWEST rate for
  every group except Endgame → it IS a valid 99-anchor. Rate-direct anchoring will work.
- **Endgame 7→13→28.6% makes the #26 endgame-move denominator MANDATORY** — without it the card tells a
  2700 they're awful at endgames, when really ~⅓ of an elite's blunders are simply endgame ones (they
  reach endgames far more; weaker players get mated first). `fifa_skill_ratings.py` MUST divide Endgame
  fires by endgame-moves-reached per band, not all blunders.
- Calculation mid-bump: re-check at full sample; if it persists it's real (gate × cp-distribution),
  if it smooths it was noise.
- Still ZERO stale catch-alls; Greedy Capture 604 fires across 3 bands.

## Status (live)
Chain VALIDATED end-to-end on 3 bands incl. the hard anchor. Proof artifacts on chess-poc:
`proof_3band{,_enrich,_sweep}.json`, scripts `pull_3band.py` / `redetect_positions_d16.py` /
`tag_aggregate_proof.py`, current tagger synced to `~/SageMaker/tagger_run/`. READY for production:
11-band pull (~9-10k mid/low, max anchors) → overnight d16 re-detect → `fifa_skill_ratings.py`
(rate-direct, Endgame-move denom) → cabbage card → ship → UI.
