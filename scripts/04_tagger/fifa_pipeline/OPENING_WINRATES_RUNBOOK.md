# Opening band win-rates rebuild (issue #76)

Replaces the noisy opening baselines on the Openings page (band mode) with a dedicated
headers-only corpus scan. **No GPU, no Stockfish** — just streams Lichess PGN headers.

## Why this exists
The old baselines came from `pull_opening_rates.py`, a 60k-*moves*/band blunder scan → ~2,400
games/band → 5-128 games/opening/band = noise. The openings page only needs win% + game count
(never blunder rate), and win% needs only PGN headers, so this scan pulls ~100x more games.

The Drill skill card still uses `fifaSkillRatings.json` for blunder rates — **do not touch it**.
This produces a separate `openingBandRates.json`.

## Run (on chess-poc)
```bash
sais -n chess-poc term          # terminal channel; use screen for long runs
cd ~/SageMaker/<pipeline dir>
export HF_TOKEN=...              # token in ~/SageMaker/hf_download.py
# 1. Scan (tune GAME_TARGET / MAX_SHARDS in the file; default 100k games/band)
python -u pull_opening_winrates.py            # -> opening_winrates.json
# 2. Aggregate into the frontend artifact
python aggregate_opening_winrates.py opening_winrates.json openingBandRates.json
#    prints per-family band coverage — eyeball that big families clear ~1k+/band
```

## Ship back to the frontend
Copy `openingBandRates.json` to `chess-deck-code/frontend/src/data/openingBandRates.json`
(overwrites the seed). No frontend code change — `src/pages/stats/openingBandRates.ts` already
imports it. Then in chess-deck-code: `npm run tailwind:build` isn't needed (data only), but run
`npm test -- --testPathPattern=openingBandRates` and screenshot band mode before committing.

## Acceptance (#76)
- ≥1,000 games per major family per band
- Win rates stable/monotonic-ish across bands for big families
- Openings band mode shows the new counts

## Tuning notes
- `GAME_TARGET` (per band): 100k is a good default. Anchor bands (600-800, 2600-2800) are rare in
  rapid — they won't hit target; the scan logs per-band n and takes what exists.
- `MAX_SHARDS`: safety cap. Mid bands fill in a few shards; raise if the anchor bands are still thin.
- `MIN_FAMILY_GAMES` / `MIN_BAND_GAMES` (aggregator): gates on shipping a family/cell. A thin band
  is dropped (frontend renders "-") rather than shown as a fake baseline.
