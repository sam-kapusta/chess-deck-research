#!/usr/bin/env python3
"""Extract a compact Opus-analysis sidecar for the atlas (run on chess-poc).

The full Opus analyses (all_positions_labeled_opus.json) are 213MB / 62,956 positions — far too big
to embed in the atlas HTML. This filters to ONLY the fen|uci pairs the atlas actually shows (the
peak + median boards across every feature, ~26K positions) and keeps the coaching fields. The atlas
fetches this lazily on feature-expand (it is gitignored — regenerable, >10MB).

Also derives a `refute_uci` per position from the Stockfish refutation_lines (first move of the top
refutation, as UCI) so the atlas can draw a BLUE arrow for the opponent's punishing reply. Only the
~19K positions in the SF cache get an arrow; all positions still get the refutation PROSE.

Run on chess-poc (inputs live there):
  cd ~/SageMaker && python3 extract_opus_sidecar.py \
    --profiles peak_median_profiles_d2048_k6.json \
    --opus all_positions_labeled_opus.json \
    --stockfish chess-stage-a/output/maia3_sae/stockfish_data_v2.json \
    --out atlas_opus_sidecar_d2048_k6.json
"""
import argparse, json, chess

ap = argparse.ArgumentParser()
ap.add_argument("--profiles", required=True)
ap.add_argument("--opus", required=True)
ap.add_argument("--stockfish", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n-boards", type=int, default=10)
a = ap.parse_args()

FIELDS = ("position_description", "move_intent", "blunder_summary",
          "best_moves_analysis", "refutation_analysis", "tactical_motif", "tags")

pm = json.load(open(a.profiles))
opus = json.load(open(a.opus))
sf = json.load(open(a.stockfish)) if a.stockfish else {}


def refute_uci(fen, sfentry):
    """First move of the top refutation line, as UCI — the line starts AFTER the blunder."""
    if not sfentry:
        return ""
    rl = sfentry.get("refutation_lines") or []
    if not rl or not rl[0].get("moves"):
        return ""
    try:
        b = chess.Board(fen); b.push_uci(sfentry["uci"])
        return b.parse_san(rl[0]["moves"][0]).uci()
    except Exception:
        return ""


keys = set()
for v in pm.values():
    for band in ("peak", "median"):
        for ex in v.get(band, [])[:a.n_boards]:
            keys.add(ex["fen"] + "|" + ex["uci"])

out, nref = {}, 0
for k in keys:
    an = opus.get(k)
    if isinstance(an, dict):
        an = an.get("analysis", an)
    rec = {f: an.get(f, "") for f in FIELDS} if isinstance(an, dict) else {}
    ru = refute_uci(k.split("|")[0], sf.get(k))
    if ru:
        rec["refute_uci"] = ru; nref += 1
    out[k] = rec

json.dump(out, open(a.out, "w"))
print(f"positions: {len(keys)} | with opus: {sum(1 for v in out.values() if v.get('blunder_summary'))} "
      f"| refute-arrow: {nref} -> {a.out}")
