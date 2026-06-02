#!/usr/bin/env python3
"""Assemble BTK explorer data: join labels + stats + profiles + enrichment + opus
into one JSON for l7only_atlas.html. Includes stats panel fields.

Usage:
    python scripts/evaluation/assemble_btk_explorer.py \
      --labels output/feature_labels_btk_2048_k16_v2.json \
      --stats output/feature_stats_btk_2048_k16_v2.json \
      --profiles output/btk_profiles_btk_2048_k16_v2.json \
      --enr /tmp/enr.json --opus /tmp/opus.json \
      --output output/btk_explorer_k16_v2.json
"""
import argparse, json, os


def cap(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--profiles", required=True)
    p.add_argument("--enr", required=True)
    p.add_argument("--opus", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    labels = json.load(open(args.labels))
    stats = json.load(open(args.stats))
    profiles = json.load(open(args.profiles))
    enr = json.load(open(args.enr))
    opus = json.load(open(args.opus))

    out = {}
    for fid, lab in labels.items():
        if "error" in lab:
            continue
        a = lab.get("analysis", lab)  # Pass-2 wraps in "analysis"
        fs = stats.get(fid, {})
        prof = profiles.get(fid, {})

        boards = []
        for ex in prof.get("examples", [])[:10]:
            key = ex["key"]
            en = enr.get(key, {})
            an = opus.get(key, {}).get("analysis", {})
            if not isinstance(an, dict):
                an = {}
            boards.append({
                "fen": ex["fen"], "uci": ex["uci"], "act": ex["act"],
                "cp_loss": ex.get("cp_loss"),
                "san": en.get("played_san", ex["uci"]),
                "best_san": en.get("best_san"),
                "side": en.get("side"), "phase": en.get("phase"),
                "eval_before": en.get("eval_before"), "eval_after": en.get("eval_after"),
                "n_good": en.get("n_good_moves"), "punish": en.get("punish_type"),
                "best_lines": [b.get("line") for b in en.get("top_3_best", [])][:3],
                "refut_lines": [r.get("line") for r in en.get("top_3_refutations", [])][:3],
                "motif": an.get("tactical_motif"),
                "pos_desc": cap(an.get("position_description"), 320),
                "blunder_summary": cap(an.get("blunder_summary"), 360),
                "best_analysis": cap(an.get("best_moves_analysis"), 360),
                "refut_analysis": cap(an.get("refutation_analysis"), 360),
            })

        out[fid] = {
            "chip": a.get("chip"), "label": a.get("label"),
            "description": cap(a.get("description"), 500),
            "why_bad": cap(a.get("why_bad"), 400),
            "move_pattern": cap(a.get("move_pattern"), 400),
            "sub_patterns": a.get("sub_patterns", []),
            "categories": a.get("categories", []),
            "confidence": a.get("confidence", 0),
            "fire_rate": prof.get("fire_rate", 0),
            "stats": {
                "n_activating": fs.get("n_activating", 0),
                "piece_type_pct": fs.get("piece_type_pct", {}),
                "is_capture_pct": fs.get("is_capture_pct"),
                "is_check_pct": fs.get("is_check_pct"),
                "piece_left_hanging_pct": fs.get("piece_left_hanging_pct", {}),
                "best_move_piece_pct": fs.get("best_move_piece_pct", {}),
                "best_move_is_capture_pct": fs.get("best_move_is_capture_pct"),
                "side_white_pct": fs.get("side_white_pct"),
                "traj_already_losing_pct": fs.get("traj_already_losing_pct"),
                "traj_made_worse_pct": fs.get("traj_made_worse_pct"),
                "traj_threw_winning_pct": fs.get("traj_threw_winning_pct"),
                "cp_loss_p50": fs.get("cp_loss_p50"),
                "cp_loss_p90": fs.get("cp_loss_p90"),
                "motif_hist": fs.get("motif_hist", {}),
                "motif_coverage_pct": fs.get("motif_coverage_pct", 0),
                "phase_hist": fs.get("phase_hist", {}),
            },
            "boards": boards,
        }

    json.dump(out, open(args.output, "w"), separators=(",", ":"))
    print(f"Written {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB, {len(out)} features)")


if __name__ == "__main__":
    main()
