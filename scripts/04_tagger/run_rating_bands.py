#!/usr/bin/env python3
"""Run the tagger over the rating-band sweep (joined with the Stockfish enrichment cache),
aggregate the 10 drill-category counts PER BAND. Output: rating_band_tag_stats.json.

Join: sweep_blunders_2000.json (band/fen/uci) × position_enrichment_cache.json (best/refutation lines).
The cache stores lines as SAN with move numbers; we strip the numbers and parse to the fields the
tagger's Mistake expects.
"""
import json, re, sys, time
import chess
sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
from mistake import Mistake
import tagger as T

SWEEP = "/home/ec2-user/SageMaker/sweep_blunders_2000.json"
CACHE = "/home/ec2-user/SageMaker/position_enrichment_cache.json"
OUT   = "/home/ec2-user/SageMaker/rating_band_tag_stats.json"

DRILL = ["Hung Piece","Missed Capture","Missed Tactic","Missed Mate","Allowed Tactic",
         "Calculation","Trading","Position","King Safety","Endgame"]

_num = re.compile(r"^\d+\.+$")  # "30." or "30..."

def line_to_sans(line):
    """'30. Rf3 Re6+ 31. Re3' -> ['Rf3','Re6+','Re3'] ; also handles '30... Rf1 31. Kc2'."""
    out = []
    for tok in line.replace("...", ". ").split():
        if _num.match(tok) or tok[0].isdigit():
            continue
        out.append(tok)
    return out

def eval_to_cp(s):
    """'+433' -> 433 ; '-48' -> -48 ; mate '#3' -> None."""
    if s is None: return None
    s = str(s).strip()
    if "#" in s or "M" in s.upper(): return None
    try: return int(s)
    except: return None

def build_mistake(sweep_row, ce):
    fen = sweep_row["fen"]; uci = sweep_row["uci"]
    b = chess.Board(fen)
    mover = b.turn
    best_uci = ""
    best_line_san = []
    if ce.get("top_3_best"):
        best_line_san = line_to_sans(ce["top_3_best"][0]["line"])
        if best_line_san:
            try:
                best_uci = b.parse_san(best_line_san[0]).uci()
            except Exception:
                best_uci = ""
    # refutation: SAN line from the board AFTER the played move
    refutation_san = []
    if ce.get("top_3_refutations"):
        refutation_san = line_to_sans(ce["top_3_refutations"][0]["line"])
    eb = eval_to_cp(ce.get("eval_before"))
    ea = eval_to_cp(ce.get("eval_after"))
    # cache evals are mover-POV strings; tagger wants white-POV centipawns
    if eb is not None and mover == chess.BLACK: eb = -eb
    if ea is not None and mover == chess.BLACK: ea = -ea
    return Mistake(
        fen_before=fen, played_uci=uci, best_uci=best_uci,
        best_line_san=best_line_san, refutation_san=refutation_san,
        eval_before=eb, eval_after=ea, cp_loss=int(ce.get("cp_loss", sweep_row.get("cp_loss", 0)) or 0),
        mover=mover, played_san=ce.get("played_san",""), best_san=ce.get("best_san",""),
    )

def main():
    t0 = time.time()
    sweep = json.load(open(SWEEP))
    cache = json.load(open(CACHE))
    print(f"sweep {len(sweep)}, cache {len(cache)}", flush=True)

    # per band: category -> set of position indices (a position can hit several categories);
    # also per band: label -> count (for the detailed view) and total positions tagged.
    from collections import defaultdict, Counter
    band_cat_pos = defaultdict(Counter)   # band -> {category: positions}
    band_label   = defaultdict(Counter)   # band -> {label: count}
    band_total   = Counter()              # band -> positions processed (enriched + tag-attempted)
    band_tagged  = Counter()              # band -> positions with >=1 non-info tag
    errors = 0

    for i, row in enumerate(sweep):
        key = row["fen"] + "|" + row["uci"]
        ce = cache.get(key)
        if ce is None:
            continue
        band = row["band"]
        band_total[band] += 1
        try:
            m = build_mistake(row, ce)
            res = T.tag_mistake_full(m, with_maia=False)
        except Exception:
            errors += 1
            continue
        cats_here = set()
        any_tag = False
        for tg in res["tags"]:
            if tg.get("direction") == "info":
                continue
            any_tag = True
            band_label[band][tg["label"]] += 1
            cat = T.categorize(tg["label"], tg.get("direction"))
            if cat in DRILL:
                cats_here.add(cat)
        if any_tag: band_tagged[band] += 1
        for c in cats_here:
            band_cat_pos[band][c] += 1
        # Checkpoint on ENRICHED count, not raw row index: 5000-multiple rows are usually NOT in the
        # 37% enriched set (key off sum(band_total) so progress always prints on healthy runs).
        nproc = sum(band_total.values())
        if nproc % 1000 == 0:
            print(f"  enriched {nproc} (row {i+1}/{len(sweep)}) | {(time.time()-t0)/60:.1f}min | errors {errors}", flush=True)

    # assemble output
    out = {"_bands": sorted(band_total.keys()), "_errors": errors, "bands": {}}
    for band in sorted(band_total.keys()):
        n = band_total[band]
        cats = {c: {"positions": band_cat_pos[band].get(c,0),
                    "pct": round(100*band_cat_pos[band].get(c,0)/n, 1)} for c in DRILL}
        out["bands"][band] = {
            "enriched_positions": n,
            "tagged_positions": band_tagged[band],
            "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["positions"])),
            "labels": dict(band_label[band].most_common()),  # ALL labels for this band, count-desc
        }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min, errors {errors} -> {OUT}", flush=True)
    # quick console table
    print(f"\n{'category':<16}" + "".join(f"{b[:4]:>7}" for b in out['_bands']))
    for c in DRILL:
        print(f"{c:<16}" + "".join(f"{out['bands'][b]['categories'][c]['pct']:>6.0f}%" for b in out['_bands']))

if __name__ == "__main__":
    main()
