#!/usr/bin/env python3
"""Per-position skill tags for the Gauntlet corpus (issue #69 — per-skill filtering).

The FIFA pipeline already tags every sweep position (fifa_skill_ratings.py) but only emits
per-BAND aggregate counts. The Gauntlet needs the tags PER POSITION so "Drill Defensive Tactics"
can filter the band pool to positions that exercise that skill.

This reuses fifa_skill_ratings.py's exact tagging (_tag_one logic) but returns, per position,
the set of 6-groups AND the specific labels it fires.

OUTPUT = SIDECAR FILES, one per band: fifa_maia/skills_<band>.json
  { "<fen>|<uci>": { "groups": [...], "labels": [...] }, ... }
Tags are VOLATILE (every tagger audit changes them — 15+ detector fixes and counting) while the
Maia/SF enrichment is STABLE (hours of compute, never invalidated by tag fixes). Keeping them in
separate files means a tagger fix re-generates ~3MB/band sidecars in minutes and the 20MB
enrichment files are never touched. The frontend fetches both and joins by fen|uci.

Group + label vocab matches the frontend (fifaSkillRatings.json clusters) so filtering lines up.

Tagger-fix workflow:
  1. sync tagger modules:  for f in predicates.py motifs.py tagger.py mistake.py chesslib_util.py;
     do sais -n chess-poc write tagger_run/$f $f; done  (from scripts/04_tagger/)
  2. re-tag (~5 min):      sais term 'screen -dmS tag_gauntlet bash -c "cd ~/SageMaker &&
     python3 fifa_pipeline/tag_gauntlet_positions.py > tag_gauntlet.log 2>&1"'
  3. download + upload the 11 skills_*.json to
     s3://chess-coach-research-data/drill-positions/gauntlet/ (chess-deck profile)
"""
import glob
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import chess

sys.path.insert(0, "/home/ec2-user/SageMaker/tagger_run")
from mistake import Mistake  # noqa: E402
import tagger as T  # noqa: E402

BASE = os.path.expanduser("~/SageMaker")
ENRICH = os.path.join(BASE, "fifa_enrich.json")
SWEEP = os.path.join(BASE, "fifa_sweep.json")
MAIA_DIR = os.path.join(BASE, "fifa_maia")

# --- tagging helpers (verbatim from fifa_skill_ratings.py so groups match exactly) ---


def line_to_sans(line):
    out = []
    for tok in (line or "").replace("...", ". ").split():
        if tok.replace(".", "").isdigit() or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        out.append(tok)
    return out


def eval_to_cp(s):
    if s is None:
        return None
    s = str(s)
    if "#" in s or "M" in s.upper():
        return -10000 if "-" in s else 10000
    try:
        return int(s)
    except ValueError:
        return None


def to_group(cat, label):
    l = label.lower()
    if cat in ("Missed Tactic", "Missed Mate"):
        return "Offensive Tactics"
    if cat == "Allowed Tactic":
        return "Defensive Tactics"
    if cat == "Calculation":
        return "Calculation"
    if cat in ("Hung Piece", "Missed Capture"):
        return "Piece Safety"
    if cat in ("Position", "Trading"):
        return "Positional"
    if cat == "Endgame":
        return "Endgame"
    if cat == "King Safety":
        if "mate" in l or "attack" in l:
            return "Defensive Tactics" if l.startswith("allowed") else "Offensive Tactics"
        return "Positional"
    return None


_enrich = None  # loaded in each worker (fork inherits parent's copy)


def _tag_one(row):
    """(fen|uci key, groups[], labels[]) for one position, or None if untaggable."""
    key = f'{row["fen"]}|{row["uci"]}'
    ce = _enrich.get(key)
    if not ce:
        return None
    try:
        fen = row["fen"]
        b = chess.Board(fen)
        mover = b.turn
        bl = line_to_sans(ce.get("top_3_best", [{}])[0].get("line", "")) if ce.get("top_3_best") else []
        bu = ""
        if bl:
            try:
                bu = b.parse_san(bl[0]).uci()
            except ValueError:
                pass
        refut = line_to_sans(ce.get("top_3_refutations", [{}])[0].get("line", "")) if ce.get("top_3_refutations") else []
        eb = eval_to_cp(ce.get("eval_before"))
        ea = eval_to_cp(ce.get("eval_after"))
        m = Mistake(fen_before=fen, played_uci=row["uci"], best_uci=bu, best_line_san=bl,
                    refutation_san=refut, eval_before=eb, eval_after=ea,
                    cp_loss=int(ce.get("cp_loss", 0) or 0), mover=mover,
                    played_san=ce.get("played_san", ""), best_san=ce.get("best_san", ""))
        groups = set()
        labels = set()
        for t in T.tag_mistake_full(m, with_maia=False)["tags"]:
            if t.get("direction") == "info":
                continue  # context/orient tags aren't skills (matches frontend filter)
            lab = t["label"]
            g = to_group(T.categorize(lab, t.get("direction")), lab)
            if g:
                groups.add(g)
                labels.add(lab)
        return (key, sorted(groups), sorted(labels))
    except Exception:
        return None


def _init():
    global _enrich
    _enrich = json.load(open(ENRICH))


def main():
    sweep = json.load(open(SWEEP))
    print(f"tagging {len(sweep)} positions across {os.cpu_count()} cores…", flush=True)

    tags_by_key = {}
    t0 = time.time()
    n = 0
    grp_counter = Counter()
    with Pool(40, initializer=_init) as pool:
        for res in pool.imap_unordered(_tag_one, sweep, chunksize=200):
            n += 1
            if n % 20000 == 0:
                print(f"  {n}/{len(sweep)} ({time.time()-t0:.0f}s)", flush=True)
            if res is None:
                continue
            key, groups, labels = res
            tags_by_key[key] = {"groups": groups, "labels": labels}
            for g in groups:
                grp_counter[g] += 1

    print(f"tagged {len(tags_by_key)} positions in {time.time()-t0:.0f}s", flush=True)
    print("group coverage:", dict(grp_counter), flush=True)

    # Write SIDECAR files per band (NOT merged into the enrichment JSONs — they're stable and
    # never re-generated, while tags change with every tagger audit). Frontend joins by key.
    key_to_band = {f'{r["fen"]}|{r["uci"]}': r["band"] for r in sweep}
    by_band: dict[str, dict] = {}
    for key, skills in tags_by_key.items():
        band = key_to_band.get(key)
        if band:
            by_band.setdefault(band, {})[key] = skills

    for band in sorted(by_band):
        out_path = os.path.join(MAIA_DIR, f"skills_{band}.json")
        json.dump(by_band[band], open(out_path, "w"))
        print(f"  {band}: {len(by_band[band])} positions → {out_path}", flush=True)

    print("DONE — sidecar skills files written. Upload skills_*.json to S3.", flush=True)


if __name__ == "__main__":
    main()
