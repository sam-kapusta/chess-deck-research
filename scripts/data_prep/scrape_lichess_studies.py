#!/usr/bin/env python3
"""Scrape popular Lichess studies and extract (FEN, commentary) training pairs.

Pipeline:
  1. Collect study IDs from /study/all/popular (40 pages × 16 = 640 studies, all 550+ likes)
  2. Export each study as PGN with comments via Lichess API
  3. Replay PGN with python-chess, extract (FEN, comment) at each annotated move
  4. Filter: English, >30 chars, not just eval/clock annotations
  5. Output: research/data/lichess_studies.jsonl

Usage:
    python scrape_lichess_studies.py                    # Full run
    python scrape_lichess_studies.py --pages 5           # First 5 pages only (test)
    python scrape_lichess_studies.py --skip-scrape        # Parse already-downloaded PGNs
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import chess
import chess.pgn
import io
import requests

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PGN_DIR = DATA_DIR / "lichess_pgns"
OUTPUT_PATH = DATA_DIR / "lichess_studies.jsonl"
STUDY_IDS_PATH = DATA_DIR / "lichess_study_ids.json"

# Lichess API
STUDY_PAGE_URL = "https://lichess.org/study/{topic}/popular?page={{page}}"
STUDY_PGN_URL = "https://lichess.org/api/study/{study_id}.pgn?comments=true&variations=true&clocks=false"
REQUEST_DELAY = 2.0  # seconds between API calls (respect rate limits)

# Topics to scrape (first few pages overlap, deeper pages diverge)
DEFAULT_TOPICS = [
    "all", "Openings", "e5", "Opening", "Opening Theory", "Tactics",
    "Endgame", "Puzzles", "e4", "d4", "d5", "Traps", "Sicilian Defense",
    "Checkmate Patterns", "London System", "Chess", "e6", "Endgames", "f5",
    "Game Study", "Checkmates", "Caro-Kann", "White", "Gambit",
    "Game Collection", "Games", "Interactive Lessons", "French Defense",
    "Opening traps", "Checkmate", "Gambits", "f4", "King's Gambit",
    "Ruy Lopez", "Black", "White Opening", "Nf6", "Strategy",
    "Caro-Kann Defense", "c5", "Beginner", "d6", "Sicilian", "Mates",
    "Traps to win your Games", "Nc6", "Black Opening", "Nc3", "c4", "Nf3",
]


# ============================================================
# Step 1: Collect study IDs
# ============================================================

def collect_study_ids(max_pages: int = 40, topics: list[str] = None) -> tuple[list[str], dict[str, int]]:
    """Scrape study IDs and like counts from Lichess popular studies pages across topics."""
    if topics is None:
        topics = ["all"]

    all_ids = []
    seen = set()
    study_likes = {}  # {study_id: like_count}

    for topic in topics:
        topic_url = f"https://lichess.org/study/topic/{topic}/popular?page={{page}}" if topic != "all" else "https://lichess.org/study/all/popular?page={page}"
        topic_new = 0

        for page in range(1, max_pages + 1):
            url = topic_url.format(page=page)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [{topic}] Page {page}: request failed ({e}), stopping topic.")
                break

            # Extract study IDs and like counts from HTML
            # Studies appear as /study/XXXXXXXX with like counts in nearby JSON
            ids = re.findall(r'/study/([A-Za-z0-9]{8})', resp.text)

            # Try to extract like counts from embedded JSON (pattern: "likes":N)
            likes_map = {}
            for match in re.finditer(r'"id"\s*:\s*"([A-Za-z0-9]{8})"[^}]*"likes"\s*:\s*(\d+)', resp.text):
                likes_map[match.group(1)] = int(match.group(2))

            unique_ids = []
            for sid in ids:
                if sid not in seen:
                    seen.add(sid)
                    unique_ids.append(sid)
                    if sid in likes_map:
                        study_likes[sid] = likes_map[sid]

            if not unique_ids:
                break

            all_ids.extend(unique_ids)
            topic_new += len(unique_ids)
            time.sleep(0.5)

        if topic_new > 0:
            print(f"  [{topic}] {topic_new} new studies (total unique: {len(all_ids)})")

    if study_likes:
        print(f"  Like counts found for {len(study_likes)} studies (min: {min(study_likes.values())}, max: {max(study_likes.values())})")

    return all_ids, study_likes


# ============================================================
# Step 2: Download PGNs
# ============================================================

def download_study_pgn(study_id: str, output_dir: Path) -> bool:
    """Download a study's PGN with comments."""
    output_file = output_dir / f"{study_id}.pgn"
    if output_file.exists() and output_file.stat().st_size > 100:
        return True  # Already downloaded

    url = STUDY_PGN_URL.format(study_id=study_id)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
    except requests.RequestException:
        return False

    # Check it's actually PGN (not HTML error page)
    if resp.text.startswith("<!DOCTYPE") or resp.text.startswith("<"):
        return False

    output_file.write_text(resp.text, encoding="utf-8")
    return True


# ============================================================
# Step 3: Parse PGN → (FEN, comment) pairs
# ============================================================

# Patterns for filtering
EVAL_PATTERN = re.compile(r'^\s*\[%(eval|clk|cal|csl)\s')
NON_ENGLISH = re.compile(r'[^\x00-\x7F]{10,}')  # 10+ consecutive non-ASCII chars


def is_text_comment(comment: str) -> bool:
    """Check if a comment is actual text (not just eval/clock/arrow annotations)."""
    comment = comment.strip()
    if not comment:
        return False

    # Remove eval/clock/arrow annotations from the comment
    cleaned = re.sub(r'\[%(eval|clk|cal|csl)[^\]]*\]', '', comment).strip()

    # Must have >30 chars of actual text after removing annotations
    if len(cleaned) < 30:
        return False

    # Skip comments that are mostly non-ASCII (likely non-English)
    if NON_ENGLISH.search(cleaned):
        return False

    return True


def clean_comment(comment: str) -> str:
    """Remove eval/clock/arrow annotations, clean whitespace."""
    cleaned = re.sub(r'\[%(eval|clk|cal|csl)[^\]]*\]', '', comment)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_pairs_from_pgn(pgn_text: str, study_id: str) -> list[dict]:
    """Extract (FEN, comment) pairs from a PGN string."""
    pairs = []
    pgn_io = io.StringIO(pgn_text)

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            break

        if game is None:
            break

        # Get metadata
        headers = dict(game.headers)
        study_name = headers.get("StudyName", "")
        chapter_name = headers.get("ChapterName", "")
        annotator = headers.get("Annotator", "")
        event = headers.get("Event", "")

        # Check for chapter-level comment (before first move)
        if game.comment and is_text_comment(game.comment):
            # This is a chapter intro — useful but no FEN/move context
            pass

        # Walk through moves
        board = game.board()
        node = game

        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            san = board.san(move)

            # Get FEN before the move
            fen_before = board.fen()

            # Apply the move
            board.push(move)

            # Check for comment on this move
            comment = next_node.comment
            if comment and is_text_comment(comment):
                cleaned = clean_comment(comment)

                # Extract eval if present
                eval_match = re.search(r'\[%eval\s+([^\]]+)\]', comment)
                eval_score = eval_match.group(1) if eval_match else None

                pairs.append({
                    "fen": fen_before,
                    "fen_after": board.fen(),
                    "move": san,
                    "comment": cleaned,
                    "eval": eval_score,
                    "study_id": study_id,
                    "study_name": study_name,
                    "chapter": chapter_name,
                    "annotator": annotator,
                    "event": event,
                    "move_number": board.fullmove_number,
                    "color": "black" if board.turn else "white",  # color that just moved
                })

            # Also check variation comments (sidelines)
            for var_idx in range(1, len(node.variations)):
                var_node = node.variation(var_idx)
                if var_node.comment and is_text_comment(var_node.comment):
                    var_cleaned = clean_comment(var_node.comment)
                    var_san = board.san(var_node.move) if var_node.move else ""

                    pairs.append({
                        "fen": fen_before,
                        "fen_after": "",
                        "move": f"({var_san})" if var_san else "",
                        "comment": var_cleaned,
                        "eval": None,
                        "study_id": study_id,
                        "study_name": study_name,
                        "chapter": chapter_name,
                        "annotator": annotator,
                        "event": event,
                        "move_number": (board.fullmove_number - 1) if board.turn else board.fullmove_number,
                        "color": "variation",
                    })

            node = next_node

    return pairs


# ============================================================
# Step 4: Dedup and quality pass
# ============================================================

def dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove duplicate (FEN, comment) pairs."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["fen"], p["comment"][:100])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape Lichess studies for training data")
    parser.add_argument("--pages", type=int, default=40, help="Number of pages per topic (default: 40)")
    parser.add_argument("--topics", action="store_true", help="Scrape all topic categories (not just /all)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip downloading, parse existing PGNs")
    parser.add_argument("--skip-download", action="store_true", help="Skip study ID collection + download")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PGN_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect study IDs
    if not args.skip_scrape and not args.skip_download:
        print(f"=== Step 1: Collecting study IDs ({args.pages} pages) ===")
        topics = DEFAULT_TOPICS if args.topics else ["all"]
        study_ids, study_likes = collect_study_ids(max_pages=args.pages, topics=topics)
        print(f"Total study IDs: {len(study_ids)}")

        # Save for resumability (include likes)
        with open(STUDY_IDS_PATH, "w") as f:
            json.dump({"ids": study_ids, "likes": study_likes}, f)
    else:
        print("=== Step 1: Loading saved study IDs ===")
        with open(STUDY_IDS_PATH) as f:
            data = json.load(f)
            if isinstance(data, list):
                study_ids = data  # Old format (just list)
                study_likes = {}
            else:
                study_ids = data["ids"]
                study_likes = data.get("likes", {})
        print(f"Loaded {len(study_ids)} study IDs")

    # Step 2: Download PGNs
    if not args.skip_scrape:
        print(f"\n=== Step 2: Downloading {len(study_ids)} study PGNs ===")
        downloaded = 0
        failed = 0
        skipped = 0

        for i, sid in enumerate(study_ids):
            pgn_file = PGN_DIR / f"{sid}.pgn"
            if pgn_file.exists() and pgn_file.stat().st_size > 100:
                skipped += 1
                continue

            success = download_study_pgn(sid, PGN_DIR)
            if success:
                downloaded += 1
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(study_ids)} (downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

            time.sleep(REQUEST_DELAY)

        print(f"Download complete: {downloaded} new, {skipped} cached, {failed} failed")
    else:
        print("\n=== Step 2: Skipped (using cached PGNs) ===")

    # Step 3: Parse PGNs
    print(f"\n=== Step 3: Parsing PGNs ===")
    all_pairs = []
    pgn_files = sorted(PGN_DIR.glob("*.pgn"))
    parse_errors = 0

    for i, pgn_file in enumerate(pgn_files):
        study_id = pgn_file.stem
        try:
            pgn_text = pgn_file.read_text(encoding="utf-8")
            pairs = extract_pairs_from_pgn(pgn_text, study_id)
            all_pairs.extend(pairs)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                print(f"  Error parsing {study_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Parsed {i+1}/{len(pgn_files)} files ({len(all_pairs)} pairs so far)")

    print(f"Parsed {len(pgn_files)} files → {len(all_pairs)} raw pairs ({parse_errors} errors)")

    # Step 4: Dedup and quality
    print(f"\n=== Step 4: Dedup + quality check ===")
    unique_pairs = dedup_pairs(all_pairs)
    print(f"After dedup: {len(unique_pairs)} pairs (removed {len(all_pairs) - len(unique_pairs)} duplicates)")

    # Add like counts to pairs
    for p in unique_pairs:
        p["likes"] = study_likes.get(p["study_id"], None)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")

    # Stats
    print(f"\n=== Stats ===")
    studies_with_pairs = len(set(p["study_id"] for p in unique_pairs))
    avg_per_study = len(unique_pairs) / max(studies_with_pairs, 1)
    comment_lengths = [len(p["comment"]) for p in unique_pairs]
    avg_len = sum(comment_lengths) / max(len(comment_lengths), 1)

    print(f"Total pairs: {len(unique_pairs)}")
    print(f"Studies with pairs: {studies_with_pairs}")
    print(f"Avg pairs per study: {avg_per_study:.1f}")
    print(f"Avg comment length: {avg_len:.0f} chars")
    print(f"Shortest comment: {min(comment_lengths)} chars")
    print(f"Longest comment: {max(comment_lengths)} chars")

    # Top annotators
    annotators = {}
    for p in unique_pairs:
        a = p["annotator"] or "unknown"
        annotators[a] = annotators.get(a, 0) + 1
    print(f"\nTop 10 annotators:")
    for a, count in sorted(annotators.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:5d} comments — {a}")

    # Sample
    print(f"\nSample pairs:")
    import random
    for p in random.sample(unique_pairs, min(3, len(unique_pairs))):
        print(f"  [{p['study_name']}] {p['move']}: {p['comment'][:120]}...")


if __name__ == "__main__":
    main()
