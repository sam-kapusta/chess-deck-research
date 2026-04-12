#!/usr/bin/env python3
"""Fetch analyzed games from DynamoDB and save to local cache.

Table: chess-coach-analysis (PK: gameOriginId)
Each record has: evaluations, moments, all_moves, pgn, etc.

Usage:
  python fetch_games_from_dynamo.py
  python fetch_games_from_dynamo.py --limit 100  # test with 100 games
"""
import json
import boto3
import argparse
from pathlib import Path
from decimal import Decimal

CACHE_DIR = Path.home() / ".chess-coach" / "games"
TABLE_NAME = "chess-coach-analysis"
REGION = "us-east-1"


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def fetch_all_games(limit=None):
    """Scan DynamoDB for all analyzed games."""
    session = boto3.Session(profile_name='chess-deck')
    dynamodb = session.resource('dynamodb', region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)

    games = []
    scan_kwargs = {}
    total = 0

    while True:
        response = table.scan(**scan_kwargs)
        items = response.get('Items', [])
        games.extend(items)
        total += len(items)

        if total % 500 == 0:
            print(f"  Scanned {total} games...")

        if limit and total >= limit:
            games = games[:limit]
            break

        if 'LastEvaluatedKey' not in response:
            break
        scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']

    return games


def save_game(game, cache_dir):
    """Save a game to local cache if not already there."""
    game_id = game.get('gameOriginId', game.get('game_id', ''))
    if not game_id:
        return False

    filepath = cache_dir / f"{game_id}.json"
    if filepath.exists():
        return False  # Already cached

    filepath.write_text(json.dumps(game, cls=DecimalEncoder, indent=None))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--cache-dir', default=str(CACHE_DIR))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(cache_dir.glob("chesscom-*.json")))
    print(f"Existing local games: {existing}")
    print(f"Fetching from DynamoDB ({TABLE_NAME})...")

    games = fetch_all_games(limit=args.limit)
    print(f"Fetched {len(games)} games from DynamoDB")

    new = 0
    for game in games:
        if save_game(game, cache_dir):
            new += 1

    print(f"New games saved: {new}")
    print(f"Total local games: {existing + new}")


if __name__ == "__main__":
    main()
