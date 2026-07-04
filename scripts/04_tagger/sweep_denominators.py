#!/usr/bin/env python3
"""Compute per-band DENOMINATORS for the rating sweep (no Stockfish): fetch the 12k distinct game
PGNs from lichess in bulk, count per band: games, player-moves (both players err, so all moves),
and moves-by-phase. Phase via piece count, matching the tagger's phase(): Opening (fullmove<=12 &
>=24 pieces), Endgame (<=12 pieces or <=4 non-pawn), else Middlegame.
Output: sweep_denominators.json {band: {games, moves, opening_moves, middle_moves, endgame_moves}}.
"""
import json, io, time, urllib.request
import chess, chess.pgn
from collections import defaultdict

SWEEP="/home/ec2-user/SageMaker/sweep_blunders_2000.json"
OUT="/home/ec2-user/SageMaker/sweep_denominators.json"
BATCH=200

s=json.load(open(SWEEP))
# band -> set of game short-ids
band_ids=defaultdict(set); id_band={}
for r in s:
    gid=r['game_id'].rstrip('/').split('/')[-1]
    band_ids[r['band']].add(gid); id_band[gid]=r['band']
all_ids=list(id_band)
print(f"{len(all_ids)} distinct games across {len(band_ids)} bands", flush=True)

def phase_of(board):
    n=len(board.piece_map())
    nonpawn=sum(1 for p in board.piece_map().values() if p.piece_type not in (chess.PAWN,chess.KING))
    if board.fullmove_number<=12 and n>=24: return 'opening'
    if n<=12 or nonpawn<=4: return 'endgame'
    return 'middle'

den=defaultdict(lambda: {'games':0,'moves':0,'opening':0,'middle':0,'endgame':0})
t0=time.time(); fetched=0
for i in range(0,len(all_ids),BATCH):
    batch=all_ids[i:i+BATCH]
    try:
        req=urllib.request.Request('https://lichess.org/api/games/export/_ids',
            data=','.join(batch).encode(), headers={'User-Agent':'chess-deck-research denominators'})
        txt=urllib.request.urlopen(req,timeout=60).read().decode()
    except Exception as e:
        print(f"  batch {i} err: {e}", flush=True); time.sleep(5); continue
    stream=io.StringIO(txt)
    while True:
        g=chess.pgn.read_game(stream)
        if g is None: break
        gid=g.headers.get('GameId') or g.headers.get('Site','').rstrip('/').split('/')[-1]
        band=id_band.get(gid)
        if not band: continue
        d=den[band]; d['games']+=1
        board=g.board()
        for mv in g.mainline_moves():
            d['moves']+=1; d[phase_of(board)]+=1
            board.push(mv)
        fetched+=1
    print(f"  {i+len(batch)}/{len(all_ids)} requested | {fetched} parsed | {(time.time()-t0)/60:.1f}min", flush=True)
    time.sleep(1)  # be polite to lichess

json.dump({b:dict(v) for b,v in den.items()}, open(OUT,'w'), indent=2)
print(f"\nDONE {fetched} games -> {OUT}", flush=True)
for b in sorted(den):
    d=den[b]; print(f"  {b}: {d['games']} games, {d['moves']} moves (O{d['opening']}/M{d['middle']}/E{d['endgame']})", flush=True)
