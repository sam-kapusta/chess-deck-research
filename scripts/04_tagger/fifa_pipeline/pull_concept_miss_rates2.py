"""Round 2: eligible-denominator miss rates for the UNTESTED thin-cluster candidates.

Same method as pull_concept_miss_rates.py (miss/eligible by band — does the weaker player err more
WHEN the chance exists). Tests candidates I asserted-but-never-measured for the thin clusters:
  Minor:      KnightActivity, KingActivity-in-minor
  Queen:      KingActivity-in-queen, AllowedPerpetual(defensive flip)
  RookMinor:  MinorActivity (the bishop/knight half)
Run on chess-poc with HF_HOME=/tmp/hfcache.
"""
import os, re, io, json, time
import chess, chess.pgn
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from collections import defaultdict

REPO = "Lichess/standard-chess-games"
BANDS = [('600-800',600,800),('800-1000',800,1000),('1000-1200',1000,1200),('1200-1400',1200,1400),
         ('1400-1600',1400,1600),('1600-1800',1600,1800),('1800-2000',1800,2000),('2000-2200',2000,2200),
         ('2200-2400',2200,2400),('2400-2600',2400,2600),('2600-2800',2600,2800)]
ELIGIBLE_TARGET = 1500
MIN_LOSS = 200
GOOD_LOSS = 30
MAX_SHARDS = 10
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]

def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''))
    if not m: return False
    b, i = int(m.group(1)), int(m.group(2)); return 480 <= b + 40*i < 1500

def band_of(e):
    for n, lo, hi in BANDS:
        if lo <= e < hi: return n
    return None

def is_endgame(board):
    pm = board.piece_map()
    if len(pm) <= 12: return True
    return sum(1 for p in pm.values() if p.piece_type not in (chess.PAWN, chess.KING)) <= 4

def material(board):
    nq=nr=nb=nn=0
    for p in board.piece_map().values():
        t=p.piece_type
        if t==chess.QUEEN: nq+=1
        elif t==chess.ROOK: nr+=1
        elif t==chess.BISHOP: nb+=1
        elif t==chess.KNIGHT: nn+=1
    minors=nb+nn
    if nq==0 and nr==0 and minors==0: return "Pawn"
    if nq==0 and nr>0 and minors==0: return "Rook"
    if nq>0 and nr==0 and minors==0: return "Queen"
    if nq==0 and nr==0 and minors>0: return "Minor"
    if nq==0 and nr>0 and minors>0: return "RookMinor"
    if nq>0 and (nr>0 or minors>0): return "Heavy"
    return "Other"

def chebyshev_center(sq):
    return min(chess.square_distance(sq, c) for c in CENTER)

def piece_activation_available(board, color, ptype):
    if board.turn != color: return False
    for sq, p in board.piece_map().items():
        if p.piece_type != ptype or p.color != color: continue
        before=len(board.attacks(sq))
        for mv in board.legal_moves:
            if mv.from_square != sq: continue
            t=board.copy(stack=False); t.push(mv)
            if len(t.attacks(mv.to_square))-before>=4: return True
    return False

def piece_activation_taken(board, mv, ptype):
    if board.piece_type_at(mv.from_square)!=ptype: return False
    before=len(board.attacks(mv.from_square))
    t=board.copy(stack=False); t.push(mv)
    return len(t.attacks(mv.to_square))-before>=4

def king_center_available(board, color):
    """A legal king move that steps toward the center exists (opportunity to activate the king)."""
    if board.turn != color: return False
    k=board.king(color)
    if k is None: return False
    cd=chebyshev_center(k)
    for mv in board.legal_moves:
        if mv.from_square==k and chebyshev_center(mv.to_square)<cd:
            return True
    return False

def king_center_taken(board, mv, color):
    k=board.king(color)
    return k is not None and mv.from_square==k and chebyshev_center(mv.to_square)<chebyshev_center(k)

# concept -> (material_set, available, taken)  — material_set None = any endgame
CONCEPTS = {
    "KnightActivity_Minor":   ({"Minor"}, lambda b,c: piece_activation_available(b,c,chess.KNIGHT),
                                          lambda b,m: piece_activation_taken(b,m,chess.KNIGHT)),
    "KingActivity_Minor":     ({"Minor"}, king_center_available, lambda b,m: king_center_taken(b,m,b.turn)),
    "KingActivity_Queen":     ({"Queen"}, king_center_available, lambda b,m: king_center_taken(b,m,b.turn)),
    "KingActivity_Rook":      ({"Rook"},  king_center_available, lambda b,m: king_center_taken(b,m,b.turn)),
    "MinorActivity_RookMinor":({"RookMinor"},
        lambda b,c: piece_activation_available(b,c,chess.BISHOP) or piece_activation_available(b,c,chess.KNIGHT),
        lambda b,m: piece_activation_taken(b,m,chess.BISHOP) or piece_activation_taken(b,m,chess.KNIGHT)),
    # Queen mistakes in ANY queen endgame (Q+P 'Queen' AND Q+pieces 'Heavy') — "queen to a better
    # square / centralization" missed. Sam: queen-endgame errors aren't only Q+P.
    "QueenActivity_QueenOrHeavy":({"Queen","Heavy"},
        lambda b,c: piece_activation_available(b,c,chess.QUEEN),
        lambda b,m: piece_activation_taken(b,m,chess.QUEEN)),
}

def pe(c):
    m = re.search(r'\[%eval\s+([#\-\d.]+)\]', c or '')
    if not m: return None
    s=m.group(1)
    if '#' in s: return -10000 if '-' in s else 10000
    try: return int(round(float(s)*100))
    except: return None

files=[f for f in list_repo_files(REPO, repo_type="dataset")
       if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$', f)]
files.sort(reverse=True)
stats={c: defaultdict(lambda:[0,0,0]) for c in CONCEPTS}  # [eligible, miss, took_good]
elig={c:{b:0 for b,_,_ in BANDS} for c in CONCEPTS}
t0=time.time(); ng=0
def done(): return all(all(elig[c][b]>=ELIGIBLE_TARGET for b,_,_ in BANDS) for c in CONCEPTS)

for fi,f in enumerate(files):
    if fi>=MAX_SHARDS or done(): break
    try:
        p=hf_hub_download(REPO,f,repo_type="dataset")
        t=pq.read_table(p,columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e:
        print('skip',str(e)[:50],flush=True); continue
    for we,be,tc,mt in zip(*[t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext']]):
        ng+=1
        if not we or not be or not is_rapid(tc) or '%eval' not in mt: continue
        if band_of(we) is None and band_of(be) is None: continue
        try: g=chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{mt}'))
        except Exception: continue
        if not g: continue
        board=g.board(); prev=None
        for node in g.mainline():
            mover=board.turn; me=we if mover==chess.WHITE else be; bn=band_of(me)
            cur=pe(node.comment)
            if bn and is_endgame(board):
                mt_type=material(board)
                for cname,(mset,availf,takenf) in CONCEPTS.items():
                    if elig[cname][bn]>=ELIGIBLE_TARGET: continue
                    if mset and mt_type not in mset: continue
                    try:
                        if availf(board,mover):
                            elig[cname][bn]+=1; cell=stats[cname][bn]; cell[0]+=1
                            took=takenf(board,node.move)
                            loss=(prev-cur) if (prev is not None and cur is not None and mover==chess.WHITE) else ((cur-prev) if (prev is not None and cur is not None) else None)
                            if took and (loss is None or loss<=GOOD_LOSS): cell[2]+=1
                            elif (not took) and loss is not None and loss>=MIN_LOSS: cell[1]+=1
                    except Exception: pass
            prev=cur; board.push(node.move)
    try: os.remove(p)
    except: pass
    if fi%2==0:
        print(f'shard {fi+1} | {time.time()-t0:.0f}s | '+' '.join(f'{c.split("_")[0][:5]}:{min(elig[c].values())}' for c in CONCEPTS),flush=True)

out={c:{b:{"eligible":stats[c][b][0],"miss":stats[c][b][1],"took":stats[c][b][2]} for b in stats[c]} for c in CONCEPTS}
json.dump(out,open("concept_miss_rates2.json","w"))
print(f"=== DONE === {(time.time()-t0)/60:.1f}min",flush=True)
BO=[b for b,_,_ in BANDS]
for c in CONCEPTS:
    rates=[round(100*out[c].get(b,{}).get('miss',0)/out[c][b]['eligible'],1) if out[c].get(b,{}).get('eligible') else 0 for b in BO]
    elg=[out[c].get(b,{}).get('eligible',0) for b in BO]
    nz=[r for r in rates if r]; mono=all(nz[i]>=nz[i+1] for i in range(len(nz)-1))
    print(f"\n{c}\n  eligible:{elg}\n  miss%:{rates}\n  falls with rating: {mono}")
