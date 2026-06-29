"""Eligible-denominator band miss-rates for the 4 NEW middlegame-tactic detectors (2026-06-27):
pin exploitation, unpinning resource, interposition, remove-the-guard. NOT endgame-gated.

Same method as pull_concept_miss_rates*.py: for each rapid eval-annotated game, at every move check
if the concept's OPPORTUNITY exists (geometric/material proxy). If yes → eligible++. Then:
  - took (good): played move realized the resource AND loss<=GOOD_LOSS
  - miss (bad):  played move did NOT realize it AND loss>=MIN_LOSS
miss_rate = miss/eligible per band. Real skill signal ⇒ miss_rate falls beginner→master.

Geometry ported from predicates.py (_ray_pin_on, king-ring, castled-king). Run on chess-poc with
HF_HOME=/tmp/hfcache + HF_TOKEN. Output: tactic_miss_rates.json = {concept:{band:{eligible,miss,took}}}.
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
MAX_SHARDS = 12
RAY = {chess.ROOK:[(1,0),(-1,0),(0,1),(0,-1)], chess.BISHOP:[(1,1),(1,-1),(-1,1),(-1,-1)],
       chess.QUEEN:[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]}
KVAL = {chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:99}

def is_rapid(tc):
    m = re.match(r'(\d+)\+(\d+)', str(tc or ''));
    if not m: return False
    b,i = int(m.group(1)),int(m.group(2)); return 480 <= b+40*i < 1500
def band_of(e):
    for n,lo,hi in BANDS:
        if lo<=e<hi: return n
    return None
def pe(c):
    m=re.search(r'\[%eval\s+([#\-\d.]+)\]', c or '')
    if not m: return None
    s=m.group(1)
    if '#' in s: return -10000 if '-' in s else 10000
    try: return int(round(float(s)*100))
    except: return None

# ---- geometry (ported from predicates.py) ----
def ray_pin_on(board, color):
    """Enemy pieces of (not color) pinned by a `color` ray piece. Returns list of pinned squares."""
    enemy = not color; found=[]
    for sq,p in board.piece_map().items():
        if p.color!=color or p.piece_type not in (chess.ROOK,chess.BISHOP,chess.QUEEN): continue
        fr,ff=chess.square_rank(sq),chess.square_file(sq)
        for dr,df in RAY[p.piece_type]:
            first=None; r,f=fr+dr,ff+df
            while 0<=r<=7 and 0<=f<=7:
                s=chess.square(f,r); q=board.piece_at(s)
                if q is not None:
                    if first is None:
                        if q.color==color or q.piece_type==chess.PAWN: break
                        first=(s,q)
                    else:
                        if q.color==enemy and KVAL[q.piece_type]>KVAL[first[1].piece_type]:
                            found.append(first[0])
                        break
                r+=dr; f+=df
    return found

def king_ring(board,color):
    k=board.king(color)
    if k is None: return set()
    return {k}|{s for s in chess.SQUARES if chess.square_distance(s,k)==1}

def king_castled(board,color):
    k=board.king(color)
    if k is None: return False
    kf,kr=chess.square_file(k),chess.square_rank(k); home=0 if color==chess.WHITE else 7
    if abs(kr-home)>1 or kf not in (0,1,2,5,6,7): return False
    for f in (kf-1,kf,kf+1):
        if 0<=f<=7:
            for r in range(8):
                pp=board.piece_at(chess.square(f,r))
                if pp and pp.color==color and pp.piece_type==chess.PAWN: return True
    return False

# ---- concept availability / taken (board=pre-move, mv=played move, color=mover) ----
def pinx_avail(b,c):
    # held enemy pin exists AND a non-capturing legal move adds an attacker onto the pinned sq
    pins=ray_pin_on(b,c)
    if not pins: return False
    opp=not c
    for psq in pins:
        if len(b.attackers(c,psq))>len(b.attackers(opp,psq)): continue  # not held
        for mv in b.legal_moves:
            if b.is_capture(mv): continue
            t=b.copy(stack=False); t.push(mv)
            if psq in t.attacks(mv.to_square) and mv.to_square not in b.attackers(c,psq):
                if ray_pin_on(t,c): return True
    return False
def pinx_taken(b,mv):
    if b.is_capture(mv): return False
    c=b.turn; opp=not c
    for psq in ray_pin_on(b,c):
        if len(b.attackers(c,psq))>len(b.attackers(opp,psq)): continue
        t=b.copy(stack=False); t.push(mv)
        if psq in t.attacks(mv.to_square) and mv.to_square not in b.attackers(c,psq) and ray_pin_on(t,c):
            return True
    return False

def unpin_avail(b,c):
    return len(ray_pin_on(b, not c))>0   # one of c's pieces is pinned by an enemy ray piece
def unpin_taken(b,mv):
    c=b.turn; before=set(ray_pin_on(b,not c))
    if not before: return False
    t=b.copy(stack=False); t.push(mv)
    return len(before - set(ray_pin_on(t,not c)))>0

def interp_avail(b,c):
    if not b.is_check(): return False
    ch=list(b.checkers())
    if len(ch)!=1: return False
    if b.piece_type_at(ch[0]) not in (chess.ROOK,chess.BISHOP,chess.QUEEN): return False
    k=b.king(c)
    return k is not None and len(chess.SquareSet.between(ch[0],k))>0
def interp_taken(b,mv):
    c=b.turn; ch=list(b.checkers())
    if len(ch)!=1: return False
    k=b.king(c); btw=chess.SquareSet.between(ch[0],k)
    return mv.from_square!=k and mv.to_square in btw

def guard_avail(b,c):
    opp=not c
    if not king_castled(b,opp): return False
    ring=king_ring(b,opp)
    for mv in b.legal_moves:
        if not b.is_capture(mv): continue
        v=b.piece_at(mv.to_square)
        if v is None or v.color!=opp or v.piece_type not in (chess.KNIGHT,chess.BISHOP): continue
        if not b.is_attacked_by(opp,mv.to_square): continue       # even trade (defended)
        t=b.copy(stack=False)
        if t.is_into_check(mv): continue
        t.push(mv)
        if t.is_check(): continue                                  # not a check
        if b.attacks(mv.to_square) & chess.SquareSet(ring): return True
    return False
def guard_taken(b,mv):
    c=b.turn; opp=not c
    if not b.is_capture(mv): return False
    v=b.piece_at(mv.to_square)
    if v is None or v.color!=opp or v.piece_type not in (chess.KNIGHT,chess.BISHOP): return False
    if not b.is_attacked_by(opp,mv.to_square): return False
    if not (b.attacks(mv.to_square) & chess.SquareSet(king_ring(b,opp))): return False
    t=b.copy(stack=False); t.push(mv)
    return not t.is_check()

CONCEPTS = {
    "PinExploitation": (pinx_avail, pinx_taken),
    "Unpinning":       (unpin_avail, unpin_taken),
    "Interposition":   (interp_avail, interp_taken),
    "RemoveTheGuard":  (guard_avail, guard_taken),
}

files=[f for f in list_repo_files(REPO,repo_type="dataset")
       if re.search(r'data/year=(2025|2024|2023)/month=\d+/.*\.parquet$', f)]
files.sort(reverse=True)
stats={c:defaultdict(lambda:[0,0,0]) for c in CONCEPTS}  # [eligible,miss,took]
elig={c:{b:0 for b,_,_ in BANDS} for c in CONCEPTS}
t0=time.time()
def done(): return all(all(elig[c][b]>=ELIGIBLE_TARGET for b,_,_ in BANDS) for c in CONCEPTS)

for fi,f in enumerate(files):
    if fi>=MAX_SHARDS or done(): break
    try:
        p=hf_hub_download(REPO,f,repo_type="dataset")
        t=pq.read_table(p,columns=['WhiteElo','BlackElo','TimeControl','movetext'])
    except Exception as e:
        print('skip',str(e)[:50],flush=True); continue
    for we,be,tc,mt in zip(*[t.column(c).to_pylist() for c in ['WhiteElo','BlackElo','TimeControl','movetext']]):
        if not we or not be or not is_rapid(tc) or '%eval' not in mt: continue
        if band_of(we) is None and band_of(be) is None: continue
        try: g=chess.pgn.read_game(io.StringIO(f'[Event "?"]\n\n{mt}'))
        except Exception: continue
        if not g: continue
        board=g.board(); prev=None
        for node in g.mainline():
            mover=board.turn; me=we if mover==chess.WHITE else be; bn=band_of(me)
            cur=pe(node.comment)
            if bn:
                for cname,(availf,takenf) in CONCEPTS.items():
                    if elig[cname][bn]>=ELIGIBLE_TARGET: continue
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
    print(f'shard {fi+1} | {time.time()-t0:.0f}s | '+' '.join(f'{c[:5]}:{min(elig[c].values())}' for c in CONCEPTS),flush=True)

out={c:{b:{"eligible":stats[c][b][0],"miss":stats[c][b][1],"took":stats[c][b][2]} for b in stats[c]} for c in CONCEPTS}
json.dump(out,open("tactic_miss_rates.json","w"))
print(f"=== DONE === {(time.time()-t0)/60:.1f}min",flush=True)
BO=[b for b,_,_ in BANDS]
for c in CONCEPTS:
    rates=[round(100*out[c].get(b,{}).get('miss',0)/out[c][b]['eligible'],1) if out[c].get(b,{}).get('eligible') else 0 for b in BO]
    elg=[out[c].get(b,{}).get('eligible',0) for b in BO]
    nz=[r for r in rates if r]; mono=all(nz[i]>=nz[i+1] for i in range(len(nz)-1))
    print(f"\n{c}\n  eligible:{elg}\n  miss%:{rates}\n  falls with rating: {mono}")
