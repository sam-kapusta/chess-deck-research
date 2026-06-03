"""Precompute SEE-on-BOTH-moves aggregate stats per feature, over a LARGE sample (top-N).

The robust signature that grounds Opus's holistic read. For each live feature, take its
top-N (default 500) activating positions and compute, on BOTH the played (blunder) and the
engine best move:
  - best_wins_material : fraction where the best move wins a hanging enemy piece (SEE>0)
                         (captures, OR a check/quiet that then wins material) + median value
  - played_capture     : fraction where the played move was itself a capture
  - blunder_hangs_own  : fraction where the played move left the player's OWN piece en prise
                         (SEE>0) + median value + piece-class distribution
  - best_is_check / best_is_capture
These disambiguate "Missed Hanging Piece" (best wins, played quiet) from "Hung Own Piece"
(played hangs own). Stats over 500 positions, not 10, so concentrations are real.

Run on chess-poc:
  python compute_feature_see_stats.py --model k4 --dict 1024 --statn 500 --out see_stats_d1024_k4.json
"""
import torch, numpy as np, json, chess, argparse, torch.nn.functional as F
from collections import Counter
from multiprocessing import Pool
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--dict', type=int, default=2048)
ap.add_argument('--statn', type=int, default=500); ap.add_argument('--out', required=True)
a = ap.parse_args()
KK = int(''.join(c for c in a.model if c.isdigit()))

VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
PIECE = {chess.KNIGHT:'knight', chess.BISHOP:'bishop', chess.ROOK:'rook', chess.QUEEN:'queen', chess.PAWN:'pawn', chess.KING:'king'}
def cls(p): return 'major' if p in ('queen','rook') else 'minor' if p in ('bishop','knight') else (p or 'none')
def see(bd, t, stm):
    aa = bd.attackers(stm, t)
    if not aa: return 0
    lva = min(aa, key=lambda s: VAL.get(bd.piece_type_at(s), 99)); cv = VAL.get(bd.piece_type_at(t), 0)
    b2 = bd.copy(); b2.remove_piece_at(t); b2.set_piece_at(t, bd.piece_at(lva)); b2.remove_piece_at(lva)
    return max(0, cv - see(b2, t, not stm))
def worst_hang(board, owner):
    opp = not owner; worst = 0; wp = None
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == owner and board.is_attacked_by(opp, sq):
            l = see(board, sq, opp)
            if l > worst: worst = l; wp = p.piece_type
    return worst, (PIECE.get(wp) if wp else None)

def see_one(args):
    fen, bl, bu = args
    try:
        b = chess.Board(fen); mover = b.turn
        bm = chess.Move.from_uci(bl)
        r = {'played_cap': b.is_capture(bm), 'best_captured_piece': 'none'}
        # DESCRIPTIVE distributions (what the player did), not just composite metrics:
        mpc = b.piece_at(bm.from_square)
        r['moved_piece'] = PIECE.get(mpc.piece_type) if mpc else 'none'   # which piece the player moved
        if b.is_capture(bm):
            tp = b.piece_at(bm.to_square)
            r['captured_piece'] = PIECE.get(tp.piece_type) if tp else 'pawn'  # en passant -> pawn
        else:
            r['captured_piece'] = 'none'
        best_win = 0; best_check = False; best_cap = False; best_piece = 'none'
        if bu and len(bu) >= 4:
            mv = chess.Move.from_uci(bu); best_check = b.gives_check(mv); best_cap = b.is_capture(mv)
            bp = b.piece_at(mv.from_square); best_piece = PIECE.get(bp.piece_type) if bp else 'none'
            if best_cap:
                tgt = mv.to_square; cap = b.piece_at(tgt)
                r['best_captured_piece'] = PIECE.get(cap.piece_type) if cap else 'pawn'  # what Maia's move captures
                cv = VAL.get(cap.piece_type, 1) if cap else 1
                bb = b.copy(); bb.push(mv); best_win = max(0, cv - see(bb, tgt, not mover))
            else:
                bb = b.copy(); bb.push(mv); best_win, _ = worst_hang(bb, not mover)  # enemy hangs after our best
        r['best_win'] = best_win; r['best_check'] = best_check; r['best_cap'] = best_cap; r['best_piece'] = best_piece
        bb = b.copy(); bb.push(bm); w, wp = worst_hang(bb, mover)
        r['own_hang'] = w; r['own_piece'] = cls(wp) if wp else 'none'
        return r
    except Exception:
        return None

c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
meta = c['metadata']; craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)
sd = torch.load(BASE + f'/output/maia3_sae/btk_{a.dict}_{a.model}_nol2.pt', map_location='cpu', weights_only=False)['state_dict']
kth = []
for i in range(0, 40000, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
    kth.append(torch.topk(z, KK, 1).values[:, -1].numpy())
th = float(np.concatenate(kth).mean())
D = sd['W_enc'].shape[1]; ACT = np.zeros((N, D), np.float32)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
    ACT[i:i+z.shape[0]] = z * (z > th)
fire = (ACT > 0).mean(0); live = np.where(fire > 0)[0]
print(f"{a.model}: {len(live)} live, computing SEE-both over top-{a.statn} each", flush=True)

# gather all (feature, position) work, dedup positions across features for SEE caching
feat_idx = {}
allpos = {}
for f in live:
    order = np.argsort(-ACT[:, f])
    n = min(a.statn, int((ACT[:, f] > 0).sum()))
    idxs = order[:n]
    feat_idx[int(f)] = idxs
    for i in idxs:
        allpos[int(i)] = None
keys = list(allpos.keys())
print(f"unique positions to SEE: {len(keys)}", flush=True)
args = [(meta[i]['fen'], meta[i]['blunder_uci'], meta[i].get('best_uci','')) for i in keys]
with Pool(16) as p:
    res = p.map(see_one, args, chunksize=256)
posstat = {keys[j]: res[j] for j in range(len(keys)) if res[j] is not None}

out = {}
for f, idxs in feat_idx.items():
    rs = [posstat[int(i)] for i in idxs if int(i) in posstat]
    n = len(rs)
    if n == 0: continue
    bw = [r['best_win'] for r in rs]; ow = [r['own_hang'] for r in rs]
    pc = Counter(r['own_piece'] for r in rs if r['own_hang'] > 0)
    med = lambda L: float(sorted(L)[len(L)//2]) if L else 0
    out[f"f{f}"] = {
        'n': n, 'fire_rate': round(float(fire[f]), 4),
        'best_wins_material_pct': round(sum(v > 0 for v in bw)/n, 3),
        'best_wins_median': med([v for v in bw if v > 0]),
        'played_capture_pct': round(sum(r['played_cap'] for r in rs)/n, 3),
        'blunder_hangs_own_pct': round(sum(v > 0 for v in ow)/n, 3),
        'own_hang_median': med([v for v in ow if v > 0]),
        'own_hang_piece_dist': dict(pc.most_common()),
        'best_is_check_pct': round(sum(r['best_check'] for r in rs)/n, 3),
        'best_is_capture_pct': round(sum(r['best_cap'] for r in rs)/n, 3),
        # DESCRIPTIVE distributions over the top-N (as % of n) — "95% queen move, 50% capture, ..."
        'moved_piece_pct': {k: round(v/n, 3) for k, v in Counter(r['moved_piece'] for r in rs).most_common()},
        'captured_piece_pct': {k: round(v/n, 3) for k, v in Counter(r['captured_piece'] for r in rs).most_common()},
        'best_piece_pct': {k: round(v/n, 3) for k, v in Counter(r['best_piece'] for r in rs).most_common()},
        'best_captured_piece_pct': {k: round(v/n, 3) for k, v in Counter(r['best_captured_piece'] for r in rs).most_common()},
    }
json.dump(out, open(a.out, 'w'), indent=1)
print(f"wrote {a.out} ({len(out)} features, top-{a.statn} SEE stats)", flush=True)
