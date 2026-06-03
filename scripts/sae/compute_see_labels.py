"""Compute SEE-derived ground-truth concept labels for ALL 168k positions, once.

These are the labels Sam trusts: computed from the cache metadata (FEN, blunder/best UCI,
cp_loss, is_white) + static-exchange-eval. Cached to see_labels_168k.npz so k-sparse
probing (and anything else) can reuse them without recomputing SEE.

Binary concepts produced (one bool array each, length N):
  any_hang, hang_major, hang_minor, hang_queen, hang_rook, hang_knight, hang_bishop,
  blunder_capture, best_capture, best_check, best_quiet, severe(cp>=300), endgame
Plus float cp_loss and an int phase code for reference.

Run on chess-poc from ~/SageMaker:  python compute_see_labels.py
"""
import torch, numpy as np, json, chess
from multiprocessing import Pool
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
meta = c['metadata']; N = len(meta)
VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
PIECE = {chess.KNIGHT:'knight', chess.BISHOP:'bishop', chess.ROOK:'rook', chess.QUEEN:'queen', chess.PAWN:'pawn', chess.KING:'king'}

def see(bd, t, stm):
    aa = bd.attackers(stm, t)
    if not aa: return 0
    lva = min(aa, key=lambda s: VAL.get(bd.piece_type_at(s), 99)); cv = VAL.get(bd.piece_type_at(t), 0)
    b2 = bd.copy(); b2.remove_piece_at(t); b2.set_piece_at(t, bd.piece_at(lva)); b2.remove_piece_at(lva)
    return max(0, cv - see(b2, t, not stm))

def one(m):
    # returns dict of concept -> 0/1 (and cp/phase) for a single position
    out = dict(any_hang=0, hang_major=0, hang_minor=0, hang_queen=0, hang_rook=0,
               hang_knight=0, hang_bishop=0, blunder_capture=0, best_capture=0,
               best_check=0, best_quiet=0, severe=0, endgame=0, phase=1, cp=0.0)
    try:
        b = chess.Board(m['fen']); mover = b.turn
        bm = chess.Move.from_uci(m['blunder_uci'])
        out['blunder_capture'] = int(b.is_capture(bm))
        bu = m.get('best_uci', '')
        if bu and len(bu) >= 4:
            bmv = chess.Move.from_uci(bu)
            if b.is_capture(bmv): out['best_capture'] = 1
            elif b.gives_check(bmv): out['best_check'] = 1
            else: out['best_quiet'] = 1
        b2 = b.copy(); b2.push(bm); opp = not mover; worst = 0; wp = 'none'
        for sq in chess.SQUARES:
            p = b2.piece_at(sq)
            if p and p.color == mover and b2.is_attacked_by(opp, sq):
                l = see(b2, sq, opp)
                if l > worst: worst = l; wp = PIECE.get(p.piece_type, '?')
        if wp != 'none':
            out['any_hang'] = 1
            if wp in ('queen','rook'): out['hang_major'] = 1
            if wp in ('bishop','knight'): out['hang_minor'] = 1
            out[f'hang_{wp}'] = out.get(f'hang_{wp}', 0) or 1 if wp in ('queen','rook','knight','bishop') else 0
        cp = float(m.get('cp_loss') or 0); out['cp'] = cp; out['severe'] = int(cp >= 300)
        npc = len(b.piece_map())
        out['phase'] = 0 if npc <= 14 else 2 if b.fullmove_number <= 12 else 1   # 0=end,1=mid,2=open
        out['endgame'] = int(npc <= 14)
    except Exception:
        pass
    return out

if __name__ == '__main__':
    print(f"computing SEE labels for {N} positions...", flush=True)
    with Pool(16) as p:
        res = p.map(one, meta, chunksize=512)
    keys = ['any_hang','hang_major','hang_minor','hang_queen','hang_rook','hang_knight',
            'hang_bishop','blunder_capture','best_capture','best_check','best_quiet','severe','endgame']
    arrs = {k: np.array([r[k] for r in res], dtype=np.int8) for k in keys}
    arrs['cp'] = np.array([r['cp'] for r in res], dtype=np.float32)
    arrs['phase'] = np.array([r['phase'] for r in res], dtype=np.int8)
    np.savez(B + '/see_labels_168k.npz', **arrs)
    print("base rates:")
    for k in keys:
        print(f"  {k:16s} {100*arrs[k].mean():5.1f}%")
    print(f"saved see_labels_168k.npz  (N={N})", flush=True)
