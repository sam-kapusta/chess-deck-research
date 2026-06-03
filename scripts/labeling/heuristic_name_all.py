"""Heuristic naming of ALL non-dead features of a z-score SAE.

Sam's spec: for every non-dead feature, pull its top-10 activating positions, compute a
multi-granularity mistake-signature (SEE-based, same axes as deep_signature), then NAME
the feature with DETERMINISTIC heuristics — explicit rules, no LLM. The name is built only
from facts that concentrate >= THRESH across a deep sample; if nothing concentrates the
feature is 'diffuse'. This is the rule-based counterpart to the Opus constrained labeler,
so we can read what the dictionary encodes without trusting a model's prose.

Output JSON: { fid: { fire_rate, name, facts, top10:[{fen,played,best,hang,cp,eval_traj}] } }

Run on chess-poc from ~/SageMaker:
  python heuristic_name_all.py --model k4 --thresh 0.7 --sigdepth 300 --out heur_names_k4.json
"""
import torch, numpy as np, json, chess, argparse, torch.nn.functional as F
from collections import Counter
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)               # k4 / k6 / k8 ... -> btk_2048_<m>_nol2.pt
ap.add_argument('--dict', type=int, default=2048)
ap.add_argument('--thresh', type=float, default=0.7)    # concentration bar for a fact to enter the name
ap.add_argument('--sigdepth', type=int, default=300)    # positions sampled for signature stats
ap.add_argument('--firefloor', type=float, default=0.0) # 0 = all non-dead; e.g. 0.0005 to drop near-dead
ap.add_argument('--out', required=True)
a = ap.parse_args()

c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
meta = c['metadata']; craw = c['activations'].float()
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)
wp = BASE + f'/output/maia3_sae/btk_{a.dict}_{a.model}_nol2.pt'
sd = torch.load(wp, map_location='cpu', weights_only=False)['state_dict']
K = sd['W_enc'].shape[1]
try:
    enr = json.load(open(B + '/position_enrichment_cache.json'))
except Exception:
    enr = {}

VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
PIECE = {chess.KNIGHT:'knight', chess.BISHOP:'bishop', chess.ROOK:'rook', chess.QUEEN:'queen', chess.PAWN:'pawn', chess.KING:'king'}
def cls(p): return 'major piece' if p in ('queen','rook') else 'minor piece' if p in ('bishop','knight') else p
def see(bd, t, stm):
    aa = bd.attackers(stm, t)
    if not aa: return 0
    lva = min(aa, key=lambda s: VAL.get(bd.piece_type_at(s), 99)); cv = VAL.get(bd.piece_type_at(t), 0)
    b2 = bd.copy(); b2.remove_piece_at(t); b2.set_piece_at(t, bd.piece_at(lva)); b2.remove_piece_at(lva)
    return max(0, cv - see(b2, t, not stm))
def evn(s):
    if not s: return 0
    s = str(s).strip()
    if s.startswith('#'): v = int(s[1:].replace('−','-')); return (10000 - abs(v)*10) * (1 if v>=0 else -1)
    try: return int(float(s)*100)
    except: return 0

# ---- single forward pass: full activation matrix at eval threshold ----
# threshold = mean k-th-largest activation per position (the calibrated eval threshold)
KK = int(''.join(ch for ch in a.model if ch.isdigit()))   # 'k4' -> 4
kth = []
for i in range(0, 40000, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
    kth.append(torch.topk(z, KK, 1).values[:, -1].numpy())
th = float(np.concatenate(kth).mean())
ACT = np.zeros((N, sd['W_enc'].shape[1]), np.float32)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
    ACT[i:i+z.shape[0]] = z * (z > th)
fire = (ACT > 0).mean(0)
live = np.where(fire > a.firefloor)[0]
print(f"{a.model}: {len(live)} features above firefloor {a.firefloor} (of {sd['W_enc'].shape[1]}), threshold={th:.4f}")

def position_facts(idx):
    """Compute the mistake-axes for one position index. Returns dict or None."""
    m = meta[int(idx)]
    try:
        b = chess.Board(m['fen']); mover = b.turn
        bm = chess.Move.from_uci(m['blunder_uci']); pc = b.piece_at(bm.from_square)
        f = {}
        f['blunder_piece'] = PIECE.get(pc.piece_type, '?') if pc else '?'
        f['iscap'] = 'capture' if b.is_capture(bm) else 'noncapture'
        bu = m.get('best_uci', '')
        if bu and len(bu) >= 4:
            bmv = chess.Move.from_uci(bu); bpc = b.piece_at(bmv.from_square)
            bpn = PIECE.get(bpc.piece_type, '?') if bpc else '?'
            f['best_piece'] = bpn; f['best_class'] = cls(bpn)
            f['best_type'] = 'capture' if b.is_capture(bmv) else 'check' if b.gives_check(bmv) else 'quiet'
        b2 = b.copy(); b2.push(bm); opp = not mover; worst = 0; wp2 = 'none'
        for sq in chess.SQUARES:
            p = b2.piece_at(sq)
            if p and p.color == mover and b2.is_attacked_by(opp, sq):
                l = see(b2, sq, opp)
                if l > worst: worst = l; wp2 = PIECE.get(p.piece_type, '?')
        f['hang_exact'] = wp2; f['hang_class'] = cls(wp2) if wp2 != 'none' else 'none'
        f['anyhang'] = 'hangs' if wp2 != 'none' else 'safe'; f['_hangval'] = worst
        npc = len(b.piece_map())
        f['phase'] = 'endgame' if npc <= 14 else 'opening' if b.fullmove_number <= 12 else 'middlegame'
        return f, m, b, bm, bu
    except Exception:
        return None

def heuristic_name(facts):
    """DETERMINISTIC name from concentrated facts (value!=None means it cleared THRESH).
    Priority: hanging-piece motif first (tightest granularity), then blunder character,
    then the missed-move character. Mirrors the constrained labeler's rules, in pure code."""
    def val(ax): return facts.get(ax, {}).get('value')
    parts_played = []; parts_missed = []
    # --- the hang motif (what the blunder leaves en prise) ---
    if val('anyhang') == 'hangs':
        if val('hang_exact'):       hang = f"hangs the {val('hang_exact')}"
        elif val('hang_class'):     hang = f"hangs a {val('hang_class')}"
        else:                       hang = "hangs a piece"
        parts_played.append(hang)
    elif val('anyhang') == 'safe':
        pass  # explicitly non-hanging feature; characterize by other axes
    # --- blunder character ---
    if val('iscap') == 'capture':       parts_played.append("a capture")
    elif val('iscap') == 'noncapture' and not parts_played:  parts_played.append("a quiet move")
    if val('blunder_piece') and val('blunder_piece') not in ('?',):
        parts_played.append(f"with the {val('blunder_piece')}")
    # --- the missed (best) move character ---
    if val('best_type') == 'capture':   parts_missed.append("a capture")
    elif val('best_type') == 'check':   parts_missed.append("a check")
    elif val('best_type') == 'quiet':   parts_missed.append("a quiet move")
    if val('best_piece') and val('best_piece') not in ('?',):
        parts_missed.append(f"with the {val('best_piece')}")
    elif val('best_class'):
        parts_missed.append(f"with a {val('best_class')}")
    # --- phase qualifier (weak; only if present and nothing else carries it) ---
    phase = val('phase')
    # assemble
    if not parts_played and not parts_missed:
        return ('diffuse — no concentrated pattern', 'diffuse')
    played = ' '.join(parts_played) if parts_played else 'a move'
    name = f"Played {played}"
    if parts_missed:
        name += f"; missed {' '.join(parts_missed)}"
    if phase and phase != 'middlegame':   # middlegame is the base rate, don't bother
        name += f" ({phase})"
    return (name, 'named')

out = {}
for f in live:
    order = np.argsort(-ACT[:, f])
    nfire = int((ACT[:, f] > 0).sum())
    d = min(a.sigdepth, nfire)
    samp = order[:d]
    AX = {k: Counter() for k in ['blunder_piece','iscap','best_piece','best_class','best_type','hang_exact','hang_class','anyhang','phase']}
    n = 0
    top10 = []
    for rank, idx in enumerate(samp):
        pf = position_facts(idx)
        if pf is None: continue
        ff, m, b, bm, bu = pf
        for k in AX:
            if k in ff: AX[k][ff[k]] += 1
        n += 1
        if rank < 10:
            try: played_san = b.san(bm)
            except Exception: played_san = m['blunder_uci']
            best_san = ''
            if bu and len(bu) >= 4:
                try: best_san = b.san(chess.Move.from_uci(bu))
                except Exception: best_san = bu
            key = m['fen'] + '|' + m['blunder_uci']; e = enr.get(key, {})
            top10.append({'fen': m['fen'], 'played': played_san, 'best': best_san,
                          'hang': f"{ff['hang_exact']}(-{ff['_hangval']})" if ff['anyhang']=='hangs' else 'none',
                          'cp_loss': m.get('cp_loss'),
                          'eval': f"{e.get('eval_before','?')}->{e.get('eval_after','?')}" if e and 'error' not in e else '?'})
    facts = {}
    for ax, cnt in AX.items():
        if not cnt: continue
        v, k = cnt.most_common(1)[0]; pct = k / n
        facts[ax] = {'value': v, 'pct': round(pct, 2)} if pct >= a.thresh else {'value': None, 'pct': round(pct, 2), 'top': v}
    name, status = heuristic_name(facts)
    out[f"f{int(f)}"] = {'fire_rate': round(nfire / N, 4), 'n_sampled': n, 'name': name,
                         'status': status, 'facts': facts, 'top10': top10}

json.dump(out, open(a.out, 'w'), indent=1)
named = sum(1 for v in out.values() if v['status'] == 'named')
print(f"{a.model}: {len(out)} features -> {named} named, {len(out)-named} diffuse. wrote {a.out}")
