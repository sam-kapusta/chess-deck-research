"""Fuse Opus per-position motifs + SEE heuristic facts into FEATURE names.

Division of labor:
  - Opus per-position labels (all_positions_labeled_opus.json) give the TACTICAL MOTIF
    (fork/pin/skewer/discovered_attack/back_rank/overloaded_defender...) that SEE cannot
    compute, plus tags + blunder_summary.
  - SEE heuristics (computed here) give the QUANTIFIED backbone: which piece hangs, value
    class, severity, is_capture, best-move character, phase.
The feature NAME is the dominant Opus motif (gated at THRESH concentration over the
feature's top-N positions) combined with the concentrated SEE facts. If neither the motif
nor any SEE axis concentrates, the feature is 'diffuse'.

Per feature output: name, motif_dist, concentrated SEE facts, and a top-10 DISPLAY list
where each position carries its full Opus analysis (blunder_summary, refutation, tags)
when covered, else the SEE facts as fallback.

Run on chess-poc from ~/SageMaker:
  python fuse_feature_names.py --model k4 --topn 50 --thresh 0.6 --out fused_names_k4.json
"""
import torch, numpy as np, json, chess, argparse, torch.nn.functional as F
from collections import Counter
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)          # k4 / k6 / k8 / k16
ap.add_argument('--dict', type=int, default=2048)
ap.add_argument('--topn', type=int, default=50)    # positions per feature for motif/SEE stats
ap.add_argument('--thresh', type=float, default=0.6)  # concentration bar for motif or SEE fact
ap.add_argument('--firefloor', type=float, default=0.0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
KK = int(''.join(ch for ch in a.model if ch.isdigit()))

c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
meta = c['metadata']; craw = c['activations'].float()
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)
sd = torch.load(BASE + f'/output/maia3_sae/btk_{a.dict}_{a.model}_nol2.pt', map_location='cpu', weights_only=False)['state_dict']
opus = json.load(open(B + '/all_positions_labeled_opus.json'))
# index opus by exact 'FEN|move' key (matches our cache key)
keyarr = [m['fen'] + '|' + m['blunder_uci'] for m in meta]

VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
PIECE = {chess.KNIGHT:'knight', chess.BISHOP:'bishop', chess.ROOK:'rook', chess.QUEEN:'queen', chess.PAWN:'pawn', chess.KING:'king'}
def cls(p): return 'major piece' if p in ('queen','rook') else 'minor piece' if p in ('bishop','knight') else p
def see(bd, t, stm):
    aa = bd.attackers(stm, t)
    if not aa: return 0
    lva = min(aa, key=lambda s: VAL.get(bd.piece_type_at(s), 99)); cv = VAL.get(bd.piece_type_at(t), 0)
    b2 = bd.copy(); b2.remove_piece_at(t); b2.set_piece_at(t, bd.piece_at(lva)); b2.remove_piece_at(lva)
    return max(0, cv - see(b2, t, not stm))

# ---- activations (single pass) ----
kth = []
for i in range(0, 40000, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
    kth.append(torch.topk(z, KK, 1).values[:, -1].numpy())
th = float(np.concatenate(kth).mean())
ACT = np.zeros((N, sd['W_enc'].shape[1]), np.float32)
for i in range(0, N, 8192):
    z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
    ACT[i:i+z.shape[0]] = z * (z > th)
fire = (ACT > 0).mean(0); live = np.where(fire > a.firefloor)[0]
print(f"{a.model}: {len(live)} live features, threshold={th:.3f}", flush=True)

def see_facts(idx):
    m = meta[int(idx)]
    try:
        b = chess.Board(m['fen']); mover = b.turn
        bm = chess.Move.from_uci(m['blunder_uci']); pc = b.piece_at(bm.from_square)
        f = {'blunder_piece': PIECE.get(pc.piece_type,'?') if pc else '?',
             'iscap': 'capture' if b.is_capture(bm) else 'noncapture'}
        bu = m.get('best_uci','')
        if bu and len(bu) >= 4:
            bmv = chess.Move.from_uci(bu); bpc = b.piece_at(bmv.from_square)
            bpn = PIECE.get(bpc.piece_type,'?') if bpc else '?'
            f['best_piece'] = bpn; f['best_class'] = cls(bpn)
            f['best_type'] = 'capture' if b.is_capture(bmv) else 'check' if b.gives_check(bmv) else 'quiet'
        b2 = b.copy(); b2.push(bm); opp = not mover; worst = 0; wp2 = 'none'
        for sq in chess.SQUARES:
            p = b2.piece_at(sq)
            if p and p.color == mover and b2.is_attacked_by(opp, sq):
                l = see(b2, sq, opp)
                if l > worst: worst = l; wp2 = PIECE.get(p.piece_type,'?')
        f['hang_exact'] = wp2; f['hang_class'] = cls(wp2) if wp2 != 'none' else 'none'
        f['anyhang'] = 'hangs' if wp2 != 'none' else 'safe'; f['_hangval'] = worst
        npc = len(b.piece_map())
        f['phase'] = 'endgame' if npc <= 14 else 'opening' if b.fullmove_number <= 12 else 'middlegame'
        return f, b, bm, bu, m
    except Exception:
        return None

def build_name(motif_top, motif_pct, facts):
    def v(ax): return facts.get(ax, {}).get('value')
    seg = []
    # 1) Opus motif if it concentrates (the tactical layer)
    MOTIF_LABEL = {'hanging_piece':'hangs a piece','fork':'walks into / misses a fork','pin':'pin',
                   'skewer':'skewer','discovered_attack':'discovered attack','back_rank':'back-rank',
                   'overloaded_defender':'overloaded defender','trapped_piece':'trapped piece',
                   'king_safety':'king-safety mistake','tempo_loss':'loses tempo','passed_pawn':'passed-pawn',
                   'promotion_error':'promotion error','positional_mistake':'positional mistake',
                   'missed_tactic':'misses a tactic','pawn_endgame':'pawn-endgame','rook_endgame':'rook-endgame'}
    motif_named = motif_top in MOTIF_LABEL and motif_pct >= 0  # always show motif w/ pct
    # 2) SEE backbone — tighten the hang with the piece class if concentrated
    hang = None
    if v('anyhang') == 'hangs':
        if v('hang_exact'):   hang = f"hangs the {v('hang_exact')}"
        elif v('hang_class'): hang = f"hangs a {v('hang_class')}"
        else:                 hang = "hangs a piece"
    # assemble: prefer SEE hang specificity, annotate with motif if it adds info
    if hang:
        seg.append(hang)
        if motif_top not in ('hanging_piece','?','other') and motif_pct >= a.thresh:
            seg.append(f"via {MOTIF_LABEL.get(motif_top, motif_top)}")
    elif motif_top in MOTIF_LABEL and motif_pct >= a.thresh:
        seg.append(MOTIF_LABEL[motif_top])
    # best-move character
    if v('best_type') == 'capture':   seg.append("missed a capture")
    elif v('best_type') == 'check':   seg.append("missed a check")
    elif v('best_type') == 'quiet':   seg.append("missed a quiet move")
    if v('best_piece') and v('best_piece') != '?':
        seg.append(f"with the {v('best_piece')}")
    elif v('best_class'):
        seg.append(f"with a {v('best_class')}")
    ph = v('phase')
    if ph and ph != 'middlegame': seg.append(f"({ph})")
    if not seg:
        return ('diffuse — no concentrated motif or SEE pattern', 'diffuse')
    return ('; '.join(seg), 'named')

out = {}
for f in live:
    order = np.argsort(-ACT[:, f])
    nfire = int((ACT[:, f] > 0).sum())
    samp = order[:min(a.topn, nfire)]
    motifs = Counter(); AX = {k: Counter() for k in ['blunder_piece','iscap','best_piece','best_class','best_type','hang_exact','hang_class','anyhang','phase']}
    n = 0; n_opus = 0; top10 = []
    for rank, idx in enumerate(samp):
        sf = see_facts(idx)
        if sf is None: continue
        ff, b, bm, bu, m = sf
        for k in AX:
            if k in ff: AX[k][ff[k]] += 1
        n += 1
        key = keyarr[int(idx)]; oa = opus.get(key, {}).get('analysis') if isinstance(opus.get(key), dict) else None
        if isinstance(oa, dict):
            motifs[oa.get('tactical_motif','?')] += 1; n_opus += 1
        if rank < 10:
            try: played_san = b.san(bm)
            except Exception: played_san = m['blunder_uci']
            best_san = ''
            if bu and len(bu) >= 4:
                try: best_san = b.san(chess.Move.from_uci(bu))
                except Exception: best_san = bu
            disp = {'fen': m['fen'], 'played': played_san, 'best': best_san,
                    'hang': f"{ff['hang_exact']}(-{ff['_hangval']})" if ff['anyhang']=='hangs' else 'none',
                    'cp_loss': m.get('cp_loss')}
            if isinstance(oa, dict):
                disp['motif'] = oa.get('tactical_motif'); disp['tags'] = oa.get('tags')
                disp['summary'] = (oa.get('blunder_summary') or '')[:240]
            top10.append(disp)
    # concentrated SEE facts
    facts = {}
    for ax, cnt in AX.items():
        if not cnt: continue
        val, k = cnt.most_common(1)[0]; pct = k / n
        facts[ax] = {'value': val, 'pct': round(pct,2)} if pct >= a.thresh else {'value': None, 'pct': round(pct,2), 'top': val}
    motif_top, motif_pct = ('?', 0.0)
    if n_opus:
        mt, mc = motifs.most_common(1)[0]; motif_top, motif_pct = mt, round(mc/n_opus, 2)
    name, status = build_name(motif_top, motif_pct, facts)
    out[f"f{int(f)}"] = {'fire_rate': round(nfire/N,4), 'n_sampled': n, 'opus_coverage': n_opus,
                         'name': name, 'status': status,
                         'motif_dist': dict(motifs.most_common()), 'motif_top': motif_top, 'motif_pct': motif_pct,
                         'see_facts': facts, 'top10': top10}

json.dump(out, open(a.out, 'w'), indent=1)
named = sum(1 for v in out.values() if v['status'] == 'named')
avgcov = np.mean([v['opus_coverage'] for v in out.values()])
print(f"{a.model}: {len(out)} feats -> {named} named, {len(out)-named} diffuse | mean Opus coverage {avgcov:.1f}/{a.topn}. wrote {a.out}", flush=True)
