"""Temporal consistency + Polysemanticity score for all SAE checkpoints.

Temporal: within a game, if feature fires on move N, probability it fires on move N+1.
  High = "sticky" (positional/structural). Low = "spiky" (tactical/one-off).

Polysemanticity: for each feature's decoder vector, how many other features have
  high cosine similarity? Pure features are orthogonal. Polysemantic ones correlate.
"""
import json, os, sys, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
GD = BASE + '/output/game_analysis'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'

class SAE_BTK(nn.Module):
    def __init__(s, di, dd, k):
        super().__init__()
        s.encoder=nn.Linear(di,dd);s.decoder=nn.Linear(dd,di,bias=False)
        s.pre_bias=nn.Parameter(torch.zeros(di));s.k=k
    def forward(s, x):
        z=s.encoder(x-s.pre_bias);z_relu=F.relu(z);flat=z_relu.reshape(-1)
        tc=s.k*x.shape[0]
        if tc>flat.shape[0]:tc=flat.shape[0]
        th=torch.topk(flat,tc).values[-1]
        sparse=z_relu*(z_relu>=th).float()
        return s.decoder(sparse)+s.pre_bias, sparse

class SAE_V1(nn.Module):
    def __init__(s, di, dd):
        super().__init__()
        s.encoder=nn.Linear(di,dd);s.decoder=nn.Linear(dd,di,bias=False)
        s.pre_bias=nn.Parameter(torch.zeros(di))
    def forward(s, x):
        z=s.encoder(x-s.pre_bias);z_relu=F.relu(z)
        return s.decoder(z_relu)+s.pre_bias, z_relu

class SAE_Gated(nn.Module):
    def __init__(s, di, dd):
        super().__init__()
        s.gate=nn.Linear(di,dd);s.magnitude=nn.Linear(di,dd)
        s.decoder=nn.Linear(dd,di,bias=False);s.pre_bias=nn.Parameter(torch.zeros(di))
    def forward(s, x):
        xc=x-s.pre_bias;g=torch.sigmoid(s.gate(xc));m=F.relu(s.magnitude(xc))
        sparse=g*m;return s.decoder(sparse)+s.pre_bias, sparse

# Load cache + games
print('Loading...')
cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
all_hidden = cache['hidden']
key_to_idx = cache['key_to_idx']

game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
game_data = [json.load(open(GD + '/' + gf)) for gf in game_files]
print(str(len(game_data)) + ' games loaded')

# Collect checkpoints
CHECKPOINTS = []
for f in sorted(os.listdir(BASE + '/output')):
    if f.startswith('sweep_sae_') and f.endswith('.pt'):
        CHECKPOINTS.append((BASE + '/output/' + f, f.replace('sweep_sae_', '').replace('.pt', '')))
for k_val in [1, 4, 16, 32]:
    f = BASE + '/output/sae_puzzle_pertoken_2048_k' + str(k_val) + '.pt'
    if os.path.exists(f):
        CHECKPOINTS.append((f, 'orig_2048_k' + str(k_val)))
print(str(len(CHECKPOINTS)) + ' checkpoints')

def load_sae(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ds = cfg['dict_size']; k = cfg.get('k'); arch = cfg.get('arch', 'btk')
    if arch == 'v1': sae = SAE_V1(1024, ds)
    elif arch == 'gated': sae = SAE_Gated(1024, ds)
    else: sae = SAE_BTK(1024, ds, k if k else 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    sae = sae.cuda().eval()
    mn = torch.tensor(ckpt['normalization']['mean'], device='cuda')
    sd = torch.tensor(ckpt['normalization']['std'], device='cuda').clamp(min=1e-8)
    return sae, mn, sd, ds

def run_polysemanticity(sae, dict_size):
    """Decoder-based polysemanticity: how many other features correlate with each one?"""
    with torch.no_grad():
        W = sae.decoder.weight.data  # [input_dim, dict_size]
        W_norm = F.normalize(W, dim=0)
        cos = torch.mm(W_norm.T, W_norm)  # [dict_size, dict_size]
        cos.fill_diagonal_(0)

    # Per feature: count features with |cosine| > 0.5
    cos_abs = cos.abs().cpu().numpy()
    poly_scores = []
    for fid in range(dict_size):
        n_correlated = int((cos_abs[fid] > 0.5).sum())
        max_corr = float(cos_abs[fid].max())
        poly_scores.append((n_correlated, max_corr))

    avg_correlated = np.mean([p[0] for p in poly_scores])
    avg_max = np.mean([p[1] for p in poly_scores])
    # % of features that are "pure" (0 correlated at 0.5 threshold)
    pure_pct = 100 * sum(1 for p in poly_scores if p[0] == 0) / len(poly_scores)
    # % that are highly polysemantic (5+ correlated)
    poly_pct = 100 * sum(1 for p in poly_scores if p[0] >= 5) / len(poly_scores)

    return {
        'avg_correlated': round(avg_correlated, 1),
        'avg_max_cosine': round(avg_max, 3),
        'pure_pct': round(pure_pct, 0),
        'polysemantic_pct': round(poly_pct, 0),
    }

def run_temporal(sae, mn, sd, dict_size):
    """Within-game temporal consistency: P(fires on move N+1 | fires on move N)."""
    # For each game, get features per move in order
    fire_then_fire = np.zeros(dict_size)  # fires on N AND N+1
    fire_total = np.zeros(dict_size)       # fires on N (where N+1 exists)

    for g in game_data:
        moves = g.get('moves', [])
        prev_feats = None
        for m in moves:
            pk = m['fen'] + '|' + m['uci']
            if pk not in key_to_idx:
                prev_feats = None
                continue

            h = all_hidden[key_to_idx[pk]].float().cuda()
            with torch.no_grad():
                _, acts = sae(((h - mn) / sd).unsqueeze(0))
            max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
            curr_feats = set(int(f) for f in np.where(max_acts > 0)[0])

            if prev_feats is not None:
                for fid in prev_feats:
                    fire_total[fid] += 1
                    if fid in curr_feats:
                        fire_then_fire[fid] += 1

            prev_feats = curr_feats

    # Compute per-feature stickiness
    stickiness = []
    for fid in range(dict_size):
        if fire_total[fid] < 10:
            continue
        stick = fire_then_fire[fid] / fire_total[fid]
        stickiness.append(stick)

    if not stickiness:
        return {'n': 0, 'avg_sticky': 0, 'median_sticky': 0, 'high_sticky_pct': 0, 'low_sticky_pct': 0}

    arr = np.array(stickiness)
    return {
        'n': len(arr),
        'avg_sticky': round(float(arr.mean()), 3),
        'median_sticky': round(float(np.median(arr)), 3),
        'high_sticky_pct': round(100 * (arr >= 0.5).sum() / len(arr), 0),  # fires >50% of next moves
        'low_sticky_pct': round(100 * (arr < 0.2).sum() / len(arr), 0),    # fires <20% of next moves
    }

# Run
all_results = {}
for path, name in CHECKPOINTS:
    print()
    print('=== ' + name + ' ===')
    sys.stdout.flush()
    sae, mn, sd, ds = load_sae(path)

    t0 = time.time()
    poly = run_polysemanticity(sae, ds)
    t_poly = time.time() - t0
    print('  Poly ({:.1f}s): avg_corr={} pure={}% polysem={}%'.format(
        t_poly, poly['avg_correlated'], poly['pure_pct'], poly['polysemantic_pct']))

    t0 = time.time()
    temp = run_temporal(sae, mn, sd, ds)
    t_temp = time.time() - t0
    print('  Temporal ({:.0f}s): n={} avg_sticky={} high={}% low={}%'.format(
        t_temp, temp['n'], temp['avg_sticky'], temp['high_sticky_pct'], temp['low_sticky_pct']))
    sys.stdout.flush()

    all_results[name] = {'polysemanticity': poly, 'temporal': temp}
    del sae; torch.cuda.empty_cache()

# Summary
print()
print('=' * 90)
print('TEMPORAL + POLYSEMANTICITY SUMMARY')
print('=' * 90)
print()
print('{:<22} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}'.format(
    'Config', 'Feats', 'Sticky', 'HiStk%', 'LoStk%', 'AvgCor', 'Pure%', 'Poly%'))
print('-' * 80)
for _, name in CHECKPOINTS:
    r = all_results.get(name)
    if not r: continue
    t = r['temporal']; p = r['polysemanticity']
    print('{:<22} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}'.format(
        name, t['n'], t['avg_sticky'], t['high_sticky_pct'], t['low_sticky_pct'],
        p['avg_correlated'], p['pure_pct'], p['polysemantic_pct']))

with open(BASE + '/output/temporal_poly_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print()
print('Saved to ' + BASE + '/output/temporal_poly_results.json')
print('DONE')
