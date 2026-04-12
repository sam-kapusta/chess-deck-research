"""Coherence sweep: run coherence test on all SAE checkpoints."""
import json, os, sys, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
GD = BASE + '/output/game_analysis'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'
TOP_K = 50

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

# Load cache
print('Loading cache...')
cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
all_hidden = cache['hidden']
key_to_idx = cache['key_to_idx']

# Load game indices
print('Loading games...')
game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
indices = []
for gf in game_files:
    g = json.load(open(GD + '/' + gf))
    for m in g.get('moves', []):
        pk = m['fen'] + '|' + m['uci']
        if pk in key_to_idx:
            indices.append(key_to_idx[pk])
print(str(len(indices)) + ' moves')

# Collect all checkpoints
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
    ds = cfg['dict_size']
    k = cfg.get('k')
    arch = cfg.get('arch', 'btk')
    if arch == 'v1': sae = SAE_V1(1024, ds)
    elif arch == 'gated': sae = SAE_Gated(1024, ds)
    else: sae = SAE_BTK(1024, ds, k if k else 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    sae = sae.cuda().eval()
    mn = torch.tensor(ckpt['normalization']['mean'], device='cuda')
    sd = torch.tensor(ckpt['normalization']['std'], device='cuda').clamp(min=1e-8)
    return sae, mn, sd, ds

def run_coherence(sae, mn, sd, dict_size):
    # Collect top-K strongest activations per feature
    feature_top = {fid: [] for fid in range(dict_size)}
    for i, idx in enumerate(indices):
        h = all_hidden[idx].float().cuda()
        tokens = (h - mn) / sd
        with torch.no_grad():
            _, acts = sae(tokens.unsqueeze(0))
        max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
        for fid in np.where(max_acts > 0)[0]:
            fid = int(fid)
            strength = float(max_acts[fid])
            tl = feature_top[fid]
            if len(tl) < TOP_K:
                tl.append((strength, idx))
            elif strength > tl[0][0]:
                tl[0] = (strength, idx)
                tl.sort()

    # Compute coherence per feature
    coherences = []
    for fid in range(dict_size):
        tl = feature_top[fid]
        if len(tl) < 10:
            continue
        cache_idx = [idx for _, idx in tl]
        vecs = torch.stack([all_hidden[idx].float().mean(dim=0) for idx in cache_idx])
        vecs_norm = F.normalize(vecs, dim=1)
        cos_sim = torch.mm(vecs_norm, vecs_norm.T)
        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        avg = float(cos_sim[mask].mean())
        coherences.append(avg)

    if not coherences:
        return {'n': 0, 'mean': 0, 'median': 0, 'high_pct': 0, 'low_pct': 0}

    arr = np.array(coherences)
    return {
        'n': len(arr),
        'mean': round(float(arr.mean()), 3),
        'median': round(float(np.median(arr)), 3),
        'high_pct': round(100 * (arr >= 0.8).sum() / len(arr), 0),
        'low_pct': round(100 * (arr < 0.6).sum() / len(arr), 0),
    }

# Run sweep
all_results = {}
for path, name in CHECKPOINTS:
    print()
    print('=== ' + name + ' ===')
    sys.stdout.flush()
    sae, mn, sd, ds = load_sae(path)
    t0 = time.time()
    r = run_coherence(sae, mn, sd, ds)
    elapsed = time.time() - t0
    all_results[name] = r
    print('  {:.0f}s — n={} mean={} median={} high={}% low={}%'.format(
        elapsed, r['n'], r['mean'], r['median'], r['high_pct'], r['low_pct']))
    del sae; torch.cuda.empty_cache()

# Summary
print()
print('=' * 70)
print('COHERENCE SWEEP SUMMARY')
print('=' * 70)
print()
print('{:<22} {:>5} {:>7} {:>7} {:>7} {:>7}'.format('Config', 'Feats', 'Mean', 'Median', 'High%', 'Low%'))
print('-' * 60)
for _, name in CHECKPOINTS:
    r = all_results.get(name)
    if not r: continue
    print('{:<22} {:>5} {:>7} {:>7} {:>7} {:>7}'.format(
        name, r['n'], r['mean'], r['median'], r['high_pct'], r['low_pct']))

with open(BASE + '/output/coherence_sweep.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print()
print('Saved to ' + BASE + '/output/coherence_sweep.json')
print('DONE')
