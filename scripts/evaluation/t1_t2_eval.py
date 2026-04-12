"""T1 + T2 + T3 evaluation across all SAE checkpoints.

T1 — Structural:
  - Alive features (fire >0 on any game position)
  - Dead features
  - L0 (avg active features per position)
  - Decoder cosine similarity (redundancy — max cosine between any two decoder columns)
  - Activation bimodality (per feature: is it bimodal or uniform?)

T2 — Concept concentration:
  - Puzzle theme Herfindahl (does a feature fire on one theme or many?)
  - Game phase concentration (opening/middle/endgame)
  - Classification concentration (good/mistake/blunder)

T3 — Feature redundancy (Jaccard):
  - For each pair of alive features, compute Jaccard overlap of firing positions
  - High overlap = redundant (two features encode the same concept)
  - Reports: redundant pairs (J>0.3), similar pairs (J>0.15), avg max overlap
"""
import json, os, sys, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
GD = BASE + '/output/game_analysis'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'
THEMES_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32_themes.json'

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

# Load cached activations
print('Loading cache...')
cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
all_hidden = cache['hidden']  # [N, 77, 1024] float16
key_to_idx = cache['key_to_idx']
print('Cache: ' + str(all_hidden.shape[0]) + ' pairs')

# Load game data
print('Loading games...')
game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
game_data = [json.load(open(GD + '/' + f)) for f in game_files]
print(str(len(game_data)) + ' games')

# Collect all SAE checkpoints
CHECKPOINTS = []
for f in sorted(os.listdir(BASE + '/output')):
    if f.startswith('sweep_sae_') and f.endswith('.pt'):
        CHECKPOINTS.append((BASE + '/output/' + f, f.replace('sweep_sae_', '').replace('.pt', '')))
# Also add original SAEs
for k_val in [1, 4, 16, 32]:
    f = BASE + '/output/sae_puzzle_pertoken_2048_k' + str(k_val) + '.pt'
    if os.path.exists(f):
        CHECKPOINTS.append((f, 'orig_2048_k' + str(k_val)))

print(str(len(CHECKPOINTS)) + ' checkpoints to evaluate')
for f, name in CHECKPOINTS:
    print('  ' + name)
sys.stdout.flush()

def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    ds = cfg['dict_size']
    k = cfg.get('k')
    arch = cfg.get('arch', 'btk')
    if arch == 'v1':
        sae = SAE_V1(1024, ds)
    elif arch == 'gated':
        sae = SAE_Gated(1024, ds)
    else:
        sae = SAE_BTK(1024, ds, k if k else 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    sae = sae.cuda().eval()
    mn = torch.tensor(ckpt['normalization']['mean'], device='cuda')
    sd = torch.tensor(ckpt['normalization']['std'], device='cuda').clamp(min=1e-8)
    return sae, mn, sd, ds, k, arch

def eval_t1(sae, mn, sd, dict_size, arch):
    """T1: Structural metrics."""
    # Sample 2000 positions from games
    positions = []
    for g in game_data:
        for m in g.get('moves', []):
            pk = m['fen'] + '|' + m['uci']
            if pk in key_to_idx:
                positions.append(key_to_idx[pk])
            if len(positions) >= 2000:
                break
        if len(positions) >= 2000:
            break

    # Run SAE on all positions, collect stats
    fire_counts = np.zeros(dict_size)  # how many positions each feature fires on
    l0_values = []  # active features per position
    all_strengths = [[] for _ in range(dict_size)]  # activation values per feature

    for idx in positions:
        h = all_hidden[idx].float().cuda()
        tokens = (h - mn) / sd
        with torch.no_grad():
            _, acts = sae(tokens.unsqueeze(0))
        # Per-token max activation per feature
        max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
        active = max_acts > 0
        fire_counts += active.astype(float)
        l0_values.append(active.sum())
        for fid in np.where(active)[0]:
            all_strengths[fid].append(max_acts[fid])

    alive = int((fire_counts > 0).sum())
    dead = dict_size - alive
    avg_l0 = round(np.mean(l0_values), 1)
    median_l0 = round(np.median(l0_values), 1)

    # Fire rate distribution
    fire_rates = fire_counts / len(positions) * 100
    fr_under1 = int((fire_rates < 1).sum()) - dead
    fr_1to5 = int(((fire_rates >= 1) & (fire_rates < 5)).sum())
    fr_5to20 = int(((fire_rates >= 5) & (fire_rates < 20)).sum())
    fr_over20 = int((fire_rates >= 20).sum())

    # Decoder cosine similarity (redundancy)
    if hasattr(sae, 'decoder'):
        with torch.no_grad():
            W = sae.decoder.weight.data  # [input_dim, dict_size]
            W_norm = F.normalize(W, dim=0)  # normalize each column
            cos_sim = torch.mm(W_norm.T, W_norm)  # [dict_size, dict_size]
            # Zero diagonal
            cos_sim.fill_diagonal_(0)
            max_cos = cos_sim.max().item()
            avg_cos = cos_sim.abs().mean().item()
            # How many pairs have >0.9 cosine?
            high_sim_pairs = int((cos_sim > 0.9).sum().item() / 2)
    else:
        max_cos = avg_cos = 0
        high_sim_pairs = 0

    # Bimodality: for alive features, is activation distribution bimodal?
    # Simple test: what fraction of positions have activation=0 vs >0?
    # Good features: fire rarely (bimodal). Bad features: fire often (not bimodal).
    bimodal_count = 0
    for fid in range(dict_size):
        if fire_counts[fid] == 0:
            continue
        fr = fire_counts[fid] / len(positions)
        if fr < 0.3:  # fires on <30% — bimodal enough
            bimodal_count += 1
    bimodal_pct = round(100 * bimodal_count / alive, 0) if alive else 0

    return {
        'alive': alive, 'dead': dead, 'avg_l0': avg_l0, 'median_l0': median_l0,
        'fr_under1': fr_under1, 'fr_1to5': fr_1to5, 'fr_5to20': fr_5to20, 'fr_over20': fr_over20,
        'max_cos_sim': round(max_cos, 3), 'avg_cos_sim': round(avg_cos, 3),
        'high_sim_pairs': high_sim_pairs, 'bimodal_pct': bimodal_pct,
    }

def eval_t2(sae, mn, sd, dict_size):
    """T2: Concept concentration."""
    # For each feature, track: game phase, classification, piece type
    phase_counts = {fid: {'opening': 0, 'middle': 0, 'endgame': 0} for fid in range(dict_size)}
    cls_counts = {fid: {} for fid in range(dict_size)}
    total_fires = np.zeros(dict_size)

    for g in game_data:
        moves = g.get('moves', [])
        for mi, m in enumerate(moves):
            pk = m['fen'] + '|' + m['uci']
            if pk not in key_to_idx:
                continue

            ply = m.get('ply', 0)
            cls = m.get('classification', '')
            phase = 'opening' if ply <= 20 else ('middle' if ply <= 60 else 'endgame')

            h = all_hidden[key_to_idx[pk]].float().cuda()
            with torch.no_grad():
                _, acts = sae(((h - mn) / sd).unsqueeze(0))
            max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
            active_fids = np.where(max_acts > 0)[0]

            for fid in active_fids:
                fid = int(fid)
                total_fires[fid] += 1
                phase_counts[fid][phase] += 1
                cls_counts[fid][cls] = cls_counts[fid].get(cls, 0) + 1

    # Compute Herfindahl index for phase and classification
    def herfindahl(counts_dict):
        total = sum(counts_dict.values())
        if total == 0:
            return 0
        return sum((v / total) ** 2 for v in counts_dict.values())

    phase_hhi = []
    cls_hhi = []
    for fid in range(dict_size):
        if total_fires[fid] < 20:
            continue
        phase_hhi.append(herfindahl(phase_counts[fid]))
        cls_hhi.append(herfindahl(cls_counts[fid]))

    return {
        'n_measured': len(phase_hhi),
        'avg_phase_hhi': round(np.mean(phase_hhi), 3) if phase_hhi else 0,
        'median_phase_hhi': round(np.median(phase_hhi), 3) if phase_hhi else 0,
        'avg_cls_hhi': round(np.mean(cls_hhi), 3) if cls_hhi else 0,
        'median_cls_hhi': round(np.median(cls_hhi), 3) if cls_hhi else 0,
        # Features concentrated in one phase (HHI > 0.5)
        'phase_concentrated': sum(1 for h in phase_hhi if h > 0.5),
        'phase_concentrated_pct': round(100 * sum(1 for h in phase_hhi if h > 0.5) / len(phase_hhi), 0) if phase_hhi else 0,
        # Features concentrated in one classification
        'cls_concentrated': sum(1 for h in cls_hhi if h > 0.5),
        'cls_concentrated_pct': round(100 * sum(1 for h in cls_hhi if h > 0.5) / len(cls_hhi), 0) if cls_hhi else 0,
    }

def eval_t3(sae, mn, sd, dict_size):
    """T3: Feature redundancy via Jaccard overlap of firing positions.

    For each alive feature, track which positions it fires on.
    Then compute pairwise Jaccard between the top-firing features.
    High Jaccard = redundant features encoding the same concept.
    """
    # Collect firing sets for alive features (sample 2000 positions)
    fire_sets = {}
    for fid in range(dict_size):
        fire_sets[fid] = set()

    n_pos = 0
    for g in game_data:
        for m in g.get('moves', []):
            pk = m['fen'] + '|' + m['uci']
            if pk not in key_to_idx:
                continue
            h = all_hidden[key_to_idx[pk]].float().cuda()
            with torch.no_grad():
                _, acts = sae(((h - mn) / sd).unsqueeze(0))
            max_acts = acts.squeeze(0).max(dim=0).values.cpu().numpy()
            active_fids = np.where(max_acts > 0)[0]
            for fid in active_fids:
                fire_sets[int(fid)].add(n_pos)
            n_pos += 1
            if n_pos >= 2000:
                break
        if n_pos >= 2000:
            break

    # Filter to features that fire on >= 10 positions
    alive_fids = [fid for fid in range(dict_size) if len(fire_sets[fid]) >= 10]
    if len(alive_fids) < 2:
        return {'n_measured': 0, 'redundant_pairs': 0, 'similar_pairs': 0, 'avg_max_jaccard': 0}

    # For efficiency, only compute pairwise for top 200 most-firing features
    alive_fids.sort(key=lambda f: -len(fire_sets[f]))
    top_fids = alive_fids[:200]

    max_jaccards = []
    redundant_pairs = 0
    similar_pairs = 0

    for i in range(len(top_fids)):
        best_j = 0
        s1 = fire_sets[top_fids[i]]
        for j in range(i + 1, len(top_fids)):
            s2 = fire_sets[top_fids[j]]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            if union == 0:
                continue
            jacc = intersection / union
            if jacc > best_j:
                best_j = jacc
            if jacc > 0.3:
                redundant_pairs += 1
            elif jacc > 0.15:
                similar_pairs += 1
        max_jaccards.append(best_j)

    return {
        'n_measured': len(top_fids),
        'redundant_pairs': redundant_pairs,
        'similar_pairs': similar_pairs,
        'avg_max_jaccard': round(np.mean(max_jaccards), 3) if max_jaccards else 0,
        'median_max_jaccard': round(np.median(max_jaccards), 3) if max_jaccards else 0,
        'pct_with_redundant': round(100 * sum(1 for j in max_jaccards if j > 0.3) / len(max_jaccards), 0) if max_jaccards else 0,
    }

# Run evaluation
all_results = {}

for ckpt_path, name in CHECKPOINTS:
    print()
    print('=' * 60)
    print('Evaluating: ' + name)
    print('=' * 60)
    sys.stdout.flush()

    sae, mn, sd, ds, k, arch = load_sae(ckpt_path)

    t0 = time.time()
    t1 = eval_t1(sae, mn, sd, ds, arch)
    t1_time = time.time() - t0
    print('  T1 ({:.0f}s): alive={} dead={} L0={} maxCos={} bimodal={}%'.format(
        t1_time, t1['alive'], t1['dead'], t1['avg_l0'], t1['max_cos_sim'], t1['bimodal_pct']))
    print('    fire rates: <1%={} 1-5%={} 5-20%={} >20%={}'.format(
        t1['fr_under1'], t1['fr_1to5'], t1['fr_5to20'], t1['fr_over20']))
    sys.stdout.flush()

    t0 = time.time()
    t2 = eval_t2(sae, mn, sd, ds)
    t2_time = time.time() - t0
    print('  T2 ({:.0f}s): phase_HHI={} cls_HHI={} phase_conc={}% cls_conc={}%'.format(
        t2_time, t2['avg_phase_hhi'], t2['avg_cls_hhi'],
        t2['phase_concentrated_pct'], t2['cls_concentrated_pct']))
    sys.stdout.flush()

    t0 = time.time()
    t3 = eval_t3(sae, mn, sd, ds)
    t3_time = time.time() - t0
    print('  T3 ({:.0f}s): redundant={} similar={} avg_max_J={} pct_redundant={}%'.format(
        t3_time, t3['redundant_pairs'], t3['similar_pairs'],
        t3['avg_max_jaccard'], t3['pct_with_redundant']))
    sys.stdout.flush()

    all_results[name] = {'t1': t1, 't2': t2, 't3': t3, 'config': {'dict_size': ds, 'k': k, 'arch': arch}}

    del sae
    torch.cuda.empty_cache()

# Summary table
print()
print('=' * 120)
print('T1 + T2 SUMMARY')
print('=' * 120)
print()
print('{:<22} {:>5} {:>5} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7}'.format(
    'Config', 'Alive', 'Dead', 'L0', 'MaxCos', 'HiSim', '<1%', '>20%', 'Bim%',
    'PhHHI', 'ClHHI', 'PhCon%', 'ClCon%'))
print('-' * 120)

for _, name in CHECKPOINTS:
    r = all_results.get(name)
    if not r: continue
    t1 = r['t1']; t2 = r['t2']
    print('{:<22} {:>5} {:>5} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7}'.format(
        name, t1['alive'], t1['dead'], t1['avg_l0'], t1['max_cos_sim'], t1['high_sim_pairs'],
        t1['fr_under1'], t1['fr_over20'], t1['bimodal_pct'],
        t2['avg_phase_hhi'], t2['avg_cls_hhi'], t2['phase_concentrated_pct'], t2['cls_concentrated_pct']))

# Save
with open(BASE + '/output/t1_t2_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print()
print('Saved to ' + BASE + '/output/t1_t2_results.json')
print('DONE')
