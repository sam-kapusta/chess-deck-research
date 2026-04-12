"""Step 2: Sweep all SAE checkpoints against cached encoder activations.

For each SAE: load, run on cached activations, compute agreement stats.
No encoder needed — just SAE forward passes on cached hidden states.
"""
import json, os, sys, time, torch
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
GD = BASE + '/output/game_analysis'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'

# All SAE checkpoints to test
SAE_FILES = [
    (BASE + '/output/sae_puzzle_pertoken_2048_k1.pt', 'k=1'),
    (BASE + '/output/sae_puzzle_pertoken_2048_k4.pt', 'k=4'),
    (BASE + '/output/sae_puzzle_pertoken_2048_k16.pt', 'k=16'),
    (BASE + '/output/sae_puzzle_pertoken_2048_k32.pt', 'k=32'),
]

# Filter to existing files
SAE_FILES = [(f, name) for f, name in SAE_FILES if os.path.exists(f)]
print('SAE checkpoints found: ' + str(len(SAE_FILES)))
for f, name in SAE_FILES:
    print('  ' + name + ': ' + f)

class SAE(nn.Module):
    def __init__(s, di, dd, k):
        super().__init__()
        s.encoder=nn.Linear(di,dd);s.decoder=nn.Linear(dd,di,bias=False)
        s.pre_bias=nn.Parameter(torch.zeros(di));s.k=k
    def forward(s, x):
        z=s.encoder(x-s.pre_bias);z_relu=F.relu(z);flat=z_relu.reshape(-1)
        tc=s.k*x.shape[0]
        if tc>flat.shape[0]:tc=flat.shape[0]
        th=torch.topk(flat,tc).values[-1]
        return s.decoder(z_relu*(z_relu>=th).float())+s.pre_bias, z_relu*(z_relu>=th).float()

# Load cache
print()
print('Loading encoder activation cache...')
cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
all_hidden = cache['hidden']  # [N, 77, 1024] float16
key_to_idx = cache['key_to_idx']
print('Cache: ' + str(all_hidden.shape[0]) + ' pairs, ' + str(cache['n_games']) + ' games')

# Load game data to map moves to pairs
print('Loading game metadata...')
game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
game_data = []
for gf in game_files:
    g = json.load(open(GD + '/' + gf))
    game_data.append(g)
print(str(len(game_data)) + ' games loaded')

def get_features_from_cache(hidden_fp32, sae, norm_mean, norm_std):
    """Run SAE on cached hidden states, return set of active feature IDs."""
    tokens = (hidden_fp32 - norm_mean) / norm_std
    with torch.no_grad():
        _, acts = sae(tokens.unsqueeze(0))
    max_per_f = acts.squeeze(0).max(dim=0).values.cpu().numpy()
    import numpy as np
    return set(int(f) for f in np.where(max_per_f > 0)[0])

def run_agreement_test(sae, norm_mean, norm_std, dict_size):
    """Run agreement test for a single SAE."""
    import numpy as np

    both = [0] * dict_size
    played_only = [0] * dict_size
    best_only = [0] * dict_size
    total = 0
    skipped = 0

    for g in game_data:
        moves = g.get('moves', [])
        for m in moves:
            fen = m['fen']
            uci = m.get('uci', '')
            best_uci = m.get('best_uci', m.get('bestMove', ''))

            if not best_uci:
                continue

            played_key = fen + '|' + uci
            best_key = fen + '|' + best_uci

            if played_key not in key_to_idx or best_key not in key_to_idx:
                skipped += 1
                continue

            total += 1

            # Get features from cache
            pi = key_to_idx[played_key]
            bi = key_to_idx[best_key]

            played_h = all_hidden[pi].float().cuda()
            best_h = all_hidden[bi].float().cuda()

            played_feats = get_features_from_cache(played_h, sae, norm_mean, norm_std)
            best_feats = get_features_from_cache(best_h, sae, norm_mean, norm_std)

            for fid in range(dict_size):
                inp = fid in played_feats
                inb = fid in best_feats
                if inp and inb:
                    both[fid] += 1
                elif inp:
                    played_only[fid] += 1
                elif inb:
                    best_only[fid] += 1

    return both, played_only, best_only, total, skipped

def analyze_agreement(both, played_only, best_only, total, dict_size):
    """Compute agreement distribution and key stats."""
    results = []
    for fid in range(dict_size):
        fires_either = both[fid] + played_only[fid] + best_only[fid]
        if fires_either < 20:
            continue
        agreement = round(100 * both[fid] / fires_either, 1)
        fire_rate = round(100 * fires_either / total, 1)
        results.append({
            'fid': fid,
            'agreement': agreement,
            'both': both[fid],
            'played_only': played_only[fid],
            'best_only': best_only[fid],
            'fire_rate': fire_rate,
        })

    # Distribution
    dist = {}
    for lo, hi, label in [(0, 20, 'pure_move'), (20, 40, 'mostly_move'), (40, 60, 'mixed'), (60, 80, 'mostly_pos'), (80, 101, 'pure_pos')]:
        dist[label] = sum(1 for r in results if lo <= r['agreement'] < hi)

    # Fire rate distribution
    fire_dist = {}
    for lo, hi, label in [(0, 1, '<1%'), (1, 5, '1-5%'), (5, 10, '5-10%'), (10, 20, '10-20%'), (20, 40, '20-40%'), (40, 100, '>40%')]:
        fire_dist[label] = sum(1 for r in results if lo <= r['fire_rate'] < hi)

    # Coaching useful: low agreement + high best/played ratio
    coaching_misses = 0
    coaching_overuse = 0
    for r in results:
        if r['agreement'] > 30:
            continue
        if r['best_only'] > r['played_only'] * 1.5:
            coaching_misses += 1
        if r['played_only'] > r['best_only'] * 1.5:
            coaching_overuse += 1

    return {
        'n_features_with_data': len(results),
        'agreement_dist': dist,
        'fire_rate_dist': fire_dist,
        'coaching_misses': coaching_misses,
        'coaching_overuse': coaching_overuse,
        'total_positions': total,
        'results': results,
    }

# Run sweep
print()
all_results = {}

for sae_file, sae_name in SAE_FILES:
    print('=' * 60)
    print('Testing: ' + sae_name + ' (' + os.path.basename(sae_file) + ')')
    print('=' * 60)

    ckpt = torch.load(sae_file, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    dict_size = cfg['dict_size']
    k = cfg['k']
    sae = SAE(1024, dict_size, k)
    sae.load_state_dict(ckpt['model_state_dict'])
    sae = sae.cuda().eval()
    mn = torch.tensor(ckpt['normalization']['mean'], dtype=torch.float32, device='cuda')
    sd = torch.tensor(ckpt['normalization']['std'], dtype=torch.float32, device='cuda').clamp(min=1e-8)

    print('  dict_size=' + str(dict_size) + ' k=' + str(k))
    sys.stdout.flush()

    # Quick sanity check: run on first valid pair
    for g in game_data:
        for m in g.get('moves', []):
            best_uci = m.get('best_uci', m.get('bestMove', ''))
            if not best_uci: continue
            pk = m['fen'] + '|' + m['uci']
            bk = m['fen'] + '|' + best_uci
            if pk in key_to_idx and bk in key_to_idx:
                ph = all_hidden[key_to_idx[pk]].float().cuda()
                bh = all_hidden[key_to_idx[bk]].float().cuda()
                pf = get_features_from_cache(ph, sae, mn, sd)
                bf = get_features_from_cache(bh, sae, mn, sd)
                print('  Sanity: played_feats={} best_feats={} overlap={}'.format(len(pf), len(bf), len(pf & bf)))
                break
        else:
            continue
        break

    t0 = time.time()
    b, po, bo, total, skipped = run_agreement_test(sae, mn, sd, dict_size)
    elapsed = time.time() - t0

    analysis = analyze_agreement(b, po, bo, total, dict_size)
    all_results[sae_name] = analysis

    print('  Time: {:.1f}s  Positions: {}  Skipped: {}'.format(elapsed, total, skipped))
    print('  Features with data: ' + str(analysis['n_features_with_data']))
    print('  Agreement distribution:')
    for label, count in analysis['agreement_dist'].items():
        print('    {:<15} {}'.format(label, count))
    print('  Fire rate distribution:')
    for label, count in analysis['fire_rate_dist'].items():
        print('    {:<10} {}'.format(label, count))
    print('  Coaching: {} missed features, {} overused features'.format(
        analysis['coaching_misses'], analysis['coaching_overuse']))
    print()
    sys.stdout.flush()

    del sae, ckpt
    torch.cuda.empty_cache()

# Summary comparison
print()
print('=' * 60)
print('SUMMARY COMPARISON')
print('=' * 60)
print()
print('{:<8} {:>6} {:>8} {:>8} {:>6} {:>6} {:>6} {:>8} {:>8}'.format(
    'Config', 'Total', 'PurePos', 'PureMov', 'Mixed', 'Miss', 'Over', 'Clean%', 'Noise%'))
print('-' * 75)

for sae_name in [n for _, n in SAE_FILES]:
    a = all_results.get(sae_name)
    if not a: continue
    d = a['agreement_dist']
    total_f = a['n_features_with_data']
    pure_pos = d.get('pure_pos', 0)
    pure_mov = d.get('pure_move', 0)
    mixed = d.get('mixed', 0) + d.get('mostly_move', 0) + d.get('mostly_pos', 0)
    clean = pure_pos + pure_mov
    fd = a['fire_rate_dist']
    noise = fd.get('>40%', 0) + fd.get('20-40%', 0)
    clean_pct = round(100 * clean / total_f, 0) if total_f else 0
    noise_pct = round(100 * noise / total_f, 0) if total_f else 0
    print('{:<8} {:>6} {:>8} {:>8} {:>6} {:>6} {:>6} {:>7.0f}% {:>7.0f}%'.format(
        sae_name, total_f, pure_pos, pure_mov, mixed, a['coaching_misses'], a['coaching_overuse'], clean_pct, noise_pct))

# Save
output_file = BASE + '/output/sae_sweep_agreement.json'
# Remove full results for saving (too large)
save_data = {}
for name, a in all_results.items():
    save_data[name] = {k: v for k, v in a.items() if k != 'results'}
with open(output_file, 'w') as f:
    json.dump(save_data, f, indent=2)
print()
print('Saved summary to ' + output_file)
print('DONE')
