"""Train SAEs at multiple scales and architectures, then run agreement test.

Sweep: dict_size × k × architecture
Uses cached encoder activations for instant agreement testing.
"""
import json, os, sys, time, math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'
GD = BASE + '/output/game_analysis'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'

# =============================================
# SAE Architectures
# =============================================

class SAE_BatchTopK(nn.Module):
    """BatchTopK SAE — variable sparsity per example."""
    def __init__(s, di, dd, k):
        super().__init__()
        s.encoder = nn.Linear(di, dd)
        s.decoder = nn.Linear(dd, di, bias=False)
        s.pre_bias = nn.Parameter(torch.zeros(di))
        s.k = k
    def forward(s, x):
        z = s.encoder(x - s.pre_bias)
        z_relu = F.relu(z)
        flat = z_relu.reshape(-1)
        tc = s.k * x.shape[0]
        if tc > flat.shape[0]: tc = flat.shape[0]
        th = torch.topk(flat, tc).values[-1]
        sparse = z_relu * (z_relu >= th).float()
        return s.decoder(sparse) + s.pre_bias, sparse

class SAE_V1(nn.Module):
    """Vanilla SAE with L1 penalty."""
    def __init__(s, di, dd):
        super().__init__()
        s.encoder = nn.Linear(di, dd)
        s.decoder = nn.Linear(dd, di, bias=False)
        s.pre_bias = nn.Parameter(torch.zeros(di))
    def forward(s, x):
        z = s.encoder(x - s.pre_bias)
        z_relu = F.relu(z)
        return s.decoder(z_relu) + s.pre_bias, z_relu

class SAE_Gated(nn.Module):
    """Gated SAE — separate gate and magnitude paths."""
    def __init__(s, di, dd):
        super().__init__()
        s.gate = nn.Linear(di, dd)
        s.magnitude = nn.Linear(di, dd)
        s.decoder = nn.Linear(dd, di, bias=False)
        s.pre_bias = nn.Parameter(torch.zeros(di))
    def forward(s, x):
        xc = x - s.pre_bias
        g = torch.sigmoid(s.gate(xc))
        m = F.relu(s.magnitude(xc))
        sparse = g * m
        return s.decoder(sparse) + s.pre_bias, sparse

# =============================================
# Training
# =============================================

def load_puzzle_data(n_positions=150000):
    """Load puzzle positions for SAE training."""
    from datasets import load_dataset

    with open(MOVE_MAP) as f: M2A = json.load(f)

    _C = list('0123456789abcdefghpnrkqPBNRQKw.')
    _I = {c:i for i,c in enumerate(_C)}; _S = frozenset('12345678')
    def tok(fen):
        p = fen.split(' ')
        while len(p)<6:
            if len(p)==4: p.append('0')
            elif len(p)==5: p.append('1')
            else: p.append('-')
        b,s,c,e,h,f = p[:6]; b = s+b.replace('/',''); ix = []
        for ch in b:
            if ch in _S: ix.extend(int(ch)*[_I['.']])
            elif ch in _I: ix.append(_I[ch])
            else: return None
        if c=='-': ix.extend(4*[_I['.']])
        else:
            for ch in c:
                if ch not in _I: return None
                ix.append(_I[ch])
            ix.extend((4-len(c))*[_I['.']])
        if e=='-': ix.extend(2*[_I['.']])
        else:
            for ch in e:
                if ch not in _I: return None
                ix.append(_I[ch])
        h+='.'*(3-len(h)); ix.extend([_I[x] for x in h[:3]])
        f+='.'*(3-len(f)); ix.extend([_I[x] for x in f[:3]])
        return ix if len(ix)==77 else None

    # Load encoder
    DIM=1024;NL=16;NH=8;HD=128;FFN=4096;FS=79
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.te=nn.Embedding(1968,DIM);self.pe=nn.Embedding(FS,DIM);self.layers=nn.ModuleList()
            for _ in range(NL):
                self.layers.append(nn.ModuleDict(dict(la=nn.LayerNorm(DIM),q=nn.Linear(DIM,DIM,bias=False),k=nn.Linear(DIM,DIM,bias=False),v=nn.Linear(DIM,DIM,bias=False),o=nn.Linear(DIM,DIM,bias=False),lm=nn.LayerNorm(DIM),g=nn.Linear(DIM,FFN,bias=False),u=nn.Linear(DIM,FFN,bias=False),d=nn.Linear(FFN,DIM,bias=False))))
            self.fn=nn.LayerNorm(DIM)
        def forward(self,t):
            B,T=t.shape;s=torch.cat([torch.zeros(B,1,dtype=t.dtype,device=t.device),t[:,:-1]],dim=1)
            x=self.te(s)*math.sqrt(DIM)+self.pe(torch.arange(T,device=t.device))
            for l in self.layers:
                xn=l['la'](x);q=l['q'](xn).reshape(B,T,NH,HD);k=l['k'](xn).reshape(B,T,NH,HD);v=l['v'](xn).reshape(B,T,NH,HD)
                a=torch.einsum('bthd,bThd->bhtT',q,k)/math.sqrt(HD);a=F.softmax(a,dim=-1)
                o=torch.einsum('bhtT,bThd->bthd',a,v).reshape(B,T,DIM);x=x+l['o'](o)
                xn=l['lm'](x);x=x+l['d'](F.silu(l['g'](xn))*l['u'](xn))
            return self.fn(x)
    def glk(i): return 'layer_norm' if i==0 else 'layer_norm_'+str(i)
    def gak(i): return 'multi_head_dot_product_attention' if i==0 else 'multi_head_dot_product_attention_'+str(i)
    def gmk(i): return 'linear' if i==0 else 'linear_'+str(i)

    print('Loading encoder for puzzle extraction...')
    pr=dict(np.load(PARAMS));enc=Enc()
    with torch.no_grad():
        enc.te.weight.copy_(torch.tensor(pr['embed/embeddings']));enc.pe.weight.copy_(torch.tensor(pr['embed_1/embeddings']))
        for i,l in enumerate(enc.layers):
            la,lm=glk(i*2),glk(i*2+1)
            l['la'].weight.copy_(torch.tensor(pr[la+'/scale']));l['la'].bias.copy_(torch.tensor(pr[la+'/offset']))
            l['lm'].weight.copy_(torch.tensor(pr[lm+'/scale']));l['lm'].bias.copy_(torch.tensor(pr[lm+'/offset']))
            ak=gak(i);l['q'].weight.copy_(torch.tensor(pr[ak+'/linear/w']).T);l['k'].weight.copy_(torch.tensor(pr[ak+'/linear_1/w']).T)
            l['v'].weight.copy_(torch.tensor(pr[ak+'/linear_2/w']).T);l['o'].weight.copy_(torch.tensor(pr[ak+'/linear_3/w']).T)
            mb=i*3;l['g'].weight.copy_(torch.tensor(pr[gmk(mb)+'/w']).T);l['u'].weight.copy_(torch.tensor(pr[gmk(mb+1)+'/w']).T)
            l['d'].weight.copy_(torch.tensor(pr[gmk(mb+2)+'/w']).T)
        fl=glk(NL*2);enc.fn.weight.copy_(torch.tensor(pr[fl+'/scale']));enc.fn.bias.copy_(torch.tensor(pr[fl+'/offset']))
    del pr; enc=enc.cuda().eval()

    print('Extracting puzzle activations...')
    ds = load_dataset('Lichess/chess-puzzles', split='train', streaming=True)
    all_acts = []
    count = 0
    BATCH = 64
    batch_seqs = []

    for d in ds:
        fen = d['FEN']
        raw_moves = d.get('Moves', '')
        moves = raw_moves.split() if isinstance(raw_moves, str) else raw_moves
        if len(moves) < 2: continue
        move = moves[1]
        if move not in M2A: continue
        ft = tok(fen)
        if ft is None: continue
        seq = ft + [M2A[move], 64]
        batch_seqs.append(seq)

        if len(batch_seqs) >= BATCH:
            with torch.no_grad():
                h = enc(torch.tensor(batch_seqs, dtype=torch.long, device='cuda'))
                acts = h[:, 1:78, :].cpu()
                all_acts.append(acts)
            batch_seqs = []
            count += len(acts)
            if count % 10000 == 0:
                print('  ' + str(count) + '/' + str(n_positions))
                sys.stdout.flush()
            if count >= n_positions:
                break

    if batch_seqs:
        with torch.no_grad():
            h = enc(torch.tensor(batch_seqs, dtype=torch.long, device='cuda'))
            all_acts.append(h[:, 1:78, :].cpu())

    del enc
    torch.cuda.empty_cache()
    all_acts = torch.cat(all_acts, dim=0).half()  # [N, 77, 1024] float16 to save RAM
    print('Extracted: ' + str(all_acts.shape) + ' (' + str(all_acts.element_size() * all_acts.nelement() // 1024 // 1024) + ' MB)')

    # Compute normalization on a subsample to save memory
    subsample = all_acts[:50000].reshape(-1, 1024).float()
    mean = subsample.mean(dim=0)
    std = subsample.std(dim=0).clamp(min=1e-8)
    del subsample
    return all_acts, mean, std

def train_sae(sae, acts, mean, std, arch_name, epochs=3, batch_size=256, lr=1e-3, l1_coeff=0.01):
    """Train a single SAE."""
    sae = sae.cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    # Subsample to 2M tokens max to stay in RAM, keep as float16 until batch
    flat = acts.reshape(-1, 1024)
    n = flat.shape[0]
    MAX_TOKENS = 2000000
    if n > MAX_TOKENS:
        perm = torch.randperm(n)[:MAX_TOKENS]
        flat = flat[perm]
        n = flat.shape[0]
        print('  Subsampled to ' + str(n) + ' tokens for training')
    # Normalize — compute in float32 chunks to save memory
    norm = ((flat.float() - mean.cpu()) / std.cpu()).half()
    del flat

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        total_l1 = 0
        batches = 0
        for i in range(0, n, batch_size):
            batch = norm[perm[i:i+batch_size]].float().cuda()
            recon, sparse = sae(batch)
            mse = F.mse_loss(recon, batch)

            if arch_name == 'btk':
                loss = mse
            else:
                l1 = sparse.abs().mean()
                loss = mse + l1_coeff * l1
                total_l1 += l1.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += mse.item()
            batches += 1

        avg_mse = total_loss / batches
        print('  epoch {}: mse={:.6f}'.format(epoch, avg_mse))
        sys.stdout.flush()

    return sae.eval()

def get_features_from_cache(hidden_fp32, sae, norm_mean, norm_std):
    tokens = (hidden_fp32 - norm_mean) / norm_std
    with torch.no_grad():
        _, acts = sae(tokens.unsqueeze(0))
    max_per_f = acts.squeeze(0).max(dim=0).values.cpu().numpy()
    return set(int(f) for f in np.where(max_per_f > 0)[0])

def run_agreement(sae, norm_mean, norm_std, dict_size, cache, game_data):
    """Quick agreement test."""
    all_hidden = cache['hidden']
    key_to_idx = cache['key_to_idx']
    mn = norm_mean.cuda()
    sd = norm_std.cuda()

    both = [0] * dict_size
    played_only = [0] * dict_size
    best_only = [0] * dict_size
    total = 0

    for g in game_data:
        for m in g.get('moves', []):
            best_uci = m.get('best_uci', '')
            if not best_uci: continue
            pk = m['fen'] + '|' + m['uci']
            bk = m['fen'] + '|' + best_uci
            if pk not in key_to_idx or bk not in key_to_idx: continue
            total += 1
            pf = get_features_from_cache(all_hidden[key_to_idx[pk]].float().cuda(), sae, mn, sd)
            bf = get_features_from_cache(all_hidden[key_to_idx[bk]].float().cuda(), sae, mn, sd)
            for fid in pf | bf:
                if fid in pf and fid in bf: both[fid] += 1
                elif fid in pf: played_only[fid] += 1
                else: best_only[fid] += 1

    # Analyze
    results = []
    for fid in range(dict_size):
        fe = both[fid] + played_only[fid] + best_only[fid]
        if fe < 20: continue
        results.append({
            'fid': fid, 'agreement': round(100 * both[fid] / fe, 1),
            'fire_rate': round(100 * fe / total, 1),
            'both': both[fid], 'played_only': played_only[fid], 'best_only': best_only[fid],
        })

    dist = {}
    for lo, hi, label in [(0, 20, 'pure_move'), (20, 40, 'mostly_move'), (40, 60, 'mixed'), (60, 80, 'mostly_pos'), (80, 101, 'pure_pos')]:
        dist[label] = sum(1 for r in results if lo <= r['agreement'] < hi)

    miss = sum(1 for r in results if r['agreement'] <= 30 and r['best_only'] > r['played_only'] * 1.5)
    over = sum(1 for r in results if r['agreement'] <= 30 and r['played_only'] > r['best_only'] * 1.5)
    noise = sum(1 for r in results if r['fire_rate'] > 20)

    return {
        'n_features': len(results), 'dist': dist, 'miss': miss, 'over': over,
        'noise': noise, 'total_positions': total, 'results': results,
    }

# =============================================
# Sweep configs
# =============================================

CONFIGS = [
    # dict_size, k, architecture
    (1024, 8, 'btk'),
    (1024, 16, 'btk'),
    (1024, 32, 'btk'),
    (2048, 16, 'btk'),  # already have this, but retrain for consistency
    (2048, 32, 'btk'),  # already have this
    (2048, 64, 'btk'),
    (4096, 32, 'btk'),
    (4096, 64, 'btk'),
    # V1 (L1) at best BTK scale for comparison
    (2048, None, 'v1'),
    # Gated at best BTK scale
    (2048, None, 'gated'),
]

# =============================================
# Main
# =============================================

# Load puzzle data
acts, mean, std = load_puzzle_data(20000)  # 20K to avoid OOM — enough for sweep comparison

# Load cached encoder activations and game data
print()
print('Loading agreement test data...')
cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
game_data = [json.load(open(GD + '/' + f)) for f in game_files]
print(str(len(game_data)) + ' games, ' + str(cache['hidden'].shape[0]) + ' cached pairs')

# Run sweep
all_results = {}
for dict_size, k, arch in CONFIGS:
    name = '{}_{}{}'.format(arch, dict_size, '_k' + str(k) if k else '')
    print()
    print('=' * 60)
    print('Training: ' + name)
    print('=' * 60)

    if arch == 'btk':
        sae = SAE_BatchTopK(1024, dict_size, k)
    elif arch == 'v1':
        sae = SAE_V1(1024, dict_size)
    elif arch == 'gated':
        sae = SAE_Gated(1024, dict_size)

    t0 = time.time()
    sae = train_sae(sae, acts, mean, std, arch)
    train_time = time.time() - t0
    print('  Trained in {:.0f}s'.format(train_time))

    # Save checkpoint
    ckpt_file = BASE + '/output/sweep_sae_' + name + '.pt'
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {'dict_size': dict_size, 'k': k, 'arch': arch},
        'normalization': {'mean': mean.numpy().tolist(), 'std': std.numpy().tolist()},
    }, ckpt_file)

    # Agreement test
    mn_cuda = mean.cuda()
    sd_cuda = std.clamp(min=1e-8).cuda()
    t0 = time.time()
    result = run_agreement(sae, mn_cuda, sd_cuda, dict_size, cache, game_data)
    agree_time = time.time() - t0

    all_results[name] = result
    d = result['dist']
    pure_pos = d.get('pure_pos', 0)
    pure_mov = d.get('pure_move', 0)
    clean = pure_pos + pure_mov
    clean_pct = round(100 * clean / result['n_features'], 0) if result['n_features'] else 0

    print('  Agreement ({:.0f}s): {} features, {}pos/{}mov/{}mix, miss={} over={} noise={} clean={}%'.format(
        agree_time, result['n_features'], pure_pos, pure_mov,
        d.get('mixed', 0) + d.get('mostly_move', 0) + d.get('mostly_pos', 0),
        result['miss'], result['over'], result['noise'], clean_pct))
    sys.stdout.flush()

    del sae
    torch.cuda.empty_cache()

# Summary
print()
print('=' * 80)
print('FULL SWEEP SUMMARY')
print('=' * 80)
print()
print('{:<20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6}'.format(
    'Config', 'Feats', 'PPos', 'PMov', 'Mixed', 'Miss', 'Over', 'Clean%', 'Noise'))
print('-' * 80)
for name in [n for _, _, _ in CONFIGS for n in ['{}_{}{}' .format(_[2], _[0], '_k' + str(_[1]) if _[1] else '')]]:
    if name not in all_results: continue
    r = all_results[name]
    d = r['dist']
    pp = d.get('pure_pos', 0)
    pm = d.get('pure_move', 0)
    mx = d.get('mixed', 0) + d.get('mostly_move', 0) + d.get('mostly_pos', 0)
    cl = round(100 * (pp + pm) / r['n_features'], 0) if r['n_features'] else 0
    print('{:<20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6.0f}% {:>6}'.format(
        name, r['n_features'], pp, pm, mx, r['miss'], r['over'], cl, r['noise']))

# Save
with open(BASE + '/output/full_sweep_results.json', 'w') as f:
    save = {n: {k: v for k, v in r.items() if k != 'results'} for n, r in all_results.items()}
    json.dump(save, f, indent=2)
print()
print('Saved to ' + BASE + '/output/full_sweep_results.json')
print('DONE')
