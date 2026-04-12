"""Step 1: Cache encoder activations for all (FEN, move) pairs from game analysis.

Run encoder ONCE, save hidden states. Then SAE sweep is instant.
"""
import json, os, sys, time, math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
GD = BASE + '/output/game_analysis'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
CACHE_FILE = BASE + '/output/encoder_activation_cache.pt'

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

print('Loading encoder...')
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
print('Encoder loaded.')

# Step 1: Collect all unique (FEN, move) pairs
print('Collecting (FEN, move) pairs from game analysis...')
pairs = {}  # key -> (fen, move_uci, list of (game_idx, move_idx, is_best))
game_meta = []  # per-game metadata

game_files = sorted([f for f in os.listdir(GD) if f.startswith('game_') and f.endswith('.json')])
total_pairs = 0
skipped = 0

for gi, gf in enumerate(game_files):
    g = json.load(open(GD + '/' + gf))
    moves = g.get('moves', [])
    game_meta.append({'file': gf, 'n_moves': len(moves)})

    for mi, m in enumerate(moves):
        fen = m['fen']
        uci = m.get('uci', '')
        best_uci = m.get('best_uci', m.get('bestMove', ''))

        # Played move
        if uci and uci in M2A:
            key = fen + '|' + uci
            if key not in pairs:
                pairs[key] = {'fen': fen, 'move': uci}
            total_pairs += 1
        else:
            skipped += 1

        # Best move
        if best_uci and best_uci in M2A:
            key = fen + '|' + best_uci
            if key not in pairs:
                pairs[key] = {'fen': fen, 'move': best_uci}
            total_pairs += 1

print('Total (FEN, move) references: ' + str(total_pairs))
print('Unique pairs: ' + str(len(pairs)))
print('Skipped (bad UCI): ' + str(skipped))
sys.stdout.flush()

# Step 2: Batch encode all unique pairs
print('Encoding ' + str(len(pairs)) + ' unique pairs in batches...')
BATCH_SIZE = 64

pair_keys = list(pairs.keys())
pair_list = [pairs[k] for k in pair_keys]

# Tokenize all
seqs = []
valid_indices = []
for i, p in enumerate(pair_list):
    ft = tok(p['fen'])
    if ft is None:
        continue
    seq = ft + [M2A[p['move']], 64]
    seqs.append(seq)
    valid_indices.append(i)

print('Valid sequences: ' + str(len(seqs)) + ' / ' + str(len(pair_list)))
sys.stdout.flush()

# Batch encode
all_hidden = torch.zeros(len(seqs), 77, DIM, dtype=torch.float16)  # float16 to save memory
t0 = time.time()

with torch.no_grad():
    for batch_start in range(0, len(seqs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(seqs))
        batch = torch.tensor(seqs[batch_start:batch_end], dtype=torch.long, device='cuda')
        h = enc(batch)  # [B, 79, 1024]
        all_hidden[batch_start:batch_end] = h[:, 1:78, :].cpu().half()

        if (batch_start // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - t0
            pct = 100 * batch_end / len(seqs)
            rate = batch_end / elapsed if elapsed > 0 else 0
            print('  {}/{} ({:.1f}%) {:.0f} pairs/sec'.format(batch_end, len(seqs), pct, rate))
            sys.stdout.flush()

elapsed = time.time() - t0
print('Encoding done in {:.1f}s ({:.0f} pairs/sec)'.format(elapsed, len(seqs) / elapsed))

# Step 3: Build index mapping pair_key -> hidden_index
key_to_idx = {}
for i, vi in enumerate(valid_indices):
    key_to_idx[pair_keys[vi]] = i

# Step 4: Save cache
print('Saving cache...')
torch.save({
    'hidden': all_hidden,  # [N, 77, 1024] float16
    'key_to_idx': key_to_idx,  # pair_key -> index into hidden
    'pair_keys': pair_keys,
    'n_games': len(game_files),
    'n_pairs': len(seqs),
}, CACHE_FILE)

size_mb = os.path.getsize(CACHE_FILE) / 1024 / 1024
print('Saved to ' + CACHE_FILE + ' ({:.0f} MB)'.format(size_mb))
print('DONE')
