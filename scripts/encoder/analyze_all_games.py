"""Run all moves from cabbagelover5566's games through the puzzle SAE.
For each move: encode (position, played_move) and (position, best_move).
Save per-game feature activations + diffs. Aggregate patterns across all games."""
import json, math, numpy as np, torch, sys, chess
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict

BASE = '/home/ec2-user/SageMaker/poc'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
GAMES_FILE = BASE + '/data/cabbagelover5566_games.json'
SAE_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32.pt'
LABELS_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32_labels.json'
OUTPUT_DIR = BASE + '/output/game_analysis'

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

SQ = []
for r in range(8, 0, -1):
    for file_ in 'abcdefgh':
        SQ.append(file_ + str(r))

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
del pr;enc=enc.cuda().eval()
print('Encoder loaded.')

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

ckpt=torch.load(SAE_FILE,map_location='cpu',weights_only=False)
cfg=ckpt['config'];sae=SAE(1024,cfg['dict_size'],cfg['k'])
sae.load_state_dict(ckpt['model_state_dict']);sae=sae.cuda().eval()
mn=torch.tensor(ckpt['normalization']['mean'],dtype=torch.float32,device='cuda')
sd=torch.tensor(ckpt['normalization']['std'],dtype=torch.float32,device='cuda').clamp(min=1e-8)
DICT_SIZE = cfg['dict_size']
print('SAE loaded.')

# Load labels
labels = {}
try:
    ld = json.load(open(LABELS_FILE))
    labels = ld.get('labels', {})
    n_good = len([v for v in labels.values() if v != 'unknown' and not v.startswith('error')])
    print('Labels loaded: ' + str(n_good) + ' good labels')
except:
    print('No labels file yet — will use feature IDs')

def get_label(fid):
    lbl = labels.get(str(fid), '')
    if lbl and lbl != 'unknown' and not lbl.startswith('error'):
        return lbl
    return 'F' + str(fid)

def encode_move(fen, move_uci):
    ft = tok(fen)
    if ft is None or move_uci not in M2A: return None
    seq = ft + [M2A[move_uci], 64]
    with torch.no_grad():
        h = enc(torch.tensor([seq], dtype=torch.long, device='cuda'))
        tokens = (h[0, 1:78, :] - mn) / sd
        _, acts = sae(tokens.unsqueeze(0))
    return acts.squeeze(0).cpu().numpy()  # [77, DICT_SIZE]

def top_features(acts, n=5):
    """Get top n features by max activation across tokens."""
    max_per_f = acts.max(axis=0)
    top_idx = np.argsort(-max_per_f)[:n]
    result = []
    for fid in top_idx:
        if max_per_f[fid] <= 0: break
        # Find which token fires strongest
        tok_acts = acts[:, fid]
        best_tok = int(np.argmax(tok_acts))
        sq = ''
        if 1 <= best_tok <= 64:
            sq = SQ[best_tok - 1]
        result.append({'fid': int(fid), 'label': get_label(fid), 'strength': round(float(max_per_f[fid]), 2), 'square': sq})
    return result

def diff_features(acts_good, acts_bad, n=5):
    """Features that good move has more than bad move."""
    mg = acts_good.max(axis=0)
    mb = acts_bad.max(axis=0)
    diff = mg - mb
    top_idx = np.argsort(-diff)[:n]
    result = []
    for fid in top_idx:
        if diff[fid] <= 0: break
        result.append({'fid': int(fid), 'label': get_label(fid), 'diff': round(float(diff[fid]), 2)})
    return result

# Load games
print('Loading games...')
with open(GAMES_FILE) as f:
    games = json.load(f)
print(str(len(games)) + ' games loaded')
sys.stdout.flush()

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aggregate counters
all_played_features = Counter()    # feature → count across all moves
all_best_features = Counter()      # feature → count across all best moves
all_missed_features = Counter()    # features good has that played doesn't → "what you miss"
all_bad_features = Counter()       # features played has that good doesn't → "what you do wrong"
blunder_missed = Counter()         # missed features on blunders/mistakes only
games_processed = 0
moves_processed = 0
moves_with_best = 0

for gi, game in enumerate(games):
    moves = game.get('moves', [])
    if not moves: continue

    game_id = game.get('id', str(gi))
    game_results = []

    board = chess.Board()
    for mi, move_data in enumerate(moves):
        san = move_data.get('san', '')
        uci = move_data.get('uci', '')
        best_uci = move_data.get('best_move', '')
        classification = move_data.get('classification', '')
        eval_val = move_data.get('eval', None)
        fen = board.fen()

        if not uci or uci not in M2A:
            try:
                board.push_san(san)
            except:
                pass
            continue

        # Encode played move
        acts_played = encode_move(fen, uci)
        if acts_played is None:
            try:
                board.push_san(san)
            except:
                pass
            continue

        played_top = top_features(acts_played, n=5)
        for ft in played_top:
            all_played_features[ft['fid']] += 1

        move_result = {
            'ply': mi + 1,
            'san': san,
            'uci': uci,
            'fen': fen,
            'classification': classification,
            'eval': eval_val,
            'played_features': played_top,
        }

        # Encode best move if available
        if best_uci and best_uci in M2A and best_uci != uci:
            acts_best = encode_move(fen, best_uci)
            if acts_best is not None:
                best_top = top_features(acts_best, n=5)
                missed = diff_features(acts_best, acts_played, n=5)
                bad_pattern = diff_features(acts_played, acts_best, n=5)

                for ft in best_top:
                    all_best_features[ft['fid']] += 1
                for ft in missed:
                    all_missed_features[ft['fid']] += 1
                    if classification in ('blunder', 'mistake'):
                        blunder_missed[ft['fid']] += 1
                for ft in bad_pattern:
                    all_bad_features[ft['fid']] += 1

                move_result['best_uci'] = best_uci
                move_result['best_features'] = best_top
                move_result['missed'] = missed
                move_result['bad_pattern'] = bad_pattern
                moves_with_best += 1

        game_results.append(move_result)
        moves_processed += 1

        try:
            board.push_san(san)
        except:
            pass

    # Save per-game results
    if game_results:
        game_output = {
            'game_id': game_id,
            'white': game.get('white', ''),
            'black': game.get('black', ''),
            'result': game.get('result', ''),
            'moves': game_results,
        }
        with open(OUTPUT_DIR + '/game_' + str(game_id) + '.json', 'w') as f:
            json.dump(game_output, f, indent=2)

    games_processed += 1
    if games_processed % 10 == 0:
        print('  ' + str(games_processed) + '/' + str(len(games)) + ' games, ' + str(moves_processed) + ' moves')
        sys.stdout.flush()

# Save aggregate results
print()
print('=' * 60)
print('AGGREGATE RESULTS')
print('=' * 60)
print(str(games_processed) + ' games, ' + str(moves_processed) + ' moves, ' + str(moves_with_best) + ' with best move comparison')
print()

print('TOP 15 FEATURES ON YOUR PLAYED MOVES (what you do):')
for fid, count in all_played_features.most_common(15):
    print('  ' + get_label(fid) + ' (' + str(count) + ' moves)')

print()
print('TOP 15 MISSED FEATURES (what good moves have that yours dont):')
for fid, count in all_missed_features.most_common(15):
    print('  ' + get_label(fid) + ' (' + str(count) + ' moves)')

print()
print('TOP 15 BAD PATTERNS (what your moves have that good moves dont):')
for fid, count in all_bad_features.most_common(15):
    print('  ' + get_label(fid) + ' (' + str(count) + ' moves)')

print()
print('TOP 15 MISSED ON BLUNDERS/MISTAKES ONLY:')
for fid, count in blunder_missed.most_common(15):
    print('  ' + get_label(fid) + ' (' + str(count) + ' blunders/mistakes)')

# Save aggregate
agg = {
    'games_processed': games_processed,
    'moves_processed': moves_processed,
    'moves_with_best': moves_with_best,
    'played_features': dict(all_played_features.most_common(50)),
    'missed_features': dict(all_missed_features.most_common(50)),
    'bad_patterns': dict(all_bad_features.most_common(50)),
    'blunder_missed': dict(blunder_missed.most_common(50)),
}
with open(OUTPUT_DIR + '/aggregate.json', 'w') as f:
    json.dump(agg, f, indent=2)

print()
print('Saved ' + str(games_processed) + ' game files to ' + OUTPUT_DIR + '/')
print('Saved aggregate.json')
print('DONE')
