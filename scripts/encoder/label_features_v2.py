"""Label SAE features by showing Haiku the actual chess positions.

Two-phase:
1. Run all puzzles through SAE, save top 10 puzzle indices per feature
2. For each feature, show Haiku the actual boards + moves + firing squares

Filters out noise features (>40% fire rate).
Includes counter-examples (positions from same theme where feature doesn't fire).
"""
import json, math, sys, time, numpy as np, torch, chess, boto3
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/poc'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
SAE_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32.pt'
THEMES_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32_themes.json'
PUZZLE_FILE = BASE + '/data/lichess_puzzles_200k.jsonl'
OUTPUT_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32_labels_v2.json'
TOP_N_FILE = BASE + '/output/sae_feature_top_puzzles.json'

N_POSITIONS = 150000
FIRE_RATE_MAX = 0.4  # skip features firing on >40% of puzzles
TOP_PER_FEATURE = 10  # save top 10 puzzles per feature
LABEL_TOP_N = 5  # show 5 positions to Haiku

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
    for f_ in 'abcdefgh':
        SQ.append(f_ + str(r))

# --- Encoder ---
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
DICT_SIZE=cfg['dict_size']
print('Encoder + SAE loaded.')

# --- Load puzzles ---
print('Loading puzzles...')
puzzles = []
with open(PUZZLE_FILE) as f:
    for line in f:
        d = json.loads(line)
        moves = d['moves'].split() if isinstance(d['moves'], str) else d['moves']
        if len(moves) < 2: continue
        try:
            board = chess.Board(d['fen'])
            board.push_uci(moves[0])
            puzzle_fen = board.fen()
        except: continue
        best_move = moves[1]
        ft = tok(puzzle_fen)
        if ft is None or best_move not in M2A: continue
        themes = d['themes'] if isinstance(d['themes'], list) else d['themes'].split()
        puzzles.append({'fen': puzzle_fen, 'best': best_move, 'themes': themes, 'seq': ft + [M2A[best_move], 64]})
        if len(puzzles) >= N_POSITIONS: break
print(str(len(puzzles)) + ' puzzles loaded')
sys.stdout.flush()

# --- Phase 1: Find top puzzles per feature (skip if cached) ---
import heapq, os
if os.path.exists(TOP_N_FILE):
    print('Phase 1: SKIPPED — loading cached ' + TOP_N_FILE)
    feature_top = json.load(open(TOP_N_FILE))
    td = json.load(open(THEMES_FILE))
    feature_freq = {}
    for fid_s, data in td['feature_details'].items():
        if fid_s in feature_top:
            feature_freq[int(fid_s)] = data.get('freq', 0)
    del enc
    torch.cuda.empty_cache()
    print('Loaded ' + str(len(feature_top)) + ' features with top puzzles')
else:
    print('Phase 1: Finding top puzzles per feature...')
    feature_heaps = {fid: [] for fid in range(DICT_SIZE)}
    for i in range(0, len(puzzles), 64):
        batch = [p['seq'] for p in puzzles[i:i+64]]
        tens = torch.tensor(batch, dtype=torch.long, device='cuda')
        with torch.no_grad():
            h = enc(tens)
            tokens = (h[:, 1:78, :] - mn) / sd
            _, acts = sae(tokens.reshape(-1, 1024))
        acts_r = acts.reshape(len(batch), 77, DICT_SIZE).cpu().numpy()
        for b in range(len(batch)):
            pidx = i + b
            max_per_f = acts_r[b].max(axis=0)
            argmax_per_f = acts_r[b].argmax(axis=0)
            for fid in range(DICT_SIZE):
                if max_per_f[fid] > 0:
                    val = (float(max_per_f[fid]), pidx, int(argmax_per_f[fid]))
                    if len(feature_heaps[fid]) < TOP_PER_FEATURE:
                        heapq.heappush(feature_heaps[fid], val)
                    elif val[0] > feature_heaps[fid][0][0]:
                        heapq.heapreplace(feature_heaps[fid], val)
        if (i // 64) % 200 == 0 and i > 0:
            print('  ' + str(i) + '/' + str(len(puzzles)))
            sys.stdout.flush()
    feature_top = {}
    feature_freq = {}
    for fid in range(DICT_SIZE):
        heap = feature_heaps[fid]
        if not heap: continue
        sorted_top = sorted(heap, key=lambda x: -x[0])
        feature_top[str(fid)] = [{'activation': round(t[0], 2), 'puzzle_idx': t[1], 'token_idx': t[2]} for t in sorted_top]
    td = json.load(open(THEMES_FILE))
    for fid_s, data in td['feature_details'].items():
        if fid_s in feature_top:
            feature_freq[int(fid_s)] = data.get('freq', 0)
    del feature_heaps, enc
    torch.cuda.empty_cache()
    with open(TOP_N_FILE, 'w') as f:
        json.dump(feature_top, f)
print('Saved top puzzles per feature to ' + TOP_N_FILE)
sys.stdout.flush()

# --- Phase 2: Label with Haiku ---
print()
print('Phase 2: Labeling with Haiku...')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
MODEL = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'

def call_haiku(prompt):
    resp = bedrock.converse(
        modelId=MODEL,
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={'maxTokens': 200, 'temperature': 0}
    )
    return resp['output']['message']['content'][0]['text'].strip()

labels = {}
errors = 0
# Filter: only label features with fire rate < 40% and with top puzzles
to_label = []
for fid_s, top_list in feature_top.items():
    fid = int(fid_s)
    freq = feature_freq.get(fid, 0)
    fire_rate = freq / len(puzzles) if len(puzzles) > 0 else 0
    if fire_rate > FIRE_RATE_MAX:
        labels[fid_s] = 'noise (fires on ' + str(round(fire_rate * 100)) + '% of positions)'
        continue
    if len(top_list) < 3:
        labels[fid_s] = 'rare (< 3 activations)'
        continue
    to_label.append((fid_s, top_list))

MAX_LABELS = 100
# Sort by frequency descending — label the most common features first
to_label.sort(key=lambda x: -feature_freq.get(int(x[0]), 0))
if len(to_label) > MAX_LABELS:
    to_label = to_label[:MAX_LABELS]
print(str(len(to_label)) + ' features to label (' + str(len(labels)) + ' pre-filtered as noise/rare)')
sys.stdout.flush()

for idx, (fid_s, top_list) in enumerate(to_label):
    # Build prompt with actual positions
    examples = []
    for entry in top_list[:LABEL_TOP_N]:
        pidx = entry['puzzle_idx']
        tok_idx = entry['token_idx']
        activation = entry['activation']
        p = puzzles[pidx]

        board = chess.Board(p['fen'])
        board_str = str(board)

        # What square does the feature fire on?
        fire_square = ''
        fire_piece = ''
        if 1 <= tok_idx <= 64:
            fire_square = SQ[tok_idx - 1]
            pc = board.piece_at(chess.parse_square(fire_square))
            fire_piece = pc.symbol() if pc else 'empty'

        # What does the best move do?
        try:
            best_san = board.san(chess.Move.from_uci(p['best']))
        except:
            best_san = p['best']

        # Does the best move involve the firing square?
        move_from = p['best'][:2]
        move_to = p['best'][2:4]
        relation = ''
        if fire_square == move_from:
            relation = 'piece MOVES FROM this square'
        elif fire_square == move_to:
            relation = 'piece MOVES TO this square'
        else:
            # Check if the move attacks the firing square
            try:
                board_after = board.copy()
                board_after.push_uci(p['best'])
                if board_after.is_attacked_by(board.turn, chess.parse_square(fire_square)):
                    relation = 'move ATTACKS this square'
            except:
                pass

        # Simple eval estimate from piece count (no Stockfish available)
        white_mat = sum({1:1,2:3,3:3,4:5,5:9}.get(board.piece_at(sq_i).piece_type, 0)
                       for sq_i in chess.SQUARES if board.piece_at(sq_i) and board.piece_at(sq_i).color == chess.WHITE)
        black_mat = sum({1:1,2:3,3:3,4:5,5:9}.get(board.piece_at(sq_i).piece_type, 0)
                       for sq_i in chess.SQUARES if board.piece_at(sq_i) and board.piece_at(sq_i).color == chess.BLACK)
        mat_balance = 'White +' + str(white_mat - black_mat) if white_mat > black_mat else ('Black +' + str(black_mat - white_mat) if black_mat > white_mat else 'Equal material')
        side = 'White' if board.turn == chess.WHITE else 'Black'

        examples.append(
            'Position ' + str(len(examples)+1) + ' (' + side + ' to move, ' + mat_balance + '):\n'
            + board_str + '\n'
            + 'Best move: ' + best_san + ' (' + p['best'] + ')\n'
            + 'Feature fires on: ' + fire_square + ' (' + fire_piece + ')' + (' — ' + relation if relation else '')
        )

    prompt = (
        'You are analyzing features from a chess neural network. '
        'Each feature detects a specific pattern in chess positions.\n\n'
        'Below are 5 chess positions where Feature F' + fid_s + ' fires most strongly. '
        'For each position, I show the board, the best move, which square the feature fires on, '
        'and what piece is there.\n\n'
        + '\n\n'.join(examples)
        + '\n\nWhat chess pattern does this feature detect? '
        'Look at what the firing squares have in common, how they relate to the best move, '
        'and what the positions share.\n\n'
        'Write TWO things:\n'
        '1. LABEL: A short (5-10 word) description of the pattern. Be specific — '
        '"back rank mate with rook" is better than "tactical position". '
        'If the feature fires on the piece that moves, say so. '
        'If it fires on the target square, say so.\n'
        '2. CONFIDENCE: high/medium/low. Say "low" if the positions seem unrelated or '
        'the pattern is vague. It is completely fine to say low — an honest "unclear" is '
        'better than a forced label.\n\n'
        'Format exactly:\nLABEL: <your label>\nCONFIDENCE: <high/medium/low>'
    )

    try:
        resp = call_haiku(prompt)
        # Parse label and confidence
        label = ''
        confidence = 'unknown'
        for line in resp.split('\n'):
            line = line.strip()
            if line.upper().startswith('LABEL:'):
                label = line[6:].strip().strip('"\'')
            elif line.upper().startswith('CONFIDENCE:'):
                confidence = line[11:].strip().lower()
        if not label:
            label = resp.split('\n')[0].strip('"\'').strip()
        labels[fid_s] = {'label': label, 'confidence': confidence}
    except Exception as e:
        labels[fid_s] = 'error: ' + str(e)[:80]
        errors += 1
        if errors == 1:
            print('  FIRST ERROR at F' + fid_s + ': ' + str(e)[:100])
            sys.stdout.flush()

    # Progress + spot check every 25
    if (idx + 1) % 25 == 0:
        n_good = len([v for v in labels.values() if isinstance(v, dict)])
        n_high = len([v for v in labels.values() if isinstance(v, dict) and v.get('confidence') == 'high'])
        print('  ' + str(idx+1) + '/' + str(len(to_label)) + ' (labeled=' + str(n_good) + ', high_conf=' + str(n_high) + ', err=' + str(errors) + ')')
        # Show last label with top 3 examples
        recent = [(k, v) for k, v in labels.items() if isinstance(v, dict)]
        if recent:
            last_fid, last_lbl = recent[-1]
            print('    F' + last_fid + ': ' + last_lbl['label'] + ' [' + last_lbl['confidence'] + ']')
            top_entries = feature_top.get(last_fid, [])[:3]
            for ei, entry in enumerate(top_entries):
                pidx = entry.get('puzzle_idx', 0)
                tok_idx = entry.get('token_idx', 0)
                p = puzzles[pidx] if pidx < len(puzzles) else None
                if not p: continue
                board = chess.Board(p['fen'])
                sq = SQ[tok_idx - 1] if 1 <= tok_idx <= 64 else '?'
                pc = board.piece_at(chess.parse_square(sq)) if sq != '?' else None
                pc_str = pc.symbol() if pc else '.'
                try:
                    best_san = board.san(chess.Move.from_uci(p['best']))
                except:
                    best_san = p['best']
                print('      ex' + str(ei+1) + ': ' + best_san + ' | fires=' + sq + '(' + pc_str + ') | ' + ','.join(p['themes'][:3]) + ' | ' + p['fen'].split(' ')[0][:30])
        sys.stdout.flush()

    # Save incrementally every 100
    if (idx + 1) % 100 == 0:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({'config': {'dict_size': DICT_SIZE, 'k': cfg['k'], 'model': MODEL, 'partial': True}, 'labels': labels}, f, indent=2)

    time.sleep(0.15)

# Final save + stats
n_labeled = len([v for v in labels.values() if isinstance(v, dict)])
n_high = len([v for v in labels.values() if isinstance(v, dict) and v.get('confidence') == 'high'])
n_med = len([v for v in labels.values() if isinstance(v, dict) and v.get('confidence') == 'medium'])
n_low = len([v for v in labels.values() if isinstance(v, dict) and v.get('confidence') == 'low'])
n_noise = len([v for v in labels.values() if isinstance(v, str) and v.startswith('noise')])
n_rare = len([v for v in labels.values() if isinstance(v, str) and v.startswith('rare')])
with open(OUTPUT_FILE, 'w') as f:
    json.dump({'config': {'dict_size': DICT_SIZE, 'k': cfg['k'], 'model': MODEL}, 'labels': labels, 'stats': {'labeled': n_labeled, 'high': n_high, 'medium': n_med, 'low': n_low, 'noise': n_noise, 'rare': n_rare, 'errors': errors}}, f, indent=2)

print()
print('Done. ' + str(n_labeled) + ' labeled (high=' + str(n_high) + ', med=' + str(n_med) + ', low=' + str(n_low) + '), ' + str(n_noise) + ' noise, ' + str(n_rare) + ' rare, ' + str(errors) + ' errors')
print('Saved to ' + OUTPUT_FILE)

# Show sample labels for the game-important features
print()
print('Game-important features:')
for fid_s in ['2012', '492', '175', '1459', '606', '245', '1790', '1738', '822', '1915']:
    v = labels.get(fid_s, '(missing)')
    if isinstance(v, dict):
        print('  F' + fid_s + ': ' + v['label'] + ' [' + v['confidence'] + ']')
    else:
        print('  F' + fid_s + ': ' + str(v))
