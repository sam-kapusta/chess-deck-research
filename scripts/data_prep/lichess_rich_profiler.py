"""Lichess profiler v2: rich examples with SAN, piece type, capture, check, phase.

Runs encoder + SAE on 100K Lichess positions, stores rich per-feature examples,
then labels with Sonnet.
"""
import json, os, sys, time, math, numpy as np, torch, chess
import torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset

BASE = '/home/ec2-user/SageMaker/poc'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
SAE_FILE = BASE + '/output/sae_puzzle_pertoken_2048_k32.pt'
OUTPUT_FILE = BASE + '/output/lichess_rich_profiles.json'
LABELS_FILE = BASE + '/output/lichess_rich_labels.json'

N_POSITIONS = 100000

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

ckpt = torch.load(SAE_FILE, map_location='cpu', weights_only=False)
cfg = ckpt['config']; sae = SAE(1024, cfg['dict_size'], cfg['k'])
sae.load_state_dict(ckpt['model_state_dict']); sae = sae.cuda().eval()
mn = torch.tensor(ckpt['normalization']['mean'], dtype=torch.float32, device='cuda')
sd = torch.tensor(ckpt['normalization']['std'], dtype=torch.float32, device='cuda').clamp(min=1e-8)
DICT_SIZE = cfg['dict_size']
print('Encoder + SAE loaded.')

def get_features(fen, move_uci):
    ft = tok(fen)
    if ft is None or move_uci not in M2A: return None
    seq = ft + [M2A[move_uci], 64]
    with torch.no_grad():
        h = enc(torch.tensor([seq], dtype=torch.long, device='cuda'))
        tokens = (h[0, 1:78, :] - mn) / sd
        _, acts = sae(tokens.unsqueeze(0))
    max_per_f = acts.squeeze(0).max(dim=0).values.cpu().numpy()
    return set(int(f) for f in np.where(max_per_f > 0)[0])

def enrich_move(fen, move_uci, cp, is_best, best_cp=None):
    """Convert UCI move to rich description using python-chess."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        san = board.san(move)

        # Piece moved
        from_sq = move.from_square
        pc = board.piece_at(from_sq)
        piece = chess.piece_name(pc.piece_type) if pc else ''

        # Properties
        is_capture = board.is_capture(move)
        board.push(move)
        is_check = board.is_check()
        board.pop()

        # Phase estimate from piece count
        total_pieces = len([s for s in chess.SQUARES if board.piece_at(s)])
        if total_pieces > 28:
            phase = 'opening'
        elif total_pieces > 16:
            phase = 'middlegame'
        else:
            phase = 'endgame'

        tag = 'best' if is_best else 'alt'
        cap_tag = 'x' if is_capture else ''
        chk_tag = '+' if is_check else ''
        loss_tag = ''
        if not is_best and best_cp is not None:
            loss = best_cp - cp
            loss_tag = ', loss=' + str(loss) + 'cp'
        return fen + ' | ' + san + ' (' + tag + ', ' + piece + cap_tag + chk_tag + ', ' + phase + ', eval=' + str(cp) + 'cp' + loss_tag + ')'
    except:
        return fen + ' | ' + move_uci + ' (' + ('best' if is_best else 'alt') + ', eval=' + str(cp) + 'cp)'

# Stream Lichess evaluations
print('Streaming Lichess position evaluations...')
ds = load_dataset('Lichess/chess-position-evaluations', split='train', streaming=True)

# Per-feature stats
fire_count = np.zeros(DICT_SIZE)
feature_examples = defaultdict(list)  # fid -> [rich description strings]

# Per-feature: move type counts for stats
feature_stats = defaultdict(lambda: {
    'n': 0, 'captures': 0, 'checks': 0, 'knight': 0, 'bishop': 0,
    'rook': 0, 'queen': 0, 'king': 0, 'pawn': 0,
    'opening': 0, 'middlegame': 0, 'endgame': 0,
    'best': 0, 'alt': 0,
})

n_processed = 0
current_fen = None
current_moves = []

print('Processing positions...')
sys.stdout.flush()
t0 = time.time()

for row in ds:
    fen = row['fen']
    cp = row.get('cp')
    line = row.get('line', '')
    if cp is None or not line: continue
    move_uci = line.split()[0]
    if move_uci not in M2A: continue

    if fen != current_fen:
        if current_fen and len(current_moves) >= 2:
            sorted_moves = sorted(current_moves, key=lambda x: -x[1])
            best_move = sorted_moves[0][0]
            best_cp = sorted_moves[0][1]

            best_feats = get_features(current_fen, best_move)
            if best_feats is not None:
                for alt_move, alt_cp in sorted_moves[1:]:
                    alt_feats = get_features(current_fen, alt_move)
                    if alt_feats is not None:
                        # Parse board for phase
                        try:
                            board = chess.Board(current_fen)
                            total_pieces = len([s for s in chess.SQUARES if board.piece_at(s)])
                            phase = 'opening' if total_pieces > 28 else ('middlegame' if total_pieces > 16 else 'endgame')
                        except:
                            phase = 'unknown'

                        for fid in best_feats | alt_feats:
                            fire_count[fid] += 1
                            in_best = fid in best_feats
                            in_alt = fid in alt_feats
                            fs = feature_stats[fid]
                            fs['n'] += 1

                            if in_best:
                                fs['best'] += 1
                                if len(feature_examples[fid]) < 50:
                                    feature_examples[fid].append(
                                        enrich_move(current_fen, best_move, best_cp, True, best_cp))
                                # Count piece/capture/check for best move
                                try:
                                    b = chess.Board(current_fen)
                                    m = chess.Move.from_uci(best_move)
                                    pc = b.piece_at(m.from_square)
                                    if pc: fs[chess.piece_name(pc.piece_type)] += 1
                                    if b.is_capture(m): fs['captures'] += 1
                                    b.push(m)
                                    if b.is_check(): fs['checks'] += 1
                                    b.pop()
                                except: pass

                            if in_alt:
                                fs['alt'] += 1
                                if len(feature_examples[fid]) < 50:
                                    feature_examples[fid].append(
                                        enrich_move(current_fen, alt_move, alt_cp, False, best_cp))

                            fs[phase] += 1

                        n_processed += 1
                        break

        current_fen = fen
        current_moves = []

    current_moves.append((move_uci, cp))

    if n_processed % 5000 == 0 and n_processed > 0:
        elapsed = time.time() - t0
        print('  ' + str(n_processed) + '/' + str(N_POSITIONS) + ' ({:.0f}/sec)'.format(n_processed/elapsed))
        sys.stdout.flush()

    if n_processed >= N_POSITIONS:
        break

print('Processed: ' + str(n_processed))
alive = int((fire_count > 0).sum())
print('Alive features: ' + str(alive))

# Build profiles
print('Building profiles...')
profiles = {}
for fid in range(DICT_SIZE):
    if fire_count[fid] < 20: continue
    fs = feature_stats.get(fid, {})
    n = fs.get('n', 0) or 1
    fr = round(100 * fire_count[fid] / n_processed, 2)

    profile = {
        'fire_rate': fr,
        'n_fires': int(fire_count[fid]),
        'examples': feature_examples.get(fid, [])[:20],
    }

    # Stats (only report significant signals)
    for key in ['captures', 'checks']:
        pct = round(100 * fs.get(key, 0) / n, 1)
        if pct > 25 or (key == 'checks' and pct > 10):
            profile[key] = str(pct) + '%'

    for piece in ['knight', 'bishop', 'rook', 'queen', 'king', 'pawn']:
        pct = round(100 * fs.get(piece, 0) / n, 1)
        if pct > 25:
            profile['piece_' + piece] = str(pct) + '%'

    for phase in ['opening', 'middlegame', 'endgame']:
        pct = round(100 * fs.get(phase, 0) / n, 1)
        if pct > 50:
            profile['phase_' + phase] = str(pct) + '%'

    # Best vs alt ratio
    best_pct = round(100 * fs.get('best', 0) / n, 1)
    alt_pct = round(100 * fs.get('alt', 0) / n, 1)
    if abs(best_pct - alt_pct) > 10:
        profile['best_pct'] = str(best_pct) + '%'
        profile['alt_pct'] = str(alt_pct) + '%'

    profiles[str(fid)] = profile

print(str(len(profiles)) + ' features profiled')

with open(OUTPUT_FILE, 'w') as f:
    json.dump(profiles, f, indent=2)
print('Saved to ' + OUTPUT_FILE)
sys.stdout.flush()

# Free GPU
del enc
torch.cuda.empty_cache()

# Label with Sonnet
print()
print('Labeling with Sonnet 4.6...')
import boto3
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
MODEL = 'us.anthropic.claude-sonnet-4-6'

to_label = [(fid_s, p) for fid_s, p in profiles.items() if p['fire_rate'] < 40]
to_label.sort(key=lambda x: -x[1]['n_fires'])
print(str(len(to_label)) + ' features to label')

labels = {}
for idx, (fid_s, p) in enumerate(to_label):
    examples = p.get('examples', [])[:10]
    examples_str = '\n'.join('  ' + e for e in examples) if examples else '  (none)'

    stats = ['Fire rate: ' + str(p['fire_rate']) + '%']
    for key in ['captures', 'checks', 'piece_knight', 'piece_bishop', 'piece_rook',
                'piece_queen', 'piece_king', 'piece_pawn',
                'phase_opening', 'phase_middlegame', 'phase_endgame',
                'best_pct', 'alt_pct']:
        if key in p:
            stats.append(key.replace('piece_', '').replace('phase_', '').replace('_pct', '') + ': ' + str(p[key]))

    # Add puzzle theme enrichment if available
    themes_file = BASE + '/output/sae_puzzle_pertoken_2048_k32_themes.json'
    theme_str = ''
    try:
        td_all = json.load(open(themes_file)).get('feature_details', {})
        td = td_all.get(fid_s, {})
        enriched = sorted(td.get('enriched', {}).items(), key=lambda x: -x[1]['enrichment'])
        if enriched:
            theme_str = 'Puzzle themes: ' + ', '.join(t + ' (' + str(round(v['enrichment'], 1)) + 'x)' for t, v in enriched[:5])
    except:
        pass

    prompt = (
        'You are a chess expert. A neural network feature fires on specific chess moves. '
        'Below are 10 example positions (FEN) with the move played and Stockfish evaluation.\n\n'
        'POSITIONS WHERE THIS FEATURE FIRES:\n' + examples_str + '\n\n'
        'STATISTICS:\n' + '\n'.join('  ' + s for s in stats) + '\n'
        + ('  ' + theme_str + '\n' if theme_str else '') +
        '\nLook at the FENs. What chess concept connects these positions and moves?\n\n'
        'EXPLANATION: 2-3 sentences. Reference specific positions.\n'
        'LABEL: 3-8 word description of the move quality. Write "unclear" if no pattern.\n'
        'CONFIDENCE: high/medium/low'
    )

    try:
        resp = bedrock.converse(modelId=MODEL, messages=[{'role':'user','content':[{'text':prompt}]}],
                                inferenceConfig={'maxTokens':300,'temperature':0})
        text = resp['output']['message']['content'][0]['text']
        explanation = label = ''
        confidence = 'unknown'
        for line in text.split('\n'):
            s = line.strip()
            clean = s.replace('**', '').strip()
            if clean.upper().startswith('EXPLANATION:'): explanation = clean[12:].strip()
            elif clean.upper().startswith('LABEL:'): label = clean[6:].strip().strip('"\'')
            elif clean.upper().startswith('CONFIDENCE:'): confidence = clean[11:].strip().lower().split()[0]
        if not label: label = 'unclear'
        labels[fid_s] = {'label': label, 'confidence': confidence, 'explanation': explanation}
    except Exception as e:
        labels[fid_s] = {'label': 'ERROR', 'confidence': 'unknown', 'explanation': str(e)[:100]}

    if (idx + 1) % 20 == 0:
        n_high = sum(1 for v in labels.values() if v['confidence'] == 'high')
        n_med = sum(1 for v in labels.values() if v['confidence'] == 'medium')
        n_unclear = sum(1 for v in labels.values() if v['label'] == 'unclear')
        print('  ' + str(idx+1) + '/' + str(len(to_label)) +
              ' (high=' + str(n_high) + ' med=' + str(n_med) + ' unclear=' + str(n_unclear) + ')')
        last = list(labels.items())[-1]
        print('    F' + last[0] + ': ' + last[1]['label'] + ' [' + last[1]['confidence'] + ']')
        sys.stdout.flush()

    time.sleep(0.2)

with open(LABELS_FILE, 'w') as f:
    json.dump(labels, f, indent=2)

n_high = sum(1 for v in labels.values() if v['confidence'] == 'high')
n_med = sum(1 for v in labels.values() if v['confidence'] == 'medium')
n_low = sum(1 for v in labels.values() if v['confidence'] == 'low')
n_unclear = sum(1 for v in labels.values() if v['label'] == 'unclear')
print()
print('Done: ' + str(len(labels)) + ' labeled')
print('  high=' + str(n_high) + ' med=' + str(n_med) + ' low=' + str(n_low) + ' unclear=' + str(n_unclear))
print('Saved to ' + LABELS_FILE)
print('DONE')
