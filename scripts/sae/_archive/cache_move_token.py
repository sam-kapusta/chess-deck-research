#!/usr/bin/env python3
"""Extract move-token activations from blunder positions. Small cache (~400MB).

Reads blunder_positions.json, encodes each (FEN, blunder_move) pair,
extracts only hidden[77] (the move token), saves as [N, 1024] float16.

Usage:
    python3 cache_move_token.py
    python3 cache_move_token.py --use-best  # best moves instead
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
POSITIONS = BASE + '/cache/blunder_positions.json'

_C = list('0123456789abcdefghpnrkqPBNRQKw.')
_I = {c: i for i, c in enumerate(_C)}
_S = frozenset('12345678')

def tok(fen):
    p = fen.split(' ')
    while len(p) < 6:
        if len(p) == 4: p.append('0')
        elif len(p) == 5: p.append('1')
        else: p.append('-')
    b, s, c, e, h, f = p[:6]; b = s + b.replace('/', ''); ix = []
    for ch in b:
        if ch in _S: ix.extend(int(ch) * [_I['.']])
        elif ch in _I: ix.append(_I[ch])
        else: return None
    if c == '-': ix.extend(4 * [_I['.']])
    else:
        for ch in c:
            if ch not in _I: return None
            ix.append(_I[ch])
        ix.extend((4 - len(c)) * [_I['.']])
    if e == '-': ix.extend(2 * [_I['.']])
    else:
        for ch in e:
            if ch not in _I: return None
            ix.append(_I[ch])
    h += '.' * (3 - len(h)); ix.extend([_I[x] for x in h[:3]])
    f += '.' * (3 - len(f)); ix.extend([_I[x] for x in f[:3]])
    return ix if len(ix) == 77 else None

DIM = 1024; NL = 16; NH = 8; HD = 128; FFN = 4096; FS = 79

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(1968, DIM); self.pe = nn.Embedding(FS, DIM); self.layers = nn.ModuleList()
        for _ in range(NL):
            self.layers.append(nn.ModuleDict(dict(
                la=nn.LayerNorm(DIM), q=nn.Linear(DIM, DIM, bias=False),
                k=nn.Linear(DIM, DIM, bias=False), v=nn.Linear(DIM, DIM, bias=False),
                o=nn.Linear(DIM, DIM, bias=False), lm=nn.LayerNorm(DIM),
                g=nn.Linear(DIM, FFN, bias=False), u=nn.Linear(DIM, FFN, bias=False),
                d=nn.Linear(FFN, DIM, bias=False))))
        self.fn = nn.LayerNorm(DIM)
    def forward(self, t):
        B, T = t.shape
        s = torch.cat([torch.zeros(B, 1, dtype=t.dtype, device=t.device), t[:, :-1]], dim=1)
        x = self.te(s) * math.sqrt(DIM) + self.pe(torch.arange(T, device=t.device))
        for l in self.layers:
            xn = l['la'](x); q = l['q'](xn).reshape(B, T, NH, HD); k = l['k'](xn).reshape(B, T, NH, HD); v = l['v'](xn).reshape(B, T, NH, HD)
            a = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(HD); a = F.softmax(a, dim=-1)
            o = torch.einsum('bhtT,bThd->bthd', a, v).reshape(B, T, DIM); x = x + l['o'](o)
            xn = l['lm'](x); x = x + l['d'](F.silu(l['g'](xn)) * l['u'](xn))
        return self.fn(x)

def glk(i): return 'layer_norm' if i == 0 else 'layer_norm_' + str(i)
def gak(i): return 'multi_head_dot_product_attention' if i == 0 else 'multi_head_dot_product_attention_' + str(i)
def gmk(i): return 'linear' if i == 0 else 'linear_' + str(i)

def load_encoder():
    print('Loading encoder...')
    pr = dict(np.load(PARAMS))
    enc = Enc()
    test_key = gak(0) + '/linear/w'
    use_linear_w = test_key in pr
    with torch.no_grad():
        enc.te.weight.copy_(torch.tensor(pr['embed/embeddings'])); enc.pe.weight.copy_(torch.tensor(pr['embed_1/embeddings']))
        for i, l in enumerate(enc.layers):
            la, lm = glk(i * 2), glk(i * 2 + 1)
            l['la'].weight.copy_(torch.tensor(pr[la + '/scale'])); l['la'].bias.copy_(torch.tensor(pr[la + '/offset']))
            l['lm'].weight.copy_(torch.tensor(pr[lm + '/scale'])); l['lm'].bias.copy_(torch.tensor(pr[lm + '/offset']))
            ak = gak(i)
            if use_linear_w:
                l['q'].weight.copy_(torch.tensor(pr[ak + '/linear/w']).T); l['k'].weight.copy_(torch.tensor(pr[ak + '/linear_1/w']).T)
                l['v'].weight.copy_(torch.tensor(pr[ak + '/linear_2/w']).T); l['o'].weight.copy_(torch.tensor(pr[ak + '/linear_3/w']).T)
                mb = i * 3; l['g'].weight.copy_(torch.tensor(pr[gmk(mb) + '/w']).T); l['u'].weight.copy_(torch.tensor(pr[gmk(mb + 1) + '/w']).T); l['d'].weight.copy_(torch.tensor(pr[gmk(mb + 2) + '/w']).T)
            else:
                for n, full in [('q', 'query'), ('k', 'key'), ('v', 'value'), ('o', 'linear')]:
                    l[n].weight.copy_(torch.tensor(pr[ak + '/' + full + '/kernel']).reshape(DIM, DIM).T)
                l['g'].weight.copy_(torch.tensor(pr[ak + '/mlp/gating_einsum'][0]).T)
                l['u'].weight.copy_(torch.tensor(pr[ak + '/mlp/gating_einsum'][1]).T)
                l['d'].weight.copy_(torch.tensor(pr[ak + '/mlp/linear/kernel']).T)
        fl = glk(NL * 2); enc.fn.weight.copy_(torch.tensor(pr[fl + '/scale'])); enc.fn.bias.copy_(torch.tensor(pr[fl + '/offset']))
    del pr; enc = enc.cuda().eval()
    print('Encoder loaded.')
    return enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-best', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--positions', default=POSITIONS)
    parser.add_argument('--output', default=BASE + '/cache/blunder_move_token_200k.pt')
    args = parser.parse_args()

    with open(MOVE_MAP) as f: m2a = json.load(f)
    with open(args.positions) as f: positions = json.load(f)
    print(f'Loaded {len(positions)} positions')

    enc = load_encoder()

    move_key = 'best_uci' if args.use_best else 'blunder_uci'
    seqs = []
    valid_idx = []
    for i, p in enumerate(positions):
        ft = tok(p['fen'])
        move = p[move_key]
        if ft is None or move not in m2a: continue
        seqs.append(ft + [m2a[move], 64])
        valid_idx.append(i)
    print(f'{len(seqs)} valid sequences')

    # Encode and extract only position 77 (move token)
    all_mt = []
    t0 = time.time()
    for i in range(0, len(seqs), args.batch_size):
        batch = seqs[i:i + args.batch_size]
        tens = torch.tensor(batch, dtype=torch.long, device='cuda')
        with torch.no_grad():
            h = enc(tens)  # [B, 79, 1024]
            mt = h[:, 77, :]  # [B, 1024] — the move token
        all_mt.append(mt.half().cpu())
        if (i // args.batch_size) % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f'  {i}/{len(seqs)} ({elapsed:.0f}s)')
            sys.stdout.flush()

    move_tokens = torch.cat(all_mt)  # [N, 1024]
    elapsed = time.time() - t0
    print(f'Encoded {move_tokens.shape[0]} move tokens in {elapsed:.0f}s')

    # Normalization
    flat = move_tokens.float()
    mean = flat.mean(dim=0)
    std = flat.std(dim=0) + 1e-8

    # Also do best moves if we did blunders
    best_mt = None
    if not args.use_best:
        print('Encoding best moves...')
        best_seqs = []
        for i in valid_idx:
            p = positions[i]
            ft = tok(p['fen'])
            if ft and p['best_uci'] in m2a:
                best_seqs.append(ft + [m2a[p['best_uci']], 64])
        all_best = []
        t0 = time.time()
        for i in range(0, len(best_seqs), args.batch_size):
            batch = best_seqs[i:i + args.batch_size]
            tens = torch.tensor(batch, dtype=torch.long, device='cuda')
            with torch.no_grad():
                h = enc(tens)
                mt = h[:, 77, :]
            all_best.append(mt.half().cpu())
            if (i // args.batch_size) % 200 == 0 and i > 0:
                print(f'  best: {i}/{len(best_seqs)} ({time.time() - t0:.0f}s)')
                sys.stdout.flush()
        best_mt = torch.cat(all_best)
        print(f'Encoded {best_mt.shape[0]} best move tokens in {time.time() - t0:.0f}s')

    metadata = [positions[i] for i in valid_idx]
    save_dict = {
        'blunder_mt': move_tokens,   # [N, 1024] float16
        'metadata': metadata,
        'n_positions': len(valid_idx),
        'mean': mean.numpy(),
        'std': std.numpy(),
    }
    if best_mt is not None:
        save_dict['best_mt'] = best_mt
        best_flat = best_mt.float()
        save_dict['best_mean'] = best_flat.mean(dim=0).numpy()
        save_dict['best_std'] = (best_flat.std(dim=0) + 1e-8).numpy()

    torch.save(save_dict, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'Saved: {args.output} ({size_mb:.0f} MB)')

    del enc; torch.cuda.empty_cache()
    print('Done.')

if __name__ == '__main__':
    main()
