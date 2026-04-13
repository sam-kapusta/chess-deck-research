"""Cache encoder activations for blunder/mistake moves from Lichess.

Streams Lichess/chess-position-evaluations, groups by FEN to find multi-PV,
identifies blunders (eval drop >= threshold) and caches encoder(position, blunder_move)
hidden states for SAE training.

Also caches encoder(position, best_move) for each position so we can train
on both or compute diffs.

Usage (on chess-research notebook):
    python3 cache_blunder_activations.py
    python3 cache_blunder_activations.py --n-positions 200000 --min-loss 200
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'

with open(MOVE_MAP) as f:
    M2A = json.load(f)

# Tokenizer (same as cache_encoder_activations.py)
_C = list('0123456789abcdefghpnrkqPBNRQKw.')
_I = {c: i for i, c in enumerate(_C)}
_S = frozenset('12345678')


def tok(fen):
    p = fen.split(' ')
    while len(p) < 6:
        if len(p) == 4: p.append('0')
        elif len(p) == 5: p.append('1')
        else: p.append('-')
    b, s, c, e, h, f = p[:6]
    b = s + b.replace('/', '')
    ix = []
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


# Encoder model (DeepMind 270M)
DIM = 1024; NL = 16; NH = 8; HD = 128; FFN = 4096; FS = 79


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(1968, DIM)
        self.pe = nn.Embedding(FS, DIM)
        self.layers = nn.ModuleList()
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
            xn = l['la'](x)
            q = l['q'](xn).reshape(B, T, NH, HD)
            k = l['k'](xn).reshape(B, T, NH, HD)
            v = l['v'](xn).reshape(B, T, NH, HD)
            a = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(HD)
            a = F.softmax(a, dim=-1)
            o = torch.einsum('bhtT,bThd->bthd', a, v).reshape(B, T, DIM)
            x = x + l['o'](o)
            xn = l['lm'](x)
            x = x + l['d'](F.silu(l['g'](xn)) * l['u'](xn))
        return self.fn(x)


def glk(i): return 'layer_norm' if i == 0 else 'layer_norm_' + str(i)
def gak(i): return 'multi_head_dot_product_attention' if i == 0 else 'multi_head_dot_product_attention_' + str(i)
def gmk(i): return 'linear' if i == 0 else 'linear_' + str(i)


def load_encoder():
    print('Loading encoder...')
    pr = dict(np.load(PARAMS))
    enc = Enc()
    # Auto-detect key format: chess-stage-a uses linear/w, some NPZs use query/kernel
    test_key = gak(0) + '/linear/w'
    use_linear_w = test_key in pr
    print(f'  Key format: {"linear/w" if use_linear_w else "query/kernel"}')
    with torch.no_grad():
        enc.te.weight.copy_(torch.tensor(pr['embed/embeddings']))
        enc.pe.weight.copy_(torch.tensor(pr['embed_1/embeddings']))
        for i, l in enumerate(enc.layers):
            la, lm = glk(i * 2), glk(i * 2 + 1)
            l['la'].weight.copy_(torch.tensor(pr[la + '/scale']))
            l['la'].bias.copy_(torch.tensor(pr[la + '/offset']))
            l['lm'].weight.copy_(torch.tensor(pr[lm + '/scale']))
            l['lm'].bias.copy_(torch.tensor(pr[lm + '/offset']))
            ak = gak(i)
            if use_linear_w:
                l['q'].weight.copy_(torch.tensor(pr[ak + '/linear/w']).T)
                l['k'].weight.copy_(torch.tensor(pr[ak + '/linear_1/w']).T)
                l['v'].weight.copy_(torch.tensor(pr[ak + '/linear_2/w']).T)
                l['o'].weight.copy_(torch.tensor(pr[ak + '/linear_3/w']).T)
                mb = i * 3
                l['g'].weight.copy_(torch.tensor(pr[gmk(mb) + '/w']).T)
                l['u'].weight.copy_(torch.tensor(pr[gmk(mb + 1) + '/w']).T)
                l['d'].weight.copy_(torch.tensor(pr[gmk(mb + 2) + '/w']).T)
            else:
                for n, full in [('q', 'query'), ('k', 'key'), ('v', 'value'), ('o', 'linear')]:
                    l[n].weight.copy_(torch.tensor(pr[ak + '/' + full + '/kernel']).reshape(DIM, DIM).T)
                l['g'].weight.copy_(torch.tensor(pr[ak + '/mlp/gating_einsum'][0]).T)
                l['u'].weight.copy_(torch.tensor(pr[ak + '/mlp/gating_einsum'][1]).T)
                l['d'].weight.copy_(torch.tensor(pr[ak + '/mlp/linear/kernel']).T)
        fl = glk(NL * 2)
        enc.fn.weight.copy_(torch.tensor(pr[fl + '/scale']))
        enc.fn.bias.copy_(torch.tensor(pr[fl + '/offset']))
    del pr
    enc = enc.cuda().eval()
    print('Encoder loaded.')
    return enc


def encode_pair(enc, fen, move_uci):
    """Encode a single (FEN, move) pair, return [77, 1024] hidden state."""
    ft = tok(fen)
    if ft is None or move_uci not in M2A:
        return None
    seq = ft + [M2A[move_uci], 64]
    with torch.no_grad():
        h = enc(torch.tensor([seq], dtype=torch.long, device='cuda'))
    return h[0, 1:78, :].cpu().half()  # [77, 1024] float16


def main():
    parser = argparse.ArgumentParser(description='Cache encoder activations for blunder moves')
    parser.add_argument('--n-positions', type=int, default=200000, help='Target number of blunder positions')
    parser.add_argument('--min-loss', type=int, default=200, help='Minimum cp loss to count as blunder (default 200 = 2.0 eval)')
    parser.add_argument('--output', default=BASE + '/cache/blunder_acts_200k.pt', help='Output path')
    args = parser.parse_args()

    enc = load_encoder()

    # Stream Lichess positions, group by FEN to find multi-PV
    ds = load_dataset('Lichess/chess-position-evaluations', split='train', streaming=True)

    # Pre-allocate to avoid OOM from list of tensors
    # 50K × [77, 1024] × float16 = ~7.5GB per tensor
    MAX_N = args.n_positions
    blunder_hidden = torch.zeros(MAX_N, 77, 1024, dtype=torch.float16)
    best_hidden = torch.zeros(MAX_N, 77, 1024, dtype=torch.float16)
    metadata = []

    current_fen = None
    current_moves = []
    n_blunders = 0
    n_scanned = 0
    t0 = time.time()

    for row in ds:
        fen = row['fen']
        cp = row.get('cp')
        line = row.get('line', '')
        if cp is None or not line:
            continue
        move_uci = line.split()[0]
        if move_uci not in M2A:
            continue

        if fen != current_fen:
            # Process previous position
            if current_fen and len(current_moves) >= 2:
                sorted_moves = sorted(current_moves, key=lambda x: -x[1])
                best_move, best_cp = sorted_moves[0]

                for alt_move, alt_cp in sorted_moves[1:]:
                    cp_loss = best_cp - alt_cp
                    if cp_loss >= args.min_loss:
                        # This is a blunder — cache both moves
                        blunder_h = encode_pair(enc, current_fen, alt_move)
                        best_h = encode_pair(enc, current_fen, best_move)
                        if blunder_h is not None and best_h is not None:
                            blunder_hidden[n_blunders] = blunder_h
                            best_hidden[n_blunders] = best_h
                            metadata.append({
                                'fen': current_fen,
                                'blunder_uci': alt_move,
                                'best_uci': best_move,
                                'blunder_cp': alt_cp,
                                'best_cp': best_cp,
                                'cp_loss': cp_loss,
                            })
                            n_blunders += 1
                            break  # one blunder per position

            current_fen = fen
            current_moves = []
            n_scanned += 1

        current_moves.append((move_uci, cp))

        if n_blunders % 5000 == 0 and n_blunders > 0:
            elapsed = time.time() - t0
            print('  ' + str(n_blunders) + '/' + str(args.n_positions) +
                  ' blunders from ' + str(n_scanned) + ' positions' +
                  ' ({:.0f} blunders/sec, {:.1f}% hit rate)'.format(
                      n_blunders / elapsed, 100 * n_blunders / n_scanned))
            sys.stdout.flush()

        if n_blunders >= args.n_positions:
            break

    print()
    print('Collected ' + str(n_blunders) + ' blunders from ' + str(n_scanned) + ' positions')
    print('Hit rate: {:.1f}%'.format(100 * n_blunders / max(n_scanned, 1)))

    # Trim to actual size
    blunder_tensor = blunder_hidden[:n_blunders]
    best_tensor = best_hidden[:n_blunders]

    # Compute normalization stats for SAE training
    # Per-token: flatten to [N*77, 1024] — do in chunks to save memory
    print('Computing normalization stats...')
    flat_blunder = blunder_tensor.reshape(-1, DIM).float()
    blunder_mean = flat_blunder.mean(dim=0)
    blunder_std = flat_blunder.std(dim=0)
    del flat_blunder

    flat_best = best_tensor.reshape(-1, DIM).float()
    best_mean = flat_best.mean(dim=0)
    best_std = flat_best.std(dim=0)
    del flat_best

    print('Saving cache...')
    torch.save({
        'blunder_hidden': blunder_tensor,   # [N, 77, 1024] float16
        'best_hidden': best_tensor,         # [N, 77, 1024] float16
        'metadata': metadata,               # list of dicts
        'n_blunders': n_blunders,
        'n_scanned': n_scanned,
        'min_loss': args.min_loss,
        'normalization': {
            'blunder_mean': blunder_mean.numpy(),
            'blunder_std': blunder_std.numpy(),
            'best_mean': best_mean.numpy(),
            'best_std': best_std.numpy(),
        },
    }, args.output)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print('Saved to ' + args.output + ' ({:.0f} MB)'.format(size_mb))

    # Distribution stats
    losses = [m['cp_loss'] for m in metadata]
    print()
    print('CP loss distribution:')
    for threshold in [100, 200, 300, 500, 1000]:
        n = sum(1 for l in losses if l >= threshold)
        print('  >= ' + str(threshold) + 'cp: ' + str(n) + ' ({:.1f}%)'.format(100 * n / len(losses)))
    print('DONE')


if __name__ == '__main__':
    main()
