"""Cache encoder activations for blunder/mistake moves from Lichess.

Two-phase pipeline:
  Phase 1: Stream HuggingFace dataset, filter blunders (CPU only, fast)
  Phase 2: Batch-encode all positions on GPU (64 at a time, same as puzzle cache)

Usage (on chess-poc notebook):
    python3 cache_blunder_activations.py
    python3 cache_blunder_activations.py --n-positions 200000 --min-loss 200
    python3 cache_blunder_activations.py --phase1-only  # just download+filter, no GPU
    python3 cache_blunder_activations.py --phase2-only  # encode from saved positions
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

BASE = '/home/ec2-user/SageMaker/chess-stage-a'
PARAMS = BASE + '/cache/deepmind_270m_params.npz'
MOVE_MAP = BASE + '/cache/move_to_action.json'
POSITIONS_FILE = BASE + '/cache/blunder_positions.json'

# Tokenizer
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


# ── Phase 1: Download + Filter (CPU only) ──

def phase1_collect(n_positions, min_loss, positions_file):
    """Stream HuggingFace dataset, collect blunder positions. No GPU needed."""
    from datasets import load_dataset

    if os.path.exists(positions_file):
        with open(positions_file) as f:
            positions = json.load(f)
        print(f'Phase 1: loaded {len(positions)} cached positions from {positions_file}')
        if len(positions) >= n_positions:
            return positions[:n_positions]
        print(f'  Need {n_positions - len(positions)} more, resuming...')

    with open(MOVE_MAP) as f:
        m2a = json.load(f)

    ds = load_dataset('Lichess/chess-position-evaluations', split='train', streaming=True)

    positions = []
    current_fen = None
    current_moves = []
    n_scanned = 0
    n_rows = 0
    t0 = time.time()
    last_print = 0

    print(f'Phase 1: streaming dataset, collecting {n_positions} blunders with ≥{min_loss}cp loss...')

    for row in ds:
        n_rows += 1
        fen = row['fen']
        cp = row.get('cp')
        line = row.get('line', '')
        if cp is None or not line:
            continue
        move_uci = line.split()[0]
        if move_uci not in m2a:
            continue

        if fen != current_fen:
            # Process previous position
            if current_fen and len(current_moves) >= 2:
                sorted_moves = sorted(current_moves, key=lambda x: -x[1])
                best_move, best_cp = sorted_moves[0]

                for alt_move, alt_cp in sorted_moves[1:]:
                    cp_loss = best_cp - alt_cp
                    if cp_loss >= min_loss:
                        ft = tok(current_fen)
                        if ft and alt_move in m2a and best_move in m2a:
                            positions.append({
                                'fen': current_fen,
                                'blunder_uci': alt_move,
                                'best_uci': best_move,
                                'blunder_cp': alt_cp,
                                'best_cp': best_cp,
                                'cp_loss': cp_loss,
                            })
                            break

            current_fen = fen
            current_moves = []
            n_scanned += 1

        current_moves.append((move_uci, cp))

        # Print every 10K blunders or every 60s
        n_found = len(positions)
        elapsed = time.time() - t0
        if (n_found >= last_print + 10000) or (elapsed > last_print / max(1, n_found) * elapsed + 60 and n_found > last_print):
            if n_found > last_print:
                rate = n_found / elapsed if elapsed > 0 else 0
                eta = (n_positions - n_found) / rate if rate > 0 else 0
                print(f'  {n_found}/{n_positions} blunders from {n_scanned} positions '
                      f'({n_rows} rows, {rate:.0f}/sec, hit {100*n_found/max(1,n_scanned):.1f}%, '
                      f'ETA {eta/60:.0f}min)')
                sys.stdout.flush()
                last_print = n_found

        if n_found >= n_positions:
            break

    elapsed = time.time() - t0
    print(f'\nPhase 1 done: {len(positions)} blunders from {n_scanned} positions '
          f'({n_rows} rows) in {elapsed:.0f}s')

    # Save positions to disk
    with open(positions_file, 'w') as f:
        json.dump(positions[:n_positions], f)
    print(f'Saved to {positions_file}')

    return positions[:n_positions]


# ── Phase 2: Batch Encode (GPU) ──

def phase2_encode(positions, output_path, batch_size=64):
    """Batch-encode all blunder+best positions. Matches puzzle cache format."""
    with open(MOVE_MAP) as f:
        m2a = json.load(f)

    enc = load_encoder()
    n = len(positions)

    # Build token sequences for blunder moves and best moves
    print(f'Phase 2: encoding {n} positions (blunder + best = {2*n} forward passes)...')
    blunder_seqs = []
    best_seqs = []
    valid_idx = []

    for i, pos in enumerate(positions):
        ft = tok(pos['fen'])
        if ft is None:
            continue
        bm = pos['blunder_uci']
        bst = pos['best_uci']
        if bm not in m2a or bst not in m2a:
            continue
        blunder_seqs.append(ft + [m2a[bm], 64])
        best_seqs.append(ft + [m2a[bst], 64])
        valid_idx.append(i)

    n_valid = len(valid_idx)
    print(f'  {n_valid}/{n} positions valid for encoding')

    # Encode in batches
    def batch_encode(seqs, label):
        all_acts = []
        t0 = time.time()
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            tens = torch.tensor(batch, dtype=torch.long, device='cuda')
            with torch.no_grad():
                h = enc(tens)[:, :77, :]  # [B, 77, 1024]
            for b in range(len(batch)):
                all_acts.append(h[b].half().cpu())
            if (i // batch_size) % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(seqs) - i) / rate
                print(f'  {label}: {i}/{len(seqs)} ({elapsed:.0f}s, ETA {eta:.0f}s)')
                sys.stdout.flush()
        elapsed = time.time() - t0
        print(f'  {label}: done ({len(seqs)} positions in {elapsed:.0f}s)')
        return torch.stack(all_acts)  # [N, 77, 1024]

    blunder_hidden = batch_encode(blunder_seqs, 'blunder')
    best_hidden = batch_encode(best_seqs, 'best')

    # Normalization stats
    print('Computing normalization stats...')
    flat_b = blunder_hidden.reshape(-1, DIM).float()
    b_mean = flat_b.mean(dim=0)
    b_std = flat_b.std(dim=0)
    del flat_b

    flat_best = best_hidden.reshape(-1, DIM).float()
    best_mean = flat_best.mean(dim=0)
    best_std = flat_best.std(dim=0)
    del flat_best

    # Build metadata for valid positions
    metadata = [positions[i] for i in valid_idx]

    print('Saving cache...')
    torch.save({
        'blunder_hidden': blunder_hidden,
        'best_hidden': best_hidden,
        'metadata': metadata,
        'n_blunders': n_valid,
        'n_scanned': len(positions),
        'min_loss': metadata[0]['cp_loss'] if metadata else 0,
        'normalization': {
            'blunder_mean': b_mean.numpy(),
            'blunder_std': b_std.numpy(),
            'best_mean': best_mean.numpy(),
            'best_std': best_std.numpy(),
        },
    }, output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'Saved: {output_path} ({size_mb:.0f} MB)')

    # Distribution stats
    losses = [m['cp_loss'] for m in metadata]
    print(f'\nCP loss distribution ({len(losses)} positions):')
    for threshold in [200, 300, 500, 1000, 2000]:
        ct = sum(1 for l in losses if l >= threshold)
        print(f'  >= {threshold}cp: {ct} ({100 * ct / len(losses):.1f}%)')

    del enc
    torch.cuda.empty_cache()
    print('Done. GPU freed.')


def main():
    parser = argparse.ArgumentParser(description='Cache encoder activations for blunder moves')
    parser.add_argument('--n-positions', type=int, default=200000)
    parser.add_argument('--min-loss', type=int, default=200, help='Min cp loss (default 200 = 2.0 eval)')
    parser.add_argument('--output', default=BASE + '/cache/blunder_acts_200k.pt')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--phase1-only', action='store_true', help='Only download+filter, no GPU')
    parser.add_argument('--phase2-only', action='store_true', help='Only encode from saved positions')
    args = parser.parse_args()

    if args.phase2_only:
        print(f'Loading positions from {POSITIONS_FILE}...')
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)
        positions = positions[:args.n_positions]
        print(f'Loaded {len(positions)} positions')
        phase2_encode(positions, args.output, args.batch_size)
        return

    positions = phase1_collect(args.n_positions, args.min_loss, POSITIONS_FILE)

    if args.phase1_only:
        print('Phase 1 only — skipping encoding.')
        return

    phase2_encode(positions, args.output, args.batch_size)


if __name__ == '__main__':
    main()
