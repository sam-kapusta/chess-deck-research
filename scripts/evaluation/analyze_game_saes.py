#!/usr/bin/env python3
"""Run blunder SAE k=32 and k=8 on a specific game's mistake positions.

Compares what features fire for played (blunder) vs best move.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Encoder imports
ENCODER_PATH = '/home/ec2-user/SageMaker/chess-stage-a/cache/deepmind_270m_params.npz'
MOVE_MAP_PATH = '/home/ec2-user/SageMaker/chess-stage-a/cache/move_to_action.json'


class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k
    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


# Simple FEN tokenizer matching the encoder
_C = list('0123456789abcdefghpnrkqPBNRQKw.')
_I = {c: i for i, c in enumerate(_C)}
_S = frozenset('12345678')

def tokenize(fen):
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


def encode_position(fen, move_uci, encoder_session, move_map):
    """Encode (FEN, move) → hidden[77] (move token)."""
    ft = tokenize(fen)
    if ft is None or move_uci not in move_map:
        return None
    seq = np.array([ft + [move_map[move_uci], 64]], dtype=np.int64)
    output = encoder_session.run(None, {"input": seq})
    hidden = output[0][0]  # [79, 1024]
    return hidden[77].astype(np.float32)  # move token


def run_sae(hidden_vec, sae, mean, std):
    """Run SAE on a hidden vector, return (feature_ids, strengths)."""
    x = torch.tensor(hidden_vec).unsqueeze(0)
    x = (x - mean) / std
    with torch.no_grad():
        _, acts = sae(x)
    acts_np = acts.numpy()[0]
    active = np.where(acts_np > 0)[0]
    strengths = acts_np[active]
    order = np.argsort(-strengths)
    return active[order].tolist(), strengths[order].tolist()


def main():
    game_file = sys.argv[1] if len(sys.argv) > 1 else '/home/ec2-user/SageMaker/game_mistakes.json'

    with open(game_file) as f:
        mistakes = json.load(f)
    print(f'Loaded {len(mistakes)} mistakes')

    # Load encoder
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1

    encoder_path = '/home/ec2-user/SageMaker/chess-stage-a/cache/encoder_270m.onnx'
    if not os.path.exists(encoder_path):
        print(f'Encoder not found at {encoder_path}')
        # Try alternate paths
        for alt in ['/home/ec2-user/SageMaker/chess-coach-scripts/data/encoder_270m.onnx',
                    '/home/ec2-user/SageMaker/chess-stage-a/encoder_270m.onnx']:
            if os.path.exists(alt):
                encoder_path = alt
                break

    print(f'Loading encoder from {encoder_path}')
    encoder = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    with open('/home/ec2-user/SageMaker/chess-stage-a/cache/move_to_action.json') as f:
        move_map = json.load(f)

    # Load both SAEs
    print('Loading SAEs...')
    # k=32
    ckpt32 = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_mt_2048_k32_aux.pt',
                         map_location='cpu', weights_only=False)
    sae32 = SAE(1024, 2048, 32)
    sae32.load_state_dict(ckpt32['model_state_dict'])
    mean32 = torch.tensor(ckpt32['mean'], dtype=torch.float32)
    std32 = torch.tensor(ckpt32['std'], dtype=torch.float32) + 1e-8

    # k=8
    ckpt8 = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_mt_512_k8.pt',
                        map_location='cpu', weights_only=False)
    sae8 = SAE(1024, 512, 8)
    sae8.load_state_dict(ckpt8['model_state_dict'])
    mean8 = torch.tensor(ckpt8['mean'], dtype=torch.float32)
    std8 = torch.tensor(ckpt8['std'], dtype=torch.float32) + 1e-8

    # Load labels
    labels32 = {}
    try:
        with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
            labels32 = json.load(f)
    except:
        print('Warning: no k=32 labels')

    print(f'k=32 labels: {len(labels32)}')
    print()

    # Process each mistake
    for m in mistakes:
        fen = m['fen']
        played = m['uci']
        best = m['best_uci']
        ply = m['ply']
        san = m['san']
        cp_loss = m['cp_loss']

        print(f'{"="*70}')
        print(f'Ply {ply}: {san} ({m["side"]}) — cp_loss={cp_loss}')
        print(f'  Played: {played}  Best: {best}')
        print(f'  FEN: {fen}')

        # Encode both moves
        h_played = encode_position(fen, played, encoder, move_map)
        h_best = encode_position(fen, best, encoder, move_map)

        if h_played is None:
            print(f'  Could not encode played move {played}')
            continue
        if h_best is None:
            print(f'  Could not encode best move {best}')
            h_best = None

        # Run k=32 SAE
        fids32_played, str32_played = run_sae(h_played, sae32, mean32, std32)
        print(f'\n  --- k=32 SAE (2048 dict) on PLAYED move ({played}) ---')
        for fid, s in zip(fids32_played[:8], str32_played[:8]):
            lbl = labels32.get(str(fid), {}).get('label', '?')[:50]
            cat = labels32.get(str(fid), {}).get('category', '?')
            print(f'    F{fid} (str={s:.2f}) [{cat}] {lbl}')

        if h_best is not None:
            fids32_best, str32_best = run_sae(h_best, sae32, mean32, std32)
            print(f'\n  --- k=32 SAE on BEST move ({best}) ---')
            for fid, s in zip(fids32_best[:8], str32_best[:8]):
                lbl = labels32.get(str(fid), {}).get('label', '?')[:50]
                cat = labels32.get(str(fid), {}).get('category', '?')
                print(f'    F{fid} (str={s:.2f}) [{cat}] {lbl}')

            # Diff: what's in best but not played?
            played_set = set(fids32_played)
            best_set = set(fids32_best)
            only_best = best_set - played_set
            if only_best:
                print(f'\n  --- DIFF: Features in best but not played (what was missed) ---')
                for fid in list(only_best)[:5]:
                    s = str32_best[fids32_best.index(fid)] if fid in fids32_best else 0
                    lbl = labels32.get(str(fid), {}).get('label', '?')[:50]
                    print(f'    F{fid} (str={s:.2f}) {lbl}')

        # Run k=8 SAE
        fids8_played, str8_played = run_sae(h_played, sae8, mean8, std8)
        print(f'\n  --- k=8 SAE (512 dict) on PLAYED move ({played}) ---')
        for fid, s in zip(fids8_played, str8_played):
            print(f'    F{fid} (str={s:.2f})')

        if h_best is not None:
            fids8_best, str8_best = run_sae(h_best, sae8, mean8, std8)
            print(f'\n  --- k=8 SAE on BEST move ({best}) ---')
            for fid, s in zip(fids8_best, str8_best):
                print(f'    F{fid} (str={s:.2f})')

            # Diff
            played8 = set(fids8_played)
            best8 = set(fids8_best)
            shared = played8 & best8
            only_played = played8 - best8
            only_best8 = best8 - played8
            print(f'\n  --- k=8 DIFF ---')
            print(f'    Shared: {len(shared)} features')
            print(f'    Only in played: {sorted(only_played)}')
            print(f'    Only in best: {sorted(only_best8)}')

        print()


if __name__ == '__main__':
    main()
