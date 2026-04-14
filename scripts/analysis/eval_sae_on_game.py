#!/usr/bin/env python3
"""Evaluate SAE(s) on a specific game. The SAE evaluation tool.

Usage:
    eval_sae_on_game.py --game game_analysis.json --sae sae1.pt [--sae sae2.pt] [--labels labels1.json]

Flow:
    1. Load game analysis (from analyze_game.py)
    2. Add Sonnet narrative to biggest mistakes (one LLM call)
    3. For each SAE: encode played+best moves, get features, label diffs
    4. Output per-mistake comparison table

This is for SAE evaluation — which SAE produces the most useful features?
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# ── Encoder + SAE setup ──

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False

try:
    import onnxruntime as ort
    ONNX = True
except ImportError:
    ONNX = False

try:
    import boto3
    BOTO = True
except ImportError:
    BOTO = False


DEFAULT_ENCODER = os.path.join(os.path.dirname(__file__), '..', '..', '..',
    'chess-coach', 'backend', 'lambda', 'sae_features', 'data', 'encoder_270m.onnx')
DEFAULT_MOVE_MAP = os.path.join(os.path.dirname(__file__), '..', '..', '..',
    'chess-coach', 'backend', 'lambda', 'sae_features', 'data', 'move_to_action.json')


# ── Tokenizer ──
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


class Encoder:
    def __init__(self, onnx_path, move_map_path):
        ort.set_default_logger_severity(3)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        with open(move_map_path) as f:
            self.move_map = json.load(f)

    def encode(self, fen, uci):
        ft = tokenize(fen)
        if ft is None or uci not in self.move_map:
            return None
        seq = np.array([ft + [self.move_map[uci], 64]], dtype=np.int64)
        out = self.session.run(None, {"input": seq})
        return out[0][0][77].astype(np.float32)


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


def load_sae(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    dd = config.get('dict_size', 2048)
    k = config.get('k', 32)
    sae = SAE(1024, dd, k)
    state = ckpt.get('model_state_dict', ckpt)
    model_keys = {'encoder.weight', 'encoder.bias', 'decoder.weight', 'decoder.bias', 'pre_bias'}
    filtered = {k_: v for k_, v in state.items() if k_ in model_keys}
    sae.load_state_dict(filtered if filtered else state, strict=False)

    mean = ckpt.get('mean')
    if mean is None:
        norm = ckpt.get('normalization', {}) or {}
        mean = norm.get('mean')
    std = ckpt.get('std')
    if std is None:
        norm = ckpt.get('normalization', {}) or {}
        std = norm.get('std')

    mean_t = torch.tensor(mean, dtype=torch.float32) if mean is not None else torch.zeros(1024)
    std_t = torch.tensor(std, dtype=torch.float32) + 1e-8 if std is not None else torch.ones(1024)

    name = os.path.basename(path).replace('.pt', '')
    return sae.eval(), mean_t, std_t, dd, k, name


def run_sae(hidden, sae, mean, std):
    x = torch.tensor(hidden).unsqueeze(0)
    x = (x - mean) / std
    with torch.no_grad():
        _, acts = sae(x)
    a = acts.numpy()[0]
    active = np.where(a > 0)[0]
    strengths = a[active]
    order = np.argsort(-strengths)
    return [(int(active[i]), round(float(strengths[i]), 2)) for i in order]


# ── Narrative ──

NARRATIVE_PROMPT = """You are a chess coach analyzing a game. Here are the biggest mistakes.
For each, explain in 1-2 sentences what went wrong and why the best move was better.
Use concrete chess language (pieces, squares, tactics). No fluff.

{mistakes}

Respond as JSON array: [{{"ply": N, "narrative": "..."}}]"""


def get_narratives(mistakes):
    if not BOTO:
        return {}
    client = boto3.client('bedrock-runtime', region_name='us-east-1')

    mistake_text = ""
    for m in mistakes:
        mistake_text += f"Ply {m['ply']}: {m['san']} ({m['side']}, {m['cp_loss']}cp loss)\n"
        mistake_text += f"  FEN: {m['fen']}\n"
        mistake_text += f"  Played: {m['uci']}  Best: {m['best_uci']}\n\n"

    prompt = NARRATIVE_PROMPT.format(mistakes=mistake_text)
    resp = client.converse(
        modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={'maxTokens': 1000},
    )
    text = resp['output']['message']['content'][0]['text']

    import re
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            return {item['ply']: item['narrative'] for item in items}
        except:
            pass
    return {}


# ── Feature labeling (direct API, per-feature) ──

LABEL_PROMPT = """This SAE feature fires on a chess blunder position.

FEN: {fen}
Played move (blunder): {played}
Best move: {best}
CP loss: {cp_loss}
Activation strength: {strength}

The feature fires on the PLAYED move but NOT on the best move (or vice versa).
What specific chess pattern does this feature detect?

Respond in exactly this format:
LABEL: 2-5 word specific label
CATEGORY: one of [hanging_pieces, overloaded_defenders, forks, pins, skewers, discovered_attacks, back_rank, king_safety, passed_pawns, rook_endgames, pawn_endgames, checkmate_patterns, quiet_moves, trapped_pieces, sacrifice, other]"""


def label_feature_in_context(fid, positions, model_id='us.anthropic.claude-sonnet-4-20250514-v1:0'):
    """Label a feature from its game positions. Fast: one short API call."""
    if not BOTO or not positions:
        return None

    # Use strongest activation position
    pos = max(positions, key=lambda p: p.get('strength', 0))

    prompt = LABEL_PROMPT.format(
        fen=pos['fen'], played=pos.get('uci', '?'),
        best=pos.get('best_uci', '?'), cp_loss=pos.get('cp_loss', 0),
        strength=pos.get('strength', 0),
    )

    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    resp = client.converse(
        modelId=model_id,
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={'maxTokens': 100},
    )
    text = resp['output']['message']['content'][0]['text']

    import re
    label_match = re.search(r'LABEL:\s*(.+)', text)
    cat_match = re.search(r'CATEGORY:\s*(\S+)', text)
    if label_match:
        return {
            'label': label_match.group(1).strip(),
            'category': cat_match.group(1).strip() if cat_match else 'other',
        }
    return None


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAE(s) on a specific game')
    parser.add_argument('--game', required=True, help='Game analysis JSON (from analyze_game.py)')
    parser.add_argument('--sae', action='append', required=True, help='SAE checkpoint(s). Can specify multiple: --sae a.pt --sae b.pt')
    parser.add_argument('--encoder', default=None, help='ONNX encoder path')
    parser.add_argument('--move-map', default=None, help='move_to_action.json path')
    parser.add_argument('--top-n', type=int, default=5, help='Show top N mistakes (default: 5)')
    parser.add_argument('--no-narrative', action='store_true', help='Skip Sonnet narrative')
    parser.add_argument('--no-labels', action='store_true', help='Skip feature labeling (just show IDs)')
    parser.add_argument('--output', '-o', help='Save results JSON')
    args = parser.parse_args()

    # Load game
    with open(args.game) as f:
        moves = json.load(f)
    if isinstance(moves, dict):
        moves = moves.get('moves', [])

    # Find biggest mistakes
    mistakes = [m for m in moves if m.get('uci') != m.get('best_uci') and m.get('cp_loss', 0) >= 50]
    mistakes.sort(key=lambda m: -m['cp_loss'])
    mistakes = mistakes[:args.top_n]
    print(f'Game: {len(moves)} moves, {len(mistakes)} mistakes (top {args.top_n})')
    print()

    # Get narrative
    narratives = {}
    if not args.no_narrative and mistakes:
        print('Getting Sonnet narrative for mistakes...')
        narratives = get_narratives(mistakes)
        print(f'  Got {len(narratives)} narratives')
        print()

    # Load encoder
    enc_path = args.encoder or DEFAULT_ENCODER
    mm_path = args.move_map or DEFAULT_MOVE_MAP
    # Resolve relative paths
    if not os.path.isabs(enc_path):
        enc_path = os.path.abspath(enc_path)
    if not os.path.isabs(mm_path):
        mm_path = os.path.abspath(mm_path)

    print(f'Loading encoder...')
    encoder = Encoder(enc_path, mm_path)

    # Load SAEs
    saes = []
    for sae_path in args.sae:
        print(f'Loading SAE: {sae_path}')
        sae, mean, std, dd, k, name = load_sae(sae_path)
        saes.append({'sae': sae, 'mean': mean, 'std': std, 'dd': dd, 'k': k, 'name': name})

    print()

    # Process each mistake
    results = []
    for m in mistakes:
        ply = m['ply']
        san = m['san']
        side = m['side']
        cp_loss = m['cp_loss']
        narrative = narratives.get(ply, '')

        print('=' * 70)
        print(f"Move {ply}: {san} ({side}) — {cp_loss}cp loss")
        print(f"  Played: {m['uci']}  Best: {m['best_uci']}")
        if narrative:
            print(f"  WHY: {narrative}")
        print()

        # Encode played + best
        h_played = encoder.encode(m['fen'], m['uci'])
        h_best = encoder.encode(m['fen'], m['best_uci']) if m['best_uci'] != m['uci'] else None

        if h_played is None:
            print(f"  Could not encode played move")
            continue

        mistake_result = {
            'ply': ply, 'san': san, 'side': side, 'cp_loss': cp_loss,
            'played': m['uci'], 'best': m['best_uci'],
            'fen': m['fen'], 'narrative': narrative,
            'saes': {},
        }

        # Run each SAE
        for s in saes:
            sae_name = s['name']
            feats_played = run_sae(h_played, s['sae'], s['mean'], s['std'])
            feats_best = run_sae(h_best, s['sae'], s['mean'], s['std']) if h_best is not None else []

            played_ids = {f[0] for f in feats_played}
            best_ids = {f[0] for f in feats_best}
            shared = played_ids & best_ids
            only_played = played_ids - best_ids
            only_best = best_ids - played_ids

            # Label diff features
            diff_labels = {}
            if not args.no_labels:
                for fid in list(only_played)[:4] + list(only_best)[:4]:
                    pos_info = {
                        'fen': m['fen'], 'uci': m['uci'], 'best_uci': m['best_uci'],
                        'cp_loss': cp_loss,
                        'strength': dict(feats_played + feats_best).get(fid, 0),
                    }
                    lbl = label_feature_in_context(fid, [pos_info])
                    if lbl:
                        diff_labels[fid] = lbl
                    time.sleep(0.3)

            # Print
            print(f"  {sae_name} (dict={s['dd']}, k={s['k']}):")
            print(f"    Shared: {len(shared)}  Only played: {len(only_played)}  Only best: {len(only_best)}")

            if only_played:
                print(f"    PLAYED-ONLY (what the blunder activated):")
                for fid in sorted(only_played):
                    strength = dict(feats_played).get(fid, 0)
                    lbl = diff_labels.get(fid, {})
                    lbl_str = f" → {lbl['label']} [{lbl['category']}]" if lbl else ""
                    print(f"      F{fid} (str={strength}){lbl_str}")

            if only_best:
                print(f"    BEST-ONLY (what the player missed):")
                for fid in sorted(only_best):
                    strength = dict(feats_best).get(fid, 0)
                    lbl = diff_labels.get(fid, {})
                    lbl_str = f" → {lbl['label']} [{lbl['category']}]" if lbl else ""
                    print(f"      F{fid} (str={strength}){lbl_str}")

            print()

            mistake_result['saes'][sae_name] = {
                'dict_size': s['dd'], 'k': s['k'],
                'n_shared': len(shared), 'n_only_played': len(only_played), 'n_only_best': len(only_best),
                'played_features': feats_played[:10],
                'best_features': feats_best[:10],
                'only_played': list(only_played),
                'only_best': list(only_best),
                'diff_labels': diff_labels,
            }

        results.append(mistake_result)

    # Summary
    print('=' * 70)
    print('SUMMARY: Which SAE gives better diff signal?')
    print()
    for s in saes:
        name = s['name']
        total_diff = sum(
            len(r['saes'].get(name, {}).get('only_played', [])) +
            len(r['saes'].get(name, {}).get('only_best', []))
            for r in results
        )
        total_labeled = sum(
            len(r['saes'].get(name, {}).get('diff_labels', {}))
            for r in results
        )
        print(f"  {name}: {total_diff} diff features across {len(results)} mistakes, {total_labeled} labeled")

    # Save
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
