#!/usr/bin/env python3
"""Evaluate SAE(s) on a specific game. Fair comparison tool.

Usage:
    eval_sae_on_game.py --game game_analysis.json --sae sae1.pt --profiles profiles1.json \
                        [--sae sae2.pt --profiles profiles2.json] [--top-n 5] [--diff-k 8]

Flow:
    1. Load game analysis (from analyze_game.py)
    2. Get Sonnet narrative for biggest mistakes (ground truth, no SAE bias)
    3. For each SAE: encode played+best, get feature diffs
    4. Look up diff features in pre-computed profiles (200K training examples)
    5. Label top-K diff features per mistake (parallel Sonnet calls)
    6. Output per-mistake comparison table with timing

Fair comparison: same number of diff features labeled per SAE (--diff-k, default 8).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxruntime as ort
except ImportError:
    print('pip install onnxruntime'); sys.exit(1)

try:
    import boto3
except ImportError:
    print('pip install boto3'); sys.exit(1)


DEFAULT_ENCODER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
    'chess-coach', 'backend', 'lambda', 'sae_features', 'data', 'encoder_270m.onnx'))
DEFAULT_MOVE_MAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
    'chess-coach', 'backend', 'lambda', 'sae_features', 'data', 'move_to_action.json'))
MODELS = {
    'sonnet': 'global.anthropic.claude-sonnet-4-6',
    'sonnet-us': 'us.anthropic.claude-sonnet-4-6',
    'sonnet4': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
    'haiku': 'global.anthropic.claude-haiku-4-5-20251001-v1:0',
}
BEDROCK_MODEL = MODELS['sonnet']  # default — Sonnet 4.6 global


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
        mean = (ckpt.get('normalization') or {}).get('mean')
    std = ckpt.get('std')
    if std is None:
        std = (ckpt.get('normalization') or {}).get('std')
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


# ── Narrative (ground truth, no SAE bias) ──

def get_narratives(mistakes):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    mistake_text = ""
    for m in mistakes:
        mistake_text += f"Ply {m['ply']}: {m['san']} ({m['side']}, {m['cp_loss']}cp loss)\n"
        mistake_text += f"  FEN: {m['fen']}\n"
        mistake_text += f"  Played: {m['uci']}  Best: {m['best_uci']}\n\n"

    prompt = f"""You are a chess coach. For each mistake, explain in 1-2 sentences what went wrong
and why the best move was better. Concrete chess language (pieces, squares, tactics). No fluff.

{mistake_text}

Respond as JSON array: [{{"ply": N, "narrative": "..."}}]"""

    import re
    resp = client.converse(
        modelId=BEDROCK_MODEL,
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={'maxTokens': 1000},
    )
    text = resp['output']['message']['content'][0]['text']
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return {item['ply']: item['narrative'] for item in json.loads(match.group())}
        except:
            pass
    return {}


# ── Feature labeling with profile examples ──

LABEL_PROMPT = """Chess SAE feature on blunder position.
{examples}
Game move {ply}: FEN: {fen}
Blunder: {played} ({side}). Best: {best}. Loss: {cp_loss}cp.
Feature fires on {move_type} NOT the other.

Two lines only:
LABEL: <2-5 words>
CATEGORY: <king_safety|hanging_pieces|forks|pins|skewers|discovered_attacks|back_rank|checkmate_patterns|overloaded_defenders|quiet_moves|trapped_pieces|sacrifice|passed_pawns|rook_endgames|pawn_endgames|other>"""

LABEL_PROMPT_NO_PROFILES = """Chess SAE feature on blunder position.
FEN: {fen}
Blunder: {played} ({side}). Best: {best}. Loss: {cp_loss}cp. Strength: {strength}.
Feature fires on {move_type} NOT the other.

Two lines only:
LABEL: <2-5 words>
CATEGORY: <king_safety|hanging_pieces|forks|pins|skewers|discovered_attacks|back_rank|checkmate_patterns|overloaded_defenders|quiet_moves|trapped_pieces|sacrifice|passed_pawns|rook_endgames|pawn_endgames|other>"""


def label_one_feature(fid, mistake, on_played, strength, profiles, client):
    """Label one diff feature. Returns (fid, label_dict) or (fid, None)."""
    import re

    # Build examples from profiles if available
    prof = profiles.get(str(fid), {}) if profiles else {}
    examples_text = ""
    if prof.get('examples'):
        for i, ex in enumerate(prof['examples'][:10]):
            examples_text += f"{i+1}. FEN: {ex.get('fen', '?')}  Move: {ex.get('blunder', ex.get('uci', '?'))}  "
            examples_text += f"Best: {ex.get('best', ex.get('best_uci', '?'))}  "
            examples_text += f"CP loss: {ex.get('cp_loss', '?')}  Strength: {ex.get('strength', '?')}\n"

    move_type = 'PLAYED' if on_played else 'BEST'
    if examples_text:
        prompt = LABEL_PROMPT.format(
            fid=fid, examples=examples_text, ply=mistake['ply'],
            fen=mistake['fen'], played=mistake['uci'], best=mistake['best_uci'],
            cp_loss=mistake['cp_loss'], side=mistake['side'], move_type=move_type,
        )
    else:
        prompt = LABEL_PROMPT_NO_PROFILES.format(
            fen=mistake['fen'], played=mistake['uci'], best=mistake['best_uci'],
            cp_loss=mistake['cp_loss'], side=mistake['side'], strength=strength,
            move_type=move_type,
        )

    try:
        resp = client.converse(
            modelId=BEDROCK_MODEL,
            messages=[{'role': 'user', 'content': [{'text': prompt}]}],
            inferenceConfig={'maxTokens': 30},
        )
        text = resp['output']['message']['content'][0]['text']
        label_match = re.search(r'LABEL:\s*(.+)', text)
        cat_match = re.search(r'CATEGORY:\s*(\S+)', text)
        if label_match:
            return fid, {
                'label': label_match.group(1).strip(),
                'category': cat_match.group(1).strip() if cat_match else 'other',
                'on_played': on_played,
                'strength': strength,
            }
    except Exception as e:
        print(f"    LABEL ERROR F{fid}: {e}", file=sys.stderr)
        return fid, None

    return fid, None


def label_diff_features_parallel(diff_features, mistake, profiles, max_workers=8):
    """Label multiple diff features in parallel. Returns dict of {fid: label_dict}."""
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    labels = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fid, strength, on_played in diff_features:
            future = executor.submit(label_one_feature, fid, mistake, on_played, strength, profiles, client)
            futures[future] = fid

        for future in as_completed(futures):
            fid, result = future.result()
            if result:
                labels[fid] = result

    return labels


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAE(s) on a specific game')
    parser.add_argument('--game', required=True, help='Game analysis JSON')
    parser.add_argument('--sae', action='append', required=True, help='SAE checkpoint(s)')
    parser.add_argument('--profiles', action='append', default=[], help='Profiles JSON per SAE (same order as --sae)')
    parser.add_argument('--encoder', default=None)
    parser.add_argument('--move-map', default=None)
    parser.add_argument('--top-n', type=int, default=5, help='Top N mistakes to analyze')
    parser.add_argument('--diff-k', type=int, default=8, help='Top K diff features to label per mistake per SAE (fair comparison)')
    parser.add_argument('--no-narrative', action='store_true')
    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--output', '-o')
    parser.add_argument('--workers', type=int, default=8, help='Parallel Sonnet workers')
    parser.add_argument('--model', default='sonnet', choices=list(MODELS.keys()),
                        help='LLM for labeling: sonnet (better) or haiku (faster/cheaper)')
    args = parser.parse_args()

    global BEDROCK_MODEL
    BEDROCK_MODEL = MODELS[args.model]
    print(f'Using model: {args.model} ({BEDROCK_MODEL})')

    timings = {}
    t_total = time.time()

    # Load game
    t0 = time.time()
    with open(args.game) as f:
        moves = json.load(f)
    if isinstance(moves, dict):
        moves = moves.get('moves', [])
    timings['load_game'] = round(time.time() - t0, 2)

    # Find biggest mistakes (played != best)
    mistakes = [m for m in moves if m.get('uci') != m.get('best_uci') and m.get('cp_loss', 0) >= 50]
    mistakes.sort(key=lambda m: -m['cp_loss'])
    mistakes = mistakes[:args.top_n]
    print(f'Game: {len(moves)} moves, {len(mistakes)} mistakes (top {args.top_n})')
    print()

    # Narrative (ground truth — no SAE bias)
    narratives = {}
    if not args.no_narrative and mistakes:
        t0 = time.time()
        print('Getting Sonnet narrative (ground truth)...')
        narratives = get_narratives(mistakes)
        timings['narrative'] = round(time.time() - t0, 2)
        print(f'  {len(narratives)} narratives ({timings["narrative"]:.1f}s)')
        print()

    # Load encoder
    t0 = time.time()
    enc_path = args.encoder or DEFAULT_ENCODER
    mm_path = args.move_map or DEFAULT_MOVE_MAP
    print(f'Loading encoder...')
    encoder = Encoder(enc_path, mm_path)
    timings['load_encoder'] = round(time.time() - t0, 2)
    print(f'  Done ({timings["load_encoder"]:.1f}s)')

    # Load SAEs + profiles
    t0 = time.time()
    sae_configs = []
    for i, sae_path in enumerate(args.sae):
        print(f'Loading SAE: {sae_path}')
        sae, mean, std, dd, k, name = load_sae(sae_path)
        profiles = {}
        if i < len(args.profiles) and args.profiles[i]:
            print(f'  Loading profiles: {args.profiles[i]}')
            with open(args.profiles[i]) as f:
                profiles = json.load(f)
            print(f'  {len(profiles)} feature profiles')
        sae_configs.append({
            'sae': sae, 'mean': mean, 'std': std,
            'dd': dd, 'k': k, 'name': name, 'profiles': profiles,
        })
    timings['load_saes'] = round(time.time() - t0, 2)
    print(f'  SAEs loaded ({timings["load_saes"]:.1f}s)')
    print()

    # Process each mistake
    t_encode = 0
    t_sae = 0
    t_label = 0
    results = []

    for m in mistakes:
        ply = m['ply']
        narrative = narratives.get(ply, '')

        print('=' * 70)
        print(f"Move {ply}: {m['san']} ({m['side']}) — {m['cp_loss']}cp loss")
        print(f"  Played: {m['uci']}  Best: {m['best_uci']}")
        if narrative:
            print(f"  WHAT HAPPENED: {narrative}")
        print()

        # Encode played + best
        t0 = time.time()
        h_played = encoder.encode(m['fen'], m['uci'])
        h_best = encoder.encode(m['fen'], m['best_uci']) if m['best_uci'] != m['uci'] else None
        t_encode += time.time() - t0

        if h_played is None:
            print(f"  Could not encode played move")
            continue

        mistake_result = {
            'ply': ply, 'san': m['san'], 'side': m['side'], 'cp_loss': m['cp_loss'],
            'played': m['uci'], 'best': m['best_uci'], 'fen': m['fen'],
            'narrative': narrative, 'saes': {},
        }

        for sc in sae_configs:
            sae_name = sc['name']

            # SAE inference
            t0 = time.time()
            feats_played = run_sae(h_played, sc['sae'], sc['mean'], sc['std'])
            feats_best = run_sae(h_best, sc['sae'], sc['mean'], sc['std']) if h_best is not None else []
            t_sae += time.time() - t0

            played_ids = {f[0] for f in feats_played}
            best_ids = {f[0] for f in feats_best}
            shared = played_ids & best_ids
            only_played = [(fid, s) for fid, s in feats_played if fid not in best_ids]
            only_best = [(fid, s) for fid, s in feats_best if fid not in played_ids]

            # Sort by strength, take top diff-k from each side
            only_played.sort(key=lambda x: -x[1])
            only_best.sort(key=lambda x: -x[1])
            top_played = only_played[:args.diff_k]
            top_best = only_best[:args.diff_k]

            # Label diff features (parallel)
            diff_labels = {}
            if not args.no_labels:
                t0 = time.time()
                to_label = [(fid, s, True) for fid, s in top_played] + \
                           [(fid, s, False) for fid, s in top_best]
                diff_labels = label_diff_features_parallel(
                    to_label, m, sc['profiles'], max_workers=args.workers)
                t_label += time.time() - t0

            # Print results
            print(f"  {sae_name} (dict={sc['dd']}, k={sc['k']}):")
            print(f"    Shared: {len(shared)}  Diff: {len(only_played)}+{len(only_best)}")

            if top_played:
                print(f"    BLUNDER activated (top {len(top_played)}):")
                for fid, strength in top_played:
                    lbl = diff_labels.get(fid)
                    lbl_str = f" → {lbl['label']} [{lbl['category']}]" if lbl else ""
                    print(f"      F{fid} (str={strength}){lbl_str}")

            if top_best:
                print(f"    BEST MOVE activated (top {len(top_best)}):")
                for fid, strength in top_best:
                    lbl = diff_labels.get(fid)
                    lbl_str = f" → {lbl['label']} [{lbl['category']}]" if lbl else ""
                    print(f"      F{fid} (str={strength}){lbl_str}")

            print()

            mistake_result['saes'][sae_name] = {
                'dict_size': sc['dd'], 'k': sc['k'],
                'n_shared': len(shared),
                'n_only_played': len(only_played),
                'n_only_best': len(only_best),
                'top_played': [{'fid': f, 'strength': s, **(diff_labels.get(f, {}))} for f, s in top_played],
                'top_best': [{'fid': f, 'strength': s, **(diff_labels.get(f, {}))} for f, s in top_best],
            }

        results.append(mistake_result)

    # Summary
    timings['encode'] = round(t_encode, 2)
    timings['sae_inference'] = round(t_sae, 2)
    timings['labeling'] = round(t_label, 2)
    timings['total'] = round(time.time() - t_total, 2)

    print('=' * 70)
    print('SUMMARY')
    print()

    for sc in sae_configs:
        name = sc['name']
        total_labeled = 0
        categories = {}
        for r in results:
            sae_data = r['saes'].get(name, {})
            for feat in sae_data.get('top_played', []) + sae_data.get('top_best', []):
                if 'label' in feat:
                    total_labeled += 1
                    cat = feat.get('category', 'other')
                    categories[cat] = categories.get(cat, 0) + 1

        n_diff = sum(
            r['saes'].get(name, {}).get('n_only_played', 0) +
            r['saes'].get(name, {}).get('n_only_best', 0)
            for r in results
        )
        print(f"  {name} (dict={sc['dd']}, k={sc['k']}):")
        print(f"    Total diff features: {n_diff} across {len(results)} mistakes")
        print(f"    Labeled: {total_labeled}")
        if categories:
            print(f"    Categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")
        print()

    print('TIMING:')
    for step, t in timings.items():
        pct = round(t / max(timings['total'], 0.01) * 100)
        bar = '█' * max(1, pct // 5)
        print(f'  {step:<20} {t:>6.1f}s  {pct:>3}% {bar}')

    if args.output:
        out = {'results': results, 'timings': timings,
               'saes': [{'name': sc['name'], 'dd': sc['dd'], 'k': sc['k']} for sc in sae_configs]}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
