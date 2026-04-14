#!/usr/bin/env python3
"""Experiment 15: Do phase-neutral features show phase preference by activation strength?

Hypothesis: Features that fire equally (binary) across phases have different strengths by phase.
Prediction: >30% of phase-neutral features have >2x strength ratio between phases.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess


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


def main():
    print('Experiment 15: Activation strength by phase for phase-neutral features')
    print('Hypothesis: Phase-neutral features have different STRENGTH by phase even if equal FIRE rate')
    print('Prediction: >30% have >2x strength ratio between strongest and weakest phase')
    print()

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:10000].float()
    metadata = cache['metadata'][:10000]

    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_2048_k32_aux.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 2048, 32)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    acts_np = acts.numpy()  # continuous activations
    fires = (acts_np > 0).astype(np.float32)

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Classify phases
    phase_idx = {'opening': [], 'middlegame': [], 'endgame': []}
    for i, md in enumerate(metadata):
        try:
            n = len(chess.Board(md['fen']).piece_map())
        except:
            n = 20
        p = 'opening' if n > 24 else ('middlegame' if n > 12 else 'endgame')
        phase_idx[p].append(i)

    print('Positions: ' + ', '.join(p + '=' + str(len(idx)) for p, idx in phase_idx.items()))

    # Find phase-neutral features (binary fire 15-55% each phase)
    neutral_fids = []
    for fid in range(2048):
        total = fires[:, fid].sum()
        if total < 20:
            continue
        lbl = labels.get(str(fid), {})
        if lbl.get('confidence') not in ['high', 'medium']:
            continue
        ratios = {p: fires[idx, fid].sum() / total for p, idx in phase_idx.items()}
        if all(0.15 < r < 0.55 for r in ratios.values()):
            neutral_fids.append(fid)

    print('Phase-neutral features: ' + str(len(neutral_fids)))
    print()

    # For each: compute MEAN activation strength per phase (among positions where it fires)
    strength_ratios = []
    phase_preference = {'opening': 0, 'middlegame': 0, 'endgame': 0}

    strong_examples = []  # features with >2x ratio

    for fid in neutral_fids:
        phase_strengths = {}
        for phase, idx in phase_idx.items():
            phase_acts = acts_np[idx, fid]
            firing = phase_acts[phase_acts > 0]
            phase_strengths[phase] = firing.mean() if len(firing) > 0 else 0

        max_phase = max(phase_strengths, key=phase_strengths.get)
        min_phase = min(phase_strengths, key=phase_strengths.get)
        ratio = phase_strengths[max_phase] / max(phase_strengths[min_phase], 1e-8)

        strength_ratios.append(ratio)
        phase_preference[max_phase] += 1

        if ratio > 2.0:
            lbl = labels.get(str(fid), {}).get('label', '?')[:50]
            strong_examples.append((fid, ratio, max_phase, phase_strengths, lbl))

    strength_ratios = np.array(strength_ratios)

    print('=== Strength ratio distribution ===')
    print('  Mean ratio (max/min phase): ' + str(round(strength_ratios.mean(), 2)))
    print('  Median: ' + str(round(np.median(strength_ratios), 2)))
    print('  >1.5x: ' + str((strength_ratios > 1.5).sum()) + ' (' + str(round((strength_ratios > 1.5).mean() * 100, 1)) + '%)')
    print('  >2.0x: ' + str((strength_ratios > 2.0).sum()) + ' (' + str(round((strength_ratios > 2.0).mean() * 100, 1)) + '%)')
    print('  >3.0x: ' + str((strength_ratios > 3.0).sum()) + ' (' + str(round((strength_ratios > 3.0).mean() * 100, 1)) + '%)')
    print()

    print('Phase preference (which phase has strongest activation):')
    for phase, n in sorted(phase_preference.items(), key=lambda x: -x[1]):
        print('  ' + phase + ': ' + str(n) + ' (' + str(round(n / len(neutral_fids) * 100, 1)) + '%)')
    print()

    print('Top features with >2x strength ratio:')
    for fid, ratio, max_p, strengths, lbl in sorted(strong_examples, key=lambda x: -x[1])[:15]:
        s_str = ', '.join(p + '=' + str(round(s, 2)) for p, s in strengths.items())
        print('  F' + str(fid) + ' (' + str(round(ratio, 1)) + 'x, strongest=' + max_p + ') ' + lbl)
        print('    Strengths: ' + s_str)

    # Verdict
    pct_above_2x = (strength_ratios > 2.0).mean() * 100
    print()
    print('=== Verdict ===')
    print('Phase-neutral features with >2x strength ratio: ' + str(round(pct_above_2x, 1)) + '%')
    print('Prediction was >30%: ' + ('CONFIRMED' if pct_above_2x > 30 else 'FAILED'))


if __name__ == '__main__':
    main()
