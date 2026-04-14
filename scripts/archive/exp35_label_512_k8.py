#!/usr/bin/env python3
"""Experiment 35: Profile + label 512 k=8 SAE. Do 16 coaching themes emerge from 143 features?

Steps:
1. Profile: get top-20 FEN examples per feature
2. Check theme keywords in the positions (piece types, phase, etc.)
3. Output profile JSON for Bedrock Batch labeling

Also: run the Lichess theme coverage test on this SAE.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from collections import Counter, defaultdict


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


def get_phase(fen):
    try:
        n = len(chess.Board(fen).piece_map())
    except:
        return 'middlegame'
    if n > 24: return 'opening'
    if n > 12: return 'middlegame'
    return 'endgame'


def get_move_piece(fen, uci):
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        piece = board.piece_at(move.from_square)
        if piece:
            return chess.piece_name(piece.piece_type)
    except:
        pass
    return 'unknown'


def main():
    print('Experiment 35: Profile 512 k=8 SAE')
    print()

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:50000].float()
    metadata = cache['metadata'][:50000]

    ckpt = torch.load('/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/sae_btk_blunder_mt_512_k8.pt',
                       map_location='cpu', weights_only=False)
    sae = SAE(1024, 512, 8)
    sae.load_state_dict(ckpt['model_state_dict'])
    mean = torch.tensor(ckpt['mean'], dtype=torch.float32)
    std = torch.tensor(ckpt['std'], dtype=torch.float32) + 1e-8

    with torch.no_grad():
        _, acts = sae((data - mean) / std)
    acts_np = acts.numpy()
    fires = (acts_np > 0).astype(np.float32)

    n_positions = fires.shape[0]
    print(f'Positions: {n_positions}')
    print(f'Features: 512')
    print(f'Alive: {(fires.sum(axis=0) > 0).sum()}')
    print()

    # Profile each alive feature
    profiles = {}
    for fid in range(512):
        n_fires = int(fires[:, fid].sum())
        if n_fires == 0:
            continue

        # Top 20 by activation strength
        strengths = acts_np[:, fid]
        top_idx = np.argsort(-strengths)[:20]

        examples = []
        phases = Counter()
        pieces = Counter()
        captures = 0
        checks = 0

        for idx in top_idx:
            if strengths[idx] <= 0:
                break
            md = metadata[idx]
            fen = md.get('fen', '')
            blunder_uci = md.get('blunder_uci', '')
            best_uci = md.get('best_uci', '')
            cp_loss = md.get('cp_loss', 0)

            phase = get_phase(fen)
            phases[phase] += 1

            piece = get_move_piece(fen, blunder_uci)
            pieces[piece] += 1

            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(blunder_uci)
                if board.is_capture(move):
                    captures += 1
                board.push(move)
                if board.is_check():
                    checks += 1
            except:
                pass

            examples.append({
                'fen': fen,
                'blunder': blunder_uci,
                'best': best_uci,
                'cp_loss': int(cp_loss) if isinstance(cp_loss, (int, float)) else int(cp_loss) if cp_loss.isdigit() else 200,
                'strength': round(float(strengths[idx]), 2),
            })

        fire_rate = round(n_fires / n_positions * 100, 2)

        profiles[str(fid)] = {
            'examples': examples,
            'fire_rate': fire_rate,
            'n_fires': n_fires,
            'phase_opening': round(phases.get('opening', 0) / max(len(examples), 1) * 100),
            'phase_middlegame': round(phases.get('middlegame', 0) / max(len(examples), 1) * 100),
            'phase_endgame': round(phases.get('endgame', 0) / max(len(examples), 1) * 100),
            'piece_pawn': round(pieces.get('pawn', 0) / max(len(examples), 1) * 100),
            'piece_knight': round(pieces.get('knight', 0) / max(len(examples), 1) * 100),
            'piece_bishop': round(pieces.get('bishop', 0) / max(len(examples), 1) * 100),
            'piece_rook': round(pieces.get('rook', 0) / max(len(examples), 1) * 100),
            'piece_queen': round(pieces.get('queen', 0) / max(len(examples), 1) * 100),
            'piece_king': round(pieces.get('king', 0) / max(len(examples), 1) * 100),
            'captures': round(captures / max(len(examples), 1) * 100),
            'checks': round(checks / max(len(examples), 1) * 100),
        }

    print(f'Profiled {len(profiles)} alive features')

    # Save profiles
    profile_path = '/home/ec2-user/SageMaker/chess-deck-research/output/profiles_512_k8.json'
    with open(profile_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f'Saved profiles to {profile_path}')

    # Fire rate distribution
    rates = [p['fire_rate'] for p in profiles.values()]
    print(f'\nFire rate distribution:')
    print(f'  Mean: {np.mean(rates):.2f}%')
    print(f'  Median: {np.median(rates):.2f}%')
    print(f'  <1%: {sum(1 for r in rates if r < 1)}')
    print(f'  1-5%: {sum(1 for r in rates if 1 <= r < 5)}')
    print(f'  5-10%: {sum(1 for r in rates if 5 <= r < 10)}')
    print(f'  >10%: {sum(1 for r in rates if r >= 10)}')

    # Phase distribution of alive features
    print(f'\nPhase distribution across all features:')
    phase_totals = Counter()
    for p in profiles.values():
        if p['phase_opening'] > 50: phase_totals['opening-dominant'] += 1
        elif p['phase_endgame'] > 50: phase_totals['endgame-dominant'] += 1
        else: phase_totals['mixed'] += 1
    for phase, count in phase_totals.most_common():
        print(f'  {phase}: {count}')

    # Piece distribution
    print(f'\nPiece distribution:')
    piece_totals = Counter()
    for p in profiles.values():
        for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
            if p.get(f'piece_{piece}', 0) > 30:
                piece_totals[piece] += 1
    for piece, count in piece_totals.most_common():
        print(f'  {piece}-related: {count} features')

    # Quick diversity check: are the 143 features diverse?
    print(f'\nCapture rate: {sum(1 for p in profiles.values() if p["captures"] > 50)} features are >50% captures')
    print(f'Check rate: {sum(1 for p in profiles.values() if p["checks"] > 30)} features are >30% checks')

    # Build Bedrock Batch input for labeling
    print(f'\n=== Building labeling batch ===')
    batch_records = []
    for fid_str, prof in profiles.items():
        examples = prof['examples'][:15]
        if len(examples) < 5:
            continue

        fen_list = '\n'.join(
            f'{i+1}. FEN: {ex["fen"]}  Blunder: {ex["blunder"]}  Best: {ex["best"]}  CP loss: {ex["cp_loss"]}'
            for i, ex in enumerate(examples)
        )

        prompt = f"""Analyze these chess positions where SAE feature {fid_str} activates strongly.
All positions are blunders (the played move lost significant evaluation).

{fen_list}

Stats: fire_rate={prof['fire_rate']}%, captures={prof['captures']}%, checks={prof['checks']}%
Phase: opening={prof['phase_opening']}%, middlegame={prof['phase_middlegame']}%, endgame={prof['phase_endgame']}%

What specific chess pattern does this feature detect? Be as specific as possible.
Respond in this exact JSON format:
{{"label": "2-6 word specific label", "category": "one of: fork, pin, skewer, discovered_attack, hanging_piece, deflection, back_rank, trapped_piece, quiet_move, sacrifice, exposed_king, passed_pawn, rook_endgame, pawn_endgame, checkmate, other", "confidence": "high/medium/low", "explanation": "1-2 sentences explaining the pattern"}}"""

        record = {
            'recordId': f'f512k8_{fid_str}',
            'modelInput': {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 300,
                'messages': [{'role': 'user', 'content': prompt}],
            }
        }
        batch_records.append(record)

    batch_path = '/home/ec2-user/SageMaker/chess-deck-research/output/label_512_k8_input.jsonl'
    with open(batch_path, 'w') as f:
        for rec in batch_records:
            f.write(json.dumps(rec) + '\n')

    print(f'Built {len(batch_records)} labeling records')
    print(f'Saved to {batch_path}')
    print(f'\nTo submit: upload to S3 then create Bedrock Batch job')


if __name__ == '__main__':
    main()
