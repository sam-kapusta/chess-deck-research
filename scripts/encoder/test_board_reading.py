#!/usr/bin/env python3
"""Board Reading Test: Can the LLM read individual piece positions through the encoder?

Tests whether the translator carries spatial information or just eval direction.
Gives the model a position via encoder tokens and asks "What piece is on [square]?"

Interpretation:
  >70%: translator carries rich spatial info, problem is LLM training data
  30-70%: partial spatial signal, per-token alignment would help
  <20% (random ~8%): translator is nearly empty, must retrain before anything else

Usage:
    # On SAIS notebook (chess-research):
    python test_board_reading.py --n 200
    python test_board_reading.py --n 200 --ablation  # zeroed encoder baseline
"""
import sys
import json
import argparse
import random
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder')
sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder/scripts')

from searchless_chess.src.tokenizer import tokenize as chess_tokenize

# Square names for board positions
SQUARES = [f"{file}{rank}" for rank in range(8, 0, -1) for file in "abcdefgh"]
# Token 0 = side to move, tokens 1-64 = board squares (a8, b8, ..., h1)

PIECE_NAMES = {
    '.': 'empty', 'P': 'white pawn', 'N': 'white knight', 'B': 'white bishop',
    'R': 'white rook', 'Q': 'white queen', 'K': 'white king',
    'p': 'black pawn', 'n': 'black knight', 'b': 'black bishop',
    'r': 'black rook', 'q': 'black queen', 'k': 'black king'
}

PIECE_LABELS = list(PIECE_NAMES.values())  # 13 classes


def fen_to_board(fen: str) -> dict:
    """Parse FEN to {square: piece} dict."""
    board_part = fen.split(' ')[0]
    board = {}
    rank = 8
    file_idx = 0
    for char in board_part:
        if char == '/':
            rank -= 1
            file_idx = 0
        elif char.isdigit():
            for _ in range(int(char)):
                square = f"{'abcdefgh'[file_idx]}{rank}"
                board[square] = '.'
                file_idx += 1
        else:
            square = f"{'abcdefgh'[file_idx]}{rank}"
            board[square] = char
            file_idx += 1
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200, help='Number of test positions')
    parser.add_argument('--ablation', action='store_true', help='Zero encoder tokens')
    parser.add_argument('--positions', default='/home/ec2-user/SageMaker/chess-research/data/eval_positions.jsonl')
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/models/qwen2.5-7b')
    parser.add_argument('--lora', default='/home/ec2-user/SageMaker/chess-stage-a/output/lora_v2')
    parser.add_argument('--contrastive', default='/home/ec2-user/SageMaker/chess-stage-a/output/contrastive_projection.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    from convert_and_validate_v2 import ChessEncoder
    from projection_layer import ChessProjection

    print("Loading encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    print("Loading projection...")
    proj_ckpt = torch.load(args.contrastive, map_location=device, weights_only=False)
    projection = ChessProjection(
        encoder_dim=proj_ckpt['config']['encoder_dim'],
        llm_dim=proj_ckpt['config']['llm_dim']
    ).to(device).float()
    projection.load_state_dict(proj_ckpt['state_dict'])
    projection.eval()

    # Norm scaling
    proj_norm = torch.nn.LayerNorm(proj_ckpt['config']['llm_dim']).to(device).float()
    scale_factor = 0.0157  # target_norm / sqrt(dim)

    print("Loading Qwen + LoRA...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    llm = PeftModel.from_pretrained(llm, args.lora)
    llm.eval()

    # Load test positions
    positions = [json.loads(l) for l in Path(args.positions).read_text().strip().split('\n')]
    random.shuffle(positions)
    positions = positions[:args.n]

    print(f"\nTesting {len(positions)} positions {'(ABLATION)' if args.ablation else ''}")
    print("=" * 60)

    correct = 0
    total = 0
    by_piece = {}  # accuracy per piece type

    for i, pos in enumerate(positions):
        fen = pos['fen']
        board = fen_to_board(fen)

        # Pick a random square to ask about
        square = random.choice(SQUARES)
        true_piece = PIECE_NAMES[board[square]]

        # Encode position
        fen_tokens = torch.tensor(
            chess_tokenize(fen).astype(np.int64), dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            chess_hidden = encoder(fen_tokens)
            chess_projected = projection(chess_hidden.float())
            chess_projected = proj_norm(chess_projected) * scale_factor

            if args.ablation:
                chess_projected = torch.zeros_like(chess_projected)

            # Build prompt
            prompt = f"What piece is on {square}? Answer with one of: {', '.join(PIECE_LABELS)}. Answer:"
            prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

            # Get text embeddings
            text_embeds = llm.get_input_embeddings()(prompt_ids)

            # Combine: chess tokens + text prompt
            combined = torch.cat([chess_projected.half(), text_embeds], dim=1)

            # Generate
            generated_ids = []
            past_kv = None
            input_embeds = combined

            for step in range(20):
                outputs = llm(inputs_embeds=input_embeds, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                next_id = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_ids.append(next_id.item())
                if next_id.item() == tokenizer.eos_token_id:
                    break
                input_embeds = llm.get_input_embeddings()(next_id.unsqueeze(0))

            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

        # Match prediction to piece labels
        pred_piece = None
        for label in PIECE_LABELS:
            if label.lower() in pred_text:
                pred_piece = label
                break

        if pred_piece is None:
            pred_piece = pred_text[:30]  # raw output for debugging

        is_correct = (pred_piece == true_piece)
        correct += int(is_correct)
        total += 1

        # Track per-piece accuracy
        if true_piece not in by_piece:
            by_piece[true_piece] = {'correct': 0, 'total': 0}
        by_piece[true_piece]['total'] += 1
        by_piece[true_piece]['correct'] += int(is_correct)

        if i < 10 or (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(positions)}] {square}: true={true_piece}, pred={pred_piece}, {'✓' if is_correct else '✗'}")
            if (i + 1) % 50 == 0:
                print(f"  Running accuracy: {correct/total:.1%}")

    # Results
    print("\n" + "=" * 60)
    print(f"BOARD READING TEST {'(ABLATION)' if args.ablation else ''}")
    print(f"=" * 60)
    print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"Random baseline: ~8% (1/13)")
    print()

    if correct / total > 0.70:
        print("RESULT: >70% — Translator carries rich spatial info.")
        print("NEXT: Scale LLM training data. Translator is fine.")
    elif correct / total > 0.30:
        print("RESULT: 30-70% — Partial spatial signal.")
        print("NEXT: Per-token alignment would help.")
    elif correct / total > 0.20:
        print("RESULT: 20-30% — Weak signal.")
        print("NEXT: Retrain translator with spatial objectives.")
    else:
        print("RESULT: <20% — Near random. Translator carries almost nothing.")
        print("NEXT: Must retrain translator before any LLM training matters.")

    print(f"\nPer-piece accuracy:")
    for piece in sorted(by_piece.keys()):
        d = by_piece[piece]
        pct = d['correct'] / d['total'] if d['total'] > 0 else 0
        print(f"  {piece:15s}: {d['correct']:3d}/{d['total']:3d} = {pct:.0%}")

    # Save results
    results = {
        'overall_accuracy': correct / total,
        'correct': correct,
        'total': total,
        'ablation': args.ablation,
        'by_piece': {k: v['correct'] / v['total'] for k, v in by_piece.items()},
        'interpretation': 'spatial' if correct/total > 0.7 else 'partial' if correct/total > 0.3 else 'weak' if correct/total > 0.2 else 'empty'
    }
    out_path = f'/home/ec2-user/SageMaker/chess-stage-a/output/board_reading_{"ablation" if args.ablation else "normal"}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
