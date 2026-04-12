#!/usr/bin/env python3
"""Generate Stage A predictions on eval set, then score with Stockfish.

Loads the Stage A model (encoder + projection + LoRA Qwen), generates
structured predictions for each eval position, saves results, and runs
Stockfish verification.

Usage:
  python generate_stage_a.py
  python generate_stage_a.py --eval-set /path/to/eval_set.json --verbose
"""
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder')
sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder/scripts')

from searchless_chess.src.tokenizer import tokenize as chess_tokenize
from convert_and_validate_v2 import ChessEncoder
from projection_layer import ChessProjection

CHECKPOINTS = '/home/ec2-user/SageMaker/chess-research/checkpoints'
MODELS = '/home/ec2-user/SageMaker/chess-research/models'
ENCODER_PATH = '/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt'


def load_model(args, device):
    """Load encoder + projection + LoRA Qwen."""
    # Encoder
    print("Loading encoder...")
    ckpt = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    # Projection
    print("Loading projection...")
    proj_path = f"{args.checkpoint}/projection_stage_a.pt"
    from transformers import AutoConfig
    qwen_config = AutoConfig.from_pretrained(f"{MODELS}/qwen2.5-7b", trust_remote_code=True)
    hidden_dim = qwen_config.hidden_size

    projection = ChessProjection(encoder_dim=1024, llm_dim=hidden_dim).to(device)
    if Path(proj_path).exists():
        proj_ckpt = torch.load(proj_path, map_location=device, weights_only=False)
        projection.load_state_dict(proj_ckpt['state_dict'])
    projection.eval()

    # Qwen + LoRA
    print("Loading Qwen + LoRA...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(f"{MODELS}/qwen2.5-7b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        f"{MODELS}/qwen2.5-7b", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    lora_path = f"{args.checkpoint}/qwen_lora_stage_a"
    if Path(lora_path).exists():
        llm = PeftModel.from_pretrained(llm, lora_path)
    llm.eval()

    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return encoder, projection, llm, tokenizer


@torch.no_grad()
def generate_prediction(encoder, projection, llm, tokenizer, fen, device, max_new_tokens=100):
    """Generate structured prediction for a position."""
    # Encode chess position
    fen_tokens = torch.tensor(chess_tokenize(fen).astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
    chess_hidden = encoder(fen_tokens)
    chess_projected = projection(chess_hidden.float()).to(llm.dtype)

    # Text prompt
    prompt = "Analyze this position."
    text_encoded = tokenizer(prompt, return_tensors='pt').to(device)
    text_embeds = llm.get_input_embeddings()(text_encoded['input_ids'])

    # Combine
    combined = torch.cat([chess_projected, text_embeds], dim=1)
    chess_mask = torch.ones(1, 77, dtype=torch.long, device=device)
    combined_mask = torch.cat([chess_mask, text_encoded['attention_mask']], dim=1)

    # Generate
    outputs = llm.generate(
        inputs_embeds=combined,
        attention_mask=combined_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=f'{CHECKPOINTS}/stage_a')
    parser.add_argument('--eval-set', default='/home/ec2-user/SageMaker/chess-research/data/stage_a_eval_set.json')
    parser.add_argument('--output', default=f'{CHECKPOINTS}/stage_a_results.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, projection, llm, tokenizer = load_model(args, device)

    # Load eval set
    eval_data = json.loads(Path(args.eval_set).read_text())
    print(f"\nGenerating predictions for {len(eval_data)} positions...")

    results = []
    for i, item in enumerate(eval_data):
        prediction = generate_prediction(encoder, projection, llm, tokenizer, item['fen'], device)

        result = {**item, 'generated': prediction}
        results.append(result)

        if args.verbose and i < 5:
            print(f"\n--- Position {i+1} ---")
            print(f"  FEN: {item['fen'][:50]}...")
            print(f"  Truth: Best={item.get('best_move')}, Eval={item.get('eval')}")
            print(f"  Model: {prediction[:150]}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_data)} done")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")

    # Run Stockfish eval
    print("\n=== Stockfish Evaluation ===")
    from eval_stage_a import evaluate
    evaluate(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
