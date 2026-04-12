#!/usr/bin/env python3
"""Evaluate LoRA Phase 2 model with contrastive projection.

Usage:
  python3 eval_lora.py \
    --lora-dir output/lora \
    --qwen models/qwen2.5-7b \
    --projection output/contrastive/projection.pt \
    --encoder /tmp/chess_encoder_270m.pt \
    --data data/eval_positions.jsonl \
    --n-eval 200 \
    --ablation  # optional: zero out encoder embeddings
"""
import sys, os, json, argparse, re, time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fen_tokenizer import tokenize as chess_tokenize
from chess_model import ChessEncoder
from projection_layer import ChessProjection


def parse_output(text):
    result = {}
    best = re.search(r'Best:\s*(\S+)', text)
    if best:
        result['best_move'] = best.group(1).rstrip('.').rstrip(',')
    eval_m = re.search(r'Eval:\s*([+-]?\d+\.?\d*)', text)
    if eval_m:
        result['eval'] = float(eval_m.group(1))
    cls_m = re.search(r'Classification:\s*(\w+)', text)
    if cls_m:
        result['classification'] = cls_m.group(1)
    return result


def is_legal_move(fen, move_str):
    try:
        import chess
        board = chess.Board(fen)
        try:
            return chess.Move.from_uci(move_str) in board.legal_moves
        except:
            pass
        try:
            board.parse_san(move_str)
            return True
        except:
            return False
    except ImportError:
        return None


def generate(model, projection, proj_norm, encoder, tokenizer, fen, device, ablation=False, scale_factor=1.0):
    parts = fen.split()
    if len(parts) == 4: fen += ' 0 1'
    elif len(parts) == 5: fen += ' 1'

    fen_tokens = torch.tensor(chess_tokenize(fen).astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        chess_hidden = encoder(fen_tokens)
        if ablation:
            chess_hidden = torch.zeros_like(chess_hidden)
        chess_proj = (proj_norm(projection(chess_hidden.float())) * scale_factor).to(torch.bfloat16)

    prompt = "Analyze this position."
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    prompt_emb = model.get_input_embeddings()(prompt_ids)
    combined = torch.cat([chess_proj, prompt_emb], dim=1)

    generated_ids = []
    with torch.no_grad():
        out = model(inputs_embeds=combined, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        generated_ids.append(next_tok.item())

        for _ in range(127):
            tok_emb = model.get_input_embeddings()(next_tok.unsqueeze(0))
            out = model(inputs_embeds=tok_emb, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            tid = next_tok.item()
            generated_ids.append(tid)
            if tid == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-dir', required=True, help='Path to LoRA adapter dir')
    parser.add_argument('--qwen', required=True, help='Base Qwen model path')
    parser.add_argument('--projection', required=True, help='Contrastive projection.pt')
    parser.add_argument('--encoder', required=True, help='Chess encoder checkpoint')
    parser.add_argument('--data', required=True, help='Eval JSONL')
    parser.add_argument('--n-eval', type=int, default=200)
    parser.add_argument('--output', default='eval_lora_results.json')
    parser.add_argument('--ablation', action='store_true', help='Zero out encoder embeddings')
    parser.add_argument('--show-samples', type=int, default=10, help='Print first N samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encoder (frozen)
    print("Loading encoder...", flush=True)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    # Base LLM + LoRA
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model %s..." % args.qwen, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    print("Loading LoRA adapter from %s..." % args.lora_dir, flush=True)
    llm = PeftModel.from_pretrained(llm, args.lora_dir)
    llm.eval()

    # Projection (frozen contrastive)
    config = AutoConfig.from_pretrained(args.qwen, trust_remote_code=True)
    projection = ChessProjection(encoder_dim=1024, llm_dim=config.hidden_size).to(device)
    proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
    projection.load_state_dict(proj_ckpt['state_dict'])
    projection.eval()

    # LayerNorm (trained during Phase 2)
    proj_norm = nn.LayerNorm(config.hidden_size).to(device)  # float32
    norm_ckpt = torch.load(os.path.join(args.lora_dir, 'proj_norm.pt'), map_location=device, weights_only=False)
    proj_norm.load_state_dict(norm_ckpt['state_dict'] if 'state_dict' in norm_ckpt else norm_ckpt)
    proj_norm.eval()
    scale_factor = norm_ckpt.get('scale_factor', 1.0) if isinstance(norm_ckpt, dict) else 1.0
    print("Scale factor: %.4f" % scale_factor, flush=True)

    print("GPU mem: %.1fGB" % (torch.cuda.memory_allocated(device) / 1e9), flush=True)
    if args.ablation:
        print("*** ABLATION MODE: encoder embeddings zeroed ***", flush=True)

    # Load eval data
    data = [json.loads(l) for l in Path(args.data).read_text().strip().split('\n')]
    eval_data = data[:args.n_eval]
    print("Evaluating %d positions..." % len(eval_data), flush=True)

    metrics = {'best_move_correct': 0, 'eval_dir_correct': 0, 'move_legal': 0,
               'format_valid': 0, 'total': 0, 'eval_errors': [], 'samples': []}

    t0 = time.time()
    for i, item in enumerate(eval_data):
        fen = item.get('fen', '')
        true_best = item.get('best_move', item.get('best_move_uci', ''))
        true_eval = item.get('eval_sf', item.get('eval', 0))
        if not fen or not true_best:
            continue

        try:
            pred_text = generate(llm, projection, proj_norm, encoder, tokenizer, fen, device,
                                ablation=args.ablation, scale_factor=scale_factor)
            parsed = parse_output(pred_text)
            metrics['total'] += 1

            has_best = bool(parsed.get('best_move'))
            if has_best:
                metrics['format_valid'] += 1

            pred_best = parsed.get('best_move', '')
            best_correct = pred_best.lower() == true_best.lower()
            if best_correct:
                metrics['best_move_correct'] += 1

            legal = is_legal_move(fen, pred_best)
            if legal:
                metrics['move_legal'] += 1

            pred_eval = parsed.get('eval', 0)
            eval_correct = False
            if true_eval is not None:
                if (pred_eval > 0) == (true_eval > 0) or (pred_eval == 0 and true_eval == 0):
                    eval_correct = True
                    metrics['eval_dir_correct'] += 1
                if pred_eval is not None:
                    metrics['eval_errors'].append(abs(pred_eval - true_eval))

            # Print samples
            if i < args.show_samples:
                print(f"\n--- Sample {i+1} ---", flush=True)
                print(f"  FEN: {fen}", flush=True)
                print(f"  True: best={true_best} eval={true_eval}", flush=True)
                print(f"  Pred: {pred_text[:200]}", flush=True)
                print(f"  Parsed: {parsed}", flush=True)
                print(f"  Match: best={'Y' if best_correct else 'N'} legal={'Y' if legal else 'N'} "
                      f"eval_dir={'Y' if eval_correct else 'N'}", flush=True)

            if (i + 1) % 20 == 0:
                n = metrics['total']
                elapsed = time.time() - t0
                rate = n / elapsed
                print(f"  [{i+1}/{len(eval_data)}] best={metrics['best_move_correct']/n:.0%} "
                      f"legal={metrics['move_legal']/n:.0%} eval_dir={metrics['eval_dir_correct']/n:.0%} "
                      f"format={metrics['format_valid']/n:.0%} ({rate:.1f} pos/s)", flush=True)
        except Exception as e:
            print(f"  Error {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    n = max(metrics['total'], 1)
    elapsed = time.time() - t0
    results = {
        'n': metrics['total'],
        'best_move_accuracy': round(metrics['best_move_correct'] / n, 3),
        'move_legality': round(metrics['move_legal'] / n, 3),
        'eval_direction': round(metrics['eval_dir_correct'] / n, 3),
        'format_validity': round(metrics['format_valid'] / n, 3),
        'eval_mae': round(float(np.mean(metrics['eval_errors'])), 2) if metrics['eval_errors'] else None,
        'ablation': args.ablation,
        'elapsed_s': round(elapsed, 1),
    }

    print(f"\n{'='*50}", flush=True)
    mode = "ABLATION (zeroed encoder)" if args.ablation else "NORMAL"
    print(f"RESULTS — {mode} ({results['n']} positions, {elapsed:.0f}s)", flush=True)
    print(f"  Best move:      {results['best_move_accuracy']:.1%}", flush=True)
    print(f"  Move legality:  {results['move_legality']:.1%}", flush=True)
    print(f"  Eval direction: {results['eval_direction']:.1%}", flush=True)
    print(f"  Format valid:   {results['format_validity']:.1%}", flush=True)
    if results['eval_mae'] is not None:
        print(f"  Eval MAE:       {results['eval_mae']:.2f}", flush=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
