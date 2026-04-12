#!/usr/bin/env python3
"""Stage A: Train model to predict chess truths from encoder embeddings.

Architecture:
  FEN → encoder (frozen) → aligned projection → [77, hidden_dim]
  Text prompt → Qwen tokenizer → embeddings → [T, hidden_dim]
  Combined: [chess_prefix, text_prompt] → Qwen (full FT) → structured prediction

Target output format:
  "Best: Nxd5 (+2.1). Line: Nxd5 exd5 Bxd5 Qe7 Bxf7+. Alt: Bg5 (+1.4), Re1 (+1.2)"

Training data: positions with Stockfish labels (eval, best_move, pv_line, alternatives)

Usage:
  python train_stage_a.py --data /path/to/stage_a_positions_labeled.jsonl
  python train_stage_a.py --data /path/to/data.jsonl --epochs 3 --lr 1e-5
"""
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder')
sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder/scripts')

from searchless_chess.src.tokenizer import tokenize as chess_tokenize
from convert_and_validate_v2 import ChessEncoder
from projection_layer import ChessProjection


def format_target(pos):
    """Format Stockfish labels as target text for the model.

    Target format:
      "Best: Nxd5 (+2.1). Played: Qd7 (-0.8). Classification: blunder. Line: Nxd5 exd5 Bxd5. Alt: Bg5 (+1.4)"

    The model predicts: best move, eval, played move assessment, classification, PV line.
    All verifiable by Stockfish.
    """
    parts = []

    best = pos.get("best_move", "")
    ev = pos.get("eval_sf", pos.get("eval", 0))
    played = pos.get("played_move", "")
    classification = pos.get("classification", "")
    pv = pos.get("pv_line", "")
    alts = pos.get("alternatives", [])

    if best:
        parts.append(f"Best: {best} ({ev:+.1f})")

    if played and played != best:
        parts.append(f"Played: {played}")

    if classification and classification not in ("opening", "excellent", "good"):
        parts.append(f"Classification: {classification}")

    if pv:
        parts.append(f"Line: {pv}")

    if alts:
        alt_strs = [f"{a.get('move', a.get('san', '?'))} ({a.get('eval', 0):+.1f})" for a in alts[:3] if a.get('move') or a.get('san')]
        if alt_strs:
            parts.append(f"Alt: {', '.join(alt_strs)}")

    return ". ".join(parts)


class StageADataset(Dataset):
    """Positions with Stockfish labels for structured prediction."""

    def __init__(self, data_path, tokenizer, max_text_len=128):
        raw = [json.loads(l) for l in Path(data_path).read_text().strip().split('\n')]
        # Filter: must have best_move and FEN
        self.data = [d for d in raw if d.get('best_move') and d.get('fen')]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Pre-tokenize all examples
        self.examples = []
        for item in self.data:
            try:
                fen_tokens = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)

                # Prompt: just "Analyze this position."
                prompt = "Analyze this position."
                target = format_target(item)
                full_text = prompt + " " + target

                text_encoded = self.tokenizer(
                    full_text, max_length=self.max_text_len,
                    padding='max_length', truncation=True, return_tensors='pt',
                )

                prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids']) + 1
                labels = text_encoded['input_ids'].squeeze().clone()
                labels[:prompt_len] = -100
                labels[text_encoded['attention_mask'].squeeze() == 0] = -100

                self.examples.append({
                    'fen_tokens': fen_tokens,
                    'text_input_ids': text_encoded['input_ids'].squeeze(),
                    'text_attention_mask': text_encoded['attention_mask'].squeeze(),
                    'labels': labels,
                })
            except:
                continue

        print(f"  Loaded {len(self.examples)} examples (from {len(raw)} raw, {len(self.data)} with best_move)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(encoder, projection, llm, tokenizer, dataset, args, device):
    """Train Stage A: predict chess truths from encoder embeddings."""
    print(f"=== Stage A: Structured Chess Prediction ===")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # LoRA on LLM (full FT needs multi-GPU — LoRA fits on A10G)
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=64,  # rank 64 (C1 showed +6.7% over rank 16)
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    projection.train()
    llm.train()

    if hasattr(llm, 'gradient_checkpointing_enable'):
        llm.gradient_checkpointing_enable()

    trainable_params = list(projection.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       drop_last=True, num_workers=0, pin_memory=True)

    total_steps = args.epochs * len(loader) // args.grad_accum
    warmup = total_steps // 10

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Dataset: {len(dataset)}, Steps/epoch: {len(loader)}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    for epoch in range(args.epochs):
        total_loss = 0
        num_valid = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            fen = batch['fen_tokens'].to(device)
            txt = batch['text_input_ids'].to(device)
            mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            bs = fen.shape[0]

            with torch.no_grad():
                chess_hidden = encoder(fen)
            chess_proj = projection(chess_hidden.float()).to(llm.dtype)

            txt_emb = llm.get_input_embeddings()(txt)
            combined = torch.cat([chess_proj, txt_emb], dim=1)
            cmask = torch.cat([torch.ones(bs, 77, dtype=torch.long, device=device), mask], dim=1)
            clabels = torch.cat([torch.full((bs, 77), -100, dtype=torch.long, device=device), labels], dim=1)

            outputs = llm(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
            loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            num_valid += 1

            if (step + 1) % 200 == 0:
                avg = total_loss / num_valid
                print(f"  E{epoch+1} | {step+1}/{len(loader)} | Loss: {avg:.4f}")

        elapsed = time.time() - t0
        avg = total_loss / max(num_valid, 1)
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg:.4f} | Time: {elapsed:.0f}s")

    return projection, llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--projection', default='/home/ec2-user/SageMaker/chess-research/checkpoints/projection_contrastive_3b.pt')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/chess-research/models/qwen2.5-3b')
    parser.add_argument('--output', default='/home/ec2-user/SageMaker/chess-research/checkpoints/stage_a')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load encoder
    print("Loading encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    # Load projection
    print("Loading projection...")
    from transformers import AutoConfig
    qwen_config = AutoConfig.from_pretrained(args.qwen, trust_remote_code=True)
    hidden_dim = qwen_config.hidden_size
    print(f"  Qwen hidden_dim: {hidden_dim}")

    projection = ChessProjection(encoder_dim=1024, llm_dim=hidden_dim).to(device)
    if Path(args.projection).exists():
        proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
        projection.load_state_dict(proj_ckpt['state_dict'])
        print(f"  Loaded projection from {args.projection}")
    else:
        print(f"  No projection checkpoint — starting fresh")

    # Load Qwen
    print(f"Loading Qwen from {args.qwen}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    # Load dataset
    print("Loading dataset...")
    dataset = StageADataset(args.data, tokenizer)

    # Train
    projection, llm = train(encoder, projection, llm, tokenizer, dataset, args, device)

    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': projection.state_dict(),
        'config': {'encoder_dim': 1024, 'llm_dim': hidden_dim},
        'stage': 'stage_a',
    }, f"{args.output}/projection_stage_a.pt")
    llm.save_pretrained(f"{args.output}/qwen_lora_stage_a")
    tokenizer.save_pretrained(f"{args.output}/qwen_lora_stage_a")
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
