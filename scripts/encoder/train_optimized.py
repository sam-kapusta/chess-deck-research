#!/usr/bin/env python3
"""Stage 2: Optimized Coaching SFT with Lichess study annotations.

Optimizations over train_lichess_stage2.py:
  1. Flash Attention 2 for Qwen — O(n) memory, ~2x attention speedup
  2. bf16 autocast — wider dynamic range than fp16, no GradScaler needed
  3. Pre-tokenized dataset — all tokenization at init, __getitem__ is pure indexing
  4. Multi-worker DataLoader — num_workers=2, pin_memory, prefetch
  5. torch.compile (optional) — kernel fusion, 10-20% speedup after warmup
  6. Tuned grad_accum — adjusted for larger batch_size

Expected: 11h → ~4h for 20K × 2 epochs on A10G.

Usage:
  python train_optimized.py --data /path/to/lichess_studies.jsonl --epochs 2
  python train_optimized.py --data /path/to/data.jsonl --batch-size 2 --grad-accum 2
  python train_optimized.py --data /path/to/data.jsonl --no-compile  # skip torch.compile
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


class PreTokenizedLichessDataset(Dataset):
    """Pre-tokenized Lichess coaching dataset. All tokenization at init.

    __getitem__ returns pre-computed tensors — no CPU work per batch.
    """

    def __init__(self, data_path: str, tokenizer, max_text_len: int = 256):
        raw = [json.loads(line) for line in Path(data_path).read_text().strip().split('\n')]
        filtered = [d for d in raw if d.get('comment') and d.get('fen') and len(d['comment']) > 30]
        print(f"  Pre-tokenizing {len(filtered)} examples (from {len(raw)} total)...")

        t0 = time.time()
        self.fen_tokens = []
        self.text_input_ids = []
        self.text_attention_mask = []
        self.labels = []

        for item in filtered:
            # Chess tokenization
            try:
                fen_tok = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)
            except Exception:
                fen_tok = torch.zeros(77, dtype=torch.long)

            # Text tokenization
            move = item.get('move', '?')
            prompt = f"Move played: {move}. "
            full_text = prompt + item['comment']
            text_encoded = tokenizer(
                full_text, max_length=max_text_len,
                padding='max_length', truncation=True, return_tensors='pt',
            )

            # Label masking (don't compute loss on prompt tokens)
            prompt_encoded = tokenizer(prompt, add_special_tokens=False)
            prompt_len = len(prompt_encoded['input_ids']) + 1
            lbl = text_encoded['input_ids'].squeeze().clone()
            lbl[:prompt_len] = -100
            lbl[text_encoded['attention_mask'].squeeze() == 0] = -100

            self.fen_tokens.append(fen_tok)
            self.text_input_ids.append(text_encoded['input_ids'].squeeze())
            self.text_attention_mask.append(text_encoded['attention_mask'].squeeze())
            self.labels.append(lbl)

        print(f"  Pre-tokenized in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.fen_tokens)

    def __getitem__(self, idx):
        return {
            'fen_tokens': self.fen_tokens[idx],
            'text_input_ids': self.text_input_ids[idx],
            'text_attention_mask': self.text_attention_mask[idx],
            'labels': self.labels[idx],
        }


def check_flash_attn_available() -> bool:
    """Check if flash-attn is installed."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def check_bf16_available() -> bool:
    """Check if bf16 is supported on current GPU."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8  # Ampere+


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--data', default='/home/ec2-user/SageMaker/chess-research/data/lichess_studies.jsonl')
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--projection', default='/home/ec2-user/SageMaker/chess-research/checkpoints/projection_distill.pt')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/chess-research/models/qwen2.5-7b')
    parser.add_argument('--output', default='/home/ec2-user/SageMaker/chess-research/checkpoints')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda')
    use_bf16 = check_bf16_available()
    use_fa2 = check_flash_attn_available()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"=== Optimized Training ===")
    print(f"  bf16: {use_bf16}, Flash Attention 2: {use_fa2}")
    print(f"  batch_size={args.batch_size}, grad_accum={args.grad_accum}, effective={args.batch_size * args.grad_accum}")
    print(f"  torch.compile: {not args.no_compile}")

    # --- Load encoder (frozen) ---
    print("Loading encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder = encoder.to(compute_dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # --- Load projection ---
    print("Loading distill-aligned projection...")
    projection = ChessProjection(encoder_dim=1024, llm_dim=3584).to(device)
    proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
    projection.load_state_dict(proj_ckpt['state_dict'])
    projection = projection.to(compute_dtype)

    # --- Load Qwen + LoRA ---
    print("Loading Qwen + LoRA...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flash Attention 2 if available, else SDPA (PyTorch native)
    model_kwargs = {
        'torch_dtype': compute_dtype,
        'trust_remote_code': True,
    }
    if use_fa2:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
        print("  Using Flash Attention 2")
    else:
        model_kwargs['attn_implementation'] = 'sdpa'
        print("  Using SDPA (flash-attn not installed, run: pip install flash-attn --no-build-isolation)")

    llm = AutoModelForCausalLM.from_pretrained(args.qwen, **model_kwargs).to(device)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    if hasattr(llm, 'gradient_checkpointing_enable'):
        llm.gradient_checkpointing_enable()

    # --- Optional torch.compile ---
    # Keep uncompiled reference for save_pretrained (compiled wrapper can't save)
    llm_unwrapped = llm
    if not args.no_compile:
        print("  Compiling model (first step will be slow)...")
        try:
            # "default" mode is safest with gradient checkpointing + LoRA
            # "reduce-overhead" uses CUDA graphs which may conflict with grad ckpt
            llm = torch.compile(llm, mode="default")
            print("  torch.compile enabled (mode=default)")
        except Exception as e:
            print(f"  torch.compile failed ({e}), continuing without it")

    # --- Optimizer ---
    projection.train()
    trainable_params = list(projection.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # --- Pre-tokenized dataset ---
    dataset = PreTokenizedLichessDataset(args.data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # --- LR schedule ---
    total_steps = args.epochs * len(loader) // args.grad_accum
    warmup = total_steps // 10

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training ---
    print(f"\n=== Stage 2: Optimized Lichess Coaching SFT ===")
    print(f"  Dataset: {len(dataset)}, Batch size: {args.batch_size}, Steps/epoch: {len(loader)}")
    print(f"  Grad accum: {args.grad_accum}, Optimizer steps: {total_steps}, Warmup: {warmup}")
    print(f"  VRAM before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    opt_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_valid = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            fen = batch['fen_tokens'].to(device, non_blocking=True)
            txt = batch['text_input_ids'].to(device, non_blocking=True)
            mask = batch['text_attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            bs = fen.shape[0]

            # Forward pass in autocast
            with torch.autocast("cuda", dtype=compute_dtype):
                with torch.no_grad():
                    chess_hidden = encoder(fen)
                chess_proj = projection(chess_hidden)

                with torch.no_grad():
                    txt_emb = llm.get_input_embeddings()(txt)

                combined = torch.cat([chess_proj, txt_emb], dim=1)
                cmask = torch.cat([
                    torch.ones(bs, 77, dtype=torch.long, device=device),
                    mask,
                ], dim=1)
                clabels = torch.cat([
                    torch.full((bs, 77), -100, dtype=torch.long, device=device),
                    labels,
                ], dim=1)

                outputs = llm(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
                loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            # Backward in native precision (autocast handles scaling)
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

            total_loss += loss.item() * args.grad_accum
            num_valid += 1

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                eta_epoch = (len(loader) - step - 1) / steps_per_sec
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"  E{epoch+1} | {step+1}/{len(loader)} | "
                    f"Loss: {total_loss/num_valid:.4f} | "
                    f"{steps_per_sec:.1f} step/s | "
                    f"ETA: {eta_epoch/60:.0f}min | "
                    f"Peak VRAM: {vram:.1f}GB"
                )

        epoch_time = time.time() - t0
        print(
            f"  Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {total_loss/max(num_valid,1):.4f} | "
            f"Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min)"
        )

    # --- Save ---
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': projection.state_dict(),
        'config': {'encoder_dim': 1024, 'llm_dim': 3584},
        'stage': 'lichess_stage2_optimized',
    }, f"{args.output}/projection_lichess.pt")
    llm_unwrapped.save_pretrained(f"{args.output}/qwen_lora_lichess")

    print(f"\nSaved to {args.output}/")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
