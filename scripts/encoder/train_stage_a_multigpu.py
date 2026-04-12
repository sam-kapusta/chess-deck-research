#!/usr/bin/env python3
"""Stage A: Train model to predict chess truths from encoder embeddings.

Multi-GPU version for g5.48xlarge (8×A10G, 192GB GPU total).
Full fine-tune Qwen2.5-7B — no LoRA needed with this much memory.

Launch:
  torchrun --nproc_per_node=8 train_stage_a_multigpu.py \
    --data /path/to/stage_a_all_positions.jsonl \
    --epochs 2 --lr 1e-5 --batch-size 4

Single GPU fallback:
  python train_stage_a_multigpu.py --data /path/to/data.jsonl --batch-size 1
"""
import sys
import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder')
sys.path.insert(0, '/home/ec2-user/SageMaker/chess-research/encoder/scripts')

from searchless_chess.src.tokenizer import tokenize as chess_tokenize
from convert_and_validate_v2 import ChessEncoder
from projection_layer import ChessProjection


def format_target(pos):
    """Format Stockfish labels as target text."""
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
    """Pre-tokenized positions with Stockfish labels."""

    def __init__(self, data_path, tokenizer, max_text_len=128):
        raw = [json.loads(l) for l in Path(data_path).read_text().strip().split('\n')]
        self.data = [d for d in raw if d.get('best_move') and d.get('fen')]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        self.examples = []
        for item in self.data:
            try:
                fen_tokens = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)
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

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(f"  Loaded {len(self.examples)} examples (from {len(raw)} raw)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    return 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--projection', default='')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/chess-research/models/qwen2.5-7b')
    parser.add_argument('--output', default='/home/ec2-user/SageMaker/chess-research/checkpoints/stage_a_full_ft')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    args = parser.parse_args()

    local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    is_main = local_rank == 0

    if is_main:
        print(f"{'='*60}")
        print(f"Stage A: Structured Chess Prediction (Full Fine-Tune)")
        print(f"{'='*60}")
        print(f"GPUs: {world_size}, Batch/GPU: {args.batch_size}, Grad accum: {args.grad_accum}")
        print(f"Effective batch: {world_size * args.batch_size * args.grad_accum}")
        print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Load encoder (same on all ranks, frozen)
    if is_main:
        print("Loading encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load projection
    from transformers import AutoConfig
    qwen_config = AutoConfig.from_pretrained(args.qwen, trust_remote_code=True)
    hidden_dim = qwen_config.hidden_size

    projection = ChessProjection(encoder_dim=1024, llm_dim=hidden_dim).to(device)
    if args.projection and Path(args.projection).exists():
        proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
        projection.load_state_dict(proj_ckpt['state_dict'])
        if is_main:
            print(f"  Loaded projection from {args.projection}")

    # Load Qwen — full fine-tune
    if is_main:
        print(f"Loading Qwen from {args.qwen}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    if hasattr(llm, 'gradient_checkpointing_enable'):
        llm.gradient_checkpointing_enable()

    # Wrap for DDP
    if is_distributed:
        llm = DDP(llm, device_ids=[local_rank], find_unused_parameters=False)
        projection = DDP(projection, device_ids=[local_rank])

    # Dataset + distributed sampler
    if is_main:
        print("Loading dataset...")
    dataset = StageADataset(args.data, tokenizer)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                       shuffle=(sampler is None), drop_last=True, num_workers=2, pin_memory=True)

    # Optimizer — all params (full FT)
    llm_module = llm.module if is_distributed else llm
    proj_module = projection.module if is_distributed else projection
    trainable_params = list(proj_module.parameters()) + list(llm_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * len(loader) // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main:
        print(f"  Dataset: {len(dataset)}, Steps/epoch: {len(loader)}")
        print(f"  Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")
        for i in range(world_size):
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f}GB)")

    # Training loop
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

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

            chess_proj = proj_module(chess_hidden.float()).to(llm_module.dtype) if not is_distributed else projection(chess_hidden.float()).to(torch.bfloat16)

            embed_fn = llm_module.get_input_embeddings() if not is_distributed else llm.module.get_input_embeddings()
            txt_emb = embed_fn(txt)

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

            if is_main and (step + 1) % 100 == 0:
                avg = total_loss / num_valid
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                eta = (len(loader) - step - 1) / steps_per_sec / 60
                print(f"  E{epoch+1} | {step+1}/{len(loader)} | Loss: {avg:.4f} | {steps_per_sec:.1f} step/s | ETA: {eta:.0f}min")

        elapsed = time.time() - t0
        avg = total_loss / max(num_valid, 1)
        if is_main:
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg:.4f} | Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save (only main rank)
    if is_main:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': proj_module.state_dict(),
            'config': {'encoder_dim': 1024, 'llm_dim': hidden_dim},
            'stage': 'stage_a_full_ft',
        }, f"{args.output}/projection_stage_a.pt")
        llm_module.save_pretrained(f"{args.output}/qwen_stage_a")
        tokenizer.save_pretrained(f"{args.output}/qwen_stage_a")
        print(f"Saved to {args.output}/")

    cleanup_distributed()


if __name__ == "__main__":
    main()
