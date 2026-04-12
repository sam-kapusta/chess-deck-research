#!/usr/bin/env python3
"""Stage 2: Coaching SFT with Lichess study annotations.

Uses distill-aligned projection + LoRA on Qwen.
Data: Lichess popular study annotations (FEN + move + human commentary).

Usage:
  python train_lichess_stage2.py --data /path/to/lichess_studies.jsonl --epochs 3
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


class LichessCoachingDataset(Dataset):
    """Lichess study annotations: (FEN, move, comment) -> coaching text."""

    def __init__(self, data_path, tokenizer, max_text_len=256):
        raw = [json.loads(line) for line in Path(data_path).read_text().strip().split('\n')]
        self.data = [d for d in raw if d.get('comment') and d.get('fen') and len(d['comment']) > 30]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        print(f"  Loaded {len(self.data)} Lichess annotations (from {len(raw)} total)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            fen_tokens = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)
        except:
            fen_tokens = torch.zeros(77, dtype=torch.long)

        move = item.get('move', '?')
        prompt = f"Move played: {move}. "
        full_text = prompt + item['comment']
        text_encoded = self.tokenizer(
            full_text, max_length=self.max_text_len,
            padding='max_length', truncation=True, return_tensors='pt',
        )

        prompt_encoded = self.tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_encoded['input_ids']) + 1
        labels = text_encoded['input_ids'].squeeze().clone()
        labels[:prompt_len] = -100
        labels[text_encoded['attention_mask'].squeeze() == 0] = -100

        return {
            'fen_tokens': fen_tokens,
            'text_input_ids': text_encoded['input_ids'].squeeze(),
            'text_attention_mask': text_encoded['attention_mask'].squeeze(),
            'labels': labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--data', default='/home/ec2-user/SageMaker/chess-research/data/lichess_studies.jsonl')
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--projection', default='/home/ec2-user/SageMaker/chess-research/checkpoints/projection_distill.pt')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/chess-research/models/qwen2.5-7b')
    parser.add_argument('--output', default='/home/ec2-user/SageMaker/chess-research/checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda')

    print("Loading encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("Loading distill-aligned projection...")
    projection = ChessProjection(encoder_dim=1024, llm_dim=3584).to(device).float()
    proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
    projection.load_state_dict(proj_ckpt['state_dict'])

    print("Loading Qwen + LoRA...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(args.qwen, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()
    if hasattr(llm, 'gradient_checkpointing_enable'):
        llm.gradient_checkpointing_enable()

    projection.train()
    trainable_params = list(projection.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    dataset = LichessCoachingDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    total_steps = args.epochs * len(loader) // args.grad_accum
    warmup = total_steps // 10
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n=== Stage 2: Lichess Coaching SFT ===")
    print(f"  Dataset: {len(dataset)}, Steps/epoch: {len(loader)}, Total opt: {total_steps}")

    for epoch in range(args.epochs):
        total_loss = 0; num_valid = 0; t0 = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            fen = batch['fen_tokens'].to(device)
            txt = batch['text_input_ids'].to(device)
            mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            bs = fen.shape[0]

            with torch.no_grad(): chess_hidden = encoder(fen)
            chess_proj = projection(chess_hidden.float()).half()
            with torch.no_grad(): txt_emb = llm.get_input_embeddings()(txt)
            combined = torch.cat([chess_proj, txt_emb], dim=1)
            cmask = torch.cat([torch.ones(bs, 77, dtype=torch.long, device=device), mask], dim=1)
            clabels = torch.cat([torch.full((bs, 77), -100, dtype=torch.long, device=device), labels], dim=1)
            outputs = llm(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
            loss = outputs.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss): optimizer.zero_grad(); continue
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item() * args.grad_accum; num_valid += 1
            if (step + 1) % 200 == 0:
                print(f"  E{epoch+1} | {step+1}/{len(loader)} | Loss: {total_loss/num_valid:.4f}")

        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/max(num_valid,1):.4f} | Time: {time.time()-t0:.0f}s")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': projection.state_dict(), 'config': {'encoder_dim': 1024, 'llm_dim': 3584},
                'stage': 'lichess_stage2'}, f"{args.output}/projection_lichess.pt")
    llm.save_pretrained(f"{args.output}/qwen_lora_lichess")
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
