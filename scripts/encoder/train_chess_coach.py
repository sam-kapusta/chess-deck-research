#!/usr/bin/env python3
"""Train the chess coaching model: projection layer + optional LoRA.

Phase 1: Train projection only (encoder + LLM frozen)
  python train_chess_coach.py --phase 1

Phase 2: Train projection + LoRA (encoder frozen)
  python train_chess_coach.py --phase 2

Requires on SAIS:
  - chess_encoder_270m.pt (converted encoder)
  - Qwen3-4B-Instruct model weights
  - coaching_training_data.jsonl (generated coaching text)
"""
import sys
import json
import argparse
import math
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


# ============================================================
# Dataset
# ============================================================

class ChessCoachingDataset(Dataset):
    """Dataset of (FEN, coaching_text) pairs."""

    def __init__(self, data_path, tokenizer, max_text_len=256):
        self.data = [json.loads(line) for line in Path(data_path).read_text().strip().split('\n')
                     if json.loads(line).get('coaching_text')]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Chess encoder input
        fen_tokens = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)

        # Build text prompt + coaching output
        prompt = f"You played {item['played_move']} ({item['classification']}). "
        if item.get('tags'):
            prompt += f"Patterns: {', '.join(item['tags'][:3])}. "
        prompt += "Explain what went wrong."

        full_text = prompt + " " + item['coaching_text']
        text_encoded = self.tokenizer(
            full_text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Labels: -100 for prompt tokens (don't compute loss on prompt)
        prompt_len = len(self.tokenizer(prompt)['input_ids'])
        labels = text_encoded['input_ids'].squeeze().clone()
        labels[:prompt_len] = -100  # Mask prompt

        return {
            'fen_tokens': fen_tokens,
            'text_input_ids': text_encoded['input_ids'].squeeze(),
            'text_attention_mask': text_encoded['attention_mask'].squeeze(),
            'labels': labels,
        }


# ============================================================
# Training Loop
# ============================================================

def train_phase1(encoder, projection, llm, tokenizer, data_path, epochs=3, lr=2e-3, batch_size=4):
    """Phase 1: Train projection only. Encoder + LLM frozen."""
    print("=== Phase 1: Projection Layer Training ===")

    # Freeze everything except projection
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in llm.parameters():
        p.requires_grad = False
    projection.train()

    dataset = ChessCoachingDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            fen_tokens = batch['fen_tokens']
            text_ids = batch['text_input_ids']
            labels = batch['labels']

            # Encode chess position
            with torch.no_grad():
                chess_hidden = encoder(fen_tokens)  # [B, 77, 1024]
            chess_projected = projection(chess_hidden)  # [B, 77, 4096]

            # Get text embeddings
            text_embeds = llm.get_input_embeddings()(text_ids)  # [B, T, 4096]

            # Combine: chess context prepended to text
            combined = torch.cat([chess_projected, text_embeds], dim=1)  # [B, 77+T, 4096]

            # Create attention mask (all 1s for chess, text mask for text)
            chess_mask = torch.ones(chess_projected.shape[0], 77, dtype=torch.long)
            text_mask = batch['text_attention_mask']
            combined_mask = torch.cat([chess_mask, text_mask], dim=1)

            # Shift labels to account for chess prefix (77 tokens)
            # Labels for chess positions = -100 (no loss)
            chess_labels = torch.full((labels.shape[0], 77), -100, dtype=torch.long)
            combined_labels = torch.cat([chess_labels, labels], dim=1)

            # Forward through LLM
            outputs = llm(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                labels=combined_labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx+1}/{len(loader)} | Loss: {avg:.4f}")

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    return projection


def train_phase2(encoder, projection, llm, tokenizer, data_path, epochs=3, lr=2e-5, batch_size=4):
    """Phase 2: Train projection + LoRA. Encoder frozen."""
    from peft import LoraConfig, get_peft_model, TaskType

    print("=== Phase 2: Projection + LoRA Training ===")

    # Apply LoRA to LLM
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # Train both projection and LoRA
    trainable_params = list(projection.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    dataset = ChessCoachingDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0
        projection.train()
        llm.train()

        for batch_idx, batch in enumerate(loader):
            fen_tokens = batch['fen_tokens']
            text_ids = batch['text_input_ids']
            labels = batch['labels']

            with torch.no_grad():
                chess_hidden = encoder(fen_tokens)
            chess_projected = projection(chess_hidden)

            text_embeds = llm.get_input_embeddings()(text_ids)
            combined = torch.cat([chess_projected, text_embeds], dim=1)

            chess_mask = torch.ones(chess_projected.shape[0], 77, dtype=torch.long)
            combined_mask = torch.cat([chess_mask, batch['text_attention_mask']], dim=1)

            chess_labels = torch.full((labels.shape[0], 77), -100, dtype=torch.long)
            combined_labels = torch.cat([chess_labels, labels], dim=1)

            outputs = llm(inputs_embeds=combined, attention_mask=combined_mask, labels=combined_labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {batch_idx+1}/{len(loader)} | Loss: {total_loss/(batch_idx+1):.4f}")

        print(f"  Epoch {epoch+1} complete | Avg Loss: {total_loss/len(loader):.4f}")

    return projection, llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--data', default='/home/ec2-user/SageMaker/chess-research/data/coaching_training_data.jsonl')
    parser.add_argument('--encoder', default='/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt')
    parser.add_argument('--qwen', default='/home/ec2-user/SageMaker/chess-research/models/qwen3-4b')
    parser.add_argument('--output', default='/home/ec2-user/SageMaker/chess-research/checkpoints')
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 2e-3 if args.phase == 1 else 2e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load encoder
    print("Loading chess encoder...")
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    # Load projection
    print("Creating projection layer...")
    projection = ChessProjection(encoder_dim=1024, llm_dim=4096).to(device)

    # Load Qwen
    print(f"Loading Qwen from {args.qwen}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # Train
    if args.phase == 1:
        projection = train_phase1(encoder, projection, llm, tokenizer, args.data,
                                  epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        torch.save(projection.state_dict(), f"{args.output}/projection_phase1.pt")
        print(f"Saved projection to {args.output}/projection_phase1.pt")
    else:
        projection, llm = train_phase2(encoder, projection, llm, tokenizer, args.data,
                                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        torch.save(projection.state_dict(), f"{args.output}/projection_phase2.pt")
        llm.save_pretrained(f"{args.output}/qwen_lora_phase2")
        print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
