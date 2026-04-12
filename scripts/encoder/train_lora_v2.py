#!/usr/bin/env python3
"""Phase 2 LoRA v2: Fix norm mismatch that killed v1.

Key fix: Scale chess embeddings to match Qwen text embedding norms (~1.0 per token).
LayerNorm gives unit variance per feature → L2 norm ≈ sqrt(dim) ≈ 59.8.
Qwen text tokens have L2 norm ≈ 0.94. 60x mismatch caused NaN gradients in v1.

Usage:
  python3 train_lora_v2.py \
    --data data/mixed_train.jsonl \
    --encoder /tmp/chess_encoder_270m.pt \
    --qwen models/qwen2.5-7b \
    --projection output/contrastive/projection.pt \
    --output output/lora_v2 \
    --epochs 1 --batch-size 2 --lr 2e-4
"""
import sys, os, json, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fen_tokenizer import tokenize as chess_tokenize
from chess_model import ChessEncoder
from projection_layer import ChessProjection


def format_target(item):
    parts = []
    if 'best_move_uci' in item:
        ev = item.get('eval', 0)
        parts.append("Eval: %+.1f" % ev)
        parts.append("Best: %s" % item['best_move_uci'])
        pv = item.get('pv_line_uci', '')
        if pv: parts.append("Line: %s" % pv)
    elif 'themes' in item:
        moves = item.get('moves', '').split()
        if moves: parts.append("Best: %s" % moves[0])
        themes = item.get('themes', '')
        if themes: parts.append("Theme: %s" % themes)
    else:
        ev = item.get('eval_sf', item.get('eval', 0))
        bm = item.get('best_move', '')
        if ev is not None: parts.append("Eval: %+.1f" % ev)
        if bm: parts.append("Best: %s" % bm)
        cls = item.get('classification', '')
        if cls: parts.append("Classification: %s" % cls)
    return '. '.join(parts) + '.' if parts else ''


class ChessDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.items = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for line in open(path):
            item = json.loads(line.strip())
            target = format_target(item)
            if not target or not item.get('fen'): continue
            fen = item['fen']
            parts = fen.split()
            if len(parts) == 4: fen += ' 0 1'
            elif len(parts) == 5: fen += ' 1'
            self.items.append({'fen': fen, 'target': target})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        fen_tokens = torch.tensor(chess_tokenize(item['fen']).astype(np.int64), dtype=torch.long)
        prompt = "Analyze this position."
        full = prompt + " " + item['target']
        enc = self.tokenizer(full, max_length=self.max_len, padding='max_length',
                            truncation=True, return_tensors='pt')
        prompt_len = len(self.tokenizer(prompt)['input_ids'])
        labels = enc['input_ids'].clone()
        labels[0, :prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'fen_tokens': fen_tokens,
            'text_input_ids': enc['input_ids'].squeeze(0),
            'text_attention_mask': enc['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--qwen', required=True)
    parser.add_argument('--projection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=0, help='Stop after N steps (0=full epoch)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encoder (frozen)
    print("Loading encoder...", flush=True)
    ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).half()
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # LLM with LoRA
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("Loading %s..." % args.qwen, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    # Apply LoRA
    lora_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # Projection (frozen contrastive) + LayerNorm + scale
    hidden_dim = AutoConfig.from_pretrained(args.qwen, trust_remote_code=True).hidden_size
    projection = ChessProjection(encoder_dim=1024, llm_dim=hidden_dim).to(device)
    proj_ckpt = torch.load(args.projection, map_location=device, weights_only=False)
    projection.load_state_dict(proj_ckpt['state_dict'])
    projection.eval()
    for p in projection.parameters(): p.requires_grad = False

    proj_norm = nn.LayerNorm(hidden_dim).to(device)  # keep float32

    # Compute target norm from Qwen text embeddings
    with torch.no_grad():
        sample_ids = tokenizer("Analyze this chess position carefully.", return_tensors='pt')['input_ids'].to(device)
        sample_emb = llm.get_input_embeddings()(sample_ids)
        target_norm = sample_emb.norm(dim=-1).mean().item()
    print("Target text embedding norm: %.3f" % target_norm, flush=True)

    # Verify chess projection norm BEFORE scaling
    with torch.no_grad():
        test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        test_tok = torch.tensor(chess_tokenize(test_fen).astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        test_hidden = encoder(test_tok)
        test_proj = proj_norm(projection(test_hidden.float()))
        raw_norm = test_proj.norm(dim=-1).mean().item()
        scale_factor = target_norm / raw_norm
    print("Raw chess proj norm: %.3f, scale factor: %.4f" % (raw_norm, scale_factor), flush=True)

    print("GPU mem: %.1fGB" % (torch.cuda.memory_allocated(device)/1e9), flush=True)

    # Data
    print("Loading dataset...", flush=True)
    dataset = ChessDataset(args.data, tokenizer)
    print("  %d examples" % len(dataset), flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       drop_last=True, num_workers=2, pin_memory=True)

    # Optimizer (LoRA params + proj_norm)
    trainable = list(filter(lambda p: p.requires_grad, llm.parameters())) + list(proj_norm.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(loader)
    warmup = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("Steps/epoch: %d, Total: %d" % (len(loader), total_steps), flush=True)
    print("Training...", flush=True)

    nan_count = 0
    for epoch in range(args.epochs):
        total_loss, n = 0, 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            fen = batch['fen_tokens'].to(device)
            txt = batch['text_input_ids'].to(device)
            mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            bs = txt.shape[0]

            with torch.no_grad():
                chess_hidden = encoder(fen)
                # LayerNorm + scale to match text embedding norms
                chess_proj = proj_norm(projection(chess_hidden.float()))
                chess_proj = chess_proj * scale_factor
                chess_proj = chess_proj.to(torch.bfloat16)

            txt_emb = llm.get_input_embeddings()(txt)
            seq_len = chess_proj.shape[1]
            combined = torch.cat([chess_proj, txt_emb], dim=1)
            cmask = torch.cat([torch.ones(bs, seq_len, dtype=torch.long, device=device), mask], dim=1)
            clabels = torch.cat([torch.full((bs, seq_len), -100, dtype=torch.long, device=device), labels], dim=1)

            outputs = llm(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue  # Skip NaN, don't mask it

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n += 1

            if step == 0:
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                # Verify norms
                chess_n = chess_proj[0].norm(dim=-1).mean().item()
                text_n = txt_emb[0].norm(dim=-1).mean().item()
                print("  Step 0: chess_norm=%.2f text_norm=%.2f peak=%.1fGB grad_norm=%.2f" %
                      (chess_n, text_n, peak, grad_norm.item()), flush=True)

            if (step + 1) % 50 == 0:
                avg = total_loss / max(n, 1)
                elapsed = time.time() - t0
                rate = (step + 1) / elapsed
                eta = int((len(loader) - step - 1) / max(rate, 0.01)) // 60
                print("  E%d | %d/%d | Loss: %.4f | %.1f step/s | ETA: %dmin | NaN: %d" %
                      (epoch + 1, step + 1, len(loader), avg, rate, eta, nan_count), flush=True)

            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                print("  Max steps reached (%d)" % args.max_steps, flush=True)
                break

        avg = total_loss / max(n, 1)
        elapsed = (time.time() - t0) / 60
        print("  Epoch %d/%d | Loss: %.4f | %.1fmin | NaN: %d" %
              (epoch + 1, args.epochs, avg, elapsed, nan_count), flush=True)

    # Save
    os.makedirs(args.output, exist_ok=True)
    llm.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    torch.save({'state_dict': proj_norm.state_dict(), 'config': {'hidden_dim': hidden_dim},
                'scale_factor': scale_factor}, os.path.join(args.output, 'proj_norm.pt'))
    print("Saved to %s (NaN count: %d)" % (args.output, nan_count), flush=True)


if __name__ == '__main__':
    main()
