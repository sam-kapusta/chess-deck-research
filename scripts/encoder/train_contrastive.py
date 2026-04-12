#!/usr/bin/env python3
"""Contrastive pre-alignment: teach the chess encoder projection semantic alignment.

InfoNCE loss on (FEN, commentary) pairs. Trains a small projection MLP to map
encoder embeddings (1024-dim) into an alignment space where they match text
embeddings from Qwen's embedding table.

Architecture:
  Chess encoder (frozen) → mean pool → [1024]
  Projection MLP (trainable) → [1024] alignment space
  Text: Qwen tokenizer → Qwen embedding layer (frozen) → mean pool → linear → [1024]
  Loss: symmetric InfoNCE (cross_entropy on cosine similarity matrix)

This is FAST — no 7B model forward pass. Just:
  - Encoder: 270M (frozen, batch encode)
  - Projection: 1024 → 1024 MLP (~2M params, trainable)
  - Text: embedding table lookup (~2GB) + mean pool + linear

Usage (on SAIS):
  # Step 1: Pre-compute embeddings (optional, for speed)
  python train_contrastive.py --precompute --data contrastive_pairs.jsonl

  # Step 2: Train
  python train_contrastive.py --data contrastive_pairs.jsonl --epochs 10 --batch-size 128

  # Step 3: Evaluate
  python train_contrastive.py --evaluate --checkpoint alignment_epoch5.pt

Requires:
  - chess_encoder_270m.pt (converted encoder checkpoint)
  - Qwen3-4B model (for tokenizer + embedding layer only)
  - contrastive_pairs.jsonl (109K+ pairs)
"""
import sys
import json
import argparse
import math
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add encoder module to path (SAIS layout)
SAIS_ENCODER_DIR = "/home/ec2-user/SageMaker/chess-research/encoder"
LOCAL_ENCODER_DIR = str(Path(__file__).parent.parent)
for p in [SAIS_ENCODER_DIR, LOCAL_ENCODER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ============================================================
# Model Components
# ============================================================

class ProjectionMLP(nn.Module):
    """Projects chess encoder embeddings into alignment space.

    Architecture: Linear(1024, 2048) → GELU → Linear(2048, 1024) → LayerNorm
    ~4M params total. The bottleneck expansion (2x) gives the projection
    enough capacity to rearrange features without being too heavy.
    """

    def __init__(self, encoder_dim=1024, align_dim=1024, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, align_dim),
            nn.LayerNorm(align_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextProjection(nn.Module):
    """Projects text embeddings (from Qwen embedding table) into alignment space.

    Takes mean-pooled Qwen token embeddings and projects to alignment dim.
    Architecture: Linear(qwen_dim, align_dim) → LayerNorm
    """

    def __init__(self, text_dim, align_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, align_dim),
            nn.LayerNorm(align_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveAligner(nn.Module):
    """InfoNCE contrastive alignment between chess and text embeddings.

    Symmetric loss: both chess→text and text→chess retrieval.
    Temperature is learnable (initialized at 0.07, clamped to [0.01, 1.0]).
    """

    def __init__(self, encoder_dim=1024, text_dim=3584, align_dim=1024):
        super().__init__()
        self.chess_proj = ProjectionMLP(encoder_dim, align_dim)
        self.text_proj = TextProjection(text_dim, align_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def temperature(self):
        # Clamp temperature to [0.01, 1.0] for stability
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=1.0)

    def forward(self, chess_embeds, text_embeds):
        """Compute symmetric InfoNCE loss.

        Args:
            chess_embeds: [B, encoder_dim] — mean-pooled encoder output
            text_embeds: [B, text_dim] — mean-pooled Qwen token embeddings

        Returns:
            loss: scalar InfoNCE loss
            metrics: dict with sim_pos, sim_neg, accuracy
        """
        z = F.normalize(self.chess_proj(chess_embeds), dim=1)  # [B, align_dim]
        t = F.normalize(self.text_proj(text_embeds), dim=1)    # [B, align_dim]

        # Similarity matrix: [B, B]
        logits = z @ t.T / self.temperature
        labels = torch.arange(len(z), device=z.device)

        # Symmetric loss
        loss_c2t = F.cross_entropy(logits, labels)
        loss_t2c = F.cross_entropy(logits.T, labels)
        loss = (loss_c2t + loss_t2c) / 2

        # Metrics
        with torch.no_grad():
            sim_pos = logits.diag().mean().item() * self.temperature.item()
            mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
            sim_neg = (logits * mask).sum().item() / mask.sum().item() * self.temperature.item()
            acc_c2t = (logits.argmax(dim=1) == labels).float().mean().item()
            acc_t2c = (logits.T.argmax(dim=1) == labels).float().mean().item()

        return loss, {
            "sim_pos": sim_pos,
            "sim_neg": sim_neg,
            "acc_c2t": acc_c2t,
            "acc_t2c": acc_t2c,
            "temperature": self.temperature.item(),
        }


# ============================================================
# Dataset
# ============================================================

class ContrastivePairsDataset(Dataset):
    """Dataset of (FEN, commentary) pairs for contrastive alignment.

    Two modes:
    1. Online: tokenizes FEN + text on the fly (slower, lower memory)
    2. Precomputed: loads pre-embedded tensors (faster training)
    """

    def __init__(self, data_path, chess_tokenize_fn=None, text_tokenizer=None,
                 text_embed_layer=None, max_text_len=128, precomputed_dir=None):
        self.max_text_len = max_text_len
        self.precomputed = precomputed_dir is not None

        if self.precomputed:
            self.chess_embeds = torch.load(Path(precomputed_dir) / "chess_embeds.pt", weights_only=True)
            self.text_embeds = torch.load(Path(precomputed_dir) / "text_embeds.pt", weights_only=True)
            log.info(f"Loaded precomputed: {len(self.chess_embeds)} chess, {len(self.text_embeds)} text")
        else:
            self.chess_tokenize = chess_tokenize_fn
            self.text_tokenizer = text_tokenizer
            self.text_embed_layer = text_embed_layer
            # Load all pairs
            log.info(f"Loading data from {data_path}...")
            self.pairs = []
            with open(data_path) as f:
                for line in f:
                    row = json.loads(line)
                    self.pairs.append((row["fen"], row["commentary"]))
            log.info(f"Loaded {len(self.pairs)} pairs")

    def __len__(self):
        if self.precomputed:
            return len(self.chess_embeds)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.precomputed:
            return {
                "chess_embed": self.chess_embeds[idx],
                "text_embed": self.text_embeds[idx],
            }

        fen, commentary = self.pairs[idx]

        # Chess: tokenize FEN for encoder
        fen_tokens = torch.tensor(
            self.chess_tokenize(fen).astype(np.int64), dtype=torch.long
        )

        # Text: tokenize + embed via Qwen embedding layer
        text_enc = self.text_tokenizer(
            commentary,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            token_embeds = self.text_embed_layer(text_enc["input_ids"].squeeze(0))  # [seq, dim]
            mask = text_enc["attention_mask"].squeeze(0).unsqueeze(-1)  # [seq, 1]
            text_embed = (token_embeds * mask).sum(0) / mask.sum(0).clamp(min=1)  # [dim]

        return {
            "fen_tokens": fen_tokens,
            "text_embed": text_embed,
        }


# ============================================================
# Precompute Embeddings
# ============================================================

@torch.no_grad()
def precompute_embeddings(args):
    """Pre-compute all chess and text embeddings. Saves to disk for fast training."""
    from chess_encoder import ChessEncoder, load_encoder, tokenize_fen
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load data
    pairs = []
    with open(args.data) as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["fen"], row["commentary"]))
    log.info(f"Loaded {len(pairs)} pairs")

    # Load chess encoder
    log.info(f"Loading chess encoder from {args.encoder}...")
    encoder = load_encoder(args.encoder, device=str(device))

    # Load Qwen tokenizer + embedding layer only
    log.info(f"Loading Qwen tokenizer + embeddings from {args.qwen_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load full model but only use embedding layer — then free
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, trust_remote_code=True, torch_dtype=torch.float16
    )
    embed_layer = model.model.embed_tokens.to(device)
    text_dim = embed_layer.embedding_dim
    del model  # Free 7B params
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log.info(f"Text embedding dim: {text_dim}")

    # Pre-compute chess embeddings
    log.info("Computing chess embeddings...")
    chess_embeds = []
    batch_size = args.batch_size
    t0 = time.time()
    for i in range(0, len(pairs), batch_size):
        batch_fens = [p[0] for p in pairs[i:i+batch_size]]
        try:
            tokens = torch.stack([tokenize_fen(f) for f in batch_fens]).to(device)
            hidden = encoder(tokens)  # [B, 77, 1024]
            pooled = hidden.mean(dim=1)  # [B, 1024]
            chess_embeds.append(pooled.cpu().float())
        except Exception as e:
            log.warning(f"Batch {i}: {e}, skipping {len(batch_fens)} fens")
            # Append zeros as placeholder — will be filtered later
            chess_embeds.append(torch.zeros(len(batch_fens), 1024))
        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + batch_size) / max(elapsed, 1)
            eta = (len(pairs) - i) / max(speed, 1)
            log.info(f"  Chess: {i+batch_size}/{len(pairs)} ({speed:.0f} pos/s, ETA {eta:.0f}s)")
    chess_embeds = torch.cat(chess_embeds, dim=0)
    log.info(f"Chess embeddings: {chess_embeds.shape} in {time.time()-t0:.1f}s")

    # Pre-compute text embeddings
    log.info("Computing text embeddings...")
    text_embeds = []
    t0 = time.time()
    for i in range(0, len(pairs), batch_size):
        batch_comments = [p[1] for p in pairs[i:i+batch_size]]
        enc = tokenizer(
            batch_comments,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        token_embeds = embed_layer(enc["input_ids"])  # [B, seq, dim]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, seq, 1]
        pooled = (token_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, dim]
        text_embeds.append(pooled.cpu().float())
        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + batch_size) / max(elapsed, 1)
            eta = (len(pairs) - i) / max(speed, 1)
            log.info(f"  Text: {i+batch_size}/{len(pairs)} ({speed:.0f} pos/s, ETA {eta:.0f}s)")
    text_embeds = torch.cat(text_embeds, dim=0)
    log.info(f"Text embeddings: {text_embeds.shape} in {time.time()-t0:.1f}s")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(chess_embeds, out_dir / "chess_embeds.pt")
    torch.save(text_embeds, out_dir / "text_embeds.pt")

    # Save metadata
    meta = {
        "num_pairs": len(pairs),
        "chess_dim": chess_embeds.shape[1],
        "text_dim": text_embeds.shape[1],
        "encoder": args.encoder,
        "qwen_model": args.qwen_model,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Saved to {out_dir}/")


# ============================================================
# Training
# ============================================================

class PrecomputedDataset(Dataset):
    """Fast dataset from pre-computed embedding tensors."""

    def __init__(self, chess_embeds, text_embeds):
        assert len(chess_embeds) == len(text_embeds)
        self.chess = chess_embeds
        self.text = text_embeds

    def __len__(self):
        return len(self.chess)

    def __getitem__(self, idx):
        return self.chess[idx], self.text[idx]


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    precomputed_dir = Path(args.precomputed_dir) if args.precomputed_dir else None

    if precomputed_dir:
        # Fast path: load precomputed embeddings
        log.info(f"Loading precomputed embeddings from {precomputed_dir}...")
        chess_embeds = torch.load(precomputed_dir / "chess_embeds.pt", weights_only=True)
        text_embeds = torch.load(precomputed_dir / "text_embeds.pt", weights_only=True)
        meta = json.loads((precomputed_dir / "metadata.json").read_text())
        encoder_dim = meta["chess_dim"]
        text_dim = meta["text_dim"]

        # Train/val split (95/5)
        n = len(chess_embeds)
        n_val = max(1, int(n * 0.05))
        perm = torch.randperm(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        train_ds = PrecomputedDataset(chess_embeds[train_idx], text_embeds[train_idx])
        val_ds = PrecomputedDataset(chess_embeds[val_idx], text_embeds[val_idx])
        log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    else:
        # Online path: encode on the fly (slower but lower disk usage)
        from chess_encoder import load_encoder, tokenize_fen
        from searchless_chess.src.tokenizer import tokenize as chess_tokenize
        from transformers import AutoTokenizer, AutoModelForCausalLM

        log.info(f"Loading encoder from {args.encoder}...")
        encoder = load_encoder(args.encoder, device=str(device))

        log.info(f"Loading Qwen from {args.qwen_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model, trust_remote_code=True, torch_dtype=torch.float16
        )
        embed_layer = model.model.embed_tokens.to(device)
        text_dim = embed_layer.embedding_dim
        encoder_dim = 1024
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        full_ds = ContrastivePairsDataset(
            args.data,
            chess_tokenize_fn=chess_tokenize,
            text_tokenizer=tokenizer,
            text_embed_layer=embed_layer,
        )
        # Split
        n = len(full_ds)
        n_val = max(1, int(n * 0.05))
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n - n_val, n_val])
        log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    aligner = ContrastiveAligner(
        encoder_dim=encoder_dim,
        text_dim=text_dim,
        align_dim=args.align_dim,
    ).to(device)
    total_params = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
    log.info(f"Aligner: {total_params:,} trainable params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        aligner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    # Cosine LR schedule with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        aligner.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # Training loop
    log.info(f"Training: {args.epochs} epochs, {len(train_loader)} steps/epoch, "
             f"batch_size={args.batch_size}, lr={args.lr}")
    best_val_loss = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        aligner.train()
        epoch_loss = 0.0
        epoch_metrics = {"sim_pos": 0, "sim_neg": 0, "acc_c2t": 0, "acc_t2c": 0}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if precomputed_dir:
                chess_e, text_e = batch[0].to(device), batch[1].to(device)
            else:
                # Online mode: need to run encoder
                fen_tokens = batch["fen_tokens"].to(device)
                with torch.no_grad():
                    hidden = encoder(fen_tokens)
                    chess_e = hidden.mean(dim=1)
                text_e = batch["text_embed"].to(device)

            loss, metrics = aligner(chess_e, text_e)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics.get(k, 0)

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"E{epoch} S{step}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} sim+={metrics['sim_pos']:.3f} "
                    f"sim-={metrics['sim_neg']:.3f} acc={metrics['acc_c2t']:.3f} "
                    f"τ={metrics['temperature']:.4f} lr={lr:.2e}"
                )

        # Epoch averages
        n_steps = len(train_loader)
        avg_loss = epoch_loss / n_steps
        for k in epoch_metrics:
            epoch_metrics[k] /= n_steps
        elapsed = time.time() - t0

        # Validation
        aligner.eval()
        val_loss = 0.0
        val_metrics = {"sim_pos": 0, "sim_neg": 0, "acc_c2t": 0, "acc_t2c": 0}
        with torch.no_grad():
            for batch in val_loader:
                if precomputed_dir:
                    chess_e, text_e = batch[0].to(device), batch[1].to(device)
                else:
                    fen_tokens = batch["fen_tokens"].to(device)
                    hidden = encoder(fen_tokens)
                    chess_e = hidden.mean(dim=1)
                    text_e = batch["text_embed"].to(device)

                loss, metrics = aligner(chess_e, text_e)
                val_loss += loss.item()
                for k in val_metrics:
                    val_metrics[k] += metrics.get(k, 0)

        n_val_steps = max(len(val_loader), 1)
        val_loss /= n_val_steps
        for k in val_metrics:
            val_metrics[k] /= n_val_steps

        log.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch} done in {elapsed:.0f}s\n"
            f"  Train: loss={avg_loss:.4f} sim+={epoch_metrics['sim_pos']:.3f} "
            f"acc={epoch_metrics['acc_c2t']:.3f}\n"
            f"  Val:   loss={val_loss:.4f} sim+={val_metrics['sim_pos']:.3f} "
            f"acc={val_metrics['acc_c2t']:.3f}\n"
            f"{'='*60}\n"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": aligner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "train_metrics": epoch_metrics,
            "val_metrics": val_metrics,
            "config": {
                "encoder_dim": encoder_dim,
                "text_dim": text_dim,
                "align_dim": args.align_dim,
            },
        }
        torch.save(ckpt, out_dir / f"alignment_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "alignment_best.pt")
            log.info(f"  New best val loss: {val_loss:.4f}")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved to {out_dir}/")


# ============================================================
# Evaluation
# ============================================================

def evaluate(args):
    """Evaluate aligned embeddings: cosine similarity + retrieval metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precomputed_dir = Path(args.precomputed_dir)
    chess_embeds = torch.load(precomputed_dir / "chess_embeds.pt", weights_only=True)
    text_embeds = torch.load(precomputed_dir / "text_embeds.pt", weights_only=True)
    meta = json.loads((precomputed_dir / "metadata.json").read_text())

    # Load trained aligner
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    aligner = ContrastiveAligner(**config).to(device)
    aligner.load_state_dict(ckpt["model_state_dict"])
    aligner.eval()
    log.info(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    # Evaluate on a random subset (5K for speed)
    n_eval = min(5000, len(chess_embeds))
    perm = torch.randperm(len(chess_embeds))[:n_eval]
    chess_sub = chess_embeds[perm].to(device)
    text_sub = text_embeds[perm].to(device)

    with torch.no_grad():
        z = F.normalize(aligner.chess_proj(chess_sub), dim=1)
        t = F.normalize(aligner.text_proj(text_sub), dim=1)

        # Cosine similarity
        sim_matrix = z @ t.T  # [N, N]
        sim_pos = sim_matrix.diag()
        mask = ~torch.eye(n_eval, dtype=torch.bool, device=device)
        sim_neg = sim_matrix[mask].reshape(n_eval, n_eval - 1)

        log.info(f"\n{'='*60}")
        log.info(f"Evaluation on {n_eval} pairs:")
        log.info(f"  Positive cosine sim: {sim_pos.mean():.4f} ± {sim_pos.std():.4f}")
        log.info(f"  Negative cosine sim: {sim_neg.mean():.4f} ± {sim_neg.std():.4f}")
        log.info(f"  Gap: {(sim_pos.mean() - sim_neg.mean()):.4f}")

        # Retrieval: MRR (chess → text)
        ranks = (sim_matrix.argsort(dim=1, descending=True) == torch.arange(n_eval, device=device).unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1.0 / ranks.float()).mean()
        r1 = (ranks == 1).float().mean()
        r5 = (ranks <= 5).float().mean()
        r10 = (ranks <= 10).float().mean()

        log.info(f"\n  Retrieval (chess → text):")
        log.info(f"    MRR: {mrr:.4f}")
        log.info(f"    R@1: {r1:.4f}")
        log.info(f"    R@5: {r5:.4f}")
        log.info(f"    R@10: {r10:.4f}")
        log.info(f"    Median rank: {ranks.median().item()}")
        log.info(f"{'='*60}")

    # Save results
    results = {
        "n_eval": n_eval,
        "sim_pos_mean": sim_pos.mean().item(),
        "sim_pos_std": sim_pos.std().item(),
        "sim_neg_mean": sim_neg.mean().item(),
        "mrr": mrr.item(),
        "r_at_1": r1.item(),
        "r_at_5": r5.item(),
        "r_at_10": r10.item(),
        "median_rank": ranks.median().item(),
        "checkpoint": args.checkpoint,
        "epoch": ckpt["epoch"],
    }
    out_path = Path(args.output_dir) / "contrastive_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved to {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Contrastive pre-alignment for chess encoder")
    sub = parser.add_subparsers(dest="command")

    # Precompute embeddings
    p_pre = sub.add_parser("precompute", help="Pre-compute chess and text embeddings")
    p_pre.add_argument("--data", required=True, help="Path to contrastive_pairs.jsonl")
    p_pre.add_argument("--encoder", default="/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt")
    p_pre.add_argument("--qwen-model", default="/home/ec2-user/SageMaker/chess-research/models/Qwen3-4B")
    p_pre.add_argument("--output-dir", default="/home/ec2-user/SageMaker/chess-research/data/precomputed")
    p_pre.add_argument("--batch-size", type=int, default=64)

    # Train
    p_train = sub.add_parser("train", help="Train contrastive alignment")
    p_train.add_argument("--data", help="Path to contrastive_pairs.jsonl (for online mode)")
    p_train.add_argument("--precomputed-dir", help="Dir with precomputed embeddings (fast mode)")
    p_train.add_argument("--encoder", default="/home/ec2-user/SageMaker/chess-research/encoder/chess_encoder_270m.pt")
    p_train.add_argument("--qwen-model", default="/home/ec2-user/SageMaker/chess-research/models/Qwen3-4B")
    p_train.add_argument("--output-dir", default="/home/ec2-user/SageMaker/chess-research/checkpoints/contrastive")
    p_train.add_argument("--align-dim", type=int, default=1024)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=0.01)
    p_train.add_argument("--warmup-steps", type=int, default=500)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--resume", help="Resume from checkpoint")

    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate alignment quality")
    p_eval.add_argument("--checkpoint", required=True, help="Path to alignment checkpoint")
    p_eval.add_argument("--precomputed-dir", required=True)
    p_eval.add_argument("--output-dir", default="/home/ec2-user/SageMaker/chess-research/results")

    args = parser.parse_args()

    if args.command == "precompute":
        precompute_embeddings(args)
    elif args.command == "train":
        if not args.data and not args.precomputed_dir:
            parser.error("Either --data or --precomputed-dir required")
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
