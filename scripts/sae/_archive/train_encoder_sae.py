#!/usr/bin/env python3
"""Train SAE on DeepMind 270M chess encoder activations.

Step 1: Extract mean-pooled encoder activations for N positions
Step 2: Train BatchTopK SAE
Step 3: Save checkpoint + normalization stats

Usage (on SAIS/notebook with GPU):
    python3 train_encoder_sae.py \
        --encoder /tmp/chess_encoder_270m.pt \
        --data /home/ec2-user/SageMaker/chess-stage-a/data/eval_positions.jsonl \
        --output /home/ec2-user/SageMaker/chess-stage-a/output/encoder_sae.pt \
        --n-positions 50000 \
        --dict-size 1024 \
        --k 64
"""
import argparse
import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# SAE Model (BatchTopK, same as Sandstone/Maia)
# ============================================================================

class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.dict_size = dict_size

        # Init decoder as normalized transpose of encoder (per SAE literature)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=-1)

    def forward(self, x):
        x_centered = x - self.pre_bias
        z = self.encoder(x_centered)

        if self.training:
            # BatchTopK: top n*k across entire batch
            batch_size = x.shape[0]
            total_k = batch_size * self.k
            z_relu = F.relu(z)
            flat_z = z_relu.reshape(-1)
            topk_vals, topk_idx = torch.topk(flat_z, k=min(int(total_k), flat_z.numel()))
            acts = torch.zeros_like(flat_z)
            acts[topk_idx] = topk_vals
            acts = acts.reshape(z.shape)
        else:
            # Eval: simple top-k per sample
            topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
            acts = torch.zeros_like(z)
            acts.scatter_(-1, topk_idx, F.relu(topk_vals))

        x_hat = self.decoder(acts) + self.pre_bias
        return x_hat, acts


# ============================================================================
# Step 1: Extract encoder activations
# ============================================================================

def extract_activations(encoder_path, data_path, n_positions, device):
    """Run encoder on positions, return mean-pooled activations."""
    # Import encoder code (must be on PYTHONPATH)
    from fen_tokenizer import tokenize as chess_tokenize
    from chess_model import ChessEncoder

    print("Loading encoder...", flush=True)
    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder = ChessEncoder(**ckpt['config']).to(device).float()  # float32 — half causes NaN on some positions
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    print(f"Extracting {n_positions} activations...", flush=True)
    activations = []
    t0 = time.time()

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n_positions:
                break
            item = json.loads(line.strip())
            fen = item.get('fen', '')
            if not fen:
                continue

            parts = fen.split()
            if len(parts) == 4: fen += ' 0 1'
            elif len(parts) == 5: fen += ' 1'

            tokens = torch.tensor(
                chess_tokenize(fen).astype(np.int64), dtype=torch.long
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                hidden = encoder(tokens)  # [1, 77, 1024]
                # Mean pool across all 77 tokens (encoder is holistic, not per-square)
                pooled = hidden.mean(dim=1)  # [1, 1024]
                activations.append(pooled.cpu())

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{n_positions} ({elapsed:.0f}s)", flush=True)

    all_acts = torch.cat(activations, dim=0)  # [N, 1024]
    elapsed = time.time() - t0
    print(f"Extracted {all_acts.shape[0]} activations in {elapsed:.0f}s", flush=True)
    print(f"Shape: {all_acts.shape}, norm: mean={all_acts.norm(dim=-1).mean():.2f}", flush=True)

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache()

    return all_acts


# ============================================================================
# Step 2: Train SAE
# ============================================================================

def train_sae(activations, dict_size, k, epochs=3, batch_size=256, lr=3e-4, device='cpu'):
    """Train BatchTopK SAE on activations."""
    input_dim = activations.shape[1]
    N = activations.shape[0]

    # Normalize
    mean = activations.mean(dim=0)
    std = activations.std(dim=0).clamp(min=1e-6)
    acts_norm = (activations - mean) / std

    print(f"\nTraining SAE: {input_dim}d → {dict_size} features, k={k}", flush=True)
    print(f"Data: {N} samples, {epochs} epochs, batch {batch_size}", flush=True)

    sae = BatchTopKSAE(input_dim, dict_size, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Auxiliary loss for dead features
    AUX_COEFF = 1/32
    DEAD_THRESHOLD = 50
    steps_since_fired = torch.zeros(dict_size, device=device)

    dataset = TensorDataset(acts_norm.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0
        total_aux = 0
        n_batches = 0
        sae.train()

        for (batch,) in loader:
            x_hat, acts = sae(batch)
            mse_loss = F.mse_loss(x_hat, batch)

            # Track dead features
            fired = (acts > 0).any(dim=0)
            steps_since_fired[fired] = 0
            steps_since_fired[~fired] += 1
            dead_mask = steps_since_fired > DEAD_THRESHOLD

            # Auxiliary loss: encourage dead features to explain the residual
            aux_loss = torch.tensor(0.0, device=device)
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                residual = (batch - x_hat).detach()
                dead_enc = sae.encoder.weight[dead_mask] @ residual.T
                dead_acts = F.relu(dead_enc).T
                dead_recon = dead_acts @ sae.decoder.weight[:, dead_mask].T
                aux_loss = F.mse_loss(dead_recon, residual)

            loss = mse_loss + AUX_COEFF * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights (per SAE literature)
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=-1)

            total_loss += mse_loss.item()
            total_aux += aux_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_aux = total_aux / n_batches

        # Eval: count active features
        sae.eval()
        with torch.no_grad():
            sample = acts_norm[:1000].to(device)
            _, z = sae(sample)
            active = (z > 0).any(dim=0).sum().item()
            dead = dict_size - active
            l0 = (z > 0).float().sum(dim=-1).mean().item()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, aux={avg_aux:.6f}, "
              f"active={active}/{dict_size}, dead={dead}, L0={l0:.1f}, "
              f"{elapsed:.0f}s", flush=True)

    return sae, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True, help='Chess encoder checkpoint path')
    parser.add_argument('--data', required=True, help='Positions JSONL')
    parser.add_argument('--output', required=True, help='Output SAE checkpoint path')
    parser.add_argument('--n-positions', type=int, default=50000)
    parser.add_argument('--dict-size', type=int, default=1024)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Step 1: Extract
    activations = extract_activations(args.encoder, args.data, args.n_positions, device)

    # Step 2: Train
    sae, mean, std = train_sae(
        activations, args.dict_size, args.k,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device
    )

    # Step 3: Save
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': activations.shape[1],
            'dict_size': args.dict_size,
            'k': args.k,
        },
        'normalization': {
            'mean': mean.numpy().tolist(),
            'std': std.numpy().tolist(),
        },
        'n_positions': activations.shape[0],
    }, args.output)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
