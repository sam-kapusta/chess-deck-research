#!/usr/bin/env python3
"""Train a Tactics SAE on Maia hidden activations from Lichess puzzle positions.

Same Maia model, same SAE architecture, different training data.
The puzzle SAE should learn tactical motif features instead of positional features.

Run on SAIS notebook (GPU available but CPU is fine for SAE training).

Steps:
1. Download Lichess puzzle CSV
2. Sample N puzzles, extract FENs + themes
3. Run FENs through Maia 1800 → extract hidden activations
4. Train BatchTopK SAE on activations
5. Save checkpoint + metadata (themes per feature for validation)
"""
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ──
PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLE_FILE = Path("lichess_db_puzzle.csv")
N_PUZZLES = 200_000  # positions to train on
BATCH_SIZE = 256
DICT_SIZE = 2048
K = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cpu"  # SAE trains fine on CPU, Maia inference too

OUTPUT_DIR = Path("tactics_sae")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── BatchTopK SAE (same as position SAE) ──
class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k

    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        acts = torch.zeros_like(z)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        x_hat = self.decoder(acts) + self.pre_bias
        return x_hat, acts

    def encode(self, x):
        z = self.encoder(x - self.pre_bias)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        acts = torch.zeros_like(z)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts


def download_puzzles():
    """Download and decompress Lichess puzzle database."""
    zst_file = Path("lichess_db_puzzle.csv.zst")
    if PUZZLE_FILE.exists():
        print(f"Puzzle file already exists: {PUZZLE_FILE}")
        return

    if not zst_file.exists():
        print(f"Downloading puzzles from {PUZZLE_URL}...")
        subprocess.run(["curl", "-L", "-o", str(zst_file), PUZZLE_URL], check=True)

    print("Decompressing...")
    # Try zstd, fall back to python
    try:
        subprocess.run(["zstd", "-d", str(zst_file), "-o", str(PUZZLE_FILE)], check=True)
    except FileNotFoundError:
        print("zstd not found, trying python zstandard...")
        import zstandard
        with open(zst_file, "rb") as compressed:
            dctx = zstandard.ZstdDecompressor()
            with open(PUZZLE_FILE, "wb") as output:
                dctx.copy_stream(compressed, output)

    print(f"Puzzles ready: {PUZZLE_FILE}")


def load_puzzles(n=N_PUZZLES):
    """Load N puzzles from CSV. Returns list of (fen, themes_list)."""
    puzzles = []
    with open(PUZZLE_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        print(f"CSV header: {header}")

        for row in reader:
            if len(row) < 8:
                continue
            fen = row[1]
            themes = row[7].split() if row[7] else []
            rating = int(row[3]) if row[3] else 1500

            # Filter: reasonable rating range, has themes
            if not themes or rating < 800 or rating > 2500:
                continue

            puzzles.append({"fen": fen, "themes": themes, "rating": rating})

            if len(puzzles) >= n:
                break

    print(f"Loaded {len(puzzles)} puzzles")

    # Theme distribution
    from collections import Counter
    theme_counts = Counter()
    for p in puzzles:
        for t in p["themes"]:
            theme_counts[t] += 1
    print(f"\nTop 20 themes:")
    for theme, count in theme_counts.most_common(20):
        print(f"  {theme:30s}  {count:6d}  ({count/len(puzzles)*100:.1f}%)")

    return puzzles


def extract_maia_activations(fens, batch_size=64):
    """Run FENs through Maia and extract hidden layer activations."""
    from maia2 import model as maia_model, inference as maia_inference

    print("Loading Maia model...")
    m = maia_model.MAIA()
    m.load("rapid", device=DEVICE)
    prepared = maia_inference.prepare_model(m, DEVICE)

    # Hook to capture hidden activations
    activations = []
    hook_handle = None

    def hook_fn(module, input, output):
        # Capture the output of the last transformer layer
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())

    # Find the right layer to hook
    # Maia2 architecture: transformer layers, we want the last one's output
    for name, module in m.model.named_modules():
        pass  # Find the last module name
    # Hook the model's output layer (before the head)
    # For Maia2, the hidden state is 1024-dim
    last_layer = None
    for name, module in m.model.named_modules():
        if "layers" in name and "norm" not in name:
            last_layer = (name, module)
    if last_layer:
        hook_handle = last_layer[1].register_forward_hook(hook_fn)
        print(f"Hooked layer: {last_layer[0]}")

    all_acts = []
    valid_indices = []

    for i in range(0, len(fens), batch_size):
        batch_fens = fens[i:i + batch_size]
        activations.clear()

        for j, fen in enumerate(batch_fens):
            try:
                # Run Maia inference
                result = maia_inference.maia_rapid(m, prepared, fen, 1800, DEVICE)
                if activations:
                    # Mean-pool the spatial dimensions
                    act = activations[-1]
                    if act.dim() == 3:
                        act = act.mean(dim=1)  # (1, seq_len, dim) → (1, dim)
                    elif act.dim() == 2:
                        act = act.mean(dim=0, keepdim=True)  # (seq_len, dim) → (1, dim)
                    all_acts.append(act.squeeze(0).cpu())
                    valid_indices.append(i + j)
                activations.clear()
            except Exception as e:
                activations.clear()
                continue

        if (i + batch_size) % 10000 < batch_size:
            print(f"  {len(all_acts)}/{len(fens)} activations extracted...")

    if hook_handle:
        hook_handle.remove()

    if not all_acts:
        print("ERROR: No activations extracted. Check Maia hook.")
        sys.exit(1)

    result = torch.stack(all_acts)
    print(f"Extracted {result.shape[0]} activations, dim={result.shape[1]}")
    return result, valid_indices


def train_sae(activations, epochs=EPOCHS):
    """Train BatchTopK SAE on activations."""
    input_dim = activations.shape[1]
    print(f"\nTraining SAE: input_dim={input_dim}, dict_size={DICT_SIZE}, k={K}")

    # Normalize
    mean = activations.mean(dim=0)
    std = activations.std(dim=0) + 1e-8
    activations_normed = (activations - mean) / std

    sae = BatchTopKSAE(input_dim, DICT_SIZE, K).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    # Auxiliary loss for dead features
    AUX_COEFF = 1/32
    DEAD_THRESHOLD = 50
    steps_since_fired = torch.zeros(DICT_SIZE, device=DEVICE)

    n = activations_normed.shape[0]
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0
        epoch_aux = 0
        batches = 0

        for i in range(0, n, BATCH_SIZE):
            batch = activations_normed[perm[i:i + BATCH_SIZE]].to(DEVICE)
            x_hat, acts = sae(batch)
            mse_loss = F.mse_loss(x_hat, batch)

            # Track dead features
            fired = (acts > 0).any(dim=0)
            steps_since_fired[fired] = 0
            steps_since_fired[~fired] += 1
            dead_mask = steps_since_fired > DEAD_THRESHOLD

            # Auxiliary loss: encourage dead features to explain the residual
            aux_loss = torch.tensor(0.0, device=DEVICE)
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

            epoch_loss += mse_loss.item()
            epoch_aux += aux_loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        avg_aux = epoch_aux / batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            with torch.no_grad():
                all_acts = sae.encode(activations_normed[:5000].to(DEVICE))
                fire_count = (all_acts > 0).sum(dim=0)
                dead = (fire_count == 0).sum().item()
                useful = DICT_SIZE - dead

            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  aux={avg_aux:.6f}  dead={dead}  useful={useful}  ({elapsed:.0f}s)")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")

    # Final stats
    with torch.no_grad():
        all_acts = sae.encode(activations_normed.to(DEVICE))
        fire_count = (all_acts > 0).sum(dim=0)
        dead = (fire_count == 0).sum().item()
        print(f"Dead features: {dead}/{DICT_SIZE}")
        print(f"Useful features: {DICT_SIZE - dead}")

    return sae, mean, std


def save_checkpoint(sae, mean, std, puzzles, valid_indices):
    """Save SAE checkpoint + puzzle metadata."""
    checkpoint = {
        "model_state_dict": sae.state_dict(),
        "config": {"input_dim": mean.shape[0], "dict_size": DICT_SIZE, "k": K},
        "normalization": {"mean": mean, "std": std},
        "n_positions": len(valid_indices),
        "epochs": EPOCHS,
        "training_data": "lichess_puzzles",
    }
    path = OUTPUT_DIR / f"tactics_sae_{DICT_SIZE}_k{K}.pt"
    torch.save(checkpoint, path)
    print(f"Saved SAE to {path}")

    # Save puzzle themes for each valid position (for feature→theme correlation)
    theme_data = []
    for idx in valid_indices:
        if idx < len(puzzles):
            theme_data.append({"themes": puzzles[idx]["themes"], "rating": puzzles[idx]["rating"]})
    with open(OUTPUT_DIR / "puzzle_themes.json", "w") as f:
        json.dump(theme_data, f)
    print(f"Saved {len(theme_data)} puzzle themes")


def analyze_features(sae, activations, mean, std, puzzles, valid_indices):
    """Correlate SAE features with puzzle themes."""
    activations_normed = (activations - mean) / (std + 1e-8)

    with torch.no_grad():
        all_acts = sae.encode(activations_normed.to(DEVICE))  # (N, dict_size)

    # For each feature, find which puzzle themes co-occur
    from collections import Counter, defaultdict
    feature_themes = defaultdict(Counter)

    for i, idx in enumerate(valid_indices):
        if idx >= len(puzzles):
            continue
        themes = puzzles[idx]["themes"]
        active_features = (all_acts[i] > 0).nonzero(as_tuple=True)[0].tolist()
        for fid in active_features:
            for theme in themes:
                feature_themes[fid][theme] += 1

    # For each feature, find its dominant theme
    feature_labels = {}
    for fid in range(DICT_SIZE):
        themes = feature_themes.get(fid, Counter())
        if themes:
            top_theme, top_count = themes.most_common(1)[0]
            total = sum(themes.values())
            specificity = top_count / total  # how specific is this feature to one theme?
            feature_labels[str(fid)] = {
                "top_theme": top_theme,
                "specificity": round(specificity, 3),
                "top_count": top_count,
                "total_activations": total,
                "top_3_themes": themes.most_common(3),
            }

    with open(OUTPUT_DIR / "feature_theme_labels.json", "w") as f:
        json.dump(feature_labels, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("FEATURE → THEME CORRELATION")
    print(f"{'='*60}")

    # How many features are specific to one theme?
    specificities = [v["specificity"] for v in feature_labels.values()]
    print(f"\nFeature specificity distribution:")
    for threshold in [0.5, 0.4, 0.3, 0.2]:
        n = sum(1 for s in specificities if s >= threshold)
        print(f"  ≥{threshold:.0%} specific: {n}/{len(specificities)} features")

    # Top features per theme
    print(f"\nMost theme-specific features:")
    sorted_features = sorted(feature_labels.items(), key=lambda x: -x[1]["specificity"])
    for fid, info in sorted_features[:20]:
        themes_str = ", ".join(f"{t}({c})" for t, c in info["top_3_themes"])
        print(f"  Feature {fid:4s}: {info['top_theme']:20s}  specificity={info['specificity']:.2f}  ({themes_str})")


def main():
    print("=" * 60)
    print("TACTICS SAE — Training on Lichess Puzzles via Maia")
    print("=" * 60)

    # Step 1: Download puzzles
    download_puzzles()

    # Step 2: Load puzzles
    puzzles = load_puzzles(N_PUZZLES)

    # Step 3: Extract Maia activations
    fens = [p["fen"] for p in puzzles]
    activations, valid_indices = extract_maia_activations(fens)

    # Save activations for reuse
    torch.save({"activations": activations, "valid_indices": valid_indices},
               OUTPUT_DIR / "puzzle_activations.pt")
    print(f"Saved activations cache")

    # Step 4: Train SAE
    sae, mean, std = train_sae(activations)

    # Step 5: Save
    save_checkpoint(sae, mean, std, puzzles, valid_indices)

    # Step 6: Analyze features
    analyze_features(sae, activations, mean, std, puzzles, valid_indices)

    print("\nDone!")


if __name__ == "__main__":
    main()
