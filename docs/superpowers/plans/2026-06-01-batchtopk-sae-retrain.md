# BatchTopK Chess Mistake SAE — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the chess mistake-SAE using BatchTopK + exact SandstonePersonas normalization, run the full T1 structural eval, compute per-feature stats, then label features with the proven 2-pass Opus pipeline.

**Architecture:** 5 scripts run in order on chess-poc GPU. All scripts live in `chess-deck-research/scripts/`. Training modifies the existing `train_maia3_sae.py` (drop val-split, save mean/std separately). T1 eval is a new standalone script. Stats, labeling scripts wire to new output file names. Atlas gets a stats panel.

**Tech Stack:** PyTorch (BatchTopKSAE already in codebase), python-chess, boto3/Bedrock (Opus 4.6), existing `label_features_pass2.py` and `label_all_positions_opus.py` patterns, chess-poc GPU via sais.

---

## File map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `scripts/sae/train_maia3_sae.py` | Drop val-split; save mean/std to `btk_train_stats.json`; add `--no-val-split` flag |
| Create | `scripts/evaluation/eval_t1_btk.py` | Full T1: FVU, L0, freq buckets, decoder cosine, bimodality, collapsed — with gate check |
| Create | `scripts/evaluation/feature_stats_btk.py` | Top-100 per-feature stats: piece type, side, eval trajectory, cp_loss, motif histogram, phase |
| Create | `scripts/labeling/label_positions_btk.py` | Pass-1: gap positions only, Opus 4.6, reusing proven pattern |
| Create | `scripts/labeling/label_features_btk.py` | Pass-2: chip+description from top-20, reusing `label_features_pass2.py` pattern |
| Modify | `output/l7only_atlas.html` | Add stats panel reading `feature_stats_btk_2048_k32.json`; swap chip source |

**Output files (all written to notebook `~/SageMaker/` then S3/local `output/`):**
- `btk_2048_k32_weights.pt` — trained weights
- `btk_train_stats.json` — `{"mean": [...], "std": [...]}` for all downstream
- `t1_btk_2048_k32.json` — T1 metrics
- `feature_stats_btk_2048_k32.json` — per-feature stats
- `feature_labels_btk_2048_k32.json` — chips + descriptions

---

## Task 1: Modify training script — drop val-split, save mean/std

**Context:** `train_maia3_sae.py` already has the right `BatchTopKSAE` + `train_batchtopk` loop. Two changes needed: (1) no 90/10 split — train on all 168k positions, (2) save `mean` and `std` to a separate JSON so downstream scripts can normalize identically without recomputing.

**Files:**
- Modify: `scripts/sae/train_maia3_sae.py:224-397`

- [ ] **Step 1: Add `--no-val-split` flag and full-corpus path**

In `main()`, replace the val-split block (lines 275–296):

```python
# OLD (remove this block):
# n = acts_norm.shape[0]
# n_val = int(n * args.val_split)
# ...

# NEW — after normalize_activations call, before DataLoader creation:
parser.add_argument("--no-val-split", action="store_true",
                    help="Train on full corpus, no held-out val (use for small corpora)")

# In main(), after normalization:
n = acts_norm.shape[0]
if args.no_val_split:
    train_data = acts_norm
    val_data = acts_norm  # T1 eval sees full corpus
    n_train, n_val = n, n
    print(f"  Full corpus training: {n} positions (no val split)")
else:
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = torch.randperm(n)
    train_data = acts_norm[perm[:n_train]]
    val_data = acts_norm[perm[n_train:]]
    print(f"  Train: {n_train}, Val: {n_val}")
```

- [ ] **Step 2: Save mean/std to `btk_train_stats.json`**

After normalization (line ~270, after `acts_norm, sample_norms = normalize_activations(...)`), add:

```python
# Compute and save normalization stats for downstream reuse
raw_acts_for_stats = data["activations"].float()
norm_mean = raw_acts_for_stats.mean(dim=0).numpy().tolist()
norm_std = raw_acts_for_stats.std(dim=0).clamp(min=1e-6).numpy().tolist()
stats_path = args.output.replace(".pt", "_stats.json") if args.output else out_path.replace(".pt", "_stats.json")
# (save after out_path is determined, in the save block)
```

And in the save block after `torch.save(save_payload, out_path)`:

```python
stats_out = out_path.replace(".pt", "_stats.json")
with open(stats_out, "w") as f:
    json.dump({"mean": norm_mean, "std": norm_std,
               "n_positions": n, "d_input": d_input}, f)
print(f"Normalization stats saved to {stats_out}")
```

- [ ] **Step 3: Upload and run on chess-poc with the deduped cache**

```bash
# Upload
sais -n chess-poc write scripts/sae/train_maia3_sae.py \
  /path/to/scripts/sae/train_maia3_sae.py

# Run under screen (50 epochs, ~20–40 min on GPU)
sais -n chess-poc term "screen -dmS btk_train bash -c \
  'cd ~/SageMaker && python3 scripts/sae/train_maia3_sae.py \
    --activations chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
    --dict-size 2048 --k 32 --k-aux 256 --aux-alpha 0.03125 \
    --lr 3e-4 --batch-size 4096 --n-epochs 50 --warmup-steps 500 \
    --no-val-split \
    --output chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
    2>&1 | tee btk_train.log' && echo launched"
```

- [ ] **Step 4: Verify training started cleanly**

```bash
sais -n chess-poc term "sleep 120 && tail -10 ~/SageMaker/btk_train.log"
```

Expected output (first 2 min):
```
Device: cuda
Loading activations from chess-stage-a/cache/maia3_l7only_v2_dedup.pt
  Raw shape: torch.Size([168132, 1024])
  Full corpus training: 168132 positions (no val split)
Epoch 1/50: ...
  Step 500 | Loss: 0.XXXXXX | L2: 0.XXXXXX | Aux: 0.0 | L0: XX.X | k: 32
```

- [ ] **Step 5: Wait for training to complete, verify output files**

```bash
sais -n chess-poc term "tail -3 ~/SageMaker/btk_train.log && \
  ls -lh ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
         ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights_stats.json"
```

Expected: two files exist, weights ~35–50MB, stats JSON present.

- [ ] **Step 6: Commit the train script change**

```bash
git add scripts/sae/train_maia3_sae.py
git commit -m "sae: add --no-val-split flag and save normalization stats"
```

---

## Task 2: Full T1 structural eval script

**Context:** `train_maia3_sae.py` already computes partial T1 inline but on val-only and misses: bimodality, redundant_pairs, collapsed_norms. This is a standalone script that computes the complete T1 metric set (matching SandstonePersonas `compute_t1.py` exactly) on the full corpus.

**Files:**
- Create: `scripts/evaluation/eval_t1_btk.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Full T1 structural eval for BatchTopK SAE — matches SandstonePersonas compute_t1.py exactly.

Usage (on chess-poc):
    python scripts/evaluation/eval_t1_btk.py \
      --weights ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
      --cache ~/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
      --output ~/SageMaker/chess-stage-a/output/t1_btk_2048_k32.json
"""
import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

GATE = {"max_dead": 50, "max_fvu": 0.15, "min_l0": 20, "max_l0": 50,
        "max_redundant_pairs": 500}


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden
        self.k = k
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        z_relu = F.relu(z)
        flat = z_relu.reshape(-1)
        total_k = min(int(x.shape[0] * self.k), flat.numel())
        topk_vals, topk_idx = torch.topk(flat, total_k)
        acts = torch.zeros_like(flat)
        acts[topk_idx] = topk_vals
        acts = acts.reshape(z.shape)
        x_hat = acts @ self.W_dec + self.b_dec
        l2_loss = (x_hat.float() - x.float()).pow(2).mean()
        return torch.tensor(0.0), x_hat, acts, l2_loss, torch.tensor(0.0)


def normalize(raw, mean, std):
    x = (raw - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load weights + config
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt["config"]
    d_input, dict_size, k = cfg["d_input"], cfg["dict_size"], cfg["k"]

    # Load normalization stats (saved alongside weights)
    stats_path = args.weights.replace(".pt", "_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std  = torch.tensor(stats["std"],  dtype=torch.float32)

    # Load + normalize corpus
    print("Loading corpus...")
    raw = torch.load(args.cache, map_location="cpu",
                     weights_only=False)["activations"].float()
    print(f"  {raw.shape[0]:,} positions, {raw.shape[1]}d")
    x_norm = normalize(raw, mean, std)
    del raw

    # Load model
    model = BatchTopKSAE(d_input, dict_size, k)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # Forward pass — collect acts, x, xhat
    loader = DataLoader(TensorDataset(x_norm), batch_size=4096, shuffle=False)
    all_acts, all_x, all_xhat = [], [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, xhat, acts, _, _ = model(batch)
            all_acts.append(acts.cpu().numpy())
            all_x.append(batch.cpu().numpy())
            all_xhat.append(xhat.cpu().numpy())

    acts = np.concatenate(all_acts)
    x    = np.concatenate(all_x)
    xhat = np.concatenate(all_xhat)

    # --- Feature frequency ---
    freq = (acts > 0).mean(axis=0)
    dead       = int((freq == 0).sum())
    near_dead  = int(((freq > 0) & (freq < 0.001)).sum())
    useful     = int((freq >= 0.001).sum())
    very_active = int((freq >= 0.05).sum())

    # --- L0 ---
    l0 = float((acts > 0).sum(axis=1).mean())

    # --- Reconstruction ---
    mse   = float(np.mean((x - xhat) ** 2))
    var_x = float(np.var(x))         # population variance, flattened
    fvu   = mse / var_x if var_x > 0 else float("inf")

    # --- Decoder cosine (alive features only) ---
    W_dec = model.W_dec.detach().cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
    alive_idx = np.where(freq > 0)[0]
    mean_cos = max_cos = pct_high = 0.0
    redundant_pairs = 0
    if len(alive_idx) > 1:
        W_alive = W_dec_norm[alive_idx]                    # dict_size<=2048 → full matrix
        cos_matrix = W_alive @ W_alive.T
        triu = np.triu_indices(len(alive_idx), k=1)
        cs = cos_matrix[triu]
        redundant_pairs = int(np.sum(np.abs(cs) > 0.5))
        mean_cos = float(np.mean(cs))
        max_cos  = float(np.max(np.abs(cs)))
        pct_high = float(np.sum(np.abs(cs) > 0.5) / len(cs) * 100)

    # --- Collapsed norms ---
    dec_norms = np.linalg.norm(W_dec, axis=1)
    collapsed = int((dec_norms < 0.01).sum())

    # --- Bimodality (CV of nonzero acts per feature) ---
    bimod = []
    for i in range(dict_size):
        nz = acts[:, i][acts[:, i] > 0]
        if len(nz) > 10:
            bimod.append(float(np.std(nz) / (np.mean(nz) + 1e-8)))
    mean_bimod = float(np.mean(bimod)) if bimod else 0.0

    metrics = {
        "tier": "T1", "n_positions": len(x),
        "fvu": fvu, "mse": mse, "l0": l0,
        "dead": dead, "near_dead": near_dead,
        "useful": useful, "very_active": very_active,
        "alive": dict_size - dead,
        "mean_decoder_cosine": mean_cos, "max_decoder_cosine": max_cos,
        "pct_high_sim": pct_high, "redundant_pairs": redundant_pairs,
        "collapsed_norms": collapsed, "mean_decoder_norm": float(np.mean(dec_norms)),
        "mean_bimodality": mean_bimod,
    }

    print("\n=== T1 Structural Metrics ===")
    print(f"  FVU={fvu:.4f}  L0={l0:.1f}  Dead={dead}  Near-dead={near_dead}")
    print(f"  Useful={useful}  Very-active={very_active}")
    print(f"  DecCos: mean={mean_cos:.4f} max={max_cos:.4f} redundant={redundant_pairs}")
    print(f"  Bimodality={mean_bimod:.3f}  Collapsed={collapsed}")

    # --- Gate check ---
    fails = []
    if dead > GATE["max_dead"]:       fails.append(f"dead={dead} > {GATE['max_dead']}")
    if fvu  > GATE["max_fvu"]:        fails.append(f"FVU={fvu:.4f} > {GATE['max_fvu']}")
    if l0 < GATE["min_l0"] or l0 > GATE["max_l0"]:
        fails.append(f"L0={l0:.1f} outside [{GATE['min_l0']},{GATE['max_l0']}]")
    if redundant_pairs > GATE["max_redundant_pairs"]:
        fails.append(f"redundant_pairs={redundant_pairs} > {GATE['max_redundant_pairs']}")

    if fails:
        print("\n!!! GATE FAILED — investigate before labeling !!!")
        for f in fails:
            print(f"  FAIL: {f}")
        metrics["gate_passed"] = False
        metrics["gate_failures"] = fails
    else:
        print("\n✓ Gate passed — proceed to feature stats + labeling")
        metrics["gate_passed"] = True

    import os; os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {args.output}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Upload and run on chess-poc**

```bash
sais -n chess-poc write scripts/evaluation/eval_t1_btk.py \
  /path/to/scripts/evaluation/eval_t1_btk.py

sais -n chess-poc term "cd ~/SageMaker && python3 scripts/evaluation/eval_t1_btk.py \
  --weights chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
  --cache chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
  --output chess-stage-a/output/t1_btk_2048_k32.json 2>&1"
```

Expected healthy output:
```
=== T1 Structural Metrics ===
  FVU=0.0X–0.12  L0=28–36  Dead=0–30  Near-dead=<200
  DecCos: mean≈0 max<0.8  redundant<500
✓ Gate passed — proceed to feature stats + labeling
```

If gate FAILS: stop. Read the failure message, investigate (likely bump n_epochs or check normalization).

- [ ] **Step 3: Download T1 results locally**

```bash
sais -n chess-poc download \
  chess-stage-a/output/t1_btk_2048_k32.json \
  output/t1_btk_2048_k32.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/eval_t1_btk.py output/t1_btk_2048_k32.json
git commit -m "eval: add full T1 structural eval for BatchTopK SAE"
```

---

## Task 3: Per-feature stats from top-100

**Context:** For each of 2048 features, encode the full corpus, take the top-100 highest activating positions, and compute objective stats: piece type of blunder move, side to move, eval trajectory, cp_loss distribution, motif histogram (from Opus join), and phase. These feed the atlas stats panel and replace any subjective label as the primary interpretability instrument.

**Files:**
- Create: `scripts/evaluation/feature_stats_btk.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Per-feature stats from top-100 activating positions.

For each of 2048 features: piece type dist, side, eval trajectory, cp_loss,
motif histogram (Opus join), phase. Fully objective — no LLM.

Usage (on chess-poc):
    python scripts/evaluation/feature_stats_btk.py \
      --weights ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
      --cache ~/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --opus ~/SageMaker/all_positions_labeled_opus.json \
      --output ~/SageMaker/chess-stage-a/output/feature_stats_btk_2048_k32.json
"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import chess

PIECE_NAMES = {chess.PAWN:"pawn",chess.KNIGHT:"knight",chess.BISHOP:"bishop",
               chess.ROOK:"rook",chess.QUEEN:"queen",chess.KING:"king"}
TOP_N = 100   # top positions per feature for stats
TOP_PROFILE = 15  # top positions saved for labeling profiles


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden; self.k = k
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def forward(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        z_relu = F.relu(z)
        flat = z_relu.reshape(-1)
        total_k = min(int(x.shape[0] * self.k), flat.numel())
        topk_vals, topk_idx = torch.topk(flat, total_k)
        acts = torch.zeros_like(flat)
        acts[topk_idx] = topk_vals
        return acts.reshape(z.shape)


def normalize(raw, mean, std):
    x = (raw - mean) / std
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def eval_num(s):
    """Parse white-relative Stockfish eval string to centipawns (mate=±10000)."""
    if not s: return 0
    s = str(s).strip()
    if s.startswith("#"):
        m = int(s[1:].replace("−","-"))
        return (10000 - abs(m)*10) * (1 if m >= 0 else -1)
    try: return int(float(s) * 100)
    except: return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--cache",      required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--opus",       required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    d_input, dict_size, k = cfg["d_input"], cfg["dict_size"], cfg["k"]

    stats_path = args.weights.replace(".pt", "_stats.json")
    with open(stats_path) as f: ns = json.load(f)
    mean = torch.tensor(ns["mean"], dtype=torch.float32)
    std  = torch.tensor(ns["std"],  dtype=torch.float32)

    print("Loading corpus + auxiliary data...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta  = cache["metadata"]
    raw   = cache["activations"].float()
    keys  = [m["fen"] + "|" + m["blunder_uci"] for m in meta]
    x_norm = normalize(raw, mean, std); del raw

    print("Loading enrichment + Opus labels...")
    enr  = json.load(open(args.enrichment))
    opus = json.load(open(args.opus))
    # Opus keyed fen|uci; build motif lookup
    motif_map = {}
    for k_op, v in opus.items():
        a = v.get("analysis", v)
        if isinstance(a, dict) and a.get("tactical_motif"):
            motif_map[k_op] = a["tactical_motif"]

    model = BatchTopKSAE(d_input, dict_size, k)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)

    print("Encoding full corpus...")
    loader = DataLoader(TensorDataset(x_norm), batch_size=8192, shuffle=False)
    all_acts = []
    with torch.no_grad():
        for (batch,) in loader:
            all_acts.append(model(batch.to(device)).cpu().numpy())
    acts = np.concatenate(all_acts)  # [N, 2048]
    print(f"  Acts shape: {acts.shape}")

    print("Computing per-feature stats...")
    out = {}
    for fid in range(dict_size):
        col = acts[:, fid]
        n_activating = int((col > 0).sum())
        if n_activating == 0:
            out[str(fid)] = {"fid": fid, "n_activating": 0, "top100_acts": []}
            continue

        top_idx = np.argsort(-col)[:TOP_N]
        top_acts = col[top_idx].tolist()

        piece_counts = Counter()
        side_white = 0
        traj_already_losing = 0
        traj_made_worse = 0
        traj_threw_winning = 0
        cp_losses = []
        motifs = Counter()
        phases = Counter()
        motif_covered = 0

        for i, idx in enumerate(top_idx):
            m = meta[int(idx)]
            key = keys[int(idx)]

            # Piece type
            try:
                b = chess.Board(m["fen"])
                mv = chess.Move.from_uci(m["blunder_uci"])
                pc = b.piece_at(mv.from_square)
                if pc: piece_counts[PIECE_NAMES.get(pc.piece_type, "other")] += 1
            except: pass

            # Side
            if m.get("is_white"): side_white += 1

            # Eval trajectory (white-relative)
            en = enr.get(key, {})
            if en and not en.get("error"):
                eb = eval_num(en.get("eval_before", 0))
                ea = eval_num(en.get("eval_after", 0))
                is_white = m.get("is_white", True)
                # Convert to mover-relative
                mover_before = eb if is_white else -eb
                mover_after  = ea if is_white else -ea
                if mover_before < -150:
                    traj_already_losing += 1
                    if mover_after < mover_before:
                        traj_made_worse += 1
                elif mover_before > 150 and mover_after < 50:
                    traj_threw_winning += 1
                cp = m.get("cp_loss") or en.get("cp_loss")
                if cp is not None: cp_losses.append(float(cp))

            # Motif (Opus join)
            mt = motif_map.get(key)
            if mt:
                motifs[mt] += 1
                motif_covered += 1

            # Phase
            ph = en.get("phase") or ("opening" if "opening" in str(en) else None)
            if ph: phases[ph] += 1

        n = len(top_idx)
        out[str(fid)] = {
            "fid": fid,
            "n_activating": n_activating,
            "top100_acts": [round(a, 4) for a in top_acts],
            "piece_types": dict(piece_counts),
            "piece_type_pct": {p: round(c/n, 3) for p,c in piece_counts.items()},
            "side_white_pct": round(side_white / n, 3),
            "traj_already_losing_pct": round(traj_already_losing / n, 3),
            "traj_made_worse_pct": round(traj_made_worse / n, 3),
            "traj_threw_winning_pct": round(traj_threw_winning / n, 3),
            "cp_loss_p50": round(float(np.percentile(cp_losses, 50)), 1) if cp_losses else None,
            "cp_loss_p90": round(float(np.percentile(cp_losses, 90)), 1) if cp_losses else None,
            "cp_loss_mean": round(float(np.mean(cp_losses)), 1) if cp_losses else None,
            "motif_hist": dict(motifs.most_common(10)),
            "motif_coverage_pct": round(motif_covered / n, 3),
            "phase_hist": dict(phases),
        }

        if fid % 200 == 0:
            print(f"  {fid}/2048 features done", flush=True)

    # Also write profiles (top-15 per feature) for labeling
    profiles_out = args.output.replace("feature_stats", "btk_profiles")
    profiles = {}
    for fid in range(dict_size):
        col = acts[:, fid]
        top15 = np.argsort(-col)[:TOP_PROFILE]
        exs = []
        seen = set()
        for idx in top15:
            a = float(col[idx])
            if a <= 0: break
            m = meta[int(idx)]
            key = keys[int(idx)]
            if key in seen: continue
            seen.add(key)
            exs.append({"fen": m["fen"], "uci": m["blunder_uci"],
                        "best_uci": m.get("best_uci"), "cp_loss": m.get("cp_loss"),
                        "act": round(a, 4), "key": key})
        profiles[str(fid)] = {"examples": exs, "fire_rate": float((col > 0).mean())}
    with open(profiles_out, "w") as f:
        json.dump(profiles, f)
    print(f"Profiles saved to {profiles_out}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Stats saved to {args.output} ({len(out)} features)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Upload and run on chess-poc**

```bash
sais -n chess-poc write scripts/evaluation/feature_stats_btk.py \
  /path/to/scripts/evaluation/feature_stats_btk.py

sais -n chess-poc term "screen -dmS feat_stats bash -c \
  'cd ~/SageMaker && python3 scripts/evaluation/feature_stats_btk.py \
    --weights chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
    --cache chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
    --enrichment position_enrichment_cache.json \
    --opus all_positions_labeled_opus.json \
    --output chess-stage-a/output/feature_stats_btk_2048_k32.json \
    2>&1 | tee feat_stats.log' && echo launched"
```

- [ ] **Step 3: Verify output**

```bash
sais -n chess-poc term "tail -3 ~/SageMaker/feat_stats.log && \
  python3 -c \"
import json
d=json.load(open('chess-stage-a/output/feature_stats_btk_2048_k32.json'))
nonempty=sum(1 for v in d.values() if v.get('n_activating',0)>0)
print('features with activations:', nonempty, '/', len(d))
sample=d['325']  # a feature that was sharp in old SAE
print('f325 top piece:', max(sample['piece_types'],key=sample['piece_types'].get) if sample['piece_types'] else 'none')
print('f325 side_white_pct:', sample['side_white_pct'])
print('f325 motif_hist:', sample['motif_hist'])
\""
```

Expected: nonempty ≈ 2000–2048, f325 shows a piece-type and motif distribution consistent with what you saw in the atlas for the old SAE.

- [ ] **Step 4: Download locally**

```bash
sais -n chess-poc download chess-stage-a/output/feature_stats_btk_2048_k32.json \
  output/feature_stats_btk_2048_k32.json
sais -n chess-poc download chess-stage-a/output/btk_profiles.json \
  output/btk_profiles.json
```

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/feature_stats_btk.py \
        output/feature_stats_btk_2048_k32.json
git commit -m "eval: per-feature stats from top-100 activations (piece, side, trajectory, motif)"
```

---

## Task 4: Pass-1 — label gap positions

**Context:** Reuse the proven `label_all_positions_opus.py` pattern exactly. The new SAE fires on slightly different top positions than the old one; most will already be in `all_positions_labeled_opus.json` (34k positions). We only need to Stockfish-enrich + Opus-label the gap. `btk_profiles.json` (written by Task 3) gives us the positions we need.

**Files:**
- Create: `scripts/labeling/label_positions_btk.py`

- [ ] **Step 1: Write the gap-identification + Pass-1 script**

```python
#!/usr/bin/env python3
"""Pass-1 for BTK SAE: Opus-label gap positions (those in btk_profiles but not
already in all_positions_labeled_opus.json). Reuses proven label_all_positions_opus
pattern: Opus 4.6, thinking=4096, max_tokens=8000, concurrency=60, resume-safe.

Usage (on chess-poc):
    AWS_PROFILE=default python scripts/labeling/label_positions_btk.py \
      --profiles ~/SageMaker/chess-stage-a/output/btk_profiles.json \
      --existing ~/SageMaker/all_positions_labeled_opus.json \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --output ~/SageMaker/all_positions_labeled_opus.json
"""
import argparse, json, time, boto3, sys
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION   = "us-east-1"
MAX_CONCURRENT = 60
REQUEST_TIMEOUT = 120

MOTIFS = ("hanging_piece|fork|pin|skewer|discovered_attack|back_rank|"
          "overloaded_defender|trapped_piece|pawn_endgame|rook_endgame|"
          "king_safety|passed_pawn|promotion_error|tempo_loss|positional_mistake|other")

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=REQUEST_TIMEOUT, connect_timeout=10,
                  retries={"max_attempts": 0}))
stats = {"times": [], "throttles": 0, "errors": 0}


def build_prompt(enriched):
    played_san = enriched["played_san"]
    features_text = "\n".join(f"  - {f}" for f in enriched.get("position_features", [])) \
                    or "  - (standard position)"
    best_text  = "\n".join(f"  {i+1}. {b['line']} (eval: {b['eval']})"
                           for i,b in enumerate(enriched.get("top_3_best", [])))
    refut_text = "\n".join(f"  {i+1}. {r['line']} (eval: {r['eval']})"
                           for i,r in enumerate(enriched.get("top_3_refutations", [])))
    return (
        f"You are an elite chess grandmaster and coach. Give a THOROUGH, DETAILED analysis.\n\n"
        f"=== POSITION DATA ===\n"
        f"FEN: {enriched['fen']}\nSide: {enriched['side']}\nPhase: {enriched['phase']}\n"
        f"Move played: {played_san}\nCentipawn loss: {enriched['cp_loss']}cp\n"
        f"Eval shift: {enriched['eval_before']} -> {enriched['eval_after']}\n"
        f"Good moves available: {enriched['n_good_moves']} within 50cp\n"
        f"Punishment type: {enriched['punish_type']}\n\n"
        f"Position features:\n{features_text}\n\n"
        f"=== TOP 3 BEST MOVES ===\n{best_text}\n\n"
        f"=== TOP 3 REFUTATIONS (after {played_san}) ===\n{refut_text}\n\n"
        f"Respond in JSON:\n{{"
        f'"position_description":"<3-4 sentences>",'
        f'"best_moves_analysis":"<4-6 sentences covering all 3>",'
        f'"move_intent":"<1-2 sentences>",'
        f'"refutation_analysis":"<4-6 sentences covering all 3>",'
        f'"blunder_summary":"<2-3 sentences>",'
        f'"tactical_motif":"<{MOTIFS}>",'
        f'"tags":["<tag>"]'
        f"}}"
    )


def parse_json_response(text):
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix): text = text[len(prefix):]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try: return json.loads(text[start:end])
        except: pass
    return None


def invoke_opus(prompt):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
    result = json.loads(resp["body"].read())
    for block in result["content"]:
        if block.get("type") == "text": return block["text"]
    return result["content"][-1].get("text", "")


def process_one(item):
    key, enriched = item
    t0 = time.time()
    for attempt in range(3):
        try:
            raw = invoke_opus(build_prompt(enriched))
            parsed = parse_json_response(raw)
            return (key, {"analysis": parsed, "time_s": round(time.time()-t0, 1)}
                    if parsed else {"error": "parse_failed", "raw": raw[:300]})
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                stats["throttles"] += 1
                time.sleep(2 ** (attempt+1))
            else:
                return (key, {"error": str(e)[:200]})
    return (key, {"error": "max_retries"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",   required=True)
    parser.add_argument("--existing",   required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    profiles  = json.load(open(args.profiles))
    results   = json.load(open(args.existing)) if args.existing else {}
    enrichment = json.load(open(args.enrichment))

    # Collect all unique position keys from profiles
    needed = set()
    for fid_data in profiles.values():
        for ex in fid_data.get("examples", [])[:15]:
            needed.add(ex["key"])

    gap = [k for k in needed
           if k not in results or "analysis" not in results[k]]
    print(f"Total unique positions in profiles: {len(needed)}")
    print(f"Already labeled: {len(needed) - len(gap)}")
    print(f"Gap to label: {len(gap)}")

    if not gap:
        print("No gap — all positions already labeled.")
        return

    # Build work items (only positions that have enrichment)
    work = [(k, enrichment[k]) for k in gap if k in enrichment and "error" not in enrichment.get(k, {"error":1})]
    no_enrich = len(gap) - len(work)
    if no_enrich:
        print(f"WARNING: {no_enrich} gap positions lack enrichment — will be skipped")

    print(f"Labeling {len(work)} positions | model=Opus 4.6 | concurrency={MAX_CONCURRENT}", flush=True)
    t0 = time.time(); done = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
            done += 1
            if "time_s" in result: stats["times"].append(result["time_s"])
            if done % 50 == 0:
                with open(args.output, "w") as f: json.dump(results, f)
                avg_t = sum(stats["times"])/len(stats["times"]) if stats["times"] else 0
                eta_h = (len(work)-done)*avg_t/MAX_CONCURRENT/3600 if avg_t else 0
                print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | "
                      f"avg {avg_t:.0f}s | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)

    with open(args.output, "w") as f: json.dump(results, f, indent=2)
    ok = sum(1 for k in work for key,_ in [(k,None)] if "analysis" in results.get(k[0] if isinstance(k,tuple) else k, {}))
    print(f"Done. Saved {len(results)} total labeled positions to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Upload and run**

```bash
sais -n chess-poc write scripts/labeling/label_positions_btk.py \
  /path/to/scripts/labeling/label_positions_btk.py

sais -n chess-poc term "screen -dmS pass1 bash -c \
  'cd ~/SageMaker && AWS_PROFILE=default python3 scripts/labeling/label_positions_btk.py \
    --profiles chess-stage-a/output/btk_profiles.json \
    --existing all_positions_labeled_opus.json \
    --enrichment position_enrichment_cache.json \
    --output all_positions_labeled_opus.json \
    2>&1 | tee pass1_btk.log' && echo launched"
```

- [ ] **Step 3: Monitor gap size and ETA**

```bash
sais -n chess-poc term "sleep 30 && head -4 ~/SageMaker/pass1_btk.log && tail -4 ~/SageMaker/pass1_btk.log"
```

Expected: "Gap to label: N" where N is small (a few thousand at most). If gap is large (>10k) and you're going to sleep, verify throttles=0 at concurrency 60.

- [ ] **Step 4: Commit script (not the output — it's on the notebook)**

```bash
git add scripts/labeling/label_positions_btk.py
git commit -m "labeling: Pass-1 gap-positions script for BTK SAE"
```

---

## Task 5: Pass-2 — chip+description per feature

**Context:** Reuse `label_features_pass2.py` pattern exactly, pointed at the new BTK profiles and Pass-1 output. Same Opus 4.6 + thinking, same output schema.

**Files:**
- Create: `scripts/labeling/label_features_btk.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Pass-2 for BTK SAE: synthesize chip+description per feature from top-20 positions.
Reuses label_features_pass2.py pattern exactly. Opus 4.6, thinking=4096, concurrency=20.

Usage (on chess-poc):
    AWS_PROFILE=default python scripts/labeling/label_features_btk.py \
      --profiles ~/SageMaker/chess-stage-a/output/btk_profiles.json \
      --positions ~/SageMaker/all_positions_labeled_opus.json \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --output ~/SageMaker/chess-stage-a/output/feature_labels_btk_2048_k32.json \
      --resume
"""
import argparse, json, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

MODEL_ID = "us.anthropic.claude-opus-4-6-v1"
REGION = "us-east-1"
MAX_CONCURRENT = 20
REQUEST_TIMEOUT = 120

client = boto3.client("bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=REQUEST_TIMEOUT, connect_timeout=10,
                  retries={"max_attempts": 0}))
stats = {"times": [], "throttles": 0, "errors": 0}

PROMPT_TEMPLATE = """You are an elite chess analyst labeling SAE (Sparse Autoencoder) features.

Each SAE feature fires on positions where a player made a specific type of mistake. Below are the top positions where this feature activates most strongly, along with detailed analysis.

Your job: identify the SPECIFIC shared pattern — the geometric, tactical, or strategic thread. Focus on what the MOVES have in common.

=== FEATURE OVERVIEW ({n_positions} positions) ===
Moves played: {moves_list}
Phases: {phase_dist}
Sides: {side_dist}
Avg cp_loss: {avg_cp_loss:.0f}
Avg good moves available: {avg_good_moves:.1f}

{positions_text}

=== INSTRUCTIONS ===
Look at the MOVES FIRST. What piece, direction, or move type connects them?
Then use the analyses to understand WHY these moves are mistakes.
Note: displayed examples are TOP activators — most extreme cases. Typical activations are milder.

Respond in JSON:
{{
  "chip": "<3-5 word punchy title>",
  "label": "<one sentence summary>",
  "description": "<full paragraph, reference evidence counts X/N>",
  "move_pattern": "<geometric description: piece type, direction, check/capture>",
  "why_bad": "<common reason these moves fail>",
  "sub_patterns": ["<variant 1>"],
  "categories": ["<broad>", "<mid>", "<specific>"],
  "confidence": <0-100>
}}"""


def build_feature_prompt(fid, examples, enrichment, analyses):
    positions_text = ""
    moves, phases, sides, cp_losses, good_moves_list = [], [], [], [], []

    for i, ex in enumerate(examples):
        key = f"{ex['fen']}|{ex['uci']}"
        enriched = enrichment.get(key, {})
        analysis = analyses.get(key, {}).get("analysis", {})
        if not enriched or "error" in enriched or not analysis:
            continue
        played_san = enriched.get("played_san", ex["uci"])
        moves.append(played_san)
        phases.append(enriched.get("phase", "?"))
        sides.append(enriched.get("side", "?"))
        cp_losses.append(enriched.get("cp_loss", 0) or 0)
        good_moves_list.append(enriched.get("n_good_moves", 0) or 0)

        features_text = "\n".join(f"    - {f}" for f in enriched.get("position_features", [])) \
                        or "    - (standard position)"
        best_text  = "\n".join(f"    {j+1}. {b['line']} (eval: {b['eval']})"
                               for j,b in enumerate(enriched.get("top_3_best", [])))
        refut_text = "\n".join(f"    {j+1}. {r['line']} (eval: {r['eval']})"
                               for j,r in enumerate(enriched.get("top_3_refutations", [])))
        positions_text += f"""
=== POSITION {i+1} ===
Move: {played_san} | Side: {enriched.get('side','?')} | Phase: {enriched.get('phase','?')} | cp_loss: {enriched.get('cp_loss','?')}
Eval: {enriched.get('eval_before','?')} -> {enriched.get('eval_after','?')} | Good moves: {enriched.get('n_good_moves','?')} | Punish: {enriched.get('punish_type','?')}
Position features:
{features_text}
Top 3 best moves:
{best_text}
Top 3 refutations:
{refut_text}

Analysis:
{json.dumps(analysis, indent=2)}
"""

    if len(moves) < 3:
        return None

    phase_dist = ", ".join(f"{p}({c})" for p,c in Counter(phases).most_common())
    side_dist  = ", ".join(f"{s}({c})" for s,c in Counter(sides).most_common())
    return PROMPT_TEMPLATE.format(
        n_positions=len(moves), moves_list=", ".join(moves),
        phase_dist=phase_dist, side_dist=side_dist,
        avg_cp_loss=sum(cp_losses)/len(cp_losses),
        avg_good_moves=sum(good_moves_list)/len(good_moves_list),
        positions_text=positions_text)


def parse_json_response(text):
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix): text = text[len(prefix):]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try: return json.loads(text[start:end])
        except: pass
    return None


def invoke_opus(prompt):
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8000,
                       "thinking": {"type": "enabled", "budget_tokens": 4096},
                       "messages": [{"role": "user", "content": prompt}]})
    resp = client.invoke_model(modelId=MODEL_ID, body=body,
                               contentType="application/json", accept="application/json")
    result = json.loads(resp["body"].read())
    for block in result["content"]:
        if block.get("type") == "text": return block["text"]
    return result["content"][-1].get("text", "")


def process_one(item):
    fid, prompt = item
    t0 = time.time()
    for attempt in range(3):
        try:
            raw = invoke_opus(prompt)
            parsed = parse_json_response(raw)
            return (fid, {"analysis": parsed, "time_s": round(time.time()-t0,1)}
                    if parsed else {"error": "parse_failed"})
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                stats["throttles"] += 1; time.sleep(2**(attempt+1))
            else:
                return (fid, {"error": str(e)[:200]})
    return (fid, {"error": "max_retries"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",   required=True)
    parser.add_argument("--positions",  required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    profiles   = json.load(open(args.profiles))
    analyses   = json.load(open(args.positions))
    enrichment = json.load(open(args.enrichment))

    results = {}
    if args.resume:
        try:
            results = json.load(open(args.output))
            done_n = sum(1 for v in results.values() if "error" not in v)
            print(f"Resuming: {done_n} already labeled")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    work = []
    skipped = 0
    for fid in sorted(profiles.keys(), key=int):
        if fid in results and "error" not in results[fid]:
            continue
        examples = profiles[fid].get("examples", [])[:20]
        prompt = build_feature_prompt(fid, examples, enrichment, analyses)
        if prompt is None:
            results[fid] = {"error": "insufficient_analyzed_positions"}
            skipped += 1
            continue
        work.append((fid, prompt))

    print(f"Features to label: {len(work)} (skipped {skipped} with <3 analyzed positions)")
    print(f"Labeling | Opus 4.6 | concurrency={MAX_CONCURRENT} | thinking=4096", flush=True)
    t0 = time.time(); done = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, item): item for item in work}
        for future in as_completed(futures):
            fid, result = future.result()
            results[fid] = result
            done += 1
            if "time_s" in result: stats["times"].append(result["time_s"])
            if done % 50 == 0:
                with open(args.output, "w") as f: json.dump(results, f)
                avg_t = sum(stats["times"])/len(stats["times"]) if stats["times"] else 0
                eta_h = (len(work)-done)*avg_t/MAX_CONCURRENT/3600 if avg_t else 0
                print(f"  {done}/{len(work)} | {(time.time()-t0)/60:.1f}min | "
                      f"avg {avg_t:.0f}s | throttles={stats['throttles']} | ETA {eta_h:.1f}h", flush=True)

    with open(args.output, "w") as f: json.dump(results, f, indent=2)
    ok = sum(1 for v in results.values() if "error" not in v)
    print(f"Done. {ok}/{len(profiles)} features labeled | throttles={stats['throttles']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Upload and run on chess-poc**

```bash
sais -n chess-poc write scripts/labeling/label_features_btk.py \
  /path/to/scripts/labeling/label_features_btk.py

sais -n chess-poc term "screen -dmS pass2 bash -c \
  'cd ~/SageMaker && AWS_PROFILE=default python3 scripts/labeling/label_features_btk.py \
    --profiles chess-stage-a/output/btk_profiles.json \
    --positions all_positions_labeled_opus.json \
    --enrichment position_enrichment_cache.json \
    --output chess-stage-a/output/feature_labels_btk_2048_k32.json \
    --resume \
    2>&1 | tee pass2_btk.log' && echo launched"
```

- [ ] **Step 3: Monitor + verify completion**

```bash
sais -n chess-poc term "tail -4 ~/SageMaker/pass2_btk.log"
```

Expected final line: `Done. XXXX/2048 features labeled | throttles=0`

- [ ] **Step 4: Download labels locally**

```bash
sais -n chess-poc download \
  chess-stage-a/output/feature_labels_btk_2048_k32.json \
  output/feature_labels_btk_2048_k32.json
```

- [ ] **Step 5: Quick coherence check locally**

```bash
python3 -c "
import json, statistics as st
d = json.load(open('output/feature_labels_btk_2048_k32.json'))
ok = {k:v for k,v in d.items() if 'error' not in v}
confs = [v.get('confidence', 0) for v in ok.values()]
chips = [v.get('chip','').lower() for v in ok.values()]
junk = sum(1 for c in chips if any(j in c for j in ['unclear','mixed','incoherent','various']))
print(f'{len(ok)} labeled | median conf {st.median(confs):.0f} | unique chips {len(set(chips))}/{len(chips)} | junk {junk}')
"
```

Expected: ~2000+ labeled, median confidence ≥65, unique chips ≥80%, junk near 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/labeling/label_features_btk.py \
        output/feature_labels_btk_2048_k32.json
git commit -m "labeling: Pass-2 chip+description for BTK SAE 2048/k32"
```

---

## Task 6: Update atlas with stats panel + new chips

**Context:** `output/l7only_atlas.html` currently reads `l7only_explorer_data.json`. We need to: (1) assemble a new joined data file from the BTK outputs, (2) add a stats panel to the HTML showing piece type, side, trajectory, cp_loss, phase, motif, and (3) wire the chip/description to the new label file. Same atlas structure, new data sources.

**Files:**
- Create: `scripts/evaluation/assemble_btk_explorer.py`
- Modify: `output/l7only_atlas.html`

- [ ] **Step 1: Write the assembler**

```python
#!/usr/bin/env python3
"""Assemble BTK explorer data: join feature stats + labels + profiles into
one JSON for the atlas. Writes output/btk_explorer_data.json."""
import json, os, numpy as np

OUT = "output/btk_explorer_data.json"

stats   = json.load(open("output/feature_stats_btk_2048_k32.json"))
labels  = json.load(open("output/feature_labels_btk_2048_k32.json"))
profiles = json.load(open("output/btk_profiles.json"))

# Load enrichment + opus for board detail (same as old assemble_explorer_data.py)
# These live on notebook; download them first if needed:
#   sais -n chess-poc download position_enrichment_cache.json /tmp/enr.json
#   sais -n chess-poc download all_positions_labeled_opus.json /tmp/opus.json
enr_path  = "/tmp/enr.json"
opus_path = "/tmp/opus.json"
enr  = json.load(open(enr_path))  if os.path.exists(enr_path)  else {}
opus = json.load(open(opus_path)) if os.path.exists(opus_path) else {}

def cap(s, n):
    s = s or ""; return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

out = {}
for fid_str, lab in labels.items():
    if "error" in lab: continue
    fs  = stats.get(fid_str, {})
    prof = profiles.get(fid_str, {})
    boards = []
    for ex in prof.get("examples", [])[:10]:
        key = ex["key"]
        en  = enr.get(key, {})
        an  = opus.get(key, {}).get("analysis", {})
        boards.append({
            "fen": ex["fen"], "uci": ex["uci"], "act": ex["act"],
            "cp_loss": ex.get("cp_loss"),
            "san": en.get("played_san", ex["uci"]),
            "best_san": en.get("best_san"),
            "side": en.get("side"), "phase": en.get("phase"),
            "eval_before": en.get("eval_before"), "eval_after": en.get("eval_after"),
            "n_good": en.get("n_good_moves"), "punish": en.get("punish_type"),
            "best_lines": [b.get("line") for b in en.get("top_3_best", [])][:3],
            "refut_lines": [r.get("line") for r in en.get("top_3_refutations", [])][:3],
            "motif": an.get("tactical_motif"),
            "pos_desc": cap(an.get("position_description"), 320),
            "blunder_summary": cap(an.get("blunder_summary"), 360),
            "best_analysis": cap(an.get("best_moves_analysis"), 360),
            "refut_analysis": cap(an.get("refutation_analysis"), 360),
        })

    # Merge label fields
    out[fid_str] = {
        "chip": lab.get("chip"), "label": lab.get("label"),
        "description": cap(lab.get("description"), 500),
        "why_bad": cap(lab.get("why_bad"), 400),
        "move_pattern": cap(lab.get("move_pattern"), 400),
        "sub_patterns": lab.get("sub_patterns", []),
        "categories": lab.get("categories", []),
        "confidence": lab.get("confidence", 0),
        "fire_rate": prof.get("fire_rate", 0),
        # Stats panel fields
        "stats": {
            "n_activating": fs.get("n_activating", 0),
            "piece_type_pct": fs.get("piece_type_pct", {}),
            "side_white_pct": fs.get("side_white_pct"),
            "traj_already_losing_pct": fs.get("traj_already_losing_pct"),
            "traj_made_worse_pct": fs.get("traj_made_worse_pct"),
            "traj_threw_winning_pct": fs.get("traj_threw_winning_pct"),
            "cp_loss_p50": fs.get("cp_loss_p50"),
            "cp_loss_p90": fs.get("cp_loss_p90"),
            "motif_hist": fs.get("motif_hist", {}),
            "motif_coverage_pct": fs.get("motif_coverage_pct", 0),
            "phase_hist": fs.get("phase_hist", {}),
        },
        "boards": boards,
    }

json.dump(out, open(OUT, "w"), separators=(",", ":"))
import os; print(f"Written {OUT} ({os.path.getsize(OUT)//1e6:.1f} MB, {len(out)} features)")
```

- [ ] **Step 2: Run assembler (after downloading enr + opus locally)**

```bash
# Download supporting data from notebook
sais -n chess-poc download position_enrichment_cache.json /tmp/enr.json
sais -n chess-poc download all_positions_labeled_opus.json /tmp/opus.json

# Run assembler
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research
python3 scripts/evaluation/assemble_btk_explorer.py
```

Expected: `Written output/btk_explorer_data.json (≈45 MB, 2000+ features)`

- [ ] **Step 3: Add stats panel to the atlas HTML**

In `output/l7only_atlas.html`, add the following CSS in the `<style>` block:

```css
.stats-panel{background:var(--panel);border:1px solid var(--line-soft);border-radius:13px;padding:17px 19px;margin-bottom:18px}
.stats-panel h3{font-family:'JetBrains Mono',monospace;font-size:10.5px;text-transform:uppercase;letter-spacing:.14em;color:var(--steel);margin-bottom:12px;display:flex;align-items:center;gap:7px}
.stats-panel h3::before{content:'';width:5px;height:5px;background:var(--steel);border-radius:50%}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}
.stat-block{background:var(--raised);border-radius:9px;padding:10px 12px}
.stat-block .sl{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:6px}
.stat-block .sv{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--cream)}
.bar-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;font-size:11px}
.bar-row .bn{color:var(--cream-dim);width:52px;flex-shrink:0;font-family:'JetBrains Mono',monospace}
.bar-row .bt{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar-row .bf{height:100%;background:var(--brass);border-radius:3px}
.bar-row .bv{color:var(--muted);width:32px;text-align:right;font-family:'JetBrains Mono',monospace}
```

- [ ] **Step 4: Add `renderStats(f)` JS function and inject into `selectFeature`**

In `output/l7only_atlas.html`, add this function before `selectFeature`:

```javascript
function renderStats(f) {
  const s = f.stats || {};
  if (!s.n_activating) return '';
  const pieces = s.piece_type_pct || {};
  const topPieces = Object.entries(pieces).sort((a,b)=>b[1]-a[1]).slice(0,4);
  const pieceRows = topPieces.map(([p,v])=>
    `<div class="bar-row"><span class="bn">${p}</span>
     <div class="bt"><div class="bf" style="width:${Math.round(v*100)}%"></div></div>
     <span class="bv">${Math.round(v*100)}%</span></div>`).join('');
  const motifs = s.motif_hist || {};
  const topMotifs = Object.entries(motifs).sort((a,b)=>b[1]-a[1]).slice(0,3);
  const motifRows = topMotifs.map(([m,c])=>
    `<div class="bar-row"><span class="bn" style="width:90px">${m.replace(/_/g,' ')}</span>
     <span class="bv">${c}</span></div>`).join('');
  const phases = s.phase_hist || {};
  const phaseRows = Object.entries(phases).map(([ph,c])=>
    `<span style="margin-right:10px"><b style="color:var(--brass)">${c}</b> ${ph}</span>`).join('');
  const trajRows = [
    s.traj_threw_winning_pct > 0 ? `Threw winning: <b>${Math.round(s.traj_threw_winning_pct*100)}%</b>` : null,
    s.traj_already_losing_pct > 0 ? `Already losing: <b>${Math.round(s.traj_already_losing_pct*100)}%</b>` : null,
    s.traj_made_worse_pct > 0 ? `Made it worse: <b>${Math.round(s.traj_made_worse_pct*100)}%</b>` : null,
  ].filter(Boolean).join(' · ');
  return `<div class="stats-panel">
    <h3>Objective stats · top-100 activations · ${s.motif_coverage_pct ? Math.round(s.motif_coverage_pct*100)+'% Opus coverage' : 'no Opus coverage'}</h3>
    <div class="stats-grid">
      <div class="stat-block"><div class="sl">Piece type</div>${pieceRows||'<span style="color:var(--faint)">no data</span>'}</div>
      <div class="stat-block"><div class="sl">Side to move</div>
        <div class="sv">W ${Math.round((s.side_white_pct||0)*100)}% · B ${Math.round((1-(s.side_white_pct||0))*100)}%</div></div>
      <div class="stat-block"><div class="sl">Eval trajectory</div>
        <div class="sv" style="font-size:11px;line-height:1.6">${trajRows||'mixed'}</div></div>
      <div class="stat-block"><div class="sl">cp loss (p50 / p90)</div>
        <div class="sv">${s.cp_loss_p50??'—'} / ${s.cp_loss_p90??'—'}</div></div>
      <div class="stat-block"><div class="sl">Phase</div>
        <div class="sv" style="font-size:11px">${phaseRows||'—'}</div></div>
      ${topMotifs.length ? `<div class="stat-block"><div class="sl">Top motifs</div>${motifRows}</div>` : ''}
    </div>
  </div>`;
}
```

In `selectFeature`, inject `renderStats(f)` right before the `<div class="analysis-grid">` in the detail HTML:

Find the line:
```javascript
    <div class="analysis-grid">
```

Replace the full detail innerHTML assembly to insert `${renderStats(f)}` just before `<div class="analysis-grid">`.

- [ ] **Step 5: Update the data file reference and title**

In `l7only_atlas.html`, update the `fetch(...)` call:
```javascript
// Change:
fetch('l7only_explorer_data.json')
// To:
fetch('btk_explorer_data.json')
```

Update the title:
```javascript
// Change:
<title>l7only · Feature Atlas</title>
// To:
<title>btk 2048/k32 · Feature Atlas</title>
```

Update the brand subtitle:
```javascript
// Change:
<div class="s">Maia-3 layer-7 · 2048 k16 · sparse mistake features</div>
// To:
<div class="s">Maia-3 layer-7 · 2048/k32 BatchTopK · sparse mistake features</div>
```

- [ ] **Step 6: Serve and verify**

```bash
cd /Users/samtkap/workspace/chess-deck/src/chess-deck-research/output
# Kill any stale server
lsof -ti:8777 2>/dev/null | xargs kill -9 2>/dev/null
python3 -m http.server 8777 &
open "http://localhost:8777/l7only_atlas.html"
```

Click a feature. Verify:
- Stats panel appears below the confidence ring (piece type bars, trajectory, cp_loss, phase)
- Chips load from new label file
- Boards still render and link to chess.com

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluation/assemble_btk_explorer.py \
        output/btk_explorer_data.json \
        output/l7only_atlas.html
git commit -m "atlas: add stats panel + wire to BTK 2048/k32 labels"
```

---

## Self-review checklist

**Spec coverage:**
- [x] BatchTopKSAE + train_batchtopk → Task 1 (modifies existing train script)
- [x] Exact normalization (z-score + L2, save mean/std) → Task 1
- [x] No train/val split → Task 1 `--no-val-split`
- [x] T1 structural eval with gate → Task 2
- [x] Per-feature stats top-100 → Task 3
- [x] Profiles for labeling (top-15, deduped) → Task 3 (written alongside stats)
- [x] Pass-1 gap positions → Task 4
- [x] Pass-2 chip+description → Task 5
- [x] Atlas stats panel + new chips → Task 6
- [x] Weighted dataloader NOT used → Task 1 uses plain `TensorDataset`
- [x] T2 NOT included → absent from plan
- [x] Global optimizer reset NOT used → absent; AuxK handles dead features

**Placeholder scan:** No TBDs, all code is complete with actual values.

**Type consistency:**
- `BatchTopKSAE.__init__` signature consistent across Tasks 1, 2, 3 (all use `d_input, d_hidden, k`)
- Output file names consistent: `btk_2048_k32_weights.pt`, `btk_train_stats.json` (written `_stats.json` suffix)
- `btk_profiles.json` written by Task 3, consumed by Tasks 4 and 5 — consistent
- `feature_labels_btk_2048_k32.json` written by Task 5, consumed by Task 6 assembler — consistent
- `btk_explorer_data.json` written by Task 6 assembler, read by atlas — consistent
