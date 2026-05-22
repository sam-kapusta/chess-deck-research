"""
Extract intermediate activations from Maia 3 for SAE training.

Probes the residual stream after the last transformer layer (layer 7)
before the policy/value head split. This 512-dim representation encodes
the model's full understanding of the position — ideal for SAE decomposition.

Usage:
  python scripts/maia3_activations.py --positions data/positions.txt --output data/maia3_activations.npy

The positions file should be one FEN per line.
"""

import argparse
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

# Maia 3 model path
MODEL_PATH = Path(__file__).parent.parent.parent / "chess-deck-code/backend/mcp/maia3_models/maia3_simplified.onnx"

# The layer to probe — output of last transformer block's residual connection.
# We find this by looking for the Add node after layers.7/linear2
PROBE_LAYER = "/model/transformer/layers.7/Add_2_output_0"


def preprocess_fen(fen: str) -> tuple[np.ndarray, int, int]:
    """Convert FEN to Maia 3 input format: (tokens, elo_self, elo_oppo)."""
    # Maia 3 input: [batch, 64, 12] one-hot piece encoding
    # Piece order: P N B R Q K p n b r q k (white first, then black)
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    board_str = fen.split()[0]
    tokens = np.zeros((64, 12), dtype=np.float32)

    # FEN goes rank 8 to rank 1 (top to bottom)
    sq = 0
    for char in board_str:
        if char == '/':
            continue
        elif char.isdigit():
            sq += int(char)
        else:
            if char in piece_map:
                tokens[sq, piece_map[char]] = 1.0
            sq += 1

    # Add side-to-move, castling, en passant as additional features
    # (Maia 3 encodes these in the 355-dim token projection input)
    # For now we just use the piece placement — the model's tokenizer
    # handles the rest internally via the full 355-dim input

    return tokens


def preprocess_fen_full(fen: str) -> np.ndarray:
    """
    Full Maia 3 preprocessing matching the TypeScript tensor.ts implementation.
    Returns [64, 12] float32 array (simplified — piece placement only).

    For full accuracy, port the complete tensor.ts preprocessing.
    This simplified version works for activation collection since
    the SAE only needs diverse position representations.
    """
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    parts = fen.split()
    board_str = parts[0]

    tokens = np.zeros((64, 12), dtype=np.float32)
    sq = 0
    for char in board_str:
        if char == '/':
            continue
        elif char.isdigit():
            sq += int(char)
        else:
            if char in piece_map:
                tokens[sq, piece_map[char]] = 1.0
            sq += 1

    return tokens


def inspect_model():
    """Print available intermediate outputs for probing."""
    import onnx
    model = onnx.load(str(MODEL_PATH))

    print("Available probe points (outputs of transformer Add nodes):")
    print("-" * 60)

    for node in model.graph.node:
        if node.op_type == "Add" and "layers" in (node.name or ""):
            for out in node.output:
                print(f"  {out}")

    print(f"\nRecommended: last layer residual = output after layers.7 FFN")

    # Find the actual node
    for node in model.graph.node:
        if "layers.7" in (node.name or "") and node.op_type == "Add":
            print(f"  → {node.output[0]}")


def extract_activations(
    fens: list[str],
    elo_self: int = 1500,
    elo_oppo: int = 1500,
    probe_layer: str = PROBE_LAYER,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run positions through Maia 3 and extract activations at the probe layer.

    Returns: np.ndarray of shape [num_positions, 64, 512] or [num_positions, 512]
    depending on the probe layer's output shape.
    """
    # Create session with intermediate output
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # We need to add the probe layer as an explicit output
    import onnx
    model = onnx.load(str(MODEL_PATH))

    # Add intermediate tensor as output (model uses float16 internally)
    probe_output = onnx.helper.make_tensor_value_info(probe_layer, onnx.TensorProto.FLOAT16, None)
    model.graph.output.append(probe_output)

    # Save modified model to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        temp_path = f.name

    try:
        session = ort.InferenceSession(temp_path, sess_options)

        all_activations = []

        for i in range(0, len(fens), batch_size):
            batch_fens = fens[i:i + batch_size]
            batch_tokens = np.stack([preprocess_fen_full(fen) for fen in batch_fens])

            # Elo inputs
            batch_elo_self = np.full(len(batch_fens), elo_self, dtype=np.float32)
            batch_elo_oppo = np.full(len(batch_fens), elo_oppo, dtype=np.float32)

            feeds = {
                "tokens": batch_tokens,
                "elo_self": batch_elo_self,
                "elo_oppo": batch_elo_oppo,
            }

            # Run inference — get both original outputs and probe layer
            output_names = [o.name for o in session.get_outputs()]
            results = session.run(output_names, feeds)

            # The probe layer is the last output (we appended it)
            probe_activations = results[-1]
            all_activations.append(probe_activations)

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i + len(batch_fens)}/{len(fens)} positions...")

        return np.concatenate(all_activations, axis=0)

    finally:
        import os
        os.unlink(temp_path)


def uci_to_square_index(uci: str) -> int:
    """Convert UCI from-square (e.g., 'e2e4' → 'e2' → index 12)."""
    file_idx = ord(uci[0]) - ord('a')  # 0-7
    rank_idx = int(uci[1]) - 1         # 0-7
    # Maia's square ordering: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
    return rank_idx * 8 + file_idx


def pool_activations(raw: np.ndarray, pool_mode: str, ucis: list = None) -> np.ndarray:
    """
    Pool (N, 64, 512) activations based on mode.

    - 'from-square': extract activation at the from-square of each UCI move → (N, 512)
    - 'mean': mean across all 64 squares → (N, 512)
    - 'all': keep full spatial representation → (N, 64, 512)
    """
    if pool_mode == "all":
        return raw
    elif pool_mode == "mean":
        return raw.mean(axis=1).astype(np.float32)
    elif pool_mode == "from-square":
        if ucis is None:
            raise ValueError("from-square pooling requires --ucis or JSONL input with uci field")
        result = np.zeros((raw.shape[0], raw.shape[2]), dtype=np.float32)
        for i, uci in enumerate(ucis):
            sq_idx = uci_to_square_index(uci)
            result[i] = raw[i, sq_idx].astype(np.float32)
        return result
    else:
        raise ValueError(f"Unknown pool mode: {pool_mode}")


def main():
    parser = argparse.ArgumentParser(description="Extract Maia 3 activations for SAE training")
    parser.add_argument("--inspect", action="store_true", help="Print available probe points")
    parser.add_argument("--positions", type=str, help="File: one FEN per line, or JSONL with {fen, uci} fields")
    parser.add_argument("--from-cache", type=str, help="Load positions from existing blunder cache .pt file (has fens + metadata)")
    parser.add_argument("--output", type=str, default="maia3_activations.pt", help="Output .pt file (torch format, matches SAE training)")
    parser.add_argument("--elo", type=int, default=1500, help="Elo to condition on")
    parser.add_argument("--probe", type=str, default=PROBE_LAYER, help="Layer to probe")
    parser.add_argument("--pool", type=str, default="from-square", choices=["from-square", "mean", "all"],
                        help="Pooling: from-square (uses blunder UCI), mean (avg 64 squares), all (64x512)")
    parser.add_argument("--limit", type=int, default=None, help="Max positions to process")
    args = parser.parse_args()

    if args.inspect:
        inspect_model()
        return

    # Load positions — either from existing cache or from a file
    fens = []
    ucis = []
    metadata = []

    if args.from_cache:
        import torch
        print(f"Loading positions from existing cache: {args.from_cache}")
        cache = torch.load(args.from_cache, map_location="cpu", weights_only=False)
        if "metadata" in cache:
            for m in cache["metadata"]:
                fens.append(m["fen"])
                ucis.append(m.get("blunder_uci", m.get("uci", "")))
                metadata.append(m)
        elif "fens" in cache:
            fens = list(cache["fens"])
            ucis = [""] * len(fens)
        print(f"  Loaded {len(fens)} positions")
    elif args.positions:
        import json
        with open(args.positions) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Try JSONL
                if line.startswith("{"):
                    obj = json.loads(line)
                    fens.append(obj["fen"])
                    ucis.append(obj.get("uci", obj.get("blunder_uci", "")))
                    metadata.append(obj)
                else:
                    fens.append(line)
                    ucis.append("")
    else:
        print("Usage:")
        print("  python maia3_activations.py --from-cache blunder_acts_200k.pt --output maia3_blunder_acts.pt")
        print("  python maia3_activations.py --positions positions.jsonl --output maia3_acts.pt")
        print("  python maia3_activations.py --inspect")
        sys.exit(1)

    if args.limit:
        fens = fens[:args.limit]
        ucis = ucis[:args.limit]
        metadata = metadata[:args.limit] if metadata else []

    if args.pool == "from-square" and not any(ucis):
        print("ERROR: --pool from-square requires UCI moves. Use --from-cache or JSONL with uci field.")
        sys.exit(1)

    print(f"Extracting activations from {len(fens)} positions")
    print(f"  Elo: {args.elo}")
    print(f"  Probe: {args.probe}")
    print(f"  Pool: {args.pool}")
    print(f"  Model: {MODEL_PATH}")

    # Extract raw activations: (N, 64, 512)
    raw = extract_activations(fens, elo_self=args.elo, elo_oppo=args.elo, probe_layer=args.probe)

    # Pool
    pooled = pool_activations(raw, args.pool, ucis if args.pool == "from-square" else None)

    print(f"\nRaw shape: {raw.shape}")
    print(f"Pooled shape: {pooled.shape}")

    # Compute normalization stats
    if pooled.ndim == 2:
        mean = pooled.mean(axis=0)
        std = pooled.std(axis=0)
        std[std < 1e-6] = 1.0  # avoid division by zero
    else:
        mean = pooled.reshape(-1, pooled.shape[-1]).mean(axis=0)
        std = pooled.reshape(-1, pooled.shape[-1]).std(axis=0)
        std[std < 1e-6] = 1.0

    # Save in torch format matching existing SAE training expectations
    import torch
    output = {
        "activations": torch.from_numpy(pooled),
        "mean": mean,
        "std": std,
        "fens": fens,
        "metadata": metadata if metadata else [{"fen": f, "uci": u} for f, u in zip(fens, ucis)],
        "config": {
            "model": "maia3",
            "probe_layer": args.probe,
            "pool": args.pool,
            "elo": args.elo,
            "n_positions": len(fens),
        },
    }

    torch.save(output, args.output)
    print(f"Saved to {args.output}")
    print(f"  activations: {pooled.shape} ({pooled.dtype})")
    print(f"  mean: {mean.shape}")
    print(f"  std: {std.shape}")
    print(f"  metadata: {len(metadata)} entries")


if __name__ == "__main__":
    main()
