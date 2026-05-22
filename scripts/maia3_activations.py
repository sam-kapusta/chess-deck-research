"""
Extract intermediate activations from Maia 3 for SAE training.

Probes the residual stream after the last transformer layer (layer 7)
before the policy/value head split. This 512-dim representation encodes
the model's full understanding of the position — ideal for SAE decomposition.

Preprocessing matches frontend tensor.ts exactly:
- Always white-to-move orientation (mirror black positions)
- Square ordering: rank-1-first (a1=0, b1=1, ..., h8=63)
- Piece channels: 0-5 = white PNBRQK, 6-11 = black pnbrqk

Usage:
  python scripts/maia3_activations.py --from-cache blunder_acts_200k.pt --pool from-square --elo-mode random
  python scripts/maia3_activations.py --positions data/positions.jsonl --pool mean --elo 1500
"""

import argparse
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

# Maia 3 model path
MODEL_PATH = Path(__file__).parent.parent.parent / "chess-deck-code/backend/mcp/maia3_models/maia3_simplified.onnx"

# The layer to probe — output of last transformer block's residual connection.
PROBE_LAYER = "/model/transformer/layers.7/Add_2_output_0"

PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']


def mirror_fen(fen: str) -> str:
    """Mirror FEN vertically and swap piece colors (black-to-move → white-to-move).

    Matches tensor.ts mirrorFen exactly.
    """
    parts = fen.split()
    pos, active, castling, ep = parts[0], parts[1], parts[2] if len(parts) > 2 else '-', parts[3] if len(parts) > 3 else '-'
    halfmove = parts[4] if len(parts) > 4 else '0'
    fullmove = parts[5] if len(parts) > 5 else '1'

    # Reverse ranks and swap piece colors
    ranks = pos.split('/')
    mirrored_ranks = []
    for rank in reversed(ranks):
        swapped = ''
        for c in rank:
            if c.isupper():
                swapped += c.lower()
            elif c.islower():
                swapped += c.upper()
            else:
                swapped += c
        mirrored_ranks.append(swapped)
    mirrored_pos = '/'.join(mirrored_ranks)

    # Swap castling rights
    if castling == '-':
        mirrored_castling = '-'
    else:
        has = set(castling)
        mc = ''
        if 'k' in has: mc += 'K'
        if 'q' in has: mc += 'Q'
        if 'K' in has: mc += 'k'
        if 'Q' in has: mc += 'q'
        mirrored_castling = mc if mc else '-'

    # Mirror en passant
    if ep != '-' and len(ep) >= 2:
        mirrored_ep = ep[0] + str(9 - int(ep[1]))
    else:
        mirrored_ep = '-'

    mirrored_active = 'b' if active == 'w' else 'w'

    return f"{mirrored_pos} {mirrored_active} {mirrored_castling} {mirrored_ep} {halfmove} {fullmove}"


def preprocess_fen(fen: str) -> tuple[np.ndarray, bool]:
    """Convert FEN to Maia 3 input format matching tensor.ts.

    Returns (tokens [64, 12], was_mirrored).
    Square ordering: rank-1-first (a1=0, b1=1, ..., h8=63).
    Always outputs white-to-move orientation.
    """
    parts = fen.split()
    active = parts[1] if len(parts) > 1 else 'w'

    was_mirrored = (active == 'b')
    effective_fen = mirror_fen(fen) if was_mirrored else fen

    piece_map = {c: i for i, c in enumerate(PIECE_CHARS)}
    board_str = effective_fen.split()[0]

    tokens = np.zeros((64, 12), dtype=np.float32)
    rows = board_str.split('/')

    # FEN ranks go 8→1 top-to-bottom. Map to rank-1-first indexing.
    for row_idx, rank_str in enumerate(rows):
        rank = 7 - row_idx  # rank 0 = rank 1 (bottom)
        file = 0
        for char in rank_str:
            if char.isdigit():
                file += int(char)
            else:
                if char in piece_map:
                    square = rank * 8 + file
                    tokens[square, piece_map[char]] = 1.0
                file += 1

    return tokens, was_mirrored


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

    for node in model.graph.node:
        if "layers.7" in (node.name or "") and node.op_type == "Add":
            print(f"  → {node.output[0]}")


def extract_activations(
    fens: list[str],
    elo_self: int | list[int] = 1500,
    elo_oppo: int | list[int] = 1500,
    probe_layer: str = PROBE_LAYER,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, list[bool]]:
    """
    Run positions through Maia 3 and extract activations at the probe layer.

    Handles board mirroring for black-to-move positions internally.
    Uses one ONNX session per chunk because the modified graph corrupts
    session state after a single batched run (known ORT bug with added outputs).

    Args:
        elo_self: scalar or per-position list of Elos for side-to-move
        elo_oppo: scalar or per-position list of Elos for opponent
        chunk_size: positions per ONNX session (limited by memory; 5K safe on 22GB)

    Returns: (activations [N, 64, 512], was_mirrored [N])
    """
    if isinstance(elo_self, (int, float)):
        elo_self_list = [int(elo_self)] * len(fens)
    else:
        elo_self_list = list(elo_self)
    if isinstance(elo_oppo, (int, float)):
        elo_oppo_list = [int(elo_oppo)] * len(fens)
    else:
        elo_oppo_list = list(elo_oppo)

    # Prepare modified ONNX model (done once)
    import onnx
    model = onnx.load(str(MODEL_PATH))
    probe_output = onnx.helper.make_tensor_value_info(probe_layer, onnx.TensorProto.FLOAT16, None)
    model.graph.output.append(probe_output)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        temp_path = f.name

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    try:
        all_activations = []
        all_mirrored = []

        for chunk_start in range(0, len(fens), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(fens))
            chunk_fens = fens[chunk_start:chunk_end]

            # Preprocess entire chunk
            chunk_tokens = []
            chunk_mirrored = []
            for fen in chunk_fens:
                tokens, was_mirrored = preprocess_fen(fen)
                chunk_tokens.append(tokens)
                chunk_mirrored.append(was_mirrored)

            chunk_tokens_np = np.stack(chunk_tokens)
            chunk_elo_self = np.array(elo_self_list[chunk_start:chunk_end], dtype=np.float32)
            chunk_elo_oppo = np.array(elo_oppo_list[chunk_start:chunk_end], dtype=np.float32)

            feeds = {
                "tokens": chunk_tokens_np,
                "elo_self": chunk_elo_self,
                "elo_oppo": chunk_elo_oppo,
            }

            # Fresh session per chunk (ORT bug: session corrupts after first batched run)
            session = ort.InferenceSession(temp_path, sess_options)
            results = session.run([probe_layer], feeds)

            all_activations.append(results[0])
            all_mirrored.extend(chunk_mirrored)
            del session

            print(f"  Processed {chunk_end}/{len(fens)} positions...")

        return np.concatenate(all_activations, axis=0), all_mirrored

    finally:
        import os
        os.unlink(temp_path)


def uci_to_square_index(uci: str, was_mirrored: bool = False) -> int:
    """Convert UCI from-square to activation tensor index.

    Square ordering: rank-1-first (a1=0, b1=1, ..., h8=63).
    If the position was mirrored (black-to-move), the from-square must also
    be mirrored vertically (e.g., e2 → e7).
    """
    file_idx = ord(uci[0]) - ord('a')  # 0-7
    rank_idx = int(uci[1]) - 1         # 0-7

    if was_mirrored:
        rank_idx = 7 - rank_idx

    return rank_idx * 8 + file_idx


def pool_activations(raw: np.ndarray, pool_mode: str, ucis: list = None,
                     was_mirrored: list[bool] = None) -> np.ndarray:
    """
    Pool (N, 64, 512) activations based on mode.

    - 'diff': (to_square - from_square) activation → (N, 512). Encodes what
      changes along the move path. Best tactical theme clustering.
    - 'from-square': extract activation at the from-square of each UCI move → (N, 512)
    - 'mean': mean across all 64 squares → (N, 512)
    - 'all': keep full spatial representation → (N, 64, 512)

    For modes using UCI squares, was_mirrored indicates which positions were
    flipped (black-to-move) so square indices are correctly mirrored.
    """
    if pool_mode == "all":
        return raw
    elif pool_mode == "mean":
        return raw.mean(axis=1).astype(np.float32)
    elif pool_mode == "diff":
        if ucis is None:
            raise ValueError("diff pooling requires UCIs")
        if was_mirrored is None:
            was_mirrored = [False] * raw.shape[0]
        result = np.zeros((raw.shape[0], raw.shape[2]), dtype=np.float32)
        for i, uci in enumerate(ucis):
            from_idx = uci_to_square_index(uci, was_mirrored[i])
            to_file = ord(uci[2]) - ord('a')
            to_rank = int(uci[3]) - 1
            if was_mirrored[i]:
                to_rank = 7 - to_rank
            to_idx = to_rank * 8 + to_file
            result[i] = raw[i, to_idx].astype(np.float32) - raw[i, from_idx].astype(np.float32)
        return result
    elif pool_mode == "from-square":
        if ucis is None:
            raise ValueError("from-square pooling requires UCIs")
        if was_mirrored is None:
            was_mirrored = [False] * raw.shape[0]
        result = np.zeros((raw.shape[0], raw.shape[2]), dtype=np.float32)
        for i, uci in enumerate(ucis):
            sq_idx = uci_to_square_index(uci, was_mirrored[i])
            result[i] = raw[i, sq_idx].astype(np.float32)
        return result
    else:
        raise ValueError(f"Unknown pool mode: {pool_mode}")


def main():
    parser = argparse.ArgumentParser(description="Extract Maia 3 activations for SAE training")
    parser.add_argument("--inspect", action="store_true", help="Print available probe points")
    parser.add_argument("--positions", type=str, help="File: one FEN per line, or JSONL with {fen, uci} fields")
    parser.add_argument("--from-cache", type=str, help="Load positions from existing blunder cache .pt file (has fens + metadata)")
    parser.add_argument("--from-json", type=str, help="Load positions from metadata JSON (lightweight, no activations)")
    parser.add_argument("--output", type=str, default="maia3_activations.pt", help="Output .pt file (torch format, matches SAE training)")
    parser.add_argument("--elo", type=int, default=None, help="Fixed Elo for all positions (overrides --elo-mode)")
    parser.add_argument("--elo-mode", type=str, default="random", choices=["fixed", "random"],
                        help="'fixed' uses --elo for all; 'random' samples uniformly from 1100-2600")
    parser.add_argument("--probe", type=str, default=PROBE_LAYER, help="Layer to probe")
    parser.add_argument("--pool", type=str, default="diff", choices=["diff", "from-square", "mean", "all"],
                        help="Pooling: diff (to-from, best for tactics), from-square, mean, all")
    parser.add_argument("--limit", type=int, default=None, help="Max positions to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Elo assignment")
    args = parser.parse_args()

    if args.inspect:
        inspect_model()
        return

    # Load positions — either from existing cache or from a file
    fens = []
    ucis = []
    metadata = []

    if args.from_json:
        import json as json_mod
        print(f"Loading positions from JSON: {args.from_json}")
        with open(args.from_json) as f:
            metadata = json_mod.load(f)
        for m in metadata:
            fens.append(m["fen"])
            ucis.append(m.get("blunder_uci", m.get("uci", "")))
        print(f"  Loaded {len(fens)} positions")
    elif args.from_cache:
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
        print("  python maia3_activations.py --from-cache blunder_acts_200k.pt --pool diff --elo-mode random")
        print("  python maia3_activations.py --positions positions.jsonl --pool mean --elo 1500")
        print("  python maia3_activations.py --inspect")
        sys.exit(1)

    if args.limit:
        fens = fens[:args.limit]
        ucis = ucis[:args.limit]
        metadata = metadata[:args.limit] if metadata else []

    if args.pool in ("from-square", "diff") and not any(ucis):
        print(f"ERROR: --pool {args.pool} requires UCI moves. Use --from-cache or JSONL with uci field.")
        sys.exit(1)

    # Determine Elo per position
    n = len(fens)
    rng = np.random.default_rng(args.seed)

    if args.elo is not None:
        elo_self_list = [args.elo] * n
        elo_oppo_list = [args.elo] * n
        elo_mode = "fixed"
    elif args.elo_mode == "random":
        elo_self_list = rng.integers(600, 2601, size=n).tolist()
        elo_oppo_list = rng.integers(600, 2601, size=n).tolist()
        elo_mode = "random"
    else:
        elo_self_list = [1500] * n
        elo_oppo_list = [1500] * n
        elo_mode = "fixed"

    print(f"Extracting activations from {len(fens)} positions")
    print(f"  Elo mode: {elo_mode}" + (f" (fixed={args.elo})" if args.elo else f" (uniform 600-2600)"))
    print(f"  Probe: {args.probe}")
    print(f"  Pool: {args.pool}")
    print(f"  Model: {MODEL_PATH}")

    # Extract raw activations: (N, 64, 512) + mirror flags
    raw, was_mirrored = extract_activations(
        fens, elo_self=elo_self_list, elo_oppo=elo_oppo_list, probe_layer=args.probe)

    # Pool
    needs_ucis = args.pool in ("from-square", "diff")
    pooled = pool_activations(
        raw, args.pool,
        ucis if needs_ucis else None,
        was_mirrored if needs_ucis else None,
    )

    print(f"\nRaw shape: {raw.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"  Mirrored positions (black-to-move): {sum(was_mirrored)}/{len(was_mirrored)}")

    # Compute normalization stats
    if pooled.ndim == 2:
        mean = pooled.mean(axis=0)
        std = pooled.std(axis=0)
        std[std < 1e-6] = 1.0
    else:
        mean = pooled.reshape(-1, pooled.shape[-1]).mean(axis=0)
        std = pooled.reshape(-1, pooled.shape[-1]).std(axis=0)
        std[std < 1e-6] = 1.0

    # Save in torch format matching SAE training expectations
    import torch
    output = {
        "activations": torch.from_numpy(pooled),
        "mean": mean,
        "std": std,
        "fens": fens,
        "elo_self": elo_self_list,
        "elo_oppo": elo_oppo_list,
        "was_mirrored": was_mirrored,
        "metadata": metadata if metadata else [{"fen": f, "uci": u} for f, u in zip(fens, ucis)],
        "config": {
            "model": "maia3",
            "probe_layer": args.probe,
            "pool": args.pool,
            "elo_mode": elo_mode,
            "elo_range": [600, 2600] if elo_mode == "random" else [args.elo or 1500],
            "n_positions": len(fens),
            "seed": args.seed,
        },
    }

    torch.save(output, args.output)
    print(f"Saved to {args.output}")
    print(f"  activations: {pooled.shape} ({pooled.dtype})")
    print(f"  mean: {mean.shape}")
    print(f"  std: {std.shape}")
    print(f"  metadata: {len(metadata)} entries")
    print(f"  elo_self/elo_oppo saved for reproducibility")


if __name__ == "__main__":
    main()
