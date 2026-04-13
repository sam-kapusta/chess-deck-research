"""Export chess encoder to ONNX for lightweight CPU inference.

ONNX Runtime (~100MB) vs PyTorch CPU (~800MB).
~2-5x faster inference on CPU.

Usage:
    python export_onnx.py --checkpoint chess_encoder_270m.pt --output chess_encoder.onnx

Inference with ONNX Runtime:
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession("chess_encoder.onnx")
    tokens = np.array([[...]], dtype=np.int64)  # [1, 77]
    hidden = session.run(None, {"tokens": tokens})[0]  # [1, 77, 1024]
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from chess_encoder import ChessEncoder


def export(checkpoint_path, output_path):
    # Load model
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model = ChessEncoder(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Dummy input
    dummy = torch.zeros(1, 77, dtype=torch.long)

    # Export
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["tokens"],
        output_names=["hidden_states"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "hidden_states": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"Exported to {output_path}")
    print(f"  PyTorch size: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.0f} MB (fp32)")
    print(f"  ONNX size: {Path(output_path).stat().st_size / 1e6:.0f} MB")


def verify(checkpoint_path, onnx_path):
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort

    # PyTorch
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ChessEncoder(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokens = torch.randint(0, 100, (1, 77))
    with torch.no_grad():
        pt_out = model(tokens).numpy()

    # ONNX
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"tokens": tokens.numpy()})[0]

    # Compare
    diff = np.abs(pt_out - onnx_out).max()
    cos = np.dot(pt_out.flatten(), onnx_out.flatten()) / (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out))
    print(f"  Max diff: {diff:.6f}")
    print(f"  Cosine sim: {cos:.6f}")
    print(f"  {'PASS' if cos > 0.999 else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="chess_encoder.onnx")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    export(args.checkpoint, args.output)
    if args.verify:
        verify(args.checkpoint, args.output)
