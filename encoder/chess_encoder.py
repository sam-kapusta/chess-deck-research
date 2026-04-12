"""Chess encoder: DeepMind 270M transformer for position understanding.

Standalone module — no SAIS dependencies. Load the checkpoint and run inference on CPU.

Usage:
    from chess_encoder import load_encoder, encode_position
    encoder = load_encoder("chess_encoder_270m.pt")
    hidden = encode_position(encoder, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # hidden: [77, 1024] tensor — the encoder's understanding of this position
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add searchless_chess to path for tokenizer
_dir = Path(__file__).parent
if (_dir / "searchless_chess").exists():
    sys.path.insert(0, str(_dir))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhtT,bThd->bthd', attn, v)
        return self.out_proj(out.reshape(B, T, D))


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUMLP(dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ChessEncoder(nn.Module):
    """DeepMind 270M chess transformer. 2895 Elo. 16 layers, 1024 dim."""

    def __init__(self, vocab_size=1968, dim=1024, num_layers=16, num_heads=8,
                 ffn_dim=4096, max_seq_len=79):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, tokens):
        B, T = tokens.shape
        bos = torch.zeros(B, 1, dtype=tokens.dtype, device=tokens.device)
        shifted = torch.cat([bos, tokens[:, :-1]], dim=1)
        x = self.token_emb(shifted) * math.sqrt(self.dim)
        x = x + self.pos_emb(torch.arange(T, device=tokens.device))
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


def load_encoder(checkpoint_path, device='cpu'):
    """Load the encoder from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    encoder = ChessEncoder(**config).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    return encoder


def tokenize_fen(fen):
    """Tokenize a FEN string into encoder input tokens."""
    from searchless_chess.src.tokenizer import tokenize
    return torch.tensor(tokenize(fen).astype(np.int64), dtype=torch.long)


@torch.no_grad()
def encode_position(encoder, fen, device='cpu'):
    """Encode a FEN position → [77, 1024] hidden states."""
    tokens = tokenize_fen(fen).unsqueeze(0).to(device)
    return encoder(tokens).squeeze(0)  # [77, 1024]


@torch.no_grad()
def encode_batch(encoder, fens, device='cpu', batch_size=32):
    """Encode multiple FEN positions → [N, 1024] mean-pooled embeddings."""
    all_embeds = []
    for i in range(0, len(fens), batch_size):
        batch_fens = fens[i:i+batch_size]
        tokens = torch.stack([tokenize_fen(f) for f in batch_fens]).to(device)
        hidden = encoder(tokens)  # [B, 77, 1024]
        pooled = hidden.mean(dim=1)  # [B, 1024]
        all_embeds.append(pooled.cpu())
    return torch.cat(all_embeds, dim=0)  # [N, 1024]
