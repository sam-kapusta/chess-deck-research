#!/usr/bin/env python3
"""Chess LLM Integration — connects chess encoder to Qwen via LLaVA-style injection.

Architecture:
1. Chess encoder (frozen) produces [B, 77, 1024] hidden states from FEN
2. Projection layer (trainable) maps to [B, 77, 4096] Qwen-compatible embeddings
3. A special <chess> token in the prompt marks where encoder features are injected
4. Qwen's text embeddings and chess embeddings are combined at the <chess> position
5. The combined sequence is fed through Qwen's transformer layers

Training:
- Phase 1: Freeze encoder + Qwen. Train only projection layer.
- Phase 2: Freeze encoder. Train projection + LoRA on Qwen.

This module provides the `ChessCoachModel` class that wraps everything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional


CHESS_TOKEN = "<chess>"
CHESS_TOKEN_ID = -100  # Placeholder, will be replaced by a real special token


class ChessCoachModel(nn.Module):
    """Full chess coaching model: encoder + projection + LLM.

    Follows LLaVA architecture: visual (chess) tokens injected into
    the LLM's embedding sequence at a placeholder position.
    """

    def __init__(self, encoder, projection, llm, tokenizer):
        super().__init__()
        self.encoder = encoder  # Frozen chess encoder
        self.projection = projection  # Trainable MLP
        self.llm = llm  # Qwen (frozen or LoRA)
        self.tokenizer = tokenizer

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode_position(self, fen_tokens: torch.Tensor) -> torch.Tensor:
        """FEN tokens → projected chess embeddings.

        Args:
            fen_tokens: [B, 77] integer FEN tokens
        Returns:
            [B, 77, llm_dim] projected embeddings ready for injection
        """
        with torch.no_grad():
            hidden = self.encoder(fen_tokens)  # [B, 77, 1024]
        return self.projection(hidden)  # [B, 77, llm_dim]

    def prepare_inputs(self, chess_embeds: torch.Tensor, text_input_ids: torch.Tensor) -> torch.Tensor:
        """Combine chess embeddings with text embeddings.

        Replaces the <chess> placeholder token(s) in the text sequence
        with the 77 chess encoder tokens.

        Args:
            chess_embeds: [B, 77, llm_dim] from encode_position
            text_input_ids: [B, T] text token ids with CHESS_TOKEN_ID placeholder
        Returns:
            [B, 77 + T - 1, llm_dim] combined embeddings (placeholder replaced)
        """
        # Get text embeddings from the LLM's embedding layer
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)  # [B, T, llm_dim]

        # Find the chess placeholder position
        # For simplicity, assume it's at position 0 (first token)
        # In practice, find CHESS_TOKEN_ID in text_input_ids

        batch_combined = []
        for b in range(text_input_ids.shape[0]):
            # Find placeholder position(s)
            chess_positions = (text_input_ids[b] == CHESS_TOKEN_ID).nonzero(as_tuple=True)[0]

            if len(chess_positions) == 0:
                # No chess token — just use text
                batch_combined.append(text_embeds[b])
            else:
                pos = chess_positions[0].item()
                # Combine: [text_before, chess_embeds, text_after]
                before = text_embeds[b, :pos]  # [pos, dim]
                after = text_embeds[b, pos+1:]  # [T-pos-1, dim]
                combined = torch.cat([before, chess_embeds[b], after], dim=0)
                batch_combined.append(combined)

        # Pad to same length
        max_len = max(c.shape[0] for c in batch_combined)
        padded = torch.zeros(len(batch_combined), max_len, text_embeds.shape[-1],
                           device=text_embeds.device, dtype=text_embeds.dtype)
        attention_mask = torch.zeros(len(batch_combined), max_len,
                                   device=text_embeds.device, dtype=torch.long)
        for b, c in enumerate(batch_combined):
            padded[b, :c.shape[0]] = c
            attention_mask[b, :c.shape[0]] = 1

        return padded, attention_mask

    def forward(self, fen_tokens, text_input_ids, labels=None):
        """Full forward pass: FEN → encoder → projection → inject into LLM → generate.

        Args:
            fen_tokens: [B, 77] FEN tokens
            text_input_ids: [B, T] text tokens with chess placeholder
            labels: [B, T'] for training (optional)
        Returns:
            LLM output (logits or loss)
        """
        chess_embeds = self.encode_position(fen_tokens)
        combined_embeds, attention_mask = self.prepare_inputs(chess_embeds, text_input_ids)

        # Run through LLM using inputs_embeds instead of input_ids
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate_coaching(self, fen: str, prompt: str, max_new_tokens=200):
        """Generate coaching text for a position.

        Args:
            fen: Chess position in FEN notation
            prompt: Coaching prompt (e.g., "You played Qd7. What went wrong?")
            max_new_tokens: Maximum tokens to generate
        Returns:
            Generated coaching text string
        """
        from searchless_chess.src.tokenizer import tokenize as chess_tokenize

        # Encode chess position
        fen_tokens = torch.tensor(chess_tokenize(fen).astype(np.int64), dtype=torch.long).unsqueeze(0)
        chess_embeds = self.encode_position(fen_tokens)

        # Tokenize text prompt with chess placeholder
        full_prompt = f"{CHESS_TOKEN} {prompt}"
        text_input_ids = self.tokenizer(full_prompt, return_tensors='pt').input_ids

        # Prepare combined inputs
        combined_embeds, attention_mask = self.prepare_inputs(chess_embeds, text_input_ids)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def describe_architecture():
    """Print the full architecture for documentation."""
    print("=" * 60)
    print("CHESS COACHING MODEL — Architecture")
    print("=" * 60)
    print("""
Components:
  1. Chess Encoder (FROZEN)
     - DeepMind 270M transformer
     - Input: FEN tokens [B, 77]
     - Output: hidden states [B, 77, 1024]
     - Encodes: check (97.7%), game phase (94.8%), material (r=0.99 at layer 1)

  2. Projection Layer (TRAINABLE)
     - 2-layer MLP with GELU
     - Input: [B, 77, 1024]
     - Output: [B, 77, 4096]
     - Params: ~21M

  3. Qwen-4B (TRAINABLE via LoRA)
     - Input: combined [B, 77+T, 4096] (chess + text embeddings)
     - Output: coaching text tokens
     - LoRA rank 16-32, ~10M additional params

Training pipeline:
  Phase 1: Train projection only (encoder + Qwen frozen)
    - LR: 2e-3, batch 128, 1-3 epochs on ~1K coaching examples
    - ~30 min on A10G

  Phase 2: Train projection + LoRA (encoder frozen)
    - LR: 2e-5, batch 32, 3 epochs
    - ~2 hours on A10G

Data:
  - 1,076 coaching moments with (FEN, played_move, best_move, tags)
  - Coaching text generated by Claude Haiku via feigned discovery prompt
  - ~$1 total generation cost

Inference:
  FEN string
    → chess_tokenize() → [77] integer tokens
    → encoder → [77, 1024] hidden states
    → projection → [77, 4096] Qwen-compatible embeddings
    → inject at <chess> placeholder in prompt
    → Qwen generates coaching text
""")


if __name__ == "__main__":
    describe_architecture()
