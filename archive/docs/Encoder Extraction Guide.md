# Encoder Extraction Guide — DeepMind 270M

**For the cloud2 agent doing the JAX→PyTorch conversion.**

## What We Want

The last hidden state `h` from the transformer, AFTER the final layer norm but BEFORE the logits projection. In the original code (transformer.py line 274):

```python
if config.apply_post_ln:
    h = layer_norm(h)  # <-- We want THIS h
logits = hk.Linear(config.output_size)(h)  # <-- NOT this
```

Shape: `[batch, sequence_length, 1024]` for the 270M model.

## Input Handling

The model takes input sequences of length 79 for action-value prediction:
- Tokens 0-76: FEN encoding (77 tokens from tokenizer.py)
- Token 77: Action token (the move being evaluated)
- Token 78: Return bucket token (dummy for inference)

**For feature extraction, we only need the FEN tokens (0-76).** Feed a sequence of length 77 (just the FEN), extract `h` of shape `[batch, 77, 1024]`.

**Important:** The model uses `use_causal_mask=False` — full bidirectional attention. This means every token's representation incorporates information from the entire board. The FEN representation is already a fully contextualized encoding.

## Two Ways to Use the Hidden State

### Option A: Full sequence (77 tokens × 1024)
Like LLaVA using all 257 image patches. Each token becomes a separate input to the projection layer. The LLM receives 77 extra tokens representing the chess position.

**Pro:** Maximum information preserved.
**Con:** Adds 77 tokens to the LLM's sequence length. For Qwen 4B with 1024 max_seq_length, this uses ~8% of the context window.

### Option B: Mean pool to single vector (1 × 1024)
Average all 77 token representations into one 1024-dim vector. Project through MLP to a single token.

**Pro:** Minimal impact on LLM context. Simple.
**Con:** Loses spatial information (which square corresponds to what).

### Recommendation: Start with Option A (full sequence)
The 77 extra tokens are cheap for modern LLMs. And the spatial information matters — "the knight on d5" is encoded in a specific position in the sequence. Pooling destroys this.

## The PyTorch Encoder Class

```python
class ChessEncoder(nn.Module):
    """Extracts position representations from the DeepMind 270M model."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        # Load converted weights
        self.embedding = nn.Embedding(vocab_size, 1024)
        self.pos_embedding = nn.Embedding(79, 1024)  # learned positional
        self.layers = nn.ModuleList([
            TransformerBlock(dim=1024, num_heads=8, ffn_dim=4096)
            for _ in range(16)
        ])
        self.final_ln = nn.LayerNorm(1024)
        # Load weights from converted checkpoint
        self._load_weights(checkpoint_path)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, 77] integer tensor of FEN tokens
        Returns:
            [batch, 77, 1024] hidden state representations
        """
        # Right-shift (prepend BOS=0, drop last)
        bos = torch.zeros(token_ids.shape[0], 1, dtype=token_ids.dtype, device=token_ids.device)
        shifted = torch.cat([bos, token_ids[:, :-1]], dim=1)

        # Embed
        x = self.embedding(shifted) * math.sqrt(1024)
        x = x + self.pos_embedding(torch.arange(x.shape[1], device=x.device))

        # Transformer blocks (no causal mask — full attention)
        for layer in self.layers:
            x = layer(x)  # pre-norm + attention + MLP

        # Final layer norm
        x = self.final_ln(x)
        return x  # [batch, 77, 1024]

    def encode_fen(self, fen: str) -> torch.Tensor:
        """Convenience: FEN string → hidden state."""
        from searchless_chess.src.tokenizer import tokenize
        tokens = torch.tensor(tokenize(fen), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            return self.forward(tokens)
```

## Validation

After conversion, verify by comparing JAX and PyTorch outputs:

```python
# For 10 test positions
for fen in test_fens:
    jax_h = extract_hidden_jax(fen)    # shape [77, 1024]
    pytorch_h = encoder.encode_fen(fen)  # shape [1, 77, 1024]

    cosine_sim = F.cosine_similarity(
        torch.tensor(jax_h).flatten(),
        pytorch_h.squeeze(0).flatten(),
        dim=0
    )
    assert cosine_sim > 0.99, f"Mismatch: {cosine_sim}"
```

## Weight Mapping (JAX Haiku → PyTorch)

Haiku uses a nested dict structure: `params['transformer_decoder/...']['w']`

Key mappings:
```
Haiku: transformer_decoder/~/embed/embeddings → PyTorch: self.embedding.weight
Haiku: transformer_decoder/~/multi_head_dot_product_attention/linear/w → PyTorch: layer.attn.q_proj.weight (transposed)
Haiku: transformer_decoder/~/multi_head_dot_product_attention/linear_1/w → PyTorch: layer.attn.k_proj.weight (transposed)
Haiku: transformer_decoder/~/multi_head_dot_product_attention/linear_2/w → PyTorch: layer.attn.v_proj.weight (transposed)
Haiku: transformer_decoder/~/multi_head_dot_product_attention/linear_3/w → PyTorch: layer.attn.out_proj.weight (transposed)
Haiku: transformer_decoder/~/linear/w → PyTorch: layer.mlp.gate_proj.weight (transposed)
Haiku: transformer_decoder/~/linear_1/w → PyTorch: layer.mlp.up_proj.weight (transposed)
Haiku: transformer_decoder/~/linear_2/w → PyTorch: layer.mlp.down_proj.weight (transposed)
Haiku: transformer_decoder/~/layer_norm/scale → PyTorch: layer.ln1.weight
Haiku: transformer_decoder/~/layer_norm/offset → PyTorch: layer.ln1.bias
```

Note: Haiku Linear stores weights as [in, out]. PyTorch stores as [out, in]. Transpose required.

## Action-Value Framing — Important Subtlety

The 270M model was trained on sequences of length 79: `[FEN_tokens(77), action_token(1), return_bucket(1)]`. The model predicts `P(return_bucket | state, action)`.

For feature extraction, we have two options:

### Option 1: FEN-only input (length 77)
Feed just the FEN tokens, extract hidden states. This is out-of-distribution since the model never saw length-77 inputs during training.

### Option 2: FEN + dummy action + dummy return (length 79)
Feed the full 79-token sequence with dummy values for action (0) and return bucket (0). This is in-distribution.

**Since the model uses bidirectional attention (no causal mask), the action/return tokens influence the FEN representations.** Option 2 is safer — the FEN representations are "as trained." But the dummy action introduces noise (it's a specific move, not "no move").

### Recommendation: Test both with the 9M model first
The 9M model is tiny and fast. Run both options, compare the hidden state quality:
1. Probe for board state accuracy (can you recover piece positions from the hidden states?)
2. Cluster similar positions and verify they're actually similar
3. Compare cosine similarity between the two options

If Option 1 works (FEN-only), prefer it — it's cleaner and faster (no dummy tokens).

**The cloud2 agent should validate this during the conversion workstream.**

## Evaluating the Encoder Representations

After conversion, validate that the representations are actually useful — not just numerically correct.

### Level 1: Board State Probing (5 min)
Train 13 linear classifiers (one per piece type + blank) on the hidden states to predict piece type at each of 64 squares. Following Karvonen's methodology.
- Expected accuracy: >95% (Karvonen got 99.6% on his 50M model; the 270M should be at least as good)
- If accuracy is low, the conversion is wrong or the representation layer is wrong

### Level 2: Strategic Concept Probing (30 min)
Train binary classifiers for: check, fork potential, pin, hanging piece, passed pawn, connected rooks, king safety (open file near king).
- Use python-chess to generate ground truth labels
- Expected accuracy: >80% for most concepts (McGrath got high accuracy on AlphaZero)

### Level 3: Similarity Clustering (10 min)
Embed 1000 diverse positions. Verify:
- Positions from the same opening cluster together
- Positions with similar material balance have closer embeddings
- Endgame positions are far from opening positions
- Visualize with t-SNE or UMAP

### Level 4: Commentary Quality (requires full pipeline)
The real test — does the encoder improve coaching text vs raw FEN input?
- Hold out 50 positions with human-written coaching commentary
- Generate commentary with: (a) just FEN text, (b) FEN + encoder representations
- Score both on the qualitative rubric (correctness, specificity, coaching value, voice, conciseness)
- The encoder should improve correctness and specificity

## Config for 270M Model

```python
TransformerConfig(
    vocab_size=4672,        # NUM_ACTIONS (all possible chess moves)
    output_size=128,        # num_return_buckets (not needed for encoder)
    embedding_dim=1024,
    num_layers=16,
    num_heads=8,
    use_causal_mask=False,  # Full bidirectional attention
    pos_encodings=LEARNED,
    max_sequence_length=79, # SEQUENCE_LENGTH + 2
    widening_factor=4,      # FFN dim = 4096
    apply_post_ln=True,
    apply_qk_layernorm=False,
)
```
