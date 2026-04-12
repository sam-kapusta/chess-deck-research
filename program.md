---
name: chess-research
description: Convert DeepMind 270M chess model to PyTorch and build Lichess annotations dataset. Two parallel workstreams.
model: opus
effort: high
memory: project
initialPrompt: >
  Begin. Read research/docs/ for context.
  Then execute the two workstreams below. Alternate between them when one is blocked.
---

# Chess Research: Encoder Conversion + Dataset Building

Two objectives tonight. Both are prerequisites for fine-tuning a chess coaching model.

## Your Tools

**SageMaker MCP** (`sm_*` tools) — execute on SAIS GPU instance (A10G, 24GB VRAM, 32 vCPU, 64GB RAM):
- `sm_terminal_execute(command, timeout, cwd, background)` — run shell commands. Use `background=True` for long-running processes.
- `sm_execute(code)` — run Python on Jupyter kernel
- `sm_write_file(path, content)` — write files to SAIS
- `sm_read_file(path)` — read files from SAIS
- `sm_list_files(path)` — list directory contents
- `sm_kernel_restart()` — restart kernel (OOM recovery)

**Web Research** — `WebSearch`, `WebFetch` for docs, repos, examples

## Workspace

```
/home/ec2-user/SageMaker/chess-research/
├── encoder/          # Workstream 1: DeepMind model conversion
│   ├── jax_weights/  # Downloaded JAX model files
│   ├── pytorch/      # Converted PyTorch model
│   └── scripts/      # Conversion + validation scripts
├── dataset/          # Workstream 2: Lichess annotations
│   ├── raw/          # Raw PGN downloads
│   ├── processed/    # Cleaned (FEN, move, commentary) triples
│   └── scripts/      # Scraping + processing scripts
└── logs/             # Experiment logs
```

---

## Workstream 1: Convert DeepMind 270M Chess Model (JAX → PyTorch)

### Goal
Get DeepMind's searchless chess transformer (270M params, 2895 Elo) running in PyTorch as a feature extractor. We need the last hidden layer representations — NOT the move/value prediction heads.

### Source
- **Repo:** github.com/google-deepmind/searchless_chess (Apache 2.0)
- **Paper:** arXiv 2402.04494
- **Architecture:** Decoder-only transformer, 270M params
- **Framework:** JAX
- **Model weights:** Available via the repo (`.bag` format or similar)
- **Board encoding:** Custom chess tokenizer

### Steps

1. **Clone the repo and understand the architecture.**
   ```bash
   cd /home/ec2-user/SageMaker/chess-research/encoder
   git clone https://github.com/google-deepmind/searchless_chess.git
   ```
   Read the model definition. Identify:
   - Number of layers, hidden dim, attention heads
   - Input tokenization (how FEN/board state is encoded)
   - Where the last hidden layer representations live (before policy/value heads)
   - Weight file format and how to load them in JAX

2. **Install JAX dependencies and load the model.**
   ```bash
   pip install jax jaxlib flax
   ```
   Load the pretrained 270M weights. Verify by running inference on a test position and comparing to expected output (paper reports 2895 Elo behavior).

3. **Extract the architecture into PyTorch.**
   Two approaches (try the easier one first):

   **Approach A: Manual conversion**
   - Map JAX model classes to PyTorch nn.Module equivalents
   - Copy weight tensors: `jax_param → torch.tensor(np.array(jax_param))`
   - Handle any naming/shape differences (JAX uses NHWC, PyTorch uses NCHW if applicable)
   - This is the standard approach. There are examples of JAX→PyTorch conversion for similar transformer models.

   **Approach B: Use a conversion library**
   - Check if `transformers` has a compatible architecture
   - Check if `flax2pytorch` or similar tools handle this model class
   - Some DeepMind models have community-maintained PyTorch ports — search HuggingFace

4. **Validate the PyTorch model.**
   For 10 test positions:
   - Run JAX model → get last hidden layer activations
   - Run PyTorch model → get last hidden layer activations
   - Compare: cosine similarity should be > 0.99 for each position
   - Also compare final move predictions — should be identical top-1

5. **Build a feature extractor wrapper.**
   ```python
   class ChessEncoder:
       """Extracts position representations from the DeepMind 270M model."""

       def __init__(self, model_path):
           ...

       def encode(self, fen: str) -> torch.Tensor:
           """FEN string → hidden representation vector."""
           ...

       def encode_batch(self, fens: list[str]) -> torch.Tensor:
           """Batch of FEN strings → batch of hidden vectors."""
           ...
   ```
   This is what the projection layer will connect to Qwen later.

6. **Save and document.**
   - Save PyTorch weights to `encoder/pytorch/deepmind_270m.pt`
   - Save the encoder class to `encoder/pytorch/chess_encoder.py`
   - Document: hidden dim size, input format, output format, any gotchas

### What Success Looks Like
- PyTorch model loads from saved weights
- `ChessEncoder.encode(fen)` returns a tensor
- Cosine similarity > 0.99 with JAX model on 10 test positions
- Runs on A10G GPU, inference < 100ms per position

### If You Get Stuck
- The model architecture is in the searchless_chess repo — read every file
- JAX weight loading can be tricky — look for `flax.serialization` or `orbax` patterns
- If weight format is `.bag`, look for the repo's data reader utilities
- Search GitHub for "searchless chess pytorch" — someone may have already converted it
- If JAX is fundamentally incompatible with the SAIS environment, try running JAX CPU-only for weight extraction, then switch to PyTorch for everything else

---

## Workstream 2: Lichess Studies Dataset

### Goal
Build a high-quality dataset of (FEN, move, coaching commentary) triples from Lichess studies. These are community-annotated chess games — humans explaining chess moves in natural language.

### Source
- **Lichess studies database:** database.lichess.org
- **License:** CC0 (public domain). Free for any use.
- **Format:** PGN with inline comments in `{curly braces}`
- **Volume:** Millions of studies exist. We want the best ones.

### Steps

1. **Download the Lichess studies database.**
   Check database.lichess.org for a studies export. It may be:
   - A bulk download (compressed PGN)
   - An API: `lichess.org/api/study/{studyId}/pgn`
   - Search for `lichess study pgn database download`

   If no bulk download exists, use the Lichess API to crawl popular studies:
   ```
   GET https://lichess.org/api/study/search?q=analysis&sort=popular
   ```

2. **Parse PGN with inline comments.**
   Use `python-chess` to parse PGN. Extract:
   - FEN before each move
   - The move (SAN)
   - Any inline comment `{text}` associated with the move
   - Study metadata: author, title, likes/views if available

3. **Filter for quality.**
   Keep entries where:
   - Comment is in English (basic language detection)
   - Comment is > 20 characters (skip "!" or "good move")
   - Comment contains chess-relevant content (mentions pieces, squares, concepts)
   - Study has > 5 likes/upvotes (if metadata available)
   - Discard pure notation comments (just move sequences with no explanation)

4. **Enrich with Stockfish eval.**
   For each (FEN, move) pair, optionally add:
   - Stockfish eval before the move
   - Best move according to Stockfish
   - Whether the annotated move matches Stockfish's recommendation
   This connects the human commentary to engine evaluation — the same data format the coaching model needs.

   Note: Stockfish eval is optional for the dataset build. Can be added later during training data preparation. If Stockfish is available on SAIS, run it. If not, skip and move on.

5. **Format and save.**
   Output format (JSONL):
   ```json
   {
     "fen": "rnbqkb1r/...",
     "move": "Nf3",
     "comment": "Developing the knight to a natural square, controlling the center...",
     "study_id": "abc123",
     "author": "username",
     "eval_before": 0.3,
     "best_move": "Nf3",
     "is_best": true
   }
   ```
   Save to `dataset/processed/lichess_studies.jsonl`

6. **Stats and quality report.**
   Report:
   - Total (FEN, move, comment) triples extracted
   - Average comment length
   - Top 20 most common opening words (filter for quality)
   - Distribution of comment lengths
   - Sample 10 random entries for manual inspection

### What Success Looks Like
- > 50K (FEN, move, comment) triples with meaningful English commentary
- Average comment length > 50 characters
- Diverse positions (not all openings, not all endgames)
- Clean JSONL format, loadable by HuggingFace datasets

### If You Get Stuck
- Lichess API docs: lichess.org/api
- If no bulk download: crawl the most popular 10K studies via API
- If API rate-limited: use the PGN bulk exports from database.lichess.org (games have fewer comments than studies, but volume is massive)
- python-chess PGN parsing: `chess.pgn.read_game(handle)` then iterate `node.comment` for each move

---

## Alternating Strategy

These workstreams have natural wait points:
- **Encoder:** downloading weights, installing JAX, running conversion — lots of waiting
- **Dataset:** downloading PGN files, parsing millions of games — lots of waiting

When one is blocked (downloading, processing), switch to the other. The goal is to have BOTH done by morning.

## Logging

Write progress to `research/research_plan.md` (survives context compression):
- What's been accomplished
- Current blockers
- What's remaining
- Any surprises or interesting findings

Also log structured results to `research/experiment_log.jsonl`.

## If Both Finish Early

If you complete both workstreams and have time left:
1. Run a quick test: feed 10 positions through the PyTorch encoder, extract representations, verify they're meaningful (not random noise — cluster similar positions together)
2. Sample 20 Lichess commentary entries, score them on the qualitative rubric (correctness, specificity, coaching value, voice, conciseness)
3. Start exploring what a projection layer from encoder hidden dim → Qwen embedding dim would look like
4. Search for any existing JAX→PyTorch conversions of DeepMind chess models on GitHub/HuggingFace

## Never Stop

Keep working until both are done or you hit an unresolvable blocker. Log everything.
