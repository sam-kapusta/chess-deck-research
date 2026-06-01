# v2 SAE rebuild pipeline (2026-05-31)

Rebuilds the 4 "what-was-missed" SAE constructions on **v2 corrected data** + **Maia-best@2600**
moves. Supersedes `../new_sae_architecture/` (which was built on v1 cache — label-inversion bug).

## Why

The 3 diff constructions need a *best move* per position. The v2 cache
(`maia3_blunder_diff_v2.pt`) dropped `best_uci`; the only 200k best_uci (v1 metadata) is
bug-affected. **Decision: best move = Maia3 policy argmax @ elo 2600**, not Stockfish — the SAE
reads Maia's activations, so Maia's own human-best is the internally-consistent, coaching-relevant
target. (~50% agreement with Stockfish on the 19k overlap; the divergence is the point.)

## Run order (chess-poc notebook, `~/SageMaker/`)

1. `build_maia_best.py` → `maia_best_200k.json` — Maia3 policy best move for all 200k. ~30min CPU.
   - **Critical:** uses maia3 pkg `get_all_possible_moves()` (4352 vocab), `tokenize_board`
     (mirrors black), `get_legal_moves_mask`, `mirror_move`. Do NOT use `move_to_action.json`
     (1968-move DeepMind vocab → garbage, 3% SF agreement).
2. `run_all_v2.sh` chains: build option_a + board_diff caches (ONNX) → l2l7 cache (79M PyTorch,
   GPU) → slice l7only (L7 half of l2l7) → train 4 SAEs @ k=16 (`train_sae_v2.py`) → eval
   (`eval_v2_html.py`).
3. `wait_and_run.sh` polls for the maia_best DONE marker then runs step 2 — for unattended runs.

## Constructions (all on v2 + maia_best@2600, forward conditioned at player's real elo)

| name | construction | dim | encoder |
|------|--------------|-----|---------|
| option_a | `h[best_to]-h[blunder_to]` before-board | 512 | ONNX L7 |
| board_diff | `mean64(h_after_best - h_after_blunder)` | 512 | ONNX L7 |
| l2l7 | `concat(L2_mean64_diff, L7_mean64_diff)` | 2048 | 79M PyTorch |
| l7only | L7 half of l2l7 | 1024 | (sliced) |

Degenerate positions (maia_best == blunder, ~14%) are dropped.

## Outputs

- Caches: `chess-stage-a/cache/maia3_{option_a_diff,board_diff,l2l7_concat,l7only}_v2.pt`
- Weights: `chess-stage-a/output/maia3_sae/maia3_{name}_v2_2048_k16.pt`
- Eval: `eval_v2.html` (chess.com FEN links + top profile examples per fired feature), `eval_v2_results.json`

Per `S3_INVENTORY.md`: upload weights to `s3://.../sae/weights/`, add inventory lines, before shipping.
