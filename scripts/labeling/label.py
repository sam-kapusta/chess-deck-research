#!/usr/bin/env python3
"""Consolidated SAE feature labeling CLI.

Subcommands:
    profile      — Extract top-N example positions per feature from SAE activations
    label        — Batch-label all features via Bedrock Batch (Sonnet)
    label-single — Label one feature interactively via Bedrock converse()
    label-game   — Label features that fire on positions in a game analysis JSON

Usage:
    # Profile features from a trained SAE + activation cache
    python3 label.py profile --sae checkpoint.pt --cache cache.pt --output profiles.json

    # Batch label via Bedrock
    python3 label.py label --profiles profiles.json --output labels.json

    # Label a single feature interactively
    python3 label.py label-single --profiles profiles.json --feature 42

    # Label features from a game
    python3 label.py label-game --sae checkpoint.pt --game game_analysis.json --output game_labels.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Repo root (for default paths) ──────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]  # chess-deck-research -> chess-coach
CHESS_COACH_ROOT = REPO_ROOT.parent / "chess-coach"
DEFAULT_ENCODER_ONNX = CHESS_COACH_ROOT / "backend" / "lambda" / "sae_features" / "data" / "encoder_270m.onnx"
DEFAULT_MOVE_MAP = CHESS_COACH_ROOT / "backend" / "lambda" / "sae_features" / "data" / "move_to_action.json"

# ── Bedrock config ──────────────────────────────────────────────────────────

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_BATCH_ROLE = "arn:aws:iam::140023406996:role/BedrockBatchInferenceRole"
BEDROCK_S3_BUCKET = "chess-stage-a-140023406996"
BEDROCK_S3_PREFIX = "sae-labeling"

# ── Structured labeling prompt (16 categories) ─────────────────────────────

CATEGORIES = {
    "hanging_pieces": "Undefended or inadequately defended pieces that can be captured",
    "overloaded_defenders": "Pieces defending too many targets simultaneously, creating deflection opportunities",
    "forks": "One piece attacking two or more targets simultaneously",
    "pins": "Piece unable to move without exposing a more valuable piece behind it",
    "skewers": "High-value piece attacked, forced to move, exposing a lower-value piece behind",
    "discovered_attacks": "Moving a piece to reveal an attack from a piece behind it",
    "back_rank": "Back rank mate threats or king trapped on home rank by own pieces",
    "king_safety": "Weak pawn shelter, exposed king, castling vulnerabilities",
    "passed_pawns": "Passed pawn creation, advancement, promotion threats, or blockade",
    "rook_endgames": "Rook endgame technique including R+P vs R, active rook play, 7th rank",
    "pawn_endgames": "King and pawn endgame technique, opposition, king activity",
    "checkmate_patterns": "Mating threats, mating nets, specific checkmate patterns",
    "quiet_moves": "Non-forcing winning moves, prophylaxis, zwischenzug, intermediate moves",
    "trapped_pieces": "Pieces with no escape squares, boxed in by enemy or own pieces",
    "sacrifice": "Giving up material for positional or tactical compensation",
    "other_tactics": "En passant, zugzwang, attraction, interference, stalemate tricks",
}

PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king", "mixed"]
PHASES = ["opening", "middlegame", "endgame", "all_phases"]
SIDES = ["white_playing", "black_playing", "either_side"]


def build_labeling_prompt(feature_id, positive_examples, negative_examples=None):
    """Build structured labeling prompt for one SAE feature."""
    cat_list = "\n".join(f"  - {k}: {v}" for k, v in CATEGORIES.items())

    pos_text = ""
    for i, ex in enumerate(positive_examples[:15]):
        pos_text += f'{i+1}. FEN: {ex["fen"]}\n'
        pos_text += f'   Move played: {ex.get("uci", "?")}'
        if ex.get("best_uci") and ex["best_uci"] != ex.get("uci"):
            pos_text += f'  (best was: {ex["best_uci"]})'
        if ex.get("cp_loss"):
            pos_text += f'  cp_loss={ex["cp_loss"]}'
        pos_text += f'\n   Activation strength: {ex.get("strength", "?")}\n\n'

    neg_text = ""
    if negative_examples:
        neg_text = "\n=== NEGATIVE EXAMPLES (feature does NOT fire on these) ===\n"
        for i, ex in enumerate(negative_examples[:5]):
            neg_text += f'{i+1}. FEN: {ex["fen"]}  Move: {ex.get("uci", "?")}\n'
        neg_text += "\nUse these to understand what the feature is NOT detecting.\n"

    prompt = f"""Analyze SAE feature F{feature_id}. This feature fires on the chess positions below (all are blunder moves — the played move lost evaluation).

=== POSITIVE EXAMPLES (feature fires strongly on these) ===
{pos_text}
{neg_text}
=== YOUR TASK ===

Classify this feature using the structured format below. Be as SPECIFIC as possible.
Do NOT use generic labels like "tactical positions" or "multiple threats."
Instead, identify the SPECIFIC pattern: what piece, what tactic, what context.

CATEGORIES (pick exactly one primary, optionally one secondary):
{cat_list}

Respond in this EXACT JSON format — nothing else:
{{
  "primary_category": "<one of the category keys above>",
  "secondary_category": "<one of the category keys above, or null>",
  "specific_label": "<2-5 words: what SPECIFIC pattern within the category>",
  "piece_involved": "<{'/'.join(PIECES)}>",
  "game_phase": "<{'/'.join(PHASES)}>",
  "side": "<{'/'.join(SIDES)}>",
  "explanation": "<1 sentence: what distinguishes this feature from other features in the same category>",
  "confidence": "<high/medium/low>"
}}

RULES:
- "specific_label" must be 2-5 words describing the SPECIFIC variant, not the category name
  BAD: "overloaded defenders" (that's just the category)
  GOOD: "queen defending rook and back rank"
  GOOD: "knight fork with check"
  GOOD: "hanging bishop after castling"
- If positions involve a specific piece type, name it
- If positions are all in one game phase, say which
- If you can't identify a specific pattern, set confidence to "low"
- secondary_category is for features that span two themes (e.g., a fork that also creates a pin)
"""
    return prompt


# ── BatchTopK SAE ───────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BatchTopKSAE(nn.Module):
        """BatchTopK Sparse Autoencoder."""

        def __init__(self, input_dim, dict_size, k):
            super().__init__()
            self.encoder = nn.Linear(input_dim, dict_size)
            self.decoder = nn.Linear(dict_size, input_dim, bias=False)
            self.pre_bias = nn.Parameter(torch.zeros(input_dim))
            self.k = k
            self.dict_size = dict_size

        def forward(self, x):
            z = self.encoder(x - self.pre_bias)
            tv, ti = torch.topk(z, self.k, dim=-1)
            a = torch.zeros_like(z)
            a.scatter_(-1, ti, F.relu(tv))
            return self.decoder(a) + self.pre_bias, a

        def encode(self, x):
            """Encode only — returns sparse activations."""
            z = self.encoder(x - self.pre_bias)
            tv, ti = torch.topk(z, self.k, dim=-1)
            a = torch.zeros_like(z)
            a.scatter_(-1, ti, F.relu(tv))
            return a


def load_sae(checkpoint_path):
    """Load SAE from a .pt checkpoint. Returns (model, config, mean, std)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for SAE operations. pip install torch")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Config can be top-level or nested
    config = ckpt.get("config", {})
    dict_size = config.get("dict_size", 2048)
    k = config.get("k", 32)
    input_dim = config.get("input_dim", 1024)

    sae = BatchTopKSAE(input_dim, dict_size, k)

    # State dict may be top-level or under model_state_dict
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Filter out non-model keys
    model_keys = {"encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias", "pre_bias"}
    filtered = {k_: v for k_, v in state.items() if k_ in model_keys}
    if filtered:
        sae.load_state_dict(filtered, strict=False)
    else:
        sae.load_state_dict(state, strict=False)

    # Normalization — may be at top level or under normalization.*
    mean = ckpt.get("mean")
    if mean is None:
        norm = ckpt.get("normalization", {}) or {}
        mean = norm.get("mean")
    std = ckpt.get("std")
    if std is None:
        norm = ckpt.get("normalization", {}) or {}
        std = norm.get("std")

    if mean is not None:
        if isinstance(mean, np.ndarray):
            mean = torch.tensor(mean, dtype=torch.float32)
        elif not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
    if std is not None:
        if isinstance(std, np.ndarray):
            std = torch.tensor(std, dtype=torch.float32)
        elif not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)

    return sae.eval(), {"dict_size": dict_size, "k": k, "input_dim": input_dim}, mean, std


# ── FEN tokenizer ───────────────────────────────────────────────────────────

_C = list("0123456789abcdefghpnrkqPBNRQKw.")
_I = {c: i for i, c in enumerate(_C)}
_S = frozenset("12345678")


def tokenize_fen(fen):
    """Tokenize a FEN string to 77 integers for the DeepMind 270M encoder."""
    p = fen.split(" ")
    while len(p) < 6:
        if len(p) == 4:
            p.append("0")
        elif len(p) == 5:
            p.append("1")
        else:
            p.append("-")
    b, s, c, e, h, f = p[:6]
    b = s + b.replace("/", "")
    ix = []
    for ch in b:
        if ch in _S:
            ix.extend(int(ch) * [_I["."]])
        elif ch in _I:
            ix.append(_I[ch])
        else:
            return None
    if c == "-":
        ix.extend(4 * [_I["."]])
    else:
        for ch in c:
            if ch not in _I:
                return None
            ix.append(_I[ch])
        ix.extend((4 - len(c)) * [_I["."]])
    if e == "-":
        ix.extend(2 * [_I["."]])
    else:
        for ch in e:
            if ch not in _I:
                return None
            ix.append(_I[ch])
    h += "." * (3 - len(h))
    ix.extend([_I[x] for x in h[:3]])
    f += "." * (3 - len(f))
    ix.extend([_I[x] for x in f[:3]])
    return ix if len(ix) == 77 else None


# ── ONNX encoder wrapper ───────────────────────────────────────────────────

class OnnxEncoder:
    """Wraps the DeepMind 270M ONNX encoder for CPU inference."""

    def __init__(self, onnx_path, move_map_path):
        os.environ["CPUINFO_DISABLED"] = "1"
        os.environ["OMP_NUM_THREADS"] = "4"
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        with open(move_map_path) as f:
            self.move_map = json.load(f)

    def encode_position(self, fen, move_uci):
        """Encode a (FEN, move) pair. Returns 1024-dim hidden vector or None."""
        tokens = tokenize_fen(fen)
        if tokens is None:
            return None
        if move_uci not in self.move_map:
            return None
        seq = np.array([tokens + [self.move_map[move_uci], 64]], dtype=np.int64)
        output = self.session.run(None, {"input": seq})
        hidden = output[0][0]  # [79, 1024]
        return hidden[77].astype(np.float32)  # move token activation

    def encode_batch(self, positions):
        """Encode a list of (fen, move_uci) pairs. Returns list of 1024-dim vectors."""
        results = []
        for fen, move_uci in positions:
            vec = self.encode_position(fen, move_uci)
            results.append(vec)
        return results


# ── SAE numpy runner (no torch needed) ──────────────────────────────────────

def run_sae_numpy(hidden_vec, encoder_weight, encoder_bias, pre_bias, mean, std, k):
    """Run BatchTopK SAE using numpy. Returns (feature_ids, strengths)."""
    x = (hidden_vec - mean) / np.clip(std, 1e-6, None)
    z = (x - pre_bias) @ encoder_weight.T + encoder_bias
    top_idx = np.argpartition(z, -k)[-k:]
    acts = np.zeros_like(z)
    acts[top_idx] = np.maximum(z[top_idx], 0)
    active = np.where(acts > 0)[0]
    strengths = acts[active]
    order = np.argsort(-strengths)
    return active[order].tolist(), strengths[order].tolist()


# ── Bedrock helpers ─────────────────────────────────────────────────────────

def call_bedrock_converse(prompt, model_id=BEDROCK_MODEL, max_tokens=400):
    """Call Bedrock converse() synchronously. Returns response text."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    return response["output"]["message"]["content"][0]["text"]


def parse_label_response(text):
    """Extract JSON from an LLM response (handles markdown fences)."""
    import re
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SUBCOMMANDS
# ═══════════════════════════════════════════════════════════════════════════


def cmd_profile(args):
    """Profile features: extract top-N example positions per feature from cache."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. pip install torch", file=sys.stderr)
        sys.exit(1)

    print(f"Loading SAE from {args.sae}...")
    sae, config, mean, std = load_sae(args.sae)
    dict_size = config["dict_size"]
    k = config["k"]
    print(f"  dict_size={dict_size}, k={k}")

    print(f"Loading cache from {args.cache}...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    # Cache format: token_acts (N, 77, 1024) or flat (N, 1024), + fens, + mean/std
    acts_tensor = cache["token_acts"]
    fens = cache["fens"]
    cache_mean = cache.get("mean")
    cache_std = cache.get("std")

    # Use cache normalization if SAE checkpoint doesn't have its own
    if mean is None and cache_mean is not None:
        mean = torch.tensor(np.array(cache_mean), dtype=torch.float32)
    if std is None and cache_std is not None:
        std = torch.tensor(np.array(cache_std), dtype=torch.float32)

    # If token_acts is 3D, take the move token (index 77 from 79-length, but
    # cache stores only 77 tokens → last token = index 76)
    if acts_tensor.ndim == 3:
        print(f"  Cache shape: {acts_tensor.shape} — using last token per position")
        flat = acts_tensor[:, -1, :].float()
    else:
        flat = acts_tensor.float()

    n_positions = flat.shape[0]
    print(f"  {n_positions} positions, {flat.shape[1]}-dim activations")

    # Normalize
    if mean is not None and std is not None:
        mean_t = mean.float()
        std_t = std.float().clamp(min=1e-6)
        flat = (flat - mean_t) / std_t

    # Run SAE on all positions in batches
    print(f"Running SAE encoder on {n_positions} positions...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    batch_size = args.batch_size

    # Accumulate per-feature top examples
    # For each feature: track (strength, position_index) tuples
    top_n = args.top_n
    from heapq import heappush, heappushpop

    feature_heaps = {}  # fid -> min-heap of (strength, idx)

    t0 = time.time()
    for i in range(0, n_positions, batch_size):
        batch = flat[i : i + batch_size].to(device)
        with torch.no_grad():
            sparse_acts = sae.encode(batch)  # (B, dict_size)

        # Process each position in batch
        sparse_np = sparse_acts.cpu().numpy()
        for b_idx in range(sparse_np.shape[0]):
            pos_idx = i + b_idx
            row = sparse_np[b_idx]
            active = np.where(row > 0)[0]
            for fid in active:
                strength = float(row[fid])
                heap = feature_heaps.setdefault(int(fid), [])
                entry = (strength, pos_idx)
                if len(heap) < top_n:
                    heappush(heap, entry)
                elif strength > heap[0][0]:
                    heappushpop(heap, entry)

        if (i // batch_size) % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            pct = 100 * i / n_positions
            print(f"  {pct:.0f}% ({i}/{n_positions}, {elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. {len(feature_heaps)} active features.")

    # Build profiles
    profiles = {}
    for fid, heap in sorted(feature_heaps.items()):
        examples = sorted(heap, key=lambda x: -x[0])  # descending by strength
        profiles[str(fid)] = {
            "fire_count": len(examples),
            "fire_rate": len(examples) / n_positions,
            "max_strength": examples[0][0] if examples else 0,
            "examples": [
                {
                    "fen": fens[idx] if idx < len(fens) else f"position_{idx}",
                    "strength": round(s, 4),
                    "position_index": idx,
                }
                for s, idx in examples
            ],
        }

    # Summary
    n_active = len(profiles)
    n_dead = dict_size - n_active
    print(f"\nFeature summary: {n_active} active, {n_dead} dead ({100*n_dead/dict_size:.1f}%)")

    with open(args.output, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved profiles to {args.output}")


def cmd_label(args):
    """Batch-label features via Bedrock Batch."""
    import boto3

    print(f"Loading profiles from {args.profiles}...")
    with open(args.profiles) as f:
        profiles = json.load(f)

    min_examples = args.min_examples
    eligible = {k: v for k, v in profiles.items() if len(v.get("examples", [])) >= min_examples}
    print(f"  {len(eligible)}/{len(profiles)} features have >= {min_examples} examples")

    # Build JSONL records
    records = []
    for fid_str, prof in sorted(eligible.items(), key=lambda x: int(x[0])):
        prompt = build_labeling_prompt(fid_str, prof["examples"])
        record = {
            "recordId": f"label_{fid_str}",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        records.append(record)

    print(f"Built {len(records)} labeling prompts")

    # Write JSONL to temp file, upload to S3
    import tempfile

    ts = time.strftime("%Y%m%d-%H%M%S")
    s3_input_key = f"{BEDROCK_S3_PREFIX}/{ts}/input.jsonl"
    s3_output_key = f"{BEDROCK_S3_PREFIX}/{ts}/output/"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp_path = tmp.name

    s3 = boto3.client("s3", region_name="us-east-1")
    print(f"Uploading {len(records)} records to s3://{BEDROCK_S3_BUCKET}/{s3_input_key}")
    s3.upload_file(tmp_path, BEDROCK_S3_BUCKET, s3_input_key)
    os.unlink(tmp_path)

    # Submit batch job
    bedrock = boto3.client("bedrock", region_name="us-east-1")
    job_name = f"sae-label-{ts}"
    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=args.model or BEDROCK_MODEL,
        roleArn=BEDROCK_BATCH_ROLE,
        inputDataConfig={
            "s3InputDataConfig": {
                "s3Uri": f"s3://{BEDROCK_S3_BUCKET}/{s3_input_key}",
                "s3InputFormat": "JSONL",
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {"s3Uri": f"s3://{BEDROCK_S3_BUCKET}/{s3_output_key}"}
        },
    )

    job_arn = response["jobArn"]
    print(f"\nBatch job submitted: {job_name}")
    print(f"  ARN: {job_arn}")
    print(f"  Input: s3://{BEDROCK_S3_BUCKET}/{s3_input_key}")
    print(f"  Output: s3://{BEDROCK_S3_BUCKET}/{s3_output_key}")
    print(f"\nCheck status:")
    print(f"  python3 label.py label-status --job-arn '{job_arn}'")
    print(f"\nParse results when done:")
    print(f"  python3 label.py label-parse --job-arn '{job_arn}' --output {args.output}")

    # Optionally poll
    if args.wait:
        print("\nWaiting for completion...")
        while True:
            status = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
            state = status["status"]
            print(f"  Status: {state}", flush=True)
            if state in ("Completed", "Failed", "Stopped"):
                break
            time.sleep(30)

        if state == "Completed":
            _parse_batch_results(s3, bedrock, job_arn, args.output, profiles)


def cmd_label_status(args):
    """Check status of a Bedrock Batch job."""
    import boto3
    bedrock = boto3.client("bedrock", region_name="us-east-1")
    status = bedrock.get_model_invocation_job(jobIdentifier=args.job_arn)
    print(f"Job: {status.get('jobName', '?')}")
    print(f"Status: {status['status']}")
    print(f"Model: {status.get('modelId', '?')}")
    if status.get("message"):
        print(f"Message: {status['message']}")
    stats = status.get("statistics", {})
    if stats:
        print(f"Records: {stats.get('inputTokenCount', '?')} input tokens, "
              f"processed={stats.get('processedRecordCount', '?')}, "
              f"failed={stats.get('failedRecordCount', '?')}")


def cmd_label_parse(args):
    """Parse completed Bedrock Batch results into labels.json."""
    import boto3

    bedrock = boto3.client("bedrock", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")

    # Load profiles for metadata
    profiles = {}
    if args.profiles:
        with open(args.profiles) as f:
            profiles = json.load(f)

    _parse_batch_results(s3, bedrock, args.job_arn, args.output, profiles)


def _parse_batch_results(s3, bedrock, job_arn, output_path, profiles):
    """Download + parse Bedrock Batch output into labels.json."""
    import gzip
    import re

    status = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
    if status["status"] != "Completed":
        print(f"Job not completed (status={status['status']}). Cannot parse.")
        return

    output_uri = status["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    # Output is at {output_uri}{input_filename}.jsonl.out
    # List objects under the output prefix
    parsed = output_uri.replace("s3://", "").split("/", 1)
    bucket = parsed[0]
    prefix = parsed[1] if len(parsed) > 1 else ""

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    output_files = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".jsonl.out")]

    if not output_files:
        print(f"No output files found under s3://{bucket}/{prefix}")
        return

    labels = {}
    n_parsed = 0
    n_failed = 0

    for key in output_files:
        print(f"Downloading s3://{bucket}/{key}...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()

        # May be gzipped
        try:
            body = gzip.decompress(body)
        except gzip.BadGzipFile:
            pass

        for line in body.decode("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_failed += 1
                continue

            record_id = record.get("recordId", "")
            fid_match = re.search(r"label_(\d+)", record_id)
            if not fid_match:
                n_failed += 1
                continue
            fid_str = fid_match.group(1)

            # Extract response text
            output = record.get("modelOutput", {})
            content = output.get("content", [])
            text = ""
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
            elif isinstance(content, str):
                text = content

            parsed_label = parse_label_response(text)
            if parsed_label is None:
                n_failed += 1
                continue

            # Merge with profile metadata
            prof = profiles.get(fid_str, {})
            labels[fid_str] = {
                **parsed_label,
                "feature_id": int(fid_str),
                "fire_rate": prof.get("fire_rate", 0),
                "max_strength": prof.get("max_strength", 0),
                "examples": prof.get("examples", [])[:20],
            }
            n_parsed += 1

    print(f"\nParsed {n_parsed} labels, {n_failed} failed")
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved to {output_path}")


def cmd_label_single(args):
    """Label a single feature interactively via Bedrock converse()."""
    with open(args.profiles) as f:
        profiles = json.load(f)

    fid_str = str(args.feature)
    prof = profiles.get(fid_str)
    if not prof:
        print(f"Feature {fid_str} not found in profiles. Available: {len(profiles)} features.")
        sys.exit(1)

    examples = prof.get("examples", [])
    print(f"Feature {fid_str}: {len(examples)} examples, fire_rate={prof.get('fire_rate', 0):.6f}")
    print(f"Top 5 strengths: {[e['strength'] for e in examples[:5]]}")
    print()

    prompt = build_labeling_prompt(fid_str, examples)

    print(f"Calling Bedrock ({args.model or BEDROCK_MODEL})...")
    t0 = time.time()
    response_text = call_bedrock_converse(prompt, model_id=args.model or BEDROCK_MODEL)
    elapsed = time.time() - t0

    print(f"\n--- Raw response ({elapsed:.1f}s) ---")
    print(response_text)

    parsed = parse_label_response(response_text)
    if parsed:
        print(f"\n--- Parsed ---")
        print(json.dumps(parsed, indent=2))
    else:
        print("\nWARNING: Could not parse JSON from response")


def cmd_label_game(args):
    """Label features that fire on positions in a game analysis JSON."""
    print(f"Loading game from {args.game}...")
    with open(args.game) as f:
        game_data = json.load(f)

    # Game is a list of move records with fen_before, classification, etc.
    if isinstance(game_data, dict):
        moves = game_data.get("moves", game_data.get("moments", []))
    else:
        moves = game_data

    # Filter to interesting positions (blunders, mistakes, inaccuracies)
    classifications = set(args.classifications.split(",")) if args.classifications else {"blunder", "mistake", "inaccuracy"}
    interesting = [m for m in moves if m.get("classification") in classifications]
    if not interesting:
        interesting = moves  # fallback: use all
    print(f"  {len(interesting)} interesting positions out of {len(moves)} total")

    # Load encoder
    encoder_path = args.encoder or str(DEFAULT_ENCODER_ONNX)
    move_map_path = args.move_map or str(DEFAULT_MOVE_MAP)

    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX encoder from {encoder_path}...")
    encoder = OnnxEncoder(encoder_path, move_map_path)

    # Load SAE
    print(f"Loading SAE from {args.sae}...")
    sae, config, mean_t, std_t = load_sae(args.sae)
    dict_size = config["dict_size"]
    k = config["k"]
    print(f"  dict_size={dict_size}, k={k}")

    # Convert normalization to numpy for run_sae_numpy
    if mean_t is not None:
        mean_np = mean_t.numpy()
        std_np = std_t.numpy()
    else:
        print("WARNING: No normalization stats in SAE checkpoint. Using zeros/ones.")
        mean_np = np.zeros(config["input_dim"], dtype=np.float32)
        std_np = np.ones(config["input_dim"], dtype=np.float32)

    # Extract SAE weights as numpy
    sae_enc_weight = sae.encoder.weight.detach().numpy()
    sae_enc_bias = sae.encoder.bias.detach().numpy()
    sae_pre_bias = sae.pre_bias.detach().numpy()

    # Encode each position and run SAE
    feature_positions = {}  # fid -> list of {fen, uci, strength, ...}
    n_encoded = 0

    for m in interesting:
        fen = m.get("fen_before", m.get("fen", ""))
        # Try to get the played move in UCI
        move_uci = m.get("move_uci", m.get("uci", ""))
        best_uci = m.get("best_uci", m.get("best_move_uci", ""))

        if not fen or not move_uci:
            continue

        hidden = encoder.encode_position(fen, move_uci)
        if hidden is None:
            continue

        fids, strengths = run_sae_numpy(hidden, sae_enc_weight, sae_enc_bias, sae_pre_bias, mean_np, std_np, k)
        n_encoded += 1

        for fid, strength in zip(fids, strengths):
            entry = {
                "fen": fen,
                "uci": move_uci,
                "best_uci": best_uci,
                "strength": round(strength, 4),
                "cp_loss": m.get("eval_loss", m.get("cp_loss", 0)),
                "classification": m.get("classification", ""),
                "ply": m.get("ply", 0),
                "move": m.get("move", m.get("label", "")),
            }
            feature_positions.setdefault(fid, []).append(entry)

    print(f"  Encoded {n_encoded} positions, found {len(feature_positions)} active features")

    # Sort features by number of positions (most common first)
    sorted_features = sorted(feature_positions.items(), key=lambda x: -len(x[1]))

    # Label each feature
    print(f"\nLabeling {min(len(sorted_features), args.max_features)} features via Bedrock...")
    labels = {}
    for i, (fid, positions) in enumerate(sorted_features[: args.max_features]):
        positions_sorted = sorted(positions, key=lambda x: -x["strength"])
        prompt = build_labeling_prompt(fid, positions_sorted)

        try:
            response_text = call_bedrock_converse(prompt, model_id=args.model or BEDROCK_MODEL)
            parsed = parse_label_response(response_text)
        except Exception as e:
            print(f"  F{fid}: ERROR — {e}")
            parsed = None

        if parsed:
            labels[str(fid)] = {
                **parsed,
                "feature_id": fid,
                "positions_in_game": len(positions),
                "max_strength": positions_sorted[0]["strength"],
                "examples": positions_sorted[:10],
            }
            label_str = parsed.get("specific_label", "?")
            cat = parsed.get("primary_category", "?")
            conf = parsed.get("confidence", "?")
            print(f"  F{fid}: {label_str} ({cat}, {conf}) — {len(positions)} positions")
        else:
            print(f"  F{fid}: FAILED to parse")

        # Brief pause to avoid throttling
        if i < len(sorted_features) - 1:
            time.sleep(0.5)

    # Save
    output = {
        "game_file": str(args.game),
        "sae_checkpoint": str(args.sae),
        "config": config,
        "n_positions_analyzed": n_encoded,
        "n_features_found": len(feature_positions),
        "n_features_labeled": len(labels),
        "labels": labels,
        "feature_summary": [
            {
                "feature_id": fid,
                "count": len(positions),
                "max_strength": max(p["strength"] for p in positions),
                "label": labels.get(str(fid), {}).get("specific_label", "unlabeled"),
            }
            for fid, positions in sorted_features
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(labels)} labels to {args.output}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAE feature labeling CLI — profile, label, and analyze chess SAE features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile features from SAE checkpoint + activation cache
  python3 label.py profile --sae puzzle_2048_k32.pt --cache puzzle_acts_200k.pt -o profiles.json

  # Batch-label all features via Bedrock Batch (Sonnet)
  python3 label.py label --profiles profiles.json -o labels.json

  # Label one feature interactively
  python3 label.py label-single --profiles profiles.json --feature 42

  # Label features from a game analysis
  python3 label.py label-game --sae puzzle_2048_k32.pt --game game_analysis.json -o game_labels.json
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── profile ──
    p_prof = subparsers.add_parser(
        "profile",
        help="Extract top-N example positions per feature from SAE activations",
    )
    p_prof.add_argument("--sae", required=True, help="SAE checkpoint (.pt)")
    p_prof.add_argument("--cache", required=True, help="Activation cache (.pt) with token_acts, fens, mean, std")
    p_prof.add_argument("--output", "-o", required=True, help="Output profiles JSON path")
    p_prof.add_argument("--top-n", type=int, default=20, help="Top N examples per feature (default: 20)")
    p_prof.add_argument("--batch-size", type=int, default=512, help="Batch size for SAE inference (default: 512)")

    # ── label ──
    p_label = subparsers.add_parser(
        "label",
        help="Batch-label all features via Bedrock Batch (Sonnet)",
    )
    p_label.add_argument("--profiles", required=True, help="Profiles JSON from 'profile' step")
    p_label.add_argument("--output", "-o", default="labels.json", help="Output labels JSON path")
    p_label.add_argument("--model", help=f"Bedrock model ID (default: {BEDROCK_MODEL})")
    p_label.add_argument("--min-examples", type=int, default=5, help="Min examples to label a feature (default: 5)")
    p_label.add_argument("--wait", action="store_true", help="Poll until batch job completes")

    # ── label-status ──
    p_status = subparsers.add_parser("label-status", help="Check Bedrock Batch job status")
    p_status.add_argument("--job-arn", required=True, help="Bedrock Batch job ARN")

    # ── label-parse ──
    p_parse = subparsers.add_parser("label-parse", help="Parse completed Bedrock Batch results")
    p_parse.add_argument("--job-arn", required=True, help="Bedrock Batch job ARN")
    p_parse.add_argument("--profiles", help="Profiles JSON (to merge metadata into labels)")
    p_parse.add_argument("--output", "-o", default="labels.json", help="Output labels JSON path")

    # ── label-single ──
    p_single = subparsers.add_parser(
        "label-single",
        help="Label one feature interactively via Bedrock converse()",
    )
    p_single.add_argument("--profiles", required=True, help="Profiles JSON from 'profile' step")
    p_single.add_argument("--feature", type=int, required=True, help="Feature ID to label")
    p_single.add_argument("--model", help=f"Bedrock model ID (default: {BEDROCK_MODEL})")

    # ── label-game ──
    p_game = subparsers.add_parser(
        "label-game",
        help="Label features that fire on positions in a game analysis JSON",
    )
    p_game.add_argument("--sae", required=True, help="SAE checkpoint (.pt)")
    p_game.add_argument("--game", required=True, help="Game analysis JSON (list of moves with fen_before, move_uci, etc.)")
    p_game.add_argument("--output", "-o", required=True, help="Output game labels JSON path")
    p_game.add_argument("--encoder", help=f"ONNX encoder path (default: {DEFAULT_ENCODER_ONNX})")
    p_game.add_argument("--move-map", help=f"move_to_action.json path (default: {DEFAULT_MOVE_MAP})")
    p_game.add_argument("--model", help=f"Bedrock model ID (default: {BEDROCK_MODEL})")
    p_game.add_argument("--max-features", type=int, default=50, help="Max features to label (default: 50)")
    p_game.add_argument(
        "--classifications",
        help="Comma-separated move classifications to include (default: blunder,mistake,inaccuracy)",
    )

    args = parser.parse_args()

    dispatch = {
        "profile": cmd_profile,
        "label": cmd_label,
        "label-status": cmd_label_status,
        "label-parse": cmd_label_parse,
        "label-single": cmd_label_single,
        "label-game": cmd_label_game,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
