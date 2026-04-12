#!/usr/bin/env python3
"""Debug LoRA generation — isolate why model outputs '!!!'."""
import sys, os, json, torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fen_tokenizer import tokenize as chess_tokenize
from chess_model import ChessEncoder
from projection_layer import ChessProjection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load encoder
print("Loading encoder...", flush=True)
ckpt = torch.load('/tmp/chess_encoder_270m.pt', map_location=device, weights_only=False)
encoder = ChessEncoder(**ckpt['config']).to(device).half()
encoder.load_state_dict(ckpt['model_state_dict'])
encoder.eval()

# Load base LLM + LoRA
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

QWEN = '/home/ec2-user/SageMaker/models/qwen2.5-7b'
LORA = '/home/ec2-user/SageMaker/chess-stage-a/output/lora'
PROJ = '/home/ec2-user/SageMaker/chess-stage-a/output/contrastive/projection.pt'

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
llm = PeftModel.from_pretrained(llm, LORA)
llm.eval()

config = AutoConfig.from_pretrained(QWEN, trust_remote_code=True)
projection = ChessProjection(encoder_dim=1024, llm_dim=config.hidden_size).to(device)
proj_ckpt = torch.load(PROJ, map_location=device, weights_only=False)
projection.load_state_dict(proj_ckpt['state_dict'])
projection.eval()

proj_norm = nn.LayerNorm(config.hidden_size).to(device)
norm_ckpt = torch.load(os.path.join(LORA, 'proj_norm.pt'), map_location=device, weights_only=False)
proj_norm.load_state_dict(norm_ckpt['state_dict'])
proj_norm.eval()

print(f"GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)

# Test FEN
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# === TEST 1: Text-only generation (no chess tokens) ===
print("\n=== TEST 1: Text-only (no chess prefix) ===", flush=True)
prompt_ids = tokenizer("Analyze this position.", return_tensors='pt')['input_ids'].to(device)
with torch.no_grad():
    out = llm.generate(prompt_ids, max_new_tokens=64, do_sample=False)
text1 = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"  Output: {text1[:300]}", flush=True)

# === TEST 2: model.generate() with inputs_embeds ===
print("\n=== TEST 2: model.generate() with inputs_embeds ===", flush=True)
parts = fen.split()
if len(parts) == 4: fen += ' 0 1'
elif len(parts) == 5: fen += ' 1'
fen_tokens = torch.tensor(chess_tokenize(fen).astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    chess_hidden = encoder(fen_tokens)
    chess_proj = proj_norm(projection(chess_hidden.float())).to(torch.bfloat16)

prompt_ids2 = tokenizer("Analyze this position.", return_tensors='pt')['input_ids'].to(device)
prompt_emb = llm.get_input_embeddings()(prompt_ids2)
combined = torch.cat([chess_proj, prompt_emb], dim=1)

print(f"  chess_proj shape: {chess_proj.shape}, norm: {chess_proj.norm():.2f}", flush=True)
print(f"  prompt_emb shape: {prompt_emb.shape}, norm: {prompt_emb.norm():.2f}", flush=True)
print(f"  combined shape: {combined.shape}", flush=True)

# Check norm per token
chess_norms = chess_proj[0].norm(dim=-1)
text_norms = prompt_emb[0].norm(dim=-1)
print(f"  chess per-token norm: mean={chess_norms.mean():.2f}, std={chess_norms.std():.2f}", flush=True)
print(f"  text per-token norm: mean={text_norms.mean():.2f}, std={text_norms.std():.2f}", flush=True)

with torch.no_grad():
    try:
        out2 = llm.generate(inputs_embeds=combined, max_new_tokens=64, do_sample=False)
        text2 = tokenizer.decode(out2[0], skip_special_tokens=True)
        print(f"  Output: {text2[:300]}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

# === TEST 3: Manual generation (same as eval) ===
print("\n=== TEST 3: Manual autoregressive ===", flush=True)
generated_ids = []
with torch.no_grad():
    out3 = llm(inputs_embeds=combined, use_cache=True)
    past = out3.past_key_values
    next_tok = out3.logits[:, -1, :].argmax(dim=-1)
    generated_ids.append(next_tok.item())
    # Get top-5 tokens
    topk = torch.topk(out3.logits[:, -1, :], 5)
    print(f"  First token logits top-5: {[(tokenizer.decode([t]), v.item()) for t, v in zip(topk.indices[0], topk.values[0])]}", flush=True)

    for _ in range(31):
        tok_emb = llm.get_input_embeddings()(next_tok.unsqueeze(0))
        out3 = llm(inputs_embeds=tok_emb, past_key_values=past, use_cache=True)
        past = out3.past_key_values
        next_tok = out3.logits[:, -1, :].argmax(dim=-1)
        generated_ids.append(next_tok.item())
        if next_tok.item() == tokenizer.eos_token_id:
            break

text3 = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"  Output: {text3[:300]}", flush=True)
print(f"  Token IDs: {generated_ids[:20]}", flush=True)

# === TEST 4: Ablation — zero chess tokens ===
print("\n=== TEST 4: Zeroed chess prefix ===", flush=True)
zero_chess = torch.zeros_like(chess_proj)
combined_zero = torch.cat([zero_chess, prompt_emb], dim=1)
with torch.no_grad():
    try:
        out4 = llm.generate(inputs_embeds=combined_zero, max_new_tokens=64, do_sample=False)
        text4 = tokenizer.decode(out4[0], skip_special_tokens=True)
        print(f"  Output: {text4[:300]}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

# === TEST 5: Check what ! token ID is ===
print("\n=== TEST 5: Token IDs ===", flush=True)
bang_id = tokenizer.encode("!", add_special_tokens=False)
print(f"  '!' encodes to: {bang_id}", flush=True)
print(f"  Token 0: {tokenizer.decode([0])}", flush=True)
print(f"  Token 1: {tokenizer.decode([1])}", flush=True)
print(f"  EOS: {tokenizer.eos_token_id}", flush=True)

print("\nDone.", flush=True)
