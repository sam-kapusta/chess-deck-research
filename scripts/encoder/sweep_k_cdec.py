#!/usr/bin/env python3
"""Sweep k values for BTK SAE, compute c_dec to find optimal L0.

Based on "Sparse but Wrong" (Chanin & Garriga-Alonso 2025):
  c_dec = mean |cos(d_i, d_j)| across all decoder weight pairs
  The correct L0 minimizes c_dec.

Runs on SAIS notebook (needs GPU for encoder, SAE trains on GPU too).
Reuses encoder activations across all k values.

Usage (on SAIS):
    python3 sweep_k_cdec.py
"""
import json, math, numpy as np, torch, time, sys, chess
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
PARAMS = BASE + "/cache/deepmind_270m_params.npz"
MOVE_MAP = BASE + "/cache/move_to_action.json"
OUTPUT = BASE + "/output/k_sweep"
PUZZLE_FILE = BASE + "/data/lichess_puzzles_200k.jsonl"

DICT_SIZE = 2048
K_VALUES = [8, 16, 32, 64, 128, 256]
EPOCHS = 3
BATCH_SIZE = 32
N_POSITIONS = 50000
LR = 1e-3

import os
os.makedirs(OUTPUT, exist_ok=True)

with open(MOVE_MAP) as f: M2A = json.load(f)

# ── Tokenizer (same as sweep_puzzle_sae.py) ──
_C = list("0123456789abcdefghpnrkqPBNRQKw.")
_I = {c:i for i,c in enumerate(_C)}; _S = frozenset("12345678")
def tok(fen):
    p = fen.split(" ")
    while len(p)<6:
        if len(p)==4: p.append("0")
        elif len(p)==5: p.append("1")
        else: p.append("-")
    b,s,c,e,h,f = p[:6]; b = s+b.replace("/",""); ix = []
    for ch in b:
        if ch in _S: ix.extend(int(ch)*[_I["."]])
        elif ch in _I: ix.append(_I[ch])
        else: return None
    if c=="-": ix.extend(4*[_I["."]])
    else:
        for ch in c:
            if ch not in _I: return None
            ix.append(_I[ch])
        ix.extend((4-len(c))*[_I["."]])
    if e=="-": ix.extend(2*[_I["."]])
    else:
        for ch in e:
            if ch not in _I: return None
            ix.append(_I[ch])
    h+="."*(3-len(h)); ix.extend([_I[x] for x in h[:3]])
    f+="."*(3-len(f)); ix.extend([_I[x] for x in f[:3]])
    return ix if len(ix)==77 else None

# ── Encoder (DeepMind 270M) ──
DIM=1024;NL=16;NH=8;HD=128;FFN=4096;FS=79
class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.te=nn.Embedding(1968,DIM);self.pe=nn.Embedding(FS,DIM);self.layers=nn.ModuleList()
        for _ in range(NL):
            self.layers.append(nn.ModuleDict(dict(la=nn.LayerNorm(DIM),q=nn.Linear(DIM,DIM,bias=False),k=nn.Linear(DIM,DIM,bias=False),v=nn.Linear(DIM,DIM,bias=False),o=nn.Linear(DIM,DIM,bias=False),lm=nn.LayerNorm(DIM),g=nn.Linear(DIM,FFN,bias=False),u=nn.Linear(DIM,FFN,bias=False),d=nn.Linear(FFN,DIM,bias=False))))
        self.fn=nn.LayerNorm(DIM)
    def forward(self,t):
        B,T=t.shape;s=torch.cat([torch.zeros(B,1,dtype=t.dtype,device=t.device),t[:,:-1]],dim=1)
        x=self.te(s)*math.sqrt(DIM)+self.pe(torch.arange(T,device=t.device))
        for l in self.layers:
            xn=l["la"](x);q=l["q"](xn).reshape(B,T,NH,HD);k=l["k"](xn).reshape(B,T,NH,HD);v=l["v"](xn).reshape(B,T,NH,HD)
            a=torch.einsum("bthd,bThd->bhtT",q,k)/math.sqrt(HD);a=F.softmax(a,dim=-1)
            o=torch.einsum("bhtT,bThd->bthd",a,v).reshape(B,T,DIM);x=x+l["o"](o)
            xn=l["lm"](x);x=x+l["d"](F.silu(l["g"](xn))*l["u"](xn))
        return self.fn(x)
def glk(i): return "layer_norm" if i==0 else "layer_norm_"+str(i)
def gak(i): return "multi_head_dot_product_attention" if i==0 else "multi_head_dot_product_attention_"+str(i)
def gmk(i): return "linear" if i==0 else "linear_"+str(i)


# ── SAE ──
class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd); self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di)); self.k = k; self.dd = dd
    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros_like(z); a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a


def compute_cdec(sae):
    """Compute c_dec: mean |cos(d_i, d_j)| across all decoder weight pairs."""
    with torch.no_grad():
        dec_w = sae.decoder.weight.data  # [input_dim, dict_size]
        dec_normed = F.normalize(dec_w, dim=0)  # normalize each feature column
        cos_sim = dec_normed.T @ dec_normed  # [dict_size, dict_size]
        mask = ~torch.eye(sae.dd, dtype=torch.bool, device=cos_sim.device)
        c_dec = cos_sim[mask].abs().mean().item()
        max_cos = cos_sim[mask].abs().max().item()
    return c_dec, max_cos


def compute_structural_metrics(sae, token_acts, n_sample=5000):
    """Compute dead features, L0, FVU on a sample."""
    sample_idx = np.random.choice(len(token_acts), min(n_sample, len(token_acts)), replace=False)
    sample = torch.stack([token_acts[i] for i in sample_idx]).cuda()
    sample_flat = sample.reshape(-1, 1024)

    with torch.no_grad():
        recon, acts = sae(sample_flat)
        mse = F.mse_loss(recon, sample_flat).item()
        var = sample_flat.var().item()
        fvu = mse / var if var > 0 else 1.0

        fire_count = (acts > 0).any(dim=0).sum().item()  # features that fire at all
        dead = sae.dd - fire_count
        l0 = (acts > 0).float().sum(dim=-1).mean().item()

    return {'dead': dead, 'alive': fire_count, 'l0': l0, 'fvu': fvu, 'mse': mse}


# ── Main ──
print("Loading encoder...")
pr=dict(np.load(PARAMS));enc=Enc()
with torch.no_grad():
    enc.te.weight.copy_(torch.tensor(pr["embed/embeddings"]));enc.pe.weight.copy_(torch.tensor(pr["embed_1/embeddings"]))
    for i,l in enumerate(enc.layers):
        la,lm=glk(i*2),glk(i*2+1)
        l["la"].weight.copy_(torch.tensor(pr[la+"/scale"]));l["la"].bias.copy_(torch.tensor(pr[la+"/offset"]))
        l["lm"].weight.copy_(torch.tensor(pr[lm+"/scale"]));l["lm"].bias.copy_(torch.tensor(pr[lm+"/offset"]))
        ak=gak(i);l["q"].weight.copy_(torch.tensor(pr[ak+"/linear/w"]).T);l["k"].weight.copy_(torch.tensor(pr[ak+"/linear_1/w"]).T)
        l["v"].weight.copy_(torch.tensor(pr[ak+"/linear_2/w"]).T);l["o"].weight.copy_(torch.tensor(pr[ak+"/linear_3/w"]).T)
        mb=i*3;l["g"].weight.copy_(torch.tensor(pr[gmk(mb)+"/w"]).T);l["u"].weight.copy_(torch.tensor(pr[gmk(mb+1)+"/w"]).T)
        l["d"].weight.copy_(torch.tensor(pr[gmk(mb+2)+"/w"]).T)
    fl=glk(NL*2);enc.fn.weight.copy_(torch.tensor(pr[fl+"/scale"]));enc.fn.bias.copy_(torch.tensor(pr[fl+"/offset"]))
del pr;enc=enc.cuda().eval()
print("Encoder loaded.")
sys.stdout.flush()

print("Loading puzzles...")
positions = []
with open(PUZZLE_FILE) as f:
    for line in f:
        d = json.loads(line)
        moves = d["moves"].split()
        if len(moves) < 2: continue
        try:
            board = chess.Board(d["fen"])
            board.push_uci(moves[0])
            puzzle_fen = board.fen()
        except: continue
        best_move = moves[1]
        ft = tok(puzzle_fen)
        if ft is None or best_move not in M2A: continue
        seq = ft + [M2A[best_move], 64]
        positions.append(seq)
        if len(positions) >= N_POSITIONS: break
print(f"Loaded {len(positions)} puzzles")
sys.stdout.flush()

# Compute normalization stats
print("Computing normalization stats...")
norm_tokens = []
for i in range(0, min(5000, len(positions)), 64):
    batch = positions[i:i+64]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)
        tokens = h[:, 1:78, :].reshape(-1, 1024).cpu().numpy()
    norm_tokens.append(tokens)
norm_tokens = np.concatenate(norm_tokens, axis=0)
mean = norm_tokens.mean(axis=0)
std = norm_tokens.std(axis=0) + 1e-8
del norm_tokens
mean_t = torch.tensor(mean, dtype=torch.float32, device="cuda")
std_t = torch.tensor(std, dtype=torch.float32, device="cuda")
print("Norm stats computed.")
sys.stdout.flush()

# Pre-compute encoder activations (reused across all k values)
print("Pre-computing encoder activations...")
all_token_acts = []
for i in range(0, len(positions), 64):
    batch = positions[i:i+64]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)
        tokens = (h[:, 1:78, :] - mean_t) / std_t
    for b in range(len(batch)):
        all_token_acts.append(tokens[b].cpu())
    if (i // 64) % 100 == 0 and i > 0:
        print(f"  encoded {i+len(batch)}/{len(positions)}")
        sys.stdout.flush()
print(f"All {len(all_token_acts)} puzzles encoded.")
del enc
torch.cuda.empty_cache()
sys.stdout.flush()

# ── Sweep ──
results = {}

for k_val in K_VALUES:
    print()
    print("=" * 70)
    print(f"TRAINING: dict={DICT_SIZE}, k={k_val}")
    print("=" * 70)
    sys.stdout.flush()
    t0 = time.time()

    sae = SAE(1024, DICT_SIZE, k_val).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=LR)

    for ep in range(EPOCHS):
        perm = np.random.permutation(len(positions))
        total_loss = 0; n_batches = 0
        for i in range(0, len(positions), BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            batch_tokens = torch.stack([all_token_acts[j] for j in batch_idx]).cuda()
            tokens_flat = batch_tokens.reshape(-1, 1024)
            recon, acts = sae(tokens_flat)
            loss = F.mse_loss(recon, tokens_flat)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / n_batches
        print(f"  ep{ep} loss={avg_loss:.6f}")
        sys.stdout.flush()

    elapsed = time.time() - t0

    # Compute c_dec
    c_dec, max_cos = compute_cdec(sae)

    # Compute structural metrics
    metrics = compute_structural_metrics(sae, all_token_acts)

    result = {
        'k': k_val,
        'c_dec': round(c_dec, 6),
        'max_cos': round(max_cos, 4),
        'dead': metrics['dead'],
        'alive': metrics['alive'],
        'l0': round(metrics['l0'], 1),
        'fvu': round(metrics['fvu'], 6),
        'mse': round(metrics['mse'], 6),
        'loss': round(avg_loss, 6),
        'time_s': round(elapsed, 1),
    }
    results[k_val] = result

    print(f"  c_dec={c_dec:.6f}  max_cos={max_cos:.4f}  dead={metrics['dead']}  alive={metrics['alive']}  L0={metrics['l0']:.1f}  FVU={metrics['fvu']:.4f}  ({elapsed:.0f}s)")
    sys.stdout.flush()

    # Save checkpoint
    ckpt_path = OUTPUT + f"/sae_btk_{DICT_SIZE}_k{k_val}.pt"
    torch.save({
        "config": {"dict_size": DICT_SIZE, "k": k_val, "type": "puzzle_pertoken", "n_positions": len(positions), "epochs": EPOCHS},
        "model_state_dict": sae.cpu().state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "metrics": result,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    del sae
    torch.cuda.empty_cache()

# ── Summary ──
print()
print("=" * 70)
print("K-SWEEP RESULTS (c_dec = paper's proxy for optimal L0)")
print("=" * 70)
print(f"{'k':>5}  {'c_dec':>8}  {'max_cos':>8}  {'dead':>5}  {'alive':>5}  {'L0':>6}  {'FVU':>8}  {'loss':>8}")
print("-" * 70)

min_cdec = min(r['c_dec'] for r in results.values())
for k_val in K_VALUES:
    r = results[k_val]
    marker = " ← OPTIMAL" if r['c_dec'] == min_cdec else ""
    print(f"{r['k']:>5}  {r['c_dec']:>8.6f}  {r['max_cos']:>8.4f}  {r['dead']:>5}  {r['alive']:>5}  {r['l0']:>6.1f}  {r['fvu']:>8.4f}  {r['loss']:>8.6f}{marker}")

# Save summary
summary_path = OUTPUT + "/k_sweep_results.json"
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved summary: {summary_path}")
print("\nDONE")
