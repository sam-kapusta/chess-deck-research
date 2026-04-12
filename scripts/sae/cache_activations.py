#!/usr/bin/env python3
"""Cache DeepMind 270M encoder activations for SAE training.

Run once on chess-poc. All future SAE training loads from cache.

Output: chess-stage-a/cache/puzzle_acts_200k.pt
  - token_acts: (N, 77, 1024) float16
  - mean: (1024,) normalization mean
  - std: (1024,) normalization std
  - fens: list of FEN strings

Usage:
    python3 cache_activations.py

Requires: deepmind_270m_params.npz, move_to_action.json, lichess_puzzles_200k.jsonl
"""
import json, math, numpy as np, torch, time, sys, os, chess
import torch.nn as nn, torch.nn.functional as F

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
PARAMS = BASE + "/cache/deepmind_270m_params.npz"
MOVE_MAP = BASE + "/cache/move_to_action.json"
PUZZLE_FILE = BASE + "/data/lichess_puzzles_200k.jsonl"
CACHE_FILE = BASE + "/cache/puzzle_acts_200k.pt"
N_POSITIONS = 200000

if os.path.exists(CACHE_FILE):
    print(f"Cache already exists: {CACHE_FILE}")
    info = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    print(f"  Positions: {len(info['fens'])}, Shape: {info['token_acts'].shape}")
    sys.exit(0)

# ── Tokenizer (matches train_and_profile_all.py) ──
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

# ── Encoder (matches train_and_profile_all.py) ──
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

def load_enc():
    p=dict(np.load(PARAMS,allow_pickle=True))
    m=Enc()
    m.te.weight.data=torch.tensor(p["embed/embeddings"],dtype=torch.float32)
    m.pe.weight.data=torch.tensor(p["embed_1/embeddings"],dtype=torch.float32)
    for i,ly in enumerate(m.layers):
        pre=f"multi_head_dot_product_attention" if i==0 else f"multi_head_dot_product_attention_{i}"
        ln_idx=i*2; ln2_idx=i*2+1
        ly["la"].weight.data=torch.tensor(p[f"layer_norm_{ln_idx}/scale" if ln_idx>0 else "layer_norm/scale"],dtype=torch.float32)
        ly["la"].bias.data=torch.tensor(p[f"layer_norm_{ln_idx}/offset" if ln_idx>0 else "layer_norm/offset"],dtype=torch.float32)
        ly["lm"].weight.data=torch.tensor(p[f"layer_norm_{ln2_idx}/scale"],dtype=torch.float32)
        ly["lm"].bias.data=torch.tensor(p[f"layer_norm_{ln2_idx}/offset"],dtype=torch.float32)
        for n in "qkvo":
            full={"q":"query","k":"key","v":"value","o":"linear"}[n]
            ly[n].weight.data=torch.tensor(p[f"{pre}/{full}/kernel"],dtype=torch.float32).reshape(DIM,DIM).T
        mlp_pre=f"linear" if i==0 else f"linear_{i}"
        ly["g"].weight.data=torch.tensor(p[f"{pre}/mlp/gating_einsum"][0],dtype=torch.float32).T
        ly["u"].weight.data=torch.tensor(p[f"{pre}/mlp/gating_einsum"][1],dtype=torch.float32).T
        ly["d"].weight.data=torch.tensor(p[f"{pre}/mlp/linear/kernel"],dtype=torch.float32).T
    ln_final=NL*2
    ly_key = f"layer_norm_{ln_final}/scale"
    ly_key_b = f"layer_norm_{ln_final}/offset"
    m.fn.weight.data=torch.tensor(p[ly_key],dtype=torch.float32)
    m.fn.bias.data=torch.tensor(p[ly_key_b],dtype=torch.float32)
    return m.cuda().eval()

# ── Main ──
print("Loading encoder...")
enc = load_enc()
with open(MOVE_MAP) as f: M2A = json.load(f)

print(f"Loading {N_POSITIONS} puzzles...")
positions = []
fens = []
with open(PUZZLE_FILE) as f:
    for line in f:
        if len(positions) >= N_POSITIONS: break
        d = json.loads(line)
        t = tok(d["fen"])
        if t and len(t) == 77:
            positions.append(t)
            fens.append(d["fen"])
print(f"Loaded {len(positions)} positions")

# Normalization from first 5K
print("Computing normalization...")
norm = []
for i in range(0, min(5000, len(positions)), 64):
    batch = positions[i:i+64]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)[:, :77, :]
    norm.append(h.cpu().numpy().reshape(-1, 1024))
norm = np.concatenate(norm)
mean = norm.mean(axis=0)
std = norm.std(axis=0) + 1e-8
del norm
mean_t = torch.tensor(mean, dtype=torch.float32, device="cuda")
std_t = torch.tensor(std, dtype=torch.float32, device="cuda")

# Encode all positions
print(f"Encoding {len(positions)} positions...")
t0 = time.time()
all_acts = []
for i in range(0, len(positions), 64):
    batch = positions[i:i+64]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)[:, :77, :]
        normed = (h - mean_t) / std_t
    for b in range(len(batch)):
        all_acts.append(normed[b].half().cpu())  # float16 to save space
    if (i // 64) % 200 == 0 and i > 0:
        print(f"  {i}/{len(positions)} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"Encoding done: {len(all_acts)} positions in {elapsed:.0f}s")

# Save
print("Saving cache...")
token_acts = torch.stack(all_acts)
torch.save({"token_acts": token_acts, "mean": mean, "std": std, "fens": fens}, CACHE_FILE)
size_mb = os.path.getsize(CACHE_FILE) / 1024 / 1024
print(f"Cached: {CACHE_FILE} ({size_mb:.0f} MB, shape {token_acts.shape})")

del enc; torch.cuda.empty_cache()
print("Done. GPU freed.")
