#!/usr/bin/env python3
"""Sweep k=4, k=16, k=1 per-token SAEs on puzzle best moves.
For each config, train SAE, then organize features by puzzle theme.
Output: which features map to which themes, grouped by theme."""
import json, math, numpy as np, torch, time, sys, chess
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
PARAMS = BASE + "/cache/deepmind_270m_params.npz"
MOVE_MAP = BASE + "/cache/move_to_action.json"
OUTPUT = BASE + "/output"
PUZZLE_FILE = BASE + "/data/lichess_puzzles_200k.jsonl"

DICT_SIZE = 2048
CONFIGS = [4, 16, 1]
EPOCHS = 3
BATCH_SIZE = 32
N_POSITIONS = 50000
LR = 1e-3

with open(MOVE_MAP) as f: M2A = json.load(f)

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

SQUARE_NAMES = []
for rank in range(8, 0, -1):
    for file_ in "abcdefgh":
        SQUARE_NAMES.append(file_ + str(rank))

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

class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd); self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di)); self.k = k
    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros_like(z); a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a

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
        themes = d.get("themes", "").split()
        positions.append((seq, puzzle_fen, best_move, themes))
        if len(positions) >= N_POSITIONS: break
print("Loaded " + str(len(positions)) + " puzzles")
sys.stdout.flush()

print("Computing normalization stats...")
norm_tokens = []
for i in range(0, min(5000, len(positions)), 64):
    batch = [p[0] for p in positions[i:i+64]]
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

all_themes = Counter()
for p in positions:
    for t in p[3]: all_themes[t] += 1
total_puzzles = len(positions)

print("Pre-computing encoder activations...")
all_token_acts = []
for i in range(0, len(positions), 64):
    batch = [p[0] for p in positions[i:i+64]]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)
        tokens = (h[:, 1:78, :] - mean_t) / std_t
    for b in range(len(batch)):
        all_token_acts.append(tokens[b].cpu())
    if (i // 64) % 100 == 0 and i > 0:
        print("  encoded " + str(i+len(batch)) + "/" + str(len(positions)))
        sys.stdout.flush()
print("All " + str(len(all_token_acts)) + " puzzles encoded.")
del enc
torch.cuda.empty_cache()
sys.stdout.flush()


def train_and_interpret(k_val):
    print()
    print("=" * 70)
    print("TRAINING: dict=" + str(DICT_SIZE) + ", k=" + str(k_val))
    print("=" * 70)
    sys.stdout.flush()

    sae = SAE(1024, DICT_SIZE, k_val).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    final_loss = 0; alive = 0

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
            if n_batches % 300 == 0:
                print("  ep" + str(ep) + " batch " + str(n_batches) + " loss=" + str(round(total_loss/n_batches, 6)))
                sys.stdout.flush()
        with torch.no_grad():
            sample = torch.stack(all_token_acts[:64]).cuda().reshape(-1, 1024)
            _, sa = sae(sample)
            alive = int((sa > 0).any(dim=0).sum().item())
        final_loss = round(total_loss/n_batches, 6)
        print("ep" + str(ep) + " done. loss=" + str(final_loss) + " alive=" + str(alive) + "/" + str(DICT_SIZE))
        sys.stdout.flush()

    ckpt_path = OUTPUT + "/sae_puzzle_pertoken_" + str(DICT_SIZE) + "_k" + str(k_val) + ".pt"
    sae_cpu = sae.cpu().eval()
    torch.save({
        "config": {"dict_size": DICT_SIZE, "k": k_val, "type": "puzzle_pertoken", "n_positions": len(positions), "epochs": EPOCHS},
        "model_state_dict": sae_cpu.state_dict(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
    }, ckpt_path)
    print("Saved: " + ckpt_path)
    sae = sae.cuda()

    print("Interpreting features...")
    feature_puzzles = defaultdict(list)
    for i in range(0, len(positions), 64):
        batch_tokens = torch.stack(all_token_acts[i:i+64]).cuda()
        with torch.no_grad():
            _, acts = sae(batch_tokens.reshape(-1, 1024))
        acts_r = acts.reshape(len(batch_tokens), 77, DICT_SIZE).cpu().numpy()
        for b in range(len(batch_tokens)):
            pidx = i + b
            max_per_f = acts_r[b].max(axis=0)
            argmax_per_f = acts_r[b].argmax(axis=0)
            for fid in range(DICT_SIZE):
                if max_per_f[fid] > 0:
                    feature_puzzles[fid].append((pidx, float(max_per_f[fid]), int(argmax_per_f[fid])))

    feature_themes = {}
    for fid, puzzles in feature_puzzles.items():
        top20 = sorted(puzzles, key=lambda x: -x[1])[:20]
        theme_counts = Counter()
        token_counts = Counter()
        for pidx, act, tok_idx in top20:
            for t in positions[pidx][3]:
                theme_counts[t] += 1
            if 1 <= tok_idx <= 64:
                token_counts[SQUARE_NAMES[tok_idx - 1]] += 1
        enriched = {}
        for theme, count in theme_counts.items():
            base_rate = all_themes[theme] / total_puzzles
            if base_rate > 0 and count >= 3:
                enrichment = (count / 20) / base_rate
                if enrichment > 1.5:
                    enriched[theme] = {"count": count, "enrichment": round(enrichment, 1)}
        feature_themes[fid] = {"enriched": enriched, "freq": len(puzzles), "top_squares": dict(token_counts.most_common(5))}

    theme_features = defaultdict(list)
    for fid, data in feature_themes.items():
        for theme, info in data["enriched"].items():
            theme_features[theme].append((fid, info["enrichment"], info["count"], data["freq"], data["top_squares"]))

    print()
    print("-" * 70)
    print("RESULTS FOR k=" + str(k_val) + ": Features organized by puzzle theme")
    print("-" * 70)

    for theme in sorted(theme_features.keys(), key=lambda t: -len(theme_features[t])):
        features = sorted(theme_features[theme], key=lambda x: -x[1])
        top3 = features[:3]
        feat_strs = []
        for fid, enrich, count, freq, squares in top3:
            sq_str = ",".join(list(squares.keys())[:3]) if squares else "?"
            feat_strs.append("F" + str(fid) + "(" + str(enrich) + "x, " + str(count) + "/20, sq=" + sq_str + ")")
        print("  " + theme + ": " + str(len(features)) + " features. Top: " + "; ".join(feat_strs))

    results = {
        "config": {"dict_size": DICT_SIZE, "k": k_val},
        "theme_features": {},
        "feature_details": {str(fid): data for fid, data in feature_themes.items()},
        "theme_baselines": dict(all_themes.most_common()),
        "alive": alive,
        "final_loss": final_loss,
    }
    for theme, feats in theme_features.items():
        results["theme_features"][theme] = [{"fid": f[0], "enrichment": f[1], "count": f[2], "freq": f[3], "top_squares": f[4]} for f in feats]

    result_path = OUTPUT + "/sae_puzzle_pertoken_" + str(DICT_SIZE) + "_k" + str(k_val) + "_themes.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: " + result_path)
    sys.stdout.flush()

    del sae
    torch.cuda.empty_cache()
    return results


all_results = {}
for k_val in CONFIGS:
    all_results[k_val] = train_and_interpret(k_val)

print()
print("=" * 70)
print("COMPARISON ACROSS k VALUES")
print("=" * 70)
key_themes = ["fork", "pin", "skewer", "mate", "mateIn1", "mateIn2", "sacrifice", "backRankMate", "discoveredAttack", "hangingPiece"]
header = "Theme".ljust(20) + "  ".join(("k=" + str(k)).ljust(14) for k in CONFIGS)
print(header)
print("-" * 70)
for theme in key_themes:
    row = theme.ljust(20)
    for k_val in CONFIGS:
        tf = all_results[k_val].get("theme_features", {}).get(theme, [])
        n_feat = len(tf)
        top_enrich = round(tf[0]["enrichment"], 1) if tf else 0
        cell = str(n_feat) + "f/" + str(top_enrich) + "x"
        row += cell.ljust(14)
    print(row)

print()
for k_val in CONFIGS:
    r = all_results[k_val]
    print("k=" + str(k_val) + ": alive=" + str(r["alive"]) + "/" + str(DICT_SIZE) + ", loss=" + str(r["final_loss"]) + ", themes=" + str(len(r.get("theme_features",{}))))

print()
print("DONE")
