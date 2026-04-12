#!/usr/bin/env python3
"""Profile k=16 and k=64 SAEs (already trained from sweep).
Run on SAIS chess-poc."""
import json, math, numpy as np, torch, sys, chess, os
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
PARAMS = BASE + "/cache/deepmind_270m_params.npz"
MOVE_MAP = BASE + "/cache/move_to_action.json"
OUTPUT = BASE + "/output/k_sweep"
PUZZLE_FILE = BASE + "/data/lichess_puzzles_200k.jsonl"
N_POSITIONS = 50000

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

PIECE_MAP = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
             chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

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

def get_phase(fen):
    board = chess.Board(fen)
    n = len(board.piece_map())
    if n > 24: return "opening"
    if n > 12: return "middlegame"
    return "endgame"

def get_move_info(fen, uci_move):
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        piece = board.piece_at(move.from_square)
        piece_name = PIECE_MAP.get(piece.piece_type, "?") if piece else "?"
        is_capture = board.is_capture(move)
        board.push(move)
        is_check = board.is_check()
        return piece_name, is_capture, is_check
    except:
        return "?", False, False

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
        positions.append({"seq": ft + [M2A[best_move], 64], "fen": puzzle_fen, "move": best_move})
        if len(positions) >= N_POSITIONS: break
print(f"Loaded {len(positions)} puzzles")

print("Computing norm stats...")
norm_tokens = []
for i in range(0, min(5000, len(positions)), 64):
    batch = [p["seq"] for p in positions[i:i+64]]
    tens = torch.tensor(batch, dtype=torch.long, device="cuda")
    with torch.no_grad():
        h = enc(tens)
        tokens = h[:, 1:78, :].reshape(-1, 1024).cpu().numpy()
    norm_tokens.append(tokens)
norm_tokens = np.concatenate(norm_tokens, axis=0)
mean_t = torch.tensor(norm_tokens.mean(axis=0), dtype=torch.float32, device="cuda")
std_t = torch.tensor(norm_tokens.std(axis=0) + 1e-8, dtype=torch.float32, device="cuda")
del norm_tokens

print("Pre-computing encoder activations...")
all_token_acts = []
for i in range(0, len(positions), 64):
    batch = [p["seq"] for p in positions[i:i+64]]
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
del enc; torch.cuda.empty_cache()

TOP_N = 20
for k_val in [16, 64]:
    ckpt_path = OUTPUT + f"/sae_btk_2048_k{k_val}.pt"
    print(f"\n--- Profiling dict=2048 k={k_val} ---")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sae = SAE(1024, 2048, k_val).cuda()
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    feature_top = defaultdict(list)
    feature_fire_count = Counter()
    total_positions = 0

    for i in range(0, len(positions), 64):
        batch_tokens = torch.stack(all_token_acts[i:i+64]).cuda()
        with torch.no_grad():
            tokens_flat = batch_tokens.reshape(-1, 1024)
            _, acts = sae(tokens_flat)
            acts_r = acts.reshape(len(batch_tokens), 77, 2048)
            max_per_feature = acts_r.max(dim=1).values
        for b in range(min(64, len(positions) - i)):
            pidx = i + b
            total_positions += 1
            active = (max_per_feature[b] > 0).nonzero(as_tuple=True)[0]
            for fid_t in active:
                fid = fid_t.item()
                strength = max_per_feature[b, fid].item()
                feature_fire_count[fid] += 1
                if len(feature_top[fid]) < TOP_N:
                    feature_top[fid].append((strength, pidx))
                elif strength > feature_top[fid][-1][0]:
                    feature_top[fid][-1] = (strength, pidx)
                feature_top[fid].sort(key=lambda x: -x[0])
        if (i // 64) % 200 == 0 and i > 0:
            print(f"    profiled {i}/{len(positions)}")
            sys.stdout.flush()

    profiles = {}
    for fid in range(2048):
        fire_rate = round(feature_fire_count.get(fid, 0) / total_positions * 100, 2)
        if fire_rate == 0: continue
        examples = []
        phase_counts = Counter(); piece_counts = Counter(); cap_count = chk_count = 0
        for strength, pidx in feature_top[fid][:TOP_N]:
            pos = positions[pidx]
            phase = get_phase(pos["fen"])
            piece, is_cap, is_chk = get_move_info(pos["fen"], pos["move"])
            phase_counts[phase] += 1; piece_counts[piece] += 1
            if is_cap: cap_count += 1
            if is_chk: chk_count += 1
            cap_str = "x" if is_cap else ""; chk_str = "+" if is_chk else ""
            examples.append(f'{pos["fen"]} | {pos["move"]} (best, {piece}{cap_str}{chk_str}, {phase})')
        n = len(examples)
        profiles[str(fid)] = {
            "examples": examples, "fire_rate": fire_rate, "n_fires": feature_fire_count.get(fid, 0),
            "phase_opening": round(phase_counts.get("opening", 0)/n*100, 1) if n else 0,
            "phase_middlegame": round(phase_counts.get("middlegame", 0)/n*100, 1) if n else 0,
            "phase_endgame": round(phase_counts.get("endgame", 0)/n*100, 1) if n else 0,
            "piece_pawn": round(piece_counts.get("pawn", 0)/n*100, 1) if n else 0,
            "piece_knight": round(piece_counts.get("knight", 0)/n*100, 1) if n else 0,
            "piece_bishop": round(piece_counts.get("bishop", 0)/n*100, 1) if n else 0,
            "piece_rook": round(piece_counts.get("rook", 0)/n*100, 1) if n else 0,
            "piece_queen": round(piece_counts.get("queen", 0)/n*100, 1) if n else 0,
            "piece_king": round(piece_counts.get("king", 0)/n*100, 1) if n else 0,
            "captures": round(cap_count/n*100, 1) if n else 0,
            "checks": round(chk_count/n*100, 1) if n else 0,
        }

    profile_path = OUTPUT + f"/profiles_btk_2048_k{k_val}.json"
    with open(profile_path, "w") as f:
        json.dump(profiles, f, indent=2)

    fire_rates = [p["fire_rate"] for p in profiles.values()]
    print(f"  {len(profiles)} alive features profiled")
    print(f"  Fire rate: mean={sum(fire_rates)/len(fire_rates):.2f}%, median={sorted(fire_rates)[len(fire_rates)//2]:.2f}%")
    print(f"  Saved: {profile_path}")

    # Upload to S3
    os.system(f"aws s3 cp {profile_path} s3://chess-stage-a-140023406996/detection-scoring/profiles_btk_2048_k{k_val}.json")
    del sae; torch.cuda.empty_cache()

print("\nDONE")
