#!/usr/bin/env python3
"""Profile a trained SAE: find top FEN examples per feature.

Runs on SAIS (needs GPU for encoder). Outputs a profiles JSON with
top 20 FEN examples per feature + stats (fire rate, phase, piece, etc).

Usage (on SAIS):
    python3 profile_sae.py --checkpoint output/k_sweep/sae_btk_2048_k128.pt
    python3 profile_sae.py --checkpoint output/k_sweep/sae_btk_2048_k256.pt
"""
import argparse, json, math, numpy as np, torch, sys, chess, os
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
PARAMS = BASE + "/cache/deepmind_270m_params.npz"
MOVE_MAP = BASE + "/cache/move_to_action.json"
PUZZLE_FILE = BASE + "/data/lichess_puzzles_200k.jsonl"

N_POSITIONS = 50000  # same as training
TOP_N = 20  # examples per feature

with open(MOVE_MAP) as f: M2A = json.load(f)

# ── Tokenizer ──
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

PIECE_MAP = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
             chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

# ── Encoder ──
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


def get_phase(fen):
    board = chess.Board(fen)
    n = len(board.piece_map())
    if n > 24: return "opening"
    if n > 12: return "middlegame"
    return "endgame"


def get_move_info(fen, uci_move):
    """Get piece type, is_capture, is_check from a move."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoder forward pass (reduce for large dicts)")
    args = parser.parse_args()

    if not args.output:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(os.path.dirname(args.checkpoint), base + "_profiles.json")

    # Load encoder
    # Two npz formats exist:
    #   chess-poc: {pre}/linear/w, {pre}/linear_1/w (q,k,v,o), linear/w (mlp gate/up/down)
    #   cache_activations.py format: {pre}/query/kernel, {pre}/mlp/gating_einsum
    # Auto-detect by checking which keys exist.
    print("Loading encoder...")
    pr = dict(np.load(PARAMS, allow_pickle=True))
    enc = Enc()

    # Detect format
    uses_kernel = "multi_head_dot_product_attention/query/kernel" in pr
    print(f"  NPZ format: {'kernel' if uses_kernel else 'linear/w'}")

    def glk(i):
        return "layer_norm" if i == 0 else f"layer_norm_{i}"

    def gak(i):
        return "multi_head_dot_product_attention" if i == 0 else f"multi_head_dot_product_attention_{i}"

    def gmk(i):
        return "linear" if i == 0 else f"linear_{i}"

    enc.te.weight.data = torch.tensor(pr["embed/embeddings"], dtype=torch.float32)
    enc.pe.weight.data = torch.tensor(pr["embed_1/embeddings"], dtype=torch.float32)

    for i, ly in enumerate(enc.layers):
        la, lm = glk(i * 2), glk(i * 2 + 1)
        ly["la"].weight.data = torch.tensor(pr[la + "/scale"], dtype=torch.float32)
        ly["la"].bias.data = torch.tensor(pr[la + "/offset"], dtype=torch.float32)
        ly["lm"].weight.data = torch.tensor(pr[lm + "/scale"], dtype=torch.float32)
        ly["lm"].bias.data = torch.tensor(pr[lm + "/offset"], dtype=torch.float32)

        ak = gak(i)
        if uses_kernel:
            for n in "qkvo":
                full = {"q": "query", "k": "key", "v": "value", "o": "linear"}[n]
                ly[n].weight.data = torch.tensor(pr[f"{ak}/{full}/kernel"], dtype=torch.float32).reshape(DIM, DIM).T
            ly["g"].weight.data = torch.tensor(pr[f"{ak}/mlp/gating_einsum"][0], dtype=torch.float32).T
            ly["u"].weight.data = torch.tensor(pr[f"{ak}/mlp/gating_einsum"][1], dtype=torch.float32).T
            ly["d"].weight.data = torch.tensor(pr[f"{ak}/mlp/linear/kernel"], dtype=torch.float32).T
        else:
            ly["q"].weight.data = torch.tensor(pr[ak + "/linear/w"], dtype=torch.float32).T
            ly["k"].weight.data = torch.tensor(pr[ak + "/linear_1/w"], dtype=torch.float32).T
            ly["v"].weight.data = torch.tensor(pr[ak + "/linear_2/w"], dtype=torch.float32).T
            ly["o"].weight.data = torch.tensor(pr[ak + "/linear_3/w"], dtype=torch.float32).T
            mb = i * 3
            ly["g"].weight.data = torch.tensor(pr[gmk(mb) + "/w"], dtype=torch.float32).T
            ly["u"].weight.data = torch.tensor(pr[gmk(mb + 1) + "/w"], dtype=torch.float32).T
            ly["d"].weight.data = torch.tensor(pr[gmk(mb + 2) + "/w"], dtype=torch.float32).T

    fl = glk(NL * 2)
    enc.fn.weight.data = torch.tensor(pr[fl + "/scale"], dtype=torch.float32)
    enc.fn.bias.data = torch.tensor(pr[fl + "/offset"], dtype=torch.float32)
    del pr
    enc = enc.cuda().eval()
    print("Encoder loaded.")

    # Load SAE
    print(f"Loading SAE from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Handle both old format (config/model_state_dict) and new format (flat keys)
    if "config" in ckpt:
        cfg = ckpt["config"]
        dict_size = cfg["dict_size"]
        k = cfg["k"]
        norm = ckpt["normalization"]
        mean_t = torch.tensor(norm["mean"], dtype=torch.float32, device="cuda")
        std_t = torch.tensor(norm["std"], dtype=torch.float32, device="cuda")
        sae = SAE(1024, dict_size, k).cuda()
        sae.load_state_dict(ckpt["model_state_dict"])
    else:
        dict_size = ckpt["dict_size"]
        k = ckpt["k"]
        mean_t = torch.tensor(ckpt["mean"], dtype=torch.float32, device="cuda")
        std_t = torch.tensor(ckpt["std"], dtype=torch.float32, device="cuda")
        sae = SAE(1024, dict_size, k)
        sae.encoder.weight.data = ckpt["encoder_weight"]
        sae.encoder.bias.data = ckpt["encoder_bias"]
        sae.decoder.weight.data = ckpt["decoder_weight"]
        sae.pre_bias.data = ckpt["pre_bias"]
        sae = sae.cuda()
    sae.eval()
    print(f"SAE loaded: dict={dict_size}, k={k}")

    # Load puzzles
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
            positions.append({"seq": seq, "fen": puzzle_fen, "move": best_move, "themes": themes})
            if len(positions) >= N_POSITIONS: break
    print(f"Loaded {len(positions)} puzzles")

    # Profile: for each feature, find top activating positions
    print("Profiling features...")
    # Track per-feature: list of (activation_strength, position_index)
    feature_top = defaultdict(list)  # fid -> [(strength, pidx), ...]
    feature_fire_count = Counter()  # fid -> total positions where it fires
    total_positions = 0

    bs = args.batch_size
    for i in range(0, len(positions), bs):
        batch = [p["seq"] for p in positions[i:i+bs]]
        tens = torch.tensor(batch, dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = enc(tens)
            tokens_norm = (h[:, 1:78, :] - mean_t) / std_t
            tokens_flat = tokens_norm.reshape(-1, 1024)
            _, acts = sae(tokens_flat)
            # Reshape back: (batch, 77, dict_size)
            acts_r = acts.reshape(len(batch), 77, dict_size)
            # Max activation per feature across tokens
            max_per_feature = acts_r.max(dim=1).values  # (batch, dict_size)

        for b in range(len(batch)):
            pidx = i + b
            total_positions += 1
            for fid in range(dict_size):
                strength = max_per_feature[b, fid].item()
                if strength > 0:
                    feature_fire_count[fid] += 1
                    # Keep top N
                    if len(feature_top[fid]) < TOP_N:
                        feature_top[fid].append((strength, pidx))
                    elif strength > feature_top[fid][-1][0]:
                        feature_top[fid][-1] = (strength, pidx)
                    feature_top[fid].sort(key=lambda x: -x[0])

        if (i // bs) % 100 == 0 and i > 0:
            print(f"  {i+len(batch)}/{len(positions)}")
            sys.stdout.flush()

    # Build profiles
    print("Building profiles...")
    profiles = {}
    for fid in range(dict_size):
        fire_rate = round(feature_fire_count.get(fid, 0) / total_positions * 100, 2)
        if fire_rate == 0:
            continue  # skip dead features

        examples = []
        phase_counts = Counter()
        piece_counts = Counter()
        capture_count = 0
        check_count = 0

        for strength, pidx in feature_top[fid][:TOP_N]:
            pos = positions[pidx]
            fen = pos["fen"]
            move = pos["move"]
            phase = get_phase(fen)
            piece, is_cap, is_chk = get_move_info(fen, move)

            phase_counts[phase] += 1
            piece_counts[piece] += 1
            if is_cap: capture_count += 1
            if is_chk: check_count += 1

            cap_str = "x" if is_cap else ""
            chk_str = "+" if is_chk else ""
            example_str = f"{fen} | {move} (best, {piece}{cap_str}{chk_str}, {phase})"
            examples.append(example_str)

        n_ex = len(examples)
        profiles[str(fid)] = {
            "examples": examples,
            "fire_rate": fire_rate,
            "n_fires": feature_fire_count.get(fid, 0),
            "phase_opening": round(phase_counts.get("opening", 0) / n_ex * 100, 1) if n_ex else 0,
            "phase_middlegame": round(phase_counts.get("middlegame", 0) / n_ex * 100, 1) if n_ex else 0,
            "phase_endgame": round(phase_counts.get("endgame", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_pawn": round(piece_counts.get("pawn", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_knight": round(piece_counts.get("knight", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_bishop": round(piece_counts.get("bishop", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_rook": round(piece_counts.get("rook", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_queen": round(piece_counts.get("queen", 0) / n_ex * 100, 1) if n_ex else 0,
            "piece_king": round(piece_counts.get("king", 0) / n_ex * 100, 1) if n_ex else 0,
            "captures": round(capture_count / n_ex * 100, 1) if n_ex else 0,
            "checks": round(check_count / n_ex * 100, 1) if n_ex else 0,
        }

    with open(args.output, "w") as f:
        json.dump(profiles, f, indent=2)

    alive = len(profiles)
    dead = dict_size - alive
    fire_rates = [p["fire_rate"] for p in profiles.values()]
    print(f"\nDone. {alive} alive features profiled, {dead} dead.")
    print(f"Fire rate: mean={sum(fire_rates)/len(fire_rates):.2f}%, median={sorted(fire_rates)[len(fire_rates)//2]:.2f}%")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
