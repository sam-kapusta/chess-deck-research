#!/usr/bin/env python3
"""Encode a game's blunders through Maia3 -> L7-diff -> a chosen SAE, label fired features.

Generalizes encode_game_blunders.py to ANY (dict,k) SAE + ANY label file format. Runs on chess-poc.
Replicates the corpus encoding (build_l2l7_v2.py): Maia3 79M, hook layers[6] (L7),
diff = mean64(L7_after_best - L7_after_blunder), z-score with the CORPUS mean/std
(maia3_l7only_v2_dedup.pt), then SAE encode (top-k by activation).

Env vars:
  SAE_WEIGHTS  (path to .pt with state_dict W_enc/b_enc/b_dec)   default d2048_k6
  SAE_K        (top-k at inference)                              default 6
  LABELS_FILE  (json: {fid: {chip, ...}} — d64_k1 v9 OR d2048_k6 v3)
  LEAF_FILE    (optional json: {fid: {bucket_name, sub}} — d2048_k6 only)
  OUT          (output json path)                               default /tmp/blunder_features.json
  TAG          (label for this run, e.g. 'd64_k1')              default basename of weights

  python3 encode_game_multi.py <blunders.json> <white_elo> <black_elo>
"""
import sys; sys.path.insert(0, "/home/ec2-user/SageMaker/maia3")
import os, json, torch, numpy as np, chess
import torch.nn.functional as F
from types import SimpleNamespace
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board, get_historical_tokens

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/my_blunders.json"
WHITE_ELO = float(sys.argv[2]) if len(sys.argv) > 2 else 1535
BLACK_ELO = float(sys.argv[3]) if len(sys.argv) > 3 else 1571

SAE_W = os.environ.get("SAE_WEIGHTS", BASE + "/output/maia3_sae/btk_2048_k6_nol2.pt")
SAE_K = int(os.environ.get("SAE_K", "6"))
LBL = os.environ.get("LABELS_FILE", "relabel_v3_5word_d2048_k6.json")
LEAF = os.environ.get("LEAF_FILE", "")
OUT = os.environ.get("OUT", "/tmp/blunder_features.json")
TAG = os.environ.get("TAG", os.path.basename(SAE_W).replace(".pt", ""))

cfg = SimpleNamespace(history=8, use_padding=True, include_time_info=False, dim_emb=128,
    num_blocks=8, mlp_ratio=2.0, dropout=0.0, use_gab=True, use_relative_bias=False,
    use_absolute_pe=False, use_rms_norm=True, omit_qkv_biases=True, activation="gelu",
    dim_vit=1024, head_hid_dim=1024, num_heads=32, gab_gen_size=128, gab_per_square_dim=32,
    gab_intermediate_dim=128)
ckpt = torch.load("/home/ec2-user/SageMaker/maia3_79m_fixed.pt", map_location="cpu", weights_only=False)
model = MAIA3Model(cfg); model.load_state_dict(ckpt); model.eval().to(DEV)
tcfg = SimpleNamespace(history=8, include_time_info=False, dim_emb=128)
print(f"Maia3 79M loaded {DEV}", flush=True)

cache = torch.load(BASE + "/cache/maia3_l7only_v2_dedup.pt", map_location="cpu", weights_only=False)
craw = cache["activations"].float()
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
sd = torch.load(SAE_W, map_location="cpu", weights_only=False)["state_dict"]
print(f"SAE: {TAG} k={SAE_K} dict={sd['W_enc'].shape[1]}", flush=True)

labels = json.load(open(LBL))
leaf = json.load(open(LEAF)) if LEAF and os.path.exists(LEAF) else {}

buf = {}
def mk(n):
    def h(m, i, o): buf[n] = o.detach()
    return h
model.transformer.layers[6].register_forward_hook(mk("L7"))

def histtok(b):
    t = tokenize_board(b)
    return get_historical_tokens([t] * 8, tcfg, 0, 0, 0, 0)

def encode_L7(boards, es, eo):
    T = torch.stack([histtok(b) for b in boards]).to(DEV)
    esT = torch.tensor(es, dtype=torch.float32, device=DEV); eoT = torch.tensor(eo, dtype=torch.float32, device=DEV)
    with torch.no_grad(): model(T, esT, eoT)
    return buf["L7"].float().cpu().numpy()

def sae_fire(diff_vec):
    x = (torch.tensor(diff_vec, dtype=torch.float32) - zmean) / zstd
    z = F.relu((x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"]).numpy()
    top = np.argsort(-z)[:SAE_K]
    return [(int(i), float(z[i])) for i in top if z[i] > 0]

blunders = json.load(open(IN))
out = []
for m in blunders:
    fen = m["fen"]; blun = m["uci"]; best = m.get("best_uci") or m.get("best_san")
    b = chess.Board(fen); pov = b.turn
    es, eo = (WHITE_ELO, BLACK_ELO) if pov else (BLACK_ELO, WHITE_ELO)
    try:
        bestmv = chess.Move.from_uci(best)
        if bestmv not in b.legal_moves: bestmv = b.parse_san(best)
    except Exception:
        bestmv = b.parse_san(m["best_san"])
    bb = b.copy(); bb.push(chess.Move.from_uci(blun))
    sb = b.copy(); sb.push(bestmv)
    L7bl = encode_L7([bb], [es], [eo])[0]; L7bs = encode_L7([sb], [es], [eo])[0]
    diff = (L7bs - L7bl).mean(0)
    feats = []
    for fid, act in sae_fire(diff):
        f = str(fid); lab = labels.get(f, {}); lf = leaf.get(f, {})
        feats.append({"fid": fid, "act": round(act, 2),
                      "chip": lab.get("chip", "?"),
                      "direction": lab.get("direction", ""),
                      "category": lf.get("bucket_name", ""),
                      "cluster": lf.get("sub", ""),
                      "review": lab.get("review", "")})
    out.append({"move_num": m["move_num"], "san": m["san"], "cp_loss": m["cp_loss"],
                "best_san": m.get("best_san"), "fen": fen, "uci": blun, "sae": TAG, "features": feats})

json.dump(out, open(OUT, "w"), indent=1)
for o in out:
    print(f"\n{o['move_num']}. {o['san']} (lost {o['cp_loss']}cp, best {o['best_san']}):")
    for f in o["features"]:
        extra = f" [{f['review']}]" if f["review"] else ""
        cat = f" | {f['category']}" if f["category"] else ""
        print(f"    f{f['fid']} act {f['act']:>5} | {f['chip']}{cat}{extra}")
print(f"\nwrote {OUT} ({TAG})", flush=True)
