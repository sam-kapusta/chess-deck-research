#!/usr/bin/env python3
"""Encode a single game's blunders through the Maia3 -> L7-diff -> d2048_k6 SAE pipeline,
then map fired features to the v3 taxonomy (cluster + category).

Replicates the corpus encoding EXACTLY (from build_l2l7_v2.py): Maia3 79M, hooks on layers[1]/[6],
diff = mean64(L7_after_best - L7_after_blunder), z-scored with the CORPUS mean/std so activations
are comparable to the trained SAE, then SAE encode (BatchTopK k=6) -> fired features.

Differences from corpus build, stated honestly:
  - best move here is STOCKFISH-best (the game was analyzed with SF), not Maia-best@2600. Still
    "strong move - blunder", a small distribution shift. Acceptable for a fresh game.
  - elos are the player's real elos from the game (not forced to 2600).

Input: /tmp/my_blunders.json (list of {fen, uci(=blunder), best_san/best_uci, cp_loss, ...}).
Run on chess-poc (Maia3 + GPU).
"""
import sys; sys.path.insert(0, "/home/ec2-user/SageMaker/maia3")
import json, torch, numpy as np, chess
from types import SimpleNamespace
from maia3.models import MAIA3Model
from maia3.dataset import tokenize_board, get_historical_tokens

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WHITE_ELO = float(sys.argv[2]) if len(sys.argv) > 2 else 1535   # player elo (mover)
BLACK_ELO = float(sys.argv[3]) if len(sys.argv) > 3 else 1571
IN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/my_blunders.json"

cfg = SimpleNamespace(history=8, use_padding=True, include_time_info=False, dim_emb=128,
    num_blocks=8, mlp_ratio=2.0, dropout=0.0, use_gab=True, use_relative_bias=False,
    use_absolute_pe=False, use_rms_norm=True, omit_qkv_biases=True, activation="gelu",
    dim_vit=1024, head_hid_dim=1024, num_heads=32, gab_gen_size=128, gab_per_square_dim=32,
    gab_intermediate_dim=128)
ckpt = torch.load("/home/ec2-user/SageMaker/maia3_79m_fixed.pt", map_location="cpu", weights_only=False)
model = MAIA3Model(cfg); model.load_state_dict(ckpt); model.eval().to(DEV)
tcfg = SimpleNamespace(history=8, include_time_info=False, dim_emb=128)
print("Maia3 79M loaded", DEV, flush=True)

# --- model/taxonomy selection via env (default = d2048_k6 v3) ---
import os
SAE_W = os.environ.get("SAE_WEIGHTS", BASE + "/output/maia3_sae/btk_2048_k6_nol2.pt")
SAE_K = int(os.environ.get("SAE_K", "6"))
LBL = os.environ.get("LABELS_FILE", "relabel_v3_5word_d2048_k6.json")
LEAF = os.environ.get("LEAF_FILE", "feature_leaf_llm_d2048_k6.json")

# corpus z-score constants + SAE weights + taxonomy
cache = torch.load(BASE + "/cache/maia3_l7only_v2_dedup.pt", map_location="cpu", weights_only=False)
craw = cache["activations"].float()
zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)   # SAME z-score the SAE was trained with
sd = torch.load(SAE_W, map_location="cpu", weights_only=False)["state_dict"]
print(f"SAE: {os.path.basename(SAE_W)} k={SAE_K} dict={sd['W_enc'].shape[1]}", flush=True)

# taxonomy (pulled local-> uploaded alongside)
labels = json.load(open(LBL))
leaf = json.load(open(LEAF))

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
    return buf["L7"].float().cpu().numpy()   # [B,64,1024]

import torch.nn.functional as F
def sae_fire(diff_vec):
    x = (torch.tensor(diff_vec, dtype=torch.float32) - zmean) / zstd
    z = F.relu((x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"]).numpy()
    # BatchTopK at inference: take top-k by activation
    top = np.argsort(-z)[:SAE_K]
    return [(int(i), float(z[i])) for i in top if z[i] > 0]

blunders = json.load(open(IN))
out = []
for m in blunders:
    fen = m["fen"]; blun = m["uci"]; best = m.get("best_uci") or m.get("best_san")
    b = chess.Board(fen); pov = b.turn
    es, eo = (WHITE_ELO, BLACK_ELO) if pov else (BLACK_ELO, WHITE_ELO)
    # resolve best to uci if given as san
    try:
        bestmv = chess.Move.from_uci(best)
        if bestmv not in b.legal_moves: bestmv = b.parse_san(best)
    except Exception:
        bestmv = b.parse_san(m["best_san"])
    bb = b.copy(); bb.push(chess.Move.from_uci(blun))
    sb = b.copy(); sb.push(bestmv)
    L7bl = encode_L7([bb], [es], [eo])[0]; L7bs = encode_L7([sb], [es], [eo])[0]
    diff = (L7bs - L7bl).mean(0)   # mean64
    fired = sae_fire(diff)
    feats = []
    for fid, act in fired:
        f = str(fid); lf = leaf.get(f, {}); lab = labels.get(f, {})
        feats.append({"fid": fid, "act": round(act, 2), "chip": lab.get("chip", "?"),
                      "category": lf.get("bucket_name", "?"), "cluster": lf.get("sub", "?"),
                      "mixed": bool(lab.get("mixed"))})
    out.append({"move_num": m["move_num"], "san": m["san"], "cp_loss": m["cp_loss"],
                "best_san": m.get("best_san"), "features": feats})

json.dump(out, open("/tmp/blunder_features.json", "w"), indent=1)
for o in out:
    print(f"\n{o['move_num']}. {o['san']} (lost {o['cp_loss']}cp, best {o['best_san']}):")
    for f in o["features"]:
        mx = " [mixed]" if f["mixed"] else ""
        print(f"    f{f['fid']} act {f['act']:>5} | {f['category']} > {f['cluster']} | {f['chip']}{mx}")
print("\nwrote /tmp/blunder_features.json", flush=True)
