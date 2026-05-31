"""
Encode 18k Opus positions through all 3 SAEs.
Builds top-20 per feature from Opus-labeled positions.
Resumes from existing partial work if available.
"""
import torch, torch.nn.functional as F, numpy as np, json, os, heapq, time, sys
import onnxruntime as ort, chess
from types import SimpleNamespace
sys.path.insert(0, '/home/ec2-user/SageMaker/maia3')
from maia3.models import MAIA3Model
import maia3.dataset as ds

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
SAE_DIR = BASE + "/output/maia3_sae"
PIECES=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]

# Load opus + source
print("Loading data...")
opus = json.load(open("/home/ec2-user/SageMaker/all_positions_labeled_opus.json"))
src  = json.load(open(BASE+"/cache/real_game_blunder_positions.json"))
def strip_fen(f): return ' '.join(f.split()[:4])
src_by_key = {strip_fen(s["fen"])+"|"+s["blunder_uci"]: s for s in src}

positions = []
for k,v in opus.items():
    parts = k.rsplit("|",1)
    if len(parts)!=2: continue
    stripped = strip_fen(parts[0])+"|"+parts[1]
    if stripped not in src_by_key: continue
    an = v.get("analysis",v)
    if not isinstance(an,dict): continue
    positions.append((stripped, src_by_key[stripped], an))
print(f"Aligned: {len(positions)} positions")

# Check for existing partial work
existing_keys = set()
existing_path = BASE+"/cache/opus_encoded_positions.pt"
all_positions_meta = []
TOP_N = 20
D = 2048
SAE_NAMES = ["option_a","board_diff","l2l7"]
top_k = {name: [[] for _ in range(D)] for name in SAE_NAMES}

if os.path.exists(existing_path):
    print("Loading existing partial work...")
    existing = torch.load(existing_path, map_location="cpu", weights_only=False)
    # existing has option_a, board_diff, l2l7 arrays + positions list
    exist_pos = existing.get("positions", [])
    exist_oa  = existing.get("option_a",  None)
    exist_bd  = existing.get("board_diff",None)
    exist_l2l7= existing.get("l2l7",     None)
    
    if exist_pos and exist_oa is not None:
        n_exist = len(exist_pos)
        print(f"  Found {n_exist} previously encoded positions")
        # Rebuild top-k heaps from existing data
        for sae_name, acts_all in [("option_a",exist_oa),("board_diff",exist_bd),("l2l7",exist_l2l7)]:
            if acts_all is None: continue
            for pos_idx in range(n_exist):
                acts = acts_all[pos_idx]
                firing = np.where(acts>0)[0]
                for fi in firing:
                    val = float(acts[fi])
                    heap = top_k[sae_name][fi]
                    if len(heap)<TOP_N: heapq.heappush(heap,(val,pos_idx))
                    elif val>heap[0][0]: heapq.heapreplace(heap,(val,pos_idx))
        all_positions_meta = exist_pos
        existing_keys = {p["key"] for p in exist_pos}
        print(f"  Restored heaps from {n_exist} positions")

# Load SAEs
def load_sae(path): return torch.load(path,map_location="cpu",weights_only=False)
saes = {
    "option_a":   load_sae(SAE_DIR+"/maia3_option_a_2048_k32.pt"),
    "board_diff": load_sae(SAE_DIR+"/maia3_board_diff_2048_k32.pt"),
    "l2l7":       load_sae(SAE_DIR+"/maia3_l2l7_2048_k32.pt"),
}

def sae_forward(x_np, sae_ck):
    norm=sae_ck.get("norm",{}); sd=sae_ck["state_dict"]
    mean=np.array(norm.get("mean",np.zeros(len(x_np))),dtype=np.float32)
    std =np.array(norm.get("std", np.ones(len(x_np))),dtype=np.float32).clip(1e-6)
    x=(x_np-mean)/std; x=x/(np.linalg.norm(x)+1e-8)
    x_t=torch.tensor(x,dtype=torch.float32)
    We,be,bd=sd["W_enc"],sd["b_enc"],sd["b_dec"]
    with torch.no_grad(): z=F.relu((x_t-bd)@We+be)
    return z.numpy()

# ONNX
sess=ort.InferenceSession("/home/ec2-user/SageMaker/maia3_with_probe.onnx",
                          providers=["CPUExecutionProvider"])
PROBE="/model/transformer/layers.7/Add_2_output_0"
with open(BASE+"/cache/move_to_action.json") as f: m2a=json.load(f)

def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t
def run_onnx(b,elo=1500.):
    return sess.run([PROBE],{"tokens":tok(b)[None],
        "elo_self":np.array([elo],np.float32),"elo_oppo":np.array([elo],np.float32)})[0][0].astype(np.float32)
def get_maia_best(b,elo=1500.):
    out=sess.run(["logits_move"],{"tokens":tok(b)[None],
        "elo_self":np.array([elo],np.float32),"elo_oppo":np.array([elo],np.float32)})[0][0]
    legal={m.uci() for m in b.legal_moves}; best=None; bl=-1e9
    for uci,idx in m2a.items():
        if uci in legal and out[idx]>bl: bl=out[idx]; best=uci
    return best

# 79M
print("Loading 79M...")
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,
    num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,
    use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',
    dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,
    gab_per_square_dim=32,gab_intermediate_dim=128)
ckpt79=torch.load("/home/ec2-user/SageMaker/maia3_79m_fixed.pt",map_location="cpu",weights_only=False)
model79=MAIA3Model(cfg); model79.load_state_dict(ckpt79); model79.eval()
tok_cfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)

def get_layers_79m(b,elo=1500.):
    t=ds.tokenize_board(b)
    tokens=ds.get_historical_tokens([t]*8,tok_cfg,0,0,0,0).unsqueeze(0)
    elo_t=torch.tensor([float(elo)],dtype=torch.float32)
    L2={};L7={}
    h2=model79.transformer.layers[1].register_forward_hook(lambda m,i,o:L2.__setitem__('h',o.detach()))
    h7=model79.transformer.layers[6].register_forward_hook(lambda m,i,o:L7.__setitem__('h',o.detach()))
    with torch.no_grad(): model79(tokens,elo_t,elo_t)
    h2.remove();h7.remove()
    return L2['h'][0].numpy(), L7['h'][0].numpy()

# Encode remaining positions
remaining = [(k,s,a) for k,s,a in positions if k not in existing_keys]
print(f"To encode: {len(remaining)} (skipping {len(existing_keys)} already done)")
t0=time.time(); n_done=0; n_err=0

for key,src_m,opus_an in remaining:
    try:
        elo=float(src_m.get("white_elo" if src_m.get("is_white") else "black_elo",1500) or 1500)
        b0=chess.Board(src_m["fen"])
        bl_mv=chess.Move.from_uci(src_m["blunder_uci"])
        best_uci=get_maia_best(b0,elo)
        if not best_uci or best_uci==src_m["blunder_uci"]: n_err+=1; continue
        bs_mv=chess.Move.from_uci(best_uci)
        bl_b=b0.copy(); bl_b.push(bl_mv)
        bs_b=b0.copy(); bs_b.push(bs_mv)

        H0 =run_onnx(b0,elo)
        H_bl=run_onnx(bl_b,elo); H_bs=run_onnx(bs_b,elo)
        L2_bl,L7_bl=get_layers_79m(bl_b,elo)
        L2_bs,L7_bs=get_layers_79m(bs_b,elo)

        diffs={
            "option_a":   H0[bs_mv.to_square]-H0[bl_mv.to_square],
            "board_diff": (H_bs-H_bl).mean(0),
            "l2l7":       np.concatenate([(L2_bs-L2_bl).mean(0),(L7_bs-L7_bl).mean(0)]),
        }

        pos_idx=len(all_positions_meta)
        all_positions_meta.append({"key":key,"fen":src_m["fen"],
                                    "blunder_uci":src_m["blunder_uci"],
                                    "best_uci":best_uci,
                                    "cp_loss":src_m.get("cp_loss",0),
                                    "analysis":opus_an})

        for name,diff_vec in diffs.items():
            acts=sae_forward(diff_vec,saes[name])
            for fi in np.where(acts>0)[0]:
                val=float(acts[fi]); heap=top_k[name][fi]
                if len(heap)<TOP_N: heapq.heappush(heap,(val,pos_idx))
                elif val>heap[0][0]: heapq.heapreplace(heap,(val,pos_idx))

        n_done+=1
        if n_done%500==0:
            elapsed=time.time()-t0; rate=n_done/elapsed
            eta=(len(remaining)-n_done)/rate
            print(f"  {n_done}/{len(remaining)} ({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s)",flush=True)
    except: n_err+=1

total=len(all_positions_meta)
print(f"\nEncoded: {n_done} new + {len(existing_keys)} existing = {total} total, {n_err} errors")

# Save profiles
print("Saving profiles...")
for name in SAE_NAMES:
    profiles={}
    for fi in range(D):
        heap=top_k[name][fi]
        if not heap: continue
        examples=[]
        for val,pos_idx in sorted(heap,key=lambda x:-x[0]):
            pm=all_positions_meta[pos_idx]
            examples.append({"activation":val,"fen":pm["fen"],"uci":pm["blunder_uci"],
                              "best_uci":pm["best_uci"],"cp_loss":pm["cp_loss"],
                              "analysis":pm["analysis"]})
        profiles[str(fi)]={"fire_rate":len(heap)/total,"examples":examples}
    out=f"{SAE_DIR}/{name}_opus_profiles.json"
    json.dump(profiles,open(out,"w"),indent=1)
    covered=sum(1 for v in profiles.values() if v["examples"])
    print(f"  {name}: {covered}/{D} features with Opus examples → {out}")

print("Done.")
