# WARNING: built on v1 cache (label-inversion bug). Must repoint to v2 cache before reuse.
"""
Build L2+L7 concat cache: concat(mean64(L2_best-L2_blunder), mean64(L7_best-L7_blunder))
2048-dim output (1024+1024). Uses 79M PyTorch model.
Based on probe results: L2 better for positional mistakes, L7 better for tactical timing.
"""
import torch, torch.nn.functional as F, numpy as np, chess, time, sys, json
from types import SimpleNamespace
sys.path.insert(0,'/home/ec2-user/SageMaker/maia3')
from maia3.models import MAIA3Model
import maia3.dataset as ds

BASE="/home/ec2-user/SageMaker/chess-stage-a"
OUT=BASE+"/cache/maia3_l2l7_concat.pt"

print("Loading 79M...")
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,
    num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,
    use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',
    dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,
    gab_per_square_dim=32,gab_intermediate_dim=128)
ckpt=torch.load("/home/ec2-user/SageMaker/maia3_79m_fixed.pt",map_location="cpu",weights_only=False)
model=MAIA3Model(cfg); model.load_state_dict(ckpt); model.eval()
tok_cfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)

v1=torch.load(BASE+"/cache/maia3_blunder_diff.pt",map_location="cpu",weights_only=False)
meta=v1["metadata"]; elo_self=v1["elo_self"]; elo_oppo=v1["elo_oppo"]
print(f"Loaded {len(meta)} positions")

def get_both_layers(boards, elos_s, elos_o):
    """Batch encode, return (L2_out, L7_out) each [N,64,1024]."""
    results_L2=[]; results_L7=[]
    for b,es,eo in zip(boards,elos_s,elos_o):
        t=ds.tokenize_board(b)
        tokens=ds.get_historical_tokens([t]*8,tok_cfg,0,0,0,0).unsqueeze(0)
        elo_s=torch.tensor([float(es)],dtype=torch.float32)
        elo_o_t=torch.tensor([float(eo)],dtype=torch.float32)
        L2_out={}; L7_out={}
        h2=model.transformer.layers[1].register_forward_hook(
            lambda m,i,o: L2_out.__setitem__('h',o.detach()))
        h7=model.transformer.layers[6].register_forward_hook(
            lambda m,i,o: L7_out.__setitem__('h',o.detach()))
        with torch.no_grad(): model(tokens,elo_s,elo_o_t)
        h2.remove(); h7.remove()
        results_L2.append(L2_out['h'][0].numpy())  # [64,1024]
        results_L7.append(L7_out['h'][0].numpy())  # [64,1024]
    return np.stack(results_L2), np.stack(results_L7)

BS=32  # smaller batch since 2 layer hooks per position
diffs=[]; metas_out=[]; n_errors=0
t0=time.time()

valid=[i for i,m in enumerate(meta) if m.get("best_uci")]
print(f"Processing {len(valid)} positions at bs={BS} (4 layer hooks each)...")

for batch_start in range(0,len(valid),BS):
    batch_idx=valid[batch_start:batch_start+BS]
    bl_boards=[]; bs_boards=[]; es=[]; eo=[]; batch_meta=[]

    for i in batch_idx:
        m=meta[i]
        try:
            b0=chess.Board(m["fen"])
            bl_b=b0.copy(); bl_b.push(chess.Move.from_uci(m["blunder_uci"]))
            bs_b=b0.copy(); bs_b.push(chess.Move.from_uci(m["best_uci"]))
            bl_boards.append(bl_b); bs_boards.append(bs_b)
            es.append(float(elo_self[i])); eo.append(float(elo_oppo[i]))
            batch_meta.append({"fen":m["fen"],"blunder_uci":m["blunder_uci"],
                               "best_uci":m["best_uci"],"cp_loss":m["cp_loss"],
                               "elo_self":float(elo_self[i]),"elo_oppo":float(elo_oppo[i])})
        except: n_errors+=1; continue

    if not bl_boards: continue
    try:
        L2_bl,L7_bl=get_both_layers(bl_boards,es,eo)   # blunder boards
        L2_bs,L7_bs=get_both_layers(bs_boards,es,eo)   # best boards

        for j in range(len(bl_boards)):
            d2=(L2_bs[j]-L2_bl[j]).mean(0)  # [1024]
            d7=(L7_bs[j]-L7_bl[j]).mean(0)  # [1024]
            d=np.concatenate([d2,d7])        # [2048]
            if np.linalg.norm(d)>1e-6:
                diffs.append(d); metas_out.append(batch_meta[j])
    except Exception as e:
        n_errors+=len(bl_boards)

    done=batch_start+len(batch_idx)
    if done%(BS*20)==0 or done>=len(valid):
        elapsed=time.time()-t0
        rate=done/max(elapsed,1); eta=(len(valid)-done)/rate
        print(f"  {done}/{len(valid)} ({elapsed:.0f}s, {rate:.0f} pos/s, ETA {eta:.0f}s)",flush=True)

print(f"\nDone: {len(diffs)} diffs, {n_errors} errors")
activations=np.stack(diffs).astype(np.float32)
mean=activations.mean(0); std=activations.std(0)
torch.save({"activations":torch.tensor(activations),"mean":mean,"std":std,
            "metadata":metas_out,
            "config":{"construction":"concat(L2_mean64_diff, L7_mean64_diff)",
                      "dims":"1024+1024=2048","source":"v1","n":len(diffs)}},OUT)
print(f"Saved: {OUT} ({__import__('os').path.getsize(OUT)//1024//1024}MB)")
