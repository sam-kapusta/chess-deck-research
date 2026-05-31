"""
Evaluate 4 test positions through all 4 SAEs.
Uses calibrated threshold (Sandstone pattern) — shows ALL features that fire.
Requires: {name}_opus_labels.json for each SAE.
"""
import torch, torch.nn.functional as F, numpy as np, json, os, sys
import onnxruntime as ort, chess
from types import SimpleNamespace
sys.path.insert(0, '/home/ec2-user/SageMaker/maia3')
from maia3.models import MAIA3Model
import maia3.dataset as ds

BASE    = "/home/ec2-user/SageMaker/chess-stage-a"
SAE_DIR = BASE + "/output/maia3_sae"
PIECES  = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
JUNK    = ["insufficient","unclassified","unavailable","unanalyzed","no data","no label","sparse"]

# ── Load everything ──
test_cases = json.load(open("/home/ec2-user/SageMaker/chess-deck-research/output/test_positions.json"))

def load_labels(primary, fallback=None):
    for p in [primary, fallback]:
        if p and os.path.exists(p):
            return {str(k):v for k,v in json.load(open(p)).items()}
    return {}

labels = {
    "v2":         load_labels(SAE_DIR+"/l2_labels_sonnet.json"),
    "option_a":   load_labels(SAE_DIR+"/option_a_opus_labels.json",  SAE_DIR+"/option_a_labels.json"),
    "board_diff": load_labels(SAE_DIR+"/board_diff_opus_labels.json", SAE_DIR+"/board_diff_labels.json"),
    "l2l7":       load_labels(SAE_DIR+"/l2l7_opus_labels.json",       SAE_DIR+"/l2l7_labels.json"),
}
for name,lbls in labels.items():
    good = sum(1 for v in lbls.values()
               if v.get("chip") and not any(j in v.get("chip", v.get("specific_label", "")).lower() for j in JUNK))
    print(f"  {name}: {len(lbls)} labels, {good} good")

def load_sae(path):
    return torch.load(path, map_location="cpu", weights_only=False)

saes = {
    "v2":         load_sae(SAE_DIR+"/maia3_sae_diff_v2_2048_k32_l2.pt"),
    "option_a":   load_sae(SAE_DIR+"/maia3_option_a_2048_k32.pt"),
    "board_diff": load_sae(SAE_DIR+"/maia3_board_diff_2048_k32.pt"),
    "l2l7":       load_sae(SAE_DIR+"/maia3_l2l7_2048_k32.pt"),
}

# ── Compute calibrated threshold per SAE (Sandstone pattern) ──
cache_map = {
    "v2":         BASE+"/cache/maia3_blunder_diff_v2.pt",
    "option_a":   BASE+"/cache/maia3_option_a_diff.pt",
    "board_diff": BASE+"/cache/maia3_board_diff_both.pt",
    "l2l7":       BASE+"/cache/maia3_l2l7_concat.pt",
}

def compute_threshold(sae_ck, cache_path, target_k=32, n=5000):
    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw = d["activations"].float()[:n]
    norm = sae_ck.get("norm", {})
    mean = torch.tensor(np.array(norm.get("mean", np.zeros(raw.shape[1]))), dtype=torch.float32)
    std  = torch.tensor(np.array(norm.get("std",  np.ones(raw.shape[1]))),  dtype=torch.float32).clamp(min=1e-6)
    x = (raw-mean)/std; x = x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    sd = sae_ck["state_dict"]
    with torch.no_grad():
        z = F.relu((x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"])
    vals = z.flatten().numpy(); vals = vals[vals>0]
    if len(vals)==0: return 0.0
    target = n * target_k
    sorted_vals = np.sort(vals)[::-1]
    thresh = float(sorted_vals[min(target, len(sorted_vals)-1)])
    avg = (z > thresh).float().sum(1).mean().item()
    print(f"    {thresh:.4f} (avg {avg:.1f} features/position)")
    return thresh

print("\nComputing thresholds...")
thresholds = {}
for name, sae_ck in saes.items():
    print(f"  {name}:", end=" ", flush=True)
    thresholds[name] = compute_threshold(sae_ck, cache_map[name])

# ── Encoders ──
sess = ort.InferenceSession("/home/ec2-user/SageMaker/maia3_with_probe.onnx",
                            providers=["CPUExecutionProvider"])
PROBE = "/model/transformer/layers.7/Add_2_output_0"
with open(BASE+"/cache/move_to_action.json") as f: m2a = json.load(f)

def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t

def run_onnx(b, elo=1500.):
    return sess.run([PROBE],{"tokens":tok(b)[None],
        "elo_self":np.array([elo],np.float32),
        "elo_oppo":np.array([elo],np.float32)})[0][0].astype(np.float32)

def get_maia_best(b, elo=1500.):
    out = sess.run(["logits_move"],{"tokens":tok(b)[None],
        "elo_self":np.array([elo],np.float32),
        "elo_oppo":np.array([elo],np.float32)})[0][0]
    legal = {m.uci() for m in b.legal_moves}
    best=None; bl=-1e9
    for uci,idx in m2a.items():
        if uci in legal and out[idx]>bl: bl=out[idx]; best=uci
    return best

print("\nLoading 79M...")
cfg=SimpleNamespace(history=8,use_padding=True,include_time_info=False,dim_emb=128,
    num_blocks=8,mlp_ratio=2.0,dropout=0.0,use_gab=True,use_relative_bias=False,
    use_absolute_pe=False,use_rms_norm=True,omit_qkv_biases=True,activation='gelu',
    dim_vit=1024,head_hid_dim=1024,num_heads=32,gab_gen_size=128,
    gab_per_square_dim=32,gab_intermediate_dim=128)
ckpt79=torch.load("/home/ec2-user/SageMaker/maia3_79m_fixed.pt",map_location="cpu",weights_only=False)
model79=MAIA3Model(cfg); model79.load_state_dict(ckpt79); model79.eval()
tok_cfg=SimpleNamespace(history=8,include_time_info=False,dim_emb=128)

def get_layers_79m(b, elo=1500.):
    t=ds.tokenize_board(b)
    tokens=ds.get_historical_tokens([t]*8,tok_cfg,0,0,0,0).unsqueeze(0)
    elo_t=torch.tensor([float(elo)],dtype=torch.float32)
    L2={};L7={}
    h2=model79.transformer.layers[1].register_forward_hook(lambda m,i,o:L2.__setitem__('h',o.detach()))
    h7=model79.transformer.layers[6].register_forward_hook(lambda m,i,o:L7.__setitem__('h',o.detach()))
    with torch.no_grad(): model79(tokens,elo_t,elo_t)
    h2.remove();h7.remove()
    return L2['h'][0].numpy(), L7['h'][0].numpy()

def sae_encode(x_np, sae_ck, threshold):
    norm=sae_ck.get("norm",{}); sd=sae_ck["state_dict"]
    mean=np.array(norm.get("mean",np.zeros(len(x_np))),dtype=np.float32)
    std =np.array(norm.get("std", np.ones(len(x_np))),dtype=np.float32).clip(1e-6)
    x=(x_np-mean)/std; x=x/(np.linalg.norm(x)+1e-8)
    x_t=torch.tensor(x,dtype=torch.float32)
    with torch.no_grad():
        z=F.relu((x_t-sd["b_dec"])@sd["W_enc"]+sd["b_enc"])
    acts=z.numpy(); acts[acts<threshold]=0.0
    return acts

def get_label(fid, lbls):
    v=lbls.get(str(fid),{}); chip=v.get("chip", v.get("specific_label", "")).strip(); desc=v.get("description","").strip()
    if not chip: return None
    if any(j in chip.lower() for j in JUNK): return None
    return chip + (" — " + desc[:100] if desc else "")

# ── Run eval ──
print("\n" + "="*70)
print("EVALUATION: 4 Real Positions vs 4 SAEs")
print("="*70)

for tc in test_cases:
    print(f"\n{'─'*65}")
    print(f"POSITION: {tc['id'].upper()}")
    print(f"Game:  {tc['game']}")
    print(f"Move:  {tc['move']}  (cp_loss={tc.get('cp_loss',0)})")
    print(f"Type:  {tc['mistake_type']}")
    print(f"Coach: {tc['coach_would_say']}")
    print(f"{'─'*65}")

    b0 = chess.Board(tc["fen"])
    bl_mv = chess.Move.from_uci(tc["blunder_uci"])
    best_uci = tc.get("best_uci") or get_maia_best(b0)
    bs_mv = chess.Move.from_uci(best_uci)

    bl_b=b0.copy(); bl_b.push(bl_mv)
    bs_b=b0.copy(); bs_b.push(bs_mv)
    elo=1500.

    H0  = run_onnx(b0, elo)
    H_bl= run_onnx(bl_b, elo)
    H_bs= run_onnx(bs_b, elo)
    L2_bl,L7_bl = get_layers_79m(bl_b, elo)
    L2_bs,L7_bs = get_layers_79m(bs_b, elo)

    diffs = {
        "v2 (current)":     H0[bl_mv.to_square] - H0[bl_mv.from_square],
        "option_a":         H0[bs_mv.to_square]  - H0[bl_mv.to_square],
        "board_diff":       (H_bs - H_bl).mean(0),
        "l2l7":             np.concatenate([(L2_bs-L2_bl).mean(0),(L7_bs-L7_bl).mean(0)]),
    }
    sae_keys = ["v2 (current)","option_a","board_diff","l2l7"]
    sae_name_map = {"v2 (current)":"v2","option_a":"option_a","board_diff":"board_diff","l2l7":"l2l7"}

    for diff_name in sae_keys:
        sname = sae_name_map[diff_name]
        acts = sae_encode(diffs[diff_name], saes[sname], thresholds[sname])
        active = np.where(acts>0)[0]
        active_sorted = active[np.argsort(acts[active])[::-1]]
        n_active = len(active_sorted)

        # Split labeled vs unlabeled
        labeled   = [(fi, acts[fi], get_label(fi,labels[sname]))
                     for fi in active_sorted if get_label(fi,labels[sname])]
        n_unlabeled = n_active - len(labeled)

        print(f"\n  ┌─ {diff_name} ─── {n_active} fired ({len(labeled)} labeled, {n_unlabeled} no label)")
        if labeled:
            for rank,(fid,act,lbl) in enumerate(labeled[:8]):
                marker = " ✓" if any(kw in lbl.lower() for kw in
                    ["hang","undefend","en prise","attack","fork","pin","tactic","king","expose","capture"]) else ""
                print(f"  │  [{rank+1}] f{fid} ({act:.3f}): {lbl}{marker}")
        else:
            print(f"  │  (no labeled features — all {n_active} above threshold are unlabeled)")
            for fi in active_sorted[:5]:
                print(f"  │  f{fi} ({acts[fi]:.3f}): (no label)")
        print(f"  └{'─'*55}")

print("\n\n✓ = likely coaching-relevant keyword detected")
print("Done.")
