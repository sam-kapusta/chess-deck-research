"""
Build Option-A diff cache: h[best_to_sq] - h[blunder_to_sq] on before-board.
Uses v1 positions (200k, all have best_uci + elo_self/elo_oppo).
Batched at bs=128 for speed (~1hr).
"""
import onnxruntime as ort, numpy as np, torch, chess, time, sys, json
BASE="/home/ec2-user/SageMaker/chess-stage-a"
OUT=BASE+"/cache/maia3_option_a_diff.pt"

v1=torch.load(BASE+"/cache/maia3_blunder_diff.pt",map_location="cpu",weights_only=False)
meta=v1["metadata"]; elo_self=v1["elo_self"]; elo_oppo=v1["elo_oppo"]
print(f"Loaded {len(meta)} positions")

# Verify best_uci presence
n_best=sum(1 for m in meta if m.get("best_uci"))
print(f"Positions with best_uci: {n_best}/{len(meta)}")

sess=ort.InferenceSession("/home/ec2-user/SageMaker/maia3_with_probe.onnx",
                          providers=["CPUExecutionProvider"])
PROBE="/model/transformer/layers.7/Add_2_output_0"
PIECES=[chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]

def tok(b):
    t=np.zeros((64,12),np.float32)
    for sq in range(64):
        p=b.piece_at(sq)
        if p: t[sq,(0 if p.color else 6)+PIECES.index(p.piece_type)]=1.0
    return t

def encode_batch(boards, es, eo):
    """Returns [N, 64, 512] layer-7 activations."""
    tokens=np.stack([tok(b) for b in boards])
    out=sess.run([PROBE],{
        "tokens": tokens,
        "elo_self": np.array(es, dtype=np.float32),
        "elo_oppo": np.array(eo, dtype=np.float32)
    })[0]  # [N, 64, 512]
    return out.astype(np.float32)

BS=128
diffs=[]; metas_out=[]; n_errors=0
t0=time.time()

# Collect valid indices
valid=[i for i,m in enumerate(meta) if m.get("best_uci")]
print(f"Processing {len(valid)} positions at bs={BS}...")

for batch_start in range(0, len(valid), BS):
    batch_idx=valid[batch_start:batch_start+BS]
    boards=[]; es=[]; eo=[]; bl_moves=[]; bs_moves=[]; batch_meta=[]

    for i in batch_idx:
        m=meta[i]
        try:
            b0=chess.Board(m["fen"])
            bl_mv=chess.Move.from_uci(m["blunder_uci"])
            bs_mv=chess.Move.from_uci(m["best_uci"])
            boards.append(b0)
            es.append(float(elo_self[i])); eo.append(float(elo_oppo[i]))
            bl_moves.append(bl_mv); bs_moves.append(bs_mv)
            batch_meta.append({
                "fen": m["fen"],
                "blunder_uci": m["blunder_uci"],
                "best_uci": m["best_uci"],
                "cp_loss": m["cp_loss"],
                "elo_self": float(elo_self[i]),
                "elo_oppo": float(elo_oppo[i])
            })
        except: n_errors+=1

    if not boards: continue
    try:
        H=encode_batch(boards, es, eo)  # [N, 64, 512]
        for j in range(len(boards)):
            best_to=bs_moves[j].to_square
            blunder_to=bl_moves[j].to_square
            d=H[j,best_to]-H[j,blunder_to]  # [512]
            if np.linalg.norm(d)>1e-6:
                diffs.append(d); metas_out.append(batch_meta[j])
    except Exception as e:
        n_errors+=len(boards); print(f"  batch error: {e}")

    done=batch_start+len(batch_idx)
    if done%(BS*25)==0 or done>=len(valid):
        elapsed=time.time()-t0
        rate=done/max(elapsed,1); eta=(len(valid)-done)/rate
        print(f"  {done}/{len(valid)} ({elapsed:.0f}s, {rate:.0f} pos/s, ETA {eta:.0f}s)",
              flush=True)

print(f"\nDone: {len(diffs)} diffs, {n_errors} errors")
activations=np.stack(diffs).astype(np.float32)
mean=activations.mean(0); std=activations.std(0)
torch.save({
    "activations": torch.tensor(activations),
    "mean": mean, "std": std,
    "metadata": metas_out,
    "config": {
        "construction": "h[best_to_sq] - h[blunder_to_sq]",
        "source": "maia3_blunder_diff_v1",
        "n": len(diffs),
        "probe_layer": "/model/transformer/layers.7/Add_2_output_0"
    }
}, OUT)
sz=__import__('os').path.getsize(OUT)/1024/1024
print(f"Saved: {OUT} ({sz:.0f} MB)")
