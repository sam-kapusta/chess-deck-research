# WARNING: built on v1 cache (label-inversion bug). Must repoint to v2 cache before reuse.
"""
Build board_diff_BOTH cache: mean64(h_after_best - h_after_blunder)
Uses v1 positions (200k, all have best_uci + elo_self/elo_oppo).
3 board encodes per position: before, after_blunder, after_best.
Batched at bs=64 (3x encode per batch so reduced from 128).
"""
import onnxruntime as ort, numpy as np, torch, chess, time, sys
BASE="/home/ec2-user/SageMaker/chess-stage-a"
OUT=BASE+"/cache/maia3_board_diff_both.pt"

v1=torch.load(BASE+"/cache/maia3_blunder_diff.pt",map_location="cpu",weights_only=False)
meta=v1["metadata"]; elo_self=v1["elo_self"]; elo_oppo=v1["elo_oppo"]
print(f"Loaded {len(meta)} positions")

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
    tokens=np.stack([tok(b) for b in boards])
    out=sess.run([PROBE],{"tokens":tokens,
        "elo_self":np.array(es,dtype=np.float32),
        "elo_oppo":np.array(eo,dtype=np.float32)})[0]
    return out.astype(np.float32)  # [N, 64, 512]

BS=64  # 3x encodes per position so smaller batch
diffs=[]; metas_out=[]; n_errors=0
t0=time.time()

valid=[i for i,m in enumerate(meta) if m.get("best_uci")]
print(f"Processing {len(valid)} positions at bs={BS} (3 encodes each)...")

for batch_start in range(0, len(valid), BS):
    batch_idx=valid[batch_start:batch_start+BS]
    b0s=[]; bl_bs=[]; bs_bs=[]; es=[]; eo=[]; batch_meta=[]

    for i in batch_idx:
        m=meta[i]
        try:
            b0=chess.Board(m["fen"])
            bl_b=b0.copy(); bl_b.push(chess.Move.from_uci(m["blunder_uci"]))
            bs_b=b0.copy(); bs_b.push(chess.Move.from_uci(m["best_uci"]))
            b0s.append(b0); bl_bs.append(bl_b); bs_bs.append(bs_b)
            es.append(float(elo_self[i])); eo.append(float(elo_oppo[i]))
            batch_meta.append({"fen":m["fen"],
                               "blunder_uci":m["blunder_uci"],
                               "best_uci":m["best_uci"],
                               "cp_loss":m["cp_loss"],
                               "elo_self":float(elo_self[i]),
                               "elo_oppo":float(elo_oppo[i])})
        except: n_errors+=1; continue

    if not b0s: continue
    try:
        H_bl=encode_batch(bl_bs, es, eo)  # [N,64,512] after blunder
        H_bs=encode_batch(bs_bs, es, eo)  # [N,64,512] after best
        for j in range(len(b0s)):
            # mean64(after_best - after_blunder)
            d=(H_bs[j]-H_bl[j]).mean(0)  # [512]
            if np.linalg.norm(d)>1e-6:
                diffs.append(d); metas_out.append(batch_meta[j])
    except Exception as e:
        n_errors+=len(b0s)

    done=batch_start+len(batch_idx)
    if done%(BS*30)==0 or done>=len(valid):
        elapsed=time.time()-t0
        rate=done/max(elapsed,1); eta=(len(valid)-done)/rate
        print(f"  {done}/{len(valid)} ({elapsed:.0f}s, {rate:.0f} pos/s, ETA {eta:.0f}s)",
              flush=True)

print(f"\nDone: {len(diffs)} diffs, {n_errors} errors")
activations=np.stack(diffs).astype(np.float32)
mean=activations.mean(0); std=activations.std(0)
torch.save({"activations":torch.tensor(activations),
            "mean":mean,"std":std,"metadata":metas_out,
            "config":{"construction":"mean64(h_after_best - h_after_blunder)",
                      "source":"v1","n":len(diffs)}}, OUT)
print(f"Saved: {OUT} ({__import__('os').path.getsize(OUT)//1024//1024}MB)")
