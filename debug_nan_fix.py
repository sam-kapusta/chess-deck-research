"""Test fixes for ONNX batch NaN issue."""
import sys, json, numpy as np, onnxruntime as ort, onnx, tempfile, os, time
sys.path.insert(0, "scripts")
from maia3_activations import preprocess_fen, MODEL_PATH, PROBE_LAYER

with open("/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_metadata_200k.json") as f:
    metadata = json.load(f)
fens = [m["fen"] for m in metadata[:200]]

# Prepare modified model once
model = onnx.load(str(MODEL_PATH))
probe_output = onnx.helper.make_tensor_value_info(PROBE_LAYER, onnx.TensorProto.FLOAT16, None)
model.graph.output.append(probe_output)
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx.save(model, f.name)
    temp_path = f.name

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# Prepare all tokens
all_tokens = np.stack([preprocess_fen(fens[i])[0] for i in range(200)])
elo = np.full(200, 1500.0, dtype=np.float32)

# FIX 1: New session per batch
print("FIX 1: New session per batch (bs=32)")
t0 = time.time()
results_fix1 = []
for start in range(0, 200, 32):
    end = min(start + 32, 200)
    session = ort.InferenceSession(temp_path, sess_options)
    output_names = [o.name for o in session.get_outputs()]
    feeds = {
        "tokens": all_tokens[start:end],
        "elo_self": elo[start:end],
        "elo_oppo": elo[start:end],
    }
    r = session.run(output_names, feeds)
    results_fix1.append(r[-1])  # probe layer
elapsed1 = time.time() - t0
all_r1 = np.concatenate(results_fix1, axis=0)
print(f"  Time: {elapsed1:.2f}s for 200 positions")
print(f"  NaN: {np.isnan(all_r1).any()}")
print(f"  Throughput: {200/elapsed1:.0f} pos/s")

# FIX 2: batch_size=1 (current workaround)
print("\nFIX 2: batch_size=1 (single session)")
session_single = ort.InferenceSession(temp_path, sess_options)
output_names = [o.name for o in session_single.get_outputs()]
t0 = time.time()
results_fix2 = []
for i in range(200):
    feeds = {
        "tokens": all_tokens[i:i+1],
        "elo_self": elo[i:i+1],
        "elo_oppo": elo[i:i+1],
    }
    r = session_single.run(output_names, feeds)
    results_fix2.append(r[-1])
elapsed2 = time.time() - t0
all_r2 = np.concatenate(results_fix2, axis=0)
print(f"  Time: {elapsed2:.2f}s for 200 positions")
print(f"  NaN: {np.isnan(all_r2).any()}")
print(f"  Throughput: {200/elapsed2:.0f} pos/s")

# FIX 3: Run the UNMODIFIED model (no probe) batched — check if probe is the issue
print("\nFIX 3: Unmodified model batched (no probe output)")
session_orig = ort.InferenceSession(str(MODEL_PATH), sess_options)
orig_output_names = [o.name for o in session_orig.get_outputs()]
t0 = time.time()
nan_found = False
for start in range(0, 200, 32):
    end = min(start + 32, 200)
    feeds = {
        "tokens": all_tokens[start:end],
        "elo_self": elo[start:end],
        "elo_oppo": elo[start:end],
    }
    r = session_orig.run(orig_output_names, feeds)
    if np.isnan(r[0]).any():
        nan_found = True
        print(f"  NaN at batch {start}!")
        break
elapsed3 = time.time() - t0
if not nan_found:
    print(f"  Time: {elapsed3:.2f}s for 200 positions — NO NaN")
    print(f"  → Probe addition causes the NaN issue")

# FIX 4: Run modified model but request ONLY probe output (not all outputs)
print("\nFIX 4: Modified model, request ONLY probe output")
session_probe_only = ort.InferenceSession(temp_path, sess_options)
t0 = time.time()
nan_found = False
for start in range(0, 200, 32):
    end = min(start + 32, 200)
    feeds = {
        "tokens": all_tokens[start:end],
        "elo_self": elo[start:end],
        "elo_oppo": elo[start:end],
    }
    r = session_probe_only.run([PROBE_LAYER], feeds)
    if np.isnan(r[0]).any():
        nan_found = True
        print(f"  NaN at batch {start}!")
        break
elapsed4 = time.time() - t0
if not nan_found:
    print(f"  Time: {elapsed4:.2f}s for 200 positions — NO NaN")
    print(f"  Throughput: {200/elapsed4:.0f} pos/s")
    print(f"  → Requesting only probe output fixes it!")
else:
    print(f"  Still NaN — probe output itself is the issue")

# FIX 5: Modified model, request only probe + one other
print("\nFIX 5: Request probe + logits_move only")
session_two = ort.InferenceSession(temp_path, sess_options)
t0 = time.time()
nan_found = False
for start in range(0, 200, 32):
    end = min(start + 32, 200)
    feeds = {
        "tokens": all_tokens[start:end],
        "elo_self": elo[start:end],
        "elo_oppo": elo[start:end],
    }
    r = session_two.run(["logits_move", PROBE_LAYER], feeds)
    if np.isnan(r[-1]).any():
        nan_found = True
        print(f"  NaN at batch {start}!")
        break
elapsed5 = time.time() - t0
if not nan_found:
    print(f"  Time: {elapsed5:.2f}s — NO NaN")
    print(f"  Throughput: {200/elapsed5:.0f} pos/s")

os.unlink(temp_path)

print("\n" + "="*60)
print("SUMMARY: Best fix for production use")
print("="*60)
