"""Debug NaN in batched ONNX inference."""
import sys, json, numpy as np, onnxruntime as ort, onnx, tempfile, os
sys.path.insert(0, "scripts")
from maia3_activations import preprocess_fen, MODEL_PATH, PROBE_LAYER

with open("/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_metadata_200k.json") as f:
    metadata = json.load(f)
fens = [m["fen"] for m in metadata[:64]]

# Create modified model exactly as the script does
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
model = onnx.load(str(MODEL_PATH))
probe_output = onnx.helper.make_tensor_value_info(PROBE_LAYER, onnx.TensorProto.FLOAT16, None)
model.graph.output.append(probe_output)
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx.save(model, f.name)
    temp_path = f.name
session = ort.InferenceSession(temp_path, sess_options)

output_names = [o.name for o in session.get_outputs()]
print("Outputs:", output_names)
print()

# Batch 1
tokens1 = np.stack([preprocess_fen(fens[i])[0] for i in range(32)])
elo1 = np.full(32, 1500.0, dtype=np.float32)
feeds1 = {"tokens": tokens1, "elo_self": elo1, "elo_oppo": elo1}
results1 = session.run(output_names, feeds1)
print("Batch 1 (positions 0-31):")
for name, r in zip(output_names, results1):
    has_nan = np.isnan(r).any()
    print(f"  {name}: shape={r.shape}, NaN={has_nan}")

# Batch 2: SAME DATA as batch 1 (tests if session state corrupts)
print("\nBatch 2 (SAME data as batch 1, same session):")
results2 = session.run(output_names, feeds1)
for name, r in zip(output_names, results2):
    has_nan = np.isnan(r).any()
    print(f"  {name}: shape={r.shape}, NaN={has_nan}")

# Batch 3: different data, same session
tokens3 = np.stack([preprocess_fen(fens[i])[0] for i in range(32, 64)])
elo3 = np.full(32, 1500.0, dtype=np.float32)
feeds3 = {"tokens": tokens3, "elo_self": elo3, "elo_oppo": elo3}
print("\nBatch 3 (positions 32-63, same session):")
results3 = session.run(output_names, feeds3)
for name, r in zip(output_names, results3):
    has_nan = np.isnan(r).any()
    print(f"  {name}: shape={r.shape}, NaN={has_nan}")

# NEW SESSION for the same positions 32-63
print("\n--- NEW SESSION for positions 32-63 ---")
session2 = ort.InferenceSession(temp_path, sess_options)
results4 = session2.run(output_names, feeds3)
for name, r in zip(output_names, results4):
    has_nan = np.isnan(r).any()
    print(f"  {name}: shape={r.shape}, NaN={has_nan}")

# Single position #32 with ORIGINAL corrupted session
print("\n--- Single position #32, original session (batch=1) ---")
tokens_single = preprocess_fen(fens[32])[0][np.newaxis, ...]
elo_single = np.array([1500.0], dtype=np.float32)
feeds_single = {"tokens": tokens_single, "elo_self": elo_single, "elo_oppo": elo_single}
results5 = session.run(output_names, feeds_single)
for name, r in zip(output_names, results5):
    has_nan = np.isnan(r).any()
    print(f"  {name}: shape={r.shape}, NaN={has_nan}")

# Test: does enabling optimization help?
print("\n--- WITH graph optimization ENABLED ---")
sess_options2 = ort.SessionOptions()
session3 = ort.InferenceSession(temp_path, sess_options2)
print("Batch 1:")
r = session3.run(output_names, feeds1)
print(f"  probe NaN: {np.isnan(r[-1]).any()}")
print("Batch 2 (same data):")
r = session3.run(output_names, feeds1)
print(f"  probe NaN: {np.isnan(r[-1]).any()}")
print("Batch 3 (different data):")
r = session3.run(output_names, feeds3)
print(f"  probe NaN: {np.isnan(r[-1]).any()}")

# Test: smaller batch sizes
print("\n--- Batch size sweep ---")
for bs in [1, 2, 4, 8, 16, 32]:
    sess_new = ort.InferenceSession(temp_path, sess_options)
    all_nan = False
    for start in range(0, 64, bs):
        end = min(start + bs, 64)
        tokens = np.stack([preprocess_fen(fens[i])[0] for i in range(start, end)])
        elo = np.full(end - start, 1500.0, dtype=np.float32)
        feeds = {"tokens": tokens, "elo_self": elo, "elo_oppo": elo}
        r = sess_new.run(output_names, feeds)
        if np.isnan(r[-1]).any():
            all_nan = True
            print(f"  bs={bs}: NaN at batch starting at position {start}")
            break
    if not all_nan:
        print(f"  bs={bs}: ALL CLEAN (64 positions, no NaN)")

os.unlink(temp_path)
