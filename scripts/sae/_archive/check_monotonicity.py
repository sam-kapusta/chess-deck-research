"""Check if activation strength is monotonically related to theme match."""
import torch
import torch.nn.functional as F

sae = torch.load("/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_raw.pt", map_location="cpu", weights_only=False)
data = torch.load("/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt", map_location="cpu", weights_only=False)
state = sae["state_dict"]
x = data["activations"].float()
metadata = data["metadata"]

z = (x - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
z_relu = F.relu(z)
topk_vals, topk_idx = torch.topk(z_relu, k=32, dim=-1)
acts = torch.zeros_like(z_relu)
acts.scatter_(-1, topk_idx, topk_vals)

# Check a few features at different activation levels
for feat_id in [500, 830, 1536, 1890]:
    feat = acts[:, feat_id]
    nonzero_idx = (feat > 0).nonzero(as_tuple=True)[0]
    sorted_idx = nonzero_idx[feat[nonzero_idx].argsort(descending=True)]
    n = len(sorted_idx)
    max_val = feat[sorted_idx[0]].item()

    print(f"\nFeature {feat_id}: {n} positions, max={max_val:.2f}")

    # Top 5
    print("  TOP (>80% of max):")
    for i in sorted_idx[:5]:
        m = metadata[i.item()]
        rel = feat[i].item() / max_val
        fen = m["fen"][:50]
        uci = m.get("blunder_uci", "")
        cp = m.get("cp_loss", 0)
        print(f"    [{rel:.2f}] {fen}  {uci}  cp={cp}")

    # Middle
    mid = int(n * 0.5)
    print("  MIDDLE (50th percentile):")
    for i in sorted_idx[mid:mid+5]:
        m = metadata[i.item()]
        rel = feat[i].item() / max_val
        fen = m["fen"][:50]
        uci = m.get("blunder_uci", "")
        cp = m.get("cp_loss", 0)
        print(f"    [{rel:.2f}] {fen}  {uci}  cp={cp}")

    # Bottom
    print("  BOTTOM (lowest activations):")
    for i in sorted_idx[-5:]:
        m = metadata[i.item()]
        rel = feat[i].item() / max_val
        fen = m["fen"][:50]
        uci = m.get("blunder_uci", "")
        cp = m.get("cp_loss", 0)
        print(f"    [{rel:.2f}] {fen}  {uci}  cp={cp}")
