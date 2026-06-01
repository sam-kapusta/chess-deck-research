"""Calibrate inference threshold using k-th percentile approach.

For small corpora, the paper's min-positive formula collapses to ~0.
Instead: compute the k-th largest pre-threshold activation per position,
then take the mean across positions. This gives a threshold where
mean L0 ≈ k at inference.
"""
import json, numpy as np, torch, torch.nn.functional as F, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

B = "/home/ec2-user/SageMaker"
WPATH = f"{B}/chess-stage-a/output/maia3_sae/btk_2048_k16_v2_weights.pt"
CPATH = f"{B}/chess-stage-a/cache/maia3_l7only_v2_dedup.pt"

class SAE(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d, h))
        self.W_dec = nn.Parameter(torch.empty(h, d))
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("num_batches_not_active", torch.zeros(h))
    def forward(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

def normalize(raw, mean, std):
    x = (raw - mean) / std
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

ckpt = torch.load(WPATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]; cfg = ckpt["config"]
k = cfg["k"]
d_input, dict_size = cfg["d_input"], cfg["dict_size"]
ns = json.load(open(WPATH.replace(".pt", "_stats.json")))
mean = torch.tensor(ns["mean"]); std = torch.tensor(ns["std"])
raw = torch.load(CPATH, map_location="cpu", weights_only=False)["activations"].float()
x_norm = normalize(raw, mean, std); del raw

m = SAE(d_input, dict_size); m.load_state_dict(sd, strict=False); m.eval()

# k-th largest activation per position → average = threshold where L0 ≈ k
kth_vals = []
with torch.no_grad():
    for (batch,) in DataLoader(TensorDataset(x_norm), batch_size=4096, shuffle=False):
        z = m(batch)
        # top-k values per row; the k-th is the cutoff for that position
        topk = torch.topk(z, k, dim=1).values[:, -1]  # shape [B]
        kth_vals.append(topk.numpy())

kth = np.concatenate(kth_vals)
theta = float(np.mean(kth))
print(f"k={k} | mean k-th activation = {theta:.6f}")
print(f"  p5={np.percentile(kth,5):.6f}  p50={np.percentile(kth,50):.6f}  p95={np.percentile(kth,95):.6f}")

# Verify L0 at this threshold
l0s = []
with torch.no_grad():
    for (batch,) in DataLoader(TensorDataset(x_norm[:8192]), batch_size=4096, shuffle=False):
        z = m(batch)
        acts = z * (z > theta)
        l0s.append((acts > 0).sum(1).float().numpy())
mean_l0 = float(np.concatenate(l0s).mean())
print(f"Mean L0 at threshold: {mean_l0:.1f}  (target: {k})")

out = {"global_threshold": theta, "mean_l0": mean_l0, "k": k,
       "dict_size": dict_size, "method": "mean_kth_largest"}
path = WPATH.replace("_weights.pt", "_calibration.json")
json.dump(out, open(path, "w"), indent=2)
print(f"Saved to {path}")
