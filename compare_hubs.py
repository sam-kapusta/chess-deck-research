"""Compare hub features between L2 and no-L2 SAE versions."""
import torch
import torch.nn.functional as F
import numpy as np

DATA_PATH = "/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt"

data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
x_raw = data["activations"].float()
mean = x_raw.mean(dim=0)
std = x_raw.std(dim=0).clamp(min=1e-6)
x_zscore = (x_raw - mean) / std

# L2 version input
norms = x_zscore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
x_l2 = x_zscore / norms

for name, sae_path, x_input in [
    ("L2 200ep", "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_l2_200ep.pt", x_l2),
    ("No-L2 100ep", "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_v2.pt", x_zscore),
]:
    sae = torch.load(sae_path, map_location="cpu", weights_only=False)
    state = sae["state_dict"]

    # Run forward on full dataset
    z = (x_input - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
    z_relu = F.relu(z)

    # Per-sample top-k (for fair fire rate comparison)
    k = 32
    topk_vals, topk_idx = torch.topk(z_relu, k=k, dim=-1)
    acts = torch.zeros_like(z_relu)
    acts.scatter_(-1, topk_idx, topk_vals)

    fire_rates = (acts > 0).float().mean(dim=0).numpy()
    fire_rates_sorted = np.sort(fire_rates)[::-1]

    n_hub_10 = (fire_rates > 0.10).sum()
    n_hub_5 = (fire_rates > 0.05).sum()
    n_good = ((fire_rates >= 0.005) & (fire_rates <= 0.03)).sum()
    gini = np.sum(np.abs(fire_rates[:, None] - fire_rates[None, :])) / (2 * len(fire_rates) * fire_rates.sum())

    print(f"\n{name}:")
    print(f"  Hubs (>10%): {n_hub_10}")
    print(f"  Hubs (>5%): {n_hub_5}")
    print(f"  Good range (0.5-3%): {n_good}")
    print(f"  Top 10 fire rates: {[f'{r:.3f}' for r in fire_rates_sorted[:10]]}")
    print(f"  Gini coefficient: {gini:.4f} (0=perfectly uniform, 1=one feature dominates)")
    print(f"  Fire rate std: {fire_rates.std():.5f}")
    print(f"  Max/median ratio: {fire_rates_sorted[0]/np.median(fire_rates[fire_rates>0]):.1f}x")
