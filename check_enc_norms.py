import torch
import torch.nn.functional as F

for name, path in [
    ("L2 50ep", "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32.pt"),
    ("L2 200ep", "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_l2_200ep.pt"),
    ("No-L2 100ep", "/home/ec2-user/SageMaker/chess-stage-a/output/maia3_sae/maia3_sae_diff_2048_k32_v2.pt"),
]:
    try:
        sae = torch.load(path, map_location="cpu", weights_only=False)
        state = sae["state_dict"]
        enc_norms = state["W_enc"].norm(dim=0)
        b_enc = state["b_enc"]
        print(f"{name}:")
        print(f"  W_enc norms: mean={enc_norms.mean():.3f}, max={enc_norms.max():.3f}, min={enc_norms.min():.3f}")
        print(f"  b_enc: mean={b_enc.mean():.4f}, range=[{b_enc.min():.4f}, {b_enc.max():.4f}]")

        # Check activation range on a sample
        data = torch.load("/home/ec2-user/SageMaker/chess-stage-a/cache/maia3_blunder_diff.pt", map_location="cpu", weights_only=False)
        x = data["activations"][:4096].float()
        mean_v = x.mean(dim=0)
        std_v = x.std(dim=0).clamp(min=1e-6)
        x = (x - mean_v) / std_v
        if "v2" not in path:  # L2 versions
            norms_x = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            x = x / norms_x

        z = (x - state["b_dec"]) @ state["W_enc"] + state["b_enc"]
        z_relu = F.relu(z)
        topk_vals, topk_idx = torch.topk(z_relu, k=32, dim=-1)
        print(f"  Activation max: {topk_vals.max():.3f}")
        print(f"  Activation mean: {topk_vals.mean():.3f}")
        print(f"  >1.0: {(topk_vals > 1.0).sum().item()}")
        print(f"  >5.0: {(topk_vals > 5.0).sum().item()}")
        print()
    except Exception as e:
        print(f"{name}: {e}\n")
