#!/usr/bin/env python3
"""Train k=8 SAE variants to find optimal dict_size and aux_loss setting.

Configs:
  A: 2048 k=8 + aux
  B: 2048 k=8 no aux
  C: 1024 k=8 no aux
  D: 512 k=8 no aux

Compare: dead features, alive features, FVU, fire rates, label specificity.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, di, dd, k):
        super().__init__()
        self.encoder = nn.Linear(di, dd)
        self.decoder = nn.Linear(dd, di, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(di))
        self.k = k
        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        z = self.encoder(x - self.pre_bias)
        tv, ti = torch.topk(z, self.k, dim=-1)
        a = torch.zeros(x.shape[0], self.encoder.out_features, device=x.device)
        a.scatter_(-1, ti, F.relu(tv))
        return self.decoder(a) + self.pre_bias, a, z


def train_sae(data, dict_size, k, use_aux, epochs=10, batch_size=256, lr=3e-4,
              aux_coeff=1/32, dead_threshold=50):
    """Train a BatchTopK SAE. Returns model + metrics."""
    n, di = data.shape
    device = data.device
    sae = SAE(di, dict_size, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    fire_counts = torch.zeros(dict_size, device=device)
    steps_since_reset = 0

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_mse = 0
        epoch_aux = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = data[idx]

            recon, acts, pre_topk = sae(x)
            mse = F.mse_loss(recon, x)

            loss = mse
            aux_loss_val = 0

            if use_aux:
                # Aux loss: encourage dead features to activate
                steps_since_reset += 1
                fire_counts += (acts > 0).float().sum(dim=0)

                if steps_since_reset > dead_threshold:
                    dead_mask = fire_counts < 1
                    if dead_mask.any():
                        dead_pre = pre_topk[:, dead_mask]
                        aux_loss_val = F.relu(dead_pre).mean()
                        loss = mse + aux_coeff * aux_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

            epoch_mse += mse.item()
            epoch_aux += aux_loss_val if isinstance(aux_loss_val, (int, float)) else aux_loss_val.item()
            n_batches += 1

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f'  ep{epoch}: mse={epoch_mse/n_batches:.4f} aux={epoch_aux/n_batches:.4f}')

    return sae


def evaluate(sae, data, name):
    """Compute structural metrics."""
    n = data.shape[0]
    dd = sae.encoder.out_features

    with torch.no_grad():
        recon, acts, pre_topk = sae(data)

    acts_np = acts.cpu().numpy()
    fires = (acts_np > 0).astype(np.float32)

    # Metrics
    fire_per_feature = fires.sum(axis=0)
    dead = (fire_per_feature == 0).sum()
    alive = dd - dead

    fvu = F.mse_loss(recon, data).item() / data.var().item()

    fire_rates = fire_per_feature / n * 100
    alive_rates = fire_rates[fire_rates > 0]

    # L0
    l0 = fires.sum(axis=1).mean()

    # Decoder cosine sim
    dec = sae.decoder.weight.data.cpu().numpy()  # (di, dd)
    dec_norm = dec / (np.linalg.norm(dec, axis=0, keepdims=True) + 1e-8)
    cos_sim = np.abs(dec_norm.T @ dec_norm)
    np.fill_diagonal(cos_sim, 0)
    c_dec = cos_sim.mean()

    # Per-position: how many features fire
    features_per_pos = fires.sum(axis=1)

    # Theme coverage: pre-topk energy in top-k
    pre_np = pre_topk.cpu().numpy()
    pre_pos = np.maximum(pre_np, 0)
    total_e = pre_pos.sum(axis=1, keepdims=True)
    total_e = np.maximum(total_e, 1e-8)
    sorted_pre = np.sort(pre_pos, axis=1)[:, ::-1]
    cum = np.cumsum(sorted_pre, axis=1) / total_e
    energy_captured = cum[:, sae.k - 1].mean() * 100

    print(f'\n=== {name} (dict={dd}, k={sae.k}) ===')
    print(f'  Dead: {dead}/{dd} ({dead/dd*100:.1f}%)')
    print(f'  Alive: {alive}')
    print(f'  FVU: {fvu:.4f}')
    print(f'  L0: {l0:.1f}')
    print(f'  c_dec: {c_dec:.4f}')
    print(f'  Energy captured: {energy_captured:.1f}%')
    print(f'  Fire rate (alive): mean={alive_rates.mean():.2f}%, median={np.median(alive_rates):.2f}%')
    print(f'  Features/position: mean={features_per_pos.mean():.1f}, min={features_per_pos.min():.0f}, max={features_per_pos.max():.0f}')

    return {
        'name': name, 'dict_size': dd, 'k': sae.k,
        'dead': int(dead), 'alive': int(alive),
        'fvu': round(fvu, 4), 'l0': round(float(l0), 1),
        'c_dec': round(float(c_dec), 4),
        'energy_pct': round(energy_captured, 1),
        'fire_rate_mean': round(float(alive_rates.mean()), 2),
        'fire_rate_median': round(float(np.median(alive_rates)), 2),
    }


def main():
    print('Training k=8 SAE sweep: 2048+aux, 2048-aux, 1024-aux, 512-aux')
    print()

    cache = torch.load('/home/ec2-user/SageMaker/chess-stage-a/cache/blunder_move_token_200k.pt',
                        map_location='cpu', weights_only=False)
    data = cache['blunder_mt'][:200000].float()
    print(f'Data: {data.shape[0]} positions, {data.shape[1]} dims')

    # Normalize
    mean = data.mean(dim=0)
    std = data.std(dim=0).clamp(min=1e-8)
    normalized = (data - mean) / std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    normalized = normalized.to(device)

    configs = [
        ('A: 2048 k=8 +aux', 2048, 8, True),
        ('B: 2048 k=8 -aux', 2048, 8, False),
        ('C: 1024 k=8 -aux', 1024, 8, False),
        ('D: 512 k=8 -aux',  512,  8, False),
    ]

    results = []
    models = {}
    for name, dd, k, aux in configs:
        print(f'\n--- Training {name} ---')
        t0 = time.time()
        sae = train_sae(normalized, dd, k, aux, epochs=10)
        elapsed = time.time() - t0
        print(f'  Trained in {elapsed:.1f}s')

        # Evaluate on subset
        eval_data = normalized[:10000]
        metrics = evaluate(sae, eval_data, name)
        metrics['train_time'] = round(elapsed, 1)
        results.append(metrics)
        models[name] = sae

        # Save checkpoint
        save_name = f'sae_btk_blunder_mt_{dd}_k{k}{"_aux" if aux else ""}.pt'
        save_path = f'/home/ec2-user/SageMaker/chess-stage-a/output/blunder_sae/{save_name}'
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': {'dict_size': dd, 'k': k, 'aux': aux},
            'mean': mean.tolist(),
            'std': std.tolist(),
            'metrics': metrics,
        }, save_path)
        print(f'  Saved: {save_path}')

    # Comparison table
    print('\n\n=== COMPARISON ===')
    print(f'{"Config":<25} {"Dead":>6} {"Alive":>6} {"FVU":>7} {"c_dec":>7} {"Energy":>7} {"FR mean":>8} {"FR med":>8}')
    print('-' * 85)
    for r in results:
        print(f'{r["name"]:<25} {r["dead"]:>6} {r["alive"]:>6} {r["fvu"]:>7.4f} {r["c_dec"]:>7.4f} {r["energy_pct"]:>6.1f}% {r["fire_rate_mean"]:>7.2f}% {r["fire_rate_median"]:>7.2f}%')

    # === Theme analysis: do the 16 categories survive in each variant? ===
    # We need labels — but these are new SAEs without labels.
    # Instead, profile the top-20 features from each model and check diversity.
    print('\n\n=== FEATURE DIVERSITY: Top features per model ===')
    for name, dd, k, aux in configs:
        sae = models[f'{name}']
        eval_data = normalized[:10000]
        with torch.no_grad():
            _, acts, _ = sae(eval_data)
        fires = (acts > 0).cpu().numpy().astype(np.float32)

        # Top 20 features by fire count
        fire_counts = fires.sum(axis=0)
        top20 = np.argsort(-fire_counts)[:20]

        # How concentrated is the firing? (Gini-like)
        alive_counts = fire_counts[fire_counts > 0]
        alive_counts_sorted = np.sort(alive_counts)[::-1]
        cumsum = np.cumsum(alive_counts_sorted)
        total = cumsum[-1]

        # How many features needed for 50% / 80% of all fires?
        n_for_50 = np.searchsorted(cumsum, total * 0.5) + 1
        n_for_80 = np.searchsorted(cumsum, total * 0.8) + 1

        # Fire rate distribution
        alive_rates = alive_counts / 10000 * 100

        print(f'\n  {name}:')
        print(f'    Features for 50% of fires: {n_for_50} (of {len(alive_counts)} alive)')
        print(f'    Features for 80% of fires: {n_for_80}')
        print(f'    Fire rate distribution: p10={np.percentile(alive_rates,10):.2f}% p50={np.percentile(alive_rates,50):.2f}% p90={np.percentile(alive_rates,90):.2f}%')
        print(f'    Top 5 fire rates: {", ".join(f"{r:.1f}%" for r in sorted(alive_rates, reverse=True)[:5])}')

        # Position uniqueness: what fraction of positions have a UNIQUE top-1 feature?
        top1 = np.argmax(acts.cpu().numpy(), axis=1)
        n_unique_top1 = len(set(top1))
        print(f'    Unique top-1 features used: {n_unique_top1}/{dd}')

    # Save results
    with open('/home/ec2-user/SageMaker/chess-deck-research/output/k8_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved results to output/k8_sweep_results.json')


if __name__ == '__main__':
    main()
