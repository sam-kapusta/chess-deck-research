#!/usr/bin/env python3
"""Phi-coefficient clustering on v2 k=16 SAE features.

Computes pairwise phi coefficients (binary co-occurrence) between all
feature pairs, then hierarchically clusters into categories and sub-clusters.
Same methodology as Sandstone persona_tree phi_clustering.

Output: JSON tree with categories -> subclusters -> features, ready for
Opus labeling and HTML visualization.

Usage (on chess-poc):
    cd ~/SageMaker && python3 scripts/sae/phi_clustering_v2.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

BASE = "/home/ec2-user/SageMaker/chess-stage-a"
OUTPUT = BASE + "/output/maia3_sae_v2"
MODEL_PATH = OUTPUT + "/sweep_v2_k16_d2048.pt"
ACTIVATIONS_PATH = BASE + "/cache/maia3_blunder_diff_v2.pt"
os.makedirs(OUTPUT, exist_ok=True)


def compute_phi_matrix(binary_acts):
    """Compute pairwise phi coefficient matrix.

    phi(i,j) = (n11*n00 - n10*n01) / sqrt(n1.*n0.*n.1*n.0)
    where n11 = both fire, n00 = neither fires, etc.
    """
    n = binary_acts.shape[0]
    d = binary_acts.shape[1]

    # Compute using matrix operations for speed
    # n11 = A^T @ A (co-occurrence count)
    A = binary_acts.astype(np.float32)
    n11 = A.T @ A

    # Marginals
    n1_dot = A.sum(axis=0)  # how often each feature fires
    n0_dot = n - n1_dot

    # n11 matrix gives co-fire counts
    # n10[i,j] = feature i fires, j doesn't = n1_dot[i] - n11[i,j]
    # n01[i,j] = i doesn't fire, j fires = n1_dot[j] - n11[i,j]
    # n00[i,j] = neither fires = n - n1_dot[i] - n1_dot[j] + n11[i,j]

    n1i = n1_dot.reshape(-1, 1)
    n1j = n1_dot.reshape(1, -1)

    numerator = n11 * (n - n1i - n1j + n11) - (n1i - n11) * (n1j - n11)
    denominator = np.sqrt(n1i * (n - n1i) * n1j * (n - n1j))
    denominator = np.maximum(denominator, 1e-10)

    phi = numerator / denominator
    np.fill_diagonal(phi, 1.0)

    return phi


def hierarchical_cluster(phi_matrix, n_top=25, target_subcluster_size=7):
    """Two-level hierarchical clustering.

    Level 1: ~n_top broad categories
    Level 2: subclusters within each category (~target_subcluster_size features each)
    """
    # Convert phi to distance (1 - phi, clipped to [0, 2])
    dist = 1.0 - phi_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)

    # Make symmetric (numerical precision)
    dist = (dist + dist.T) / 2

    # Condensed distance matrix for scipy
    condensed = squareform(dist, checks=False)

    # Ward linkage
    Z = linkage(condensed, method='ward')

    # Level 1: cut into n_top clusters
    top_labels = fcluster(Z, t=n_top, criterion='maxclust')

    # Level 2: sub-cluster within each top cluster
    tree = {}
    for cat_id in range(1, n_top + 1):
        members = np.where(top_labels == cat_id)[0]
        if len(members) <= target_subcluster_size:
            # Too small to sub-cluster
            tree[cat_id] = [members.tolist()]
        else:
            # Sub-cluster this group
            n_subclusters = max(2, len(members) // target_subcluster_size)
            sub_dist = dist[np.ix_(members, members)]
            sub_condensed = squareform(sub_dist, checks=False)
            sub_Z = linkage(sub_condensed, method='ward')
            sub_labels = fcluster(sub_Z, t=n_subclusters, criterion='maxclust')

            subclusters = []
            for sub_id in range(1, n_subclusters + 1):
                sub_members = members[sub_labels == sub_id]
                subclusters.append(sub_members.tolist())
            tree[cat_id] = subclusters

    return tree, top_labels


def main():
    print("=" * 60)
    print("PHI-COEFFICIENT CLUSTERING ON V2 k=16 SAE")
    print("=" * 60)
    sys.stdout.flush()

    # Load model
    print("Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    We = sd["W_enc"]
    Wd = sd["W_dec"]
    be = sd["b_enc"]
    bd = sd["b_dec"]

    # Load activations
    print("Loading v2 activations...")
    data = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    raw = data["activations"].float()
    meta = data["metadata"]

    # Normalize
    mean = raw.mean(dim=0)
    std = raw.std(dim=0).clamp(min=1e-6)
    x = (raw - mean) / std
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x = x / norms
    del raw

    # Forward pass
    print("Running forward pass (k=16)...")
    sys.stdout.flush()
    z = F.relu((x - bd) @ We + be)
    n = x.shape[0]
    flat = z.reshape(-1)
    k = 16
    topk_vals, topk_idx = torch.topk(flat, k=min(n * k, flat.numel()))
    acts = torch.zeros_like(flat)
    acts[topk_idx] = topk_vals
    acts = acts.reshape(z.shape)

    # Binary activations
    binary = (acts > 0).numpy().astype(np.float32)
    print(f"  Activations shape: {binary.shape}")
    print(f"  Mean L0: {binary.sum(axis=1).mean():.1f}")
    sys.stdout.flush()

    # Filter dead/rare features
    fire_rates = binary.mean(axis=0)
    alive = fire_rates > 0.001  # at least 0.1% fire rate
    alive_idx = np.where(alive)[0]
    print(f"  Alive features (>0.1%): {len(alive_idx)}/{binary.shape[1]}")
    sys.stdout.flush()

    binary_alive = binary[:, alive_idx]

    # Compute phi matrix
    print("Computing phi coefficient matrix...")
    t0 = time.time()
    phi = compute_phi_matrix(binary_alive)
    print(f"  Phi matrix: {phi.shape}, computed in {time.time()-t0:.1f}s")
    print(f"  Mean phi: {phi[np.triu_indices_from(phi, k=1)].mean():.4f}")
    print(f"  Max phi (off-diag): {phi[np.triu_indices_from(phi, k=1)].max():.4f}")
    sys.stdout.flush()

    # Cluster
    print("Hierarchical clustering...")
    tree, top_labels = hierarchical_cluster(phi, n_top=25, target_subcluster_size=7)

    # Build output structure (matching persona tree format)
    print("Building tree structure...")
    output = {
        "method": "phi_coefficient_ward",
        "n_positions": int(n),
        "n_features_total": int(binary.shape[1]),
        "n_features_alive": int(len(alive_idx)),
        "n_categories": len(tree),
        "categories": [],
    }

    total_subclusters = 0
    for cat_id, subclusters in sorted(tree.items()):
        cat_features = []
        for sc in subclusters:
            cat_features.extend(sc)

        # Category-level stats
        cat_fire_rates = fire_rates[alive_idx[cat_features]]

        cat_data = {
            "id": int(cat_id),
            "n_features": len(cat_features),
            "mean_fire_rate": float(cat_fire_rates.mean()),
            "max_fire_rate": float(cat_fire_rates.max()),
            "subclusters": [],
        }

        for sc_idx, sc_members in enumerate(subclusters):
            total_subclusters += 1
            sc_features = []
            for feat_local_idx in sc_members:
                global_idx = int(alive_idx[feat_local_idx])
                sc_features.append({
                    "feature_id": global_idx,
                    "fire_rate": float(fire_rates[global_idx]),
                })

            # Mean phi within subcluster
            if len(sc_members) > 1:
                sc_phi_vals = []
                for i in range(len(sc_members)):
                    for j in range(i + 1, len(sc_members)):
                        sc_phi_vals.append(phi[sc_members[i], sc_members[j]])
                mean_internal_phi = float(np.mean(sc_phi_vals))
            else:
                mean_internal_phi = 1.0

            cat_data["subclusters"].append({
                "id": sc_idx,
                "n_features": len(sc_members),
                "features": sc_features,
                "mean_internal_phi": mean_internal_phi,
            })

        output["categories"].append(cat_data)

    output["n_subclusters"] = total_subclusters
    print(f"  {len(tree)} categories, {total_subclusters} subclusters")
    sys.stdout.flush()

    # Save phi matrix for later use
    np.save(f"{OUTPUT}/phi_matrix_v2.npy", phi)
    print(f"  Saved phi matrix: {OUTPUT}/phi_matrix_v2.npy")

    # Save tree
    tree_path = f"{OUTPUT}/phi_tree_v2.json"
    with open(tree_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved tree: {tree_path}")

    # Also save feature->category mapping for quick lookup
    feat_to_cat = {}
    for cat in output["categories"]:
        for sc in cat["subclusters"]:
            for feat in sc["features"]:
                feat_to_cat[feat["feature_id"]] = {
                    "category": cat["id"],
                    "subcluster": sc["id"],
                }
    with open(f"{OUTPUT}/feat_to_category_v2.json", "w") as f:
        json.dump(feat_to_cat, f)

    # Summary stats
    print("\n=== CATEGORY SUMMARY ===")
    for cat in sorted(output["categories"], key=lambda c: -c["n_features"]):
        print(f"  Cat {cat['id']:2d}: {cat['n_features']:3d} features, "
              f"fire={cat['mean_fire_rate']*100:.2f}%, "
              f"{len(cat['subclusters'])} subclusters")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
