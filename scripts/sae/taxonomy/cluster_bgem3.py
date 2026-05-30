"""Semantic clustering baseline (persona method) for the chess SAE taxonomy.

Embeds each feature's label-text (chip + description) with BAAI/bge-m3, then
agglomerative-clusters into fine sub-clusters. This is the BASELINE — categories
emerge bottom-up, replacing the broken top-down per-feature assignment.

Two-level output:
  - fine sub-clusters (~N/8 features each) = sub-categories
  - coarse clusters = candidate top-level categories

Run locally (bge-m3 is cached in ~/.cache/huggingface).
    python3 scripts/sae/taxonomy/cluster_bgem3.py \
        --input output/taxonomy_v2/cluster_input.json \
        --out-embeddings output/taxonomy_v2/embeddings.npy \
        --out-clusters output/taxonomy_v2/clusters.json
"""
import argparse
import json

import numpy as np


def embed(texts, model_name="BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return m.encode(texts, batch_size=32, show_progress_bar=True,
                    normalize_embeddings=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-embeddings", required=True)
    ap.add_argument("--out-clusters", required=True)
    ap.add_argument("--n-fine", type=int, default=240, help="fine sub-clusters")
    ap.add_argument("--n-coarse", type=int, default=18, help="coarse candidate categories")
    args = ap.parse_args()

    import re
    rows = json.load(open(args.input))
    fids = sorted(rows, key=lambda k: int(k))

    # CONCEPT embedding: embed the MECHANISM (first ~2 sentences of description)
    # with piece-nouns neutralized to "piece". Embedding the chip (which leads with
    # the piece name) clusters by piece, not by mistake-concept — the persona
    # method's "shared vocabulary, different concept" failure mode. Stripping the
    # piece exposes the coaching concept (hang / abandon-defense / greedy / king-walk)
    # so categories cut ACROSS pieces. Verified 2026-05-30: piece-based -> concept-based,
    # largest category 13% -> 10%.
    _PIECES = re.compile(r"\b(pawns?|knights?|bishops?|rooks?|queens?|kings?|"
                         r"kingside|queenside)\b", re.I)
    texts = []
    for f in fids:
        mechanism = ". ".join(rows[f]["description"].split(". ")[:2])
        texts.append(_PIECES.sub("piece", mechanism))

    import os
    if os.path.exists(args.out_embeddings):
        emb = np.load(args.out_embeddings)
        print(f"loaded cached embeddings {emb.shape}")
    else:
        print(f"embedding {len(texts)} features with bge-m3...")
        emb = embed(texts)
        np.save(args.out_embeddings, emb)
        print(f"embeddings {emb.shape} -> {args.out_embeddings}")

    from sklearn.cluster import AgglomerativeClustering

    # Ward linkage (minimizes within-cluster variance) gives BALANCED clusters —
    # avoids average-linkage chaining everything into a few mega-blobs. Embeddings
    # are L2-normalized so Euclidean Ward ~ spherical/cosine clustering.
    fine = AgglomerativeClustering(n_clusters=args.n_fine, linkage="ward").fit_predict(emb)
    coarse = AgglomerativeClustering(n_clusters=args.n_coarse, linkage="ward").fit_predict(emb)

    # coherence per fine cluster = mean pairwise cosine of members (bge-m3 bar 0.593)
    def coherence(idxs):
        if len(idxs) < 2:
            return 1.0
        sub = emb[idxs]
        sim = sub @ sub.T
        n = len(idxs)
        return float((sim.sum() - n) / (n * (n - 1)))

    fine_groups = {}
    for ci in range(args.n_fine):
        members = [i for i, c in enumerate(fine) if c == ci]
        fids_m = [int(fids[i]) for i in members]
        fine_groups[str(ci)] = {
            "feature_ids": fids_m,
            "size": len(members),
            "coherence": round(coherence(members), 3),
            "coarse": int(np.bincount(coarse[members]).argmax()),  # dominant coarse cluster
        }

    out = {
        "n_fine": args.n_fine,
        "n_coarse": args.n_coarse,
        "fine_to_coarse": {str(ci): fine_groups[str(ci)]["coarse"] for ci in range(args.n_fine)},
        "fine": fine_groups,
        "feature_fine": {fids[i]: int(fine[i]) for i in range(len(fids))},
        "feature_coarse": {fids[i]: int(coarse[i]) for i in range(len(fids))},
    }
    json.dump(out, open(args.out_clusters, "w"), indent=1)

    sizes = [g["size"] for g in fine_groups.values()]
    cohs = [g["coherence"] for g in fine_groups.values()]
    print(f"\nfine sub-clusters: {args.n_fine}")
    print(f"  size: min {min(sizes)} max {max(sizes)} median {int(np.median(sizes))} mean {np.mean(sizes):.1f}")
    print(f"  coherence: median {np.median(cohs):.3f}, below 0.593: {sum(c<0.593 for c in cohs)}/{args.n_fine}")
    coarse_sizes = np.bincount(coarse)
    print(f"coarse categories: {args.n_coarse}, sizes: {sorted(coarse_sizes.tolist(), reverse=True)}")
    print(f"  -> {args.out_clusters}")


if __name__ == "__main__":
    main()
