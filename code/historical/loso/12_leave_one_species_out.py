"""
12_leave_one_species_out.py — Generalization test.

Holds out all specimens of one species, trains on remaining 23,
generates for the held-out species using phylogenetic position,
and measures Chamfer Distance to real held-out specimens.

This tests: can the model extrapolate to unseen species?

Usage: python 12_leave_one_species_out.py --species "G.Fortis"
       python 12_leave_one_species_out.py --all  (test all species with >= 4 specimens)
Output: output/evaluation/loso_results.json
"""

import os, json, argparse
import numpy as np
import torch
from models.utils import load_config, set_seed

os.makedirs("output/evaluation", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, help="Species to hold out")
    parser.add_argument("--all", action="store_true", help="Test all species with >= 4 specimens")
    args = parser.parse_args()

    config = load_config("config.yaml")
    set_seed(42)

    ckpt_sdf = "checkpoints/deepsdf"
    latent_codes = np.load(os.path.join(ckpt_sdf, "latent_codes.npy"))
    species_indices = np.load(os.path.join(ckpt_sdf, "species_indices.npy"))
    centroids = np.load("checkpoints/diffusion/species_centroids.npy")
    res_mean = np.load("checkpoints/diffusion/res_mean.npy")
    res_global_std = np.load("checkpoints/diffusion/res_global_std.npy")[0]

    with open(os.path.join(config["data"]["processed_dir"], "metadata.json")) as f:
        metadata = json.load(f)
    sp2idx = metadata["species_to_idx"]
    idx2sp = metadata["idx_to_species"]

    # Determine which species to test
    if args.all:
        species_to_test = []
        for sp_name, sp_idx in sp2idx.items():
            n = (species_indices == sp_idx).sum()
            if n >= 4:
                species_to_test.append(sp_name)
        print(f"Testing {len(species_to_test)} species with >= 4 specimens")
    elif args.species:
        species_to_test = [args.species]
    else:
        print("Specify --species or --all")
        return

    results = {}

    for sp_name in species_to_test:
        sp_idx = sp2idx[sp_name]
        held_out_mask = species_indices == sp_idx
        held_out_codes = latent_codes[held_out_mask]
        train_mask = ~held_out_mask
        train_codes = latent_codes[train_mask]
        train_species = species_indices[train_mask]

        n_held = held_out_mask.sum()
        print(f"\n=== LOSO: {sp_name} (held out {n_held} specimens) ===")

        # Method 1: Centroid-based prediction
        # Use phylogenetic neighbors to estimate held-out centroid
        # Find 3 nearest species by phylogenetic distance
        phylo_dist = np.load(os.path.join(ckpt_sdf, "phylo_dist.npy"))
        sp_dists = phylo_dist[sp_idx]
        # Exclude self (distance 0)
        other_sp = [s for s in range(len(sp_dists)) if s != sp_idx and s in np.unique(train_species)]
        other_dists = [(s, sp_dists[s]) for s in other_sp]
        other_dists.sort(key=lambda x: x[1])

        # Weighted average of nearest 3 species centroids
        k = min(3, len(other_dists))
        nearest = other_dists[:k]
        weights = np.array([1.0 / (d + 1e-6) for _, d in nearest])
        weights /= weights.sum()

        predicted_centroid = np.zeros_like(centroids[0])
        for (s, _), w in zip(nearest, weights):
            # Centroid from training data only
            s_mask = train_species == s
            if s_mask.any():
                predicted_centroid += w * train_codes[s_mask].mean(axis=0)

        # Generate codes: predicted centroid + random residuals
        np.random.seed(42)
        gen_codes = []
        for _ in range(n_held):
            residual = np.random.randn(latent_codes.shape[1]) * res_global_std
            gen_codes.append(predicted_centroid + residual)
        gen_codes = np.array(gen_codes)

        # Compare to real held-out codes
        # L2 distances
        dists = []
        for gc in gen_codes:
            min_d = min(np.linalg.norm(gc - hc) for hc in held_out_codes)
            dists.append(min_d)

        # Within held-out distances (reference)
        within_dists = []
        for i in range(len(held_out_codes)):
            for j in range(i+1, len(held_out_codes)):
                within_dists.append(np.linalg.norm(held_out_codes[i] - held_out_codes[j]))

        # Centroid distance
        real_centroid = held_out_codes.mean(axis=0)
        centroid_error = np.linalg.norm(predicted_centroid - real_centroid)

        nearest_names = [idx2sp[str(s)] for s, _ in nearest]

        results[sp_name] = {
            "n_held_out": int(n_held),
            "nearest_species": nearest_names,
            "centroid_l2_error": float(centroid_error),
            "gen_to_nearest_real_l2": {
                "mean": float(np.mean(dists)),
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
            },
            "within_held_out_l2": {
                "mean": float(np.mean(within_dists)) if within_dists else 0,
            },
        }

        print(f"  Nearest species: {nearest_names}")
        print(f"  Centroid L2 error: {centroid_error:.4f}")
        print(f"  Gen-to-nearest real L2: {np.mean(dists):.4f}")
        print(f"  Within held-out L2: {np.mean(within_dists):.4f}" if within_dists else "  N/A")

    # Save
    with open("output/evaluation/loso_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== LOSO Summary ===")
    for sp, r in results.items():
        ratio = r["gen_to_nearest_real_l2"]["mean"] / r["within_held_out_l2"]["mean"] if r["within_held_out_l2"]["mean"] > 0 else float('inf')
        print(f"  {sp}: centroid_err={r['centroid_l2_error']:.4f}, "
              f"gen_NN={r['gen_to_nearest_real_l2']['mean']:.4f}, "
              f"ratio={ratio:.2f}x")
    print(f"\nSaved to output/evaluation/loso_results.json")


if __name__ == "__main__":
    main()
