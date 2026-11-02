"""
ARAP-Inspired Latent Space Augmentation.

Adapts the geometric augmentation idea from GLASS (CVPR 2022) to work
in the DeepSDF latent space, without requiring mesh correspondence.

Key idea: perturb latent codes along directions that respect the local
geometry of the latent manifold, guided by the Hessian structure of
the decoder. This produces augmented codes that decode to plausible
shapes rather than random noise.

Methods:
1. Local Hessian-guided perturbation
2. Phylogenetically-aware interpolation
3. Species-centroid extrapolation
"""

import torch
import numpy as np
from itertools import combinations


def augment_latent_codes(
    latent_codes,           # (N, D) numpy array of original latent codes
    species_indices,        # (N,) species index for each specimen
    phylo_dist_matrix,      # (S, S) phylogenetic distance matrix
    bbox_dims,              # (N, 3) bounding box dimensions
    config,                 # augmentation config dict
    decoder=None,           # Optional: DeepSDF decoder for Hessian computation
    device="cuda",
):
    """
    Augment latent codes using multiple strategies.

    Returns:
        aug_codes: (M, D) augmented latent codes
        aug_species: (M,) species indices
        aug_bbox: (M, 3) bounding box dimensions
        aug_labels: (M,) string labels for each augmented code's origin
    """
    N, D = latent_codes.shape
    num_species = phylo_dist_matrix.shape[0]
    noise_std = config.get("arap_noise_std", 0.3)
    num_aug_per = config.get("num_augmented_per_specimen", 10)
    interp_samples = config.get("interpolation_samples", 5)
    total_target = config.get("total_target", 2000)

    all_codes = [latent_codes.copy()]
    all_species = [species_indices.copy()]
    all_bbox = [bbox_dims.copy()]
    all_labels = [np.array(["original"] * N)]

    # ------------------------------------------------------------------
    # Strategy 1: Hessian-guided local perturbation
    # Perturb each code along principal directions of local curvature
    # ------------------------------------------------------------------
    if decoder is not None:
        print("  Augmenting via Hessian-guided perturbation...")
        hessian_codes = _hessian_guided_augment(
            latent_codes, decoder, device,
            num_per=min(num_aug_per, 5), noise_std=noise_std
        )
    else:
        print("  Augmenting via PCA-guided perturbation...")
        hessian_codes = _pca_guided_augment(
            latent_codes, species_indices, num_species,
            num_per=min(num_aug_per, 5), noise_std=noise_std
        )

    all_codes.append(hessian_codes)
    all_species.append(np.repeat(species_indices, len(hessian_codes) // N))
    all_bbox.append(np.repeat(bbox_dims, len(hessian_codes) // N, axis=0))
    all_labels.append(np.array(["hessian_perturb"] * len(hessian_codes)))

    # ------------------------------------------------------------------
    # Strategy 2: Phylogenetically-aware interpolation
    # Interpolate between specimens of related species
    # ------------------------------------------------------------------
    print("  Augmenting via phylogenetic interpolation...")
    interp_codes, interp_sp, interp_bbox = _phylo_interpolation(
        latent_codes, species_indices, phylo_dist_matrix, bbox_dims,
        num_samples=interp_samples
    )
    if len(interp_codes) > 0:
        all_codes.append(interp_codes)
        all_species.append(interp_sp)
        all_bbox.append(interp_bbox)
        all_labels.append(np.array(["phylo_interp"] * len(interp_codes)))

    # ------------------------------------------------------------------
    # Strategy 3: Species-centroid extrapolation
    # Move along the direction between species centroids
    # ------------------------------------------------------------------
    print("  Augmenting via species centroid extrapolation...")
    extrap_codes, extrap_sp, extrap_bbox = _centroid_extrapolation(
        latent_codes, species_indices, bbox_dims, num_species,
        num_per_direction=3, scale_range=(0.3, 0.7)
    )
    if len(extrap_codes) > 0:
        all_codes.append(extrap_codes)
        all_species.append(extrap_sp)
        all_bbox.append(extrap_bbox)
        all_labels.append(np.array(["centroid_extrap"] * len(extrap_codes)))

    # Concatenate and optionally subsample to target
    aug_codes = np.concatenate(all_codes, axis=0)
    aug_species = np.concatenate(all_species, axis=0)
    aug_bbox = np.concatenate(all_bbox, axis=0)
    aug_labels = np.concatenate(all_labels, axis=0)

    if len(aug_codes) > total_target:
        # Keep all originals, subsample augmented
        orig_mask = aug_labels == "original"
        aug_mask = ~orig_mask
        n_keep = total_target - orig_mask.sum()
        aug_idx = np.where(aug_mask)[0]
        keep_idx = np.random.choice(aug_idx, size=n_keep, replace=False)
        all_idx = np.concatenate([np.where(orig_mask)[0], keep_idx])
        aug_codes = aug_codes[all_idx]
        aug_species = aug_species[all_idx]
        aug_bbox = aug_bbox[all_idx]
        aug_labels = aug_labels[all_idx]

    print(f"  Augmentation complete: {N} original -> {len(aug_codes)} total")
    label_counts = {l: (aug_labels == l).sum() for l in np.unique(aug_labels)}
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

    return aug_codes, aug_species, aug_bbox, aug_labels


def _hessian_guided_augment(latent_codes, decoder, device, num_per=5, noise_std=0.3):
    """
    Perturb latent codes along directions informed by the decoder's Hessian.
    Uses finite-difference approximation of the Hessian diagonal.
    """
    N, D = latent_codes.shape
    codes_t = torch.tensor(latent_codes, device=device, dtype=torch.float32)
    aug_codes = []

    # Sample a fixed set of query points for Hessian estimation
    query_points = torch.randn(100, 3, device=device) * 0.5

    for i in range(N):
        z = codes_t[i:i+1]  # (1, D)
        # Estimate Hessian diagonal via finite differences
        eps = 0.01
        hess_diag = torch.zeros(D, device=device)

        z_expand = z.expand(100, -1)  # (100, D)
        with torch.no_grad():
            f0 = decoder(z_expand, query_points).squeeze(-1)  # (100,)

            for d in range(D):
                z_plus = z.clone()
                z_plus[0, d] += eps
                z_minus = z.clone()
                z_minus[0, d] -= eps

                fp = decoder(z_plus.expand(100, -1), query_points).squeeze(-1)
                fm = decoder(z_minus.expand(100, -1), query_points).squeeze(-1)

                # Second derivative approximation
                hess_diag[d] = ((fp - 2 * f0 + fm) / (eps ** 2)).abs().mean()

        # Low Hessian = flat directions = safe to perturb more
        # High Hessian = sensitive directions = perturb less
        sensitivity = hess_diag / (hess_diag.max() + 1e-8)
        inv_sensitivity = 1.0 / (sensitivity + 0.1)
        inv_sensitivity = inv_sensitivity / inv_sensitivity.norm() * D ** 0.5

        for _ in range(num_per):
            noise = torch.randn(D, device=device) * noise_std * inv_sensitivity
            aug_codes.append((z.squeeze() + noise).cpu().numpy())

    return np.array(aug_codes)


def _pca_guided_augment(latent_codes, species_indices, num_species,
                        num_per=5, noise_std=0.3):
    """
    Fallback: perturb along PCA directions of within-species variation.
    """
    N, D = latent_codes.shape
    aug_codes = []

    # Compute global PCA
    mean = latent_codes.mean(axis=0)
    centered = latent_codes - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    for i in range(N):
        for _ in range(num_per):
            # Sample noise weighted by eigenvalue structure
            noise_coeffs = np.random.randn(D) * noise_std
            noise_coeffs *= np.sqrt(np.maximum(eigenvalues, 0) + 1e-8)
            noise = eigenvectors @ noise_coeffs
            aug_codes.append(latent_codes[i] + noise)

    return np.array(aug_codes)


def _phylo_interpolation(latent_codes, species_indices, phylo_dist_matrix,
                         bbox_dims, num_samples=5):
    """
    Interpolate between specimens of phylogenetically related species.
    Weight interpolation by phylogenetic proximity — closer species
    get more even interpolation, distant species stay closer to endpoints.
    """
    N = len(latent_codes)
    num_species = phylo_dist_matrix.shape[0]

    # Group specimens by species
    species_groups = {}
    for i in range(N):
        sp = species_indices[i]
        if sp not in species_groups:
            species_groups[sp] = []
        species_groups[sp].append(i)

    interp_codes = []
    interp_sp = []
    interp_bbox = []

    # For each pair of species, interpolate representative specimens
    species_list = sorted(species_groups.keys())
    for sp_a, sp_b in combinations(species_list, 2):
        phylo_d = phylo_dist_matrix[sp_a, sp_b]

        # Only interpolate between reasonably related species
        if phylo_d > 0.7:
            continue

        # Pick random specimens from each species
        idx_a = np.random.choice(species_groups[sp_a])
        idx_b = np.random.choice(species_groups[sp_b])

        code_a, code_b = latent_codes[idx_a], latent_codes[idx_b]
        bbox_a, bbox_b = bbox_dims[idx_a], bbox_dims[idx_b]

        # Interpolation weights: more uniform for closer species
        max_alpha = 0.5 - 0.3 * phylo_d  # Closer → wider range
        alphas = np.linspace(0.1, max(0.2, max_alpha), num_samples)

        for alpha in alphas:
            interp_code = (1 - alpha) * code_a + alpha * code_b
            interp_bbox_val = (1 - alpha) * bbox_a + alpha * bbox_b
            interp_codes.append(interp_code)
            interp_sp.append(sp_a)  # Assign to closer species
            interp_bbox.append(interp_bbox_val)

    if not interp_codes:
        return np.array([]).reshape(0, latent_codes.shape[1]), np.array([]), np.array([]).reshape(0, 3)

    return np.array(interp_codes), np.array(interp_sp), np.array(interp_bbox)


def _centroid_extrapolation(latent_codes, species_indices, bbox_dims,
                            num_species, num_per_direction=3,
                            scale_range=(0.3, 0.7)):
    """
    Generate new codes by extrapolating along directions between species centroids.
    """
    # Compute species centroids
    centroids = {}
    centroid_bbox = {}
    for sp in range(num_species):
        mask = species_indices == sp
        if mask.sum() > 0:
            centroids[sp] = latent_codes[mask].mean(axis=0)
            centroid_bbox[sp] = bbox_dims[mask].mean(axis=0)

    extrap_codes = []
    extrap_sp = []
    extrap_bbox = []

    species_list = sorted(centroids.keys())
    for sp in species_list:
        center = centroids[sp]
        center_bb = centroid_bbox[sp]

        # Extrapolate along direction from global mean to species centroid
        global_mean = latent_codes.mean(axis=0)
        direction = center - global_mean
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            continue
        direction = direction / direction_norm

        for scale in np.linspace(scale_range[0], scale_range[1], num_per_direction):
            new_code = center + scale * direction * direction_norm
            extrap_codes.append(new_code)
            extrap_sp.append(sp)
            extrap_bbox.append(center_bb)

    if not extrap_codes:
        return np.array([]).reshape(0, latent_codes.shape[1]), np.array([]), np.array([]).reshape(0, 3)

    return np.array(extrap_codes), np.array(extrap_sp), np.array(extrap_bbox)
