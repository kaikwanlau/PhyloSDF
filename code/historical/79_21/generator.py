"""Train-only proxy-conditioned residual flow for the four-species holdout.

Preparation receives exactly the new 79 learned training codes. It has no API
for heldout fitted codes, bounding boxes, geometry, historical embeddings, or
species lookup parameters. The frozen taxonomy proxy is indexed as provided;
its historical sorted-key construction defect is deliberately not repaired.

``prepare_pool`` and ``controls`` return only NPZ-safe NumPy arrays. Save those
arrays before fitting/testing. Caller owns seeds, RNG/checkpoint restoration,
the fixed 10,000-update loop, and the cosine scheduler (eta_min=1e-6). Each
update samples 79 training rows with replacement using ``sampling_weights``.
Condition columns follow the supplied ``train_species20`` order; labels are
the original metadata species indices, never implicitly remapped indices.

Residual moments use uniform specimen weights and a population covariance.
Distances use one scalar: the maximum in the train20-by-train20 submatrix.
Targets use that same scalar without clipping or target-derived statistics.
The 64-dimensional covariance can be singular: no diagonal jitter is added.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # Preparation/control validation also works without Torch.
    torch = None
    nn = None
    F = None


LATENT_DIM = 64
PROXY_DIM = 20
FIXED_UPDATES = 10_000
TRAINING_BATCH_SIZE = 79
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VARIANCE_WEIGHT = 0.1
GRADIENT_CLIP = 1.0
EULER_STEPS = 50


def _finite_array(value, shape, name, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _names(values: Sequence[str], count: int, name: str) -> tuple[str, ...]:
    values = tuple(values)
    if (len(values) != count or len(set(values)) != count
            or any(not isinstance(x, str) or not x for x in values)):
        raise ValueError(f"{name} requires {count} distinct nonempty species names")
    return values


def prepare_pool(codes79, train_labels79, train_species20, target_species4,
                 full_distance24, metadata_species_to_idx: Mapping[str, int]):
    """Freeze training residuals, distance inputs, and target centroids.

    Integer ``train_labels79`` must use metadata indices. The supplied training
    name order is the condition-column order, and target name order defines
    target-local indices 0..3 used by ``controls``/``compose_cfm_codes``.
    Neighbors are chosen by (raw proxy distance, alphabetical species name).
    Weights are inverse(distance + 1e-6), normalized across exactly 3 neighbors.
    """
    train_names = _names(train_species20, PROXY_DIM, "train_species20")
    target_names = _names(target_species4, 4, "target_species4")
    if set(train_names) & set(target_names):
        raise ValueError("Training and target species must be disjoint")
    mapping = dict(metadata_species_to_idx)
    if (set(mapping) != set(train_names) | set(target_names)
            or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool)
                   for x in mapping.values())
            or sorted(mapping.values()) != list(range(24))):
        raise ValueError("Metadata must map the exact 24 species bijectively to 0..23")
    codes = _finite_array(codes79, (79, LATENT_DIM), "codes79").copy()
    labels_source = np.asarray(train_labels79)
    if labels_source.shape != (79,) or not np.issubdtype(labels_source.dtype, np.integer):
        raise ValueError("train_labels79 must contain 79 integer metadata indices")
    labels = labels_source.astype(np.int64, copy=True)
    train_idx = np.asarray([mapping[name] for name in train_names], dtype=np.int64)
    target_idx = np.asarray([mapping[name] for name in target_names], dtype=np.int64)
    if set(labels.tolist()) != set(train_idx.tolist()):
        raise ValueError("Codes must cover all 20 training species and no target species")
    local_by_metadata = {int(value): local for local, value in enumerate(train_idx)}
    local_labels = np.asarray([local_by_metadata[int(x)] for x in labels], dtype=np.int64)

    distance = _finite_array(full_distance24, (24, 24), "full_distance24")
    if (np.any(distance < 0) or not np.allclose(distance, distance.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(distance), 0, rtol=0, atol=1e-12)):
        raise ValueError("Proxy distances must be nonnegative, symmetric, and zero on diagonal")
    train_distance = distance[np.ix_(train_idx, train_idx)].copy()
    target_distance = distance[np.ix_(target_idx, train_idx)].copy()
    distance_scale = float(train_distance.max())
    if distance_scale <= 0:
        raise ValueError("Training proxy distances require a positive normalization scale")

    centroids = np.stack([codes[local_labels == i].mean(axis=0)
                          for i in range(PROXY_DIM)])
    residuals = codes - centroids[local_labels]
    weights = np.full(79, 1.0 / 79, dtype=np.float64)
    mean = np.sum(weights[:, None] * residuals, axis=0)
    centered = residuals - mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_tolerance = 1e-10 * max(1.0, float(np.linalg.norm(covariance, ord=2)))
    if eigenvalues.min() < -eigen_tolerance:
        raise ValueError("Residual population covariance is not positive semidefinite")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    if not np.allclose(factor @ factor.T, covariance, rtol=1e-10, atol=eigen_tolerance):
        raise ValueError("Residual covariance factor does not reproduce covariance")
    residual_scale = float(np.sqrt(np.trace(covariance) / LATENT_DIM))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("Training residual pool must have positive finite variance")

    neighbors = np.asarray([
        sorted(range(PROXY_DIM), key=lambda i: (float(row[i]), train_names[i]))[:3]
        for row in target_distance
    ], dtype=np.int64)
    neighbor_distances = np.take_along_axis(target_distance, neighbors, axis=1)
    neighbor_weights = 1.0 / (neighbor_distances + 1e-6)
    neighbor_weights /= neighbor_weights.sum(axis=1, keepdims=True)
    predicted_centroids = np.sum(centroids[neighbors] * neighbor_weights[:, :, None], axis=1)

    return {
        "train_codes": codes,
        "train_labels": labels,
        "train_local_labels": local_labels,
        "train_species": np.asarray(train_names),
        "target_species": np.asarray(target_names),
        "train_species_indices": train_idx,
        "target_species_indices": target_idx,
        "train_species_counts": np.bincount(local_labels, minlength=PROXY_DIM),
        "train_centroids": centroids,
        "residuals": residuals,
        "normalized_residuals": centered / residual_scale,
        "sampling_weights": weights,
        "residual_mean": mean,
        "residual_covariance": covariance,
        "residual_factor": factor,
        "residual_global_std": np.asarray(residual_scale),
        "train_proxy_distances": train_distance,
        "target_proxy_distances": target_distance,
        "proxy_distance_scale": np.asarray(distance_scale),
        "train_proxy_conditioning": train_distance / distance_scale,
        "target_proxy_conditioning": target_distance / distance_scale,
        "target_neighbor_local_indices": neighbors,
        "target_neighbor_metadata_indices": train_idx[neighbors],
        "target_neighbor_species": np.asarray(train_names)[neighbors],
        "target_neighbor_distances": neighbor_distances,
        "target_neighbor_weights": neighbor_weights,
        "predicted_target_centroids": predicted_centroids,
    }


def _target_schedule(target_local_indices):
    indices = np.asarray(target_local_indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= 4)):
        raise ValueError("Target schedule requires integer target-local indices 0..3")
    return indices.astype(np.int64, copy=True)


def controls(pool, target_local_indices, seed: int):
    """Generate preserved empirical donors and common Gaussian/CFM base noise.

    Use the returned ``common_gaussian_base_noise`` verbatim as the flow ODE
    initial state. Empirical donors use a separate deterministic RNG stream.
    Each method uses the same target schedule and predicted centroid arrays.
    """
    targets = _target_schedule(target_local_indices)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Generation seed must be a nonnegative integer")
    donor_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    donors = donor_rng.choice(79, len(targets), p=pool["sampling_weights"])
    base = noise_rng.standard_normal((len(targets), LATENT_DIM))
    centroids = pool["predicted_target_centroids"][targets]
    empirical = centroids + pool["residuals"][donors]
    gaussian = centroids + pool["residual_mean"] + base @ pool["residual_factor"].T
    if not np.isfinite(empirical).all() or not np.isfinite(gaussian).all():
        raise ValueError("Nonfinite matched controls")
    return {
        "empirical_residual": empirical,
        "pooled_gaussian": gaussian,
        "empirical_donor_indices": donors,
        "common_gaussian_base_noise": base,
        "target_local_indices": targets,
        "target_species_indices": pool["target_species_indices"][targets],
        "predicted_centroids": centroids.copy(),
        "proxy_conditioning": pool["target_proxy_conditioning"][targets].copy(),
    }


def compose_cfm_codes(pool, normalized_samples, target_local_indices):
    """Return generated latent codes from ODE outputs in normalized units."""
    targets = _target_schedule(target_local_indices)
    samples = _finite_array(normalized_samples, (len(targets), LATENT_DIM), "normalized_samples")
    result = (pool["predicted_target_centroids"][targets] + pool["residual_mean"]
              + samples * float(pool["residual_global_std"]))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite generated flow codes")
    return result


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for flow-network training and sampling")


if torch is not None:
    class SinusoidalTimeEmbedding(nn.Module):
        """The archived flow network's exact time embedding."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            frequencies = torch.exp(torch.arange(half, device=t.device)
                                    * -(math.log(10000) / (half - 1)))
            phase = t[:, None].float() * frequencies[None, :]
            return torch.cat([phase.sin(), phase.cos()], dim=-1)


    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.05):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout))

        def forward(self, x):
            return x + self.net(x)


    class ProxyFlowNet(nn.Module):
        """Historical residual-flow architecture with only 20-distance input.

        This model has no learned species lookup, geometry/size input, or
        embedded historical phylogeny. Three historical residual hidden blocks
        operate at width128; all dropout probabilities are0.05.
        """
        def __init__(self, latent_dim=64, model_dim=128, num_layers=3,
                     proxy_dim=20, dropout=0.05):
            super().__init__()
            if (latent_dim, model_dim, num_layers, proxy_dim, dropout) != (64, 128, 3, 20, 0.05):
                raise ValueError("Primary holdout flow architecture is fixed at64/128/3/20/dropout0.05")
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(model_dim), nn.Linear(model_dim, model_dim),
                nn.GELU(), nn.Linear(model_dim, model_dim))
            self.proxy_proj = nn.Sequential(
                nn.Linear(proxy_dim, 64), nn.GELU(), nn.Linear(64, 32))
            self.input_proj = nn.Linear(latent_dim + model_dim + 32, model_dim)
            self.blocks = nn.ModuleList([ResidualBlock(model_dim, dropout)
                                        for _ in range(num_layers)])
            self.output_proj = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))

        def forward(self, x_t, t, proxy_distances):
            if (x_t.ndim != 2 or x_t.shape[1] != LATENT_DIM or t.shape != (len(x_t),)
                    or proxy_distances.shape != (len(x_t), PROXY_DIM)):
                raise ValueError("Flow inputs must have shapes(B,64),(B,),(B,20)")
            hidden = self.input_proj(torch.cat(
                [x_t, self.time_embed(t), self.proxy_proj(proxy_distances)], dim=-1))
            for block in self.blocks:
                hidden = block(hidden)
            return self.output_proj(hidden)
else:
    class ProxyFlowNet:
        def __init__(self, *args, **kwargs):
            _require_torch()


def flow_loss(model, x_1, proxy_conditioning, t, x_0, *, return_components=False):
    """Historical straight-path CFM MSE plus locked midpoint variance proxy.

    All random values are explicit inputs so caller can preserve their RNG.
    The variance operator is ``torch.var(correction=1)`` over all batch and
    latent entries. The archived B>=8 guard is retained. Network dropout still
    consumes Torch RNG in training mode; preserve that RNG on checkpoints.
    """
    _require_torch()
    if (x_1.ndim != 2 or x_1.shape[1] != LATENT_DIM or x_0.shape != x_1.shape
            or t.shape != (len(x_1),) or proxy_conditioning.shape != (len(x_1), PROXY_DIM)):
        raise ValueError("CFM inputs require(B,64) endpoints,(B,) times,(B,20) conditions")
    if not all(bool(torch.isfinite(value).all()) for value in (x_1, x_0, t, proxy_conditioning)):
        raise ValueError("CFM inputs must be finite")
    if bool(torch.any((t < 0) | (t > 1))):
        raise ValueError("CFM interpolation times must be in[0,1]")
    x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
    mse = F.mse_loss(model(x_t, t, proxy_conditioning), x_1 - x_0)
    penalty = mse.new_zeros(())
    if len(x_1) >= 8:
        midpoint = 0.5 * (x_0 + x_1)
        proxy = midpoint + 0.5 * model(midpoint, torch.full_like(t, 0.5), proxy_conditioning)
        penalty = VARIANCE_WEIGHT * torch.relu(
            torch.var(x_1, correction=1) - torch.var(proxy, correction=1))
    loss = mse + penalty
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite CFM loss")
    if return_components:
        return loss, {"cfm_mse": mse, "variance_penalty": penalty}
    return loss


def make_optimizer(model):
    """Fixed AdamW; caller attaches CosineAnnealingLR(10000,eta_min=1e-6)."""
    _require_torch()
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def training_step(model, optimizer, x_1, proxy_conditioning, t, x_0):
    """One explicit-input optimizer update, with the archived gradient clip1.

    Caller selects rows, sets ``model.train()``, generates noise/times, advances
    the scheduler once per committed update, and saves all RNG and optimizer
    state. This function does not implement a checkpoint or stopping policy.
    """
    _require_torch()
    optimizer.zero_grad(set_to_none=True)
    loss, terms = flow_loss(model, x_1, proxy_conditioning, t, x_0, return_components=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP,
                                               error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Nonfinite flow update; do not commit this update")
    return {"loss": float(loss.detach()), "cfm_mse": float(terms["cfm_mse"].detach()),
            "variance_penalty": float(terms["variance_penalty"].detach()),
            "pre_clip_gradient_norm": float(grad_norm)}


def euler_samples(model, common_base_noise, proxy_conditioning, *, num_steps=50,
                  guidance_scale=1.0, temperature=1.0):
    """Integrate normalized residuals using preserved noise and 50 Euler steps.

    Guidance1 and temperature1 are fixed primary settings. There is no
    unconditional branch or classifier-free dropout in this proxy-only model.
    This function consumes no new RNG; model must already be in eval mode.
    Caller denormalizes with ``compose_cfm_codes`` using the same target schedule
    used by ``controls``. Input noise is cloned and never modified in place.
    """
    _require_torch()
    if num_steps != EULER_STEPS or guidance_scale != 1.0 or temperature != 1.0:
        raise ValueError("Primary sampling is fixed at50 Euler steps,CFG1,temperature1")
    if model.training:
        raise ValueError("Set the flow model to eval mode before sampling")
    if (common_base_noise.ndim != 2 or common_base_noise.shape[1] != LATENT_DIM
            or proxy_conditioning.shape != (len(common_base_noise), PROXY_DIM)):
        raise ValueError("Sampling inputs require noise(N,64) and conditions(N,20)")
    if not all(bool(torch.isfinite(value).all()) for value in (common_base_noise, proxy_conditioning)):
        raise ValueError("Sampling inputs must be finite")
    with torch.no_grad():
        x = common_base_noise.clone()
        dt = 1.0 / EULER_STEPS
        for i in range(EULER_STEPS):
            t = torch.full((len(x),), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * model(x, t, proxy_conditioning)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite generated flow residuals")
    return x
