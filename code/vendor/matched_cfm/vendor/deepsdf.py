"""
DeepSDF Auto-Decoder with Phylogenetic Consistency Loss.

Extension of Park et al. (2019) with a novel phylogenetic regularization:
latent code distances are encouraged to correlate with phylogenetic distances.
"""

import torch
import torch.nn as nn
import numpy as np


class DeepSDFDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=512, num_layers=8,
                 skip_connection=None, weight_norm=True):
        super().__init__()
        if skip_connection is None:
            skip_connection = [4]

        self.latent_dim = latent_dim
        self.skip_connection = skip_connection
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = latent_dim + 3
            elif i in skip_connection:
                in_dim = hidden_dim + latent_dim + 3
            else:
                in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1

            linear = nn.Linear(in_dim, out_dim)
            if weight_norm and i < num_layers - 1:
                linear = nn.utils.weight_norm(linear)
            layers.append(linear)

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'weight_g'):
                nn.init.kaiming_normal_(layer.weight_v)
            else:
                nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, latent_code, points):
        x = torch.cat([latent_code, points], dim=-1)
        input_x = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connection:
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.relu(x)
        return self.tanh(x)


class LatentCodeLibrary(nn.Module):
    def __init__(self, num_specimens, latent_dim, init_std=0.01):
        super().__init__()
        self.latent_codes = nn.Embedding(num_specimens, latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=init_std)

    def forward(self, indices):
        return self.latent_codes(indices)

    def get_all_codes(self):
        return self.latent_codes.weight.detach().cpu().numpy()


def clamp_sdf(sdf_values, delta):
    return torch.clamp(sdf_values, -delta, delta)


def deepsdf_loss(pred_sdf, gt_sdf, latent_codes, delta, reg_weight):
    pred_clamped = clamp_sdf(pred_sdf, delta)
    gt_clamped = clamp_sdf(gt_sdf, delta)
    recon_loss = torch.abs(pred_clamped - gt_clamped).mean()
    reg_loss = (latent_codes ** 2).mean()
    total = recon_loss + reg_weight * reg_loss
    return total, recon_loss, reg_loss


def phylogenetic_consistency_loss(latent_lib, species_indices, phylo_dist_matrix):
    """
    Novel phylogenetic consistency loss.

    Encourages latent code distances to correlate with phylogenetic distances.
    For each pair of specimens, the L2 distance between their latent codes
    should be proportional to the phylogenetic distance between their species.

    Args:
        latent_lib: LatentCodeLibrary
        species_indices: (N,) species index for each specimen
        phylo_dist_matrix: (S, S) normalized phylogenetic distance matrix

    Returns:
        loss: scalar — negative Pearson correlation between latent and phylo distances
    """
    codes = latent_lib.latent_codes.weight  # (N, D)
    N = codes.shape[0]

    if N < 3:
        return torch.tensor(0.0, device=codes.device)

    # Compute pairwise latent distances (upper triangle)
    latent_dists = []
    phylo_dists = []

    indices = torch.triu_indices(N, N, offset=1)
    for k in range(indices.shape[1]):
        i, j = indices[0, k].item(), indices[1, k].item()
        ld = torch.norm(codes[i] - codes[j])
        latent_dists.append(ld)

        sp_i, sp_j = species_indices[i], species_indices[j]
        pd = phylo_dist_matrix[sp_i, sp_j]
        phylo_dists.append(pd)

    latent_dists = torch.stack(latent_dists)
    phylo_dists = torch.tensor(phylo_dists, device=codes.device, dtype=torch.float32)

    # Negative Pearson correlation (minimize = maximize correlation)
    ld_mean = latent_dists.mean()
    pd_mean = phylo_dists.mean()
    ld_centered = latent_dists - ld_mean
    pd_centered = phylo_dists - pd_mean

    numerator = (ld_centered * pd_centered).sum()
    denominator = (torch.sqrt((ld_centered ** 2).sum() * (pd_centered ** 2).sum()) + 1e-8)

    correlation = numerator / denominator
    return -correlation  # Minimize negative correlation = maximize correlation
