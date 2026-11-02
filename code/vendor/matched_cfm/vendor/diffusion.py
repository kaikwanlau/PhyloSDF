"""
Phylogenetically-Conditioned Latent Generation.

Three generation methods:
  1. Flow Matching (CFM) - preferred, single-step ODE, no error accumulation
  2. GMM baseline - per-species Gaussian, simple but effective  
  3. DDPM - kept for backward compatibility

CFM replaces DDPM because DDPM with 1000 timesteps fails on small datasets
(100-1000 points in 64-dim) — the iterative denoising accumulates errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ================================================================
# Shared network architecture (used by both CFM and DDPM)
# ================================================================

class PhyloFlowNet(nn.Module):
    """Velocity/noise prediction network with phylogenetic conditioning."""
    def __init__(self, latent_dim=64, model_dim=256, num_layers=4,
                 num_species=0, condition_on_species=True,
                 condition_on_size=True, condition_on_phylo=True,
                 species_embed_dim=32, phylo_embed_dim=32, dropout=0.05):
        super().__init__()
        self.condition_on_species = condition_on_species
        self.condition_on_size = condition_on_size
        self.condition_on_phylo = condition_on_phylo

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_dim),
            nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim))

        cond_dim = 0
        if condition_on_species and num_species > 0:
            self.species_embed = nn.Embedding(num_species + 1, species_embed_dim)
            cond_dim += species_embed_dim
        else:
            self.species_embed = None

        if condition_on_size:
            self.size_proj = nn.Sequential(
                nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
            cond_dim += 32
        else:
            self.size_proj = None

        if condition_on_phylo:
            self.phylo_proj = nn.Sequential(
                nn.Linear(phylo_embed_dim, 64), nn.GELU(), nn.Linear(64, 32))
            cond_dim += 32
        else:
            self.phylo_proj = None

        self.input_proj = nn.Linear(latent_dim + model_dim + cond_dim, model_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(model_dim, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))

    def forward(self, x_t, t, species_idx=None, bbox_dims=None, phylo_embed=None):
        t_emb = self.time_embed(t)
        parts = [x_t, t_emb]

        if self.condition_on_species and self.species_embed is not None:
            if species_idx is not None:
                parts.append(self.species_embed(species_idx))
            else:
                unc = torch.full((x_t.shape[0],), self.species_embed.num_embeddings - 1,
                                 device=x_t.device, dtype=torch.long)
                parts.append(self.species_embed(unc))

        if self.condition_on_size and self.size_proj is not None:
            if bbox_dims is not None:
                parts.append(self.size_proj(bbox_dims))
            else:
                parts.append(self.size_proj(torch.zeros(x_t.shape[0], 3, device=x_t.device)))

        if self.condition_on_phylo and self.phylo_proj is not None:
            if phylo_embed is not None:
                parts.append(self.phylo_proj(phylo_embed))
            else:
                dim = self.phylo_proj[0].in_features
                parts.append(self.phylo_proj(torch.zeros(x_t.shape[0], dim, device=x_t.device)))

        x = self.input_proj(torch.cat(parts, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# Backward compat alias
PhyloDiffusionNet = PhyloFlowNet


# ================================================================
# Conditional Flow Matching (preferred)
# ================================================================

class ConditionalFlowMatching:
    """
    OT-CFM: train v(x_t, t) where x_t = (1-t)*noise + t*data.
    Sample by integrating dx/dt = v(x, t) from t=0 to t=1.
    """
    def __init__(self, device="cuda"):
        self.device = device

    def training_loss(self, model, x_1, species_idx=None, bbox_dims=None,
                      phylo_embed=None):
        B = x_1.shape[0]
        t = torch.rand(B, device=self.device)
        x_0 = torch.randn_like(x_1)
        t_e = t[:, None]
        x_t = (1 - t_e) * x_0 + t_e * x_1
        target = x_1 - x_0
        v_pred = model(x_t, t, species_idx, bbox_dims, phylo_embed)
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, model, num_samples, latent_dim, num_steps=50,
               species_idx=None, bbox_dims=None, phylo_embed=None,
               guidance_scale=1.0, temperature=1.0):
        x = torch.randn(num_samples, latent_dim, device=self.device) * temperature
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=self.device)
            if guidance_scale > 1.0 and species_idx is not None:
                vc = model(x, t, species_idx, bbox_dims, phylo_embed)
                vu = model(x, t, None, bbox_dims, phylo_embed)
                v = vu + guidance_scale * (vc - vu)
            else:
                v = model(x, t, species_idx, bbox_dims, phylo_embed)
            x = x + v * dt
        return x


# ================================================================
# GMM Baseline
# ================================================================

class PhyloGMM:
    """Per-species Gaussian with Ledoit-Wolf shrinkage. No neural net needed."""
    def __init__(self):
        self.means = {}
        self.covs = {}
        self.global_mean = None
        self.global_cov = None

    def fit(self, latent_codes, species_indices):
        D = latent_codes.shape[1]
        self.global_mean = latent_codes.mean(axis=0)
        self.global_cov = np.cov(latent_codes.T) + np.eye(D) * 1e-4

        for sp in np.unique(species_indices):
            sp_codes = latent_codes[species_indices == sp]
            self.means[sp] = sp_codes.mean(axis=0)
            if len(sp_codes) >= 3:
                alpha = 2.0 / (len(sp_codes) + 2.0)
                self.covs[sp] = (1 - alpha) * np.cov(sp_codes.T) + alpha * self.global_cov
            else:
                self.covs[sp] = self.global_cov * 0.5
        print(f"GMM fit: {len(self.means)} species")

    def sample(self, num_samples, species_idx=None):
        m = self.means.get(species_idx, self.global_mean)
        c = self.covs.get(species_idx, self.global_cov)
        return np.random.multivariate_normal(m, c, size=num_samples).astype(np.float32)

    def sample_ancestral(self, sp_a, sp_b, num_samples):
        m = 0.5 * self.means[sp_a] + 0.5 * self.means[sp_b]
        c = 0.5 * self.covs[sp_a] + 0.5 * self.covs[sp_b]
        return np.random.multivariate_normal(m, c, size=num_samples).astype(np.float32)


# ================================================================
# DDPM (backward compatibility)
# ================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 0.0001, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 beta_schedule="cosine", device="cuda"):
        self.timesteps = timesteps
        self.device = device
        betas = cosine_beta_schedule(timesteps) if beta_schedule == "cosine" \
            else linear_beta_schedule(timesteps, beta_start, beta_end)
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def q_sample(self, z_0, t, noise=None):
        if noise is None: noise = torch.randn_like(z_0)
        return self.sqrt_alphas_cumprod[t][:, None] * z_0 + \
               self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise

    def training_loss(self, model, z_0, species_idx=None, bbox_dims=None, phylo_embed=None):
        B = z_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        return F.mse_loss(model(z_t, t, species_idx, bbox_dims, phylo_embed), noise)

    @torch.no_grad()
    def sample(self, model, num_samples, latent_dim, species_idx=None,
               bbox_dims=None, phylo_embed=None, guidance_scale=1.0, temperature=1.0):
        z = torch.randn(num_samples, latent_dim, device=self.device) * temperature
        for t in reversed(range(self.timesteps)):
            tt = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            if guidance_scale > 1.0 and species_idx is not None:
                nc = model(z, tt, species_idx, bbox_dims, phylo_embed)
                nu = model(z, tt, None, bbox_dims, phylo_embed)
                np_ = nu + guidance_scale * (nc - nu)
            else:
                np_ = model(z, tt, species_idx, bbox_dims, phylo_embed)
            mean = self.sqrt_recip_alphas[t] * (
                z - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * np_)
            z = mean + (torch.sqrt(self.posterior_variance[t]) * torch.randn_like(z) if t > 0 else 0)
        return z
