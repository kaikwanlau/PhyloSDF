"""
Shared utilities for the phylogenetically-conditioned skull generation pipeline.
"""

import os
import re
import yaml
import torch
import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from scipy.spatial import KDTree


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_species_from_filename(filename):
    basename = os.path.basename(str(filename))
    for ext in ['.stl', '.STL', '.mat', '.npz']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    species = re.sub(r'[A-Z]$', '', basename)
    return species


def load_size_data(excel_path):
    df = pd.read_excel(excel_path)
    size_dict = {}
    for _, row in df.iterrows():
        stem = Path(str(row['Filename'])).stem
        size_dict[stem] = np.array([
            row['length_x'], row['width_y'], row['height_z']
        ], dtype=np.float32)
    return size_dict


def build_species_mapping(filenames):
    species_set = sorted(set(
        extract_species_from_filename(f) for f in filenames
    ))
    species_to_idx = {s: i for i, s in enumerate(species_set)}
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    return species_to_idx, idx_to_species, len(species_set)


# ================================================================
# Phylogenetic distance utilities
# ================================================================

def build_phylo_distance_matrix(species_list):
    """
    Build a phylogenetic distance matrix for Darwin's Finch species.

    Uses taxonomic grouping as a proxy for phylogenetic distance.
    Genus-level groupings from established Darwin's Finch phylogeny:
      - Geospiza (G.): Ground finches
      - Camarhynchus (C.): Tree finches
      - Certhidea (C2.): Warbler finches
      - Platyspiza (P.): Vegetarian finch
      - Pinaroloxias (P2.): Cocos finch
      - Other genera: Outgroup tanagers

    For a real paper, replace this with actual phylogenetic distances
    from a published tree (e.g., Burns et al. 2002, Lamichhaney et al. 2015).
    """
    n = len(species_list)
    dist = np.zeros((n, n), dtype=np.float32)

    # Extract genus from species name
    def get_genus(species):
        parts = species.split('.')
        return parts[0] if parts else species

    # Genus-level phylogenetic distances (approximate, from literature)
    # Scale: 0 = same species, 1 = same genus, 2 = sister genera, 3 = distant
    genus_distances = {
        ('G', 'G'): 0.0,     # Within Geospiza
        ('C', 'C'): 0.0,     # Within Camarhynchus
        ('G', 'C'): 1.5,     # Geospiza <-> Camarhynchus
        ('G', 'P'): 2.0,     # Geospiza <-> Platyspiza
        ('C', 'P'): 1.5,     # Camarhynchus <-> Platyspiza
        ('G', 'C2'): 2.5,    # Geospiza <-> Certhidea
        ('C', 'C2'): 2.0,    # Camarhynchus <-> Certhidea
        ('P', 'C2'): 2.5,
        ('G', 'P2'): 3.0,    # Geospiza <-> Pinaroloxias
        ('C', 'P2'): 2.5,
        ('P', 'P2'): 3.0,
        ('C2', 'P2'): 3.0,
    }
    # Outgroup tanagers (L., T., M., E.)
    outgroup_genera = {'L', 'T', 'M', 'E'}
    finch_genera = {'G', 'C', 'P', 'C2', 'P2'}

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0.0
                continue

            sp_i, sp_j = species_list[i], species_list[j]
            g_i, g_j = get_genus(sp_i), get_genus(sp_j)

            if sp_i == sp_j:
                dist[i, j] = 0.0  # Same species
            elif g_i == g_j:
                dist[i, j] = 0.5  # Same genus, different species
            else:
                # Look up genus pair distance
                key = tuple(sorted([g_i, g_j]))
                if key in genus_distances:
                    dist[i, j] = genus_distances[key]
                elif g_i in outgroup_genera and g_j in outgroup_genera:
                    dist[i, j] = 1.0  # Within outgroup
                elif g_i in outgroup_genera or g_j in outgroup_genera:
                    dist[i, j] = 4.0  # Finch <-> outgroup
                else:
                    dist[i, j] = 3.0  # Default

    # Normalize to [0, 1]
    if dist.max() > 0:
        dist = dist / dist.max()

    return dist


def load_or_build_phylo_distances(config, species_list):
    """Load phylogenetic distance matrix from CSV, or build from taxonomy."""
    phylo_path = config["data"].get("phylo_file", "")

    if phylo_path and os.path.exists(phylo_path):
        print(f"Loading phylogenetic distances from {phylo_path}")
        df = pd.read_csv(phylo_path, index_col=0)
        # Reorder to match species_list
        dist = np.zeros((len(species_list), len(species_list)), dtype=np.float32)
        for i, sp_i in enumerate(species_list):
            for j, sp_j in enumerate(species_list):
                if sp_i in df.index and sp_j in df.columns:
                    dist[i, j] = df.loc[sp_i, sp_j]
        if dist.max() > 0:
            dist = dist / dist.max()
        return dist
    else:
        print("Building phylogenetic distances from taxonomy...")
        return build_phylo_distance_matrix(species_list)


def phylo_distance_to_embedding(phylo_dist_matrix, embed_dim=32):
    """
    Convert phylogenetic distance matrix to dense embeddings via MDS.
    Each species gets an embedding vector such that Euclidean distances
    approximate phylogenetic distances.
    """
    from scipy.spatial.distance import squareform
    from sklearn.manifold import MDS

    n = phylo_dist_matrix.shape[0]
    dim = min(embed_dim, n - 1)

    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto')
    embeddings = mds.fit_transform(phylo_dist_matrix)

    # Pad to embed_dim if needed
    if dim < embed_dim:
        pad = np.zeros((n, embed_dim - dim), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)

    return embeddings.astype(np.float32)


# ================================================================
# Mesh utilities
# ================================================================

def normalize_mesh(mesh):
    vertices = np.array(mesh.vertices)
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    max_extent = np.abs(vertices).max()
    scale = 1.0 / max_extent
    vertices *= scale
    normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    transform = {"centroid": centroid, "scale": scale}
    return normalized, transform


def denormalize_mesh(mesh, bbox_dims):
    vertices = np.array(mesh.vertices)
    current_extent = vertices.max(axis=0) - vertices.min(axis=0)
    target_extent = np.array(bbox_dims)
    scale = target_extent / (current_extent + 1e-8)
    vertices *= scale
    vertices -= vertices.mean(axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# ================================================================
# SDF sampling (pure numpy, no rtree)
# ================================================================

def sample_sdf_points(mesh, num_points, surface_ratio=0.5):
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    num_surface = int(num_points * surface_ratio)
    num_uniform = num_points - num_surface

    num_ref_points = min(500000, max(100000, len(vertices) * 50))
    ref_points = _sample_surface_simple(vertices, faces, num_ref_points)
    tree = KDTree(ref_points)

    surface_points = _sample_surface_simple(vertices, faces, num_surface)
    num_tight = int(num_surface * 0.7)
    noise_tight = np.random.normal(0, 0.003, (num_tight, 3))
    noise_wide = np.random.normal(0, 0.03, (num_surface - num_tight, 3))
    noise = np.concatenate([noise_tight, noise_wide], axis=0)
    near_surface_points = surface_points + noise

    uniform_points = np.random.uniform(-1, 1, (num_uniform, 3))
    all_points = np.concatenate([near_surface_points, uniform_points], axis=0)
    distances, _ = tree.query(all_points)

    inside = _ray_casting_contains(vertices, faces, all_points)
    signs = np.ones(len(all_points), dtype=np.float64)
    signs[inside] = -1.0
    sdf_values = signs * distances

    return all_points.astype(np.float32), sdf_values.astype(np.float32)


def _ray_casting_contains(vertices, faces, points, batch_size=10000):
    triangles = vertices[faces]
    ray_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    votes = np.zeros(len(points), dtype=np.int32)
    for ray_dir in ray_dirs:
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            counts = _count_ray_intersections(batch, triangles, ray_dir)
            votes[i:i + batch_size] += (counts % 2 == 1).astype(np.int32)
    return votes >= 2


def _count_ray_intersections(points, triangles, ray_dir):
    P = len(points)
    counts = np.zeros(P, dtype=np.int32)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0
    F = len(triangles)
    h = np.cross(np.broadcast_to(ray_dir, (F, 3)), edge2)
    a = np.sum(edge1 * h, axis=1)
    eps = 1e-10
    valid_idx = np.where(np.abs(a) > eps)[0]
    if len(valid_idx) == 0:
        return counts

    f_inv = 1.0 / a[valid_idx]
    e1_v, v0_v, h_v = edge1[valid_idx], v0[valid_idx], h[valid_idx]
    e2_v = edge2[valid_idx]

    sub_batch = min(500, P)
    for pi in range(0, P, sub_batch):
        pj = min(pi + sub_batch, P)
        pts = points[pi:pj]
        S = pj - pi
        s = pts[:, None, :] - v0_v[None, :, :]
        u = f_inv[None, :] * np.sum(s * h_v[None, :, :], axis=2)
        u_valid = (u >= 0.0) & (u <= 1.0)
        q = np.cross(s, e1_v[None, :, :])
        v = f_inv[None, :] * np.sum(
            np.broadcast_to(ray_dir, (S, len(valid_idx), 3)) * q, axis=2)
        v_valid = (v >= 0.0) & ((u + v) <= 1.0)
        t = f_inv[None, :] * np.sum(e2_v[None, :, :] * q, axis=2)
        hit = u_valid & v_valid & (t > eps)
        counts[pi:pj] = np.sum(hit, axis=1)

    return counts


def _sample_surface_simple(vertices, faces, num_samples):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()
    sampled_faces = np.random.choice(len(faces), size=num_samples, p=probs)
    r1, r2 = np.random.rand(num_samples, 1), np.random.rand(num_samples, 1)
    sqrt_r1 = np.sqrt(r1)
    bary_a, bary_b, bary_c = 1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2
    tri_verts = vertices[faces[sampled_faces]]
    return bary_a * tri_verts[:, 0] + bary_b * tri_verts[:, 1] + bary_c * tri_verts[:, 2]
