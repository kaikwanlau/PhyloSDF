"""Method-separated PhyloSDF geometry and audited handcrafted MorphFID.

Inputs are immutable decode summaries and exactly the frozen reference cohort.
No training, checkpoint selection, mesh repair or component removal occurs here.
External archival is the caller's responsibility.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import time
import warnings

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree
import trimesh

FEATURE_NAMES = ("surface_area", "volume", "bb_x", "bb_y", "bb_z", "elongation",
                 "flatness", "compactness", "n_vertices", "centroid_x", "centroid_y", "centroid_z")
FEATURE_SUBSETS = {"all12": tuple(range(12)), "no_vertex_count11": tuple(i for i in range(12) if i != 8),
                   "geometry8_no_count_or_centroid": tuple(range(8))}
HISTORICAL_SOURCE_SHA256 = "15dde6aa115638da3152e733388212a2e0e590f18c02cd087e350a8f4eadeada"
METHODS = ("pooled_residual_CFM", "empirical_residual", "pooled_full_gaussian")
SURFACE_SEEDS = (20261014, 20261015)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def checked_path(base, value, expected):
    path = Path(value)
    path = path if path.is_absolute() else Path(base) / path
    path = path.resolve()
    if sha(path) != expected:
        raise ValueError("Input checksum mismatch: " + str(path))
    return path


def read_binary_stl(path):
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("STL too short")
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + count * 50:
        raise ValueError("Expected exact binary STL byte count")
    dtype = np.dtype([("normal", "<f4", 3), ("triangles", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["triangles"].copy()
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(np.float64), inverse.reshape(-1, 3).astype(np.int32)


def validate_mesh(vertices, faces):
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected triangle mesh arrays")
    if len(v) < 3 or not len(f) or not np.isfinite(v).all() or f.dtype.kind not in "iu":
        raise ValueError("Empty, nonfinite or noninteger mesh")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("Mesh face index out of bounds")
    return v, f.astype(np.int64)


def topology(vertices, faces):
    v, f = validate_mesh(vertices, faces)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.sort(np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(v), len(v)))
    t = v[f]
    areas = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Invalid or zero-area mesh")
    return {"vertices": len(v), "faces": len(f), "open_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "connected_components": int(connected_components(graph, directed=False, return_labels=False)),
            "degenerate_faces": int((areas <= 1e-14).sum()), "area": float(areas.sum()),
            "boundary_contact_vertices": int(np.isclose(np.abs(v), 1, atol=2e-6, rtol=0).any(1).sum()),
            "bounds": [v.min(0).tolist(), v.max(0).tolist()], "component_filtering": "none"}


def sample_surface(vertices, faces, n, seed):
    """Same area/Barycentric formula as frozen benchmark; no re-normalization."""
    v, f = validate_mesh(vertices, faces)
    rng = np.random.default_rng(seed)
    tri = v[f]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Invalid sampling surface")
    t = tri[rng.choice(len(f), n, p=areas / areas.sum())]
    u, w = np.sqrt(rng.random(n)), rng.random(n)
    return (1 - u[:, None]) * t[:, 0] + (u * (1 - w))[:, None] * t[:, 1] + (u * w)[:, None] * t[:, 2]


def cd(left, right, left_tree=None, right_tree=None):
    left_tree = cKDTree(left) if left_tree is None else left_tree
    right_tree = cKDTree(right) if right_tree is None else right_tree
    value = float(np.square(left_tree.query(right, workers=1)[0]).mean()
                  + np.square(right_tree.query(left, workers=1)[0]).mean())
    if not np.isfinite(value) or value < 0:
        raise ValueError("Invalid Chamfer value")
    return value


def handcrafted_features(vertices, faces):
    """Exact twelve executable historical descriptors, preserving hull fallback."""
    vertices, faces = validate_mesh(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=True)
    area = float(mesh.area)
    fallback = not bool(mesh.is_watertight)
    volume = float(mesh.convex_hull.volume if fallback else mesh.volume)
    bb = mesh.bounding_box.extents
    b = sorted(bb)
    values = np.array([area, volume, *bb.tolist(), b[2] / (b[0] + 1e-8), b[1] / (b[0] + 1e-8),
                       (36 * np.pi * volume ** 2) ** (1 / 3) / (area + 1e-8), len(mesh.vertices),
                       *mesh.centroid.tolist()], dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("Nonfinite historical MorphFID feature row")
    return values, {"watertight": bool(mesh.is_watertight), "volume_fallback": "convex_hull" if fallback else None,
                    "signed_volume_negative": volume < 0, "winding_consistent": bool(mesh.is_winding_consistent),
                    "vertices_before_trimesh_processing": len(vertices), "vertices_after_trimesh_processing": len(mesh.vertices),
                    "faces_after_trimesh_processing": len(mesh.faces)}


def species_weights(records):
    counts = collections.Counter(str(x["species"]) for x in records)
    return np.array([1 / (len(counts) * counts[str(x["species"])]) for x in records], dtype=np.float64)


def distribution(values, weights=None):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("A covariance needs two finite feature rows")
    w = np.full(len(x), 1 / len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid feature weights")
    w = w / w.sum()
    mean = w @ x
    centred = x - mean
    denom = 1 - float(w @ w)
    if denom <= 0:
        raise ValueError("Covariance weights have zero effective degrees of freedom")
    cov = (centred.T * w) @ centred / denom
    cov = (cov + cov.T) / 2
    eigen = np.linalg.eigvalsh(cov)
    return mean, cov, {"rows": len(x), "dimension": x.shape[1], "effective_sample_count": float(1 / (w @ w)),
                       "rank": int(np.linalg.matrix_rank(cov)), "minimum_eigenvalue": float(eigen.min()),
                       "maximum_eigenvalue": float(eigen.max()), "rank_deficient": int(np.linalg.matrix_rank(cov)) < x.shape[1],
                       "covariance_convention": "weighted unbiased with denominator 1-sum(normalized_weights^2)"}


def psd_sqrt(matrix):
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2
    eigen, vec = np.linalg.eigh(matrix)
    tolerance = 1e-10 * max(1., float(np.max(np.abs(eigen))))
    if float(eigen.min()) < -tolerance:
        raise ValueError("Covariance is not positive semidefinite")
    return (vec * np.sqrt(np.maximum(eigen, 0))) @ vec.T


def frechet(left_mean, left_cov, right_mean, right_cov):
    """Real PSD Bures calculation plus checked historical sqrtm comparison."""
    diff = np.asarray(left_mean) - np.asarray(right_mean)
    root = psd_sqrt(left_cov)
    middle = (root @ right_cov @ root)
    eigen = np.linalg.eigvalsh((middle + middle.T) / 2)
    tol = 1e-10 * max(1., float(np.max(np.abs(eigen))))
    if eigen.min() < -tol:
        raise ValueError("Invalid covariance sandwich")
    cross_trace = float(np.sqrt(np.maximum(eigen, 0)).sum())
    scale = float(diff @ diff + np.trace(left_cov) + np.trace(right_cov))
    result = scale - 2 * cross_trace
    if not np.isfinite(result) or result < -1e-8 * max(scale, 1):
        raise ValueError("Invalid Fréchet score")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        historical = sqrtm(left_cov @ right_cov)
    imaginary = float(np.max(np.abs(historical.imag))) if np.iscomplexobj(historical) else 0.
    legacy = float(diff @ diff + np.trace(left_cov + right_cov - 2 * historical.real))
    if not np.isfinite(legacy) or not np.isfinite(imaginary):
        raise ValueError("Nonfinite historical covariance square root")
    tolerance = 2e-6 * max(1., abs(result), scale)
    return max(0., float(result)), {"stable_psd_value_before_roundoff_clamp": float(result),
                                  "historical_sqrtm_real_part_value": legacy, "sqrtm_max_abs_imaginary": imaginary,
                                  "stable_vs_historical_absolute_difference": abs(result - legacy),
                                  "historical_agrees_at_recorded_tolerance": abs(result - legacy) <= tolerance,
                                  "agreement_absolute_tolerance": tolerance,
                                  "historical_sqrtm_warnings": [str(x.message) for x in recorded],
                                  "roundoff_clamp_only": result < 0}


def normalization(features, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = w @ features
    std_unregularized = np.sqrt(w @ np.square(features - mean))
    return mean, std_unregularized + 1e-8, {"mean": mean.tolist(), "population_std_before_epsilon": std_unregularized.tolist(),
                                         "std_with_epsilon": (std_unregularized + 1e-8).tolist(), "epsilon": 1e-8,
                                         "near_zero_std_indices": np.flatnonzero(std_unregularized <= 1e-8).tolist(),
                                         "fitted_on": "all frozen100 real references only; never generated meshes"}


def score_features(reference, generated, norm, reference_weights, subset):
    indices = FEATURE_SUBSETS[subset]
    mu, std = norm
    r = ((reference - mu) / std)[:, indices]
    g = ((generated - mu) / std)[:, indices]
    rm, rc, rd = distribution(r, reference_weights)
    gm, gc, gd = distribution(g)
    value, check = frechet(rm, rc, gm, gc)
    return {"value": value, "features": [FEATURE_NAMES[i] for i in indices], "reference_covariance": rd,
            "generated_covariance": gd, "square_root_check": check,
            "reference_mean": rm.tolist(), "generated_mean": gm.tolist(),
            "reference_covariance_matrix": rc.tolist(), "generated_covariance_matrix": gc.tolist(),
            "inference": "descriptive fitted-distribution score; finite-sample bias and covariance singularity retained"}


def load_references(manifest):
    manifest = Path(manifest).resolve()
    cfg = json.loads(manifest.read_text())
    rows = cfg["references"]
    if len(rows) != 100 or cfg.get("expected_specimens", 100) != 100 or len({x["specimen"] for x in rows}) != len(rows):
        raise ValueError("Incomplete or duplicated locked reference cohort")
    if cfg.get("scope", "training") != "training":
        raise ValueError("This evaluator's full reference cohort is training data")
    loaded, transforms = [], []
    for row in rows:
        raw = checked_path(manifest.parent, row["raw_stl"], row["raw_sha256"])
        processed = checked_path(manifest.parent, row["processed_npz"], row["processed_sha256"])
        v, f = read_binary_stl(raw)
        with np.load(processed, allow_pickle=False) as data:
            center, scale = np.array(data["centroid"]), float(data["scale"])
            center_error = float(np.abs(v.mean(0) - center).max())
            scale_error = float(abs(1 / np.abs(v - v.mean(0)).max() - scale))
            if center_error > 1e-10 or scale_error > 1e-12 or len(v) != int(data["num_vertices"]) or len(f) != int(data["num_faces"]):
                raise ValueError("Raw/saved preprocessing transform mismatch: " + row["specimen"])
        normalized = (v - center) * scale
        ref = {**row, "raw_path": raw, "processed_path": processed, "vertices": normalized, "faces": f}
        ref["features_decoder_frame"], ref["feature_flags_decoder_frame"] = handcrafted_features(normalized, f)
        ref["features_physical_frame"], ref["feature_flags_physical_frame"] = handcrafted_features(v, f)
        ref["topology"] = topology(normalized, f)
        loaded.append(ref)
        transforms.append({"specimen": row["specimen"], "species": row["species"], "centroid": center.tolist(), "scale_multiplier": scale,
                           "centroid_max_abs_error": center_error, "scale_abs_error": scale_error,
                           "raw_vertices": len(v), "raw_faces": len(f), "decoder_frame_bounds": ref["topology"]["bounds"]})
    return loaded, transforms


def load_decode(decode):
    decode = Path(decode).resolve()
    summary_path, config_path = decode / "summary.json", decode / "run_config.json"
    summary, config = json.loads(summary_path.read_text()), json.loads(config_path.read_text())
    records = summary["records"]
    tasks = config.get("tasks", config.get("task_manifest", []))
    if not tasks and (decode / "preselection.json").exists():
        tasks = json.loads((decode / "preselection.json").read_text())["tasks"]
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    if tasks and {x["task"] for x in records} != {x["task"] for x in tasks}:
        raise ValueError("Decode summary does not cover every committed task")
    if len({x["task"] for x in records}) != len(records):
        raise ValueError("Duplicated decode task")
    loaded = []
    for row in records:
        if row.get("method") not in (*METHODS, "reference_reconstruction", "paired_reference_reconstruction") and not str(row.get("method", "")).startswith("historical_"):
            raise ValueError("Unknown method or reconstruction kind in decode manifest")
        rec = {**row, "decode_root": str(decode), "paired_reference": row.get("paired_reference", config.get("paired_reference", config.get("representation"))),
               "training_seed": row.get("training_seed", config.get("training_seed")), "evaluation_status": row["status"]}
        if row["status"] == "completed":
            path = checked_path(decode, row.get("mesh_path", row["task"] + "/mesh.npz"), row["output_sha256"]["mesh.npz"])
            with np.load(path, allow_pickle=False) as mesh:
                v, f = validate_mesh(mesh["vertices"], mesh["faces"])
            rec.update(vertices=v, faces=f, mesh_path=str(path), topology=topology(v, f))
            try:
                rec["features"], rec["feature_flags"] = handcrafted_features(v, f)
            except Exception as error:
                rec.update(feature_failure={"type": type(error).__name__, "message": str(error)})
        loaded.append(rec)
    return loaded, {"decode_root": str(decode), "summary_sha256": sha(summary_path), "config_sha256": sha(config_path), "records": len(records)}


def metric_from_matrix(matrix, reference_weights):
    matrix, w = np.asarray(matrix), np.asarray(reference_weights)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Incomplete distance matrix")
    nearest_reference = matrix.argmin(1)
    nearest_generated = matrix.argmin(0)
    covered = np.unique(nearest_reference)
    return {"generated_to_reference_nn_CD_mean": float(matrix.min(1).mean()),
            "MMD_CD_reference_to_generated": float(w @ matrix.min(0)),
            "COV_reference_fraction": float(len(covered) / len(w)),
            "COV_species_balanced_reference_weight": float(w[covered].sum()),
            "covered_reference_indices": covered.tolist(), "generated_nearest_reference_indices": nearest_reference.tolist(),
            "reference_nearest_generated_indices": nearest_generated.tolist(),
            "nearest_generated_CD_each_reference": matrix.min(0).tolist(),
            "nearest_reference_CD_each_generated": matrix.min(1).tolist()}


def compact_record(row):
    return {k: v for k, v in row.items() if k not in ("vertices", "faces", "features", "raw_path", "processed_path",
                                                    "features_decoder_frame", "features_physical_frame")}


def run(references, decode, out, points=10000, surface_seeds=SURFACE_SEEDS, resume=False, sensitivity_decode=None):
    start = time.perf_counter()
    references, out = Path(references).resolve(), Path(out).resolve()
    decode = [Path(x).resolve() for x in decode]
    if points < 2 or len(surface_seeds) != 2 or len(set(surface_seeds)) != 2:
        raise ValueError("Need two fixed distinct area-sampling seeds and at least two points")
    if any(out.is_relative_to(x) for x in [references.parent, *decode]):
        raise ValueError("Evaluation output must be outside immutable input trees")
    ref, transforms = load_references(references)
    generated, input_decodes = [], []
    for root in decode:
        rows, manifest = load_decode(root)
        generated.extend(rows)
        input_decodes.append(manifest)
    sensitivity = []
    input_sensitivity = []
    for root in sensitivity_decode or []:
        rows, manifest = load_decode(root)
        sensitivity.extend(rows)
        input_sensitivity.append(manifest)
    key_rows = [(x["decode_root"], x["task"]) for x in generated]
    if len(set(key_rows)) != len(key_rows):
        raise ValueError("Duplicate decode input")
    config = {"schema_version": 7, "source_sha256": sha(__file__), "reference_manifest_sha256": sha(references),
              "input_decodes": input_decodes, "input_sensitivity_decodes": input_sensitivity,
              "points_per_surface": points, "surface_seeds": list(surface_seeds),
              "reference_cohort": [x["specimen"] for x in ref], "reference_count": len(ref),
              "methods": list(METHODS), "historical_morphfid_source_sha256": HISTORICAL_SOURCE_SHA256,
              "frame": "raw references=(raw-saved centroid)*saved scale; generated vertices unchanged; no per-mesh centering or scaling",
              "CD": "sum of each directional mean squared nearest sampled-point distance",
              "MMD_CD": "reference-to-generated nearest CD weighted mean; not kernel maximum mean discrepancy",
              "COV": "fraction and species-balanced weight of reference meshes selected by generated nearest-reference argmin; ties first locked index",
              "diversity": "every unordered valid-generated pair in a cohort; all components retained",
              "mixture_weights": "reference uniform species then uniform specimen within species; generated locked uniform-species schedule",
              "conditional_weights": "uniform training specimen within target species",
              "MorphFID": "12 exact handcrafted executable descriptors, global100 reference-only zscore, unbiased weighted covariance and checked real PSD Frechet",
              "MorphFID_primary": "decoder-frame, matching cohort references, species-balanced global zscore; full12 plus fixed count/centroid omission sensitivities",
              "MorphFID_historical_replication": "physical raw100 vs normalized generated, specimen-uniform100 global zscore; flagged incompatible coordinate frame",
              "failure_policy": "all attempts preserved; geometric/MorphFID scores explicitly conditional on finite valid meshes/features; no silent omission",
              "component_filtering": "none", "checkpoint_selection": "caller fixed final generator at fixed budget before evaluation",
              "scope": "known species and training-reference shape distribution comparison; not heldout generalization",
              "uncertainty": "two surface seeds show sampling sensitivity; across training seeds descriptive dispersion; no confidence intervals implied",
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "trimesh")}, "python": platform.python_version()}
    lock_path = out / "evaluation_config.json"
    if out.exists() and any(out.iterdir()):
        if not resume or not lock_path.exists() or json.loads(lock_path.read_text()) != config:
            raise ValueError("Use new empty output or resume identical locked evaluation")
    out.mkdir(parents=True, exist_ok=True)
    atomic(lock_path, config)
    atomic(out / "transform_checks.json", transforms)
    atomic(out / "attempt_audit.json", [compact_record(x) for x in generated])
    rf = np.stack([x["features_decoder_frame"] for x in ref])
    rawf = np.stack([x["features_physical_frame"] for x in ref])
    balanced = species_weights(ref)
    norm_mu, norm_std, norm_info = normalization(rf, balanced)
    raw_mu, raw_std, raw_info = normalization(rawf, np.full(len(ref), 1 / len(ref)))
    atomic(out / "normalization_stats.json", {"decoder_frame_primary": norm_info, "historical_physical_frame": raw_info,
            "feature_names": list(FEATURE_NAMES), "decoder_frame_reference_weights": balanced.tolist()})
    np.savez_compressed(out / "reference_features.npz", decoder_frame=rf, physical_frame=rawf,
                        specimen=np.asarray([x["specimen"] for x in ref]), species=np.asarray([x["species"] for x in ref]))
    atomic(out / "reference_feature_flags.json", [{"specimen": x["specimen"], "decoder": x["feature_flags_decoder_frame"],
                                                   "physical": x["feature_flags_physical_frame"], "topology": x["topology"]} for x in ref])
    samples_dir = out / "surface_samples"
    samples_dir.mkdir(exist_ok=True)
    cache = {}

    def clouds(row, reference=False):
        digest = (row["raw_sha256"] + row["processed_sha256"]) if reference else row["output_sha256"]["mesh.npz"]
        key = hashlib.sha256((digest + "_" + str(points) + "_" + str(tuple(surface_seeds))).encode()).hexdigest()
        if key not in cache:
            path = samples_dir / (key + ".npz")
            sidecar = samples_dir / (key + ".json")
            if path.exists():
                if not sidecar.exists():
                    raise ValueError("Cached samples lack integrity sidecar")
                cached = json.loads(sidecar.read_text())
                if cached["source_digest"] != digest or cached["points"] != points or cached["seeds"] != list(surface_seeds):
                    raise ValueError("Cached surface sampling contract mismatch")
                checked_path(samples_dir, path.name, cached["sha256"])
                with np.load(path, allow_pickle=False) as saved:
                    values = np.array(saved["points"])
                if values.shape != (2, points, 3) or not np.isfinite(values).all():
                    raise ValueError("Invalid cached surface samples")
            else:
                values = np.stack([sample_surface(row["vertices"], row["faces"], points, seed) for seed in surface_seeds])
                np.savez_compressed(path, points=values)
                atomic(sidecar, {"source_digest": digest, "points": points, "seeds": list(surface_seeds), "sha256": sha(path)})
            cache[key] = (values, [cKDTree(x) for x in values])
        return cache[key]

    reference_clouds = [clouds(x, True) for x in ref]
    groups = collections.defaultdict(list)
    reconstruction = []
    for row in generated:
        if row["method"] in ("reference_reconstruction", "paired_reference_reconstruction"):
            reconstruction.append(row)
        else:
            groups[(str(row.get("group", "generated")), str(row.get("historical_arm", "primary")),
                    str(row["paired_reference"]), row["method"], row["cohort"], row["training_seed"])].append(row)
    results = []
    for (study_group, arm, pair, method, cohort, training_seed), attempts in sorted(groups.items(), key=lambda x: str(x[0])):
        attempts.sort(key=lambda x: (int(x["sample_slot"]), x["task"]))
        if len({int(x["sample_slot"]) for x in attempts}) != len(attempts):
            raise ValueError("Duplicated method/cohort slots")
        if cohort.startswith("conditional_"):
            species = cohort[len("conditional_"):]
            if any(str(x.get("species")) != species for x in attempts):
                raise ValueError("Conditional cohort/species mismatch")
            ref_indices = [i for i, row in enumerate(ref) if row["species"] == species]
            weights = np.full(len(ref_indices), 1 / len(ref_indices)) if ref_indices else None
        elif cohort in ("mixture", "species_mixture", "uniform_species_mixture", "unconditional_species_mixture"):
            ref_indices, weights = list(range(len(ref))), balanced
        else:
            raise ValueError("Unknown cohort: " + cohort)
        if not ref_indices:
            raise ValueError("No matched training reference cohort: " + cohort)
        valid = [x for x in attempts if x["evaluation_status"] == "completed"]
        valid_features = [x for x in valid if "features" in x]
        group_id = hashlib.sha256(json.dumps([study_group, arm, pair, method, cohort, training_seed]).encode()).hexdigest()[:20]
        group_out = out / "groups" / group_id
        group_out.mkdir(parents=True, exist_ok=True)
        completed_path = group_out / "COMPLETED.json"
        if completed_path.exists():
            complete = json.loads(completed_path.read_text())
            for filename, digest in complete["files"].items():
                checked_path(group_out, filename, digest)
            results.append(json.loads((group_out / "summary.json").read_text()))
            print("reused completed metric group", group_id, flush=True)
            continue
        group_start = time.perf_counter()
        summary = {"group_id": group_id, "study_group": study_group, "arm": arm,
                   "paired_reference": pair, "method": method, "cohort": cohort,
                   "training_seed": training_seed, "attempts": len(attempts), "valid_meshes": len(valid),
                   "failure_rate": 1 - len(valid) / len(attempts), "valid_feature_rows": len(valid_features),
                   "open_meshes": sum(x["topology"]["open_edges"] > 0 for x in valid),
                   "multi_component_meshes": sum(x["topology"]["connected_components"] > 1 for x in valid),
                   "reference_specimens": [ref[i]["specimen"] for i in ref_indices], "reference_weights": weights.tolist(),
                   "failure_task_ids": [x["task"] for x in attempts if x["evaluation_status"] != "completed"],
                   "feature_failure_task_ids": [x["task"] for x in valid if "features" not in x],
                   "inclusion": "every valid extracted mesh incl open/multiple-component surfaces; scores conditional on successful geometry; failure denominator all attempts"}
        if valid:
            cl = [clouds(x) for x in valid]
            distance = np.empty((2, len(valid), len(ref_indices)), dtype=np.float64)
            diversity = np.empty((2, len(valid) * (len(valid) - 1) // 2), dtype=np.float64)
            distance_times, diversity_times = [], []
            for seed_i in range(2):
                seed_start = time.perf_counter()
                for i, (pts, trees) in enumerate(cl):
                    for j, ref_i in enumerate(ref_indices):
                        rpts, rtrees = reference_clouds[ref_i]
                        distance[seed_i, i, j] = cd(pts[seed_i], rpts[seed_i], trees[seed_i], rtrees[seed_i])
                    if (i + 1) % 10 == 0 or i + 1 == len(cl):
                        atomic(group_out / "progress.json", {"stage": "generated_reference_distances", "surface_seed": surface_seeds[seed_i],
                                "generated_rows_complete": i + 1, "generated_rows_total": len(cl), "group_elapsed_seconds": time.perf_counter() - group_start, "utc": utc()})
                        print("metric progress", group_id, "seed", surface_seeds[seed_i], "rows", i + 1, "/", len(cl), flush=True)
                distance_times.append(time.perf_counter() - seed_start)
                diversity_start = time.perf_counter()
                k = 0
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        diversity[seed_i, k] = cd(cl[i][0][seed_i], cl[j][0][seed_i], cl[i][1][seed_i], cl[j][1][seed_i])
                        k += 1
                    if (i + 1) % 10 == 0:
                        atomic(group_out / "progress.json", {"stage": "generated_pairwise_diversity", "surface_seed": surface_seeds[seed_i],
                                "pairs_complete": k, "pairs_total": diversity.shape[1], "group_elapsed_seconds": time.perf_counter() - group_start, "utc": utc()})
                        print("metric diversity progress", group_id, "seed", surface_seeds[seed_i], "pairs", k, "/", diversity.shape[1], flush=True)
                diversity_times.append(time.perf_counter() - diversity_start)
                atomic(group_out / "progress.json", {"completed_surface_seeds": seed_i + 1, "total_surface_seeds": 2, "utc": utc()})
            np.savez_compressed(group_out / "distance_matrices.npz", generated_to_reference_CD=distance, pairwise_generated_CD=diversity,
                                valid_tasks=np.asarray([x["task"] for x in valid]), sample_slots=np.asarray([x["sample_slot"] for x in valid]),
                                references=np.asarray(summary["reference_specimens"]), reference_weights=weights, surface_seeds=np.asarray(surface_seeds))
            seed_scores = []
            for seed_i, seed in enumerate(surface_seeds):
                result = metric_from_matrix(distance[seed_i], weights)
                result.update(surface_seed=seed, diversity_all_pairs_CD_mean=float(diversity[seed_i].mean()) if diversity.shape[1] else None,
                              diversity_pairs=diversity.shape[1])
                seed_scores.append(result)
            summary["geometry_by_surface_seed"] = seed_scores
            summary["distance_matrix_seconds_by_surface_seed"] = distance_times
            summary["pairwise_diversity_seconds_by_surface_seed"] = diversity_times
            numeric = ("generated_to_reference_nn_CD_mean", "MMD_CD_reference_to_generated", "COV_reference_fraction", "COV_species_balanced_reference_weight", "diversity_all_pairs_CD_mean")
            summary["geometry_surface_seed_summary"] = {k: {"mean": float(np.mean([x[k] for x in seed_scores])),
                        "range": float(np.ptp([x[k] for x in seed_scores]))} for k in numeric if seed_scores[0][k] is not None}
        if len(valid_features) >= 2:
            gf = np.stack([x["features"] for x in valid_features])
            np.savez_compressed(group_out / "generated_features.npz", features=gf, task=np.asarray([x["task"] for x in valid_features]),
                                sample_slot=np.asarray([x["sample_slot"] for x in valid_features]), feature_names=np.asarray(FEATURE_NAMES))
            summary["morphfid_decoder_frame_primary"] = {k: score_features(rf[ref_indices], gf, (norm_mu, norm_std), weights, k) for k in FEATURE_SUBSETS} if len(ref_indices) >= 2 else {"status": "undefined_less_than_two_references"}
            summary["morphfid_historical_frame_replication"] = score_features(rawf, gf, (raw_mu, raw_std), np.full(len(ref), 1 / len(ref)), "all12")
            summary["morphfid_historical_frame_replication"]["validity"] = "incompatible physical/decoder coordinate frames; diagnostic historical-formula replication, not a fair quality comparison"
        else:
            summary["morphfid_decoder_frame_primary"] = {"status": "undefined_less_than_two_valid_generated_feature_rows"}
        summary["elapsed_seconds_since_evaluation_start"] = time.perf_counter() - start
        summary["group_wall_seconds"] = time.perf_counter() - group_start
        atomic(group_out / "summary.json", summary)
        atomic(completed_path, {"status": "complete", "files": {str(p.relative_to(group_out)): sha(p)
                                for p in sorted(group_out.rglob("*")) if p.is_file() and p.name != "COMPLETED.json"}})
        results.append(summary)
        atomic(out / "progress.json", {"complete_groups": len(results), "expected_groups": len(groups), "utc": utc()})
        print("evaluated", pair, method, cohort, training_seed, len(valid), "/", len(attempts), flush=True)
        # Release potentially many generated clouds/trees while reference caches stay small.
        keep = {hashlib.sha256((x["raw_sha256"] + x["processed_sha256"] + "_" + str(points) + "_" + str(tuple(surface_seeds))).encode()).hexdigest() for x in ref}
        cache = {k: v for k, v in cache.items() if k in keep}

    recon_results = []
    by_specimen = {x["specimen"]: i for i, x in enumerate(ref)}
    for row in reconstruction:
        record = compact_record(row)
        if row["evaluation_status"] == "completed":
            if row["specimen"] not in by_specimen:
                raise ValueError("Reconstruction reference not in frozen cohort")
            i = by_specimen[row["specimen"]]
            pts, trees = clouds(row)
            refpts, reftrees = reference_clouds[i]
            values = [cd(pts[k], refpts[k], trees[k], reftrees[k]) for k in range(2)]
            record.update(matched_specimen_CD_by_surface_seed=values, mean_matched_specimen_CD=float(np.mean(values)),
                          CD_surface_seed_range=float(np.ptp(values)))
        recon_results.append(record)
    atomic(out / "reconstruction_per_specimen.json", recon_results)
    sensitivity_results = []
    primary_tasks = {x["task"]: x for x in generated}
    for row in sensitivity:
        if row["task"] not in primary_tasks:
            raise ValueError("Sensitivity task not in committed primary schedule")
        primary = primary_tasks[row["task"]]
        if row["decoder_code_sha256"] != primary["decoder_code_sha256"] or row["paired_reference"] != primary["paired_reference"]:
            raise ValueError("Resolution sensitivity changed the decoder/code pairing")
        record = {"task": row["task"], "paired_reference": row["paired_reference"], "method": row["method"],
                  "cohort": row["cohort"], "sample_slot": row["sample_slot"], "training_seed": row["training_seed"],
                  "primary_status": primary["evaluation_status"], "sensitivity_status": row["evaluation_status"],
                  "primary_resolution": primary["resolution"], "sensitivity_resolution": row["resolution"],
                  "decoder_code_sha256": row["decoder_code_sha256"], "selection": "fixed focal slot0 subset; no per-result selection"}
        if primary["evaluation_status"] == row["evaluation_status"] == "completed":
            pts, trees = clouds(primary)
            high_pts, high_trees = clouds(row)
            vals = [cd(pts[k], high_pts[k], trees[k], high_trees[k]) for k in range(2)]
            record.update(between_resolutions_CD_by_surface_seed=vals,
                          between_resolutions_CD_mean=float(np.mean(vals)),
                          primary_topology=primary["topology"], sensitivity_topology=row["topology"])
            if "features" in primary and "features" in row:
                record.update(primary_features=primary["features"].tolist(), sensitivity_features=row["features"].tolist(),
                              feature_delta=(row["features"] - primary["features"]).tolist())
            if row["method"] in ("reference_reconstruction", "paired_reference_reconstruction"):
                i = by_specimen[row["specimen"]]
                refpts, reftrees = reference_clouds[i]
                record["primary_reference_CD_by_surface_seed"] = [cd(pts[k], refpts[k], trees[k], reftrees[k]) for k in range(2)]
                record["sensitivity_reference_CD_by_surface_seed"] = [cd(high_pts[k], refpts[k], high_trees[k], reftrees[k]) for k in range(2)]
        sensitivity_results.append(record)
    atomic(out / "resolution_sensitivity.json", {"records": sensitivity_results, "tasks": len(sensitivity_results),
            "scope": "fixed focal32 resolution sensitivity; does not constitute full256 evaluation or establish resolution superiority",
            "feature_names": list(FEATURE_NAMES)})
    dispersion = []
    dispersion_groups = collections.defaultdict(list)
    for group in results:
        if group["study_group"] == "generated" and group["method"] == "pooled_residual_CFM":
            dispersion_groups[(group["paired_reference"], group["cohort"])].append(group)
    for (pair, cohort), groups_seed in sorted(dispersion_groups.items()):
        by_seed = sorted(groups_seed, key=lambda x: x["training_seed"])
        values = {"failure_rate": [x["failure_rate"] for x in by_seed]}
        for metric in ("generated_to_reference_nn_CD_mean", "MMD_CD_reference_to_generated", "COV_reference_fraction", "COV_species_balanced_reference_weight", "diversity_all_pairs_CD_mean"):
            if all(metric in x.get("geometry_surface_seed_summary", {}) for x in by_seed):
                values[metric] = [x["geometry_surface_seed_summary"][metric]["mean"] for x in by_seed]
        for subset in FEATURE_SUBSETS:
            if all(subset in x["morphfid_decoder_frame_primary"] for x in by_seed):
                values["MorphFID_" + subset] = [x["morphfid_decoder_frame_primary"][subset]["value"] for x in by_seed]
        dispersion.append({"paired_reference": pair, "cohort": cohort, "training_seeds": [x["training_seed"] for x in by_seed],
                           "metrics": {k: {"values": v, "mean": float(np.mean(v)), "minimum": float(np.min(v)), "maximum": float(np.max(v)),
                                           "sample_std": float(np.std(v, ddof=1)) if len(v) > 1 else None} for k, v in values.items()},
                           "interpretation": "descriptive across training seeds; controls share identical generated code schedule; no confidence interval or convergence claim"})
    atomic(out / "training_seed_dispersion.json", dispersion)
    # Full input rechecks keep the final record tied to the same immutable bytes.
    for row in ref:
        checked_path(references.parent, row["raw_stl"], row["raw_sha256"])
        checked_path(references.parent, row["processed_npz"], row["processed_sha256"])
    if sha(references) != config["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed during evaluation")
    for manifest in [*input_decodes, *input_sensitivity]:
        checked_path(manifest["decode_root"], "summary.json", manifest["summary_sha256"])
        checked_path(manifest["decode_root"], "run_config.json", manifest["config_sha256"])
    for row in [*generated, *sensitivity]:
        if row["evaluation_status"] == "completed":
            checked_path(row["decode_root"], row["mesh_path"], row["output_sha256"]["mesh.npz"])
    overall = {"schema_version": 7, "status": "all_committed_method_groups_evaluated", "created_utc": utc(),
               "config_sha256": sha(lock_path), "elapsed_seconds": time.perf_counter() - start, "groups": results,
               "reconstruction_attempts": len(reconstruction), "reconstruction_valid_meshes": sum(x["evaluation_status"] == "completed" for x in reconstruction),
               "sensitivity_attempts": len(sensitivity_results),
               "all_input_hashes_reverified_after_evaluation": True,
               "scope": config["scope"], "MorphFID_assets": "historical executable12 descriptors; no learned architecture, weights, PCA or curvature features used",
               "interpretation": "fixed-budget matched comparison; convergence and independent generalization are separate claims; open-mesh hull fallbacks and tessellation-sensitive descriptors are explicit",
               "archival_status": "caller must preserve, independently download, restore and verify outputs"}
    atomic(out / "summary.json", overall)
    atomic(out / "OUTPUT_MANIFEST.json", {"schema_version": 7, "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_MANIFEST.json"}})
    return overall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", "--reference-root", type=Path, required=True)
    p.add_argument("--decode", "--decode-root", type=Path, nargs="+", required=True)
    p.add_argument("--sensitivity-decode", "--sensitivity-decode-root", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--points", type=int, default=10000)
    p.add_argument("--surface-seeds", type=int, nargs=2, default=SURFACE_SEEDS)
    p.add_argument("--resume", action="store_true")
    result = run(**vars(p.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
