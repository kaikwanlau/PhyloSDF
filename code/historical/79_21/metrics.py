"""Fixed-checkpoint group-heldout geometry and handcrafted MorphFID evaluation.

This module deliberately uses the frozen v7 numerical definitions, but never
calls its all100-reference evaluator. Normalization is fitted on training79.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.dont_write_bytecode = True
try:
    from . import vendor_metrics as vm
except ImportError:
    import vendor_metrics as vm

SURFACE_SEEDS = (20261014, 20261015)
POINTS = 10000
ALLOWED_GROUPS = {"train_reconstruction", "heldout_reconstruction", "generation"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe_path(base, name):
    path = Path(name)
    if path.is_absolute():
        raise ValueError("Expected relative input path: " + str(name))
    base = Path(base).resolve()
    candidate = base / path
    if candidate.is_symlink():
        raise ValueError("Input symlinks are not accepted: " + str(name))
    path = candidate.resolve()
    if not path.is_relative_to(base):
        raise ValueError("Input path escapes its root: " + str(name))
    return path


def checkpoint_gate(study_dir):
    """Validate locked artifacts; this attests hashes, not historical chronology."""
    root = Path(study_dir).resolve()
    path = root / "FINAL_CHECKPOINTS_LOCK.json"
    lock = json.loads(path.read_text())
    if lock.get("status") != "fixed_before_heldout_evaluation":
        raise ValueError("Final checkpoints have not been fixed for evaluation")
    for key in ("decoder_sha256", "generator_sha256", "fitted_latent_sha256"):
        if not lock.get(key):
            raise ValueError("Checkpoint lock missing " + key)
    if set(str(k) for k in lock["generator_sha256"]) != {"42", "43", "44"}:
        raise ValueError("Lock must contain all three fixed generator seeds")
    protocol = json.loads((root / "protocol.json").read_text())
    required_fit_ids = {str(row["filename"]) + "__" + str(seed) for row in protocol["heldout"] for seed in (42, 43, 44)}
    if not required_fit_ids.issubset(lock["fitted_latent_sha256"]):
        raise ValueError("Lock is missing fixed-final heldout latent fits")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Checkpoint lock must provide its files for rehashing")
    observed = set()
    for row in files:
        name = row.get("path", row.get("relative_path"))
        if name in observed:
            raise ValueError("Duplicate locked checkpoint path")
        observed.add(name)
        path = _safe_path(root, name)
        if vm.sha(path) != row["sha256"]:
            raise ValueError("Locked checkpoint integrity mismatch: " + name)
    required = {lock["decoder_sha256"], *lock["generator_sha256"].values(), *lock["fitted_latent_sha256"].values()}
    if not required.issubset({r["sha256"] for r in files}):
        raise ValueError("Scientific checkpoint hashes are absent from lock file list")
    return lock, vm.sha(root / "FINAL_CHECKPOINTS_LOCK.json")


def fit_train_normalization(feature_rows, train_rows):
    features = np.asarray(feature_rows, dtype=np.float64)
    if features.shape != (len(train_rows), 12) or not np.isfinite(features).all():
        raise ValueError("Expected finite exact12 training-reference descriptors")
    weights = vm.species_weights(train_rows)
    mu, std, info = vm.normalization(features, weights)
    info.update(fitted_on="79 training references / 20 training species only",
                training_specimens=[r.get("filename", r.get("specimen")) for r in train_rows],
                training_species=sorted({r["species"] for r in train_rows}), weights=weights.tolist(),
                feature_names=list(vm.FEATURE_NAMES), feature_rows_sha256=hashlib.sha256(features.astype("<f8").tobytes()).hexdigest())
    return mu, std, info


def frechet_checked(left_mean, left_cov, right_mean, right_cov):
    """Equivalent PSD Bures trace via SVD, stable for tiny rank-deficient cohorts."""
    diff = np.asarray(left_mean) - np.asarray(right_mean)
    left_root, right_root = vm.psd_sqrt(left_cov), vm.psd_sqrt(right_cov)
    # ||sqrt(C1) sqrt(C2)||_* = tr sqrt(sqrt(C1) C2 sqrt(C1)).
    # SVD avoids square-root amplification of spurious positive null eigenvalues.
    cross_trace = float(np.linalg.svd(left_root @ right_root, compute_uv=False).sum())
    scale = float(diff @ diff + np.trace(left_cov) + np.trace(right_cov))
    result = scale - 2 * cross_trace
    if not np.isfinite(result) or result < -1e-8 * max(scale, 1):
        raise ValueError("Invalid Fréchet score")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        historical = vm.sqrtm(left_cov @ right_cov)
    imaginary = float(np.max(np.abs(historical.imag))) if np.iscomplexobj(historical) else 0.
    legacy = float(diff @ diff + np.trace(left_cov + right_cov - 2 * historical.real))
    if not np.isfinite(legacy) or not np.isfinite(imaginary):
        raise ValueError("Nonfinite historical covariance square root")
    tolerance = 2e-6 * max(1., abs(result), scale)
    return max(0., result), {"stable_psd_value_before_roundoff_clamp": result,
                            "stable_cross_trace_algorithm": "PSD square roots plus SVD nuclear norm; no ridge regularization",
                            "historical_sqrtm_real_part_value": legacy, "sqrtm_max_abs_imaginary": imaginary,
                            "stable_vs_historical_absolute_difference": abs(result - legacy),
                            "historical_agrees_at_recorded_tolerance": abs(result - legacy) <= tolerance,
                            "agreement_absolute_tolerance": tolerance,
                            "historical_sqrtm_warnings": [str(w.message) for w in recorded], "roundoff_clamp_only": result < 0}


def feature_score(reference, generated, norm, reference_weights, generated_weights=None):
    """Both populations can be species-balanced; moments retain small sample rank."""
    r, g = np.asarray(reference), np.asarray(generated)
    if len(r) < 2 or len(g) < 2:
        return {"status": "undefined_less_than_two_finite_rows", "reference_rows": len(r), "generated_rows": len(g)}
    mu, std = norm
    scores = {}
    for name, subset in vm.FEATURE_SUBSETS.items():
        indices = list(subset)
        rm, rc, rd = vm.distribution(((r - mu) / std)[:, indices], reference_weights)
        gm, gc, gd = vm.distribution(((g - mu) / std)[:, indices], generated_weights)
        value, check = frechet_checked(rm, rc, gm, gc)
        scores[name] = {"value": value, "features": [vm.FEATURE_NAMES[i] for i in indices],
                        "reference_mean": rm.tolist(), "generated_mean": gm.tolist(),
                        "reference_covariance_matrix": rc.tolist(), "generated_covariance_matrix": gc.tolist(),
                        "reference_covariance": rd, "generated_covariance": gd, "square_root_check": check,
                        "inference": "descriptive handcrafted feature score; finite-sample bias and rank deficiency retained"}
    return scores


def _reference(row, root):
    raw = _safe_path(root, row["raw_stl"])
    processed = _safe_path(root, row["processed_npz"])
    if vm.sha(raw) != row["raw_sha256"] or vm.sha(processed) != row["processed_sha256"]:
        raise ValueError("Reference input checksum mismatch: " + row["specimen"])
    v, f = vm.read_binary_stl(raw)
    with np.load(processed, allow_pickle=False) as data:
        center, scale = np.array(data["centroid"]), float(data["scale"])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid saved reference transform")
        if np.max(np.abs(v.mean(0) - center)) > 1e-10 or abs(1 / np.abs(v - v.mean(0)).max() - scale) > 1e-12:
            raise ValueError("Saved reference normalization differs from raw unique vertices")
        if len(v) != int(data["num_vertices"]) or len(f) != int(data["num_faces"]):
            raise ValueError("Reference topology differs from saved preprocessing")
    v = (v - center) * scale
    features, flags = vm.handcrafted_features(v, f)
    return {**row, "vertices": v, "faces": f, "features": features, "feature_flags": flags,
            "topology": vm.topology(v, f), "surface_key": _digest({"raw_sha256": row["raw_sha256"], "processed_sha256": row["processed_sha256"]}),
            "transform": {"centroid": center.tolist(), "scale_multiplier": scale}}


def _load_attempts(decode_dir, lock_sha, protocol):
    root = Path(decode_dir).resolve()
    manifest_path = root / "decode_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("checkpoint_lock_sha256") != lock_sha:
        raise ValueError("Decode manifest used a different final-checkpoint lock")
    if vm.sha(root/'codes.npz') != manifest.get('codes_sha256'):raise ValueError('Decode code archive checksum differs')
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or len({t["task"] for t in tasks}) != len(tasks):
        raise ValueError("Incomplete or duplicate committed decode tasks")
    expected = {str(t["task"]) for t in tasks}
    result_files = list(root.glob("*/result.json"))
    if {p.parent.name for p in result_files} != expected:
        raise ValueError("Decode results do not cover every committed attempt")
    train = {r["filename"]: r["species"] for r in protocol["train"]}
    heldout = {r["filename"]: r["species"] for r in protocol["heldout"]}
    target_species = set(heldout.values())
    attempts = []
    for task in tasks:
        task_id = task["task"]
        directory = _safe_path(root, task_id)
        rec = json.loads((directory / "result.json").read_text())
        if rec.get("task") != task_id:
            raise ValueError("Result task ID mismatch")
        for key in ("group", "species", "seed", "method", "specimen"):
            if key in task and rec.get(key) != task[key]:
                raise ValueError("Result disagrees with committed task: " + key)
        group = rec.get("group")
        if group not in ALLOWED_GROUPS:
            raise ValueError("Unknown decode group")
        if group == "generation":
            if rec.get("species") not in target_species:
                raise ValueError("Generation target is outside four heldout species")
        else:
            allowed = train if group == "train_reconstruction" else heldout
            if allowed.get(rec.get("specimen")) != rec.get("species"):
                raise ValueError("Reconstruction specimen is in wrong split")
        if rec.get("status") not in {"mesh_complete", "failure"}:
            raise ValueError("Unfinished or unknown decode attempt status")
        rec = {**rec, "result_sha256": vm.sha(directory / "result.json"), "evaluation_status": rec["status"]}
        if rec["status"] == "mesh_complete":
            path = _safe_path(directory, rec.get("mesh_npz", "mesh.npz"))
            if vm.sha(path) != rec["mesh_sha256"]:
                raise ValueError("Decoded mesh checksum mismatch: " + task_id)
            try:
                with np.load(path, allow_pickle=False) as data:
                    v, f = vm.validate_mesh(data["vertices"], data["faces"])
                info = vm.topology(v, f)
                rec.update(vertices=v, faces=f, topology=info, surface_key=rec["mesh_sha256"])
            except Exception as error:
                rec.update(evaluation_status="geometry_failure", geometry_failure={"type": type(error).__name__, "message": str(error)})
            if rec["evaluation_status"] == "mesh_complete":
                try:
                    rec["features"], rec["feature_flags"] = vm.handcrafted_features(v, f)
                except Exception as error:
                    rec["feature_failure"] = {"type": type(error).__name__, "message": str(error)}
        attempts.append(rec)
    return attempts, {"decode_manifest_sha256": vm.sha(manifest_path), "result_sha256": {r["task"]: r["result_sha256"] for r in attempts}}


def _compact(record):
    return {k: v for k, v in record.items() if k not in {"vertices", "faces", "features", "surface_key"}}


def _summary_attempts(rows):
    valid = [r for r in rows if r["evaluation_status"] == "mesh_complete"]
    return {"attempts": len(rows), "valid_meshes": len(valid), "valid_feature_rows": sum("features" in r for r in valid),
            "failure_rate": 1 - len(valid) / len(rows),
            "failed_tasks": [r["task"] for r in rows if r["evaluation_status"] != "mesh_complete"],
            "feature_failed_tasks": [r["task"] for r in valid if "features" not in r],
            "conditional_scores": "valid extracted meshes/features only; every attempt remains in failure denominators"}


def weighted_geometry(distance, reference_weights, generated_weights=None):
    matrix = np.asarray(distance, dtype=float)
    rw = np.asarray(reference_weights, dtype=float)
    gw = np.full(len(matrix), 1 / len(matrix)) if generated_weights is None else np.asarray(generated_weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(gw), len(rw)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid generated-by-reference geometry matrix")
    rw, gw = rw / rw.sum(), gw / gw.sum()
    out = vm.metric_from_matrix(matrix, rw)
    out["generated_to_reference_nn_CD_mean"] = float(gw @ matrix.min(1))
    out["generated_weights"] = gw.tolist()
    out["reference_weights"] = rw.tolist()
    return out


def _evaluate(study_dir, reference_root, decode_dir, out_dir, points=POINTS, surface_seeds=SURFACE_SEEDS):
    """Internal small-fixture entry point; public evaluate enforces fixed10000."""
    study, root, out = Path(study_dir).resolve(), Path(reference_root).resolve(), Path(out_dir).resolve()
    if points < 2 or tuple(surface_seeds) != SURFACE_SEEDS:
        raise ValueError("Requires two fixed integration seeds and at least two surface points")
    if out.exists() or any(out.is_relative_to(p) for p in (root, Path(decode_dir).resolve())):
        raise ValueError("Metrics output must be a new directory outside inputs")
    protocol_path = study / "protocol.json"
    protocol_sha = vm.sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    train_rows, held_rows = protocol["train"], protocol["heldout"]
    train_names, held_names = [r["filename"] for r in train_rows], [r["filename"] for r in held_rows]
    if len(train_rows) != 79 or len(held_rows) != 21 or len(set(train_names + held_names)) != 100:
        raise ValueError("Fixed79/21 split changed")
    if len({r["species"] for r in train_rows}) != 20 or len({r["species"] for r in held_rows}) != 4:
        raise ValueError("Fixed20/4 species split changed")
    if {r["species"] for r in held_rows} != {"G.Fortis", "G.Fuliginosa", "G.Magnirostris", "C.Parvulus"}:
        raise ValueError("The four predeclared focal heldout species changed")
    if {r["species"] for r in train_rows} & {r["species"] for r in held_rows}:
        raise ValueError("Species overlap across training and holdout")
    manifest_path = root / "reference_manifest.json"
    if vm.sha(manifest_path) != protocol["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    source = json.loads(manifest_path.read_text())["references"]
    if len(source) != 100 or len({r["specimen"] for r in source}) != 100:
        raise ValueError("Reference manifest is incomplete or duplicated")
    reference_rows = {r["specimen"]: r for r in source}
    for row in train_rows + held_rows:
        if reference_rows[row["filename"]]["species"] != row["species"]:
            raise ValueError("Reference/protocol species mismatch")
    # Only training raw/processed files are opened before checkpoint-gate proof.
    training = [_reference(reference_rows[name], root) for name in train_names]
    train_features = np.stack([r["features"] for r in training])
    mu, std, norm_info = fit_train_normalization(train_features, train_rows)
    lock, lock_sha = checkpoint_gate(study)
    heldout = [_reference(reference_rows[name], root) for name in held_names]
    attempts, decode_provenance = _load_attempts(decode_dir, lock_sha, protocol)
    out.mkdir(parents=True)
    vm.atomic(out / "normalization_train_only.json", norm_info)
    norm_file_sha = vm.sha(out / "normalization_train_only.json")
    np.savez_compressed(out / "reference_features.npz", train_features=train_features,
                        heldout_features=np.stack([r["features"] for r in heldout]), mean=mu, std=std,
                        train_specimen=np.asarray(train_names), heldout_specimen=np.asarray(held_names), feature_names=np.asarray(vm.FEATURE_NAMES))
    vm.atomic(out / "reference_flags.json", {"training": [_compact(r) for r in training], "heldout": [_compact(r) for r in heldout]})
    vm.atomic(out / "attempts.json", [_compact(r) for r in attempts])
    finite_feature_rows = [r for r in attempts if "features" in r]
    if finite_feature_rows:
        np.savez_compressed(out / "attempt_features.npz", features=np.stack([r["features"] for r in finite_feature_rows]),
                            task=np.asarray([r["task"] for r in finite_feature_rows]),
                            group=np.asarray([r["group"] for r in finite_feature_rows]),
                            species=np.asarray([r["species"] for r in finite_feature_rows]), feature_names=np.asarray(vm.FEATURE_NAMES))
    sample_root = out / "surface_samples"
    sample_root.mkdir()
    samples = {}

    def surface(row, seed):
        key = _digest({"mesh": row["surface_key"], "points": points, "seed": seed, "sampler": "frozen_area_barycentric_v7"})
        if key not in samples:
            pts = vm.sample_surface(row["vertices"], row["faces"], points, seed)
            from scipy.spatial import cKDTree
            samples[key] = (pts, cKDTree(pts))
            np.savez_compressed(sample_root / (key + ".npz"), points=pts, seed=np.asarray(seed), count=np.asarray(points))
        return samples[key]

    def distance(a, b, seed):
        ap, at = surface(a, seed)
        bp, bt = surface(b, seed)
        return vm.cd(ap, bp, at, bt)

    refs = {r["specimen"]: r for r in training + heldout}
    reconstruction = []
    for row in attempts:
        if row["group"] == "generation":
            continue
        rec = _compact(row)
        if row["evaluation_status"] == "mesh_complete":
            values = [distance(row, refs[row["specimen"]], seed) for seed in surface_seeds]
            rec.update(matched_specimen_CD_by_surface_seed=[{"seed": seed, "CD": value} for seed, value in zip(surface_seeds, values)],
                       mean_matched_specimen_CD=float(np.mean(values)))
        reconstruction.append(rec)
    vm.atomic(out / "reconstruction_geometry.json", reconstruction)
    reconstruction_groups = []
    rgroups = collections.defaultdict(list)
    for row in reconstruction:
        rgroups[(row["group"], row["method"], row.get("seed"))].append(row)
    for (group, method, seed), rows in sorted(rgroups.items(), key=lambda p: str(p[0])):
        expected_species = sorted({r["species"] for r in (train_rows if group == "train_reconstruction" else held_rows)})
        feature_rows = [r for r in attempts if r["group"] == group and r["method"] == method and r.get("seed") == seed and "features" in r]
        reference_group = training if group == "train_reconstruction" else heldout
        by_species = {}
        for sp in expected_species:
            subset = [r for r in rows if r["species"] == sp]
            if not subset:
                by_species[sp] = {"status": "no_committed_attempts"}
                continue
            vals = [r["mean_matched_specimen_CD"] for r in subset if "mean_matched_specimen_CD" in r]
            by_species[sp] = {**_summary_attempts(subset), "matched_specimen_CD_mean": float(np.mean(vals)) if vals else None}
            gf = [r["features"] for r in feature_rows if r["species"] == sp]
            rf = [r["features"] for r in reference_group if r["species"] == sp]
            by_species[sp]["MorphFID"] = feature_score(np.asarray(rf).reshape(-1, 12), np.asarray(gf).reshape(-1, 12), (mu, std), np.full(len(rf), 1 / len(rf)))
        means = [by_species[sp].get("matched_specimen_CD_mean") for sp in expected_species]
        pooled_morphfid = feature_score(np.stack([r["features"] for r in reference_group]), np.stack([r["features"] for r in feature_rows]),
                                       (mu, std), vm.species_weights(reference_group), vm.species_weights(feature_rows)) if feature_rows and {r["species"] for r in feature_rows} == set(expected_species) else {"status": "undefined_species_without_finite_feature"}
        reconstruction_groups.append({"group": group, "method": method, "seed": seed, **_summary_attempts(rows), "by_species": by_species,
                                      "equal_species_matched_CD_mean": float(np.mean(means)) if all(v is not None for v in means) else None,
                                      "MorphFID_equal_species_pooled": pooled_morphfid,
                                      "scope": "training reconstruction context" if group == "train_reconstruction" else "heldout reconstruction after observing fit SDF queries"})
    vm.atomic(out / "reconstruction_summary.json", reconstruction_groups)

    gg = collections.defaultdict(list)
    for row in attempts:
        if row["group"] == "generation":
            gg[(row["method"], row.get("seed"))].append(row)
    summaries = []
    target_species = sorted({r["species"] for r in held_rows})
    for (method, training_seed), rows in sorted(gg.items(), key=lambda p: str(p[0])):
        for cohort in target_species + ["all4_equal_species"]:
            selected = rows if cohort == "all4_equal_species" else [r for r in rows if r["species"] == cohort]
            reference = heldout if cohort == "all4_equal_species" else [r for r in heldout if r["species"] == cohort]
            if not selected:
                summaries.append({"method": method, "seed": training_seed, "cohort": cohort, "status": "no_committed_attempts"})
                continue
            group_id = _digest([method, training_seed, cohort])[:20]
            directory = out / "groups" / group_id
            directory.mkdir(parents=True)
            valid = [r for r in selected if r["evaluation_status"] == "mesh_complete"]
            feature_valid = [r for r in valid if "features" in r]
            summary = {"method": method, "seed": training_seed, "cohort": cohort, **_summary_attempts(selected),
                       "reference_rows": len(reference), "reference_specimens": [r["specimen"] for r in reference],
                       "task_ids": [r["task"] for r in selected]}
            balanced = cohort == "all4_equal_species"
            valid_species = {r["species"] for r in valid}
            feature_species = {r["species"] for r in feature_valid}
            geometry_complete = not balanced or valid_species == set(target_species)
            features_complete = not balanced or feature_species == set(target_species)
            reference_weights = vm.species_weights(reference) if balanced else np.full(len(reference), 1 / len(reference))
            if valid and geometry_complete:
                generated_weights = vm.species_weights(valid) if balanced else np.full(len(valid), 1 / len(valid))
                scores = []
                for seed in surface_seeds:
                    matrix = np.array([[distance(g, r, seed) for r in reference] for g in valid], dtype=np.float64)
                    diversity = np.zeros((len(valid), len(valid)), dtype=np.float64)
                    for i in range(len(valid)):
                        for j in range(i):
                            diversity[i, j] = diversity[j, i] = distance(valid[i], valid[j], seed)
                    np.savez_compressed(directory / ("geometry_seed" + str(seed) + ".npz"), generated_reference_CD=matrix,
                                        generated_pair_CD=diversity, generated_task=np.asarray([r["task"] for r in valid]),
                                        reference_specimen=np.asarray([r["specimen"] for r in reference]),
                                        generated_weights=generated_weights, reference_weights=reference_weights)
                    score = weighted_geometry(matrix, reference_weights, generated_weights)
                    denom = 1 - float(generated_weights @ generated_weights)
                    score.update(surface_seed=seed, diversity_valid_pair_weighted_CD=float(generated_weights @ diversity @ generated_weights / denom) if denom > 0 else None)
                    scores.append(score)
                summary["geometry_by_surface_seed"] = scores
            else:
                summary["geometry_status"] = "undefined_no_valid_meshes" if not valid else "undefined_species_without_valid_mesh"
            if feature_valid and features_complete:
                gf = np.stack([r["features"] for r in feature_valid])
                generated_weights = vm.species_weights(feature_valid) if balanced else None
                rf = np.stack([r["features"] for r in reference])
                summary["MorphFID"] = feature_score(rf, gf, (mu, std), reference_weights, generated_weights)
                np.savez_compressed(directory / "generated_features.npz", features=gf, task=np.asarray([r["task"] for r in feature_valid]),
                                    species=np.asarray([r["species"] for r in feature_valid]), feature_names=np.asarray(vm.FEATURE_NAMES))
            else:
                summary["MorphFID"] = {"status": "undefined_no_finite_features" if not feature_valid else "undefined_species_without_finite_feature"}
            summary["failure_rate_equal_species"] = float(np.mean([1 - sum(r["evaluation_status"] == "mesh_complete" for r in selected if r["species"] == sp) / sum(r["species"] == sp for r in selected) for sp in target_species])) if balanced and all(any(r["species"] == sp for r in selected) for sp in target_species) else summary["failure_rate"] if not balanced else None
            summary["normalization_train_only_sha256"] = norm_file_sha
            vm.atomic(directory / "summary.json", summary)
            summaries.append(summary)
            print("heldout metrics", method, training_seed, cohort, len(valid), "/", len(selected), flush=True)
    vm.atomic(out / "generation_summary.json", summaries)
    # Detect changed checkpoint/reference/decode bytes before declaring completion.
    if checkpoint_gate(study)[1] != lock_sha or vm.sha(protocol_path) != protocol_sha or vm.sha(manifest_path) != protocol["reference_manifest_sha256"]:
        raise ValueError("Scientific checkpoint lock changed during evaluation")
    for ref in training + heldout:
        if vm.sha(_safe_path(root, ref["raw_stl"])) != ref["raw_sha256"] or vm.sha(_safe_path(root, ref["processed_npz"])) != ref["processed_sha256"]:
            raise ValueError("Reference changed during evaluation")
    if _load_attempts(decode_dir, lock_sha, protocol)[1] != decode_provenance:
        raise ValueError("Decode inputs changed during evaluation")
    if vm.sha(out / "normalization_train_only.json") != norm_file_sha:
        raise ValueError("Train-only normalization changed during evaluation")
    summary = {"schema": "group_holdout_metrics_v8", "status": "completed", "protocol_sha256": protocol_sha,
               "checkpoint_lock_sha256": lock_sha, "checkpoint_lock_chronology_caveat": "Hash lock records caller commitment; this evaluator cannot independently attest historical chronology.",
               "metric_source_sha256": vm.sha(__file__), "vendor_metric_source_sha256": vm.sha(vm.__file__), "decode_provenance": decode_provenance,
               "points": points, "surface_seeds": list(surface_seeds), "feature_names": list(vm.FEATURE_NAMES),
               "normalization_train_only_sha256": norm_file_sha, "generation_groups": len(summaries), "reconstruction_attempts": len(reconstruction),
               "scope": "fixed four-species retrospective group holdout, not full LOSO or prospective specimen collection",
               "uncertainty": "Surface seeds are integration sensitivity; fitting/generator seeds are optimization replicates. Specimens and species remain biological units.",
               "limitations": ["Exact12 handcrafted MorphFID, no learned feature model; signed volume/hull fallbacks retained.",
                               "Per-heldout-species reference counts5-6 imply rank-deficient12-dimensional covariances and finite-sample bias.",
                               "Mesh/feature scores are conditional on successful extraction; exact attempted denominators remain separate.",
                               "All4 pooled scores use equal species weights on both successful generated rows and references, and are undefined if a species has no valid rows.",
                               "No per-generated-mesh recentering, rescaling, repair or component filtering."]}
    vm.atomic(out / "summary.json", summary)
    vm.atomic(out / "OUTPUT_MANIFEST.json", {"schema": "group_holdout_metrics_v8", "files": {str(p.relative_to(out)): vm.sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_MANIFEST.json"}})
    return summary


def evaluate(study_dir, reference_root, decode_dir, out_dir):
    return _evaluate(study_dir, reference_root, decode_dir, out_dir, POINTS, SURFACE_SEEDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluate(args.study, args.reference_root, args.decode, args.out)


if __name__ == "__main__":
    main()
