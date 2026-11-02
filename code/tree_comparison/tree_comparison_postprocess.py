"""CPU-only additions to the completed, frozen PhyloSDF tree sampler study.

Does not train, decode, or change runner/helper/frozen metric files. Computes
reference-reference Chamfer distances once with the same surfaces and adds
leave-one-out 1-NNA plus descriptive across-seed summaries. Call only after
tree_sampler_comparison.run() returns successfully.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

SCHEMA = "phylosdf_tree_cpu_postprocess_20261024_v1"
MIXTURE = "unconditional_species_mixture"
DEFAULT_BASE = Path("/content/PHYLOSDF_TREE_COMPARISON_20261024")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def one_nna(reference_reference, generated_reference, generated_condensed):
    """Return classifier diagnostics; rows of GR and condensed GG have same order.

    The union order is references first then generated samples. The diagonal is
    excluded. np.argmin breaks exact ties by that locked union order. Raw
    accuracy is the paper's 1-NNA; balanced accuracy is also retained to expose
    class-size effects when some generated meshes failed.
    """
    rr = np.asarray(reference_reference, dtype=np.float64)
    gr = np.asarray(generated_reference, dtype=np.float64)
    condensed = np.asarray(generated_condensed, dtype=np.float64)
    if rr.ndim != 2 or rr.shape[0] != rr.shape[1]:
        raise ValueError("Reference distance matrix must be square")
    if gr.ndim != 2 or gr.shape[1] != len(rr):
        raise ValueError("Generated/reference distance dimensions differ")
    nr, ng = len(rr), len(gr)
    if nr < 1 or ng < 1:
        raise ValueError("Both classes need at least one valid mesh")
    if condensed.shape != (ng * (ng - 1) // 2,):
        raise ValueError("Condensed generated distance order/size mismatch")
    if not all(np.isfinite(x).all() and np.all(x >= 0) for x in (rr, gr, condensed)):
        raise ValueError("Nonfinite or negative distances")
    if not np.allclose(rr, rr.T, rtol=0, atol=1e-14) or not np.allclose(np.diag(rr), 0):
        raise ValueError("Reference matrix symmetry/diagonal mismatch")
    gg = np.zeros((ng, ng), dtype=np.float64)
    ix = np.triu_indices(ng, 1)
    gg[ix] = condensed
    gg[(ix[1], ix[0])] = condensed
    union = np.block([[rr, gr.T], [gr, gg]])
    np.fill_diagonal(union, np.inf)
    nearest = np.argmin(union, axis=1)
    labels = np.r_[np.zeros(nr, dtype=np.int8), np.ones(ng, dtype=np.int8)]
    correct = labels[nearest] == labels
    real_acc = float(correct[:nr].mean())
    gen_acc = float(correct[nr:].mean())
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": (real_acc + gen_acc) / 2,
        "reference_class_accuracy": real_acc,
        "generated_class_accuracy": gen_acc,
        "reference_count": nr,
        "generated_valid_count": ng,
        "correct_count": int(correct.sum()),
        "nearest_union_indices": nearest.tolist(),
        "reference_approximate_chance": 0.5,
        "finite_sample_random_label_accuracy_expectation":
            (nr * (nr - 1) + ng * (ng - 1)) / ((nr + ng) * (nr + ng - 1)),
    }


def synthetic_checks():
    """Check self-exclusion and condensed-vector order using exact 1-D distances."""
    def distances(left, right):
        return (np.asarray(left)[:, None] - np.asarray(right)[None, :]) ** 2

    rr = distances([0, 0.1], [0, 0.1])
    gr = distances([10, 10.1], [0, 0.1])
    separated = one_nna(rr, gr, np.array([0.01]))
    if separated["accuracy"] != 1:
        raise AssertionError("Separated-class 1-NNA check failed")
    rr = distances([0, 10], [0, 10])
    gr = distances([1, 11], [0, 10])
    paired = one_nna(rr, gr, np.array([100.0]))
    if paired["accuracy"] != 0:
        raise AssertionError("Opposite-class nearest neighbours/self-exclusion check failed")
    # Three generated samples exercise condensed entries (0,1),(0,2),(1,2).
    rr = distances([0, 20], [0, 20])
    gen = np.array([1.0, 2.0, 19.0])
    gg = distances(gen, gen)
    gr = distances(gen, [0, 20])
    result = one_nna(rr, gr, gg[np.triu_indices(3, 1)])
    expected_nearest = [2, 4, 0, 2, 1]
    if result["nearest_union_indices"] != expected_nearest:
        raise AssertionError("Condensed order/tie ordering check failed")
    return {"status": "passed", "checks": ["class separation", "self exclusion",
            "opposite-class closest pairs", "condensed order", "locked tie order"]}


def import_engine(path):
    spec = importlib.util.spec_from_file_location("tree_cpu_postprocess_engine", str(path))
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return engine


def verify_study(engine, base):
    """Read the existing lock and all terminal markers without changing them."""
    lock_path = base / "PROTOCOL_LOCK.json"
    lock = json.loads(lock_path.read_text())
    core = {k: v for k, v in lock.items() if k != "protocol_and_source_sha256"}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    if digest != lock["protocol_and_source_sha256"]:
        raise RuntimeError("Study protocol lock checksum mismatch")
    for name, expected in lock["sources"].items():
        path = Path(engine.__file__).with_name(name) if name in (
            "tree_sampler_comparison.py", "rep_helper.py") else base / name
        if file_sha(path) != expected:
            raise RuntimeError("Study source identity mismatch: " + name)
    complete = json.loads((base / "results/ALL_COMPLETE.json").read_text())
    expected_arms = {(seed, method) for seed in engine.SEEDS for method in engine.METHODS}
    actual_arms = {(row["seed"], row["method"]) for row in complete["statuses"]}
    if complete.get("status") != "completed" or actual_arms != expected_arms or complete.get("arms") != 12:
        raise RuntimeError("Main comparison is not complete")
    for seed, method in sorted(expected_arms):
        folder = base / f"results/seed{seed}_{method}"
        marker = json.loads((folder / "ARM_COMPLETE.json").read_text())
        if marker["status"] != "completed" or marker["attempts"] != 140:
            raise RuntimeError("Incomplete comparison arm")
        if file_sha(folder / "summary.json") != marker["summary_sha256"]:
            raise RuntimeError("Terminal summary checksum mismatch")
    return lock


def reference_matrix(rep, vm, references, out, identity):
    """Compute once and resume completed upper-triangle rows for both surface seeds."""
    path = out / "REFERENCE_PAIRWISE_CD.npz"
    meta_path = out / "REFERENCE_PAIRWISE_CD.json"
    names = [row["specimen"] for row in references]
    n = len(names)
    seeds = list(rep.SURFACE_SEEDS)
    if path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity or meta["sha256"] != file_sha(path):
            raise RuntimeError("Reference-pair cache identity/checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            matrix = np.array(saved["reference_reference_CD"])
            if saved["references"].astype(str).tolist() != names or saved["surface_seeds"].tolist() != seeds:
                raise RuntimeError("Reference-pair cache ordering mismatch")
        done = meta["rows_completed"]
    else:
        matrix = np.full((2, n, n), np.nan, dtype=np.float64)
        for seed_i in range(2):
            np.fill_diagonal(matrix[seed_i], 0)
        done = [0, 0]

    def save():
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, reference_reference_CD=matrix,
                                references=np.asarray(names), surface_seeds=np.asarray(seeds))
        temp.replace(path)
        atomic_json(meta_path, {"identity": identity, "sha256": file_sha(path),
                    "rows_completed": done, "status": "completed" if done == [n, n] else "running"})

    if matrix.shape != (2, n, n) or len(done) != 2 or any(x < 0 or x > n for x in done):
        raise RuntimeError("Malformed reference pair cache")
    from scipy.spatial import cKDTree
    for seed_i, surface_seed in enumerate(seeds):
        if done[seed_i] == n:
            continue
        clouds = [vm.sample_surface(row["vertices"], row["faces"], 10000, surface_seed)
                  for row in references]
        trees = [cKDTree(cloud) for cloud in clouds]
        began = time.time()
        for i in range(done[seed_i], n):
            for j in range(i + 1, n):
                value = rep.kd_cd(clouds[i], clouds[j], trees[i], trees[j])
                matrix[seed_i, i, j] = matrix[seed_i, j, i] = value
            done[seed_i] = i + 1
            if (i + 1) % 10 == 0 or i + 1 == n:
                save()
                print(f"Reference pairs surface seed {surface_seed}: rows {i + 1}/{n}; "
                      f"elapsed {time.time() - began:.0f}s", flush=True)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Reference distance cache incomplete")
    for item in matrix:
        if not np.array_equal(item, item.T):
            raise RuntimeError("Reference matrix lost symmetry")
    return matrix, {"path": str(path.relative_to(out.parent.parent)), "sha256": file_sha(path),
                    "references": names, "surface_seeds": seeds}


def group_score(base, seed, method, expected_references, reference_distances, surface_seeds,
                expected_metric_source_sha=None, expected_reference_manifest_sha=None):
    folder = base / f"results/seed{seed}_{method}"
    config = json.loads((folder / "evaluation_config.json").read_text())
    if config["points_per_surface"] != 10000 or config["surface_seeds"] != list(surface_seeds):
        raise RuntimeError("Surface metric protocol mismatch")
    if expected_metric_source_sha is not None and config["source_sha256"] != expected_metric_source_sha:
        raise RuntimeError("Frozen evaluator source identity mismatch")
    if expected_reference_manifest_sha is not None and config["reference_manifest_sha256"] != expected_reference_manifest_sha:
        raise RuntimeError("Frozen evaluator reference identity mismatch")
    groups = [g for g in json.loads((folder / "summary.json").read_text())["groups"]
              if g["cohort"] == MIXTURE and g["method"] == method]
    if len(groups) != 1:
        raise RuntimeError("Expected one species-mixture group per arm")
    group = groups[0]
    audits = json.loads((folder / "attempt_audit.json").read_text())
    attempts = [r for r in audits if r["method"] == method and r["cohort"] == MIXTURE]
    valid = sorted([r for r in attempts if r["status"] == "completed"],
                   key=lambda row: (int(row["sample_slot"]), row["task"]))
    if group["attempts"] != 100 or len(attempts) != 100 or len(valid) != group["valid_meshes"]:
        raise RuntimeError("Attempt denominator differs from the committed study")
    if not valid:
        return {"seed": seed, "method": method, "cohort": MIXTURE, "attempts": 100,
                "valid_meshes": 0, "failed_meshes": 100, "status": "undefined_no_valid_meshes",
                "one_NNA_accuracy": None, "one_NNA_balanced_accuracy": None}
    group_dir = folder / "groups" / group["group_id"]
    path = group_dir / "distance_matrices.npz"
    terminal = json.loads((group_dir / "COMPLETED.json").read_text())
    if file_sha(path) != terminal["files"]["distance_matrices.npz"]:
        raise RuntimeError("Generated/reference distance matrix checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        if data["references"].astype(str).tolist() != expected_references:
            raise RuntimeError("Reference-column order mismatch")
        if data["valid_tasks"].astype(str).tolist() != [r["task"] for r in valid]:
            raise RuntimeError("Generated-row order mismatch")
        if data["sample_slots"].tolist() != [int(r["sample_slot"]) for r in valid]:
            raise RuntimeError("Generated-slot order mismatch")
        if data["surface_seeds"].tolist() != list(surface_seeds):
            raise RuntimeError("Generated surface-seed order mismatch")
        gr, gg = np.array(data["generated_to_reference_CD"]), np.array(data["pairwise_generated_CD"])
    scores = [dict(surface_seed=surface_seed, **one_nna(reference_distances[i], gr[i], gg[i]))
              for i, surface_seed in enumerate(surface_seeds)]
    return {"seed": seed, "method": method, "cohort": MIXTURE, "attempts": 100,
            "valid_meshes": len(valid), "failed_meshes": 100 - len(valid), "status": "completed",
            "one_NNA_accuracy": float(np.mean([r["accuracy"] for r in scores])),
            "one_NNA_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in scores])),
            "by_surface_seed": scores,
            "distance_matrix_source": str(path.relative_to(base)), "distance_matrix_sha256": file_sha(path),
            "failure_policy": "All100 attempts retained in counts; classifier uses valid geometry only."}


def descriptive_summaries(rows, seeds):
    metrics = ("nn_CD", "MMD_CD", "COV", "diversity_CD", "MorphFID8",
               "within_species_diversity_percent", "one_NNA_accuracy", "one_NNA_balanced_accuracy")
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["cohort"]), []).append(row)
    result = []
    for (method, cohort), items in sorted(groups.items()):
        items.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in items] != list(seeds):
            raise RuntimeError("Missing or duplicated seed in descriptive summary")
        entry = {"method": method, "cohort": cohort, "seeds": list(seeds),
                 "attempts_by_seed": [row["attempts"] for row in items],
                 "valid_meshes_by_seed": [row["valid_meshes"] for row in items],
                 "failed_meshes_by_seed": [row["attempts"] - row["valid_meshes"] for row in items],
                 "metrics": {}}
        for metric in metrics:
            values = [float(row[metric]) for row in items if row.get(metric) is not None]
            if values and not np.isfinite(values).all():
                raise RuntimeError("Nonfinite metric in summary")
            if not values:
                continue
            entry["metrics"][metric] = {"values": values, "n": len(values),
                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        result.append(entry)
    return result


def run(svc=None, base=DEFAULT_BASE, folder_id=None, engine=None):
    """Run CPU postprocessing after main completion; publish if a Drive svc is passed.

    Typical call: post.run(svc, base=R, folder_id=OUTPUT_FOLDER_ID, engine=e).
    """
    base = Path(base)
    if engine is None:
        engine = import_engine(Path(__file__).with_name("tree_sampler_comparison.py"))
    lock = verify_study(engine, base)
    rep = engine.configure(base, svc)
    vm = rep.vendor()
    refs, _ = vm.load_references(rep.REFS / "reference_manifest.json")
    if len(refs) != 100 or len({r["specimen"] for r in refs}) != 100:
        raise RuntimeError("Expected100 distinct reference records")
    out = base / "results/postprocess"
    out.mkdir(parents=True, exist_ok=True)
    checks = synthetic_checks()
    identity = {"schema": SCHEMA, "main_protocol_sha256": lock["protocol_and_source_sha256"],
        "reference_manifest_sha256": file_sha(rep.REFS / "reference_manifest.json"),
        "frozen_metric_source_sha256": file_sha(rep.MESHSRC / "vendor/vendor_metrics.py"),
        "postprocess_source_sha256": file_sha(__file__), "points": 10000,
        "surface_seeds": list(rep.SURFACE_SEEDS), "reference_names": [r["specimen"] for r in refs]}
    protocol = {"identity": identity, "synthetic_checks": checks,
        "one_NNA": "Leave-one-out nearest-neighbour class accuracy on union(real,generated); no threshold; diagonal excluded.",
        "tie_rule": "First locked union index, references first, then generated rows in frozen sample-slot order.",
        "aggregation": "Classifier computed separately for each surface seed, then averaged; mean and sampleSD across generator seeds42–44.",
        "scope": "Descriptive uncertainty conditional on one tree-trained representation and training-reference collection; no confidence intervals or anatomical validity implied.",
        "failure_policy": "Preserve all attempted/failed counts; geometric classifier conditional on valid extracted meshes."}
    existing = out / "POSTPROCESS_PROTOCOL.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise RuntimeError("Postprocess identity changed in existing result directory")
    atomic_json(existing, protocol)
    reference_distances, cache = reference_matrix(rep, vm, refs, out, identity)
    scores = [group_score(base, seed, method, identity["reference_names"], reference_distances,
                          rep.SURFACE_SEEDS, identity["frozen_metric_source_sha256"],
                          identity["reference_manifest_sha256"])
              for seed in engine.SEEDS for method in engine.METHODS]
    atomic_json(out / "ONE_NNA_BY_ARM.json", {"identity": identity, "reference_cache": cache, "rows": scores})
    source_rows = json.loads((base / "results/COMPARISON_ROWS.json").read_text())
    expected_row_keys = {(seed, method, cohort) for seed in engine.SEEDS for method in engine.METHODS
                         for cohort in [MIXTURE, *("conditional_" + name for name in rep.FOCAL)]}
    observed_row_keys = [(row["seed"], row["method"], row["cohort"]) for row in source_rows]
    if len(observed_row_keys) != len(expected_row_keys) or set(observed_row_keys) != expected_row_keys:
        raise RuntimeError("Main comparison rows do not cover each committed method/seed/cohort exactly once")
    score_map = {(row["seed"], row["method"]): row for row in scores}
    merged = []
    for original in source_rows:
        row = dict(original)
        if row["cohort"] == MIXTURE:
            score = score_map[(row["seed"], row["method"])]
            if row["attempts"] != score["attempts"] or row["valid_meshes"] != score["valid_meshes"]:
                raise RuntimeError("Merged metric denominator mismatch")
            row.update(one_NNA_accuracy=score["one_NNA_accuracy"], one_NNA_balanced_accuracy=score["one_NNA_balanced_accuracy"])
        merged.append(row)
    aggregates = descriptive_summaries(merged, engine.SEEDS)
    atomic_json(out / "COMPARISON_WITH_ONE_NNA.json", {"identity": identity, "rows": merged})
    atomic_json(out / "MEAN_SD_ACROSS_SEEDS.json", {"identity": identity, "groups": aggregates,
        "interpretation": protocol["scope"]})
    fields = sorted({key for row in merged for key in row})
    with (out / "COMPARISON_WITH_ONE_NNA.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(merged)
    flat = []
    for group in aggregates:
        row = {"method": group["method"], "cohort": group["cohort"], "seeds": "/".join(map(str, group["seeds"])),
               "attempts_by_seed": "/".join(map(str, group["attempts_by_seed"])),
               "valid_meshes_by_seed": "/".join(map(str, group["valid_meshes_by_seed"]))}
        for name, value in group["metrics"].items():
            row.update({name + "_mean": value["mean"], name + "_sample_sd": value["sample_sd"], name + "_n": value["n"]})
        flat.append(row)
    fields = sorted({key for row in flat for key in row})
    with (out / "MEAN_SD_ACROSS_SEEDS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
    shutil.copyfile(__file__, out / "tree_comparison_postprocess.py")
    atomic_json(out / "POSTPROCESS_COMPLETE.json", {"status": "completed", "identity": identity,
                "one_NNA_arms": len(scores), "comparison_rows": len(merged), "summary_groups": len(aggregates),
                "main_COMPARISON_ROWS_sha256": file_sha(base / "results/COMPARISON_ROWS.json")})
    if svc is not None:
        saved = engine.publish([out], "cpu_1nna_postprocess_complete", folder_id or engine.OUTPUT_FOLDER)
        atomic_json(out / "POSTPROCESS_DRIVE_SAVED.json", saved)
    print("CPU postprocess complete: 1-NNA for12 mixture arms plus across-seed summaries.", flush=True)
    return {"rows": scores, "aggregates": aggregates, "directory": str(out)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-checks", action="store_true")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    if args.synthetic_checks:
        print(json.dumps(synthetic_checks(), indent=2))
    else:
        run(base=args.base)
