#!/usr/bin/env python3
"""Local entry point for PhyloSDF studies with private storage locators redacted.

All input files must be staged locally. This entry point does not authenticate
to Google Drive, download files, publish results, or change numerical settings.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
REFERENCE_MANIFEST = HERE.parent / "vendor/new_mesh_metrics/vendor/reference_manifest.json"
STUDIES = {
    "final": ("fm_engine", "Proxy representation: neutral residual flow and component comparisons"),
    "corrected-size": ("fmfix_engine", "Species-conditioned comparisons with corrected specimen sizes"),
    "loss": ("rep_engine", "All-100 DeepSDF loss comparison: none, proxy, dated tree"),
    "heldout": ("rep_engine", "Strict 79/21 split: representation and unseen-species generation"),
    "shuffled-proxy": ("shuf_engine", "All-100 shuffled proxy-label controls"),
    "shuffled-tree": ("st_engine", "Strict 79/21 shuffled dated-tree controls"),
    "anatomy": ("anat_engine", "Orbit, braincase and beak measurements on original and generated skulls"),
}
DEFAULT_ARMS = {
    "final": ("pooled_residual", "pooled_residual_no_var", "pooled_residual_aug", "pooled_residual_tcal"),
    "corrected-size": ("species_residual_fixsize", "species_fullcode_fixsize", "species_residual_tokenonly"),
    "loss": ("beta0", "beta_proxy", "beta_tree"),
    "heldout": ("ho_beta0", "ho_beta_proxy", "ho_beta_tree"),
    "shuffled-proxy": ("beta_shuf1", "beta_shuf2", "beta_shuf3"),
    "shuffled-tree": ("ho_beta_shuffle1", "ho_beta_shuffle2", "ho_beta_shuffle3"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_specs(study: str) -> dict:
    name = STUDIES[study][0]
    if name == "st_engine":
        name = "rep_engine"
    for node in ast.parse((HERE / f"{name}.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INPUTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"No input specification in {name}")


def raw_source(path: str) -> Path:
    return REPOSITORY / "data/source_meshes" / Path(path).name


def local_reference(args, relative: str) -> Path | None:
    candidates = [args.references / relative, args.workspace / "refs" / relative]
    if relative.startswith("raw_stl/"):
        candidates.append(raw_source(relative))
    return next((p for p in candidates if p.is_file()), None)


def check_inputs(args) -> list[str]:
    errors = []
    for name, (_, expected) in input_specs(args.study).items():
        path = args.inputs / name
        if not path.is_file():
            errors.append(f"Missing archive: {path}")
        elif digest(path) != expected:
            errors.append(f"Archive checksum mismatch: {path}")
    if args.study == "anatomy":
        if args.skull_repo is None:
            errors.append("Anatomy also requires --skull-repo pointing to the pinned skull-morphology checkout.")
        elif not (args.skull_repo / "6_verification/verify_all.py").is_file():
            errors.append(f"Missing anatomy measurement script in {args.skull_repo}")
        else:
            import subprocess
            expected_commit = "1a60c310a88574941da35ad99aacf29266aba060"
            result = subprocess.run(["git", "-C", str(args.skull_repo), "rev-parse", "HEAD"],
                                    capture_output=True, text=True)
            if result.returncode or result.stdout.strip() != expected_commit:
                errors.append(f"skull-morphology must be checked out at {expected_commit}")
            bad = []
            for row in json.loads(REFERENCE_MANIFEST.read_text())["references"]:
                path = args.skull_repo / "data/DF_and_their_relatives" / Path(row["raw_stl"]).name
                if not path.is_file() or digest(path) != row["raw_sha256"]:
                    bad.append(path.name)
            if bad:
                errors.append(f"Anatomy source mesh mismatch for {len(bad)} files; first: {bad[0]}")
    else:
        rows = json.loads(REFERENCE_MANIFEST.read_text())["references"]
        missing, wrong = [], []
        for row in rows:
            for key, sha_key in (("raw_stl", "raw_sha256"), ("processed_npz", "processed_sha256")):
                path = local_reference(args, row[key])
                if path is None:
                    missing.append(row[key])
                elif digest(path) != row[sha_key]:
                    wrong.append(row[key])
        if missing:
            errors.append(f"Missing {len(missing)} reference files under {args.references}; first: {missing[0]}")
        if wrong:
            errors.append(f"Reference checksum mismatch for {len(wrong)} files; first: {wrong[0]}")
    return errors


def load_engine(args):
    """Route local files while preserving scientific calculations and settings."""
    sys.path.insert(0, str(HERE))
    module = importlib.import_module(STUDIES[args.study][0])
    engine = module.rep if args.study == "shuffled-tree" else module
    engine.ROOT = args.workspace
    engine.DL = args.inputs
    engine.SRC = args.workspace / "src"
    if args.study != "anatomy":
        engine.SRC /= "matched_cfm"
    engine.MESHSRC = args.workspace / "meshsrc/new_mesh_metrics"
    engine.REFS = args.workspace / "refs"
    engine.WORK = args.workspace / "work"
    engine.OUT = args.workspace / "results"
    if hasattr(engine, "UPLOAD_DIR"):
        engine.UPLOAD_DIR = args.workspace / "resume"
    if hasattr(module, "UPLOAD_DIR"):
        module.UPLOAD_DIR = args.workspace / "resume"

    def get_local_input(name):
        specs = engine.INPUTS
        path = args.inputs / name
        if not path.is_file():
            raise FileNotFoundError(f"Stage the original archive locally: {path}")
        if digest(path) != specs[name][1]:
            raise RuntimeError(f"Archive checksum mismatch: {path}")
        return path

    by_id = {}
    for row in json.loads(REFERENCE_MANIFEST.read_text())["references"]:
        by_id[row["raw_drive_id"]] = row["raw_stl"]
        by_id[row["processed_drive_id"]] = row["processed_npz"]

    reference_paths = set(by_id.values())

    def fetch_local(file_id, target):
        relative = by_id.get(file_id)
        # Private archives may retain their original identifiers. Route them
        # by an approved destination path without distributing those IDs.
        if relative is None:
            try:
                candidate = Path(target).resolve().relative_to(engine.REFS.resolve()).as_posix()
            except ValueError:
                candidate = None
            if candidate in reference_paths:
                relative = candidate
        source = local_reference(args, relative) if relative else None
        if source is None:
            raise FileNotFoundError(f"Stage reference {relative or file_id} under {args.references}")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)

    engine.get_input = get_local_input
    engine.fetch = fetch_local
    engine.drive = lambda: (_ for _ in ()).throw(RuntimeError("This launcher uses local input files only."))
    if hasattr(engine, "download"):
        engine.download = lambda path: print(f"Saved: {path}")
    if args.study == "anatomy":
        engine.REPO = args.skull_repo

        def existing_skull_repo():
            if engine.REPO is None or not (engine.REPO / "6_verification/verify_all.py").is_file():
                raise FileNotFoundError("Provide the pinned skull-morphology checkout with --skull-repo.")
            # setup() separately verifies all 100 mesh SHA-256 values.
            import subprocess
            head = subprocess.run(["git", "-C", str(engine.REPO), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            if head != engine.REPO_COMMIT:
                raise RuntimeError(f"skull-morphology must be at commit {engine.REPO_COMMIT}")

        engine.clone_repo = existing_skull_repo
    return engine, module


def execute(args):
    errors = check_inputs(args)
    if errors:
        raise RuntimeError("\n".join(errors))
    if args.action == "check":
        print(f"All staged inputs verified for {args.study}.")
        return
    engine, extension = load_engine(args)
    # The original setup validates the GPU, decoder identity, references and environment.
    engine.setup()
    if args.action == "setup":
        return
    if args.action == "validate":
        if args.study == "anatomy":
            print("Anatomy setup verified the mesh checksums, measurement implementation and embedded generated codes.")
        else:
            engine.validate()
        return
    arms = args.arms or list(DEFAULT_ARMS.get(args.study, ()))
    if args.study in ("final", "corrected-size"):
        unknown = set(arms) - set(engine.ARMS)
        if unknown:
            raise ValueError(f"Unknown arms: {', '.join(sorted(unknown))}")
        order = [(seed, arm) for seed in args.seeds for arm in arms]
        engine.run(order=order, download=False)
    elif args.study == "anatomy":
        if args.arms or args.seeds != [42, 43, 44]:
            raise ValueError("Anatomy uses the fixed embedded arms and seeds; omit --arms and --seeds.")
        engine.run(download=False)
    else:
        if args.seeds != [42, 43, 44]:
            raise ValueError("Representation studies use their original fixed seed 42; omit --seeds.")
        study = "loss_ablation" if args.study == "loss" else "shuffle_control" if args.study == "shuffled-proxy" else "heldout"
        engine.run(study, arms)
        if args.study == "shuffled-tree":
            extension.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "check", "setup", "validate", "run"))
    parser.add_argument("--study", choices=tuple(STUDIES))
    parser.add_argument("--inputs", type=Path, default=REPOSITORY / "runs/input_archives")
    parser.add_argument("--references", type=Path, default=REPOSITORY / "data/runtime_references")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--skull-repo", type=Path, help="Pinned skull-morphology checkout for the anatomy study")
    parser.add_argument("--arms", nargs="+", help="Explicit arm names; otherwise use the paper's reported arms")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44],
                        help="Flow seeds for final/corrected-size only; other studies retain their fixed protocol")
    args = parser.parse_args(argv)
    if args.action == "list":
        for name, (_, description) in STUDIES.items():
            print(f"{name:17s} {description}")
            print("  arms: " + ", ".join(DEFAULT_ARMS.get(name, ("fixed anatomy cohorts",))))
        return 0
    if args.study is None:
        parser.error("--study is required for this action")
    if args.study == "shuffled-proxy" and args.arms:
        extra = set(args.arms) - set(DEFAULT_ARMS["shuffled-proxy"])
        if extra:
            parser.error("The paper release includes shuffled-proxy arms beta_shuf1, beta_shuf2 and beta_shuf3 only.")
    args.inputs = args.inputs.resolve()
    args.references = args.references.resolve()
    args.workspace = (args.workspace or REPOSITORY / "runs" / args.study).resolve()
    if args.skull_repo:
        args.skull_repo = args.skull_repo.resolve()
    try:
        execute(args)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
