"""Local entry point for the unchanged, archived dated-tree comparison.

This wrapper only selects local paths and disables the original Drive publishing.
It does not alter the sampler, training, extraction or metric implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("action", choices=("protocol", "preflight", "run", "postprocess"))
    command.add_argument("--base", type=Path, default=Path("work/tree_comparison"),
                         help="New local experiment directory; use the same path for all stages.")
    command.add_argument("--archives", type=Path,
                         help="Directory containing loss_ablation.zip, matched_inputs.zip and mesh_inputs.zip.")
    command.add_argument("--references", type=Path,
                         help="Reference root containing the raw_stl and processed_npz paths in reference_manifest.json.")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    import tree_sampler_comparison as engine
    if args.action == "protocol":
        print(json.dumps(engine.PROTOCOL, indent=2))
        return 0

    base = args.base.resolve()
    archive_root = args.archives.resolve() if args.archives else None
    reference_root = args.references.resolve() if args.references else None

    def local_input(file_id, destination):
        """Resolve the original archived file from an explicitly supplied local root."""
        destination = Path(destination)
        try:
            relative = destination.relative_to(base / "downloads")
            source = archive_root / relative if archive_root else None
        except ValueError:
            try:
                relative = destination.relative_to(base / "refs")
                source = reference_root / relative if reference_root else None
            except ValueError:
                source = None
        if source is None or not source.is_file():
            raise FileNotFoundError(
                f"Missing locked input {destination}. Supply --archives and --references "
                "as described in code/tree_comparison/README.md. "
                f"The local input reference is {file_id}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

    # No authentication, network requests or remote writes from this entry point.
    engine.download = local_input
    if args.action == "preflight":
        engine.preflight(None, base=base, folder_id=None)
    elif args.action == "run":
        engine.run(None, base=base, folder_id=None)
    else:
        import tree_comparison_postprocess as postprocess
        postprocess.run(base=base, engine=engine)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
