#!/usr/bin/env python3
"""Verify the released original inputs and inspect one saved generated skull."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def verify_sources():
    base = ROOT / "data/source_meshes"
    with (base / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise ValueError("Expected the 100 original source meshes")
    for row in rows:
        data = (base / row["filename"]).read_bytes()
        if len(data) != int(row["bytes"]) or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Source mesh checksum mismatch: {row['filename']}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, help="An alternative saved NPZ mesh")
    parser.add_argument("--output", type=Path, default=ROOT / "output/quickstart")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    count = verify_sources()
    print(f"Verified {count} original source STL meshes.")
    if args.verify_only:
        return
    from tools.export_stl import load_mesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    manifest = json.loads((ROOT / "outputs/figure_meshes/manifest.json").read_text())
    row = next(item for item in manifest["meshes"] if item["species"] == "C.Parvulus" and item["slot"] == 0)
    path = args.mesh or ROOT / "outputs/figure_meshes" / row["file"]
    if args.mesh is None and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Saved figure mesh checksum mismatch")
    vertices, faces = load_mesh(path)
    chosen = np.arange(len(faces))
    triangles = vertices[faces[chosen]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, norms, out=normals, where=norms > 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (a, b, depth, title) in zip(axes, [(0, 2, 1, "Side"), (0, 1, 2, "Top"), (1, 2, 0, "Front")]):
        order = np.argsort(triangles[:, :, depth].mean(axis=1))
        polygons = triangles[order][:, :, [a, b]]
        brightness = 0.55 + 0.45 * np.abs(normals[order, depth])
        colors = brightness[:, None] * np.array([0.84, 0.80, 0.72])
        collection = PolyCollection(polygons, facecolors=colors, edgecolors="none")
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Saved generated skull · normalized decoder coordinates", fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    preview = args.output / "inspection.png"
    fig.savefig(preview, dpi=160, facecolor="white")
    plt.close(fig)
    summary = {"source_meshes_verified": count, "mesh": str(path), "vertices": len(vertices),
               "faces": len(faces), "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
               "preview_face_count": len(chosen), "preview": str(preview)}
    (args.output / "mesh_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Loaded {len(vertices):,} vertices and {len(faces):,} triangles.")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
