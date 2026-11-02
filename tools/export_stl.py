#!/usr/bin/env python3
"""Export a saved PhyloSDF mesh without changing its coordinates or topology."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_mesh(path):
    with np.load(path, allow_pickle=False) as mesh:
        vertices = mesh["vertices"].copy()
        faces = mesh["faces"].copy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Expected finite N-by-3 vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Expected integer M-by-3 faces")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Face indices are out of bounds")
    return vertices, faces


def export(path, vertices, faces):
    triangles = np.asarray(vertices, dtype=np.float32)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, out=normals, where=lengths > 0)
    records = np.zeros(len(faces), dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records["normal"] = normals
    records["vertices"] = triangles
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"PhyloSDF unchanged saved triangle coordinates".ljust(80, b" "))
        stream.write(np.asarray([len(faces)], dtype="<u4").tobytes())
        stream.write(records.tobytes())
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    vertices, faces = load_mesh(args.input)
    print(export(args.output, vertices, faces))
