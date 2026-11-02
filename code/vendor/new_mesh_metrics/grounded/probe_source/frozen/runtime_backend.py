"""Safe checkpoint loading and portable float32 DeepSDF inference.

PyTorch, when installed, always loads with weights_only=True and map_location=cpu.
The CPU fallback never imports checkpoint-defined code: it preflights pickle
opcodes, rejects all globals except the three explicitly observed in this
handoff, and implements storage/tensor reconstruction locally. It intentionally
does not support arbitrary PyTorch pickles or optimizer/object checkpoints.
NumPy/PyTorch numerical parity must still be checked on a PyTorch runtime.
"""
from __future__ import annotations

import collections
import argparse
import importlib.util
import io
import math
import os
import pickle
import pickletools
from pathlib import Path
import zipfile

if __name__ == "__main__":
    _thread_parser = argparse.ArgumentParser(add_help=False)
    _thread_parser.add_argument("--threads", type=int, default=2)
    _early_threads, _ = _thread_parser.parse_known_args()
    if not 1 <= _early_threads.threads <= 8:
        raise SystemExit("--threads must be between 1 and 8")
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = str(_early_threads.threads)

import numpy as np

ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "FloatStorage"),
}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_PICKLE_BYTES = 8 * 1024 * 1024


def inspect_checkpoint(path):
    """Inspect ZIP/pickle metadata without executing its pickle instructions."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
            raise ValueError("Checkpoint exceeds the restricted-loader size limit")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members are forbidden")
        candidates = [n for n in names if n.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one PyTorch data.pkl")
        name = candidates[0]
        if archive.getinfo(name).file_size > MAX_PICKLE_BYTES:
            raise ValueError("Checkpoint pickle is too large")
        data = archive.read(name)
        ops = list(pickletools.genops(data))
        globals_seen = set()
        for opcode, value, _ in ops:
            if opcode.name == "GLOBAL":
                module, obj = value.split(" ", 1)
                if (module, obj) not in ALLOWED_GLOBALS:
                    raise ValueError(f"Forbidden pickle global: {module}.{obj}")
                globals_seen.add((module, obj))
            if opcode.name in {"STACK_GLOBAL", "EXT1", "EXT2", "EXT4", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "PERSID"}:
                raise ValueError(f"Unsupported pickle opcode: {opcode.name}")
        prefix = name.rsplit("/", 1)[0]
        order_name = f"{prefix}/byteorder"
        byteorder = archive.read(order_name).decode("ascii") if order_name in names else "little"
        if byteorder not in {"little", "big"}:
            raise ValueError("Invalid storage byte order")
        return {
            "data_pickle_member": name,
            "pickle_protocol": next((v for o, v, _ in ops if o.name == "PROTO"), 0),
            "globals": [".".join(g) for g in sorted(globals_seen)],
            "opcode_counts": dict(collections.Counter(o.name for o, _, _ in ops)),
            "storage_members": len([n for n in names if n.startswith(f"{prefix}/data/")]),
            "uncompressed_bytes": sum(i.file_size for i in infos),
            "byteorder": byteorder,
            "safety": "preflight plus exact-global restricted unpickler; no unrestricted pickle.load",
        }


class _FloatStorage:
    pass


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    if not isinstance(storage, np.ndarray) or storage.ndim != 1:
        raise ValueError("Unexpected tensor storage")
    if len(size) != len(stride) or len(size) > 8:
        raise ValueError("Invalid tensor rank/strides")
    if any(type(v) is not int or v < 0 for v in (*size, *stride, storage_offset)):
        raise ValueError("Invalid tensor shape/offset/strides")
    if requires_grad not in (False, True) or not isinstance(backward_hooks, (dict, collections.OrderedDict)):
        raise ValueError("Unsupported tensor rebuild parameters")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Unsupported tensor metadata")
    # Use Python integers so malicious dimensions cannot overflow int64 and
    # bypass the allocation bound before copying the reconstructed view.
    numel = math.prod(size) if size else 1
    max_index = storage_offset + sum((s - 1) * st for s, st in zip(size, stride)) if numel else storage_offset
    if numel > MAX_ARCHIVE_BYTES // 4 or max_index >= storage.size or storage_offset > storage.size:
        raise ValueError("Tensor view exceeds storage")
    view = np.ndarray(shape=tuple(size), dtype=storage.dtype, buffer=storage,
                      offset=storage_offset * storage.itemsize,
                      strides=tuple(st * storage.itemsize for st in stride))
    return np.array(view, dtype=np.float32, order="C", copy=True)


class _RestrictedTorchUnpickler(pickle.Unpickler):
    def __init__(self, file, archive, prefix, byteorder):
        super().__init__(file)
        self.archive, self.prefix, self.byteorder = archive, prefix, byteorder
        self.storages = {}

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
            ("torch", "FloatStorage"): _FloatStorage,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Forbidden global {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, identity):
        if not isinstance(identity, tuple) or len(identity) != 5:
            raise pickle.UnpicklingError("Unexpected persistent storage identity")
        kind, storage_type, key, location, numel = identity
        if kind != "storage" or storage_type is not _FloatStorage:
            raise pickle.UnpicklingError("Only FloatStorage is supported")
        if not isinstance(key, str) or not key.isdigit() or not isinstance(location, str):
            raise pickle.UnpicklingError("Invalid storage identifier")
        if type(numel) is not int or not 0 <= numel <= MAX_ARCHIVE_BYTES // 4:
            raise pickle.UnpicklingError("Invalid storage length")
        if key not in self.storages:
            member = f"{self.prefix}/data/{key}"
            if self.archive.getinfo(member).file_size != numel * 4:
                raise pickle.UnpicklingError("Storage byte count mismatch")
            raw = self.archive.read(member)
            dtype = np.dtype("<f4" if self.byteorder == "little" else ">f4")
            self.storages[key] = np.frombuffer(raw, dtype=dtype)
        elif self.storages[key].size != numel:
            raise pickle.UnpicklingError("Inconsistent shared storage length")
        return self.storages[key]


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(v) for v in value)
    return value


def load_checkpoint(path, force_numpy=False):
    """Return the checkpoint mapping with all tensor leaves converted to NumPy."""
    path = Path(path)
    info = inspect_checkpoint(path)
    if not force_numpy and importlib.util.find_spec("torch") is not None:
        import torch
        # Do not retry with weights_only=False if a restricted load fails.
        result = _to_numpy(torch.load(path, map_location="cpu", weights_only=True))
    else:
        with zipfile.ZipFile(path) as archive:
            name = info["data_pickle_member"]
            prefix = name.rsplit("/", 1)[0]
            result = _RestrictedTorchUnpickler(io.BytesIO(archive.read(name)), archive, prefix, info["byteorder"]).load()
    if not isinstance(result, dict):
        raise ValueError("Expected a checkpoint dictionary")
    return result


class NumpyDecoder:
    """Exact source architecture, float32 inference; parity pending PyTorch QA.

    Legacy nn.utils.weight_norm defaults to dim=0: each output row of weight_v
    is normalized across its input columns, then multiplied by weight_g.
    Matrix-multiply rounding can differ from PyTorch CPU/CUDA kernels.
    """
    def __init__(self, checkpoint):
        cfg = checkpoint["config"]
        self.latent_dim = int(cfg["latent_dim"])
        self.skip_connection = set(cfg.get("skip_connection", [4]))
        self.num_layers = int(cfg["num_layers"])
        state = checkpoint["decoder_state_dict"]
        self.layers = []
        for i in range(self.num_layers):
            prefix = f"layers.{i}."
            if prefix + "weight_v" in state:
                v = np.asarray(state[prefix + "weight_v"], dtype=np.float32)
                g = np.asarray(state[prefix + "weight_g"], dtype=np.float32)
                norms = np.sqrt(np.sum(v * v, axis=1, keepdims=True, dtype=np.float32))
                if np.any(norms == 0):
                    raise ValueError("Zero weight_norm row")
                w = np.asarray(v * (g / norms), dtype=np.float32)
            else:
                w = np.asarray(state[prefix + "weight"], dtype=np.float32)
            b = np.asarray(state[prefix + "bias"], dtype=np.float32)
            expected_input = self.latent_dim + 3 if i == 0 else int(cfg["hidden_dim"]) + (self.latent_dim + 3 if i in self.skip_connection else 0)
            expected_output = 1 if i == self.num_layers - 1 else int(cfg["hidden_dim"])
            if w.shape != (expected_output, expected_input) or b.shape != (expected_output,):
                raise ValueError(f"Layer {i} shape mismatch: {w.shape}, {b.shape}")
            self.layers.append((np.ascontiguousarray(w.T), b))

    def predict(self, points, code):
        points = np.asarray(points, dtype=np.float32)
        code = np.asarray(code, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or code.shape != (self.latent_dim,):
            raise ValueError("Expected points Nx3 and a single latent vector")
        if not np.isfinite(points).all() or not np.isfinite(code).all():
            raise ValueError("Non-finite decoder input")
        input_x = np.concatenate((np.broadcast_to(code, (len(points), self.latent_dim)), points), axis=1)
        x = input_x
        for i, (wt, bias) in enumerate(self.layers):
            if i in self.skip_connection:
                x = np.concatenate((x, input_x), axis=1)
            x = np.matmul(x, wt)
            x += bias
            if i < self.num_layers - 1:
                np.maximum(x, np.float32(0), out=x)
        np.tanh(x, out=x)
        return x.reshape(-1)


def _main():
    parser = argparse.ArgumentParser(description="Audit actual PhyloSDF checkpoints without modifying source inputs")
    parser.add_argument("--root", type=Path, required=True, help="Extracted handoff root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    from checkpoint_audit import run_audit
    run_audit(args.root, args.out, args.threads)


if __name__ == "__main__":
    _main()
