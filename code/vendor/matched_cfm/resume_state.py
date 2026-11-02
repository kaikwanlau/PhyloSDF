"""Exact RNG restoration and checksummed checkpoint I/O for matched CFM.

Checkpoints are trusted local experiment artifacts: the checksum detects corruption,
but does not make an untrusted pickle safe. A checkpoint and its SHA sidecar are
individually atomic. A crash between the two replacements fails verification.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterator

import numpy as np
import torch


RNG_SCHEMA_VERSION = 1


def capture_rng(sampler_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture independent copies of global and optional sampler RNG states.

    Other explicit generators, such as a device-side training-noise generator,
    must be checkpointed by their owner with ``generator.get_state().clone()``.
    """
    np_state = np.random.get_state()
    np_state = (np_state[0], np_state[1].copy(), *np_state[2:])
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return {
        "schema_version": RNG_SCHEMA_VERSION,
        "python": random.getstate(),
        "numpy": np_state,
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_states,
        "sampler": None if sampler_generator is None else sampler_generator.get_state().clone(),
        "sampler_device": None if sampler_generator is None else str(sampler_generator.device),
    }


def restore_rng(state: dict[str, Any], sampler_generator: torch.Generator | None = None) -> None:
    """Restore an RNG snapshot, rejecting missing generators or device changes."""
    required = {"schema_version", "python", "numpy", "torch_cpu", "torch_cuda", "sampler", "sampler_device"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Incomplete RNG snapshot: missing {sorted(missing)}")
    if state["schema_version"] != RNG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported RNG snapshot version: {state['schema_version']}")
    saved_sampler = state["sampler"]
    if saved_sampler is not None and sampler_generator is None:
        raise ValueError("Snapshot contains sampler RNG state but no sampler generator was supplied")
    if saved_sampler is None and sampler_generator is not None:
        raise ValueError("A sampler generator was supplied but the snapshot contains no sampler state")
    if sampler_generator is not None and state["sampler_device"] != str(sampler_generator.device):
        raise ValueError("Sampler RNG device differs from the saved snapshot")
    saved_cuda = state["torch_cuda"]
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(saved_cuda) != current_cuda_count:
        raise ValueError(
            f"CUDA device count differs from RNG snapshot: saved={len(saved_cuda)}, current={current_cuda_count}"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if saved_cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved_cuda])
    if sampler_generator is not None:
        sampler_generator.set_state(saved_sampler.cpu())


@contextmanager
def isolated_rng(sampler_generator: torch.Generator | None = None) -> Iterator[None]:
    """Restore global RNG streams after diagnostics, including on exceptions."""
    snapshot = capture_rng(sampler_generator)
    try:
        yield
    finally:
        restore_rng(snapshot, sampler_generator)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> str:
    """Durably replace a checkpoint and its ``.sha256`` sidecar; return its hash.

    Only one writer may own a checkpoint path. A failed write before replacement
    preserves the previous checkpoint. Interrupted two-file commits fail closed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = Path(str(target) + ".sha256")
    checkpoint_temp: Path | None = None
    checksum_temp: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        checkpoint_temp = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_file(checkpoint_temp)

        descriptor, name = tempfile.mkstemp(prefix=f".{checksum_path.name}.", suffix=".tmp", dir=target.parent)
        checksum_temp = Path(name)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(checkpoint_temp, target)
        _fsync_directory(target.parent)
        os.replace(checksum_temp, checksum_path)
        _fsync_directory(target.parent)
        return digest
    finally:
        for temporary in (checkpoint_temp, checksum_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def verified_torch_load(path: str | os.PathLike[str], *, map_location: Any = "cpu") -> Any:
    """Verify SHA-256 before loading those same bytes with ``weights_only=False``."""
    target = Path(path)
    checksum_path = Path(str(target) + ".sha256")
    try:
        expected_digest = checksum_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Missing or unreadable checkpoint checksum: {checksum_path}") from exc
    if len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest):
        raise ValueError(f"Malformed checkpoint checksum: {checksum_path}")
    checkpoint_bytes = target.read_bytes()
    actual_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {target}")
    return torch.load(io.BytesIO(checkpoint_bytes), map_location=map_location, weights_only=False)
