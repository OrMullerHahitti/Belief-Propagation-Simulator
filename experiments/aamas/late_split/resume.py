"""Checksummed phase records and exact continuation on a saved runtime graph."""

from pathlib import Path
import json

import numpy as np

from .core import (
    Checkpoint,
    input_fingerprint,
    load_input,
    make_engine,
    save_input,
    write_json,
)
from .run import sha


def read_arrays(path: Path) -> dict:
    """Read numeric evidence without allowing executable pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def seal(directory: Path, name: str, files: list[str], **metadata) -> dict:
    """Commit a phase only after all of its evidence has been saved."""
    record = {**metadata, "files_sha256": {p: sha(directory / p) for p in files}}
    write_json(directory / name, record)
    return record


def checked(directory: Path, name: str) -> dict | None:
    """Return a complete phase, rejecting changed or missing evidence."""
    path = directory / name
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    for relative, checksum in record["files_sha256"].items():
        if not (directory / relative).exists() or sha(directory / relative) != checksum:
            raise ValueError(f"changed phase evidence: {directory / relative}")
    return record


def save_runtime(directory: Path, engine, next_iteration: int, trace: dict) -> None:
    """Save the graph and dynamic state together after native cycle events."""
    directory.mkdir(parents=True, exist_ok=True)
    save_input(engine.graph, directory / "graph.npz")
    Checkpoint.capture(
        engine, next_iteration, input_fingerprint(engine.graph), runtime_graph=True
    ).save(directory / "state.json.gz")
    np.savez_compressed(directory / "trace.npz", **trace)
    seal(
        directory,
        "complete.json",
        ["graph.npz", "state.json.gz", "trace.npz"],
        next_iteration=next_iteration,
        split_applied=getattr(engine, "_split_applied", False),
        split_at=getattr(engine, "split_at_iter", 10**12),
        split_events=getattr(engine, "split_events", []),
    )


def restore_runtime(directory: Path, original, config):
    """Restore equivalent Q damping or released damping and original-cost scoring."""
    record = checked(directory, "complete.json")
    if record is None:
        raise ValueError("runtime checkpoint is incomplete")
    cp = Checkpoint.load(directory / "state.json.gz")
    engine = make_engine(load_input(directory / "graph.npz"), config, 10**12)
    # the normalization period is part of dynamic execution, not recomputed state
    engine.graph_diameter = cp.graph_diameter
    cp.restore(engine)
    engine.graph._original_factors = original.original_factors
    engine._split_applied = record["split_applied"]
    engine.split_at_iter = record["split_at"]
    engine.split_events = record["split_events"]
    return engine, read_arrays(directory / "trace.npz"), cp.next_iteration
