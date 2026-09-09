"""Run the paired experiment, then build its offline interactive reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from propflow.snapshots.types import EngineSnapshot

from .analysis import StabilityWindow, applied_coefficients
from .graph import (
    Settings,
    canonical_json,
    create_problem,
    fingerprint,
    restore_problem,
)


class CostSnapshotManager:
    """Use the engine snapshot protocol without retaining duplicate graph data."""

    def capture_step(self, step_index, step, engine) -> EngineSnapshot:
        return EngineSnapshot(
            step=step_index, lambda_=0.0, dom={}, N_var={}, N_fac={}, Q={}, R={}
        )


def run_variant(problem: dict, settings: Settings, variant: str, output: Path) -> dict:
    """Run one variant with exact inputs, streamed capture, and shared defaults."""
    import torch
    from propflow.integrations.dabp import DABPEngine, DABPEngineSymSplit

    classes = {"symmetric": DABPEngineSymSplit, "asymmetric": DABPEngine}
    if variant not in classes:
        raise ValueError(f"unknown variant: {variant}")
    torch.manual_seed(settings.model_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    engine = classes[variant](
        factor_graph=restore_problem(problem),
        device="cpu",
        record_weights=True,
        update_interval=settings.update_interval,
        restart_period=max(2000, settings.max_iterations),
        snapshot_manager=CostSnapshotManager(),
    )
    metadata = engine.weight_metadata()
    state_hash = hashlib.sha256()
    for key, value in engine._require_abp().state_dict().items():
        state_hash.update(key.encode())
        state_hash.update(value.detach().cpu().numpy().tobytes())
    metadata["initial_model_sha256"] = state_hash.hexdigest()
    metadata["graph_sha256"] = fingerprint(problem)
    metadata["split_ratio"] = engine.split_ratio
    metadata["settings"] = asdict(settings)
    source_target = np.asarray(metadata["src_trg_idxes"], dtype=np.int32)
    n_targets = len(metadata["trg_var_names"])
    n_sources = len(source_target)
    n_heads = metadata["num_heads"]
    window = StabilityWindow(settings.stable_window, settings.tolerance)
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(
        prefix=".capture-", dir=output.parent
    ) as temporary:
        shapes = {
            "damped": (settings.max_iterations, n_targets, 2, n_heads),
            "attention": (settings.max_iterations, n_sources, n_heads),
            "assignments": (settings.max_iterations, settings.nodes),
            "costs": (settings.max_iterations,),
        }
        arrays = {
            key: np.lib.format.open_memmap(
                Path(temporary) / f"{key}.npy",
                mode="w+",
                dtype=np.int32 if key == "assignments" else np.float64,
                shape=shape,
            )
            for key, shape in shapes.items()
        }
        first_assignment_stable = None
        first_weight_stable = None
        for iteration in range(settings.max_iterations):
            engine.step(iteration)
            record = engine.weights_log.pop()
            if record["iteration"] != iteration:
                raise ValueError("recorded iteration does not match the engine step")
            damped = record["damped_weights"]
            attention = record["attention_weight"]
            damping, _, coefficients = applied_coefficients(
                damped, attention, source_target
            )
            assignment = np.array(
                [engine.assignments[n] for n in metadata["ordered_names"]],
                dtype=np.int32,
            )
            snapshot = engine.get_snapshot(iteration)
            if snapshot is None or not np.isfinite(snapshot.global_cost):
                raise ValueError("engine failed to provide a finite cost snapshot")
            arrays["damped"][iteration] = damped
            arrays["attention"][iteration] = attention
            arrays["assignments"][iteration] = assignment
            arrays["costs"][iteration] = snapshot.global_cost
            stability = window.update(
                assignment, np.concatenate([damping, coefficients])
            )
            if stability.assignments and first_assignment_stable is None:
                first_assignment_stable = iteration + 1
            if stability.weights and first_weight_stable is None:
                first_weight_stable = iteration + 1
            if (iteration + 1) % 100 == 0 or stability.converged:
                print(
                    f"{variant}: iteration {iteration + 1}; "
                    f"assignments stable={stability.assignments}; "
                    f"weights stable={stability.weights}; "
                    f"{time.monotonic() - start:.1f}s",
                    flush=True,
                )
            if stability.converged:
                break
        count = iteration + 1
        outcome = {
            "variant": variant,
            "iterations": count,
            "stop_reason": (
                "stable_assignments_and_weights"
                if stability.converged
                else "iteration_limit"
            ),
            "assignment_stable": stability.assignments,
            "weights_stable": stability.weights,
            "window_max_range": stability.range_max,
            "first_assignment_stable_iteration": first_assignment_stable,
            "first_weights_stable_iteration": first_weight_stable,
            "elapsed_seconds": time.monotonic() - start,
            "split_ratio": engine.split_ratio,
            "initial_model_sha256": metadata["initial_model_sha256"],
            "graph_sha256": metadata["graph_sha256"],
        }
        metadata["outcome"] = outcome
        np.savez_compressed(
            output,
            metadata_json=np.array(canonical_json(metadata)),
            iteration=np.arange(1, count + 1, dtype=np.int32),
            **{name: value[:count] for name, value in arrays.items()},
        )
        arrays.clear()
    print(f"saved {output.name}: {count} iterations", flush=True)
    return outcome


def main() -> None:
    """Create an immutable output directory or regenerate its report from data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--graph-file", type=Path)
    for key, value in asdict(Settings()).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=value
        )
    args = parser.parse_args()
    from .report import build_report

    if args.report_only:
        build_report(args.out_dir)
        return
    settings = Settings(**{key: getattr(args, key) for key in asdict(Settings())})
    if args.out_dir.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    problem = (
        json.loads(args.graph_file.read_text())
        if args.graph_file
        else create_problem(settings)
    )
    graph_dimensions = (len(problem["nodes"]), problem["generation"]["domain"])
    if graph_dimensions != (settings.nodes, settings.domain):
        raise SystemExit("saved graph node count/domain do not match settings")
    restore_problem(problem)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "graph.json").write_text(canonical_json(problem) + "\n")
    import networkx
    import torch
    import torch_geometric

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "graph_sha256": fingerprint(problem),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "networkx": networkx.__version__,
            "torch_geometric": torch_geometric.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.iterdir()
            if path.suffix in {".py", ".js", ".html", ".css"}
        },
        "runs": {},
    }
    manifest_path = args.out_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for variant in ("symmetric", "asymmetric"):
        manifest["runs"][variant] = run_variant(
            problem, settings, variant, args.out_dir / f"{variant}.npz"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    initial_hashes = {r["initial_model_sha256"] for r in manifest["runs"].values()}
    if len(initial_hashes) != 1:
        raise RuntimeError(
            "paired variants did not start with identical network parameters"
        )
    build_report(args.out_dir)


if __name__ == "__main__":
    main()
