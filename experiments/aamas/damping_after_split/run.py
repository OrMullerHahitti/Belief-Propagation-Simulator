"""Run only the approved damped continuation, retaining portable recovery data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from experiments.aamas.late_split.core import (
    Config,
    advance,
    input_fingerprint,
    load_input,
    tail_kind,
    trace_arrays,
    verify_costs,
    write_json,
)
from experiments.aamas.late_split.population import dependencies
from experiments.aamas.late_split.resume import checked, read_arrays, save_runtime, seal
from experiments.aamas.late_split.run import ROOT, sha
from .state import (
    first_update_check,
    install_previous_q,
    recover_terminal,
    restore_damped,
    serialized_state,
)


def source_hashes() -> dict:
    """Freeze native dependencies and this continuation implementation."""
    paths = dependencies() + [
        Path(__file__),
        Path(__file__).with_name("state.py"),
        Path(__file__).with_name("__init__.py"),
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def prepare(parent: Path, out: Path) -> dict:
    """Validate and copy parent evidence, preserving the original run byte-for-byte."""
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if (
        parent_manifest["seeds"] != list(range(50))
        or parent_manifest["domain_size"] != 20
        or parent_manifest["config"] != asdict(Config())
    ):
        raise ValueError("parent is not the approved 50-seed domain-20 study")
    for relative, checksum in parent_manifest["source_sha256"].items():
        if sha(ROOT / relative) != checksum:
            raise ValueError(f"parent dependency changed: {relative}")
    expected = {
        "parent_manifest_sha256": sha(parent / "manifest.json"),
        "parent_checksum_index_sha256": sha(parent / "SHA256SUMS"),
        "source_sha256": source_hashes(),
        "config": parent_manifest["config"],
        "seeds": parent_manifest["seeds"],
        "additional_updates": 1000,
        "damping": 0.9,
        "checkpoint_every": 250,
    }
    if (out / "manifest.json").exists():
        saved = json.loads((out / "manifest.json").read_text())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "saved protocol or source differs; use a new output directory"
            )
        checked(out, "parent_evidence.json")
        return saved
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for line in (parent / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        if sha(parent / relative) != checksum:
            raise ValueError(f"parent evidence changed: {relative}")
        index[relative] = checksum
    copied = []
    for seed in expected["seeds"]:
        case = parent / f"random_dense_{seed}"
        paths = [
            case / name
            for name in [
                "input.npz",
                "prefix.npz",
                "best_trace.npz",
                "best_checkpoint.json.gz",
                "best_result.json",
                "DMS_split_0.5_trace.npz",
            ]
        ]
        for checkpoint in ["000750", "001000"]:
            checkpoint_dir = case / "checkpoints/best" / checkpoint
            if checkpoint_dir.exists():
                paths.extend(p for p in checkpoint_dir.iterdir() if p.is_file())
        paths.append(parent / "mgm_tail_values" / f"seed_{seed}_best_mgm.json")
        for source in paths:
            relative = str(source.relative_to(parent))
            target = out / "parent_evidence" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != index[relative]:
                raise ValueError("copied evidence differs")
            copied.append(str(target.relative_to(out)))
    shutil.copyfile(parent / "manifest.json", out / "parent_manifest.json")
    shutil.copyfile(parent / "SHA256SUMS", out / "parent_SHA256SUMS")
    seal(
        out,
        "parent_evidence.json",
        copied + ["parent_manifest.json", "parent_SHA256SUMS"],
        validated_parent_files=len(index),
    )
    for relative in expected["source_sha256"]:
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(Path(__file__).with_name("README.md"), out / "PROTOCOL.md")
    manifest = {
        **expected,
        "parent": str(parent),
        "agents": 50,
        "domain_size": 20,
        "density": 0.6,
        "mode": "best",
        "new_mgm_or_bb": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "previous_q": "last actually emitted undamped Q; recovered by checked suffix replay",
        "axis": "1000 observation + 1000 undamped split + 1000 damped split updates",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def run_seed(task: tuple) -> dict:
    """Recover one branch or resume it, then save all new native BP outcomes."""
    output, seed, config, horizon, every = task
    out = Path(output)
    case = out / f"random_dense_{seed}"
    case.mkdir(exist_ok=True)
    previous = checked(case, "result.json")
    if previous is not None:
        print(f"REUSE seed={seed} cost={previous['final_cost']:.6f}", flush=True)
        return previous
    started = time.monotonic()
    source = out / "parent_evidence" / case.name
    original = load_input(source / "input.npz")
    old = read_arrays(source / "best_trace.npz")
    empty = {
        "variable_names": old["variable_names"],
        "costs": np.empty(0),
        "iterations": np.empty(0, dtype=np.int64),
        "assignments": np.empty((0, len(old["variable_names"])), dtype=np.int64),
    }
    initial = case / "checkpoints/000000"
    if checked(initial, "complete.json") is None:
        engine, last_q, next_i, recovery = recover_terminal(source, original, config)
        save_runtime(case / "recovered_undamped", engine, next_i, empty)
        damped, _, _ = restore_damped(case / "recovered_undamped", original)
        install_previous_q(damped, last_q)
        damped.damping_factor = config.damping
        state_before = serialized_state(damped, next_i)
        recovery["first_update_damping"] = first_update_check(damped)
        if state_before != serialized_state(damped, next_i):
            raise ValueError("damping verification mutated the starting state")
        write_json(case / "recovery.json", recovery)
        save_runtime(initial, damped, next_i, empty)
        print(
            f"RECOVERED seed={seed} replay={recovery['replayed_updates']} "
            f"native_next={next_i}",
            flush=True,
        )
    checkpoints = sorted((case / "checkpoints").glob("*/complete.json"))
    engine, trace, next_i = restore_damped(checkpoints[-1].parent, original)
    if engine.damping_factor != config.damping:
        raise ValueError("restored damping differs")
    completed = len(trace["costs"])
    start_iteration = next_i - completed
    fingerprint = input_fingerprint(engine.graph)
    snapshots = []
    for count in range(completed + 1, horizon + 1):
        snapshots.append(advance(engine, start_iteration + count - 1))
        if count % every == 0 or count == horizon:
            added = trace_arrays(snapshots, list(trace["variable_names"]))
            for key in ["iterations", "costs", "assignments"]:
                trace[key] = np.concatenate([trace[key], added[key]], axis=0)
            snapshots.clear()
            verify_costs(original, trace)
            save_runtime(
                case / "checkpoints" / f"{count:06d}",
                engine,
                start_iteration + count,
                trace,
            )
            print(
                f"PROGRESS seed={seed} damped_updates={count} "
                f"cost={trace['costs'][-1]:.6f}",
                flush=True,
            )
    if len(trace["costs"]) != horizon or input_fingerprint(engine.graph) != fingerprint:
        raise ValueError("continuation horizon or split graph changed")
    error = verify_costs(original, trace)
    np.savez_compressed(case / "damped_trace.npz", **trace)
    recovery = json.loads((case / "recovery.json").read_text())
    files = ["damped_trace.npz", "recovery.json"]
    files += [
        str(p.relative_to(case))
        for p in (case / "checkpoints").rglob("*")
        if p.is_file()
    ]
    files += [
        str(p.relative_to(case))
        for p in (case / "recovered_undamped").iterdir()
        if p.is_file()
    ]
    result = seal(
        case,
        "result.json",
        files,
        seed=seed,
        additional_updates=horizon,
        final_cost=float(trace["costs"][-1]),
        best_new_cost=float(trace["costs"].min()),
        initial_cost=float(old["costs"][-1]),
        damping=config.damping,
        native_start_iteration=start_iteration,
        runtime_graph_sha256=fingerprint,
        original_input_sha256=input_fingerprint(original),
        tail_kind=tail_kind(trace["assignments"], min(config.tail_steps, horizon)),
        original_cost_max_error=error,
        seconds=time.monotonic() - started,
        replayed_updates=recovery["replayed_updates"],
    )
    print(
        f"DONE seed={seed} cost={result['final_cost']:.6f} "
        f"seconds={result['seconds']:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    """Execute the approved population using independent resumable seed jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest = prepare(args.parent.resolve(), args.out.resolve())
    cfg = Config(**manifest["config"])
    tasks = [
        (
            str(args.out.resolve()),
            seed,
            cfg,
            manifest["additional_updates"],
            manifest["checkpoint_every"],
        )
        for seed in manifest["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [
            future.result()
            for future in as_completed([pool.submit(run_seed, task) for task in tasks])
        ]
    write_json(
        args.out / "COMPLETION.json",
        {
            "seeds": sorted(r["seed"] for r in results),
            "completed": len(results),
            "new_bp_updates": sum(r["additional_updates"] for r in results),
            "recovery_replayed_updates": sum(r["replayed_updates"] for r in results),
            "mean_final_cost": float(np.mean([r["final_cost"] for r in results])),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("COMPLETE all 50 damped continuations saved", flush=True)


if __name__ == "__main__":
    main()
