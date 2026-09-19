"""Run the approved fixed-time pilot before its best-checkpoint counterpart."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.aaai.code.problems import BENCHMARKS
from .core import (
    Checkpoint,
    Config,
    advance,
    input_fingerprint,
    load_input,
    make_engine,
    merge_tail,
    save_input,
    trace_arrays,
    verify_costs,
    write_json,
)

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = (
    "random_dense",
    "random_sparse",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
)
REFERENCE_LABELS = (
    "DMS",
    "MS_split_0.5",
    "MS_split_MGM_200",
    "MS_split_opt_200",
    "DMS_split_0.5",
    "DMS_split_at_1000",
    "DMS_split_0.95",
    "DMS_split_pulse",
)


def sha(path: Path) -> str:
    """Hash an input or source file without reading a full CSV into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files() -> list[Path]:
    """All native and experiment dependencies used by this runner."""
    files = list((ROOT / "src/propflow").rglob("*.py"))
    files += [
        Path(__file__).parent / name for name in ["__init__.py", "core.py", "run.py"]
    ]
    files += [
        ROOT / "experiments/aaai/code" / name
        for name in ["engines.py", "merge.py", "problems.py"]
    ]
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return sorted(files)


def prepare(out: Path, config: Config, benchmarks: list[str], seeds: list[int]) -> None:
    """Freeze code and compact prior-data slices; never modify previous results."""
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    expected = {"config": asdict(config), "benchmarks": benchmarks, "seeds": seeds}
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        if any(previous[k] != v for k, v in expected.items()):
            raise ValueError("existing run has different inputs/configuration")
        if previous["source_sha256"] != hashes:
            raise ValueError("source changed; use a new output directory")
        return
    out.mkdir(parents=True, exist_ok=False)
    for path in source_files():
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    baseline_hashes = {}
    for benchmark in benchmarks:
        path = ROOT / "experiments/aaai/data" / f"{benchmark}_raw_costs.csv"
        baseline_hashes[str(path.relative_to(ROOT))] = sha(path)
        pieces = []
        for chunk in pd.read_csv(path, chunksize=100000):
            selected = chunk[
                chunk.seed.isin(seeds) & chunk.algorithm.isin(REFERENCE_LABELS)
            ]
            if not selected.empty:
                pieces.append(selected)
        if not pieces:
            raise ValueError(f"no baseline rows for {benchmark}")
        rows = pd.concat(pieces)
        for seed in seeds:
            case = out / f"{benchmark}_{seed}"
            case.mkdir()
            selected = rows[rows.seed.eq(seed)]
            if selected.duplicated(["algorithm", "iteration"]).any():
                raise ValueError("duplicate baseline rows")
            arrays = {}
            for label, group in selected.groupby("algorithm"):
                group = group.sort_values("iteration")
                arrays[label] = group.cost.to_numpy()
                arrays[label + "_iterations"] = group.iteration.to_numpy()
            np.savez_compressed(case / "references.npz", **arrays)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    write_json(
        out / "manifest.json",
        {
            **expected,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "source_sha256": hashes,
            "baseline_sha256": baseline_hashes,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "normalization": "native cycle events at i % current_graph_diameter == 0",
            "order": "all fixed-time cases, then all best-checkpoint cases",
        },
    )


def validate_prefix(case: Path, trace: dict) -> float:
    """Require the real unsplit prefix to match the retained corrected DMS line."""
    with np.load(case / "references.npz", allow_pickle=False) as ref:
        costs, iterations = ref["DMS"], ref["DMS_iterations"]
        n = len(trace["costs"])
        if len(costs) < n or not np.array_equal(iterations[:n], trace["iterations"]):
            raise RuntimeError("baseline DMS prefix is missing or misindexed")
        error = float(np.max(np.abs(costs[:n] - trace["costs"])))
        # the inherited harness writes raw costs with f"{cost:.4f}"
        serialized = np.array([float(f"{value:.4f}") for value in trace["costs"]])
        if not np.isfinite(error) or not np.array_equal(serialized, costs[:n]):
            raise RuntimeError(f"corrected DMS prefix mismatch: {error}")
        return error


def prepare_prefix(case: Path, benchmark: str, seed: int, config: Config) -> dict:
    """Capture the earliest best checkpoint during the approved search window."""
    graph = BENCHMARKS[benchmark](seed)
    fingerprint = input_fingerprint(graph)
    save_input(graph, case / "input.npz")
    if input_fingerprint(load_input(case / "input.npz")) != fingerprint:
        raise RuntimeError("saved input roundtrip changed tables/order")
    engine = make_engine(graph, config, config.prefix_steps)
    snapshots, best, best_cost = [], None, float("inf")
    for i in range(config.prefix_steps):
        snapshot = advance(engine, i)
        snapshots.append(snapshot)
        if snapshot.global_cost < best_cost:
            best_cost = snapshot.global_cost
            best = Checkpoint.capture(engine, i + 1, fingerprint)
    fixed = Checkpoint.capture(engine, config.prefix_steps, fingerprint)
    names = [v.name for v in engine.var_nodes]
    trace = trace_arrays(snapshots, names)
    original = load_input(case / "input.npz")
    cost_error = verify_costs(original, trace)
    baseline_error = validate_prefix(case, trace)
    best.save(case / "best_checkpoint.json.gz")
    fixed.save(case / "fixed_checkpoint.json.gz")
    np.savez_compressed(case / "prefix.npz", **trace)
    info = {
        "input_sha256": fingerprint,
        "prefix_best_cost": best_cost,
        "best_checkpoint_index": best.next_iteration - 1,
        "prefix_final_cost": float(trace["costs"][-1]),
        "prefix_steps": config.prefix_steps,
        "normalization_period": fixed.graph_diameter,
        "original_cost_max_error": cost_error,
        "retained_dms_max_error": baseline_error,
    }
    write_json(case / "prefix.json", info)
    return info


def validate_restored_prefix(case: Path, config: Config) -> float:
    """Replay the saved best state to the window end before releasing damping."""
    checkpoint = Checkpoint.load(case / "best_checkpoint.json.gz")
    engine = make_engine(load_input(case / "input.npz"), config, config.prefix_steps)
    checkpoint.restore(engine)
    with np.load(case / "prefix.npz", allow_pickle=False) as reference:
        names = list(reference["variable_names"])
        error = 0.0
        for i in range(checkpoint.next_iteration, config.prefix_steps):
            snapshot = advance(engine, i)
            error = max(error, abs(snapshot.global_cost - reference["costs"][i]))
            assignment = np.array([snapshot.assignments[name] for name in names])
            if not np.array_equal(assignment, reference["assignments"][i]):
                raise RuntimeError(f"checkpoint assignment replay mismatch at {i}")
    if error > 1e-8:
        raise RuntimeError(f"checkpoint cost replay mismatch: {error}")
    # exact final mailboxes and retained Q history catch errors hidden by decoding
    actual = Checkpoint.capture(engine, config.prefix_steps, checkpoint.input_sha256)
    expected = Checkpoint.load(case / "fixed_checkpoint.json.gz")
    from .core import json_value

    if json.dumps(asdict(actual), default=json_value, sort_keys=True) != json.dumps(
        asdict(expected), default=json_value, sort_keys=True
    ):
        raise RuntimeError("checkpoint replay final dynamic state differs")
    return error


def run_case(task: tuple) -> dict:
    """Run one immutable fixed-time or best-checkpoint continuation."""
    out, benchmark, seed, mode, config, resume = task
    case = Path(out) / f"{benchmark}_{seed}"
    target = case / f"{mode}_result.json"
    if target.exists():
        if resume:
            saved = json.loads(target.read_text())
            if any(sha(case / p) != h for p, h in saved["files_sha256"].items()):
                raise ValueError(f"saved result evidence changed: {case}")
            return saved
        raise FileExistsError(f"result already exists: {target}; use --resume")
    start = time.perf_counter()
    print(f"START {mode} {benchmark} seed={seed}", flush=True)
    if mode == "fixed":
        info = prepare_prefix(case, benchmark, seed, config)
        replay_error = None
    else:
        fixed_result = json.loads((case / "fixed_result.json").read_text())
        if any(sha(case / p) != h for p, h in fixed_result["files_sha256"].items()):
            raise ValueError("fixed-time evidence changed before best-checkpoint phase")
        info = json.loads((case / "prefix.json").read_text())
        replay_error = validate_restored_prefix(case, config)
    checkpoint = Checkpoint.load(case / f"{mode}_checkpoint.json.gz")
    graph = load_input(case / "input.npz")
    engine = make_engine(graph, config, checkpoint.next_iteration)
    checkpoint.restore(engine)
    snapshots = [
        advance(engine, i)
        for i in range(
            checkpoint.next_iteration, checkpoint.next_iteration + config.post_steps
        )
    ]
    trace = trace_arrays(snapshots, [v.name for v in engine.var_nodes])
    original = load_input(case / "input.npz")
    error = verify_costs(original, trace)
    if len(engine.split_events) != 1 or engine.damping_factor != 0:
        raise RuntimeError("expected exactly one split followed by zero damping")
    np.savez_compressed(case / f"{mode}_trace.npz", **trace)
    print(
        f"MERGE {mode} {benchmark} seed={seed}; B&B cap={config.bb_seconds}s",
        flush=True,
    )
    merge = merge_tail(original, trace, config)
    result = {
        "benchmark": benchmark,
        "seed": seed,
        "mode": mode,
        "input_sha256": info["input_sha256"],
        "split_before_iteration": checkpoint.next_iteration,
        "prefix_search_updates": config.prefix_steps,
        "post_split_updates": config.post_steps,
        "prefix_best_cost": info["prefix_best_cost"],
        "selected_checkpoint_cost": checkpoint.last_cost,
        "final_cost": float(trace["costs"][-1]),
        "anytime_cost": min(info["prefix_best_cost"], float(trace["costs"].min())),
        "normalization_period_before": checkpoint.graph_diameter,
        "normalization_period_after": engine.graph_diameter,
        "split_event": engine.split_events[0],
        "cost_verification_max_error": error,
        "checkpoint_replay_max_error": replay_error,
        "merge": merge,
        "elapsed_seconds": time.perf_counter() - start,
        "files_sha256": {
            name: sha(case / name)
            for name in [
                "input.npz",
                "references.npz",
                "prefix.npz",
                "prefix.json",
                "best_checkpoint.json.gz",
                "fixed_checkpoint.json.gz",
                f"{mode}_trace.npz",
            ]
        },
    }
    write_json(target, result)
    print(
        f"DONE {mode} {benchmark} seed={seed}: tail={merge['tail_kind']}, "
        f"final={result['final_cost']:.6f}, seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    """CLI for separately reviewable pilot and full runs; no implicit large suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=FAMILIES, default=["random_dense"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phase", choices=["fixed", "best", "both"], default="both")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prefix-steps", type=int, default=1000)
    parser.add_argument("--post-steps", type=int, default=1000)
    parser.add_argument("--tail-steps", type=int, default=100)
    parser.add_argument("--bb-seconds", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
    ):
        parser.error("positive workers and distinct nonnegative seeds required")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("duplicate benchmarks")
    config = Config(
        args.prefix_steps, args.post_steps, args.tail_steps, bb_seconds=args.bb_seconds
    )
    prepare(args.out, config, args.benchmarks, args.seeds)
    for mode in ["fixed", "best"] if args.phase == "both" else [args.phase]:
        tasks = [
            (str(args.out), b, s, mode, config, args.resume)
            for b in args.benchmarks
            for s in args.seeds
        ]
        if args.workers == 1:
            for task in tasks:
                run_case(task)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run_case, tasks))
    print("COMPLETE requested phases; no further studies launched", flush=True)


if __name__ == "__main__":
    main()
