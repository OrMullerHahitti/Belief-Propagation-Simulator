"""Resume the domain-20 population study by seed and phase; no B&B execution."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from propflow import DampingEngine, DampingSCFGEngine, MinSumComputator
from experiments.aaai.code.problems import capture_original
from .core import (
    Checkpoint,
    Config,
    TraceSnapshots,
    advance,
    input_fingerprint,
    load_input,
    make_engine,
    save_input,
    tail_kind,
    trace_arrays,
    verify_costs,
    write_json,
)
from .domains import build_dense
from .mgm import mgm_menu_search, observed_menus
from .resume import checked, read_arrays, restore_runtime, save_runtime, seal
from .run import ROOT, sha, source_files, validate_prefix, validate_restored_prefix


def dependencies() -> list[Path]:
    """Freeze simulation dependencies independently of downstream plotting code."""
    return [p for p in source_files() if not p.name.startswith("plot_")]


def prepare(out: Path, seeds: list[int], config: Config, pilot: Path | None) -> None:
    """Validate reusable pilot evidence and freeze the exact execution sources."""
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies()}
    expected = {
        "config": asdict(config),
        "domain_size": 20,
        "seeds": seeds,
        "benchmarks": ["random_dense"],
        "source_sha256": hashes,
    }
    if (out / "manifest.json").exists():
        saved = json.loads((out / "manifest.json").read_text())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "protocol or simulation sources changed; use a new run directory"
            )
        return
    out.mkdir(parents=True, exist_ok=False)
    for path in dependencies():
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    imported = []
    (out / "mgm_tail_values").mkdir()
    if pilot is not None:
        old = json.loads((pilot / "manifest.json").read_text())
        if old["config"] != asdict(config) or old["domain_size"] != 20:
            raise ValueError("pilot protocol differs")
        # algorithm and input generator must match; checkpoint storage is instrumented
        for relative, checksum in old["source_sha256"].items():
            if relative.startswith("src/propflow/") or relative.endswith(
                ("/engines.py", "/problems.py", "/domains.py", "/merge.py")
            ):
                if sha(ROOT / relative) != checksum:
                    raise ValueError(f"pilot dependency differs: {relative}")
        for seed in sorted(set(seeds).intersection(old["seeds"])):
            source_case = pilot / f"random_dense_{seed}"
            for mode in ["fixed", "best"]:
                checked(source_case, f"{mode}_result.json")
                mgm_path = pilot / "mgm_tail_values" / f"seed_{seed}_{mode}_mgm.json"
                mgm = json.loads(mgm_path.read_text())
                if mgm["trace_sha256"] != sha(source_case / f"{mode}_trace.npz") or (
                    mgm["mgm_source_sha256"] != sha(Path(__file__).with_name("mgm.py"))
                ):
                    raise ValueError("pilot MGM evidence or algorithm changed")
                shutil.copyfile(mgm_path, out / "mgm_tail_values" / mgm_path.name)
            shutil.copytree(source_case, out / source_case.name)
            imported.append(seed)
        shutil.copyfile(pilot / "manifest.json", out / "pilot_manifest.json")
    for seed in seeds:
        (out / f"random_dense_{seed}").mkdir(exist_ok=True)
    write_json(
        out / "manifest.json",
        {
            **expected,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "pilot": str(pilot.resolve()) if pilot else None,
            "reused_seeds": imported,
            "agents": 50,
            "density": 0.6,
            "bb_run": False,
            "phase_order": "all fixed BP, all best-checkpoint BP, all tail-value MGM",
            "post_checkpoint_every": 250,
            "mgm": "all per-variable values and all distinct starts in final 100 updates",
            "mean_axis": "1000 observation updates + 1000 continuation updates + MGM rounds",
            "numpy_version": np.__version__,
        },
    )
    write_json(
        out / "reuse_audit.json",
        {
            "imported_seeds": imported,
            "new_seeds": sorted(set(seeds) - set(imported)),
            "pilot_evidence_hashes_verified": True,
            "pilot_algorithm_and_generator_sources_identical": True,
            "pilot_full_post_split_states_available": False,
            "new_full_post_split_states": "every 250 updates, including terminal state",
            "scope": "imported pilot contains legacy merge metadata; use mgm_tail_values only",
        },
    )


def prepare_case(case: Path, seed: int, cfg: Config) -> None:
    """Save each baseline separately and capture prefix states during plain DMS."""
    if not (case / "input.npz").exists():
        save_input(build_dense(seed, 20), case / "input.npz")
    original = load_input(case / "input.npz")
    fingerprint = input_fingerprint(original)
    total = cfg.prefix_steps + cfg.post_steps
    for label, cls, extra in [
        ("DMS", DampingEngine, {}),
        ("DMS_split_0.5", DampingSCFGEngine, {"split_factor": cfg.split}),
    ]:
        if checked(case, f"{label}_complete.json") is not None:
            continue
        print(f"START seed={seed} {label}", flush=True)
        engine = cls(
            deepcopy(original),
            computator=MinSumComputator(),
            damping_factor=cfg.damping,
            normalize_messages=True,
            anytime=False,
            snapshot_manager=TraceSnapshots(),
            **extra,
        )
        snapshots, best, best_cost = [], None, float("inf")
        for i in range(total):
            snapshot = advance(engine, i)
            snapshots.append(snapshot)
            if label == "DMS" and i < cfg.prefix_steps:
                if snapshot.global_cost < best_cost:
                    best_cost = snapshot.global_cost
                    best = Checkpoint.capture(engine, i + 1, fingerprint)
                if i + 1 == cfg.prefix_steps:
                    best.save(case / "best_checkpoint.json.gz")
                    Checkpoint.capture(engine, i + 1, fingerprint).save(
                        case / "fixed_checkpoint.json.gz"
                    )
        trace = trace_arrays(snapshots, [v.name for v in engine.var_nodes])
        error = verify_costs(original, trace)
        np.savez_compressed(case / f"{label}_trace.npz", **trace)
        save_runtime(case / "checkpoints" / label / "terminal", engine, total, trace)
        files = ["input.npz", f"{label}_trace.npz"]
        if label == "DMS":
            prefix = trace_arrays(
                snapshots[: cfg.prefix_steps], [v.name for v in engine.var_nodes]
            )
            np.savez_compressed(case / "prefix.npz", **prefix)
            write_json(
                case / "prefix.json",
                {
                    "input_sha256": fingerprint,
                    "prefix_best_cost": best_cost,
                    "best_checkpoint_index": best.next_iteration - 1,
                    "prefix_final_cost": float(prefix["costs"][-1]),
                    "prefix_steps": cfg.prefix_steps,
                    "normalization_period": engine.graph_diameter,
                    "original_cost_max_error": error,
                    "retained_dms_max_error": 0.0,
                },
            )
            files += [
                "prefix.npz",
                "prefix.json",
                "best_checkpoint.json.gz",
                "fixed_checkpoint.json.gz",
            ]
        seal(
            case,
            f"{label}_complete.json",
            files,
            input_fingerprint=fingerprint,
            original_cost_max_error=error,
        )
        print(f"DONE seed={seed} {label}", flush=True)
    refs = {}
    for label in ["DMS", "DMS_split_0.5"]:
        trace = read_arrays(case / f"{label}_trace.npz")
        refs[label], refs[label + "_iterations"] = trace["costs"], trace["iterations"]
    np.savez_compressed(case / "references.npz", **refs)
    write_json(
        case / "references.json",
        {
            "input_sha256": fingerprint,
            "domain_size": 20,
            "agents": len(original.variables),
            "density": 0.6,
            "origin": "matched native execution on retained original tables",
        },
    )
    validate_prefix(case, read_arrays(case / "prefix.npz"))


def run_bp(task) -> dict:
    """Run a missing continuation or resume its last complete message checkpoint."""
    out, seed, mode, cfg = task
    case = Path(out) / f"random_dense_{seed}"
    saved = checked(case, f"{mode}_result.json")
    if saved is not None:
        print(f"REUSE seed={seed} {mode}", flush=True)
        return saved
    start = time.perf_counter()
    if mode == "fixed":
        prepare_case(case, seed, cfg)
    else:
        if checked(case, "fixed_result.json") is None:
            raise ValueError("fixed phase must finish first")
    info = json.loads((case / "prefix.json").read_text())
    original = load_input(case / "input.npz")
    cp = Checkpoint.load(case / f"{mode}_checkpoint.json.gz")
    replay_error = validate_restored_prefix(case, cfg) if mode == "best" else None
    root = case / "checkpoints" / mode
    available = sorted(root.glob("*/complete.json")) if root.exists() else []
    pieces = []
    if available:
        engine, previous, next_i = restore_runtime(available[-1].parent, original, cfg)
        pieces.append(previous)
        print(f"RESUME seed={seed} {mode} next={next_i}", flush=True)
    else:
        engine = make_engine(deepcopy(original), cfg, cp.next_iteration)
        cp.restore(engine)
        next_i = cp.next_iteration
    snapshots = []
    for i in range(next_i, cp.next_iteration + cfg.post_steps):
        snapshots.append(advance(engine, i))
        count = i + 1 - cp.next_iteration
        if count % 250 == 0 or count == cfg.post_steps:
            part = trace_arrays(snapshots, [v.name for v in engine.var_nodes])
            pieces.append(part)
            trace = {
                k: (
                    part[k]
                    if k == "variable_names"
                    else np.concatenate([p[k] for p in pieces])
                )
                for k in part
            }
            save_runtime(root / f"{count:06d}", engine, i + 1, trace)
            pieces, snapshots = [trace], []
    trace = pieces[-1]
    error = verify_costs(original, trace)
    if any(
        (
            len(trace["costs"]) != cfg.post_steps,
            len(engine.split_events) != 1,
            engine.damping_factor != 0,
        )
    ):
        raise RuntimeError("invalid released-damping continuation")
    np.savez_compressed(case / f"{mode}_trace.npz", **trace)
    result = seal(
        case,
        f"{mode}_result.json",
        [
            "input.npz",
            "references.npz",
            "references.json",
            "DMS_trace.npz",
            "DMS_split_0.5_trace.npz",
            "prefix.npz",
            "prefix.json",
            "best_checkpoint.json.gz",
            "fixed_checkpoint.json.gz",
            f"{mode}_trace.npz",
        ],
        benchmark="random_dense",
        seed=seed,
        mode=mode,
        domain_size=20,
        input_sha256=info["input_sha256"],
        split_before_iteration=cp.next_iteration,
        prefix_search_updates=cfg.prefix_steps,
        post_split_updates=cfg.post_steps,
        prefix_best_cost=info["prefix_best_cost"],
        selected_checkpoint_cost=cp.last_cost,
        final_cost=float(trace["costs"][-1]),
        anytime_cost=min(info["prefix_best_cost"], float(trace["costs"].min())),
        split_event=engine.split_events[0],
        cost_verification_max_error=error,
        checkpoint_replay_max_error=replay_error,
        merge={
            "tail_kind": tail_kind(trace["assignments"], cfg.tail_steps),
            "deferred_to": "mgm_tail_values",
            "bb_run": False,
        },
        elapsed_seconds=time.perf_counter() - start,
    )
    print(
        f"DONE seed={seed} {mode} tail={result['merge']['tail_kind']} seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )
    return result


def run_mgm(task) -> dict:
    """Reuse or compute all-tail-values MGM without any BP execution."""
    out, seed, mode, cfg = task
    case = Path(out) / f"random_dense_{seed}"
    saved = checked(case, f"{mode}_result.json")
    if saved is None:
        raise ValueError("BP evidence missing")
    path = Path(out) / "mgm_tail_values" / f"seed_{seed}_{mode}_mgm.json"
    source_hash = sha(Path(__file__).with_name("mgm.py"))
    if path.exists():
        result = json.loads(path.read_text())
        if any(
            (
                result["trace_sha256"] != sha(case / f"{mode}_trace.npz"),
                result["mgm_source_sha256"] != source_hash,
                result["window"] != cfg.tail_steps,
            )
        ):
            raise ValueError("MGM dependencies differ")
        return result
    original = load_input(case / "input.npz")
    names, axes, tables = capture_original(original)
    trace = read_arrays(case / f"{mode}_trace.npz")
    verify_costs(original, trace)
    menus, indices = observed_menus(trace, cfg.tail_steps)
    starts = []
    for index in indices:
        assignment = dict(
            zip(trace["variable_names"], map(int, trace["assignments"][index]))
        )
        result = mgm_menu_search(
            assignment, menus, names, axes, tables, max_rounds=cfg.mgm_rounds
        )
        if abs(result["costs"][0] - trace["costs"][index]) > 1e-8:
            raise RuntimeError("MGM initial cost mismatch")
        if result["hit_round_cap"]:
            raise RuntimeError("MGM did not finish within its round cap")
        result.update(
            source_trace_index=index, source_iteration=int(trace["iterations"][index])
        )
        starts.append(result)
    best = min(starts, key=lambda r: r["cost"])
    result = dict(
        seed=seed,
        mode=mode,
        window=cfg.tail_steps,
        menus=menus,
        tail_kind=tail_kind(trace["assignments"], cfg.tail_steps),
        starts=starts,
        best=best,
        baseline=float(read_arrays(case / "references.npz")["DMS_split_0.5"][-1]),
        prefix_best=saved["prefix_best_cost"],
        best_bp_seen=saved["anytime_cost"],
        input_sha256=saved["input_sha256"],
        trace_sha256=sha(case / f"{mode}_trace.npz"),
        mgm_source_sha256=source_hash,
        bb_run=False,
    )
    write_json(path, result)
    print(
        f"MGM seed={seed} {mode}: starts={len(starts)} rounds={best['rounds']} cost={best['cost']:.6f}",
        flush=True,
    )
    return result


def main() -> None:
    """Run the approved study, or just its missing BP/MGM phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--phase", choices=["all", "fixed", "best", "mgm"], default="all"
    )
    args = parser.parse_args()
    if any(
        (args.workers < 1, len(set(args.seeds)) != len(args.seeds), min(args.seeds) < 0)
    ):
        parser.error("require positive workers and unique nonnegative seeds")
    cfg = Config()
    prepare(args.out, args.seeds, cfg, args.pilot)
    for phase in ["fixed", "best", "mgm"] if args.phase == "all" else [args.phase]:
        modes = ["fixed", "best"] if phase == "mgm" else [phase]
        tasks = [
            (str(args.out), seed, mode, cfg) for seed in args.seeds for mode in modes
        ]
        worker = run_mgm if phase == "mgm" else run_bp
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"PROGRESS {phase} {count}/{len(tasks)}", flush=True)
    print("COMPLETE requested phases", flush=True)


if __name__ == "__main__":
    main()
