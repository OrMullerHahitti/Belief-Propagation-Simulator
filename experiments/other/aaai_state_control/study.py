"""Reproducible staged execution for the state-control research protocol."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np

from . import core


ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
PRIOR = ROOT / "results/aaai_derived_control_20260915/paper_confirmation/source"
FAMILIES = ("k4_d10", "bowtie_frustrated", "random_sparse", "random_dense")
RULES = ("state_cross10", "state_commit50", "state_plateau", "state_restore")
LEARNED = (
    "frozen",
    "frozen_explore",
    "online",
    "scratch_online",
    "scratch_frozen_explore",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def freeze(out: Path, args: argparse.Namespace):
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source / path.name)
    for name in ("kernel.py", "problems.py", "control.py", "intervals.py"):
        shutil.copyfile(PRIOR / name, source / name)
    shutil.copyfile(Path(__file__).with_name("PROTOCOL.md"), out / "PROTOCOL.md")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "source_sha256": {p.name: sha(p) for p in source.glob("*.py")},
        "protocol_sha256": sha(out / "PROTOCOL.md"),
        "native_source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (ROOT / "src/propflow").rglob("*.py")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    package = types.ModuleType("state_study_frozen")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    runtime = importlib.import_module("state_study_frozen.kernel")
    problems = importlib.import_module("state_study_frozen.problems")
    for name in ("inputs", "trajectories", "events", "observations"):
        (out / name).mkdir()
    return runtime, problems


def problem_for(runtime, problems, family: str, seed: int):
    if family == "k4_d10":
        problem = runtime.make_problem("k4", seed, d=10)
    elif family == "bowtie_frustrated":
        problem = runtime.make_small_problem("bowtie", "frustrated", seed)
    else:
        graph = getattr(problems, "build_" + family)(seed)
        problem = runtime.extract_problem(graph, family, seed)
    problem.family = family
    return problem


def save_problem(out: Path, p) -> str:
    key = f"{p.family}_{p.seed}"
    path = out / "inputs" / f"{key}.npz"
    np.savez_compressed(
        path,
        edges=p.edges,
        costs=p.costs,
        unary=p.unary,
        variable_names=p.variable_names,
        factor_names=p.factor_names,
        unary_names=p.unary_names,
    )
    return key


def advance_block(kernel, arm: int, steps: int = 256):
    edges = core.action_edges(kernel, arm)
    costs, assignments, defects = [], [], []
    initial = kernel.cost
    prior_rows = None
    for elapsed in range(steps):
        if elapsed == steps - 8:
            prior_rows = core.row_state(kernel)[0]
        core.apply_arm(kernel, arm, elapsed, edges)
        kernel.step()
        costs.append(float(kernel.cost))
        assignments.append(kernel.assignment.copy())
        defects.append(core.fixed_defect(kernel))
    target = core.block_target(
        initial, costs, assignments, defects, kernel.problem.scale
    )
    return target, costs, assignments, prior_rows


def run_case(p, runtime, method: str, horizon: int, model_path: Path | None):
    k = runtime.PairwiseKernel(p)
    schedule = {s.name: s for s in core.schedules()}.get(method)
    if schedule is None and method not in (*RULES, *LEARNED):
        raise ValueError(f"unknown method {method}")
    model = None
    if method in LEARNED:
        model = (
            core.LinearSelector()
            if method.startswith("scratch_")
            else core.LinearSelector.load(model_path)
        )
    rng = np.random.default_rng(p.seed + 909031)
    draws = [(float(rng.random()), int(rng.integers(len(core.ARMS)))) for _ in range(6)]
    costs, assignments, residuals, defects = [], [], [], []
    events, observations, decisions = [], [], []
    previous_rows = None
    last_weights, last_damping = None, None
    trigger, stop = None, None
    arm, edges, block_start, feature_at_choice = 0, np.empty(0, dtype=int), None, None
    block_initial_cost = None
    start = time.perf_counter()
    records = []
    for i in range(horizon):
        observation_end = 1568 if method in LEARNED else 512
        observe_now = i < observation_end and i % 8 == 0
        if observe_now:
            feature, previous_rows, extra = core.observe(
                k, previous_rows, costs[-8:], assignments[-8:]
            )
            observations.append({"step": i, "feature": feature.tolist(), **extra})
        if schedule is not None:
            schedule.apply(k)
        elif method in RULES:
            if trigger is None and 16 <= i <= 256 and i % 8 == 0:
                if core.state_trigger(method, feature):
                    trigger, stop = i, i + 192
            if method == "state_restore" and trigger is not None and stop is not None:
                if trigger + 32 <= i < stop and i % 8 == 0:
                    if feature[1] >= 0.9 and feature[2] <= 0.01:
                        stop = i
            k.weights[:] = 0.95 if trigger is not None and trigger <= i < stop else 0.5
            k.damping = 0.9
        else:
            if i in range(32, 1568, 256):
                index = (i - 32) // 256
                arm, predictions = model.choose(
                    feature, method != "frozen", *draws[index]
                )
                edges = core.action_edges(k, arm)
                block_start, feature_at_choice = i, feature.copy()
                block_initial_cost = float(k.cost)
                decisions.append(
                    {
                        "step": i,
                        "arm": core.ARMS[arm],
                        "features": feature.tolist(),
                        "predictions": predictions,
                        "exploration_draw": draws[index][0],
                        "selected_edges": edges.tolist(),
                    }
                )
            if block_start is not None and i < block_start + 256:
                core.apply_arm(k, arm, i - block_start, edges)
            else:
                k.weights[:] = 0.5
                k.damping = 0.9
        changed = any(
            (
                last_weights is None,
                not np.array_equal(k.weights, last_weights),
                k.damping != last_damping,
            )
        )
        if changed:
            change = 0.0
            if last_weights is not None:
                prior = k.clone()
                prior.weights, prior.damping = last_weights, last_damping
                change = float(
                    np.mean(core.row_state(prior)[0] != core.row_state(k)[0])
                )
            events.append(
                {
                    "step": i,
                    "weights": k.weights.tolist(),
                    "damping": float(k.damping),
                    "predicted_row_change": change,
                }
            )
            last_weights, last_damping = k.weights.copy(), k.damping
        k.step()
        costs.append(float(k.cost))
        assignments.append(k.assignment.copy())
        residuals.append(k.message_residual / p.scale)
        defects.append(core.fixed_defect(k))
        if block_start is not None and i + 1 == block_start + 256:
            target = core.block_target(
                block_initial_cost,
                costs[-256:],
                assignments[-256:],
                defects[-256:],
                p.scale,
            )
            decisions[-1]["observed_target"] = target.tolist()
            if method in ("online", "scratch_online"):
                model.update(arm, feature_at_choice, target)
                decisions[-1]["online_update"] = True
        if i + 1 in (2000, horizon):
            tail = min(100, i + 1)
            astable = bool(np.all(np.array(assignments[-tail:]) == assignments[-1]))
            rstable, dstable = (
                max(residuals[-tail:]) < 1e-7,
                max(defects[-tail:]) < 1e-7,
            )
            records.append(
                {
                    "family": p.family,
                    "seed": p.seed,
                    "method": method,
                    "horizon": i + 1,
                    "terminal_cost": costs[-1],
                    "best_cost": min(costs),
                    "assignment_stable": astable,
                    "strict_stable": bool(astable and rstable and dstable),
                    "message_residual": max(residuals[-tail:]),
                    "undamped_defect": max(defects[-tail:]),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
    trace = dict(
        costs=np.array(costs),
        assignments=np.array(assignments, dtype=np.int16),
        residuals=np.array(residuals),
        defects=np.array(defects),
    )
    return records, trace, events, observations, decisions, model


def verify_costs(p, trace: dict) -> float:
    x = trace["assignments"].astype(int)
    values = p.unary[np.arange(p.n)[None, :], x].sum(axis=1)
    for e, (u, v) in enumerate(p.edges):
        values += p.costs[e, x[:, u], x[:, v]]
    error = float(np.max(np.abs(values - trace["costs"])))
    if error > 1e-7:
        raise AssertionError(f"original-cost mismatch {error}")
    return error


def summarize(out: Path, records: list[dict]) -> None:
    summary = []
    paired = []
    rng = np.random.default_rng(210915)
    for family in sorted({r["family"] for r in records}):
        for horizon in sorted({r["horizon"] for r in records}):
            subset = [
                r for r in records if (r["family"], r["horizon"]) == (family, horizon)
            ]
            methods = sorted({r["method"] for r in subset})
            grouped = {
                m: {r["seed"]: r for r in subset if r["method"] == m} for m in methods
            }
            for m, rows in grouped.items():
                summary.append(
                    {
                        "family": family,
                        "horizon": horizon,
                        "method": m,
                        "n": len(rows),
                        "mean_cost": float(
                            np.mean([r["terminal_cost"] for r in rows.values()])
                        ),
                        "assignment_stable": sum(
                            r["assignment_stable"] for r in rows.values()
                        ),
                        "strict_stable": sum(r["strict_stable"] for r in rows.values()),
                    }
                )
                for control in (
                    "baseline",
                    "pulse",
                    "frozen_explore",
                    "scratch_frozen_explore",
                ):
                    if control not in grouped or control == m:
                        continue
                    seed_list = sorted(rows)
                    control_rows = grouped[control]
                    delta = np.array(
                        [
                            rows[s]["terminal_cost"] - control_rows[s]["terminal_cost"]
                            for s in seed_list
                        ]
                    )
                    boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(
                        axis=1
                    )
                    regressions = [
                        s
                        for s in seed_list
                        if control_rows[s]["strict_stable"]
                        if not rows[s]["strict_stable"]
                    ]
                    paired.append(
                        {
                            "family": family,
                            "horizon": horizon,
                            "method": m,
                            "control": control,
                            "mean_change": float(delta.mean()),
                            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                            "wins": int((delta < -1e-7).sum()),
                            "ties": int((abs(delta) <= 1e-7).sum()),
                            "losses": int((delta > 1e-7).sum()),
                            "stability_regressions": regressions,
                        }
                    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")


def evaluate(args, out, runtime, problems):
    methods = args.methods or (
        [s.name for s in core.schedules()]
        if args.stage == "ablate"
        else ["baseline", "pulse", *RULES, *LEARNED]
    )
    records, maximum_error, count = [], 0.0, 0
    with (out / "metrics.csv").open("w") as stream:
        writer = None
        for family in args.families:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                p = problem_for(runtime, problems, family, seed)
                key = save_problem(out, p)
                for method in methods:
                    rows, trace, events, observations, decisions, model = run_case(
                        p, runtime, method, args.horizon, args.model
                    )
                    error = verify_costs(p, trace)
                    maximum_error = max(maximum_error, error)
                    count += len(trace["costs"])
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    prefix = key + "_" + method
                    np.savez_compressed(out / "trajectories" / f"{prefix}.npz", **trace)
                    write_json(
                        out / "events" / f"{prefix}.json",
                        {"events": events, "decisions": decisions},
                    )
                    (out / "observations" / f"{prefix}.json").write_text(
                        json.dumps(observations) + "\n"
                    )
                    if model is not None:
                        model.save(out / "events" / f"{prefix}_final_model.npz")
                    records.extend(rows)
                    print(json.dumps(rows[-1]), flush=True)
    summarize(out, records)
    write_json(
        out / "verification.json",
        {"reconstructed_costs": count, "maximum_error": maximum_error},
    )


def train(args, out, runtime, problems):
    features, targets, arms, keys, times = [], [], [], [], []
    for family in args.families:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            p = problem_for(runtime, problems, family, seed)
            key = save_problem(out, p)
            kernel = runtime.PairwiseKernel(p)
            previous_rows, recent_costs, recent_assignments = None, [], []
            for i in range(32):
                if i == 24:
                    previous_rows = core.row_state(kernel)[0]
                kernel.step()
                recent_costs.append(float(kernel.cost))
                recent_assignments.append(kernel.assignment.copy())
            rng = np.random.default_rng(seed + 411)
            block_costs, block_assignments = [], []
            for decision in range(6):
                feature, previous_rows, _ = core.observe(
                    kernel, previous_rows, recent_costs[-8:], recent_assignments[-8:]
                )
                continuations = []
                for arm in range(len(core.ARMS)):
                    trial = kernel.clone()
                    target, costs, assignments, prior_rows = advance_block(trial, arm)
                    verify_costs(
                        p,
                        {
                            "costs": np.array(costs),
                            "assignments": np.array(assignments),
                        },
                    )
                    block_costs.append(costs)
                    block_assignments.append(assignments)
                    features.append(feature.copy())
                    targets.append(target)
                    arms.append(arm)
                    keys.append(key)
                    times.append(kernel.t)
                    continuations.append((trial, costs, assignments, prior_rows))
                next_arm = 0 if seed % 2 == 0 else int(rng.integers(len(core.ARMS)))
                kernel, recent_costs, recent_assignments, previous_rows = continuations[
                    next_arm
                ]
            np.savez_compressed(
                out / "trajectories" / f"{key}_training.npz",
                costs=np.array(block_costs),
                assignments=np.array(block_assignments, dtype=np.int16),
            )
            print(json.dumps({"training_input": key, "outcomes": 30}), flush=True)
    dataset = dict(
        features=np.array(features),
        targets=np.array(targets),
        arms=np.array(arms),
        keys=np.array(keys),
        times=np.array(times),
    )
    np.savez_compressed(out / "training_data.npz", **dataset)
    model = core.LinearSelector()
    for arm, feature, target in zip(arms, features, targets):
        model.update(arm, feature, target)
    model.save(out / "offline_model.npz")
    write_json(
        out / "training_summary.json",
        {
            "cases": len(set(keys)),
            "counterfactual_blocks": len(arms),
            "training_solver_updates": len(arms) * 256 + len(set(keys)) * 32,
            "prediction_weights": 90,
            "features": core.FEATURES,
            "arms": core.ARMS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ablate", "train", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon < 2000:
        parser.error("positive seed count and at least2000 updates are required")
    if args.seed_start + args.seeds > 19000 and not args.confirmation:
        parser.error(
            "reserved confirmation requires --confirmation after settings freeze"
        )
    original_args = vars(args).copy()
    for key in ("out", "model"):
        original_args[key] = (
            str(original_args[key]) if original_args[key] is not None else None
        )
    runtime, problems = freeze(args.out, argparse.Namespace(**original_args))
    if args.model is not None:
        shutil.copyfile(args.model, args.out / "source" / "input_model.npz")
        args.model = args.out / "source" / "input_model.npz"
    if args.stage == "train":
        train(args, args.out, runtime, problems)
    else:
        evaluate(args, args.out, runtime, problems)
    manifest = json.loads((args.out / "manifest.json").read_text())
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["artifact_sha256"] = {
        str(p.relative_to(args.out)): sha(p)
        for p in args.out.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
