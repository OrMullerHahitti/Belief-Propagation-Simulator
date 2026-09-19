"""Replay saved control events through the native split-and-damped BP engine."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from . import core, study


def load_runtime(stage: Path):
    package = ModuleType("state_replay_frozen")
    package.__path__ = [str(stage / "source")]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".kernel")


def load_problem(runtime, path: Path, family: str, seed: int):
    with np.load(path) as data:
        return runtime.PairwiseProblem(
            edges=data["edges"],
            costs=data["costs"],
            unary=data["unary"],
            family=family,
            seed=seed,
            topology=family,
            variable_names=tuple(data["variable_names"].tolist()),
            factor_names=tuple(data["factor_names"].tolist()),
            unary_names=tuple(data["unary_names"].tolist()),
        )


def replay(stage: Path, family: str, seed: int, method: str) -> dict:
    """Check every saved assignment/cost and sampled pairwise/unary messages."""
    from propflow import DampingEngine, SplitEngine

    class NativeEngine(DampingEngine, SplitEngine):
        pass

    class ReplaySnapshots:
        def capture_step(self, step_index, step, engine):
            return SimpleNamespace(
                step=step_index,
                global_cost=None,
                assignments={v.name: v.curr_assignment for v in engine.var_nodes},
            )

    key = f"{family}_{seed}"
    paths = {
        "input": stage / "inputs" / f"{key}.npz",
        "events": stage / "events" / f"{key}_{method}.json",
        "trajectory": stage / "trajectories" / f"{key}_{method}.npz",
    }
    runtime = load_runtime(stage)
    p = load_problem(runtime, paths["input"], family, seed)
    k = runtime.PairwiseKernel(p)
    engine = NativeEngine(
        p.to_native(),
        damping_factor=0.9,
        split_factor=0.5,
        anytime=False,
        normalize_messages=True,
        snapshot_manager=ReplaySnapshots(),
    )
    factors = {f.name: f for f in engine.factor_nodes}
    variables = {v.name: v for v in engine.var_nodes}
    events = {e["step"]: e for e in json.loads(paths["events"].read_text())["events"]}
    report = dict(
        family=family,
        seed=seed,
        method=method,
        inputs_sha256={name: study.sha(path) for name, path in paths.items()},
        assignment_mismatch_steps=[],
        kernel_assignment_mismatch_steps=[],
        maximum_original_cost_error=0.0,
        maximum_native_cost_error=0.0,
        maximum_gauged_message_error=0.0,
        message_sample_steps=[],
    )
    with np.load(paths["trajectory"]) as trace:
        report["steps"] = len(trace["costs"])
        for i, (expected_x, expected_cost) in enumerate(
            zip(trace["assignments"], trace["costs"])
        ):
            if i in events:
                event = events[i]
                k.weights[:] = event["weights"]
                k.damping = event["damping"]
                engine.damping_factor = k.damping
                for e, name in enumerate(p.factor_names):
                    factors[name + "'"].cost_table = k.weights[e] * p.costs[e]
                    factors[name + "''"].cost_table = (1 - k.weights[e]) * p.costs[e]
            k.step()
            engine.step(i)
            snapshot = engine.latest_snapshot()
            actual_x = np.array([snapshot.assignments[n] for n in p.variable_names])
            if not np.array_equal(actual_x, expected_x):
                report["assignment_mismatch_steps"].append(i)
            if not np.array_equal(k.assignment, expected_x):
                report["kernel_assignment_mismatch_steps"].append(i)
            for name, error in (
                ("maximum_original_cost_error", abs(p.cost(actual_x) - expected_cost)),
                (
                    "maximum_native_cost_error",
                    abs(snapshot.global_cost - expected_cost),
                ),
            ):
                report[name] = max(report[name], float(error))
            try:
                engine._handle_cycle_events(i)
            except StopIteration:
                pass
            if i in events or i % 128 == 0 or i + 1 == report["steps"]:
                report["message_sample_steps"].append(i)
                comparisons = []
                for e, (u, v) in enumerate(p.edges):
                    for clone, suffix in enumerate(("'", "''")):
                        for axis, endpoint in enumerate((u, v)):
                            comparisons.append(
                                (
                                    variables[p.variable_names[endpoint]],
                                    p.factor_names[e] + suffix,
                                    k.q[2 * e + clone, axis],
                                    k.r[2 * e + clone, axis],
                                )
                            )
                for u, name in enumerate(p.unary_names):
                    for clone, suffix in enumerate(("'", "''")):
                        comparisons.append(
                            (
                                variables[p.variable_names[u]],
                                name + suffix,
                                k.unary_q[u, clone],
                                k.unary_r[u, clone],
                            )
                        )
                for var, name, q, r in comparisons:
                    actual_q = next(
                        m.data for m in var.last_iteration if m.recipient.name == name
                    )
                    actual_r = next(m.data for m in var.inbox if m.sender.name == name)
                    error = max(
                        float(np.max(np.abs(core.gauge(a) - core.gauge(b))))
                        for a, b in ((actual_q, q), (actual_r, r))
                    )
                    report["maximum_gauged_message_error"] = max(
                        report["maximum_gauged_message_error"], error
                    )
    report["passed"] = all(
        (
            not report["assignment_mismatch_steps"],
            not report["kernel_assignment_mismatch_steps"],
            report["maximum_original_cost_error"] < 1e-7,
            report["maximum_native_cost_error"] < 1e-7,
            report["maximum_gauged_message_error"] < 1e-7 * p.scale,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", nargs=3, action="append", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for family, seed, method in args.case:
        report = replay(args.stage, family, int(seed), method)
        reports.append(report)
        study.write_json(args.out / f"{family}_{seed}_{method}.json", report)
        print(json.dumps(report), flush=True)
    study.write_json(
        args.out / "manifest.json",
        {
            "all_passed": all(r["passed"] for r in reports),
            "native_source_sha256": {
                str(p.relative_to(study.ROOT)): study.sha(p)
                for p in (study.ROOT / "src/propflow").rglob("*.py")
            },
            "replay_source_sha256": study.sha(Path(__file__)),
        },
    )
    if not all(r["passed"] for r in reports):
        raise SystemExit("native replay mismatch; see saved reports")


if __name__ == "__main__":
    main()
