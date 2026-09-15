"""Native, finite-horizon damping interventions on fully specified tiny graphs.

The runner uses the paper's Q-then-R update, periodic normalization and raw
argmin decoder. Diagnostics read snapshots and subtract each message's first
label; this gauge transformation does not change the engine's execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from experiments.other.aaai_derived_control.code.kernel import PairwiseProblem
from propflow import DampingEngine, DampingSCFGEngine
from propflow.snapshots import EngineSnapshot


def fixtures() -> dict[str, PairwiseProblem]:
    """Return explicit inputs; the tiny-unary counterexample fixes RNG seed 3."""
    path = np.array([[0, 1], [1, 2]])
    return {
        "single_edge": PairwiseProblem(
            np.array([[0, 1]]),
            np.array([[[0, 3], [4, 1]]]),
            np.array([[0, 0.6], [0.5, 0]]),
            "damping_not_required",
            0,
            "edge",
        ),
        "robust_path": PairwiseProblem(
            path,
            np.tile([[16, 0], [0, 16]], (2, 1, 1)),
            np.array([[12, 0], [13, 0], [4, 0]]),
            "damping_causality",
            0,
            "path",
        ),
        "tiny_unary_path": PairwiseProblem(
            path,
            np.array([[[5, 9], [8, 5]], [[2, 0], [0, 3]]]),
            np.array([[0, 0.001], [0, 0.002], [0, 0.004]]),
            "damping_causality",
            64,
            "path",
        ),
        "frustrated_triangle": PairwiseProblem(
            np.array([[0, 1], [1, 2], [2, 0]]),
            np.tile([[10, 0], [0, 10]], (3, 1, 1)),
            np.random.default_rng(3).uniform(0, 0.01, (3, 2)),
            "damping_counterexample",
            3,
            "triangle",
        ),
    }


def reference_gauge(values: np.ndarray) -> np.ndarray:
    """Express every vector relative to label zero."""
    return values - values[..., :1]


def undamped_map_defect(snapshot: EngineSnapshot) -> tuple[float, float]:
    """Evaluate one mathematical undamped update on the captured Q/R state.

    The result is the maximum reference-gauge difference for Q and R. Small
    damped step lengths alone cannot make this diagnostic appear converged.
    Ordered ``cost_labels`` preserve the native tensor-axis meaning.
    """
    r_now = {key: reference_gauge(value) for key, value in snapshot.R.items()}
    beliefs = {
        variable: sum(r_now[(factor, variable)] for factor in neighbors)
        for variable, neighbors in snapshot.N_var.items()
    }
    next_q = {
        (variable, factor): beliefs[variable] - r_now[(factor, variable)]
        for variable, factor in snapshot.Q
    }
    next_r = {}
    for factor, ordered in snapshot.cost_labels.items():
        aggregate = snapshot.cost_tables[factor].copy()
        for axis, variable in enumerate(ordered):
            shape = [1] * len(ordered)
            shape[axis] = len(next_q[(variable, factor)])
            aggregate += next_q[(variable, factor)].reshape(shape)
        for axis, variable in enumerate(ordered):
            shape = [1] * len(ordered)
            shape[axis] = len(next_q[(variable, factor)])
            local = aggregate - next_q[(variable, factor)].reshape(shape)
            other_axes = tuple(i for i in range(len(ordered)) if i != axis)
            next_r[(factor, variable)] = (
                local.min(axis=other_axes) if other_axes else local
            )
    defects = []
    for proposed, current in ((next_q, snapshot.Q), (next_r, snapshot.R)):
        defects.append(
            max(
                float(
                    np.max(
                        np.abs(reference_gauge(proposed[key]) - reference_gauge(value))
                    )
                )
                for key, value in current.items()
            )
        )
    return defects[0], defects[1]


@dataclass
class NativeRun:
    """Portable per-update evidence, including every Q/R message and belief."""

    q_keys: tuple[tuple[str, str], ...]
    r_keys: tuple[tuple[str, str], ...]
    variables: tuple[str, ...]
    q: np.ndarray
    r: np.ndarray
    beliefs: np.ndarray
    assignments: np.ndarray
    costs: np.ndarray
    damping: np.ndarray
    undamped_defect: np.ndarray

    @property
    def messages(self) -> np.ndarray:
        """Concatenate gauge Q/R vectors for convergence diagnostics."""
        return np.concatenate((self.q, self.r), axis=1)

    def summary(self, tail: int = 300) -> dict:
        """Describe a finite observed tail without claiming asymptotic proof."""
        tail = min(tail, len(self.costs) - 1)
        messages = self.messages
        step_delta = np.max(np.abs(np.diff(messages, axis=0)), axis=(1, 2))
        flips = np.any(self.assignments[1:] != self.assignments[:-1], axis=1)
        tail_messages = messages[-tail:]
        period = next(
            (
                lag
                for lag in range(1, min(64, tail // 3) + 1)
                if np.max(np.abs(tail_messages[lag:] - tail_messages[:-lag])) < 1e-10
            ),
            None,
        )
        last_change = np.flatnonzero(flips)
        tail_flip_start = len(flips) - tail + 1
        return {
            "updates": len(self.costs),
            "tail_updates": tail,
            "final_assignment": self.assignments[-1].tolist(),
            "final_cost": float(self.costs[-1]),
            "tail_cost_min": float(self.costs[-tail:].min()),
            "tail_cost_max": float(self.costs[-tail:].max()),
            "tail_assignments": np.unique(self.assignments[-tail:], axis=0).tolist(),
            "tail_assignment_switches": int(flips[tail_flip_start:].sum()),
            "last_assignment_switch_update": (
                int(last_change[-1] + 2) if len(last_change) else None
            ),
            "max_tail_message_step_delta": float(step_delta[-tail:].max()),
            "final_message_step_delta": float(step_delta[-1]),
            "max_tail_undamped_map_defect": float(self.undamped_defect[-tail:].max()),
            "final_undamped_map_defect": self.undamped_defect[-1].tolist(),
            "observed_tail_message_period_at_1e-10": period,
            "final_min_belief_margin": float(
                np.min(np.abs(self.beliefs[-1, :, 1] - self.beliefs[-1, :, 0]))
            ),
        }


def run_native(
    problem: PairwiseProblem,
    *,
    split: bool,
    schedule: dict[int, float],
    steps: int = 2000,
) -> NativeRun:
    """Replay native updates; schedule keys count already-completed updates.

    Changing only ``engine.damping_factor`` preserves every current message.
    All scheduled runs start from identical zero inboxes and can be checked
    for exactly equal message prefixes up to their first intervention.
    """
    if problem.d != 2:
        raise ValueError("the diagnostic fixture runner requires binary variables")
    if steps < 2 or any(step < 0 or step >= steps for step in schedule):
        raise ValueError("require at least two updates and interventions in horizon")
    if any(not np.isfinite(value) or not 0 <= value < 1 for value in schedule.values()):
        raise ValueError("old-Q retention must be finite and lie in [0,1)")
    kwargs = dict(
        damping_factor=schedule.get(0, 0.0), normalize_messages=True, anytime=False
    )
    if split:
        # this override is essential: the combined engine defaults to 0.6
        kwargs["split_factor"] = 0.5
    engine = (DampingSCFGEngine if split else DampingEngine)(
        problem.to_native(), **kwargs
    )
    q_rows, r_rows, belief_rows, assignments, costs, damping, defects = (
        [] for _ in range(7)
    )
    q_keys = r_keys = ()
    engine.convergence_monitor.reset()
    for step in range(steps):
        if step in schedule:
            engine.damping_factor = schedule[step]
        engine.step(step)
        snapshot = engine.latest_snapshot()
        if snapshot is None:
            raise RuntimeError("native update did not capture a snapshot")
        if step == 0:
            q_keys, r_keys = tuple(sorted(snapshot.Q)), tuple(sorted(snapshot.R))
        assignment = np.array(
            [snapshot.assignments[name] for name in problem.variable_names]
        )
        cost = problem.cost(assignment)
        if not np.isclose(cost, snapshot.global_cost, rtol=0, atol=1e-10):
            raise AssertionError("split snapshot cost differs from original objective")
        q_rows.append(reference_gauge(np.array([snapshot.Q[key] for key in q_keys])))
        r_rows.append(reference_gauge(np.array([snapshot.R[key] for key in r_keys])))
        belief_rows.append(
            reference_gauge(
                np.array([snapshot.beliefs[name] for name in problem.variable_names])
            )
        )
        assignments.append(assignment)
        costs.append(cost)
        damping.append(snapshot.lambda_)
        defects.append(undamped_map_defect(snapshot))
        try:
            engine._handle_cycle_events(step)
        except StopIteration:
            # match the AAAI full-horizon runner: normalize but ignore early stops
            pass
    return NativeRun(
        q_keys,
        r_keys,
        problem.variable_names,
        np.array(q_rows),
        np.array(r_rows),
        np.array(belief_rows),
        np.array(assignments),
        np.array(costs),
        np.array(damping),
        np.array(defects),
    )


def save_run(path: Path, run: NativeRun) -> None:
    """Write lossless gauge diagnostics and an accessible cost trajectory."""
    np.savez_compressed(
        path.with_suffix(".npz"),
        q_keys=np.array(run.q_keys),
        r_keys=np.array(run.r_keys),
        variables=np.array(run.variables),
        q=run.q,
        r=run.r,
        beliefs=run.beliefs,
        assignments=run.assignments,
        costs=run.costs,
        damping=run.damping,
        undamped_map_defect=run.undamped_defect,
    )
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["update", "cost", "damping", *run.variables])
        for step, (cost, damping, assignment) in enumerate(
            zip(run.costs, run.damping, run.assignments), 1
        ):
            writer.writerow([step, cost, damping, *assignment])


def run_suite(out: Path, steps: int = 2000, counter_steps: int = 20000) -> dict:
    """Run tiny causal comparisons and preserve complete reproducible evidence."""
    out.mkdir(parents=True, exist_ok=True)
    primary_methods = {
        "unsplit_d0": (False, {0: 0.0}),
        "split05_d0": (True, {0: 0.0}),
        "split05_d05": (True, {0: 0.5}),
        "split05_d09": (True, {0: 0.9}),
        "split05_d001": (True, {0: 0.01}),
        "split05_d0016": (True, {0: 0.016}),
        "split05_d002": (True, {0: 0.02}),
        "split05_d0_to_d05_at32": (True, {0: 0.0, 32: 0.5}),
        "split05_d0_to_d09_at32": (True, {0: 0.0, 32: 0.9}),
        "split05_d0_to_d05_at32_to_d0_at256": (True, {0: 0.0, 32: 0.5, 256: 0.0}),
    }
    report = {"fixtures": {}, "runs": {}, "prefix_checks": {}}
    for fixture, problem in fixtures().items():
        report["fixtures"][fixture] = {
            "edges": problem.edges.tolist(),
            "cost_tables": problem.costs.tolist(),
            "unary": problem.unary.tolist(),
            "original_objective": [
                {"assignment": list(x), "cost": problem.cost(np.array(x))}
                for x in product(range(problem.d), repeat=problem.n)
            ],
        }
        methods = primary_methods
        horizon = steps
        if fixture == "single_edge":
            methods = {
                key: primary_methods[key] for key in ("unsplit_d0", "split05_d0")
            }
        elif fixture == "tiny_unary_path":
            methods = dict(list(primary_methods.items())[:4])
        elif fixture == "frustrated_triangle":
            methods = {
                key: primary_methods[key] for key in ("split05_d0", "split05_d09")
            }
            horizon = counter_steps
        runs = {}
        for method, (split, schedule) in methods.items():
            run = run_native(problem, split=split, schedule=schedule, steps=horizon)
            runs[method] = run
            save_run(out / f"{fixture}_{method}", run)
            summary = run.summary()
            summary["schedule_after_completed_updates"] = schedule
            report["runs"][f"{fixture}_{method}"] = summary
            print(f"{fixture}/{method}: {json.dumps(summary)}", flush=True)
        if fixture == "robust_path":
            for method in methods:
                if "at32" in method:
                    for field in ("q", "r", "beliefs", "assignments", "costs"):
                        np.testing.assert_array_equal(
                            getattr(runs[method], field)[:32],
                            getattr(runs["split05_d0"], field)[:32],
                        )
                    report["prefix_checks"][
                        method
                    ] = "all Q/R, beliefs, assignments and costs identical through update32"
            resumed = runs["split05_d0_to_d05_at32_to_d0_at256"]
            damped = runs["split05_d0_to_d05_at32"]
            np.testing.assert_array_equal(resumed.messages[:256], damped.messages[:256])
            report["prefix_checks"][
                "remove_damping_after256"
            ] = "all Q/R identical through update256"
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    root = Path(__file__).resolve().parents[4]
    source_paths = [
        "experiments/other/damping_causality/code/native_experiments.py",
        "experiments/other/aaai_derived_control/code/kernel.py",
        "src/propflow/bp/engines.py",
        "src/propflow/bp/engine_base.py",
        "src/propflow/bp/computators.py",
        "src/propflow/policies/splitting.py",
        "src/propflow/policies/damping.py",
        "src/propflow/core/agents.py",
        "src/propflow/core/components.py",
        "src/propflow/snapshots/builder.py",
        "src/propflow/snapshots/types.py",
    ]
    provenance = json.dumps(
        {
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in source_paths
            },
            "native_engines": ["DampingEngine", "DampingSCFGEngine"],
            "split_factor": 0.5,
            "message_gauge": "subtract reference label zero; diagnostics only",
            "update_order": "Q, R, snapshot, scheduled native cycle normalization",
            "initialization": "native zero inboxes, no external unary warm start",
            "convergence_stops": "ignored, matching AAAI full horizon",
            "cost": "original tables plus original unary; actual decoded assignment",
            "undamped_map_defect": (
                "one undamped mathematical Q/R update from snapshot R "
                "versus current snapshot Q/R"
            ),
            "outputs_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.glob("*"))
                if path.suffix in (".csv", ".npz")
            },
        },
        indent=2,
    )
    (out / "provenance.json").write_text(f"{provenance}\n")
    return report


def append_control(
    out: Path,
    fixture: str,
    method: str,
    *,
    split: bool,
    schedule: dict[int, float],
    steps: int = 2000,
) -> dict:
    """Append a named control without replaying or overwriting frozen runs."""
    method = f"{fixture}_{method}"
    if (out / f"{method}.npz").exists():
        raise FileExistsError(f"a frozen run already exists for {method}")
    report = json.loads((out / "summary.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    problem = fixtures()[fixture]
    run = run_native(problem, split=split, schedule=schedule, steps=steps)
    save_run(out / method, run)
    summary = run.summary()
    summary["schedule_after_completed_updates"] = schedule
    report["runs"][method] = summary
    report["fixtures"].setdefault(
        fixture,
        {
            "edges": problem.edges.tolist(),
            "cost_tables": problem.costs.tolist(),
            "unary": problem.unary.tolist(),
            "original_objective": [
                {"assignment": list(x), "cost": problem.cost(np.array(x))}
                for x in product(range(problem.d), repeat=problem.n)
            ],
        },
    )
    provenance.setdefault("appended_runs", {})[method] = {
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "updates": steps,
    }
    for suffix in (".npz", ".csv"):
        path = (out / method).with_suffix(suffix)
        provenance["outputs_sha256"][path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return summary


def append_robust_damping(out: Path, damping: float, steps: int = 2000) -> dict:
    """Append a new damping control on the robust path fixture."""
    return append_control(
        out,
        "robust_path",
        f"split05_d{str(damping).replace('.', '')}",
        split=True,
        schedule={0: damping},
        steps=steps,
    )


def main() -> None:
    """Run the frozen causal fixture suite from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("results/damping_causality_20260915/native")
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--counter-steps", type=int, default=20000)
    parser.add_argument("--append-robust-damping", type=float)
    args = parser.parse_args()
    if args.append_robust_damping is None:
        run_suite(args.out, args.steps, args.counter_steps)
    else:
        print(
            json.dumps(
                append_robust_damping(args.out, args.append_robust_damping, args.steps),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
