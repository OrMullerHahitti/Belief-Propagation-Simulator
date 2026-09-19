"""Small development runs for split-boundary control; confirmation is explicit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .control import choose_split
from .kernel import PairwiseKernel, make_small_problem


def centered(array: np.ndarray) -> np.ndarray:
    """Remove the reference-label constant for residual comparison."""
    return array - array[..., :1]


def run_one(problem, method: str, horizon: int, lookahead: int, budget: int):
    """Run actual message updates, retaining independent objective evidence."""
    valid_modes = {"baseline", "boundary", "grid", "random"}
    if method.removesuffix("_guarded") not in valid_modes and not method.startswith(
        "fixed_"
    ):
        raise ValueError(f"unknown experiment method: {method}")
    damping = 0.9
    weight = 0.5
    if method.startswith("fixed_"):
        _, weight_text, damping_text = method.split("_")
        weight, damping = float(weight_text), float(damping_text)
    kernel = PairwiseKernel(problem, weights=weight, damping=damping)
    rows = []
    traces = []
    events = []
    assignments = []
    residuals = []
    auxiliary_steps = 0
    started = time.perf_counter()
    for i in range(horizon):
        mode = method.removesuffix("_guarded")
        if mode in {"boundary", "grid", "random"} and i in (64, 128, 256, 512):
            action, evidence = choose_split(
                kernel,
                mode,
                budget,
                lookahead,
                problem.seed * 1000 + i,
                guarded=method.endswith("_guarded"),
            )
            if action is not None:
                kernel.weights[action.edge] = action.weight
            auxiliary_steps += evidence["simulated_steps"]
            events.append({"step": i, **evidence})
        names = ("q", "r", "unary_q", "unary_r")
        old_messages = [centered(getattr(kernel, name)) for name in names]
        kernel.step()
        residual = max(
            float(np.max(np.abs(centered(getattr(kernel, name)) - old)))
            for name, old in zip(names, old_messages)
        )
        residuals.append(residual)
        assignments.append(kernel.assignment.copy())
        reconstructed = problem.cost(kernel.assignment)
        if abs(reconstructed - kernel.cost) > 1e-9:
            raise AssertionError("terminal cost disagrees with original objective")
        traces.append(float(reconstructed))
        if i + 1 in (128, 512, horizon):
            tail = min(100, i + 1)
            rows.append(
                {
                    "topology": problem.topology,
                    "family": problem.family,
                    "seed": problem.seed,
                    "method": method,
                    "horizon": i + 1,
                    "cost": float(reconstructed),
                    "best_cost": float(min(traces)),
                    "assignment_stable": bool(
                        np.all(np.array(assignments[-tail:]) == assignments[-1])
                    ),
                    "message_stable": bool(
                        max(residuals[-tail:]) < 1e-7 * problem.scale
                    ),
                    "residual": max(residuals[-tail:]) / problem.scale,
                    "auxiliary_steps": auxiliary_steps,
                    "seconds": time.perf_counter() - started,
                }
            )
    return rows, {
        "costs": traces,
        "assignments": [x.tolist() for x in assignments],
        "weights": kernel.weights.tolist(),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--topologies", nargs="+", default=["k4", "bowtie"])
    parser.add_argument("--families", nargs="+", default=["random", "frustrated"])
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "boundary", "grid", "random"]
    )
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--lookahead", type=int, default=64)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon < 1:
        parser.error("seeds and horizon must be positive")
    if args.seed_start + args.seeds > 6000 and not args.allow_confirmation:
        parser.error("confirmation requires --allow-confirmation after settings freeze")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    manifest = {
        "arguments": vars(args),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.glob("*.py")
        },
    }
    (out / "source").mkdir()
    for path in source.glob("*.py"):
        (out / "source" / path.name).write_bytes(path.read_bytes())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inputs = {}
    records = []
    with (
        (out / "traces.jsonl").open("w") as trace_file,
        (out / "metrics.csv").open("w") as metric_file,
    ):
        writer = None
        for topology, family, seed in itertools.product(
            args.topologies,
            args.families,
            range(args.seed_start, args.seed_start + args.seeds),
        ):
            problem = make_small_problem(topology, family, seed)
            key = f"{topology}_{family}_{seed}"
            inputs[key + "_edges"] = problem.edges
            inputs[key + "_costs"] = problem.costs
            inputs[key + "_unary"] = problem.unary
            for method in args.methods:
                rows, trace = run_one(
                    problem, method, args.horizon, args.lookahead, args.budget
                )
                if writer is None:
                    writer = csv.DictWriter(metric_file, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                metric_file.flush()
                trace_file.write(
                    json.dumps({"key": key, "method": method, **trace}) + "\n"
                )
                trace_file.flush()
                records.extend(rows)
                print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(out / "inputs.npz", **inputs)
    summary = []
    for topology, family, method in itertools.product(
        args.topologies, args.families, args.methods
    ):
        expected = dict(
            topology=topology, family=family, method=method, horizon=args.horizon
        )
        subset = [
            r
            for r in records
            if all(r[key] == value for key, value in expected.items())
        ]
        summary.append(
            {
                "topology": topology,
                "family": family,
                "method": method,
                "n": len(subset),
                "cost": float(np.mean([r["cost"] for r in subset])),
                "stable_assignments": sum(r["assignment_stable"] for r in subset),
                "stable_messages": sum(r["message_stable"] for r in subset),
            }
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
