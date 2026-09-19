"""Paired one-update interventions immediately inside/outside row boundaries."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np

from . import core, study


def continuation(initial, prefix: dict, edge: int = -1, weight: float = 0.5):
    k = initial.clone()
    if edge >= 0:
        k.weights[edge] = weight
    rows = core.row_state(k)[0]
    k.step()
    first_belief = core.gauge(k.beliefs())
    costs = list(prefix["costs"]) + [k.cost]
    assignments = list(prefix["assignments"]) + [k.assignment.copy()]
    residuals = list(prefix["residuals"]) + [k.message_residual / k.problem.scale]
    defects = list(prefix["defects"]) + [core.fixed_defect(k)]
    k.weights[:] = 0.5
    while k.t < 2000:
        k.step()
        costs.append(k.cost)
        assignments.append(k.assignment.copy())
        residuals.append(k.message_residual / k.problem.scale)
        defects.append(core.fixed_defect(k))
    trace = dict(
        costs=np.array(costs),
        assignments=np.array(assignments, dtype=np.int16),
        residuals=np.array(residuals),
        defects=np.array(defects),
    )
    stable = all(
        (
            np.all(trace["assignments"][-100:] == trace["assignments"][-1]),
            max(residuals[-100:]) < 1e-7,
            max(defects[-100:]) < 1e-7,
        )
    )
    return trace, rows, first_belief, bool(stable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    runtime, problems = study.freeze(
        args.out,
        argparse.Namespace(
            stage="mechanism", seeds=args.seeds, seed_start=18000, out=str(args.out)
        ),
    )
    boundaries_module = importlib.import_module("state_study_frozen.control")
    records, missing = [], []
    maximum_cost_error = 0.0
    for family in study.FAMILIES:
        for seed in range(18000, 18000 + args.seeds):
            p = study.problem_for(runtime, problems, family, seed)
            key = study.save_problem(args.out, p)
            k = runtime.PairwiseKernel(p)
            prefix = {
                name: [] for name in ("costs", "assignments", "residuals", "defects")
            }
            for _ in range(64):
                k.step()
                prefix["costs"].append(k.cost)
                prefix["assignments"].append(k.assignment.copy())
                prefix["residuals"].append(k.message_residual / p.scale)
                prefix["defects"].append(core.fixed_defect(k))
            baseline, base_rows, base_belief, base_stable = continuation(k, prefix)
            np.savez_compressed(
                args.out / "trajectories" / f"{key}_hold.npz", **baseline
            )
            maximum_cost_error = max(
                maximum_cost_error, study.verify_costs(p, baseline)
            )
            q = boundaries_module.prospective_q(k).reshape(-1, 2, 2, p.d)
            crossing = (
                (core.row_state(k, 0.95)[0] != base_rows)
                .reshape(len(p.edges), -1)
                .mean(axis=1)
            )
            examined = 0
            for edge in np.argsort(-crossing, kind="stable"):
                boundaries = boundaries_module.effective_boundaries(
                    p.costs[edge], q[edge]
                )
                upper = boundaries[(boundaries > 0.5 + 1e-8) & (boundaries < 1 - 1e-8)]
                if not len(upper):
                    continue
                boundary = float(upper[0])
                next_bound = float(upper[1]) if len(upper) > 1 else 1.0
                epsilon = min(1e-5, (boundary - 0.5) / 4, (next_bound - boundary) / 4)
                for side, weight in (
                    ("inside", boundary - epsilon),
                    ("outside", boundary + epsilon),
                ):
                    trace, rows, belief, stable = continuation(
                        k, prefix, int(edge), weight
                    )
                    error = study.verify_costs(p, trace)
                    maximum_cost_error = max(maximum_cost_error, error)
                    np.savez_compressed(
                        args.out / "trajectories" / f"{key}_edge{edge}_{side}.npz",
                        **trace,
                    )
                    belief_difference = float(np.max(np.abs(belief - base_belief)))
                    record = dict(
                        family=family,
                        seed=seed,
                        edge=int(edge),
                        side=side,
                        boundary=boundary,
                        weight=weight,
                        epsilon=epsilon,
                        immediate_row_changes=int(np.count_nonzero(rows != base_rows)),
                        immediate_belief_difference=belief_difference / p.scale,
                        immediate_assignment_changes=int(
                            np.count_nonzero(
                                trace["assignments"][64] != baseline["assignments"][64]
                            )
                        ),
                        final_cost=float(trace["costs"][-1]),
                        baseline_cost=float(baseline["costs"][-1]),
                        strict_stable=stable,
                        baseline_strict_stable=base_stable,
                    )
                    records.append(record)
                examined += 1
                if examined == 3:
                    break
            if not examined:
                missing.append(key)
            print(json.dumps({"input": key, "boundary_pairs": examined}), flush=True)
    with (args.out / "comparisons.csv").open("w") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = dict(
        pairs=len(records) // 2,
        no_effective_boundaries=missing,
        inside_max_belief_difference=max(
            (
                r["immediate_belief_difference"]
                for r in records
                if r["side"] == "inside"
            ),
            default=0,
        ),
        outside_changed_rows=sum(
            r["immediate_row_changes"] > 0 for r in records if r["side"] == "outside"
        ),
        maximum_cost_error=maximum_cost_error,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = json.loads((args.out / "manifest.json").read_text())
    manifest["artifact_sha256"] = {
        str(p.relative_to(args.out)): study.sha(p)
        for p in args.out.rglob("*")
        if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
