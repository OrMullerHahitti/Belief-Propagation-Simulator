"""Save causal feedback ablations and a damping scan on the exact small path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.other.aaai_derived_control.code.kernel import PairwiseKernel
from experiments.other.damping_causality.code.feedback import (
    FeedbackState,
    active_jacobian,
    advance,
    belief_differences,
)
from experiments.other.damping_causality.code.native_experiments import fixtures


def main() -> None:
    """Run small prescribed interventions without changing original objectives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    problem = fixtures()["robust_path"]
    np.savez(
        out / "input.npz", edges=problem.edges, costs=problem.costs, unary=problem.unary
    )
    summaries = []
    ablations = [(1, 0), (2, 0), (1, 1), (2, 1), (2, -1)]
    cases = [
        (f"g{g}_s{s}_from{start}", start, 0.0, g, s, None)
        for start in (0, 32)
        for g, s in ablations
    ]
    cases += [
        (f"damping_{damping:g}", 0, damping, 2, 1, None)
        for damping in (
            0,
            0.001,
            0.005,
            0.01,
            0.015,
            0.016,
            0.02,
            0.05,
            0.1,
            0.5,
            0.9,
            0.99,
        )
    ]
    cases += [
        (f"remove_sibling_{duration}_steps", 32, 0.0, 2, 0, 32 + duration)
        for duration in (1, 2, 4, 8)
    ]
    for name, intervention, damping, external, sibling, restore in cases:
        state = FeedbackState.zeros(problem)
        states, rows = [], []
        for t in range(2000):
            g, s = (2, 1) if t < intervention else (external, sibling)
            if restore is not None and t >= restore:
                g, s = 2, 1
            state = advance(problem, state, damping, g, s)
            beliefs = belief_differences(problem, state.r)
            assignment = (beliefs < 0).astype(int)
            defect = np.max(np.abs(advance(problem, state, 0, g, s).q - state.q))
            states.append(state.q.copy())
            rows.append(
                [
                    state.step,
                    problem.cost(assignment),
                    *assignment,
                    *beliefs,
                    defect,
                    *state.q.ravel(),
                ]
            )
        states = np.asarray(states)
        rows = np.asarray(rows)
        np.savetxt(
            out / f"{name}.csv",
            rows,
            delimiter=",",
            header="update,cost,x1,x2,x3,b1,b2,b3,undamped_defect,q01,q10,q12,q21",
            comments="",
        )
        final_g, final_s = (2, 1) if restore is not None else (external, sibling)
        jacobian, margin = active_jacobian(problem, state.q, damping, final_g, final_s)
        period = next(
            (
                period
                for period in range(1, 33)
                if np.max(abs(states[-100:] - states[:-period][-100:])) < 1e-8
            ),
            None,
        )
        summaries.append(
            {
                "case": name,
                "damping": damping,
                "external_gain": external,
                "sibling_gain": sibling,
                "intervention_after": intervention,
                "restore_after": restore,
                "final_assignment": "".join(str(int(x)) for x in rows[-1, 2:5]),
                "tail_cost_min": float(rows[-100:, 1].min()),
                "tail_cost_max": float(rows[-100:, 1].max()),
                "period_up_to_32": period,
                "tail_q_lag1": float(np.max(abs(states[-100:] - states[-101:-1]))),
                "tail_q_lag2": float(np.max(abs(states[-100:] - states[-102:-2]))),
                "tail_undamped_defect": float(rows[-100:, 8].max()),
                "min_final_selector_margin": margin,
                "selected_branch_spectral_radius": float(
                    np.max(abs(np.linalg.eigvals(jacobian)))
                ),
            }
        )
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    parity = {}
    for damping in (0, 0.01, 0.02, 0.5, 0.9):
        state = FeedbackState.zeros(problem)
        native_kernel = PairwiseKernel(problem, damping=damping)
        error = 0.0
        for _ in range(2000):
            state = advance(problem, state, damping)
            native_kernel.step()
            for reduced, actual in (
                (state.q, native_kernel.q),
                (state.r, native_kernel.r),
            ):
                target = 2 * (actual[::2, :, 1] - actual[::2, :, 0])
                error = max(error, float(np.max(abs(reduced - target))))
        if error > 1e-8:
            raise AssertionError(f"full-scale equal-clone parity failure: {error}")
        parity[str(damping)] = error
    (out / "equal_clone_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    (out / "source").mkdir()
    here = Path(__file__).parent
    for source in (
        Path(__file__),
        here / "feedback.py",
        here / "native_experiments.py",
    ):
        (out / "source" / source.name).write_bytes(source.read_bytes())
    hashes = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    manifest = json.dumps(
        {
            "description": "selected exact small fixture; causal examples, not prevalence sampling",
            "q_units": "twice the per-clone Q difference in the equally split run",
            "parity_note": "direct native parity is separately saved by native_experiments.py",
            "sha256": hashes,
        },
        indent=2,
    )
    (out / "manifest.json").write_text(manifest + "\n")
    print(json.dumps({"cases": len(summaries), "parity": parity, "output": str(out)}))


if __name__ == "__main__":
    main()
