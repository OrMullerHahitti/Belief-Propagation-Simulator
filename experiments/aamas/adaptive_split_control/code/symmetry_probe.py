"""Check when genuine split perturbations can affect the combined beliefs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lab import Action, PairwiseLab, make_problem
from .run import FAMILIES, save_csv


def selectors(lab):
    """Return the active sender value for each clone, direction, and output label."""
    left = (lab.tables + lab.q[:, 1, None, :]).argmin(axis=2)
    right = (lab.tables + lab.q[:, 0, :, None]).argmin(axis=1)
    return np.stack((left, right), axis=1)


def probe(problem, checkpoint, delta):
    baseline = PairwiseLab(problem, 0.5, 0.5)
    baseline.advance(checkpoint)
    trial = baseline.clone()
    edge = int(np.argmax(baseline.residual))
    trial.act(Action(edge, 0.5 + delta, 0.5))
    first_active_change, first_belief_change = None, None
    same_region_error = 0.0
    clone_change = 0.0
    for step in range(checkpoint, 128):
        baseline.step()
        trial.step()
        different = not np.array_equal(selectors(baseline), selectors(trial))
        if different and first_active_change is None:
            first_active_change = step
        difference = float(np.max(np.abs(baseline.beliefs() - trial.beliefs())))
        if difference > 1e-8 and first_belief_change is None:
            first_belief_change = step
        if first_active_change is None:
            same_region_error = max(same_region_error, difference)
            clone_change = max(
                clone_change, float(np.max(np.abs(baseline.r - trial.r)))
            )
    if same_region_error > 1e-8:
        raise AssertionError("combined beliefs changed without an active-region change")
    if first_belief_change is not None:
        assert first_active_change is not None
        assert first_active_change <= first_belief_change
    return {
        "family": problem.family,
        "seed": problem.seed,
        "checkpoint": checkpoint,
        "delta": delta,
        "edge": edge,
        "first_active_change": first_active_change,
        "first_belief_change": first_belief_change,
        "same_region_max_error": same_region_error,
        "same_region_clone_change": clone_change,
        "final_cost_change": trial.cost - baseline.cost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        probe(make_problem("bowtie", family, seed), checkpoint, delta)
        for family in FAMILIES
        for seed in range(24)
        for checkpoint in (0, 8, 24)
        for delta in (1e-6, 0.01, 0.45)
    ]
    save_csv(args.output / "symmetry_probe.csv", rows)
    summary = []
    for checkpoint in (0, 8, 24):
        for delta in (1e-6, 0.01, 0.45):
            subset = [
                r for r in rows if r["checkpoint"] == checkpoint and r["delta"] == delta
            ]
            summary.append(
                {
                    "checkpoint": checkpoint,
                    "delta": delta,
                    "n": len(subset),
                    "active_choices_changed": sum(
                        r["first_active_change"] is not None for r in subset
                    ),
                    "beliefs_changed": sum(
                        r["first_belief_change"] is not None for r in subset
                    ),
                    "final_cost_changed": sum(
                        abs(r["final_cost_change"]) > 1e-8 for r in subset
                    ),
                }
            )
    result = {
        "runs": len(rows),
        "maximum_same_region_belief_error": max(
            r["same_region_max_error"] for r in rows
        ),
        "summary": summary,
    }
    (args.output / "symmetry_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
