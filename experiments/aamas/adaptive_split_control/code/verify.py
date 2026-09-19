"""Independently reconstruct saved objective scores and replay selected policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .confirm import load_model, schedule
from .lab import Problem, native_parity
from .run import run_policy


def verify_partition(directory, pilot, confirmation):
    traces = json.loads((directory / "test_traces.json").read_text())
    with (directory / "test_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values_checked = 0
    max_error = 0.0
    with np.load(directory / "inputs.npz") as inputs:
        for row in rows:
            identity = f"{row['topology']}_{row['family']}_{row['seed']}"
            key = identity if confirmation else "test_" + identity
            edges, tables = inputs[key + "_edges"], inputs[key + "_costs"]
            trace = traces[identity + "_" + row["method"]]
            assignments = np.array(trace["assignments"])
            recomputed = np.zeros(len(assignments))
            for edge, table in zip(edges, tables):
                recomputed += table[assignments[:, edge[0]], assignments[:, edge[1]]]
            error = float(np.max(np.abs(recomputed - trace["cost"])))
            max_error = max(max_error, error)
            np.testing.assert_allclose(recomputed, trace["cost"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                recomputed[-1], float(row["final_cost"]), atol=1e-10
            )
            initial = float(tables[:, 0, 0].sum())
            np.testing.assert_allclose(
                min(initial, recomputed.min()), float(row["best_cost"]), atol=1e-10
            )
            stable_assignments = bool(np.all(assignments[-16:] == assignments[-1]))
            scale = max(float(np.ptp(tables, axis=(1, 2)).sum()), 1e-12)
            stable_messages = max(trace["qr_residual"][-16:]) / scale < 1e-7
            assert stable_assignments == (row["assignment_stable"] == "True")
            assert stable_messages == (row["message_stable"] == "True")
            values_checked += len(assignments)
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, expected in manifest["source_sha256"].items():
        path = Path(__file__).parent / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    replayed = 0
    if confirmation:
        model = load_model(pilot / "offline_model.npz")
        with np.load(directory / "inputs.npz") as inputs:
            for topology in ("bowtie", "k4"):
                for family in ("random", "frustrated"):
                    seed = 2007
                    identity = f"{topology}_{family}_{seed}"
                    p = Problem(
                        inputs[identity + "_edges"],
                        inputs[identity + "_costs"],
                        family,
                        seed,
                        topology,
                    )
                    for method in ("tuned_fixed", "tuned_schedule", "settled_online"):
                        if method == "tuned_fixed":
                            lab = run_policy(
                                p, "fixed", fixed_config=manifest["chosen_fixed"]
                            )[0]
                        elif method == "tuned_schedule":
                            config = manifest["selected_schedule"]
                            lab = schedule(p, config["switch"], config["damping"])
                        else:
                            config = manifest["selected_settle"]
                            lab = run_policy(
                                p,
                                "online",
                                model,
                                rng_seed=seed + 5000,
                                settle_after=config["onset"],
                                settle_damping=config["damping"],
                            )[0]
                        np.testing.assert_array_equal(
                            lab.costs, traces[identity + "_" + method]["cost"]
                        )
                        replayed += 1
                    for split in (None, 0.5, 0.95):
                        for damping in (0, 0.9):
                            native_parity(p, split, damping)
    return {
        "trajectories": len(rows),
        "cost_values_checked": values_checked,
        "maximum_cost_error": max_error,
        "policies_replayed": replayed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "pilot": verify_partition(args.pilot, args.pilot, False),
        "confirmation": verify_partition(args.pilot / "confirmation", args.pilot, True),
    }
    (args.pilot / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
