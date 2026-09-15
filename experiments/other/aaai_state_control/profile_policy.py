"""Separate policy-entry time from complete instrumented validation replay time."""

from __future__ import annotations

import argparse
import cProfile
import importlib
from pathlib import Path
import pstats

import numpy as np

from . import native_replay, study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    runtime = native_replay.load_runtime(args.stage)
    runner = importlib.import_module("state_replay_frozen.study")
    policy_entries = {
        "observe",
        "apply",
        "choose",
        "update",
        "action_edges",
        "apply_arm",
        "state_trigger",
    }
    result = []
    for family in ("k4_d10", "random_sparse"):
        p = native_replay.load_problem(
            runtime, args.stage / "inputs" / f"{family}_18200.npz", family, 18200
        )
        for method in ("baseline", "pulse", "online", "scratch_online"):
            for repetition in range(2):
                profiler = cProfile.Profile()
                output = profiler.runcall(
                    runner.run_case,
                    p,
                    runtime,
                    method,
                    2000,
                    args.stage / "source" / "input_model.npz",
                )
                with np.load(
                    args.stage / "trajectories" / f"{family}_18200_{method}.npz"
                ) as expected:
                    np.testing.assert_array_equal(
                        output[1]["assignments"], expected["assignments"]
                    )
                    np.testing.assert_allclose(
                        output[1]["costs"], expected["costs"], rtol=0, atol=1e-7
                    )
                statistics = pstats.Stats(profiler)
                entries = {
                    name: value[3]
                    for (file, line, name), value in statistics.stats.items()
                    if Path(file).name == "core.py" and name in policy_entries
                }
                result.append(
                    dict(
                        family=family,
                        method=method,
                        repetition=repetition,
                        total_instrumented_seconds=statistics.total_tt,
                        policy_entry_seconds=sum(entries.values()),
                        policy_breakdown_seconds=entries,
                    )
                )
    study.write_json(
        args.out / "profile.json",
        {
            "note": (
                "cProfile instrumentation affects timings; observations used only "
                "for diagnostics are included for fixed methods. This is harness "
                "accounting, not a production latency benchmark."
            ),
            "profile_source_sha256": study.sha(Path(__file__)),
            "records": result,
        },
    )


if __name__ == "__main__":
    main()
