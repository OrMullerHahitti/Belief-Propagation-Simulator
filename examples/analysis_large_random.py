"""Large random snapshot analysis exporting CSV summaries only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyzer.reporting import AnalysisReport, SnapshotAnalyzer, parse_snapshots


def generate_random_snapshot(num_vars: int = 200, degree: int = 3, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    messages = []
    assignments = {}
    factors = [f"f{i}" for i in range(num_vars)]

    for idx in range(num_vars):
        var = f"x{idx}"
        assignments[var] = int(rng.integers(0, 2))
        attached = set()
        for offset in range(degree):
            factor = factors[(idx + offset) % num_vars]
            attached.add(factor)
            values_q = rng.random(2)
            messages.append(
                {
                    "flow": "variable_to_factor",
                    "sender": var,
                    "recipient": factor,
                    "values": values_q.tolist(),
                }
            )
        for factor in attached:
            values_r = rng.random(2)
            messages.append(
                {
                    "flow": "factor_to_variable",
                    "sender": factor,
                    "recipient": var,
                    "values": values_r.tolist(),
                }
            )

    snapshot = {
        "step": 0,
        "messages": messages,
        "assignments": assignments,
        "cost": None,
        "neutral_messages": 0,
        "step_neutral": False,
    }
    return [snapshot]


def main() -> None:
    raw = generate_random_snapshot()
    records = parse_snapshots(raw)
    analyzer = SnapshotAnalyzer(records, max_cycle_len=4)
    report = AnalysisReport(analyzer)

    out_dir = Path("results/analysis_random")
    out_dir.mkdir(parents=True, exist_ok=True)

    report.to_csv(out_dir, step_idx=0)
    summary = report.to_json(0, include_cover=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Random analysis written to", out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
