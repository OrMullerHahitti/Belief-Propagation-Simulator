"""Output writers for the chain diagnostics package."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .diagonal_analyzer import analyze_trace_diagonals, summarize_diagonal_diagnostics
from .oscillation_detector import classification_details
from .trace_recorder import write_jsonl


def write_outputs(
    run_results: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    route_analysis: dict[str, Any],
    tolerance: float,
) -> dict[str, Path]:
    """Write traces, summary files, and markdown condition report."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    for result in run_results:
        run_name = result["run_name"]
        artifact_stem = result.get("artifact_stem")
        trace_name = f"{artifact_stem}_trace.jsonl" if artifact_stem else "trace.jsonl"
        snapshots_name = (
            f"{artifact_stem}_snapshots.jsonl" if artifact_stem else "snapshots.jsonl"
        )
        summary_name = (
            f"{artifact_stem}_summary.json" if artifact_stem else "summary.json"
        )
        plots_dir = (
            f"{run_name}/plots/{artifact_stem}"
            if artifact_stem
            else f"{run_name}/plots"
        )
        run_dir = out / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        paths[f"{run_name}/{trace_name}"] = write_jsonl(
            result["trace"], run_dir / trace_name
        )
        paths[f"{run_name}/{snapshots_name}"] = _write_jsonl(
            _snapshots_from_result(result), run_dir / snapshots_name
        )
        details = classification_details(result["trace"], tolerance=tolerance)
        diag_trace = analyze_trace_diagonals(result["trace"])
        diag_summary = summarize_diagonal_diagnostics(
            diag_trace, details.get("tail_start")
        )
        summary = {
            "run_name": run_name,
            "artifact_stem": artifact_stem,
            "experiment_dir": run_name,
            "trace_path": f"{run_name}/{trace_name}",
            "snapshots_path": f"{run_name}/{snapshots_name}",
            "plots_dir": plots_dir,
            **details,
            **result.get("summary", {}),
            "opposite_diagonals_seen": diag_summary.get("opposite_pairs_seen", False),
            "diagonal_summary": diag_summary,
        }
        summary = _jsonable(summary)
        summaries.append(summary)
        paths[f"{run_name}/{summary_name}"] = _write_json(
            summary, run_dir / summary_name
        )

    paths["summary.json"] = _write_json(
        {"runs": summaries, "route_analysis": route_analysis}, out / "summary.json"
    )
    paths["summary.csv"] = _write_summary_csv(summaries, out / "summary.csv")
    paths["condition_report.md"] = _write_condition_report(
        summaries, route_analysis, out / "condition_report.md"
    )
    return paths


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return path


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    return path


def _snapshots_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    engine = result.get("engine")
    if engine is None or not hasattr(engine, "snapshots"):
        return []
    snapshots_attr = getattr(engine, "snapshots")
    snapshots = snapshots_attr() if callable(snapshots_attr) else snapshots_attr
    return [_snapshot_to_dict(snapshot) for snapshot in snapshots]


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "step": snapshot.step,
        "lambda": snapshot.lambda_,
        "dom": snapshot.dom,
        "N_var": snapshot.N_var,
        "N_fac": snapshot.N_fac,
        "Q": _message_dict(snapshot.Q),
        "R": _message_dict(snapshot.R),
        "unary": snapshot.unary,
        "beliefs": snapshot.beliefs,
        "assignments": snapshot.assignments,
        "global_cost": snapshot.global_cost,
        "metadata": snapshot.metadata,
        "cost_tables": snapshot.cost_tables,
        "cost_labels": snapshot.cost_labels,
        "winners": snapshot.winners,
        "min_idx": snapshot.min_idx,
        "bct_metadata": snapshot.bct_metadata,
        "captured_at": snapshot.captured_at,
    }


def _message_dict(messages: dict[Any, Any]) -> dict[str, Any]:
    return {f"{src}->{dst}": values for (src, dst), values in sorted(messages.items())}


def _write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> Path:
    cols = [
        "run_name",
        "artifact_stem",
        "experiment_dir",
        "trace_path",
        "snapshots_path",
        "plots_dir",
        "random_experiment_family",
        "random_seed",
        "random_num_variable_nodes",
        "random_domain_size",
        "random_density",
        "random_num_factor_nodes",
        "random_graph_fingerprint",
        "classification",
        "period",
        "tail_start",
        "assignment_converged",
        "belief_converged",
        "even_odd_oscillation",
        "immediate_even_odd_oscillation",
        "opposite_diagonals_seen",
        "split_at_iter",
        "transfer_mode",
        "split_fraction",
        "split_percentage",
        "damping_factor",
        "stop_on_convergence",
        "first_forced_convergence_step",
        "oscillation_returned_after_apparent_convergence",
        "measured_mechanism",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in summaries:
            writer.writerow({col: row.get(col) for col in cols})
    return path


def _write_condition_report(
    summaries: list[dict[str, Any]], route_analysis: dict[str, Any], path: Path
) -> Path:
    standard = next((row for row in summaries if row["run_name"] == "standard"), {})
    lines = [
        "# Symmetric Split Chain Diagnostic Report",
        "",
        "This report summarizes empirical diagnostics for the tested configuration. "
        "It reports candidate conditions and diagnostic signatures; it does not prove a closed-form theorem.",
        "",
        "## Symmetric Chain Standard Min-Sum",
        f"- Classification: `{standard.get('classification', 'not_run')}`",
        f"- Detected period: `{standard.get('period')}`",
        f"- Detected tail/t0: `{standard.get('tail_start')}`",
        f"- Immediate even/odd oscillation: `{standard.get('immediate_even_odd_oscillation')}`",
        f"- Opposite selected diagonals observed: `{standard.get('opposite_diagonals_seen')}`",
    ]

    lines.extend(["", "## Damping"])
    damping_rows = [
        row for row in summaries if row["run_name"].startswith("damping_lambda")
    ]
    if damping_rows:
        for row in damping_rows:
            lines.append(
                f"- `{row['run_name']}`: classification=`{row.get('classification')}`, "
                f"returned_to_oscillation=`{row.get('oscillation_returned_after_apparent_convergence')}`, "
                f"mechanism=`{row.get('measured_mechanism')}`"
            )
    else:
        lines.append("- No damping runs were executed.")

    lines.extend(["", "## Random Graph Split Experiments"])
    random_rows = [
        row for row in summaries if row["run_name"].startswith("random_graph")
    ]
    if random_rows:
        for row in random_rows:
            lines.append(
                f"- `{row['run_name']}`: classification=`{row.get('classification')}`, "
                f"period=`{row.get('period')}`, tail/t0=`{row.get('tail_start')}`, "
                f"family=`{row.get('random_experiment_family')}`, "
                f"split_at=`{row.get('split_at_iter')}`, "
                f"split_pct=`{row.get('split_percentage')}`, "
                f"mode=`{row.get('transfer_mode')}`"
            )
    else:
        lines.append("- No random graph runs were executed.")

    lines.extend(
        [
            "",
            "## Cost-Table-Only Route Analysis",
            f"- Status: `{route_analysis.get('status')}`",
            f"- Warning: {route_analysis.get('warning', '')}",
            f"- Binary inequality summary: {route_analysis.get('binary_inequality_summary', '')}",
            "",
            "## Candidate Diagnostic Signatures",
            "- Treat opposite selected diagonals as an observed diagnostic signature only when the trace reports them.",
            "- Treat damping behavior as empirical for this configuration; no guarantee is claimed.",
            "- For the symmetric split chain, exact periodic route classification is skipped unless a single-cycle graph is supplied.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
