"""Command-line entry point for Analyzer reporting workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import AnalysisReport, SnapshotAnalyzer, load_snapshots, parse_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PropFlow snapshot analysis helper")
    parser.add_argument("--snapshots", required=True, help="Path to snapshot JSON produced by EngineSnapshotRecorder")
    parser.add_argument("--out", default="results", help="Directory for generated artefacts")
    parser.add_argument("--step", type=int, default=0, help="Snapshot index to analyse")
    parser.add_argument("--max-cycle-len", type=int, default=12, help="Maximum simple cycle length to enumerate")
    parser.add_argument("--compute-jac", action="store_true", help="Persist the dense Jacobian to CSV")
    parser.add_argument("--cover", action="store_true", help="Export neutral cover information")
    parser.add_argument("--plot", action="store_true", help="Generate plots for belief trajectories and dependency graph")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    raw = load_snapshots(args.snapshots)
    records = parse_snapshots(raw)
    analyzer = SnapshotAnalyzer(records, max_cycle_len=args.max_cycle_len)
    report = AnalysisReport(analyzer)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = report.to_json(args.step, include_cover=args.cover)
    with (out_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report.to_csv(out_dir, step_idx=args.step)
    if args.plot:
        report.plots(out_dir, step_idx=args.step, include_graph=args.cover)

    if args.cover:
        cover, residual = analyzer.scc_greedy_neutral_cover(args.step, alpha={})
        cover_path = out_dir / "neutral_cover.json"
        with cover_path.open("w", encoding="utf-8") as handle:
            json.dump({"cover": cover, "residual_nodes": residual.number_of_nodes()}, handle, indent=2)

    if args.compute_jac:
        matrix = analyzer.jacobian(args.step)
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        np.savetxt(out_dir / "jacobian.csv", dense, delimiter=",")


if __name__ == "__main__":  # pragma: no cover
    main()
