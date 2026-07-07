"""Measure DABP's per-iteration wall-clock cost relative to plain DMS.

DABP runs a full graph-attention forward pass (and a periodic backward +
optimizer step) per BP iteration, so it is far more expensive per iteration
than min-sum. This script times one instance (seed 0) of each benchmark for
both engines for DABP-supported benchmarks and writes the per-iteration ratio
to ``data/dabp_timing.csv``.

The plotter (``plot_results.py``) reads that ratio and stretches the DABP curve
onto a wall-clock-equivalent x-axis: DABP iteration ``k`` is drawn at
``x = ratio * k`` (so a 1:3 ratio makes each DABP step jump three iterations).

Timing excludes a short warm-up so DABP's one-time model build / message-state
restart is not charged to the per-iteration figure, while a full training
window (optimizer step every ``update_interval`` iterations) is included.

Example:
  uv run python experiments/aaai/code/time_dabp.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csv_backups import backup_existing_csvs
from problems import BENCHMARKS
from run_experiments import RANDOM_TERNARY_BENCHMARK, make_engine

N_WARMUP = 5
N_TIMED = 40


def _time_steps(engine, n_warmup: int, n_timed: int) -> float:
    """per-iteration wall-clock of engine.step (+ cycle events), after warm-up."""
    engine.convergence_monitor.reset()
    i = 0
    for _ in range(n_warmup):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        i += 1
    start = time.perf_counter()
    for _ in range(n_timed):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            pass
        i += 1
    return (time.perf_counter() - start) / n_timed


def _time_dabp_variant(
    label: str, benchmark: str, seed: int, dms: float
) -> tuple[float, float]:
    """per-iteration seconds and DABP/DMS ratio for one DABP engine label."""
    try:
        secs = _time_steps(
            make_engine(label, BENCHMARKS[benchmark](seed), seed), N_WARMUP, N_TIMED
        )
        ratio = secs / dms if dms > 0 else float("nan")
    except Exception as exc:  # noqa: BLE001
        print(
            f"  {benchmark}: {label} timing failed ({exc!r}); ratio left blank",
            flush=True,
        )
        secs, ratio = float("nan"), float("nan")
    return secs, ratio


def time_benchmark(benchmark: str, seed: int) -> dict:
    dms = _time_steps(
        make_engine("DMS", BENCHMARKS[benchmark](seed), seed), N_WARMUP, N_TIMED
    )
    dabp, ratio = _time_dabp_variant("Attentive", benchmark, seed, dms)
    nosplit, nosplit_ratio = _time_dabp_variant(
        "Attentive_NoSplit", benchmark, seed, dms
    )
    symsplit, symsplit_ratio = _time_dabp_variant(
        "Attentive_SymSplit", benchmark, seed, dms
    )
    return {
        "benchmark": benchmark,
        "dms_s_per_iter": dms,
        "dabp_s_per_iter": dabp,
        "ratio": ratio,
        "dabp_nosplit_s_per_iter": nosplit,
        "nosplit_ratio": nosplit_ratio,
        "dabp_symsplit_s_per_iter": symsplit,
        "symsplit_ratio": symsplit_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="do not copy existing CSVs to experiments/aaai/backups before writing",
    )
    args = parser.parse_args()

    benchmarks = list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    unknown = set(benchmarks) - set(BENCHMARKS)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")

    unsupported = [b for b in benchmarks if b == RANDOM_TERNARY_BENCHMARK]
    for benchmark in unsupported:
        print(
            f"skipping {benchmark}: DABP supports only unary/binary factors",
            flush=True,
        )
    benchmarks = [b for b in benchmarks if b not in unsupported]
    if not benchmarks:
        print("no DABP-supported benchmarks requested; nothing to time", flush=True)
        return

    rows = []
    for benchmark in benchmarks:
        print(f"timing {benchmark} (seed {args.seed}) ...", flush=True)
        row = time_benchmark(benchmark, args.seed)
        rows.append(row)
        print(
            f"  DMS {row['dms_s_per_iter'] * 1e3:.2f} ms/iter, "
            f"DABP {row['dabp_s_per_iter'] * 1e3:.2f} ms/iter (ratio {row['ratio']:.2f}), "
            f"DABP-no-split {row['dabp_nosplit_s_per_iter'] * 1e3:.2f} ms/iter "
            f"(ratio {row['nosplit_ratio']:.2f}), "
            f"DABP-sym-split {row['dabp_symsplit_s_per_iter'] * 1e3:.2f} ms/iter "
            f"(ratio {row['symsplit_ratio']:.2f})",
            flush=True,
        )

    out_path = Path(args.out_dir) / "dabp_timing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_backup:
        backup_dir = backup_existing_csvs(out_path.parent, label="data_before_timing")
        if backup_dir is not None:
            print(f"BACKUP existing CSVs -> {backup_dir}", flush=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark",
                "dms_s_per_iter",
                "dabp_s_per_iter",
                "ratio",
                "dabp_nosplit_s_per_iter",
                "nosplit_ratio",
                "dabp_symsplit_s_per_iter",
                "symsplit_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["benchmark"],
                    f"{row['dms_s_per_iter']:.6f}",
                    f"{row['dabp_s_per_iter']:.6f}",
                    f"{row['ratio']:.4f}",
                    f"{row['dabp_nosplit_s_per_iter']:.6f}",
                    f"{row['nosplit_ratio']:.4f}",
                    f"{row['dabp_symsplit_s_per_iter']:.6f}",
                    f"{row['symsplit_ratio']:.4f}",
                ]
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
