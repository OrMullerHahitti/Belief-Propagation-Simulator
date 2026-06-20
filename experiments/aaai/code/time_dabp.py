"""Measure DABP's per-iteration wall-clock cost relative to plain DMS.

DABP runs a full graph-attention forward pass (and a periodic backward +
optimizer step) per BP iteration, so it is far more expensive per iteration
than min-sum. This script times one instance (seed 0) of each benchmark for
both engines and writes the per-iteration ratio to ``data/dabp_timing.csv``.

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

from problems import BENCHMARKS
from run_experiments import make_engine

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


def time_benchmark(benchmark: str, seed: int) -> dict:
    dms = _time_steps(
        make_engine("DMS", BENCHMARKS[benchmark](seed), seed), N_WARMUP, N_TIMED
    )
    try:
        dabp = _time_steps(
            make_engine("Attentive", BENCHMARKS[benchmark](seed), seed),
            N_WARMUP,
            N_TIMED,
        )
        ratio = dabp / dms if dms > 0 else float("nan")
    except Exception as exc:  # noqa: BLE001
        print(f"  {benchmark}: DABP timing failed ({exc!r}); ratio left blank", flush=True)
        dabp, ratio = float("nan"), float("nan")
    return {
        "benchmark": benchmark,
        "dms_s_per_iter": dms,
        "dabp_s_per_iter": dabp,
        "ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    args = parser.parse_args()

    benchmarks = list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    unknown = set(benchmarks) - set(BENCHMARKS)
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)}")

    rows = []
    for benchmark in benchmarks:
        print(f"timing {benchmark} (seed {args.seed}) ...", flush=True)
        row = time_benchmark(benchmark, args.seed)
        rows.append(row)
        print(
            f"  DMS {row['dms_s_per_iter'] * 1e3:.2f} ms/iter, "
            f"DABP {row['dabp_s_per_iter'] * 1e3:.2f} ms/iter, "
            f"ratio {row['ratio']:.2f}",
            flush=True,
        )

    out_path = Path(args.out_dir) / "dabp_timing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "dms_s_per_iter", "dabp_s_per_iter", "ratio"])
        for row in rows:
            writer.writerow(
                [
                    row["benchmark"],
                    f"{row['dms_s_per_iter']:.6f}",
                    f"{row['dabp_s_per_iter']:.6f}",
                    f"{row['ratio']:.4f}",
                ]
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
