"""build one result folder for the AAMAS Section 6 numbers.

Copies the post-fix results in data/ and, on random_dense and random_sparse,
replaces the lines rerun with float cost tables (MS, DMS and every
DMS_split_at_K; the K grid was completed on 2026-09-28) by the rows in
data_float_tables_20260923/. DMS_split_pulse also starts on the unsplit
integer tables of those two benchmarks and was not rerun, so it is left out
rather than mixed in.
dabp_timing.csv comes from data_cuda/ (the RTX 4090 timing run used by
plot_final.py). Nothing outside the new folder is written.

Dry run by default; pass --write to create the folder.

Example:
  uv run python experiments/aaai/code/build_paper_data.py
  uv run python experiments/aaai/code/build_paper_data.py --write
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data"
RERUN = ROOT / "data_float_tables_20260923"
TIMING = ROOT / "data_cuda" / "dabp_timing.csv"
BENCHMARKS = [
    "random_dense",
    "random_sparse",
    "scale_free",
    "graph_coloring",
    "meeting_scheduling",
]
FLOAT_BENCHMARKS = {"random_dense", "random_sparse"}
RERUN_LINES = {"MS", "DMS"} | {
    f"DMS_split_at_{k}" for k in (50, 100, 300, 500, 1000, 1500)
}
# lines that run on the unsplit integer tables for part or all of the run and were not rerun
DROPPED_LINES = {"DMS_split_pulse"}
# meeting scheduling: every line except DABP was rerun with the current code on 2026-09-28
# (the June 28 lines in data/ cannot be reproduced by the current generator); the two DABP
# lines are kept from data/ and named in the metadata as old
MEETING_RERUN = ROOT / "data_meeting_rerun_20260928"
# Optimal is the centralized branch-and-bound value of the instances, independent of the BP code
MEETING_OLD_LINES = {"Attentive", "Attentive_NoSplit", "Optimal"}


def rerun_dir(b: str) -> Path | None:
    if b in FLOAT_BENCHMARKS:
        return RERUN
    if b == "meeting_scheduling":
        return MEETING_RERUN
    return None


def plan_benchmark(b: str) -> dict:
    final = pd.read_csv(BASE / f"{b}_final_costs.csv")
    lines = sorted(final.algorithm.unique())
    src = rerun_dir(b)
    if src is None:
        return {"benchmark": b, "copied": lines, "replaced": [], "dropped": []}
    new_final = pd.read_csv(src / f"{b}_final_costs.csv")
    counts = new_final.groupby("algorithm").seed.nunique()
    assert (counts == 50).all(), f"{b}: rerun incomplete"
    if b == "meeting_scheduling":
        replaced = sorted(set(counts.index) & set(lines))
        copied = sorted(set(lines) - set(replaced))
        assert set(copied) <= MEETING_OLD_LINES, f"{b}: not rerun: {copied}"
        return {"benchmark": b, "copied": copied, "replaced": replaced, "dropped": []}
    # the rerun folder holds only rerun lines, each with all 50 seeds (DMS_split_at_1500 is dense only)
    assert set(counts.index) <= RERUN_LINES, f"{b}: unexpected rerun lines"
    replaced = sorted(set(counts.index) & set(lines))
    assert not (RERUN_LINES & set(lines)) - set(replaced), (
        f"{b}: a rerun line is missing from the rerun folder"
    )
    dropped = sorted(DROPPED_LINES & set(lines))
    copied = sorted(set(lines) - RERUN_LINES - DROPPED_LINES)
    return {"benchmark": b, "copied": copied, "replaced": replaced, "dropped": dropped}


def write_benchmark(b: str, out: Path, plan: dict) -> None:
    keep = set(plan["copied"])
    src = rerun_dir(b)
    for kind in ("final_costs", "raw_costs"):
        old = pd.read_csv(BASE / f"{b}_{kind}.csv")
        parts = [old[old.algorithm.isin(keep)]]
        if plan["replaced"]:
            new = pd.read_csv(src / f"{b}_{kind}.csv")
            parts.append(new[new.algorithm.isin(plan["replaced"])])
        merged = pd.concat(parts, ignore_index=True)
        merged.to_csv(out / f"{b}_{kind}.csv", index=False)
        print(f"  wrote {b}_{kind}.csv: {len(merged):,} rows")
    meta = json.loads((BASE / f"{b}_metadata.json").read_text())
    meta["algorithms"] = sorted(keep | set(plan["replaced"]))
    meta["paper_folder"] = {
        "source": "data/",
        f"replaced_from_{src.name}" if src else "replaced_from": plan["replaced"],
        "dropped_integer_table_lines_not_rerun": plan["dropped"],
    }
    if b == "meeting_scheduling":
        meta["paper_folder"]["old_june_lines_kept"] = sorted(keep)
    (out / f"{b}_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "data_paper_20260923"))
    parser.add_argument(
        "--write", action="store_true", help="create the folder (default: dry run)"
    )
    args = parser.parse_args()
    out = Path(args.out_dir)

    plans = [plan_benchmark(b) for b in BENCHMARKS]
    print(f"target folder: {out} ({'WRITE' if args.write else 'dry run'})")
    for p in plans:
        print(f"\n{p['benchmark']}: copy {len(p['copied'])} lines unchanged")
        if p["replaced"]:
            print(f"  replace with float-table rerun: {', '.join(p['replaced'])}")
        if p["dropped"]:
            print(f"  leave out (integer tables, not rerun): {', '.join(p['dropped'])}")
    print(f"\ndabp_timing.csv from {TIMING.relative_to(ROOT)}")
    if not args.write:
        return
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    out.mkdir(parents=True)
    for p in plans:
        write_benchmark(p["benchmark"], out, p)
    shutil.copy2(TIMING, out / "dabp_timing.csv")
    print("copied dabp_timing.csv")


if __name__ == "__main__":
    main()
