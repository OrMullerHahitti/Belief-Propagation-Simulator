"""Separate cost-over-iteration comparisons with split labels on the x-axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402
from .core import load_input, verify_costs  # noqa: E402
from .run import sha  # noqa: E402


def plot_comparison(run_dir: Path) -> None:
    """Validate saved evidence and export one native-iteration line plot per seed."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    output = run_dir / "plots"
    output.mkdir(exist_ok=True)
    report = [
        "# Dense late-split comparison",
        "",
        "Costs use the original objective; lower is better. Each plot uses the",
        "native iteration axis. The best-state method observes all 1,000 prefix",
        "updates before restoring its checkpoint; that discarded search suffix",
        "is not drawn on this trajectory axis. It still counts as search work.",
        "MGM/B&B follow the plotted BP continuation and are not part of its line.",
        "",
        "| Seed | Mode | Split after update | Prefix best | Final BP | Tail | MGM | B&B | B&B complete |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark in manifest["benchmarks"]:
        for seed in manifest["seeds"]:
            case = run_dir / f"{benchmark}_{seed}"
            original = load_input(case / "input.npz")
            with np.load(case / "prefix.npz", allow_pickle=False) as f:
                prefix = dict(f)
            with np.load(case / "references.npz", allow_pickle=False) as f:
                refs = dict(f)
            verify_costs(original, prefix)
            for label in ["DMS", "DMS_split_0.5"]:
                path = case / f"{label}_trace.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as f:
                        trace = dict(f)
                    verify_costs(original, trace)
                    if not np.array_equal(trace["costs"], refs[label]):
                        raise ValueError("baseline trace differs from reference line")
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(
                refs["DMS_split_0.5_iterations"] + 1,
                refs["DMS_split_0.5"],
                color="#484848",
                lw=1.35,
                label="Standard split + damping",
            )
            split_points = {}
            for mode, color, style, label in [
                ("fixed", "#2563A6", "-", "Late split at 1,000"),
                ("best", "#BE6B21", "--", "Late split from best state"),
            ]:
                result = json.loads((case / f"{mode}_result.json").read_text())
                for name, checksum in result["files_sha256"].items():
                    if sha(case / name) != checksum:
                        raise ValueError(f"evidence changed: {case / name}")
                with np.load(case / f"{mode}_trace.npz", allow_pickle=False) as f:
                    trace = dict(f)
                verify_costs(original, trace)
                split = result["split_before_iteration"]
                split_points[mode] = split
                mask = prefix["iterations"] < split
                x = (
                    np.concatenate([prefix["iterations"][mask], trace["iterations"]])
                    + 1
                )
                y = np.concatenate([prefix["costs"][mask], trace["costs"]])
                if not np.all(np.diff(x) == 1):
                    raise ValueError("nonconsecutive native iteration axis")
                if mode == "fixed":
                    label = f"Late split at {split:,}"
                ax.plot(x, y, color=color, ls=style, lw=1.35, label=label)
                merge = result["merge"]
                mgm = "—" if merge["mgm"] is None else f"{merge['mgm']['cost']:,.3f}"
                bb = "—" if merge["bb"] is None else f"{merge['bb']['cost']:,.3f}"
                complete = "—" if merge["bb"] is None else str(merge["bb"]["complete"])
                report.append(
                    f"| {seed} | {mode} | {split} | {result['prefix_best_cost']:,.3f} | "
                    f"{result['final_cost']:,.3f} | {merge['tail_kind']} | {mgm} | {bb} | {complete} |"
                )
            horizon = len(refs["DMS_split_0.5"])
            ticks = {0: "0\nBaseline split"}
            fixed, best = split_points["fixed"], split_points["best"]
            if fixed == best:
                ticks[fixed] = f"{fixed:,}\nBoth late splits"
            else:
                ticks[fixed] = f"{fixed:,}\nFixed-time split"
                ticks[best] = f"{best:,}\nBest-state split"
            for value in [int(0.75 * horizon), horizon]:
                if all(abs(value - tick) > horizon * 0.08 for tick in ticks):
                    ticks[value] = f"{value:,}"
            # keep nearby split labels distinct without changing their coordinates
            positions = sorted(ticks)
            ax.set_xticks(positions, [ticks[t] for t in positions])
            if best < horizon * 0.12:
                ax.get_xticklabels()[positions.index(0)].set_ha("right")
                ax.get_xticklabels()[positions.index(best)].set_ha("left")
            if fixed != best and abs(fixed - best) < horizon * 0.12:
                ax.get_xticklabels()[positions.index(best)].set_ha("right")
                ax.get_xticklabels()[positions.index(fixed)].set_ha("left")
            ax.set_xlim(0, horizon)
            ax.set_xlabel("Iterations", labelpad=8)
            ax.set_ylabel("Cost")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            domain = manifest.get("domain_size", original.variables[0].domain)
            ax.set_title(
                f"Random dense — seed {seed} · 50 agents · domain {domain}",
                loc="left",
                fontsize=14,
                pad=44,
            )
            ax.grid(axis="y", color="#e7e7e7", lw=0.6)
            remove_frame(ax)
            ax.legend(
                frameon=False,
                fontsize=9.5,
                loc="lower left",
                bbox_to_anchor=(0, 1.01),
                borderaxespad=0,
                ncol=3,
            )
            fig.tight_layout()
            for ext in ["png", "pdf"]:
                fig.savefig(output / f"cost_lines_seed_{seed}.{ext}", dpi=180)
            plt.close(fig)
    (run_dir / "COMPARISON.md").write_text("\n".join(report) + "\n")
    # retain the renderer used for the delivered figures independently of runtime code
    shutil.copyfile(__file__, output / "plot_comparison.py")
    (output / "renderer_sha256.txt").write_text(sha(Path(__file__)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_comparison(parser.parse_args().run_dir)
