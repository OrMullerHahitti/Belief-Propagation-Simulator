"""Verify pilot artifacts and export cost/tail figures and a compact report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aamas.late_split.core import load_input, verify_costs  # noqa: E402
from experiments.aamas.late_split.run import sha  # noqa: E402
from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402

BLUE = "#2563A6"
ORANGE = "#BE6B21"
INK = "#262626"
GREY = "#858585"


def money(value):
    """Format objective cost; no monetary units are implied."""
    return "—" if value is None else f"{value:,.3f}"


def summarize(run_dir: Path) -> None:
    """Read completed results, validate all costs/hashes, and export two figures."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cases = [(b, s) for b in manifest["benchmarks"] for s in manifest["seeds"]]
    if len(cases) > 6:
        raise ValueError("this per-instance pilot figure is limited to six cases")
    collected = []
    rows = []
    for benchmark, seed in cases:
        directory = run_dir / f"{benchmark}_{seed}"
        original = load_input(directory / "input.npz")
        with np.load(directory / "prefix.npz", allow_pickle=False) as f:
            prefix = dict(f)
        verify_costs(original, prefix)
        with np.load(directory / "references.npz", allow_pickle=False) as f:
            references = dict(f)
        modes = {}
        for mode in ["fixed", "best"]:
            result = json.loads((directory / f"{mode}_result.json").read_text())
            for file, checksum in result["files_sha256"].items():
                if sha(directory / file) != checksum:
                    raise ValueError(f"modified evidence: {directory / file}")
            with np.load(directory / f"{mode}_trace.npz", allow_pickle=False) as f:
                trace = dict(f)
            verify_costs(original, trace)
            merge = result["merge"]
            row = {
                "benchmark": benchmark,
                "seed": seed,
                "mode": mode,
                "checkpoint_index": result["split_before_iteration"] - 1,
                "prefix_best": result["prefix_best_cost"],
                "tail_kind": merge["tail_kind"],
                "branch_a": merge["branch_costs"][0],
                "branch_b": merge["branch_costs"][1],
                "mgm": None if merge["mgm"] is None else merge["mgm"]["cost"],
                "bb": None if merge["bb"] is None else merge["bb"]["cost"],
                "bb_complete": None if merge["bb"] is None else merge["bb"]["complete"],
            }
            rows.append(row)
            modes[mode] = (result, trace)
        collected.append((benchmark, seed, prefix, references, modes))
    with (run_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
        }
    )
    output = run_dir / "plots"
    output.mkdir(exist_ok=True)
    for kind in ["trajectories", "tails_and_merges"]:
        fig, axes = plt.subplots(
            len(cases), 2, figsize=(13, 3.25 * len(cases)), squeeze=False
        )
        for row_index, (benchmark, seed, prefix, refs, modes) in enumerate(collected):
            for col, mode in enumerate(["fixed", "best"]):
                result, trace = modes[mode]
                ax = axes[row_index, col]
                remove_frame(ax)
                n = len(prefix["costs"])
                post = np.arange(1, len(trace["costs"]) + 1)
                if kind == "trajectories":
                    ax.plot(
                        np.arange(1, n + 1),
                        prefix["costs"],
                        color=INK,
                        lw=1.1,
                        label="DMS prefix",
                    )
                    ax.plot(
                        post + n,
                        trace["costs"],
                        color=BLUE,
                        lw=1.0,
                        zorder=3,
                        label="Undamped after split",
                    )
                    ax.plot(
                        refs["DMS_iterations"] + 1,
                        refs["DMS"],
                        color=GREY,
                        ls="--",
                        lw=1,
                        label="DMS retained",
                    )
                    ax.axvline(n, color=GREY, ls=":", lw=1)
                    ax.set_xlabel("Completed updates (selection window + continuation)")
                    ax.set_title(
                        f"{benchmark}, seed {seed} · {mode} checkpoint\n"
                        f"restored native index {result['split_before_iteration'] - 1}",
                        fontsize=10,
                    )
                else:
                    tail = min(50, len(post))
                    ax.plot(
                        post[-tail:],
                        trace["costs"][-tail:],
                        color=BLUE,
                        lw=1,
                        marker=".",
                        markersize=3,
                        label="Tail assignment cost",
                    )
                    merge = result["merge"]
                    if merge["mgm"]:
                        ax.axhline(
                            merge["mgm"]["cost"],
                            color=ORANGE,
                            ls="--",
                            lw=1.2,
                            label="MGM",
                        )
                    if merge["bb"]:
                        ax.axhline(
                            merge["bb"]["cost"],
                            color=INK,
                            ls="-.",
                            lw=1.1,
                            label="B&B incumbent",
                        )
                    status = (
                        "not applicable"
                        if merge["bb"] is None
                        else (
                            "menu optimum proved"
                            if merge["bb"]["complete"]
                            else "B&B timed out"
                        )
                    )
                    if merge["status"] == "no_op":
                        status = "one assignment; both merges are no-ops"
                    visible = list(trace["costs"][-tail:]) + [
                        result["prefix_best_cost"]
                    ]
                    if merge["mgm"]:
                        visible += [merge["mgm"]["cost"], merge["bb"]["cost"]]
                    if max(visible) - min(visible) < 1e-8:
                        value = visible[0]
                        ax.set_ylim(value - 1, value + 1)
                        ax.set_yticks([value])
                        ax.text(
                            0.5,
                            0.78,
                            f"All costs coincide at {value:,.3f}",
                            transform=ax.transAxes,
                            ha="center",
                            fontsize=10,
                        )
                    ax.set_title(
                        f"seed {seed} · {mode}: {merge['tail_kind']}\n{status}",
                        fontsize=10,
                    )
                    ax.set_xlabel("Post-split update")
                ax.axhline(
                    result["prefix_best_cost"],
                    color=GREY,
                    ls=":",
                    lw=1.2,
                    label="Prefix incumbent",
                )
                ax.set_ylabel("Original objective cost")
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.grid(axis="y", color="#E4E4E4", lw=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        title = (
            "Late split: native cost trajectories"
            if kind == "trajectories"
            else "Late split: final assignment tails and menu merges"
        )
        fig.suptitle(
            title
            + f"\nDense pilot, {len(cases)} instances; damping 0.9 → 0, equal splitting",
            fontsize=13,
            y=0.99,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.944),
            ncol=5,
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.5,
            0.009,
            "Per-instance evidence; two-cycle classification uses the last 100 assignments. "
            "B&B is restricted to their menus.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.032, 1, 0.905))
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{kind}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    lines = [
        "# Late-split pilot — three dense inputs",
        "",
        "Both approved phases completed. Costs below use the original tables, including unary preferences. "
        "The results describe these inputs only; the 50-instance study remains unrun.",
        "",
        "| Seed | Mode | Checkpoint index | Prefix best | Tail | "
        "Branch A | Branch B | MGM | B&B | Menu optimum proved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['mode']} | {row['checkpoint_index']} | "
            f"{money(row['prefix_best'])} | {row['tail_kind']} | {money(row['branch_a'])} | "
            f"{money(row['branch_b'])} | {money(row['mgm'])} | {money(row['bb'])} | {row['bb_complete']} |"
        )
    lines += [
        "",
        "## Retained corrected comparisons",
        "",
        "All entries use the same seed. Previous merges use the immediate split at updates 198/199; "
        "they have a different extraction time from this pilot's last post-split pair.",
        "",
        "| Seed | DMS final | Damped split at 1000 | Fixed 0.95 split | Pulse | Earlier MGM | Earlier B&B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, seed, _, refs, _ in collected:
        labels = [
            "DMS",
            "DMS_split_at_1000",
            "DMS_split_0.95",
            "DMS_split_pulse",
            "MS_split_MGM_200",
            "MS_split_opt_200",
        ]
        lines.append(
            f"| {seed} | "
            + " | ".join(
                money(refs[label][-1]) if label in refs else "—" for label in labels
            )
            + " |"
        )
    lines += [
        "",
        "## Verification",
        "",
        "Every saved cost was independently reconstructed. Each unsplit prefix matches the retained DMS CSV "
        "at its recorded four-decimal precision. Each best checkpoint was replayed to the prefix end, "
        "matching costs, assignments and final dynamic state exactly. "
        "Evidence file hashes were checked again before plotting.",
        "",
        "The best-point method observes all 1,000 prefix updates before restoring its checkpoint; "
        "the plots count this search work. Its additional validation replay is instrumentation overhead. "
        "Native normalization continues on the original update index.",
        "",
        "The original launch stopped at an overly strict CSV-precision check before producing a continuation. "
        "Its artifacts are preserved in the sibling `late_split_pilot_20260919_csv_precision_check` directory.",
        "",
        "## Figures",
        "",
        "![Cost trajectories](plots/trajectories.png)",
        "",
        "![Tails and merges](plots/tails_and_merges.png)",
        "",
    ]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))
    print(f"Validated {len(rows)} continuations; wrote {run_dir / 'SUMMARY.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir)


if __name__ == "__main__":
    main()
