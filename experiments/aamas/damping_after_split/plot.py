"""Plot the approved equal-seed mean comparison from saved evidence only."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402
from experiments.aamas.late_split.core import (  # noqa: E402
    input_fingerprint,
    load_input,
    tail_kind,
    verify_costs,
    write_json,
)
from experiments.aamas.late_split.plot_population import align_history  # noqa: E402
from experiments.aamas.late_split.resume import checked, read_arrays  # noqa: E402
from experiments.aamas.late_split.run import sha  # noqa: E402


def comparison(left: np.ndarray, right: np.ndarray) -> list[int]:
    """Count paired lower, equal, and higher costs with the existing tolerance."""
    delta = left - right
    return [
        int((delta < -1e-8).sum()),
        int((abs(delta) <= 1e-8).sum()),
        int((delta > 1e-8).sum()),
    ]


def plot(run_dir: Path) -> dict:
    """Validate saved traces and create full and handoff figures plus numeric data."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    checked(run_dir, "parent_evidence.json")
    seeds, horizon = manifest["seeds"], manifest["additional_updates"]
    n = manifest["config"]["prefix_steps"]
    handoff = n + manifest["config"]["post_steps"]
    rows, prefix, baseline, bp, continuation, mgm, selected, rounds = (
        [] for _ in range(8)
    )
    errors, recoveries = [], []
    for seed in seeds:
        case = run_dir / f"random_dense_{seed}"
        source = run_dir / "parent_evidence" / case.name
        result = checked(case, "result.json")
        if result is None:
            raise ValueError(f"incomplete seed {seed}")
        original = load_input(source / "input.npz")
        traces = [
            read_arrays(source / name)
            for name in ["prefix.npz", "best_trace.npz", "DMS_split_0.5_trace.npz"]
        ]
        trace = read_arrays(case / "damped_trace.npz")
        recovery = json.loads((case / "recovery.json").read_text())
        recoveries.append(recovery)
        if (
            result["final_cost"] != trace["costs"][-1]
            or result["original_input_sha256"] != input_fingerprint(original)
            or result["tail_kind"] != tail_kind(trace["assignments"], 100)
            or recovery["first_update_damping"]["formula_max_error"] != 0
            or not recovery["every_replayed_assignment_and_cost_exact"]
            or recovery["replayed_updates"] != (1000 if seed < 3 else 250)
            or recovery["full_terminal_state_matches_saved_checkpoint"] != (seed >= 3)
        ):
            raise ValueError("continuation result or recovery verification differs")
        for t in [*traces, trace]:
            errors.append(verify_costs(original, t))
        if len(trace["costs"]) != horizon:
            raise ValueError("incomplete continuation")
        np.testing.assert_array_equal(
            trace["iterations"], traces[1]["iterations"][-1] + np.arange(1, horizon + 1)
        )
        record = json.loads((source / "best_result.json").read_text())
        winner = json.loads(
            (
                run_dir
                / "parent_evidence/mgm_tail_values"
                / f"seed_{seed}_best_mgm.json"
            ).read_text()
        )
        if winner["trace_sha256"] != sha(source / "best_trace.npz") or winner[
            "input_sha256"
        ] != input_fingerprint(original):
            raise ValueError("MGM evidence belongs to a different run")
        prefix.append(traces[0]["costs"])
        bp.append(traces[1]["costs"])
        baseline.append(traces[2]["costs"])
        continuation.append(trace["costs"])
        mgm.append(align_history(winner["best"]["costs"], horizon))
        selected.append(record["selected_checkpoint_cost"])
        rounds.append(winner["best"]["rounds"])
        rows.append(
            {
                "seed": seed,
                "normal_baseline": baseline[-1][-1],
                "undamped_terminal": bp[-1][-1],
                "mgm": mgm[-1][-1],
                "restored_damping": continuation[-1][-1],
                "damping_tail": result["tail_kind"],
            }
        )
    arrays = {
        "seeds": np.array(seeds),
        "prefix_costs": np.array(prefix),
        "undamped_costs": np.array(bp),
        "baseline_costs": np.array(baseline),
        "damped_costs": np.array(continuation),
        "mgm_costs": np.array(mgm),
        "selected_checkpoint_costs": np.array(selected),
    }
    np.savez_compressed(run_dir / "aligned_costs.npz", **arrays)
    means = {k: v.mean(axis=0) for k, v in arrays.items() if k != "seeds"}
    end = handoff + horizon
    damping_final = arrays["damped_costs"][:, -1]
    mgm_final, base_final = arrays["mgm_costs"][:, -1], arrays["baseline_costs"][:, -1]
    summary = {
        "seeds": len(seeds),
        "mean_restored_damping": float(damping_final.mean()),
        "mean_mgm": float(mgm_final.mean()),
        "mean_normal_baseline": float(base_final.mean()),
        "mean_undamped_terminal": float(arrays["undamped_costs"][:, -1].mean()),
        "damping_better_equal_worse_vs_mgm": comparison(damping_final, mgm_final),
        "damping_better_equal_worse_vs_baseline": comparison(damping_final, base_final),
        "damping_tail_counts": dict(Counter(r["damping_tail"] for r in rows)),
        "lower_mean_cost_than_mgm_percent": float(
            100 * (mgm_final.mean() - damping_final.mean()) / mgm_final.mean()
        ),
        "lower_mean_cost_than_baseline_percent": float(
            100 * (base_final.mean() - damping_final.mean()) / base_final.mean()
        ),
    }
    write_json(run_dir / "summary.json", summary)
    with (run_dir / "per_seed_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "mean_costs.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "iteration",
                "common_best_split_bp",
                "restored_damping",
                "mgm",
                "normal_baseline_or_held_final",
            ]
        )
        common = np.r_[means["prefix_costs"], means["undamped_costs"]]
        for x in range(1, end + 1):
            writer.writerow(
                [
                    x,
                    common[x - 1] if x <= handoff else "",
                    means["damped_costs"][x - handoff - 1] if x > handoff else "",
                    means["mgm_costs"][x - handoff] if x >= handoff else "",
                    means["baseline_costs"][min(x, handoff) - 1],
                ]
            )

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    orange, blue, gray = "#BE6B21", "#2563A6", "#484848"
    for zoom in [False, True]:
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax.plot(
            np.arange(1, handoff + 1),
            means["baseline_costs"],
            color=gray,
            lw=1.5,
            label="Normal split + damping 0.9",
        )
        ax.plot(
            [handoff, end],
            [means["baseline_costs"][-1]] * 2,
            color=gray,
            ls=":",
            lw=1.5,
        )
        ax.plot(
            np.arange(1, n + 1),
            means["prefix_costs"],
            color=orange,
            lw=1.5,
            label="Best split → MGM",
        )
        ax.plot(
            [n, n],
            [means["prefix_costs"][-1], means["selected_checkpoint_costs"]],
            color=orange,
            ls=":",
            lw=1.5,
        )
        ax.plot(
            np.arange(n, handoff + 1),
            np.r_[means["selected_checkpoint_costs"], means["undamped_costs"]],
            color=orange,
            lw=1.5,
        )
        ax.plot(
            [handoff, handoff],
            [means["undamped_costs"][-1], means["mgm_costs"][0]],
            color=orange,
            ls=":",
            lw=1.5,
        )
        last_mgm = max(rounds)
        ax.plot(
            handoff + np.arange(last_mgm + 1),
            means["mgm_costs"][: last_mgm + 1],
            color=orange,
            lw=2,
        )
        ax.plot(
            [handoff + last_mgm, end],
            [means["mgm_costs"][-1]] * 2,
            color=orange,
            ls=":",
            lw=1.6,
        )
        ax.plot(
            np.arange(handoff, end + 1),
            np.r_[means["undamped_costs"][-1], means["damped_costs"]],
            color=blue,
            lw=1.7,
            label="Best split → restore damping 0.9",
        )
        ax.set_title(
            f"Best-checkpoint split: restore damping or use MGM\n"
            f"Random dense · 50 agents · domain 20 · mean of {len(seeds)} seeds",
            pad=12,
        )
        if zoom:
            ax.set_xlim(handoff - 50, handoff + 200)
            visible = np.r_[
                means["baseline_costs"][-50:],
                means["undamped_costs"][-50:],
                means["damped_costs"][:200],
                means["mgm_costs"][:201],
            ]
            padding = max(10, 0.1 * np.ptp(visible))
            ax.set_ylim(visible.min() - padding, visible.max() + padding)
            ticks = [
                handoff - 50,
                handoff,
                handoff + 50,
                handoff + 100,
                handoff + 150,
                handoff + 200,
            ]
        else:
            ax.set_xlim(0, end)
            ticks = [0, 500, n, 1500, handoff, 2500, end]
        labels = [f"{x:,}" for x in ticks]
        labels[ticks.index(handoff)] += "\nMGM / restore damping"
        if n in ticks:
            labels[ticks.index(n)] += "\nBest checkpoint → split"
        ax.set_xticks(ticks, labels)
        ax.set_xlabel(
            "Iterations (BP updates; orange after 2,000: MGM rounds)", labelpad=10
        )
        ax.set_ylabel("Mean original cost")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=0.18)
        ax.legend(
            loc="center right" if zoom else "upper right", frameon=False, fontsize=9
        )
        remove_frame(ax)
        fig.text(
            0.5,
            0.02,
            "Dotted tails hold completed costs; the normal baseline ends at 2,000 updates.",
            ha="center",
            fontsize=9,
            color=gray,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        name = "mean_best_restore_damping" + ("_handoff_zoom" if zoom else "")
        fig.savefig(out / f"{name}.png", dpi=210)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
    shutil.copyfile(Path(__file__), out / "plot.py")
    write_json(
        run_dir / "VALIDATION.json",
        {
            "seeds_checked": seeds,
            "original_cost_max_error": max(errors),
            "new_costs_and_assignments_checked": len(seeds) * horizon,
            "all_native_iteration_sequences_match_parent": True,
            "exact_full_terminal_state_replays": sum(
                r["full_terminal_state_matches_saved_checkpoint"] for r in recoveries
            ),
            "first_damped_q_messages_verified": sum(
                r["first_update_damping"]["messages_checked"] for r in recoveries
            ),
            "recovery_replayed_updates": sum(r["replayed_updates"] for r in recoveries),
            "plot_files_sha256": {
                p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()
            },
            "aggregation": "every plotted mean includes all 50 seeds; no smoothing",
        },
    )
    text = f"""# Restoring damping after the best-checkpoint split

Same 50 random-dense instances, 50 agents, domain 20. Lower original cost is better.

| Method | Mean final cost |
| --- | ---: |
| Normal split + damping (2,000 measured updates) | {summary['mean_normal_baseline']:,.6f} |
| Best split: terminal undamped BP | {summary['mean_undamped_terminal']:,.6f} |
| Best split → existing all-tail-values MGM | {summary['mean_mgm']:,.6f} |
| Best split → restore damping 0.9 for 1,000 more updates | {summary['mean_restored_damping']:,.6f} |

Restored damping is lower/equal/higher than MGM on
{summary['damping_better_equal_worse_vs_mgm']} seeds, and than the normal baseline on
{summary['damping_better_equal_worse_vs_baseline']} seeds.
The last 100 damped assignments classify as {summary['damping_tail_counts']}.
These finite-window assignment labels do not prove message convergence.

Execution: reuse the first 1,000 DMS observations; restore the earliest best
checkpoint; reuse 1,000 undamped equal-split updates. At plot x=2,000, continue
that exact terminal state with native old-Q damping 0.9 for 1,000 new updates.
The split graph stays intact. The actual preceding Q is recovered by exact
suffix replay: 250 updates for 47 seeds, 1,000 for the three imported pilot seeds.
All replayed costs/assignments match; all 47 available full terminal states match.
Every first damped Q was checked against the blend formula without changing state.
All 50,000 new costs were reconstructed independently from the original tables.
Resumed BP retains the full 20-value domains; the reused MGM searches retain
their original menus of every value observed in the undamped final 100 updates.

The MGM results and normal baseline are reused. No fixed-time experiment, new
MGM search or B&B was run. All 50 seeds contribute equally at every plot point.
After 2,000 the new branch uses BP updates and the MGM branch uses winning-start
MGM rounds; work across its other starts is not counted. Dotted tails retain
finished costs. The baseline stops at 2,000; no 3,000-update baseline is implied.
This is a trajectory comparison, not an equal-CPU-time comparison.

- [Full mean plot](plots/mean_best_restore_damping.png)
- [Handoff close-up](plots/mean_best_restore_damping_handoff_zoom.png)
- Matching vector PDFs are in `plots/`.
- `mean_costs.csv`, `per_seed_summary.csv`, `aligned_costs.npz` retain plot data.
- Every seed retains a recovered undamped state, its damped starting state, full
  checkpoints every 250 new updates, and all per-update assignments and costs.
- `parent_evidence/` holds copied exact inputs and comparison data; `source/`
  and `manifest.json` record execution provenance. See `PROTOCOL.md` for reruns.
"""
    (run_dir / "RESULTS.md").write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
