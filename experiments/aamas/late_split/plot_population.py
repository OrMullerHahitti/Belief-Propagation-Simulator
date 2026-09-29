"""Validate saved population evidence and plot equal-seed mean cost trajectories."""

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
import numpy as np  # noqa: E402

from experiments.aaai.code.merge import score_assignment  # noqa: E402
from experiments.aaai.code.problems import capture_original  # noqa: E402
from experiments.aaai.code.utils.plot_helpers import remove_frame  # noqa: E402
from .core import input_fingerprint, load_input, verify_costs, write_json  # noqa: E402
from .mgm import observed_menus  # noqa: E402
from .resume import checked, read_arrays  # noqa: E402
from .run import sha  # noqa: E402


def align_history(history: list[float], rounds: int) -> np.ndarray:
    """Hold terminated trajectories fixed without dropping seeds from the mean."""
    if len(history) > rounds + 1 or not history:
        raise ValueError("invalid common MGM horizon")
    return np.pad(history, (0, rounds + 1 - len(history)), mode="edge")


def write_report(run_dir: Path, summary: dict, manifest: dict) -> None:
    """Record the comparison, its limits, and links to reusable evidence."""
    count = len(manifest["seeds"])
    baseline = summary["mean_standard_baseline"]
    lines = [
        "# Random-dense domain-20 population results",
        "",
        f"{count} seeds: {manifest['seeds']}. Lower original-objective cost is better.",
        f"Reused pilot seeds: {manifest.get('reused_seeds', [])}; all remaining seeds are new executions.",
        "",
        "| Method | Mean final cost | Change versus standard baseline | Better / equal / worse seeds |",
        "| --- | ---: | ---: | --- |",
        f"| Standard equal split + damping 0.9 | {baseline:,.6f} | — | — |",
    ]
    for mode, label in [
        ("fixed", "Fixed-time late split + MGM"),
        ("best", "Best-checkpoint late split + MGM"),
    ]:
        result = summary[mode]
        comparison = " / ".join(map(str, result["better_equal_worse_vs_standard"]))
        lines.append(
            f"| {label} | {result['mean_mgm_cost']:,.6f} | "
            f"{-result['relative_mean_improvement_percent']:+.3f}% | {comparison} |"
        )
    lines += [
        "",
        "The percentage compares means over the same seeds; negative is better.",
        "These are descriptive results on the saved instances, not a convergence or significance claim.",
        "",
        "## Exact execution",
        "",
        "50 agents, domain 20, random-dense edge probability 0.6. Both controls run for 2,000 updates.",
        "Both late-split methods observe 1,000 unsplit updates with damping 0.9, equally split factors",
        "with equal transfer of the existing R messages, remove damping, then run 1,000 updates.",
        "The fixed method starts from update 1,000; the best method restores the earliest minimum-cost",
        "full message state in that observation window. Native normalization phase is preserved.",
        "MGM uses every per-variable value observed in the final 100 post-split assignments and starts",
        "from every distinct joint assignment in that window. Every start is saved; the winner has",
        "the lowest final cost. No new B&B was run. See [PROTOCOL.md](PROTOCOL.md) for commands and storage.",
        "",
        "## BP history and MGM",
        "",
        f"Damping-only DMS: mean terminal cost {summary['mean_damping_only']:,.6f}; "
        f"mean per-seed best cost seen {summary['mean_damping_only_best_seen']:,.6f}.",
        f"Standard damped splitting: mean per-seed best cost seen {summary['mean_standard_baseline_best_seen']:,.6f}.",
        "",
    ]
    for mode in ["fixed", "best"]:
        r = summary[mode]
        lines += [
            f"- **{mode}:** mean terminal BP cost {r['mean_terminal_bp_cost']:,.6f}; "
            f"mean best BP cost seen {r['mean_best_bp_seen']:,.6f}; "
            f"MGM {r['mean_mgm_cost']:,.6f}. "
            f"MGM better/equal/worse than its own best BP: {r['better_equal_worse_vs_own_best_bp']}. "
            f"Tail classifications: {r['tail_counts']}. "
            f"Winning MGM improving rounds (min/median/max): {r['winning_mgm_rounds_min_median_max']}."
        ]
    lines += [
        "",
        "An earlier BP incumbent is not silently substituted for a worse MGM outcome.",
        "For unsettled tails, the last 100 assignments are an observation window, not a proof of a full cycle.",
        "Tail labels use those 100 assignments: fixed means identical assignments, "
        "period_two means two alternating assignments, and other includes longer periods or irregular behavior.",
        "None of these finite-window labels proves message convergence or indefinite future behavior.",
        "",
        "## Plots and saved data",
        "",
        "- [Fixed-time mean](plots/mean_fixed_pipeline.png) and "
        "[MGM close-up](plots/mean_fixed_pipeline_mgm_zoom.png).",
        "- [Best-checkpoint mean](plots/mean_best_pipeline.png) and "
        "[MGM close-up](plots/mean_best_pipeline_mgm_zoom.png).",
        "- Matching vector PDFs are in `plots/`.",
        "- `per_seed_summary.csv`, `mean_costs.csv`, `aligned_costs.npz`, `summary.json` "
        "retain the numerical comparison.",
        "- Per-case directories retain original inputs, every BP cost/assignment and split checkpoints.",
        "- New seeds additionally retain full runtime states every 250 post-split updates and at baseline termination.",
        "- `mgm_tail_values/` retains all MGM starts, menus, round histories and endpoints.",
        "",
        f"Every point in each mean includes all {count} seeds. Finished MGM runs are held at their final cost.",
        "The best-checkpoint plot counts all 1,000 observation updates before restoration; "
        "native indices remain saved.",
        "After x=2,000 one x-unit is one MGM round. "
        "The winning-start trajectory does not count work across all starts.",
        "Dotted extensions hold completed costs; "
        "a dotted vertical segment denotes initialization, not an improving move.",
        "Baselines stop at 2,000 updates. This is a cost trajectory comparison, "
        "not equal CPU time or total search work.",
        "",
        "`VALIDATION.json` records independent original-cost checks, menu/start checks, and artifact hashes.",
        "`manifest.json` and `source/` retain exact simulation provenance; plotting can be repeated without BP or MGM.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def validate_case(run_dir: Path, seed: int, config: dict) -> dict:
    """Check numeric traces and all MGM starts against the saved original tables."""
    case = run_dir / f"random_dense_{seed}"
    original = load_input(case / "input.npz")
    names, axes, tables = capture_original(original)
    prefix = read_arrays(case / "prefix.npz")
    verify_costs(original, prefix)
    n, post = config["prefix_steps"], config["post_steps"]
    if len(prefix["costs"]) != n:
        raise ValueError("prefix length differs")
    data = {"prefix": prefix["costs"], "modes": {}}
    for label in ["DMS", "DMS_split_0.5"]:
        trace = read_arrays(case / f"{label}_trace.npz")
        verify_costs(original, trace)
        if len(trace["costs"]) != n + post:
            raise ValueError("baseline length differs")
        data[label] = trace["costs"]
    np.testing.assert_array_equal(prefix["costs"], data["DMS"][:n])
    for mode in ["fixed", "best"]:
        result = checked(case, f"{mode}_result.json")
        if result is None:
            raise ValueError("missing completed BP result")
        trace = read_arrays(case / f"{mode}_trace.npz")
        verify_costs(original, trace)
        if len(trace["costs"]) != post:
            raise ValueError("continuation length differs")
        source = run_dir / "mgm_tail_values" / f"seed_{seed}_{mode}_mgm.json"
        mgm = json.loads(source.read_text())
        if mgm["trace_sha256"] != sha(case / f"{mode}_trace.npz"):
            raise ValueError("MGM trace hash differs")
        if mgm["input_sha256"] != input_fingerprint(original) or mgm["bb_run"]:
            raise ValueError("MGM input/protocol differs")
        menus, indices = observed_menus(trace, config["tail_steps"])
        if any(
            (
                mgm["menus"] != menus,
                [s["source_trace_index"] for s in mgm["starts"]] != indices,
            )
        ):
            raise ValueError("missing tail values or starts")
        for index, start in zip(indices, mgm["starts"]):
            initial = dict(
                zip(trace["variable_names"], map(int, trace["assignments"][index]))
            )
            if initial != start["initial_assignment"] or start["hit_round_cap"]:
                raise ValueError("MGM start or stopping condition differs")
            if abs(start["costs"][0] - trace["costs"][index]) > 1e-8:
                raise ValueError("MGM initial cost differs")
            if len(start["costs"]) != start["rounds"] + 1 or not np.all(
                np.diff(start["costs"]) < -1e-9
            ):
                raise ValueError("MGM history does not contain improving rounds")
            score = score_assignment(start["assignment"], tables, axes)
            if abs(score - start["cost"]) > 1e-8 or score != start["costs"][-1]:
                raise ValueError("MGM final score differs")
            if any(start["assignment"][v] not in menus[v] for v in names):
                raise ValueError("MGM left its menus")
        winner = min(mgm["starts"], key=lambda row: row["cost"])
        if winner != mgm["best"]:
            raise ValueError("MGM selected winner differs")
        # check the full objective independently of MGM's local gain computation
        for name in names:
            for value in menus[name]:
                if value == winner["assignment"][name]:
                    continue
                alternative = {**winner["assignment"], name: value}
                if score_assignment(alternative, tables, axes) < winner["cost"] - 1e-8:
                    raise ValueError(
                        "winning MGM assignment has an improving menu move"
                    )
        data["modes"][mode] = {"trace": trace, "result": result, "mgm": mgm}
    return data


def plot_population(run_dir: Path) -> None:
    """Render full and MGM-close-up figures with every seed in every mean."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    seeds, config = manifest["seeds"], manifest["config"]
    data = [validate_case(run_dir, seed, config) for seed in seeds]
    count = len(seeds)
    n, total = config["prefix_steps"], config["prefix_steps"] + config["post_steps"]
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    baseline = np.array([d["DMS_split_0.5"] for d in data])
    damping = np.array([d["DMS"] for d in data])
    arrays = {
        "seeds": np.array(seeds),
        "baseline_costs": baseline,
        "dms_costs": damping,
    }
    rows, summary, figure_files = [], {}, []
    for mode, color, title in [
        ("fixed", "#2563A6", "Split after 1,000 damped updates"),
        ("best", "#BE6B21", "Split from the best checkpoint"),
    ]:
        records = [d["modes"][mode] for d in data]
        rounds = max(r["mgm"]["best"]["rounds"] for r in records)
        bp = np.array(
            [
                np.concatenate([d["prefix"], r["trace"]["costs"]])
                for d, r in zip(data, records)
            ]
        )
        mgm = np.array(
            [align_history(r["mgm"]["best"]["costs"], rounds) for r in records]
        )
        arrays[mode + "_bp_costs"] = bp
        arrays[mode + "_mgm_costs"] = mgm
        selected = np.array([r["result"]["selected_checkpoint_cost"] for r in records])
        arrays[mode + "_selected_checkpoint_costs"] = selected
        mean_bp, mean_mgm = bp.mean(axis=0), mgm.mean(axis=0)
        difference = baseline[:, -1] - mgm[:, -1]
        anytime = np.array([r["result"]["anytime_cost"] for r in records])
        summary[mode] = {
            "mean_mgm_cost": float(mean_mgm[-1]),
            "mean_terminal_bp_cost": float(mean_bp[-1]),
            "mean_best_bp_seen": float(anytime.mean()),
            "mean_improvement_vs_standard_baseline": float(difference.mean()),
            "relative_mean_improvement_percent": float(
                100 * difference.mean() / baseline[:, -1].mean()
            ),
            "better_equal_worse_vs_standard": [
                int((difference > 1e-8).sum()),
                int((abs(difference) <= 1e-8).sum()),
                int((difference < -1e-8).sum()),
            ],
            "better_equal_worse_vs_own_best_bp": [
                int((anytime - mgm[:, -1] > 1e-8).sum()),
                int((abs(anytime - mgm[:, -1]) <= 1e-8).sum()),
                int((anytime - mgm[:, -1] < -1e-8).sum()),
            ],
            "tail_counts": dict(Counter(r["mgm"]["tail_kind"] for r in records)),
            "winning_mgm_rounds_min_median_max": [
                int(min(r["mgm"]["best"]["rounds"] for r in records)),
                float(np.median([r["mgm"]["best"]["rounds"] for r in records])),
                rounds,
            ],
        }
        for seed, record, item in zip(seeds, records, data):
            winner, result = record["mgm"]["best"], record["result"]
            rows.append(
                dict(
                    seed=seed,
                    mode=mode,
                    split_before_native_iteration=result["split_before_iteration"],
                    standard_baseline=item["DMS_split_0.5"][-1],
                    damping_only=item["DMS"][-1],
                    damping_only_best=item["DMS"].min(),
                    best_bp_seen=result["anytime_cost"],
                    terminal_bp=result["final_cost"],
                    mgm=winner["cost"],
                    mgm_initial=winner["costs"][0],
                    mgm_rounds=winner["rounds"],
                    mgm_starts=len(record["mgm"]["starts"]),
                    total_mgm_rounds=sum(s["rounds"] for s in record["mgm"]["starts"]),
                    tail=record["mgm"]["tail_kind"],
                    maximum_menu_size=max(map(len, record["mgm"]["menus"].values())),
                )
            )
        for zoom in [False, True]:
            fig, ax = plt.subplots(figsize=(11.5, 5.4))
            bp_x = np.arange(1, total + 1)
            mgm_x = total + np.arange(rounds + 1)
            endpoint = total + max(rounds, 25)
            ax.plot(
                bp_x,
                baseline.mean(axis=0),
                color="#484848",
                lw=1.5,
                label="Standard split + damping 0.9",
            )
            ax.plot(
                [total, endpoint],
                [baseline[:, -1].mean()] * 2,
                color="#484848",
                lw=1.5,
                ls=":",
            )
            if mode == "best":
                ax.plot(
                    bp_x[:n],
                    mean_bp[:n],
                    color=color,
                    lw=1.5,
                    label="Damping → late split → MGM",
                )
                ax.plot(
                    [n, n],
                    [mean_bp[n - 1], selected.mean()],
                    color=color,
                    lw=1.3,
                    ls=":",
                )
                ax.plot(
                    np.r_[n, bp_x[n:]],
                    np.r_[selected.mean(), mean_bp[n:]],
                    color=color,
                    lw=1.5,
                )
            else:
                ax.plot(
                    bp_x,
                    mean_bp,
                    color=color,
                    lw=1.5,
                    label="Damping → late split → MGM",
                )
            ax.plot(
                [total, total], [mean_bp[-1], mean_mgm[0]], color=color, lw=1.3, ls=":"
            )
            ax.plot(mgm_x, mean_mgm, color=color, lw=2)
            ax.plot(
                [mgm_x[-1], endpoint], [mean_mgm[-1]] * 2, color=color, lw=1.3, ls=":"
            )
            ax.scatter(mgm_x[-1], mean_mgm[-1], color=color, s=35, zorder=5)
            if zoom:
                ax.set_xlim(total - 50, endpoint + 2)
                ticks = [total - 50, total - 25, total, total + 10, total + 20]
                if endpoint >= total + 30:
                    ticks.append(total + 30)
                values = np.concatenate(
                    [baseline.mean(axis=0)[-50:], mean_bp[-50:], mean_mgm]
                )
                margin = max(float(np.ptp(values)) * 0.12, 20)
                ax.set_ylim(float(values.min()) - margin, float(values.max()) + margin)
            else:
                ax.set_xlim(0, endpoint + 25)
                ticks = [0, 500, n, 1500, total]
            ax.set_xticks(
                ticks,
                [
                    f"{v:,}\n" + ("Split" if v == n else "MGM" if v == total else "")
                    for v in ticks
                ],
            )
            ax.set_xlabel("Iterations (BP updates, then MGM rounds)", labelpad=8)
            ax.set_ylabel(f"Mean cost ({count} seeds)")
            ax.set_title(
                f"Random dense · 50 agents · domain 20\n{title}",
                loc="left",
                fontsize=13,
            )
            ax.grid(axis="y", alpha=0.18)
            ax.legend(frameon=False, loc="upper right")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            remove_frame(ax)
            note = (
                "All seeds retained; completed runs held at final cost. "
                "MGM: best final result across all tail starts."
            )
            if mode == "best":
                note += (
                    "\nThe first 1,000 updates include checkpoint selection; "
                    "earlier-state restoration occurs at Split."
                )
            fig.text(0.09, 0.025, note, fontsize=8, color="#555555")
            fig.tight_layout(rect=(0, 0.09, 1, 1))
            stem = f"mean_{mode}_pipeline" + ("_mgm_zoom" if zoom else "")
            for suffix in ["png", "pdf"]:
                path = out / f"{stem}.{suffix}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                figure_files.append(path)
            plt.close(fig)
    np.savez_compressed(run_dir / "aligned_costs.npz", **arrays)
    with (run_dir / "per_seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "mean_costs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "stage", "iteration", "mean_cost", "seeds"])
        for mode in ["fixed", "best"]:
            for stage, offset in [("bp", 1), ("mgm", total)]:
                for i, cost in enumerate(arrays[f"{mode}_{stage}_costs"].mean(axis=0)):
                    writer.writerow([mode, stage, i + offset, cost, count])
        for label, series in [
            ("standard_split_damping", baseline),
            ("damping_only", damping),
        ]:
            for i, cost in enumerate(series.mean(axis=0), 1):
                writer.writerow([label, "bp", i, cost, count])
    summary.update(
        seeds=seeds,
        mean_standard_baseline=float(baseline[:, -1].mean()),
        mean_standard_baseline_best_seen=float(baseline.min(axis=1).mean()),
        mean_damping_only=float(damping[:, -1].mean()),
        mean_damping_only_best_seen=float(damping.min(axis=1).mean()),
    )
    write_json(run_dir / "summary.json", summary)
    write_report(run_dir, summary, manifest)
    shutil.copyfile(Path(__file__), out / "plot_population.py")
    figure_files.extend(
        run_dir / name
        for name in [
            "aligned_costs.npz",
            "mean_costs.csv",
            "per_seed_summary.csv",
            "summary.json",
            "RESULTS.md",
        ]
    )
    write_json(
        run_dir / "VALIDATION.json",
        {
            "seeds": count,
            "bp_traces_verified": count * 5,
            "mgm_results_verified": count * 2,
            "all_original_costs_reconstructed": True,
            "all_tail_values_and_starts_verified": True,
            "all_mgm_rounds_strictly_improving": True,
            "all_mgm_endpoints_scored_on_original_tables": True,
            "winning_mgm_menu_local_optima_verified": True,
            "constant_n_in_means": count,
            "renderer_sha256": sha(Path(__file__)),
            "artifacts_sha256": {
                str(p.relative_to(run_dir)): sha(p) for p in figure_files
            },
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_population(parser.parse_args().run_dir)
