"""Figure functions shared by the small (50-seed) and bigger (seed-0) plot sets.

Every panel shows a weight value itself, on its own 0-1 scale, or a cost.
No ratios, logs or normalized differences anywhere. ``runs`` is a list of
the dicts produced by ``common.load_small_run`` / ``common.load_bigger_run``
(50 seeds for the small experiment, a single run for the bigger one); the
first run is the one used for single-run panels.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BLUE,
    GRAY,
    GREEN,
    HALF_COLORS,
    HEAD_COLORS,
    SPLIT_COLORS,
    START_WEIGHT,
    VERMILION,
    first_fixed_iteration,
    half_labels,
    pair_halves,
    pair_source_halves,
    plain_axes,
    pooled_over_iterations,
    save,
    split_label,
    twin_mask,
    uniform_share,
    update_guides,
)

RNG = np.random.default_rng(0)
MAX_LINES = 200  # edges drawn in a spaghetti panel
MAX_POINTS = 20000  # points drawn in a scatter panel


def _title(ax, text: str) -> None:
    ax.set_title(textwrap.fill(text, 62), fontsize=9, loc="left")


def _subsample(n: int, k: int) -> np.ndarray:
    return np.sort(RNG.choice(n, size=min(n, k), replace=False))


def _identity(ax, lo: float, hi: float, label: str) -> None:
    pad = (hi - lo) * 0.05 or 1e-6
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        lw=0.8,
        ls="--",
        label=label,
    )


def _pairs(run: dict) -> tuple[np.ndarray, np.ndarray]:
    if "pair_a" not in run:
        run["pair_a"], run["pair_b"] = pair_halves(run)
    return run["pair_a"], run["pair_b"]


def _edge_name(run: dict, k: int) -> str:
    var = run["var_names"][run["trg_var_idx"][k]]
    factor = run["factor_names"][run["fn_orig_idx"][run["trg_fn"][k]]]
    return f"{var} to {factor}"


def _runs_note(runs: list[dict]) -> str:
    return f"{len(runs)} seeds" if len(runs) > 1 else runs[0]["label"]


def _pooled_note(runs: list[dict]) -> str:
    if len(runs) > 1:
        return f"{len(runs)} seeds; a seed drops out once its run stops"
    return runs[0]["label"]


# ------------------------------------------------------------- damping weight
def fig_damping_weights(runs: list[dict], out: Path) -> None:
    final = np.concatenate([run["lam"][-1] for run in runs])
    final_heads = np.concatenate([run["damped"][-1, :, 1, :] for run in runs])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(final, bins=50, color=BLUE, alpha=0.85)
    ax.axvline(START_WEIGHT, color="black", lw=1.0, ls="--", label="start value 0.5")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(
        "damping weight at the last iteration (weight on the previous message)"
    )
    ax.set_ylabel("number of edges")
    _title(
        ax,
        f"Damping weight of every edge at the last iteration ({final.size} edges, {_runs_note(runs)})",
    )

    ax = axes[0, 1]
    n_heads = final_heads.shape[1]
    boxes = ax.boxplot(
        [final_heads[:, h] for h in range(n_heads)],
        tick_labels=[f"head {h}" for h in range(n_heads)],
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], HEAD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(START_WEIGHT, color="black", lw=0.8, ls="--")
    ax.set_ylabel(
        "damping weight at the last iteration (weight on the previous message)"
    )
    _title(ax, "The same weight, separately for each attention head")

    ax = axes[1, 0]
    run = runs[0]
    lam = run["lam"]
    cols = _subsample(lam.shape[1], MAX_LINES)
    ax.plot(lam[:, cols], color=BLUE, lw=0.7, alpha=0.3)
    ax.axhline(START_WEIGHT, color="black", lw=0.8, ls="--")
    update_guides(ax, run["update_interval"], lam.shape[0])
    ax.set_xlabel("iteration")
    ax.set_ylabel("damping weight (weight on the previous message)")
    shown = (
        f"{len(cols)} of {lam.shape[1]} edges"
        if len(cols) < lam.shape[1]
        else f"all {lam.shape[1]} edges"
    )
    _title(
        ax,
        f"Each edge's damping weight over the run ({run['label']}, {shown}; gray = network update)",
    )

    ax = axes[1, 1]
    t, lo, mid, hi = pooled_over_iterations(runs, lambda r: r["lam"])
    ax.fill_between(
        t, lo, hi, color=BLUE, alpha=0.25, lw=0, label="lowest to highest edge"
    )
    ax.plot(t, mid, color=BLUE, lw=1.5, label="median edge")
    ax.axhline(START_WEIGHT, color="black", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("damping weight (weight on the previous message)")
    _title(
        ax,
        f"Lowest, median and highest edge at each iteration ({_pooled_note(runs)})",
    )

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- split halves, damping
def fig_split_halves_damping(runs: list[dict], out: Path) -> None:
    label_a, label_b = half_labels(runs[0]["split_ratio"])
    final_a = np.concatenate([run["lam"][-1, _pairs(run)[0]] for run in runs])
    final_b = np.concatenate([run["lam"][-1, _pairs(run)[1]] for run in runs])
    lo, hi = min(final_a.min(), final_b.min()), max(final_a.max(), final_b.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    idx = _subsample(final_a.size, MAX_POINTS)
    ax.scatter(final_a[idx], final_b[idx], s=6, alpha=0.4, color=BLUE)
    _identity(ax, lo, hi, "equal weights")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(f"damping weight on the edge to {label_a}")
    ax.set_ylabel(f"damping weight on the edge to {label_b}")
    _title(
        ax,
        f"Damping weight of the two edges of each split factor at the last iteration ({final_a.size} pairs)",
    )

    ax = axes[0, 1]
    bins = np.linspace(lo, hi, 50) if hi > lo else 50
    ax.hist(final_a, bins=bins, color=HALF_COLORS[0], alpha=0.6, label=label_a)
    ax.hist(final_b, bins=bins, color=HALF_COLORS[1], alpha=0.6, label=label_b)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(
        "damping weight at the last iteration (weight on the previous message)"
    )
    ax.set_ylabel("number of edges")
    _title(
        ax,
        "The same damping weights as histograms (fully overlapping bars = identical halves)",
    )

    ax = axes[1, 0]
    run = runs[0]
    k_a, k_b = _pairs(run)
    prev = run["damped"][:, :, 1, :]
    # a degree-1 variable's pair never leaves 0.5, so show the pair that moved the most
    moved = np.abs(prev[:, k_a, :] - START_WEIGHT) + np.abs(
        prev[:, k_b, :] - START_WEIGHT
    )
    j = int(moved.max(axis=(0, 2)).argmax())
    for h, color in enumerate(HEAD_COLORS[: run["num_heads"]]):
        ax.plot(prev[:, k_a[j], h], color=color, lw=1.2, label=f"head {h}, half A")
        ax.plot(
            prev[:, k_b[j], h], color=color, lw=1.2, ls="--", label=f"head {h}, half B"
        )
    update_guides(ax, run["update_interval"], run["n_iter"])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("damping weight (weight on the previous message)")
    _title(
        ax,
        f"The damping pair that moved the most over the run ({run['label']}, {_edge_name(run, k_a[j])}; "
        "solid = half A, dashed = half B)",
    )

    ax = axes[1, 1]
    for half, color, label in (
        (0, HALF_COLORS[0], label_a),
        (1, HALF_COLORS[1], label_b),
    ):
        t, lo_h, mid_h, hi_h = pooled_over_iterations(
            runs, lambda r, half=half: r["lam"][:, _pairs(r)[half]]
        )
        ax.fill_between(t, lo_h, hi_h, color=color, alpha=0.2, lw=0)
        ax.plot(
            t,
            mid_h,
            color=color,
            lw=1.5,
            label=f"{label}: median (band = lowest to highest)",
        )
    ax.axhline(START_WEIGHT, color="black", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("damping weight (weight on the previous message)")
    _title(
        ax,
        f"Damping weight of half A against half B over the run, all pairs ({_pooled_note(runs)})",
    )

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- split halves, edge weights
def _source_pairs(run: dict) -> tuple[np.ndarray, np.ndarray]:
    if "src_pair_a" not in run:
        run["src_pair_a"], run["src_pair_b"] = pair_source_halves(run)
    return run["src_pair_a"], run["src_pair_b"]


def _source_pair_name(run: dict, s: int) -> str:
    """'x3 to f12, halves of f7': the outgoing message and the factor whose halves are compared."""
    target = int(run["src_trg"][s])
    source_factor = run["factor_names"][run["fn_orig_idx"][run["src_fn"][s]]]
    return f"{_edge_name(run, target)}, halves of {source_factor}"


def fig_split_halves_edge(runs: list[dict], out: Path) -> None:
    """attention share of half A against half B of the same factor inside one outgoing message."""
    label_a, label_b = half_labels(runs[0]["split_ratio"])
    final_a = np.concatenate(
        [run["attention"][-1, _source_pairs(run)[0], :].mean(axis=1) for run in runs]
    )
    final_b = np.concatenate(
        [run["attention"][-1, _source_pairs(run)[1], :].mean(axis=1) for run in runs]
    )
    lo, hi = min(final_a.min(), final_b.min()), max(final_a.max(), final_b.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    idx = _subsample(final_a.size, MAX_POINTS)
    ax.scatter(final_a[idx], final_b[idx], s=6, alpha=0.4, color=GREEN)
    _identity(ax, lo, hi, "equal shares")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(f"attention share of {label_a}")
    ax.set_ylabel(f"attention share of {label_b}")
    _title(
        ax,
        "Edge weights: attention share of the two halves of a factor inside the same "
        f"outgoing message, last iteration ({final_a.size} pairs)",
    )

    ax = axes[0, 1]
    bins = np.linspace(lo, hi, 50) if hi > lo else 50
    ax.hist(final_a, bins=bins, color=HALF_COLORS[0], alpha=0.6, label=label_a)
    ax.hist(final_b, bins=bins, color=HALF_COLORS[1], alpha=0.6, label=label_b)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("attention share (last iteration)")
    ax.set_ylabel("number of pairs")
    _title(
        ax, "The same shares as histograms (fully overlapping bars = identical halves)"
    )

    ax = axes[1, 0]
    run = runs[0]
    s_a, s_b = _source_pairs(run)
    shares = run["attention"]
    apart = np.abs(shares[:, s_a, :] - shares[:, s_b, :]).max(axis=(0, 2))
    j = int(apart.argmax())
    n_in = int(np.bincount(run["src_trg"])[run["src_trg"][s_a[j]]])
    for h, color in enumerate(HEAD_COLORS[: run["num_heads"]]):
        ax.plot(shares[:, s_a[j], h], color=color, lw=1.2, label=f"head {h}, half A")
        ax.plot(
            shares[:, s_b[j], h],
            color=color,
            lw=1.2,
            ls="--",
            label=f"head {h}, half B",
        )
    ax.axhline(1.0 / n_in, color="black", lw=0.8, ls="--", label=f"uniform = 1/{n_in}")
    update_guides(ax, run["update_interval"], run["n_iter"])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("attention share")
    _title(
        ax,
        f"The pair whose shares drifted apart the most ({run['label']}, "
        f"{_source_pair_name(run, s_a[j])}; solid = half A, dashed = half B)",
    )

    ax = axes[1, 1]
    # shares depend on how many neighbors the message has, so pool only messages
    # with the most common neighbor count
    group_size = {id(r): np.bincount(r["src_trg"])[r["src_trg"]] for r in runs}
    pair_sizes = np.concatenate([group_size[id(r)][_source_pairs(r)[0]] for r in runs])
    n_in = int(np.bincount(pair_sizes).argmax())
    for half, color, label in (
        (0, HALF_COLORS[0], label_a),
        (1, HALF_COLORS[1], label_b),
    ):
        selected = {
            id(r): _source_pairs(r)[half][
                group_size[id(r)][_source_pairs(r)[0]] == n_in
            ]
            for r in runs
        }
        t, lo_h, mid_h, hi_h = pooled_over_iterations(
            [r for r in runs if selected[id(r)].size],
            lambda r, half=half: r["attention"][:, selected[id(r)], :].mean(axis=2),
        )
        ax.fill_between(t, lo_h, hi_h, color=color, alpha=0.2, lw=0)
        ax.plot(
            t,
            mid_h,
            color=color,
            lw=1.5,
            label=f"{label}: median (band = lowest to highest)",
        )
    ax.axhline(1.0 / n_in, color="black", lw=0.8, ls="--", label=f"uniform = 1/{n_in}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("attention share")
    _title(
        ax,
        f"Share of half A against half B over the run, messages with {n_in} incoming "
        f"neighbors ({int((pair_sizes == n_in).sum())} of {pair_sizes.size} pairs, {_pooled_note(runs)})",
    )

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- attention
def fig_attention(runs: list[dict], out: Path) -> None:
    share = np.concatenate([run["attention"][-1].mean(axis=1) for run in runs])
    uniform = np.concatenate([uniform_share(run) for run in runs])
    twin = np.concatenate([twin_mask(run) for run in runs])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    idx = _subsample(share.size, MAX_POINTS)
    ax.scatter(uniform[idx], share[idx], s=6, alpha=0.3, color=GREEN)
    _identity(ax, 0.0, max(uniform.max(), share.max()), "learned = uniform")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("share if attention were uniform (1 / number of incoming neighbors)")
    ax.set_ylabel("learned attention share (last iteration)")
    _title(
        ax,
        f"Attention share of each incoming neighbor against uniform ({share.size} neighbor rows)",
    )

    ax = axes[0, 1]
    bins = np.linspace(0.0, share.max(), 50)
    ax.hist(share[~twin], bins=bins, color=GREEN, alpha=0.6, label="the other factors")
    ax.hist(
        share[twin],
        bins=bins,
        color=VERMILION,
        alpha=0.6,
        label="the twin half of the same factor",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("attention share (last iteration)")
    ax.set_ylabel("number of neighbor rows")
    _title(ax, "Share given to the twin half compared with the other factors")

    ax = axes[1, 0]
    run = runs[0]
    counts = np.bincount(run["src_trg"], minlength=run["damped"].shape[1])
    target = int(np.argmax(counts))
    rows = np.nonzero(run["src_trg"] == target)[0]
    shares = run["attention"][:, rows, :].mean(axis=2)
    is_twin = twin_mask(run)[rows]
    for j in range(len(rows)):
        ax.plot(
            shares[:, j],
            color=VERMILION if is_twin[j] else GREEN,
            lw=1.2 if is_twin[j] else 0.7,
            alpha=0.9 if is_twin[j] else 0.5,
        )
    ax.axhline(
        1.0 / len(rows),
        color="black",
        lw=0.8,
        ls="--",
        label=f"uniform = 1/{len(rows)}",
    )
    update_guides(ax, run["update_interval"], run["n_iter"])
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("attention share")
    _title(
        ax,
        f"All {len(rows)} incoming shares of one edge ({run['label']}, {_edge_name(run, target)}; red = twin half)",
    )

    ax = axes[1, 1]
    # the uniform share is 1/(number of incoming neighbors), so only edges with the
    # same neighbor count are comparable; take the most common count among twin rows
    group_size = {id(r): np.bincount(r["src_trg"])[r["src_trg"]] for r in runs}
    twin_sizes = np.concatenate([group_size[id(r)][twin_mask(r)] for r in runs])
    n_in = int(np.bincount(twin_sizes).argmax())
    selected = {id(r): twin_mask(r) & (group_size[id(r)] == n_in) for r in runs}
    t, lo, mid, hi = pooled_over_iterations(
        [r for r in runs if selected[id(r)].any()],
        lambda r: r["attention"][:, selected[id(r)], :].mean(axis=2),
    )
    ax.fill_between(
        t, lo, hi, color=VERMILION, alpha=0.2, lw=0, label="lowest to highest"
    )
    ax.plot(t, mid, color=VERMILION, lw=1.5, label="median")
    ax.axhline(1.0 / n_in, color="black", lw=0.8, ls="--", label=f"uniform = 1/{n_in}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("attention share of the twin half")
    _title(
        ax,
        f"Share given to the twin half over the run, edges with {n_in} incoming "
        f"neighbors ({int(twin_sizes.size)} twin rows in total, {_pooled_note(runs)})",
    )

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- cost
def _cost_panel(ax, run: dict, color: str, label: str | None = None) -> None:
    ax.plot(run["costs"], color=color, lw=1.2, label=label)
    t_fix = first_fixed_iteration(run)
    if t_fix is not None:
        ax.axvline(
            t_fix,
            color=color,
            lw=1.0,
            ls=":",
            label=f"assignment fixed from iteration {t_fix}",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost of the current assignment")


def fig_cost_small(runs: list[dict], out: Path) -> None:
    n_iter = np.array([run["n_iter"] for run in runs])
    settled = np.array([run["converged"] for run in runs])
    best = np.array([run["costs"].min() for run in runs])
    final = np.array([run["costs"][-1] for run in runs])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    _cost_panel(ax, runs[0], BLUE)
    ax.legend(frameon=False, fontsize=8)
    _title(ax, f"Cost over the run ({runs[0]['label']})")

    ax = axes[0, 1]
    ax.hist(
        [n_iter[settled], n_iter[~settled]],
        bins=30,
        stacked=True,
        color=[BLUE, VERMILION],
        label=["assignment settled", "never settled (hit the cap)"],
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iterations run")
    ax.set_ylabel("number of seeds")
    _title(
        ax,
        f"How long each seed ran ({settled.sum()} of {len(runs)} settled; a run stops "
        "25 iterations after the assignment last changed)",
    )

    ax = axes[1, 0]
    for run in runs:
        ax.plot(
            run["costs"],
            color=BLUE if run["converged"] else VERMILION,
            lw=0.6,
            alpha=0.4,
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost of the current assignment")
    _title(ax, f"Cost over the run, all {len(runs)} seeds (red = never settled)")

    ax = axes[1, 1]
    ax.scatter(best[settled], final[settled], s=18, color=BLUE, label="settled")
    ax.scatter(
        best[~settled], final[~settled], s=18, color=VERMILION, label="never settled"
    )
    _identity(
        ax,
        min(best.min(), final.min()),
        max(best.max(), final.max()),
        "kept its best solution",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("best cost seen during the run")
    ax.set_ylabel("cost at the last iteration")
    _title(ax, "Did each seed keep the best solution it found?")

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


def fig_cost_single(run: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    _cost_panel(ax, run, BLUE)
    ax.legend(frameon=False, fontsize=8)
    _title(
        ax,
        f"Cost over the run ({run['label']}, split {split_label(run['split_ratio'])})",
    )

    ax = axes[1]
    ax.plot(run["changes_per_iter"], color=BLUE, lw=1.0, drawstyle="steps-mid")
    update_guides(ax, run["update_interval"], run["n_iter"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("variables that changed value")
    _title(
        ax,
        f"How many of the {run['assignments'].shape[1]} variables still change each iteration",
    )

    for ax in axes:
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- structure (small only)
def fig_structure_small(structure_csv: Path, stats_csv: Path, out: Path) -> None:
    structure = pd.read_csv(structure_csv)
    stats = pd.read_csv(stats_csv).set_index("feature")
    ylabel = "pair damping weight (mean of both halves)"

    def annotate(ax, feature: str) -> None:
        rho, p = stats.loc[feature, "spearman_rho"], stats.loc[feature, "p_value"]
        ax.text(
            0.02,
            0.98,
            f"rank correlation {rho:.3f} (p = {p:.2g})",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    degrees = sorted(structure["var_degree"].unique())
    ax.boxplot(
        [structure.loc[structure["var_degree"] == d, "lam_final"] for d in degrees],
        tick_labels=[str(d) for d in degrees],
    )
    annotate(ax, "var_degree")
    ax.set_xlabel("number of neighbors of the variable")
    ax.set_ylabel(ylabel)
    _title(ax, "Damping weight against how connected the variable is")

    ax = axes[0, 1]
    ax.boxplot(
        [structure.loc[structure["edge_on_cycle"] == v, "lam_final"] for v in (0, 1)],
        tick_labels=["factor is a bridge", "factor lies on a cycle"],
    )
    annotate(ax, "edge_on_cycle")
    ax.set_ylabel(ylabel)
    _title(ax, "Damping weight for bridge factors against factors on cycles")

    ax = axes[1, 0]
    ax.scatter(structure["ct_std"], structure["lam_final"], s=8, alpha=0.4, color=BLUE)
    annotate(ax, "ct_std")
    ax.set_xlabel("spread of the factor's cost table (standard deviation)")
    ax.set_ylabel(ylabel)
    _title(ax, "Damping weight against how varied the factor's costs are")

    ax = axes[1, 1]
    ax.scatter(
        structure["ct_range"], structure["lam_final"], s=8, alpha=0.4, color=BLUE
    )
    annotate(ax, "ct_range")
    ax.set_xlabel("range of the factor's cost table (max minus min)")
    ax.set_ylabel(ylabel)
    _title(ax, "Damping weight against the factor's cost range")

    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------- split comparison
def _split_legend_label(split_ratio: float) -> str:
    return f"{split_label(split_ratio)} split"


def _compare_weight_panels(axes_pair, runs_by_split: dict[float, list[dict]]) -> None:
    """histogram of final damping weights + half A vs half B scatter, one color per split."""
    ax_hist, ax_scatter = axes_pair
    finals = {
        s: np.concatenate([run["lam"][-1] for run in runs])
        for s, runs in runs_by_split.items()
    }
    lo = min(v.min() for v in finals.values())
    hi = max(v.max() for v in finals.values())
    bins = np.linspace(lo, hi, 50) if hi > lo else 50
    for s, values in finals.items():
        ax_hist.hist(
            values,
            bins=bins,
            color=SPLIT_COLORS[s],
            alpha=0.55,
            label=_split_legend_label(s),
        )
    ax_hist.axvline(START_WEIGHT, color="black", lw=0.8, ls="--")
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.set_xlabel(
        "damping weight at the last iteration (weight on the previous message)"
    )
    ax_hist.set_ylabel("number of edges")
    _title(
        ax_hist, "Damping weight of every edge at the last iteration, under each split"
    )

    lo_s, hi_s = np.inf, -np.inf
    for s, runs in runs_by_split.items():
        a = np.concatenate([run["lam"][-1, _pairs(run)[0]] for run in runs])
        b = np.concatenate([run["lam"][-1, _pairs(run)[1]] for run in runs])
        idx = _subsample(a.size, MAX_POINTS // 2)
        ax_scatter.scatter(
            a[idx],
            b[idx],
            s=6,
            alpha=0.4,
            color=SPLIT_COLORS[s],
            label=_split_legend_label(s),
        )
        lo_s, hi_s = min(lo_s, a.min(), b.min()), max(hi_s, a.max(), b.max())
    _identity(ax_scatter, lo_s, hi_s, "equal weights")
    ax_scatter.legend(frameon=False, fontsize=8)
    ax_scatter.set_xlabel(
        "damping weight on the edge to half A (the bigger share of the cost)"
    )
    ax_scatter.set_ylabel("damping weight on the edge to half B (the smaller share)")
    _title(
        ax_scatter,
        "Damping weight of the two edges of each split factor at the last iteration, under each split",
    )


def fig_compare_splits_small(runs_by_split: dict[float, list[dict]], out: Path) -> None:
    splits = sorted(runs_by_split)
    seeds = [[run["seed"] for run in runs_by_split[s]] for s in splits]
    assert all(
        sd == seeds[0] for sd in seeds
    ), "the two splits must cover the same seeds"
    final = {
        s: np.array([run["costs"][-1] for run in runs_by_split[s]]) for s in splits
    }
    n_iter = {s: np.array([run["n_iter"] for run in runs_by_split[s]]) for s in splits}
    settled = {
        s: np.array([run["converged"] for run in runs_by_split[s]]) for s in splits
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    x, y = final[splits[0]], final[splits[1]]
    both = settled[splits[0]] & settled[splits[1]]
    ax.scatter(x[both], y[both], s=18, color=BLUE, label="both settled")
    ax.scatter(
        x[~both],
        y[~both],
        s=22,
        facecolors="none",
        edgecolors=VERMILION,
        label="one of them never settled",
    )
    _identity(ax, min(x.min(), y.min()), max(x.max(), y.max()), "same cost")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(f"cost at the last iteration, {_split_legend_label(splits[0])}")
    ax.set_ylabel(f"cost at the last iteration, {_split_legend_label(splits[1])}")
    _title(ax, f"Same {len(x)} problems: final cost under each split")

    ax = axes[0, 1]
    for s in splits:
        ax.plot(
            np.sort(n_iter[s]),
            color=SPLIT_COLORS[s],
            lw=1.5,
            drawstyle="steps-post",
            label=_split_legend_label(s),
        )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("seeds, sorted from shortest to longest run")
    ax.set_ylabel("iterations run (cap 1000)")
    _title(ax, "How long the runs took under each split")

    _compare_weight_panels((axes[1, 0], axes[1, 1]), runs_by_split)
    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)


def fig_compare_splits_single(runs_by_split: dict[float, dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    for s, run in sorted(runs_by_split.items()):
        _cost_panel(ax, run, SPLIT_COLORS[s], _split_legend_label(s))
    ax.legend(frameon=False, fontsize=8)
    _title(
        ax,
        f"Cost over the run on the same graph ({next(iter(runs_by_split.values()))['label']})",
    )

    ax = axes[0, 1]
    for s, run in sorted(runs_by_split.items()):
        ax.plot(
            run["changes_per_iter"],
            color=SPLIT_COLORS[s],
            lw=1.0,
            alpha=0.8,
            drawstyle="steps-mid",
            label=_split_legend_label(s),
        )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("variables that changed value")
    _title(ax, "How many variables still change each iteration, under each split")

    _compare_weight_panels(
        (axes[1, 0], axes[1, 1]), {s: [run] for s, run in runs_by_split.items()}
    )
    for ax in axes.ravel():
        plain_axes(ax)
    fig.tight_layout()
    save(fig, out)
