"""Standalone CLI to reproduce Figures 5a, 5b, and 8 from Zivan et al.

Usage::

    uv run python expiriments/reproduce_figure.py --figure 5a
    uv run python expiriments/reproduce_figure.py --figure 8 --n-examples 20
    uv run python expiriments/reproduce_figure.py --figure 5b --output /tmp/fig5b.png
    uv run python expiriments/reproduce_figure.py --figure 5a --cycle-size 6

    # batch B&W mode: generates all 12 PNGs (3 figures x 4 cycle sizes)
    uv run python expiriments/reproduce_figure.py --batch-bw
    uv run python expiriments/reproduce_figure.py --batch-bw --n-examples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# sys.path guard so the script works when run directly from repo root
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from expiriments.utils.fig58_repro import (  # noqa: E402
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    GEN_MOTIF_REPEAT,
    GEN_RANDOM_FULL,
    aggregate_example_records,
    find_cases,
    find_cases_cycle,
    run_experiment_examples,
    run_experiment_examples_cycle,
)

# ── figure-specific config ────────────────────────────────────────────────────

FIGURE_CONFIGS = {
    "5a": {
        "case_name": CASE_CONSISTENT_NO_TAIL,
        "seed_start": 42,
        "max_iter": 50,
        "key_order": ["x1_route", "x2_route", "x3_route"],
        "labels": ["v1", "v2", "v3"],
        "styles": ["-", "-", "-"],
        "colors": ["tab:blue", "tab:orange", "tab:green"],
        "title": "Figure 5(a): consistent, no tail",
    },
    "5b": {
        "case_name": CASE_CONSISTENT_WITH_TAIL,
        "seed_start": 42,
        "max_iter": 50,
        "key_order": ["x1_route", "x2_route", "x3_route"],
        "labels": ["v1", "v2", "v3"],
        "styles": ["-", "-", "-"],
        "colors": ["tab:blue", "tab:orange", "tab:green"],
        "title": "Figure 5(b): consistent, with tail",
    },
    "8": {
        "case_name": CASE_INCONSISTENT_NO_TAIL,
        "seed_start": 42,
        "max_iter": 70,
        "key_order": ["x1_v0", "x2_v0", "x3_v0", "x1_v1", "x2_v1", "x3_v1"],
        "labels": ["v1-a", "v2-a", "v3-a", "v1-b", "v2-b", "v3-b"],
        "styles": ["-", "-", "-", "--", "--", "--"],
        "colors": ["tab:blue", "tab:orange", "tab:green", "tab:blue", "tab:orange", "tab:green"],
        "title": "Figure 8: inconsistent, no tail",
    },
}

# ── B&W line style palette ───────────────────────────────────────────────────
# each entry is (linestyle, linewidth) — enough combos for up to 24 lines

_BW_STYLES = [
    ("-", 2.0),
    ("--", 2.0),
    ("-.", 2.0),
    (":", 2.0),
    ((0, (5, 1)), 2.0),
    ((0, (1, 1)), 2.0),
    ((0, (3, 1, 1, 1)), 2.0),
    ((0, (5, 1, 1, 1, 1, 1)), 2.0),
    ("-", 1.0),
    ("--", 1.0),
    ("-.", 1.0),
    (":", 1.0),
    ((0, (5, 1)), 1.0),
    ((0, (1, 1)), 1.0),
    ((0, (3, 1, 1, 1)), 1.0),
    ((0, (5, 1, 1, 1, 1, 1)), 1.0),
    ((0, (8, 2)), 2.0),
    ((0, (8, 2)), 1.0),
    ((0, (3, 2, 1, 2)), 2.0),
    ((0, (3, 2, 1, 2)), 1.0),
    ((0, (1, 2)), 2.0),
    ((0, (1, 2)), 1.0),
    ((0, (5, 2, 1, 2, 1, 2)), 2.0),
    ((0, (5, 2, 1, 2, 1, 2)), 1.0),
]


def _build_bw_config(figure_id: str, cycle_size: int, display_vars: int = 3) -> dict:
    """Build a figure config with dynamic key_order/labels for any cycle size.

    Only the first ``display_vars`` variables are included in the plot keys,
    even though BP runs on the full cycle graph.
    """
    base = FIGURE_CONFIGS[figure_id]
    case_name = base["case_name"]

    # always show only the first display_vars variables
    n_show = min(display_vars, cycle_size)

    if case_name in {CASE_CONSISTENT_NO_TAIL, CASE_CONSISTENT_WITH_TAIL}:
        key_order = [f"x{i+1}_route" for i in range(n_show)]
        labels = [f"v{i+1}" for i in range(n_show)]
    else:
        key_order = [f"x{i+1}_v{v}" for v in range(2) for i in range(n_show)]
        labels = [f"v{i+1}-{'a' if v == 0 else 'b'}" for v in range(2) for i in range(n_show)]

    n_lines = len(key_order)
    bw_styles = [_BW_STYLES[i % len(_BW_STYLES)] for i in range(n_lines)]

    gen_strategy = GEN_RANDOM_FULL
    if case_name == CASE_INCONSISTENT_NO_TAIL:
        gen_strategy = GEN_MOTIF_REPEAT

    return {
        "case_name": case_name,
        "seed_start": base["seed_start"],
        "max_iter": base["max_iter"],
        "key_order": key_order,
        "labels": labels,
        "bw_styles": bw_styles,
        "generation_strategy": gen_strategy,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_aggregate_bw(ax, agg, key_order, labels, bw_styles):
    """Plot averaged belief traces in black-and-white with distinct dash patterns."""
    iters = np.arange(len(next(iter(agg["mean_records"].values()))))

    for key, label, (lstyle, lw) in zip(key_order, labels, bw_styles):
        mean = np.asarray(agg["mean_records"][key], dtype=float)
        y_plot = mean.copy()
        if y_plot.size:
            y_plot[0] = 0.0
        ax.plot(iters, y_plot, linestyle=lstyle, linewidth=lw, color="black", label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8, ncol=max(1, len(labels) // 6))


def plot_aggregate(ax, runs, agg, key_order, labels, styles, colors, title, show_band=True):
    """Plot averaged belief traces with faint individual lines.

    copied verbatim from notebook cell 79126b84.
    """
    iters = np.arange(len(next(iter(agg["mean_records"].values()))))

    # faint individual traces for context
    for run in runs:
        for key, style, color in zip(key_order, styles, colors):
            y_raw = np.asarray(run["records"][key], dtype=float)
            y_plot = y_raw.copy()
            if y_plot.size:
                y_plot[0] = 0.0  # visualization-only anchor
            ax.plot(iters, y_plot, linestyle=style, color=color, alpha=0.12, linewidth=1.0)

    # bold mean + optional std band
    for key, label, style, color in zip(key_order, labels, styles, colors):
        mean = np.asarray(agg["mean_records"][key], dtype=float)
        std = np.asarray(agg["std_records"][key], dtype=float)

        y_plot = mean.copy()
        if y_plot.size:
            y_plot[0] = 0.0  # visualization-only anchor
        ax.plot(iters, y_plot, linestyle=style, color=color, linewidth=2.2, label=label)

        if show_band:
            upper = mean + std
            lower = mean - std
            if upper.size:
                upper = upper.copy(); upper[0] = 0.0
                lower = lower.copy(); lower[0] = 0.0
            ax.fill_between(iters, lower, upper, color=color, alpha=0.14)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)


# ── dispatch ──────────────────────────────────────────────────────────────────

def _collect_and_run(cfg: dict, cycle_size: int, n_examples: int) -> tuple:
    """Return (runs, agg) for the given figure config."""
    case_name = cfg["case_name"]
    seed_start = cfg["seed_start"]
    max_iter = cfg["max_iter"]
    key_order = cfg["key_order"]
    gen_strategy = cfg.get("generation_strategy", GEN_RANDOM_FULL)

    if cycle_size == 3:
        # exact notebook path — uses 3-variable helpers
        examples = find_cases(
            case_name,
            n_examples=n_examples,
            seed_start=seed_start,
        )
        runs = run_experiment_examples(
            examples,
            case_name=case_name,
            max_iter=max_iter,
            normalize_messages=False,
            subtract_initial=False,
        )
    else:
        # generalised cycle path
        examples = find_cases_cycle(
            case_name,
            n_examples=n_examples,
            cycle_size=cycle_size,
            seed_start=seed_start,
            generation_strategy=gen_strategy,
        )
        runs = run_experiment_examples_cycle(
            examples,
            case_name=case_name,
            max_iter=max_iter,
            normalize_messages=False,
            subtract_initial=False,
        )

    agg = aggregate_example_records(runs, key_order)
    return runs, agg


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce paper figures (5a, 5b, 8) via averaged belief traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--figure", choices=["5a", "5b", "8"],
                   help="which figure to reproduce (single-figure mode)")
    p.add_argument("--cycle-size", type=int, default=3,
                   help="cycle length; 3 matches paper exactly, >3 uses generalised helpers")
    p.add_argument("--n-examples", type=int, default=10,
                   help="number of random instances to average over")
    p.add_argument("--output", type=str, default=None,
                   help="save plot to this path (PNG/PDF); omit to display interactively")
    p.add_argument("--no-band", action="store_true",
                   help="suppress ±1 std shading around the mean")
    p.add_argument("--batch-bw", action="store_true",
                   help="generate all 12 B&W PNGs (3 figures x 4 cycle sizes)")
    return p


BATCH_FIGURES = ["5a", "5b", "8"]
BATCH_CYCLE_SIZES = [3, 6, 9, 12]


def _run_batch_bw(n_examples: int) -> None:
    """Generate all 12 B&W plots."""
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(BATCH_FIGURES) * len(BATCH_CYCLE_SIZES)
    count = 0

    for figure_id in BATCH_FIGURES:
        for cycle_size in BATCH_CYCLE_SIZES:
            count += 1
            print(f"\n[batch-bw] ({count}/{total}) figure={figure_id}, cycle_size={cycle_size}, "
                  f"n_examples={n_examples}")

            cfg = _build_bw_config(figure_id, cycle_size)
            runs, agg = _collect_and_run(cfg, cycle_size=cycle_size, n_examples=n_examples)
            print(f"[batch-bw] collected {agg['n_examples']} examples, "
                  f"running {cfg['max_iter']} iterations each")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_aggregate_bw(
                ax,
                agg=agg,
                key_order=cfg["key_order"],
                labels=cfg["labels"],
                bw_styles=cfg["bw_styles"],
            )
            plt.tight_layout()

            fname = f"fig{figure_id}_cycle{cycle_size}.png"
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[batch-bw] saved {out_path}")

    print(f"\n[batch-bw] done — {total} plots saved to {out_dir}/")


def main() -> None:
    args = build_parser().parse_args()

    if args.batch_bw:
        _run_batch_bw(n_examples=args.n_examples)
        return

    if not args.figure:
        print("error: --figure is required (or use --batch-bw)", file=sys.stderr)
        sys.exit(1)

    cfg = FIGURE_CONFIGS[args.figure]

    print(f"[reproduce_figure] figure={args.figure}, cycle_size={args.cycle_size}, "
          f"n_examples={args.n_examples}")

    runs, agg = _collect_and_run(cfg, cycle_size=args.cycle_size, n_examples=args.n_examples)
    print(f"[reproduce_figure] collected {agg['n_examples']} examples, "
          f"running {cfg['max_iter']} iterations each")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_aggregate(
        ax,
        runs=runs,
        agg=agg,
        key_order=cfg["key_order"],
        labels=cfg["labels"],
        styles=cfg["styles"],
        colors=cfg["colors"],
        title=f"{cfg['title']} (mean over {agg['n_examples']})",
        show_band=not args.no_band,
    )
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[reproduce_figure] saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
