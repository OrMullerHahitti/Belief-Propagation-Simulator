"""Standalone CLI to reproduce Figures 5a, 5b, and 8 from Zivan et al.

Usage::

    uv run python experiments/reproduce_figure.py --figure 5a
    uv run python experiments/reproduce_figure.py --figure 8 --n-examples 20
    uv run python experiments/reproduce_figure.py --figure 5b --output /tmp/fig5b.png
    uv run python experiments/reproduce_figure.py --figure 5a --cycle-size 6

    # batch B&W mode: generates all 12 PNGs (3 figures x 4 cycle sizes)
    uv run python experiments/reproduce_figure.py --batch-bw
    uv run python experiments/reproduce_figure.py --batch-bw --n-examples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# sys.path guard so the script works when run directly from repo root
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.utils.plot_helpers import remove_frame as _remove_frame  # noqa: E402
from experiments.utils.fig58_repro import (  # noqa: E402
    CASE_CONSISTENT_NO_TAIL,
    CASE_CONSISTENT_WITH_TAIL,
    CASE_INCONSISTENT_NO_TAIL,
    construct_consistent_no_tail_examples,
    construct_inconsistent_no_tail_examples,
    derive_tail_examples,
    run_experiment_examples_cycle,
)

# ── figure-specific config ────────────────────────────────────────────────────

FIGURE_CONFIGS = {
    "5a": {
        "case_name": CASE_CONSISTENT_NO_TAIL,
        "seed_start": 42,
        "max_iter": 100,
        "key_order": ["x1_route", "x2_route", "x3_route"],
        "labels": ["v1", "v2", "v3"],
        "styles": ["-", "-", "-"],
        "colors": ["tab:blue", "tab:orange", "tab:green"],
        "title": "Figure 5(a): consistent, no tail",
    },
    "5b": {
        "case_name": CASE_CONSISTENT_WITH_TAIL,
        "seed_start": 42,
        "max_iter": 100,
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

    # for inconsistent cases, show all variables to reveal cycle-size effects;
    # for consistent cases, cap at display_vars (traces are visually redundant)
    if case_name == CASE_INCONSISTENT_NO_TAIL:
        n_show = cycle_size
    else:
        n_show = min(display_vars, cycle_size)

    if case_name in {CASE_CONSISTENT_NO_TAIL, CASE_CONSISTENT_WITH_TAIL}:
        key_order = [f"x{i+1}_route" for i in range(n_show)]
        labels = [f"v{i+1}" for i in range(n_show)]
    else:
        key_order = [f"x{i+1}_v{v}" for v in range(2) for i in range(n_show)]
        labels = [f"v{i+1}-{'a' if v == 0 else 'b'}" for v in range(2) for i in range(n_show)]

    n_lines = len(key_order)
    bw_styles = [_BW_STYLES[i % len(_BW_STYLES)] for i in range(n_lines)]

    return {
        "case_name": case_name,
        "seed_start": base["seed_start"],
        "max_iter": base["max_iter"],
        "key_order": key_order,
        "labels": labels,
        "bw_styles": bw_styles,
    }


# ── plotting ──────────────────────────────────────────────────────────────────


# ── single-example plotting ──────────────────────────────────────────────────


def plot_single_run_bw(ax, run, key_order, labels, bw_styles):
    """Plot a single example's belief traces in black-and-white."""
    n = len(next(iter(run["records"].values())))
    iters = np.arange(n + 1)  # 0 … max_iter (prepend pre-computation state)

    for key, label, (lstyle, lw) in zip(key_order, labels, bw_styles):
        y_raw = np.asarray(run["records"][key], dtype=float)
        # prepend 0 for the natural pre-computation state (all messages are zero vectors)
        y_plot = np.concatenate([[0.0], y_raw])
        ax.plot(iters, y_plot, linestyle=lstyle, linewidth=lw, color="black", label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8, ncol=max(1, len(labels) // 6))
    _remove_frame(ax)


def plot_single_run(ax, run, key_order, labels, styles, colors, title):
    """Plot a single example's belief traces in color."""
    n = len(next(iter(run["records"].values())))
    iters = np.arange(n + 1)  # 0 … max_iter (prepend pre-computation state)

    for key, label, style, color in zip(key_order, labels, styles, colors):
        y_raw = np.asarray(run["records"][key], dtype=float)
        # prepend 0 for the natural pre-computation state (all messages are zero vectors)
        y_plot = np.concatenate([[0.0], y_raw])
        ax.plot(iters, y_plot, linestyle=style, color=color, linewidth=2.2, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _remove_frame(ax)


# ── example selection ────────────────────────────────────────────────────────


def select_representative_run(runs, case_name, *, match_seed=None):
    """Pick one representative example from N runs.

    if *match_seed* is given, prefer the run with that seed (used to align
    5a and 5b so both show the same base graph).
    for consistent_with_tail, select the run with the largest periodic_start
    (most visible tail). for other cases, pick the first run.
    """
    if not runs:
        raise ValueError("no runs to select from")
    if len(runs) == 1:
        return runs[0]

    if match_seed is not None:
        matched = next((r for r in runs if r["seed"] == match_seed), None)
        if matched:
            return matched

    if case_name == CASE_CONSISTENT_WITH_TAIL:
        return max(runs, key=lambda r: r["classification"].get("periodic_start", 0) or 0)

    return runs[0]


# ── dispatch ──────────────────────────────────────────────────────────────────

# cache paired 5a/5b examples so both figures use the same base graphs
_paired_cache: dict[tuple[int, int], tuple[list, list]] = {}


def _get_paired_5a_5b(cycle_size: int, n_examples: int, seed_start: int) -> tuple[list, list]:
    """Return (5a_examples, 5b_examples) from the same base graphs."""
    key = (cycle_size, seed_start)
    if key in _paired_cache:
        return _paired_cache[key]

    # generate extra bases since not all convert to tail
    bases = construct_consistent_no_tail_examples(
        n_examples * 4,
        cycle_size=cycle_size,
        seed_start=seed_start,
    )
    derived = derive_tail_examples(bases, tail_length=2)

    # only keep bases that successfully produced a tail
    valid_seeds = {d["base_seed"] for d in derived}
    paired_5a = [b for b in bases if b["seed"] in valid_seeds][:n_examples]
    paired_5b = [d for d in derived if d["base_seed"] in {b["seed"] for b in paired_5a}]

    _paired_cache[key] = (paired_5a, paired_5b)
    return paired_5a, paired_5b


def _collect_and_run(cfg: dict, cycle_size: int, n_examples: int) -> tuple:
    """Return (runs, examples) for the given figure config."""
    case_name = cfg["case_name"]
    seed_start = cfg["seed_start"]
    max_iter = cfg["max_iter"]

    if case_name in {CASE_CONSISTENT_NO_TAIL, CASE_CONSISTENT_WITH_TAIL}:
        # 5a and 5b use the same base graphs
        paired_5a, paired_5b = _get_paired_5a_5b(cycle_size, n_examples, seed_start)
        examples = paired_5a if case_name == CASE_CONSISTENT_NO_TAIL else paired_5b
    elif case_name == CASE_INCONSISTENT_NO_TAIL:
        # fig 8: one off-diagonal factor + rest diagonal
        examples = construct_inconsistent_no_tail_examples(
            n_examples,
            cycle_size=cycle_size,
            seed_start=seed_start,
        )
    else:
        raise ValueError(f"Unknown case: {case_name}")

    runs = run_experiment_examples_cycle(
        examples,
        case_name=case_name,
        max_iter=max_iter,
        normalize_messages=False,
        subtract_initial=False,
    )

    return runs, examples


def _save_cost_tables(example: dict, out_path: Path) -> None:
    """Save the cost tables used to generate a plot alongside it."""
    tables = [ct.tolist() for ct in example["cost_tables"]]
    data = {
        "seed": int(example["seed"]),
        "cycle_size": int(example.get("cycle_size", len(tables))),
        "case_name": example["case_name"],
        "cost_tables": tables,
    }
    if "base_seed" in example:
        data["base_seed"] = int(example["base_seed"])
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved cost tables → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce paper figures (5a, 5b, 8) via representative belief traces.",
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
    p.add_argument("--batch-bw", action="store_true",
                   help="generate all 12 B&W PNGs (3 figures x 4 cycle sizes)")
    return p


BATCH_FIGURES = ["5a", "5b", "8"]
BATCH_CYCLE_SIZES = [3, 6, 9, 12]


def _plot_and_save_bw(out_dir, figure_id, cycle_size, cfg, representative, examples):
    """Save a single B&W plot + its cost-table JSON."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_single_run_bw(
        ax,
        run=representative,
        key_order=cfg["key_order"],
        labels=cfg["labels"],
        bw_styles=cfg["bw_styles"],
    )
    plt.tight_layout()

    fname = f"fig{figure_id}_cycle{cycle_size}.pdf"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[batch-bw] saved {out_path}")

    ct_path = out_dir / f"fig{figure_id}_cycle{cycle_size}_cost_tables.json"
    _save_cost_tables(examples[representative["example_index"]], ct_path)


def _run_batch_bw(n_examples: int) -> None:
    """Generate all 12 B&W plots.

    5a and 5b are processed together per cycle size so they share the same
    representative base graph: 5b picks the best tail, then 5a matches that seed.
    """
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(BATCH_FIGURES) * len(BATCH_CYCLE_SIZES)
    count = 0

    for cycle_size in BATCH_CYCLE_SIZES:
        # --- 5b first: pick the representative with the best tail ---
        count += 1
        print(f"\n[batch-bw] ({count}/{total}) figure=5b, cycle_size={cycle_size}, "
              f"n_examples={n_examples}")
        cfg_5b = _build_bw_config("5b", cycle_size)
        runs_5b, examples_5b = _collect_and_run(cfg_5b, cycle_size=cycle_size, n_examples=n_examples)
        rep_5b = select_representative_run(runs_5b, cfg_5b["case_name"])
        print(f"[batch-bw] selected 1 of {len(runs_5b)} examples, "
              f"{cfg_5b['max_iter']} iterations each")
        _plot_and_save_bw(out_dir, "5b", cycle_size, cfg_5b, rep_5b, examples_5b)

        # --- 5a: match the same seed so both show the same base graph ---
        count += 1
        print(f"\n[batch-bw] ({count}/{total}) figure=5a, cycle_size={cycle_size}, "
              f"n_examples={n_examples}")
        cfg_5a = _build_bw_config("5a", cycle_size)
        runs_5a, examples_5a = _collect_and_run(cfg_5a, cycle_size=cycle_size, n_examples=n_examples)
        rep_5a = select_representative_run(runs_5a, cfg_5a["case_name"], match_seed=rep_5b["seed"])
        print(f"[batch-bw] selected 1 of {len(runs_5a)} examples (matched seed={rep_5b['seed']}), "
              f"{cfg_5a['max_iter']} iterations each")
        _plot_and_save_bw(out_dir, "5a", cycle_size, cfg_5a, rep_5a, examples_5a)

        # --- fig 8: independent (skip if generation is not yet supported) ---
        if "8" in BATCH_FIGURES:
            count += 1
            print(f"\n[batch-bw] ({count}/{total}) figure=8, cycle_size={cycle_size}, "
                  f"n_examples={n_examples}")
            try:
                cfg_8 = _build_bw_config("8", cycle_size)
                runs_8, examples_8 = _collect_and_run(cfg_8, cycle_size=cycle_size, n_examples=n_examples)
                rep_8 = select_representative_run(runs_8, cfg_8["case_name"])
                print(f"[batch-bw] selected 1 of {len(runs_8)} examples, "
                      f"{cfg_8['max_iter']} iterations each")
                _plot_and_save_bw(out_dir, "8", cycle_size, cfg_8, rep_8, examples_8)
            except Exception as exc:
                print(f"[batch-bw] skipping figure=8, cycle_size={cycle_size}: {exc}")

    print(f"\n[batch-bw] done — plots saved to {out_dir}/")


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

    runs, examples = _collect_and_run(cfg, cycle_size=args.cycle_size, n_examples=args.n_examples)
    representative = select_representative_run(runs, cfg["case_name"])
    example = examples[representative["example_index"]]
    print(f"[reproduce_figure] selected 1 of {len(runs)} examples, "
          f"running {cfg['max_iter']} iterations each")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_single_run(
        ax,
        run=representative,
        key_order=cfg["key_order"],
        labels=cfg["labels"],
        styles=cfg["styles"],
        colors=cfg["colors"],
        title=f"{cfg['title']} (1 of {len(runs)})",
    )
    plt.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[reproduce_figure] saved to {out_path}")
        ct_path = out_path.with_name(out_path.stem + "_cost_tables.json")
        _save_cost_tables(example, ct_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
