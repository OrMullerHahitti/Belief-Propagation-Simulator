"""Shared loading, pairing and style for the DABP weight figures.

Both data sources record the same per-iteration tensors — ``damped
[n_iter, T, 2, H]`` (per target edge and head: weight on the new message,
weight on the previous message) and ``attention [n_iter, S, H]`` (share of
each incoming source inside its target edge's group) — together with the
index provenance returned by ``DABPEngine.weight_metadata()``. The two
loaders below normalize either file into one plain dict so every figure
function works unchanged on a 10-agent seed and on the 50-node run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PLOTS_ROOT = Path(__file__).resolve().parents[1]
SMALL_DATA = REPO / "experiments" / "dabp_weights"
BIGGER_DATA = REPO / "experiments" / "dabp_node_dynamics" / "outputs"

# okabe-ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#7f7f7f"
HEAD_COLORS = [BLUE, ORANGE, GREEN, VERMILION]
HALF_COLORS = (BLUE, VERMILION)
SPLIT_COLORS = {0.5: BLUE, 0.95: VERMILION}
START_WEIGHT = 0.5  # every damping weight starts at the softmax midpoint


def remove_frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plain_axes(ax) -> None:
    """no frame and no '1e-5+5e-1' offset notation: weights print as they are."""
    remove_frame(ax)
    # category axes (boxplots) carry a fixed formatter that has no offset setting
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        if isinstance(axis.get_major_formatter(), ScalarFormatter):
            ax.ticklabel_format(useOffset=False, style="plain", axis=name)
    # weight axes span ~1e-4, so their tick labels are 7 characters wide; five fit
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"wrote {shown}", flush=True)


def split_label(split_ratio: float) -> str:
    return f"{split_ratio:g}/{1 - split_ratio:g}"


def split_dirname(split_ratio: float) -> str:
    return f"split_{split_ratio:g}_{1 - split_ratio:g}"


def half_labels(split_ratio: float) -> tuple[str, str]:
    """what each of the two split halves of an original factor carries."""
    return (
        f"half A ({split_ratio:g} of the cost)",
        f"half B ({1 - split_ratio:g} of the cost)",
    )


def _orig_factor_index(fn_factor_names: list[str]) -> tuple[list[str], np.ndarray]:
    """order-preserving unique factor names + original index per function node."""
    names: list[str] = []
    idx: list[int] = []
    for fname in fn_factor_names:
        if fname not in names:
            names.append(fname)
        idx.append(names.index(fname))
    return names, np.asarray(idx, dtype=np.int32)


def _finish(run: dict) -> dict:
    run["n_iter"] = int(run["damped"].shape[0])
    run["num_heads"] = int(run["damped"].shape[3])
    # weight on the previous message, head mean = the damping actually applied
    run["lam"] = run["damped"][:, :, 1, :].mean(axis=2)
    assignments = run["assignments"]
    changes = np.zeros(run["n_iter"], dtype=int)
    changes[1:] = (assignments[1:] != assignments[:-1]).sum(axis=1)
    run["changes_per_iter"] = changes
    return run


def load_small_run(path: Path) -> dict:
    """one seed of experiments/dabp_weights (data/ or data_asym/)."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "label": f"seed {int(z['seed'])}",
            "damped": z["damped"],
            "attention": z["attention"],
            "costs": z["costs"],
            "assignments": z["assignments"].astype(np.int32),
            "trg_var_idx": z["trg_var_idx"],
            "trg_fn": z["trg_fn"],
            "fn_orig_idx": z["fn_orig_idx"],
            "fn_half": z["fn_half"],
            "src_fn": z["src_fn"],
            "src_trg": z["src_trg"],
            "var_names": [str(v) for v in z["var_names"]],
            "factor_names": [str(f) for f in z["factor_names"]],
            "split_ratio": float(z["split_ratio"]),
            "update_interval": int(z["update_interval"]),
            "converged": bool(z["converged"]),
            "seed": int(z["seed"]),
        }
    return _finish(run)


def load_bigger_run(path: Path) -> dict:
    """one variant of experiments/dabp_node_dynamics (symmetric.npz / asymmetric.npz)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"]))
        arrays = {k: z[k] for k in ("damped", "attention", "costs", "assignments")}
    var_names = list(meta["ordered_names"])
    name_to_idx = {n: k for k, n in enumerate(var_names)}
    factor_names, fn_orig_idx = _orig_factor_index(meta["fn_factor_names"])
    outcome = meta["outcome"]
    run = {
        "label": f"{meta['settings']['nodes']} nodes, seed {meta['settings']['graph_seed']}",
        **arrays,
        "assignments": arrays["assignments"].astype(np.int32),
        "trg_var_idx": np.array(
            [name_to_idx[v] for v in meta["trg_var_names"]], dtype=np.int32
        ),
        "trg_fn": np.asarray(meta["trg_fn_idxes"], dtype=np.int32),
        "fn_orig_idx": fn_orig_idx,
        "fn_half": np.asarray(meta["fn_half"], dtype=np.int8),
        "src_fn": np.asarray(meta["src_fn_idxes"], dtype=np.int32),
        "src_trg": np.asarray(meta["src_trg_idxes"], dtype=np.int32),
        "var_names": var_names,
        "factor_names": factor_names,
        "split_ratio": float(meta["split_ratio"]),
        "update_interval": int(meta["settings"]["update_interval"]),
        "converged": outcome["stop_reason"] == "stable_assignments_and_weights",
        "seed": int(meta["settings"]["graph_seed"]),
    }
    return _finish(run)


def pair_halves(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """edge rows of half A and half B for every (variable, original factor) pair."""
    slots: dict[tuple[int, int], list] = {}
    for k in range(run["damped"].shape[1]):
        fn = int(run["trg_fn"][k])
        key = (int(run["trg_var_idx"][k]), int(run["fn_orig_idx"][fn]))
        slot = slots.setdefault(key, [None, None])
        half = int(run["fn_half"][fn])
        assert slot[half] is None, f"duplicate half {half} for pair {key}"
        slot[half] = k
    assert all(a is not None and b is not None for a, b in slots.values()), (
        "unpaired half"
    )
    keys = sorted(slots)
    k_a = np.array([slots[key][0] for key in keys], dtype=np.int32)
    k_b = np.array([slots[key][1] for key in keys], dtype=np.int32)
    return k_a, k_b


def uniform_share(run: dict) -> np.ndarray:
    """the share every source would get if attention were uniform: 1 / group size."""
    counts = np.bincount(run["src_trg"], minlength=run["damped"].shape[1])
    return 1.0 / counts[run["src_trg"]]


def twin_mask(run: dict) -> np.ndarray:
    """sources that are the other half of the target edge's own factor."""
    trg_fn = run["trg_fn"][run["src_trg"]]
    return run["fn_orig_idx"][run["src_fn"]] == run["fn_orig_idx"][trg_fn]


def first_fixed_iteration(run: dict) -> int | None:
    """iteration from which the assignment never changed again (None if it kept changing)."""
    if not run["converged"]:
        return None
    changed = np.nonzero(run["changes_per_iter"])[0]
    return int(changed[-1]) if changed.size else 0


def pooled_over_iterations(
    runs: list[dict], values_fn
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """per-iteration min / median / max of values_fn(run) [n_iter, ...] pooled over runs."""
    n_max = max(run["n_iter"] for run in runs)
    per_run = [np.asarray(values_fn(run)).reshape(run["n_iter"], -1) for run in runs]
    lo, mid, hi = np.empty(n_max), np.empty(n_max), np.empty(n_max)
    for t in range(n_max):
        chunk = np.concatenate([v[t] for v in per_run if v.shape[0] > t])
        lo[t], mid[t], hi[t] = chunk.min(), np.median(chunk), chunk.max()
    return np.arange(n_max), lo, mid, hi


def update_guides(ax, update_interval: int, xmax: float) -> None:
    """thin vertical lines where the network takes an optimizer step."""
    for x in range(update_interval, int(xmax) + 1, update_interval):
        ax.axvline(x, color=GRAY, lw=0.5, alpha=0.2, zorder=0)
