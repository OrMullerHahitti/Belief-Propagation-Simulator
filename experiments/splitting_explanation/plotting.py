"""plot conventions shared by the explanation experiments: no top/right frame, MS / DMS names,
PDF output, one figure per question."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
RESULTS = HERE / "results"

LABELS = {
    "MS": "MS",
    "DMS": "DMS",
    "MS_split": "MS + split",
    "DMS_split": "DMS + split",
    "DMS05_split": "DMS($\\lambda$=0.5) + split",
    "cav": "MS (cavity)",
    "belief": "echo only (Q = belief)",
    "2cav": "doubling only (Q = 2 cavity)",
    "cav+belief": "split = cavity + belief",
}
STYLE = {
    "MS": dict(color="0.55", ls=":"),
    "DMS": dict(color="0.2", ls="--"),
    "MS_split": dict(color="tab:orange", ls=":"),
    "DMS_split": dict(color="tab:red", ls="-"),
    "DMS05_split": dict(color="tab:purple", ls="-."),
    "cav": dict(color="0.2", ls="--"),
    "belief": dict(color="tab:blue", ls="-."),
    "2cav": dict(color="tab:green", ls=":"),
    "cav+belief": dict(color="tab:red", ls="-"),
}
BENCH_TITLE = {
    "random_dense": "random dense",
    "random_sparse": "random sparse",
    "graph_coloring": "graph coloring",
    "scale_free": "scale free",
    "meeting_scheduling": "meeting scheduling",
}


def remove_frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def new_fig(ncols=1, nrows=1, width=4.2, height=3.2):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False
    )
    for ax in axes.ravel():
        remove_frame(ax)
    return fig, axes


def save(fig, name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
