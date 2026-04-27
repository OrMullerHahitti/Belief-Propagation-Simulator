"""Step 2: Run BPEngine and DampingEngine on all 11 pickled graphs, write CSVs.

Prerequisites::

    uv run python expiriments/structured_vs_random/generate_graphs.py

Run::

    uv run python expiriments/structured_vs_random/run_experiment.py

Outputs
-------
results/results_graph_XX_<EngineName>.csv  (22 CSV files, one per graph × engine)
Each CSV has columns: variant_idx, pct_random, engine, iteration, cost
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import copy

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from propflow import BPEngine, DampingEngine  # noqa: E402
from propflow.utils.fg_utils import load_pickle_safely  # noqa: E402
from experiments.aij.code._ct_utils import _FixedCostTable  # noqa: E402, F401

# ── config ────────────────────────────────────────────────────────────────────

ENGINE_CONFIGS = {
    "BPEngine": {
        "class": BPEngine,
        "kwargs": {"normalize_messages": True},
    },
    "DampingEngine": {
        "class": DampingEngine,
        "kwargs": {"damping_factor": 0.9, "normalize_messages": True},
    },
}

MAX_ITER = 2000


# ── runner ────────────────────────────────────────────────────────────────────

def run_single(fg, engine_class, engine_kwargs: dict, max_iter: int) -> list[float]:
    """Run an engine up to max_iter iterations (stops early on convergence)."""
    engine = engine_class(factor_graph=fg, **engine_kwargs)
    engine.run(max_iter=max_iter)
    return list(engine.history.costs)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    aij_dir     = Path(__file__).resolve().parent.parent  # experiments/aij/
    graphs_dir  = aij_dir / "data" / "structured_vs_random_graphs"
    results_dir = aij_dir / "data"
    results_dir.mkdir(parents=True, exist_ok=True)

    # load metadata written by generate_structured_vs_random_graphs.py
    meta_path = aij_dir / "data" / "structured_vs_random_metadata.json"
    if not meta_path.exists():
        print("[run_experiment] structured_vs_random_metadata.json not found — run generate_structured_vs_random_graphs.py first")
        return
    with open(meta_path) as fh:
        meta = json.load(fh)
    n_factors        = meta["n_factors"]
    replacement_step = meta["replacement_step"]

    pkl_files = sorted(graphs_dir.glob("graph_*.pkl"))
    if not pkl_files:
        print("[run_experiment] no graphs found — run generate_graphs.py first")
        return

    print(f"[run_experiment] found {len(pkl_files)} graph files ({n_factors} total factors, step={replacement_step})")

    for pkl_path in pkl_files:
        # derive variant index from filename (graph_00.pkl → 0)
        variant_idx = int(pkl_path.stem.split("_")[1])
        n_random    = variant_idx * replacement_step
        pct_random  = round(100 * n_random / n_factors)

        fg = load_pickle_safely(str(pkl_path))
        if fg is None:
            print(f"  [!] could not load {pkl_path.name}, skipping")
            continue

        for engine_name, cfg in ENGINE_CONFIGS.items():
            print(f"  graph_{variant_idx:02d} ({pct_random:3d}% random) × {engine_name} …", end=" ", flush=True)

            # deepcopy so each engine starts from a completely fresh graph state
            costs = run_single(
                copy.deepcopy(fg),
                engine_class=cfg["class"],
                engine_kwargs=cfg["kwargs"],
                max_iter=MAX_ITER,
            )

            df = pd.DataFrame({
                "variant_idx": variant_idx,
                "pct_random": pct_random,
                "engine": engine_name,
                "iteration": np.arange(len(costs)),
                "cost": costs,
            })

            out_path = results_dir / f"structured_vs_random_graph_{variant_idx:02d}_{engine_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"final_cost={costs[-1]:.2f}")

    print(f"[run_experiment] done — CSVs in {results_dir}")


if __name__ == "__main__":
    main()
