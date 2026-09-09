"""Run DABP on 10-agent random problems, recording learned edge weights.

For each seed this builds the problem, runs the selected variant
(``--engine symsplit``: ``DABPEngineSymSplit``, 0.5/0.5 factor split;
``--engine asym``: ``DABPEngine``, DABP's native 0.95/0.05 split) on CPU
float64, one BP iteration per step, until the assignment vector is unchanged
for ``--stable-iters`` consecutive iterations or ``--max-iter`` is reached,
and writes the per-iteration ``damped_weights`` / ``attention_weight`` tensors
plus index provenance to ``data/raw/seed{NNN}.npz`` (``data_asym/raw/`` for
the asymmetric engine). The cap must stay below the engine's
``restart_period`` so no message-state restart happens inside the run.

Example:
    uv run python experiments/dabp_weights/code/run_weights.py --n-problems 50
    uv run python experiments/dabp_weights/code/run_weights.py --engine asym
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problems import DENSITY, DOMAIN_SIZE, NUM_AGENTS, build_random_10  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
# engine key -> class name and default output dir; classes resolve lazily
# inside run_seed so this module stays importable without torch
ENGINE_NAMES = {"symsplit": "DABPEngineSymSplit", "asym": "DABPEngine"}
ENGINE_DIRS = {"symsplit": BASE_DIR / "data", "asym": BASE_DIR / "data_asym"}


class CostOnlySnapshot:
    """minimal snapshot holding only the per-step global cost."""

    def __init__(self, step: int) -> None:
        self.step = int(step)
        self.global_cost: float | None = None
        self.metadata: dict[str, Any] = {}


class CostOnlySnapshotManager:
    """capture only per-step global cost to keep long runs small and fast."""

    def capture_step(self, step_index: int, step: Any, engine: Any) -> CostOnlySnapshot:
        del step, engine
        return CostOnlySnapshot(step_index)


def run_seed(
    seed: int, engine_key: str, max_iter: int, stable_iters: int, out_path: Path
) -> dict:
    """run one seed to convergence and write its npz; returns a summary dict."""
    import torch
    from propflow.integrations import dabp

    engine_cls = getattr(dabp, ENGINE_NAMES[engine_key])
    torch.manual_seed(seed)
    fg = build_random_10(seed)
    engine = engine_cls(
        factor_graph=fg,
        # cpu keeps float64 and determinism; auto-select would pick MPS/float32
        device="cpu",
        record_weights=True,
        snapshot_manager=CostOnlySnapshotManager(),
    )
    assert max_iter <= engine.restart_period, (
        "a restart inside the run would reset DABP's message state"
    )

    meta = engine.weight_metadata()
    names = list(meta["ordered_names"])

    prev_vec: list[int] | None = None
    stable = 0
    converged = False
    assign_rows: list[list[int]] = []
    for i in range(max_iter):
        engine.step(i)
        assignment = engine.assignments
        vec = [assignment[n] for n in names]
        assign_rows.append(vec)
        stable = stable + 1 if vec == prev_vec else 1
        prev_vec = vec
        if stable >= stable_iters:
            converged = True
            break

    n_iter = len(assign_rows)
    t_stop = n_iter - 1
    t_first_stable = t_stop - stable_iters + 1 if converged else -1

    log = engine.weights_log
    damped = np.stack([rec["damped_weights"] for rec in log])
    attention = np.stack([rec["attention_weight"] for rec in log])
    costs = np.array(
        [engine._snapshots[i].global_cost for i in range(n_iter)], dtype=float
    )

    # order-preserving unique original-factor names + per-function-node index
    factor_names: list[str] = []
    fn_orig_idx: list[int] = []
    for fname in meta["fn_factor_names"]:
        if fname not in factor_names:
            factor_names.append(fname)
        fn_orig_idx.append(factor_names.index(fname))
    name_to_idx = {n: k for k, n in enumerate(names)}

    np.savez_compressed(
        out_path,
        damped=damped,
        attention=attention,
        costs=costs,
        assignments=np.array(assign_rows, dtype=np.int16),
        trg_var_idx=np.array(
            [name_to_idx[v] for v in meta["trg_var_names"]], dtype=np.int32
        ),
        trg_fn=np.array(meta["trg_fn_idxes"], dtype=np.int32),
        fn_orig_idx=np.array(fn_orig_idx, dtype=np.int32),
        fn_half=np.array(meta["fn_half"], dtype=np.int8),
        src_fn=np.array(meta["src_fn_idxes"], dtype=np.int32),
        src_trg=np.array(meta["src_trg_idxes"], dtype=np.int32),
        var_names=np.array(names),
        factor_names=np.array(factor_names),
        seed=seed,
        t_stop=t_stop,
        t_first_stable=t_first_stable,
        converged=converged,
        num_heads=meta["num_heads"],
        update_interval=engine.update_interval,
        max_iter=max_iter,
        stable_iters=stable_iters,
        split_ratio=engine.split_ratio,
    )
    return {
        "seed": seed,
        "converged": converged,
        "t_stop": t_stop,
        "final_cost": float(costs[-1]),
        "best_cost": float(costs.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=tuple(ENGINE_NAMES),
        default="symsplit",
        help="symsplit = 0.5/0.5 split; asym = DABP's native 0.95/0.05 split",
    )
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--stable-iters", type=int, default=25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/ for symsplit, data_asym/ for asym",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run seeds whose npz already exists (default: skip them)",
    )
    args = parser.parse_args()

    import torch

    out_dir = args.out_dir if args.out_dir is not None else ENGINE_DIRS[args.engine]
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seeds = range(args.seed_start, args.seed_start + args.n_problems)
    t0 = time.time()
    n_run = 0
    for seed in seeds:
        out_path = raw_dir / f"seed{seed:03d}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"seed {seed}: exists, skipping", flush=True)
            continue
        t_seed = time.time()
        summary = run_seed(
            seed, args.engine, args.max_iter, args.stable_iters, out_path
        )
        n_run += 1
        print(
            f"seed {seed}: converged={summary['converged']} "
            f"t_stop={summary['t_stop']} final={summary['final_cost']:.2f} "
            f"best={summary['best_cost']:.2f} ({time.time() - t_seed:.1f}s)",
            flush=True,
        )

    metadata = {
        "experiment": "dabp_weights",
        "engine": ENGINE_NAMES[args.engine],
        "num_agents": NUM_AGENTS,
        "domain_size": DOMAIN_SIZE,
        "density": DENSITY,
        "n_problems": args.n_problems,
        "seed_start": args.seed_start,
        "max_iter": args.max_iter,
        "stable_iters": args.stable_iters,
        "device": "cpu",
        "torch_version": torch.__version__,
        "seeds_run": n_run,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"done: {n_run} seeds in {metadata['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
