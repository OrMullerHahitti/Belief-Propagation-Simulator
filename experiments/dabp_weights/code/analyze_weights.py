"""Analyze recorded DABP-SymSplit edge weights.

Reads ``data/raw/seed*.npz`` (written by run_weights.py, no torch needed here)
and writes summary CSVs into ``data/``:

- pair_asymmetry.csv          final-iteration weight pair per (var, factor, head)
- pair_asymmetry_dynamics.csv pooled |log-ratio| quantiles per iteration
- edge_weights.csv            per-edge final/last-25 damping stats + settle iter
- attention_final.csv         final-iteration attention rows (incl. twin half)
- structure_correlation.csv   per-pair effective damping vs graph structure
- correlation_stats.csv       Spearman rho of structure features vs damping
- run_summary.csv             per-seed convergence and cost summary

The "previous-message" damping component is ``damped[:, :, 1, :]``; its head
mean is the effective per-edge damping factor lambda. For each variable and
original factor, the symmetric split yields two edges (half 0 / half 1) whose
weights are paired to measure symmetry breaking.

Example:
    uv run python experiments/dabp_weights/code/analyze_weights.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problems import build_random_10  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LAST_WINDOW = 25
SETTLE_TOL = 0.01


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def damping_pairs(run: dict) -> dict[tuple[int, int], list[int]]:
    """map (variable idx, original factor idx) -> [edge row of half 0, of half 1]."""
    T = run["damped"].shape[1]
    NF = len(run["fn_half"])
    assert T == 2 * NF, "pairing assumes the symmetric split (every var degree >= 2)"
    pairs: dict[tuple[int, int], list] = {}
    for k in range(T):
        fn = int(run["trg_fn"][k])
        key = (int(run["trg_var_idx"][k]), int(run["fn_orig_idx"][fn]))
        slot = pairs.setdefault(key, [None, None])
        half = int(run["fn_half"][fn])
        assert slot[half] is None, f"duplicate half {half} for pair {key}"
        slot[half] = k
    assert len(pairs) == T // 2
    assert all(a is not None and b is not None for a, b in pairs.values())
    return pairs


def settle_iteration(series: np.ndarray, tol: float = SETTLE_TOL) -> int:
    """first index from which the remaining tail of the series has std < tol."""
    rev = np.asarray(series, dtype=float)[::-1]
    m = np.arange(1, rev.size + 1)
    var = np.cumsum(rev * rev) / m - (np.cumsum(rev) / m) ** 2
    std_tail = np.sqrt(np.maximum(var, 0.0))[::-1]
    return int(np.nonzero(std_tail < tol)[0][0])


def structure_for_seed(seed: int) -> tuple[dict[str, dict], dict[str, int]]:
    """per-factor structure info and per-variable primal degree, rebuilt from seed."""
    fg = build_random_10(seed)
    primal = nx.Graph()
    primal.add_nodes_from(v.name for v in fg.variables)
    info: dict[str, dict] = {}
    for f in fg.factors:
        cn = getattr(f, "connection_number", {}) or {}
        if len(cn) != 2:
            continue
        u, w = list(cn.keys())
        primal.add_edge(u, w)
        ct = np.asarray(f.cost_table, dtype=float)
        info[f.name] = {
            "vars": (u, w),
            "ct_mean": float(ct.mean()),
            "ct_std": float(ct.std()),
            "ct_range": float(ct.max() - ct.min()),
        }
    bridges = {frozenset(e) for e in nx.bridges(primal)}
    for d in info.values():
        d["edge_on_cycle"] = int(frozenset(d["vars"]) not in bridges)
    return info, dict(primal.degree())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    raw_paths = sorted((args.data_dir / "raw").glob("seed*.npz"))
    if not raw_paths:
        raise SystemExit(f"no raw npz files under {args.data_dir / 'raw'}")

    pair_rows = []
    edge_rows = []
    attention_rows = []
    structure_rows = []
    summary_rows = []
    # pooled |log(w0/w1)| values per iteration, across seeds/pairs/heads
    dyn_values: list[list[np.ndarray]] = []
    dyn_seeds: list[int] = []

    for path in raw_paths:
        run = load_run(path)
        seed = int(run["seed"])
        damped = run["damped"]  # [n_iter, T, 2, H]
        n_iter, T, _, H = damped.shape
        var_names = [str(v) for v in run["var_names"]]
        factor_names = [str(f) for f in run["factor_names"]]
        pairs = damping_pairs(run)
        prev_w = damped[:, :, 1, :]  # previous-message weight per edge/head
        lam = prev_w.mean(axis=2)  # effective damping, [n_iter, T]

        # --- split-pair asymmetry (final iteration + full dynamics) ----------
        k0s = np.array([slot[0] for slot in pairs.values()])
        k1s = np.array([slot[1] for slot in pairs.values()])
        log_ratio_all = np.log(prev_w[:, k0s, :] / prev_w[:, k1s, :])  # [n_iter, P, H]
        for t in range(n_iter):
            while len(dyn_values) <= t:
                dyn_values.append([])
                dyn_seeds.append(0)
            dyn_values[t].append(np.abs(log_ratio_all[t]).ravel())
            dyn_seeds[t] += 1

        for (v_idx, f_idx), (k0, k1) in pairs.items():
            for h in range(H):
                w0 = float(prev_w[-1, k0, h])
                w1 = float(prev_w[-1, k1, h])
                pair_rows.append(
                    {
                        "seed": seed,
                        "var": var_names[v_idx],
                        "factor": factor_names[f_idx],
                        "head": h,
                        "w_half0": w0,
                        "w_half1": w1,
                        "log_ratio": float(np.log(w0 / w1)),
                        "asym": abs(w0 - w1) / (w0 + w1),
                    }
                )

        # --- per-edge damping stats ------------------------------------------
        window = min(LAST_WINDOW, n_iter)
        for k in range(T):
            fn = int(run["trg_fn"][k])
            edge_settle = settle_iteration(lam[:, k])
            for h in range(H):
                series = prev_w[:, k, h]
                edge_rows.append(
                    {
                        "seed": seed,
                        "var": var_names[int(run["trg_var_idx"][k])],
                        "factor": factor_names[int(run["fn_orig_idx"][fn])],
                        "half": int(run["fn_half"][fn]),
                        "head": h,
                        "w_final": float(series[-1]),
                        "last25_mean": float(series[-window:].mean()),
                        "last25_std": float(series[-window:].std()),
                        "settle_iter": edge_settle,
                    }
                )

        # --- final attention rows (source share per target edge) -------------
        attention = run["attention"][-1]  # [S, H]
        for r in range(attention.shape[0]):
            k = int(run["src_trg"][r])
            trg_fn = int(run["trg_fn"][k])
            src_fn = int(run["src_fn"][r])
            for h in range(H):
                attention_rows.append(
                    {
                        "seed": seed,
                        "var": var_names[int(run["trg_var_idx"][k])],
                        "trg_factor": factor_names[int(run["fn_orig_idx"][trg_fn])],
                        "trg_half": int(run["fn_half"][trg_fn]),
                        "src_factor": factor_names[int(run["fn_orig_idx"][src_fn])],
                        "src_half": int(run["fn_half"][src_fn]),
                        "head": h,
                        "weight": float(attention[r, h]),
                        "is_twin": int(
                            run["fn_orig_idx"][src_fn] == run["fn_orig_idx"][trg_fn]
                        ),
                    }
                )

        # --- structure correlation (per pair, head+half mean damping) --------
        info, degree = structure_for_seed(seed)
        assert set(info) == set(factor_names), "rebuilt graph does not match run"
        for (v_idx, f_idx), (k0, k1) in pairs.items():
            fname = factor_names[f_idx]
            vname = var_names[v_idx]
            structure_rows.append(
                {
                    "seed": seed,
                    "var": vname,
                    "factor": fname,
                    "lam_final": float((lam[-1, k0] + lam[-1, k1]) / 2.0),
                    "var_degree": degree[vname],
                    "edge_on_cycle": info[fname]["edge_on_cycle"],
                    "ct_mean": info[fname]["ct_mean"],
                    "ct_std": info[fname]["ct_std"],
                    "ct_range": info[fname]["ct_range"],
                }
            )

        costs = run["costs"]
        summary_rows.append(
            {
                "seed": seed,
                "converged": bool(run["converged"]),
                "t_stop": int(run["t_stop"]),
                "t_first_stable": int(run["t_first_stable"]),
                "n_iter": n_iter,
                "final_cost": float(costs[-1]),
                "best_cost": float(costs.min()),
            }
        )

    # --- pooled asymmetry dynamics -------------------------------------------
    dyn_rows = []
    for t, chunks in enumerate(dyn_values):
        values = np.concatenate(chunks)
        dyn_rows.append(
            {
                "iteration": t,
                "n_seeds": dyn_seeds[t],
                "abs_log_ratio_median": float(np.median(values)),
                "abs_log_ratio_q25": float(np.percentile(values, 25)),
                "abs_log_ratio_q75": float(np.percentile(values, 75)),
                "abs_log_ratio_max": float(values.max()),
            }
        )

    structure_df = pd.DataFrame(structure_rows)
    corr_rows = []
    for feature in ("var_degree", "edge_on_cycle", "ct_mean", "ct_std", "ct_range"):
        rho, p = spearmanr(structure_df[feature], structure_df["lam_final"])
        corr_rows.append(
            {
                "feature": feature,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n": len(structure_df),
            }
        )

    outputs = {
        "pair_asymmetry.csv": pd.DataFrame(pair_rows),
        "pair_asymmetry_dynamics.csv": pd.DataFrame(dyn_rows),
        "edge_weights.csv": pd.DataFrame(edge_rows),
        "attention_final.csv": pd.DataFrame(attention_rows),
        "structure_correlation.csv": structure_df,
        "correlation_stats.csv": pd.DataFrame(corr_rows),
        "run_summary.csv": pd.DataFrame(summary_rows),
    }
    for name, df in outputs.items():
        df.to_csv(args.data_dir / name, index=False)
        print(f"wrote {name}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
