"""Monte Carlo convergence experiment for the butterfly factor graph.

estimates P(convergence) for min-sum BP on a symmetric butterfly (SCFG) topology:
3 variables (x1, x2, x3, domain=2), 4 factors (f1=f5 on x1-x2, f4=f6 on x2-x3).
cost table entries drawn from Uniform[0,1], symmetric pairs share identical tables.
compares plain BPEngine vs DampingEngine (damping=0.9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ensure propflow is importable when running from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from propflow import BPEngine, DampingEngine, FactorAgent, FactorGraph, VariableAgent

# -- configuration --
N_SAMPLES = 10000
MAX_ITER = 200
DOMAIN = 2
BELIEF_EQ_TOL = 1e-10
SEED_START = 0
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ENGINE_CONFIGS = {
    "BPEngine": {"cls": BPEngine, "kwargs": {}},
    "DampingEngine": {"cls": DampingEngine, "kwargs": {"damping_factor": 0.9}},
}


def build_butterfly_graph(seed: int) -> tuple[FactorGraph, dict]:
    """builds the symmetric butterfly factor graph with random Uniform[0,1] cost tables.

    topology: x1--f1--x2, x1--f5--x2, x2--f4--x3, x2--f6--x3
    symmetric pairs: f1=f5 (same cost table), f4=f6 (same cost table).
    """
    rng = np.random.RandomState(seed)
    ct_left = rng.uniform(0, 1, (DOMAIN, DOMAIN))
    ct_right = rng.uniform(0, 1, (DOMAIN, DOMAIN))

    x1 = VariableAgent("x1", domain=DOMAIN)
    x2 = VariableAgent("x2", domain=DOMAIN)
    x3 = VariableAgent("x3", domain=DOMAIN)

    f1 = FactorAgent.create_from_cost_table("f1", ct_left)
    f5 = FactorAgent.create_from_cost_table("f5", ct_left.copy())
    f4 = FactorAgent.create_from_cost_table("f4", ct_right)
    f6 = FactorAgent.create_from_cost_table("f6", ct_right.copy())

    edges = {f1: [x1, x2], f5: [x1, x2], f4: [x2, x3], f6: [x2, x3]}
    fg = FactorGraph([x1, x2, x3], [f1, f4, f5, f6], edges)

    return fg, {"ct_left": ct_left, "ct_right": ct_right}


def has_belief_equality(beliefs: dict[str, np.ndarray], tol: float = BELIEF_EQ_TOL) -> bool:
    """checks if any variable has ambiguous argmin (tied belief values)."""
    for belief in beliefs.values():
        if belief is not None and len(belief) == DOMAIN:
            if np.abs(belief[0] - belief[1]) < tol:
                return True
    return False


def run_single(fg: FactorGraph, engine_cls, engine_kwargs: dict, max_iter: int):
    """runs a single BP trial and returns convergence info."""
    engine = engine_cls(factor_graph=fg, **engine_kwargs)
    engine.run(max_iter=max_iter)
    beliefs = engine.get_beliefs()
    converged = engine.convergence_monitor.stable_count >= engine.convergence_monitor.config.patience
    return converged, beliefs, engine.iteration_count


def run_experiment():
    """main Monte Carlo loop over both engines."""
    stats = {name: {"converged": 0, "not_converged": 0, "belief_equality": 0} for name in ENGINE_CONFIGS}
    non_converging = []

    for i in range(N_SAMPLES):
        seed = SEED_START + i

        for eng_name, eng_cfg in ENGINE_CONFIGS.items():
            # build a fresh graph per engine (agents carry mutable state)
            fg, cost_tables = build_butterfly_graph(seed)
            converged, beliefs, iters = run_single(fg, eng_cfg["cls"], eng_cfg["kwargs"], MAX_ITER)

            if has_belief_equality(beliefs):
                stats[eng_name]["belief_equality"] += 1
                continue

            if converged:
                stats[eng_name]["converged"] += 1
            else:
                stats[eng_name]["not_converged"] += 1
                non_converging.append({
                    "seed": seed,
                    "engine": eng_name,
                    "ct_left": cost_tables["ct_left"].tolist(),
                    "ct_right": cost_tables["ct_right"].tolist(),
                    "iteration_count": iters,
                    "final_beliefs": {k: v.tolist() for k, v in beliefs.items() if v is not None},
                })

        if (i + 1) % 1000 == 0:
            print(f"  [{i + 1}/{N_SAMPLES}]")

    # compute P(convergence) per engine
    results = {"n_samples": N_SAMPLES, "max_iter": MAX_ITER, "domain": DOMAIN, "engines": {}}
    for eng_name, s in stats.items():
        effective = s["converged"] + s["not_converged"]
        p_conv = s["converged"] / effective if effective > 0 else 0.0
        results["engines"][eng_name] = {
            "converged": s["converged"],
            "not_converged": s["not_converged"],
            "belief_equality_excluded": s["belief_equality"],
            "effective_total": effective,
            "p_convergence": round(p_conv, 6),
        }

    return results, non_converging


def main():
    print("butterfly convergence experiment")
    print(f"  samples: {N_SAMPLES}, max_iter: {MAX_ITER}, domain: {DOMAIN}")
    print(f"  engines: {', '.join(ENGINE_CONFIGS.keys())}")
    print()

    results, non_converging = run_experiment()

    # print summary
    print()
    for eng_name, eng_results in results["engines"].items():
        print(f"  {eng_name}:")
        print(f"    converged:         {eng_results['converged']}")
        print(f"    not converged:     {eng_results['not_converged']}")
        print(f"    belief eq (excl):  {eng_results['belief_equality_excluded']}")
        print(f"    effective total:   {eng_results['effective_total']}")
        print(f"    P(convergence) =   {eng_results['p_convergence']}")
        print()

    # save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "convergence_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved: {results_path}")

    non_conv_path = RESULTS_DIR / "non_converging.json"
    with open(non_conv_path, "w") as f:
        json.dump(non_converging, f, indent=2)
    print(f"  saved: {non_conv_path} ({len(non_converging)} instances)")


if __name__ == "__main__":
    main()
