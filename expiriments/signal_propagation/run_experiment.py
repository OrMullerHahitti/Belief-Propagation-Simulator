"""Signal propagation experiment.

Activate one factor at a time (cost table = 1s, rest = 0s), run min-sum BP,
and record the final Q-message from the observer variable (x1) to a fixed
target factor (f13).

Supports multiple engine configurations (e.g. undamped vs damped) and produces
per-engine and combined CSV results for comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from propflow import BPEngine, DampingEngine, FactorAgent, FactorGraph, FGBuilder, VariableAgent

# ── configuration ────────────────────────────────────────────────────────────
TOPOLOGY_SEED = 42
NUM_VARS = 20
DOMAIN_SIZE = 3
DENSITY = 0.25
MAX_ITER = 50

ENGINE_CONFIGS = {
    "BPEngine": {"class": BPEngine, "kwargs": {}},
    "DampingEngine": {"class": DampingEngine, "kwargs": {"damping_factor": 0.9}},
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _extract_topology(fg):
    """Return (factor_name, var_a, var_b) tuples and sorted var names."""
    edges = []
    for f in fg.factors:
        vs = sorted(f.connection_number.keys())
        if len(vs) == 2:
            edges.append((f.name, vs[0], vs[1]))
    return edges, sorted(v.name for v in fg.variables)


def _build_signal_graph(topo_edges, var_names, domain, active_factor):
    """Build a FactorGraph with only `active_factor` set to ones."""
    variables = {n: VariableAgent(name=n, domain=domain) for n in var_names}
    edge_dict = {}
    for fname, va, vb in topo_edges:
        ct = np.ones((domain, domain)) if fname == active_factor else np.zeros((domain, domain))
        f = FactorAgent.create_from_cost_table(fname, ct)
        edge_dict[f] = [variables[va], variables[vb]]
    return FactorGraph(list(variables.values()), list(edge_dict.keys()), edge_dict)


def run_signal_experiment(
    topology_seed=TOPOLOGY_SEED,
    num_vars=NUM_VARS,
    domain_size=DOMAIN_SIZE,
    density=DENSITY,
    max_iter=MAX_ITER,
    engine_configs=ENGINE_CONFIGS,
    results_dir=RESULTS_DIR,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(topology_seed)
    ref_fg = FGBuilder.build_random_graph(
        num_vars=num_vars, domain_size=domain_size,
        ct_factory="random_int", ct_params={"low": 0, "high": 10},
        density=density, seed=topology_seed,
    )
    topo_edges, var_names = _extract_topology(ref_fg)
    observer_name = var_names[0]
    target_factor = "f13"
    factor_names = sorted(f.name for f in ref_fg.factors)

    print(f"topology: {len(var_names)} vars, {len(factor_names)} factors, "
          f"observer={observer_name}, target_factor={target_factor}")

    all_rows = {}

    for engine_name, cfg in engine_configs.items():
        engine_class = cfg["class"]
        engine_kwargs = {"normalize_messages": False, **cfg["kwargs"]}
        rows = {}

        print(f"\n── {engine_name} ({engine_kwargs}) ──")

        for fi, fname in enumerate(factor_names):
            fg = _build_signal_graph(topo_edges, var_names, domain_size, fname)
            engine = engine_class(factor_graph=fg, **engine_kwargs)

            last_step = None
            for it in range(max_iter):
                last_step = engine.step(it)

            # extract Q-message from observer to target factor
            q_val = 0.0
            if last_step and observer_name in last_step.q_messages:
                for msg in last_step.q_messages[observer_name]:
                    if msg.recipient.name.lower() == target_factor:
                        q_val = float(msg.data[0])
                        break
            rows[fname] = q_val

            if (fi + 1) % 20 == 0 or fi == 0:
                print(f"  [{fi+1}/{len(factor_names)}]")

        all_rows[engine_name] = rows

        # save per-engine csv
        csv_path = results_dir / f"signal_results_{engine_name}.csv"
        with open(csv_path, "w") as f:
            f.write("activated_factor,q_message_value\n")
            for fname in factor_names:
                f.write(f"{fname},{rows[fname]:.6f}\n")
        print(f"  saved: {csv_path.name}")

    # save legacy csv (first engine)
    first_engine = next(iter(engine_configs))
    csv_path = results_dir / "signal_results.csv"
    with open(csv_path, "w") as f:
        f.write("activated_factor,q_message_value\n")
        for fname in factor_names:
            f.write(f"{fname},{all_rows[first_engine][fname]:.6f}\n")

    # save combined csv with engine column
    combined_path = results_dir / "signal_results_combined.csv"
    with open(combined_path, "w") as f:
        f.write("engine,activated_factor,q_message_value\n")
        for engine_name, rows in all_rows.items():
            for fname in factor_names:
                f.write(f"{engine_name},{fname},{rows[fname]:.6f}\n")
    print(f"\nsaved: {combined_path.name}")

    # save metadata
    meta = {
        "topology_seed": topology_seed, "num_vars": num_vars,
        "domain_size": domain_size, "density": density, "max_iter": max_iter,
        "observer": observer_name, "target_factor": target_factor,
        "metric": "Q-message from observer to target_factor, domain value 0",
        "factor_names": factor_names,
        "topology_edges": topo_edges,
        "engines": {name: {k: str(v) for k, v in cfg["kwargs"].items()} for name, cfg in engine_configs.items()},
    }
    with open(results_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("saved: metadata.json")
    return all_rows


if __name__ == "__main__":
    run_signal_experiment()
