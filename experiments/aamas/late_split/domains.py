"""Dense domain-size variants with freshly measured, matched native baselines."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.aaai.code.problems import NUM_AGENTS, PREF_SCALE
from propflow import DampingEngine, DampingSCFGEngine, FGBuilder, MinSumComputator

from .core import (
    Config,
    TraceSnapshots,
    advance,
    input_fingerprint,
    save_input,
    trace_arrays,
    verify_costs,
    write_json,
)


def build_dense(seed: int, domain_size: int):
    """Use the historical dense generator's RNG, costs and unary preferences."""
    if domain_size < 2:
        raise ValueError("domain size must be at least two")
    np.random.seed(seed)
    graph = FGBuilder.build_random_graph(
        num_vars=NUM_AGENTS,
        domain_size=domain_size,
        ct_factory="random_int",
        ct_params={"low": 100, "high": 200},
        density=0.6,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    unary = {
        v.name: rng.uniform(0.0, PREF_SCALE, size=v.domain) for v in graph.variables
    }
    return FGBuilder.build_with_unary_costs(graph, unary)


def generate_references(case: Path, seed: int, domain_size: int, config: Config):
    """Save the input and run DMS and damped splitting on identical original tables."""
    graph = build_dense(seed, domain_size)
    save_input(graph, case / "input.npz")
    references, checks = {}, {}
    for label, cls, extra in [
        ("DMS", DampingEngine, {}),
        ("DMS_split_0.5", DampingSCFGEngine, {"split_factor": config.split}),
    ]:
        print(f"BASELINE {label} seed={seed} domain={domain_size}", flush=True)
        engine = cls(
            deepcopy(graph),
            computator=MinSumComputator(),
            damping_factor=config.damping,
            normalize_messages=True,
            anytime=False,
            snapshot_manager=TraceSnapshots(),
            **extra,
        )
        trace = trace_arrays(
            [
                advance(engine, i)
                for i in range(config.prefix_steps + config.post_steps)
            ],
            [v.name for v in engine.var_nodes],
        )
        checks[label] = verify_costs(graph, trace)
        np.savez_compressed(case / f"{label}_trace.npz", **trace)
        references[label] = trace["costs"]
        references[label + "_iterations"] = trace["iterations"]
    np.savez_compressed(case / "references.npz", **references)
    write_json(
        case / "references.json",
        {
            "input_sha256": input_fingerprint(graph),
            "domain_size": domain_size,
            "agents": NUM_AGENTS,
            "density": 0.6,
            "cost_verification_max_error": checks,
            "origin": "fresh matched native execution; no domain-10 CSV reuse",
        },
    )
    return graph
