import copy

import numpy as np

from propflow import BPEngine, FactorAgent, FGBuilder, MidRunSplitEngine, VariableAgent
from propflow.policies import split_factors


def _factor(name, table=None):
    table = np.asarray(table if table is not None else [[0, 1], [2, 0]], dtype=float)
    return FactorAgent.create_from_cost_table(name, table)


def _chain_graph():
    x1 = VariableAgent("X1", 2)
    x2 = VariableAgent("X2", 2)
    x3 = VariableAgent("X3", 2)
    f12 = _factor("F12", [[0, 4], [5, 1]])
    f23 = _factor("F23", [[0, 2], [3, 0]])
    return FGBuilder.build_from_edges(
        [x1, x2, x3],
        [f12, f23],
        {f12: [x1, x2], f23: [x2, x3]},
    )


def _four_factor_graph():
    variables = [VariableAgent(f"X{i}", 2) for i in range(1, 5)]
    factors = [_factor(f"F{i}{i+1}") for i in range(1, 4)]
    factors.append(_factor("F41"))
    edges = {
        factors[0]: [variables[0], variables[1]],
        factors[1]: [variables[1], variables[2]],
        factors[2]: [variables[2], variables[3]],
        factors[3]: [variables[3], variables[0]],
    }
    return FGBuilder.build_from_edges(variables, factors, edges)


def test_split_factors_defaults_to_all_targets():
    graph = _chain_graph()

    mapping = split_factors(graph, p=0.5)

    assert sorted(mapping) == ["F12", "F23"]
    assert sorted(f.name for f in graph.factors) == ["F12'", "F12''", "F23'", "F23''"]


def test_split_factors_named_target_only():
    graph = _chain_graph()

    mapping = split_factors(graph, p=0.5, factor_names=["F12"])

    assert sorted(mapping) == ["F12"]
    assert sorted(f.name for f in graph.factors) == ["F12'", "F12''", "F23"]


def test_split_factors_percentage_is_seeded_and_deterministic():
    first = _four_factor_graph()
    second = _four_factor_graph()

    first_mapping = split_factors(first, p=0.5, split_fraction=0.5, seed=123)
    second_mapping = split_factors(second, p=0.5, split_fraction=0.5, seed=123)

    assert sorted(first_mapping) == sorted(second_mapping)
    assert len(first_mapping) == 2


def test_midrun_split_reset_splits_and_continues():
    graph = _chain_graph()
    engine = MidRunSplitEngine(
        factor_graph=graph,
        split_at_iter=1,
        split_targets=["F12"],
        transfer_mode="reset",
        normalize_messages=False,
    )

    engine.run(max_iter=4)

    assert engine.iteration_count == 4
    assert engine.split_events[0]["iteration"] == 1
    assert engine.split_events[0]["split_mapping"] == {"F12": ["F12'", "F12''"]}
    assert sorted(f.name for f in graph.factors) == ["F12'", "F12''", "F23"]


def test_midrun_split_transfer_mode_runs_or_clearly_fails():
    engine = MidRunSplitEngine(
        factor_graph=_chain_graph(),
        split_at_iter=1,
        split_targets=["F12"],
        transfer_mode="transfer",
        normalize_messages=False,
    )

    engine.run(max_iter=4)

    assert engine.iteration_count == 4
    assert engine.split_events[0]["transfer_mode"] == "transfer"


def test_midrun_split_does_not_mutate_unrelated_graph_copy():
    original = _chain_graph()
    untouched = copy.deepcopy(original)
    engine = MidRunSplitEngine(
        factor_graph=original,
        split_at_iter=1,
        split_targets=["F12"],
        normalize_messages=False,
    )

    engine.run(max_iter=2)

    assert sorted(f.name for f in untouched.factors) == ["F12", "F23"]
    assert sorted(f.name for f in original.factors) == ["F12'", "F12''", "F23"]


def test_standard_engine_still_works_on_chain_after_new_imports():
    engine = BPEngine(_chain_graph(), normalize_messages=False)
    engine.run(max_iter=3)

    assert engine.iteration_count == 3
