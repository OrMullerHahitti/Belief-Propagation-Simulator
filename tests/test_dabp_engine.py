"""Tests for the optional DABP engine integration.

Skipped automatically when the ``[dabp]`` extra (torch + torch-geometric) is not
installed, so the core test suite is unaffected.
"""

from __future__ import annotations

import itertools
import importlib

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from propflow import FGBuilder  # noqa: E402
from propflow.bp.factor_graph import FactorGraph  # noqa: E402
from propflow.configs import create_random_int_table  # noqa: E402
from propflow.core.agents import FactorAgent, VariableAgent  # noqa: E402
from propflow.integrations.dabp import (  # noqa: E402
    DABPEngine,
    DABPEngineNoSplit,
    DABPEngine_No_Split,
)
from propflow.integrations.dabp.build import build_dabp_inputs  # noqa: E402
from propflow.integrations.dabp.constant import SCALE, SPLIT_RATIO  # noqa: E402


def _small_cycle(seed: int = 0, num_vars: int = 5, domain: int = 3):
    np.random.seed(seed)
    return FGBuilder.build_cycle_graph(
        num_vars=num_vars,
        domain_size=domain,
        ct_factory=create_random_int_table,
        ct_params={"low": 1, "high": 10},
    )


def _single_binary_graph():
    v1 = VariableAgent("x1", domain=2)
    v2 = VariableAgent("x2", domain=2)
    table = np.array([[2.0, 4.0], [6.0, 8.0]])
    factor = FactorAgent.create_from_cost_table("f12", cost_table=table)
    graph = FactorGraph([v1, v2], [factor], edges={factor: [v1, v2]})
    return graph, table


def _cost(fg, assignment: dict) -> float:
    total = 0.0
    for f in fg.original_factors:
        idx = [None] * f.cost_table.ndim
        for vname, dim in f.connection_number.items():
            idx[dim] = assignment[vname]
        total += float(f.cost_table[tuple(idx)])
    return total


def test_build_default_mode_splits_binary_factors():
    fg, table = _single_binary_graph()

    data, names, domain = build_dabp_inputs(fg)

    assert names == ["x1", "x2"]
    assert domain == 2
    assert data["NF"] == 2
    np.testing.assert_allclose(data["cost_tensors"][0], table / SCALE * SPLIT_RATIO)
    np.testing.assert_allclose(
        data["cost_tensors"][1],
        table / SCALE * (1.0 - SPLIT_RATIO),
    )


def test_build_no_split_mode_keeps_one_factor_tensor():
    fg, table = _single_binary_graph()

    data, names, domain = build_dabp_inputs(fg, factor_splitting_enabled=False)

    assert names == ["x1", "x2"]
    assert domain == 2
    assert data["NF"] == 1
    assert len(data["func_embed"]) == 1
    assert data["cost_tensors"].shape == (1, 2, 2)
    np.testing.assert_allclose(data["cost_tensors"][0], table / SCALE)

    for key in ("msg_rv2f_idxes", "msg_cv2f_idxes", "msg_f2rv_idxes", "msg_f2cv_idxes"):
        assert len(data[key]) == 1
    assert len(data["msg_v2f_idxes"]) == 2
    assert len(data["msg_f2v_idxes"]) == 2
    assert len(data["edge_index"][0]) == 8


def test_dabp_no_split_public_api_exports():
    import propflow
    import propflow.engines as engines

    assert DABPEngine_No_Split is DABPEngineNoSplit
    assert propflow.DABPEngineNoSplit is DABPEngineNoSplit
    assert propflow.DABPEngine_No_Split is DABPEngineNoSplit
    assert engines.DABPEngineNoSplit is DABPEngineNoSplit
    assert engines.DABPEngine_No_Split is DABPEngineNoSplit
    assert engines.ENGINES["DABPEngineNoSplit"] is DABPEngineNoSplit
    assert engines.ENGINES["DABPEngine_No_Split"] is DABPEngineNoSplit

    reloaded = importlib.import_module("propflow.integrations.dabp")
    assert reloaded.DABPEngineNoSplit is DABPEngineNoSplit


@pytest.mark.slow
def test_dabp_records_cost_and_assignment():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle()
    eng = DABPEngine(
        factor_graph=fg, update_interval=4, restart_period=20, device="cpu"
    )

    n = 12
    for i in range(n):
        eng.step(i)

    # every step records a finite global cost
    costs = [eng._snapshots[i].global_cost for i in range(n)]
    assert len(costs) == n
    assert all(np.isfinite(c) for c in costs)

    # assignment covers all variables with valid domain indices
    assignment = eng.assignments
    assert set(assignment) == {v.name for v in fg.variables}
    assert all(0 <= a < fg.variables[0].domain for a in assignment.values())

    # the recorded cost equals the cost of the reported assignment on the
    # original tables, and is never better than the brute-force optimum
    assert abs(eng._snapshots[n - 1].global_cost - _cost(fg, assignment)) < 1e-9
    names = [v.name for v in fg.variables]
    optimal = min(
        _cost(fg, dict(zip(names, c)))
        for c in itertools.product(range(3), repeat=len(names))
    )
    assert min(costs) >= optimal - 1e-9


@pytest.mark.slow
def test_dabp_no_split_records_cost_without_mutating_graph():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle(seed=2, num_vars=4, domain=2)
    original_factor_count = len(fg.factors)
    eng = DABPEngineNoSplit(
        factor_graph=fg,
        update_interval=3,
        restart_period=10,
        device="cpu",
    )

    assert len(fg.factors) == original_factor_count
    assert eng._data["NF"] == original_factor_count

    n = 6
    for i in range(n):
        eng.step(i)

    costs = [eng._snapshots[i].global_cost for i in range(n)]
    assert all(np.isfinite(c) for c in costs)
    assert len(fg.factors) == original_factor_count

    assignment = eng.assignments
    assert set(assignment) == {v.name for v in fg.variables}
    assert all(0 <= a < fg.variables[0].domain for a in assignment.values())


@pytest.mark.slow
def test_dabp_reaches_optimum_on_small_cycle():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle(seed=1)
    names = [v.name for v in fg.variables]
    optimal = min(
        _cost(fg, dict(zip(names, c)))
        for c in itertools.product(range(3), repeat=len(names))
    )

    eng = DABPEngine(
        factor_graph=fg, update_interval=5, restart_period=30, device="cpu"
    )
    for i in range(40):
        eng.step(i)
    costs = [eng._snapshots[i].global_cost for i in range(40)]
    assert min(costs) == pytest.approx(optimal, abs=1e-6)


def test_build_rejects_higher_arity():
    """a factor with arity > 2 is rejected with a clear error."""

    v1, v2, v3 = (VariableAgent(f"x{i}", domain=2) for i in (1, 2, 3))
    table = np.ones((2, 2, 2))
    f = FactorAgent("f", domain=2, ct_creation_func=lambda *a, **k: table)
    fg = FactorGraph([v1, v2, v3], [f], edges={f: [v1, v2, v3]})

    with pytest.raises(ValueError, match="unary/binary"):
        build_dabp_inputs(fg)
