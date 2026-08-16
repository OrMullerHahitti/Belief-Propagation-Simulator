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
    DABPEngineSymSplit,
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


def test_build_symmetric_split_mode_halves_binary_factors():
    fg, table = _single_binary_graph()

    data, _, _ = build_dabp_inputs(
        fg,
        split_ratio=DABPEngineSymSplit.split_ratio,
        factor_splitting_enabled=DABPEngineSymSplit.factor_splitting_enabled,
    )

    assert data["NF"] == 2
    np.testing.assert_allclose(data["cost_tensors"][0], table / SCALE * 0.5)
    np.testing.assert_allclose(data["cost_tensors"][1], table / SCALE * 0.5)


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
    assert reloaded.DABPEngineSymSplit is DABPEngineSymSplit


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


def test_build_exposes_split_pair_provenance():
    fg, _ = _single_binary_graph()

    data, _, _ = build_dabp_inputs(fg, split_ratio=0.5, factor_splitting_enabled=True)

    assert data["fn_factor_names"] == ["f12", "f12"]
    assert data["fn_half"] == [0, 1]
    assert data["trg_var_names"] == ["x1", "x1", "x2", "x2"]
    assert data["trg_fn_idxes"] == [0, 1, 0, 1]

    # no-split: single function node, and degree-1 variables produce no target rows
    fg2, _ = _single_binary_graph()
    data2, _, _ = build_dabp_inputs(fg2, factor_splitting_enabled=False)
    assert data2["fn_factor_names"] == ["f12"]
    assert data2["fn_half"] == [0]
    assert data2["trg_var_names"] == []
    assert data2["trg_fn_idxes"] == []


@pytest.mark.slow
def test_symsplit_records_weights_with_expected_shapes():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle()
    eng = DABPEngineSymSplit(
        factor_graph=fg,
        update_interval=4,
        restart_period=100,
        device="cpu",
        record_weights=True,
    )

    n = 6
    for i in range(n):
        eng.step(i)

    meta = eng.weight_metadata()
    T = len(meta["trg_var_names"])  # directed v2f edges of variables with degree >= 2
    S = len(meta["src_fn_idxes"])  # (source fn, target edge) attention rows
    H = meta["num_heads"]
    assert T == 20 and S == 60 and H == 4

    log = eng.weights_log
    assert len(log) == n
    src_trg = np.asarray(meta["src_trg_idxes"])
    for step_idx, rec in enumerate(log):
        assert rec["iteration"] == step_idx
        damped = rec["damped_weights"]
        attention = rec["attention_weight"]
        assert damped.shape == (T, 2, H)
        assert attention.shape == (S, H)
        # the two damping components are a softmax pair per edge/head
        np.testing.assert_allclose(damped.sum(axis=1), np.ones((T, H)), atol=1e-9)
        # attention weights are a softmax within each target group
        for h in range(H):
            sums = np.zeros(T)
            np.add.at(sums, src_trg, attention[:, h])
            np.testing.assert_allclose(sums, np.ones(T), atol=1e-9)

    # every (variable, original factor) incidence pairs both split halves
    fn_orig = meta["fn_factor_names"]
    fn_half = meta["fn_half"]
    pairs: dict = {}
    for k in range(T):
        fn = meta["trg_fn_idxes"][k]
        key = (meta["trg_var_names"][k], fn_orig[fn])
        slot = pairs.setdefault(key, [None, None])
        assert slot[fn_half[fn]] is None
        slot[fn_half[fn]] = k
    assert len(pairs) == T // 2
    assert all(a is not None and b is not None for a, b in pairs.values())


@pytest.mark.slow
def test_record_weights_off_is_default_and_identical():
    import torch

    n = 8

    torch.manual_seed(0)
    eng_off = DABPEngineSymSplit(
        factor_graph=_small_cycle(), update_interval=4, restart_period=100, device="cpu"
    )
    for i in range(n):
        eng_off.step(i)

    torch.manual_seed(0)
    eng_on = DABPEngineSymSplit(
        factor_graph=_small_cycle(),
        update_interval=4,
        restart_period=100,
        device="cpu",
        record_weights=True,
    )
    for i in range(n):
        eng_on.step(i)

    assert eng_off.record_weights is False
    assert eng_off.weights_log == []
    assert len(eng_on.weights_log) == n

    # recording must not change the numbers: exact equality, not approx
    costs_off = [eng_off._snapshots[i].global_cost for i in range(n)]
    costs_on = [eng_on._snapshots[i].global_cost for i in range(n)]
    assert costs_off == costs_on
    assert eng_off.assignments == eng_on.assignments
