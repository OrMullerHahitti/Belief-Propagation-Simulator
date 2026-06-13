"""Tests for the optional DABP engine integration.

Skipped automatically when the ``[dabp]`` extra (torch + torch-geometric) is not
installed, so the core test suite is unaffected.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from propflow import FGBuilder  # noqa: E402
from propflow.configs import create_random_int_table  # noqa: E402
from propflow.integrations.dabp import DABPEngine  # noqa: E402
from propflow.integrations.dabp.build import build_dabp_inputs  # noqa: E402


def _small_cycle(seed: int = 0, num_vars: int = 5, domain: int = 3):
    np.random.seed(seed)
    return FGBuilder.build_cycle_graph(
        num_vars=num_vars,
        domain_size=domain,
        ct_factory=create_random_int_table,
        ct_params={"low": 1, "high": 10},
    )


def _cost(fg, assignment: dict) -> float:
    total = 0.0
    for f in fg.original_factors:
        idx = [None] * f.cost_table.ndim
        for vname, dim in f.connection_number.items():
            idx[dim] = assignment[vname]
        total += float(f.cost_table[tuple(idx)])
    return total


@pytest.mark.slow
def test_dabp_records_cost_and_assignment():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle()
    eng = DABPEngine(factor_graph=fg, update_interval=4, restart_period=20, device="cpu")

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
    optimal = min(_cost(fg, dict(zip(names, c))) for c in itertools.product(range(3), repeat=len(names)))
    assert min(costs) >= optimal - 1e-9


@pytest.mark.slow
def test_dabp_reaches_optimum_on_small_cycle():
    import torch

    torch.manual_seed(0)
    fg = _small_cycle(seed=1)
    names = [v.name for v in fg.variables]
    optimal = min(_cost(fg, dict(zip(names, c))) for c in itertools.product(range(3), repeat=len(names)))

    eng = DABPEngine(factor_graph=fg, update_interval=5, restart_period=30, device="cpu")
    for i in range(40):
        eng.step(i)
    costs = [eng._snapshots[i].global_cost for i in range(40)]
    assert min(costs) == pytest.approx(optimal, abs=1e-6)


def test_build_rejects_higher_arity():
    """a factor with arity > 2 is rejected with a clear error."""
    from propflow.core.agents import FactorAgent, VariableAgent
    from propflow.bp.factor_graph import FactorGraph

    v1, v2, v3 = (VariableAgent(f"x{i}", domain=2) for i in (1, 2, 3))
    table = np.ones((2, 2, 2))
    f = FactorAgent("f", domain=2, ct_creation_func=lambda *a, **k: table)
    fg = FactorGraph([v1, v2, v3], [f], edges={f: [v1, v2, v3]})

    with pytest.raises(ValueError, match="unary/binary"):
        build_dabp_inputs(fg)
