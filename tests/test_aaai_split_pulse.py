"""End-to-end pulse execution through the unchanged native message machinery."""

import numpy as np
import pytest

from experiments.aaai.code.engines import CostOnlySnapshotManager
from experiments.other.aaai_derived_control.code.kernel import (
    PairwiseKernel,
    make_small_problem,
)
from experiments.other.aaai_derived_control.code.pulse import SplitPulseEngine


@pytest.mark.parametrize("family", ["random", "frustrated"])
def test_native_pulse_matches_saved_experiment_schedule(family):
    problem = make_small_problem("k4", family, 5000)
    kernel = PairwiseKernel(problem)
    engine = SplitPulseEngine(
        problem.to_native(), snapshot_manager=CostOnlySnapshotManager()
    )
    initial_unary = {
        f.name: f.cost_table.copy()
        for f in engine.graph.factors
        if len(f.connection_number) == 1
    }
    for step in range(350):
        if step == 64:
            kernel.weights[:] = 0.95
        if step == 256:
            kernel.weights[:] = 0.5
        kernel.step()
        engine.step(step)
        try:
            engine._handle_cycle_events(step)
        except StopIteration:
            pass
        native = np.array(
            [
                next(
                    v for v in engine.graph.variables if v.name == name
                ).curr_assignment
                for name in problem.variable_names
            ]
        )
        assert np.array_equal(native, kernel.assignment)
        assert engine.get_snapshot(step).global_cost == pytest.approx(
            kernel.cost, abs=1e-10
        )
    assert engine.split_events == [
        {"step": 64, "weight": 0.95},
        {"step": 256, "weight": 0.5},
    ]
    assert engine.damping_factor == 0.9
    for f in engine.graph.factors:
        if f.name in initial_unary:
            np.testing.assert_array_equal(f.cost_table, initial_unary[f.name])
    for original, clones in engine._pulse_pairs:
        np.testing.assert_array_equal(
            clones[0].cost_table + clones[1].cost_table, original
        )


@pytest.mark.parametrize(
    "kwargs", [{"pulse_stop": 64}, {"pulse_weight": 1}, {"pulse_weight": np.nan}]
)
def test_invalid_pulse_is_rejected_before_mutating_graph(kwargs):
    graph = make_small_problem("k4", "random", 3).to_native()
    original_names = [f.name for f in graph.factors]
    with pytest.raises(ValueError):
        SplitPulseEngine(graph, **kwargs)
    assert [f.name for f in graph.factors] == original_names
