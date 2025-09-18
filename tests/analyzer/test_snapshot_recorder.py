import json
from types import SimpleNamespace

import numpy as np
import pytest

from analyzer.snapshot_recorder import EngineSnapshotRecorder
from propflow.bp.engine_components import Step
from propflow.core.components import Message


class DummyEngine:
    def __init__(self, steps, assignments, costs):
        self._steps = steps
        self._assignments = assignments
        self._costs = costs
        self._cursor = -1

    def step(self, iteration):
        if iteration >= len(self._steps):
            raise IndexError("iteration exceeds prepared steps")
        self._cursor = iteration
        return self._steps[iteration]

    @property
    def assignments(self):
        if self._cursor < 0:
            return {}
        return self._assignments[self._cursor]

    def calculate_global_cost(self):
        if self._cursor < 0:
            return None
        return self._costs[self._cursor]


@pytest.fixture
def sample_engine():
    var_x1 = SimpleNamespace(name="X1", type="variable")
    var_x2 = SimpleNamespace(name="X2", type="variable")
    factor_f1 = SimpleNamespace(name="F1", type="factor")

    step0 = Step(num=0)
    step0.q_messages["X1"] = [Message(np.array([1.0, 2.0]), var_x1, factor_f1)]
    step0.r_messages["F1"] = [Message(np.array([3.0, 3.0]), factor_f1, var_x2)]

    step1 = Step(num=1)
    step1.q_messages["X1"] = [Message(np.array([0.5, 1.5]), var_x1, factor_f1)]
    step1.r_messages["F1"] = [Message(np.array([2.0, 4.0]), factor_f1, var_x2)]

    assignments = [
        {"X1": 0, "X2": 1},
        {"X1": 0, "X2": 0},
    ]
    costs = [10.5, 8.0]
    return DummyEngine([step0, step1], assignments, costs)


def test_record_run_collects_messages_and_assignments(sample_engine):
    recorder = EngineSnapshotRecorder(sample_engine)
    snapshots = recorder.record_run(max_steps=2)

    assert len(snapshots) == 2
    first = snapshots[0]

    assert first["step"] == 0
    assert first["assignments"] == {"X1": 0, "X2": 1}
    assert pytest.approx(first["cost"], rel=1e-9) == 10.5

    message_flows = {msg["flow"] for msg in first["messages"]}
    assert message_flows == {"variable_to_factor", "factor_to_variable"}

    neutral_flags = [msg["neutral"] for msg in first["messages"]]
    assert neutral_flags.count(True) == 1
    assert first["neutral_messages"] == 1
    assert first["step_neutral"] is False


def test_to_json_writes_expected_payload(sample_engine, tmp_path):
    recorder = EngineSnapshotRecorder(sample_engine)
    recorder.record_run(max_steps=1)

    out_path = tmp_path / "snapshots.json"
    recorder.to_json(out_path)

    with out_path.open() as fh:
        payload = json.load(fh)

    assert payload == recorder.snapshots
