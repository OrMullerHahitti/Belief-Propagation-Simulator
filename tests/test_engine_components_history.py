import json
import numpy as np

from propflow.bp.engine_components import Cycle, History, Step
from propflow.core.agents import VariableAgent, FactorAgent
from propflow.core.components import Message


def _make_agents():
    var = VariableAgent("x1", domain=2)
    factor = FactorAgent(
        "f1",
        domain=2,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((2, 2)),
        param={},
    )
    return var, factor


def test_step_and_cycle_equality():
    var, factor = _make_agents()
    msg = Message(np.array([1.0, 2.0]), sender=factor, recipient=var)

    step_a = Step(num=0)
    step_a.add(var, msg)

    step_b = Step(num=0)
    step_b.add(var, msg)

    cycle_1 = Cycle(number=0)
    cycle_1.add(step_a)

    cycle_2 = Cycle(number=0)
    cycle_2.add(step_b)

    assert cycle_1 == cycle_2


def test_history_legacy_serialization(tmp_path):
    var, factor = _make_agents()
    history = History(engine_type="TestEngine", use_bct_history=False)

    cycle0 = Cycle(number=0)
    step0 = Step(num=0)
    step0.add(var, Message(np.array([0.0, 1.0]), sender=factor, recipient=var))
    cycle0.add(step0)
    history[0] = cycle0
    history.beliefs[0] = {var.name: np.array([1.0, 2.0])}
    history.assignments[0] = {var.name: 1}

    cycle1 = Cycle(number=1)
    step1 = Step(num=1)
    step1.add(var, Message(np.array([1.0, 3.0]), sender=factor, recipient=var))
    cycle1.add(step1)
    history[1] = cycle1
    history.assignments[1] = {var.name: 1}

    history.initialize_cost(5.0)
    assert len(history.costs) == 5
    assert history.compare_last_two_cycles() is True

    output_path = history.to_json(str(tmp_path / "legacy_history.json"))
    assert (tmp_path / "legacy_history.json").exists()
    with open(output_path) as fh:
        payload = json.load(fh)
    assert payload["engine_type"] == "TestEngine"
    assert payload["cycles"]["0"]["steps"][0]["num"] == 0

    bct_payload = history.get_bct_data()
    assert bct_payload["metadata"]["has_step_data"] is False
    assert var.name in bct_payload["beliefs"]


def test_history_bct_tracking_and_export(tmp_path):
    history = History(engine_type="BCTEngine", use_bct_history=True)
    var, factor = _make_agents()

    class DummyEngine:
        def __init__(self):
            self.assignments = {var.name: 0}

        def get_beliefs(self):
            return {var.name: np.array([0.2, 0.8])}

        def calculate_global_cost(self):
            return 1.5

    engine = DummyEngine()
    step = Step(num=0)
    step.add(var, Message(np.array([0.4, 0.6]), sender=factor, recipient=var))

    history.track_step_data(0, step, engine)
    bct = history.get_bct_data()

    assert bct["metadata"]["has_step_data"] is True
    assert bct["beliefs"][var.name] == [0.2]
    assert bct["assignments"][var.name] == [0]
    assert bct["messages"][f"{factor.name}->{var.name}"] == [0.4]
    assert history.step_costs[-1] == 1.5

    path = history.to_json(str(tmp_path / "bct_history.json"))
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["metadata"]["has_step_data"] is True
