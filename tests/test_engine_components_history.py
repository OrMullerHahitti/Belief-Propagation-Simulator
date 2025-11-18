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
