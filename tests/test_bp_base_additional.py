import numpy as np
import pickle
import os

import pytest

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message, MailHandler
from bp_base.factor_graph import FactorGraph
from bp_base.engine_components import Step, Cycle, History
from bp_base.computators import MinSumComputator


def create_simple_graph():
    v1 = VariableAgent("v1", 2)
    v2 = VariableAgent("v2", 2)
    ct = np.array([[0, 1], [2, 3]])
    f = FactorAgent("f", 2, ct_creation_func=lambda n, d: ct, param={})
    edges = {f: [v1, v2]}
    fg = FactorGraph(variable_li=[v1, v2], factor_li=[f], edges=edges)
    return fg, v1, v2, f


def test_factor_agent_properties():
    fg, v1, v2, f = create_simple_graph()
    # connection numbers should be set
    assert f.connection_number == {"v1": 0, "v2": 1}
    # cost table created by init
    np.testing.assert_array_equal(f.cost_table, np.array([[0, 1], [2, 3]]))
    # save original and modify
    f.save_original()
    f.cost_table = f.cost_table + 1
    np.testing.assert_array_equal(f.original_cost_table, np.array([[0, 1], [2, 3]]))
    # mean and total cost
    assert f.mean_cost == pytest.approx(np.mean(f.cost_table))
    assert f.total_cost == pytest.approx(np.sum(f.cost_table))


def test_factor_graph_global_cost_and_pickle(tmp_path):
    fg, v1, v2, f = create_simple_graph()
    # assignments default to 0 so expected cost is cost_table[0,0]
    assert fg.global_cost == 0

    state = fg.__getstate__()
    # simulate missing graph during unpickle
    del state["G"]
    new_fg = FactorGraph.__new__(FactorGraph)
    new_fg.__setstate__(state)
    assert new_fg.diameter == 2
    assert new_fg.global_cost == 0

    # pickle/unpickle roundtrip
    pkl = tmp_path / "fg.pkl"
    with open(pkl, "wb") as f_handle:
        pickle.dump(fg, f_handle)
    with open(pkl, "rb") as f_handle:
        loaded = pickle.load(f_handle)
    assert loaded.global_cost == fg.global_cost


def test_mailhandler_deduplication():
    sender = VariableAgent("s", 2)
    recipient = VariableAgent("r", 2)
    msg1 = Message(np.array([1, 0]), sender, recipient)
    msg2 = Message(np.array([0, 1]), sender, recipient)
    recipient.mailer.receive_messages(msg1)
    # second message from same sender replaces first
    recipient.mailer.receive_messages(msg2)
    assert len(recipient.mailer.inbox) == 1
    assert np.array_equal(recipient.mailer.inbox[0].data, [0, 1])
    recipient.empty_mailbox()
    assert len(recipient.mailer.inbox) == 0


def test_variable_and_factor_compute_messages():
    fg, v1, v2, f = create_simple_graph()
    computator = MinSumComputator()
    v1.computator = computator
    f.computator = computator
    # prepare messages from factor to variable
    m = Message(np.array([1.0, 0.5]), sender=f, recipient=v1)
    v1.mailer.receive_messages(m)
    v1.compute_messages()
    assert len(v1.mailer.outbox) == 1
    # send to factor and compute R messages
    v1.mailer.send()
    f.mailer.receive_messages(v1.mailer.outbox[0])
    f.compute_messages()
    assert len(f.mailer.outbox) == 1


def test_step_cycle_history(tmp_path):
    sender = VariableAgent("s", 2)
    recipient = VariableAgent("r", 2)
    msg = Message(np.array([1, 2]), sender, recipient)
    step = Step(0)
    step.add(recipient, msg)

    cycle = Cycle(0)
    cycle.add(step)

    hist = History(engine_type="Test")
    hist[0] = cycle
    hist.initialize_cost(1.0)
    fn = hist.save_results(tmp_path / "results.json")
    assert os.path.exists(fn)
    loaded = pickle.load(open(fn, "rb"))
    assert "beliefs" in loaded

