import numpy as np
import pytest

from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.factor_graph import FactorGraph

from policies.damping import damp, TD
from policies.cost_reduction import discount_attentive
from policies.message_pruning import MessagePruningPolicy
from policies.splitting import split_all_factors


def setup_var_factor_graph():
    v1 = VariableAgent("v1", 2)
    v2 = VariableAgent("v2", 2)
    ct = np.array([[1, 2], [3, 4]])
    f = FactorAgent("f", 2, ct_creation_func=lambda n, d: ct, param={})
    edges = {f: [v1, v2]}
    fg = FactorGraph(variable_li=[v1, v2], factor_li=[f], edges=edges)
    return fg, v1, v2, f


def test_message_pruning_policy():
    policy = MessagePruningPolicy(prune_threshold=0.1, min_iterations=0, adaptive_threshold=False)
    v = VariableAgent("v", 2)
    sender = FactorAgent("f", 2, ct_creation_func=lambda n,d: np.zeros((2,2)), param={})
    msg1 = Message(np.array([0.0, 0.0]), sender, v)
    msg2 = Message(np.array([0.05, 0.05]), sender, v)
    assert policy.should_accept_message(v, msg1) is True
    v.mailer.receive_messages(msg1)
    assert policy.should_accept_message(v, msg2) is False
    policy.step_completed()
    stats = policy.get_stats()
    assert stats["pruned_messages"] == 1
    policy.reset()
    assert policy.iteration_count == 0


def test_damp_and_td():
    fg, v1, v2, f = setup_var_factor_graph()
    msg_prev = Message(np.array([1.0, 2.0]), sender=v1, recipient=f)
    msg_curr = Message(np.array([3.0, 4.0]), sender=v1, recipient=f)
    v1.mailer.outbox = [msg_curr]
    v1._history.append([msg_prev])
    damp(v1, 0.5)
    np.testing.assert_array_almost_equal(msg_curr.data, 0.5 * msg_prev.data + 0.5 * np.array([3.0, 4.0]))

    # TD on list of variables
    msg_curr2 = Message(np.array([5.0, 6.0]), sender=v1, recipient=f)
    v1.mailer.outbox = [msg_curr2]
    v1._history.append([msg_curr])
    TD([v1], 0.2, diameter=1)
    np.testing.assert_array_almost_equal(msg_curr2.data, 0.2 * msg_curr.data + 0.8 * np.array([5.0,6.0]))


def test_discount_attentive_and_split():
    fg, v1, v2, f = setup_var_factor_graph()
    # each variable receives a message to test weight
    m1 = Message(np.array([1.0,1.0]), sender=f, recipient=v1)
    m2 = Message(np.array([2.0,2.0]), sender=f, recipient=v2)
    v1.mailer.receive_messages(m1)
    v2.mailer.receive_messages(m2)
    discount_attentive(fg)
    # both variables degree=1 so weight=1
    np.testing.assert_array_almost_equal(v1.mailer.inbox[0].data, np.array([1.0,1.0]))
    np.testing.assert_array_almost_equal(v2.mailer.inbox[0].data, np.array([2.0,2.0]))
    # check splitting keeps edges
    orig_edges = list(fg.G.edges())
    split_all_factors(fg, 0.5)
    assert len(fg.factors) == 2
    for _, var in orig_edges:
        assert fg.G.degree(var) == 2
