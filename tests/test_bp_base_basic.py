import numpy as np
from bp_base.components import Message, MailHandler
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.computators import MinSumComputator


def test_message_mailhandler_flow():
    sender = VariableAgent("s", 2)
    recipient = VariableAgent("r", 2)
    msg = Message(np.array([1.0, 0.0]), sender, recipient)
    sender.mailer.stage_sending([msg])
    sender.mailer.send()
    assert len(recipient.mailer.inbox) == 1
    recipient.empty_mailbox()
    assert len(recipient.mailer.inbox) == 0


def test_min_sum_computator_q_r():
    comp = MinSumComputator()
    v = VariableAgent("v", 2)
    f1 = FactorAgent("f1", 2, ct_creation_func=lambda n, d: np.zeros((d, d)), param={})
    f2 = FactorAgent("f2", 2, ct_creation_func=lambda n, d: np.zeros((d, d)), param={})
    m1 = Message(np.array([1.0, 2.0]), sender=f1, recipient=v)
    m2 = Message(np.array([0.5, 1.5]), sender=f2, recipient=v)
    q_msgs = comp.compute_Q([m1, m2])
    assert len(q_msgs) == 2

    f = FactorAgent("f", 2, ct_creation_func=lambda n, d: np.zeros((d, d)), param={})
    f.cost_table = np.zeros((2, 2))
    f.connection_number = {"v1": 0, "v2": 1}
    v1 = VariableAgent("v1", 2)
    v2 = VariableAgent("v2", 2)
    r_msgs = comp.compute_R(f.cost_table, [
        Message(np.array([0.0, 1.0]), sender=v1, recipient=f),
        Message(np.array([1.0, 0.0]), sender=v2, recipient=f),
    ])
    assert len(r_msgs) == 2
