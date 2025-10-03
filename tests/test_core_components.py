import numpy as np

from propflow.core.agents import VariableAgent, FactorAgent
from propflow.core.components import MailHandler, Message


class AcceptAllPolicy:
    def should_accept_message(self, owner, message):
        return True


class RejectAllPolicy:
    def should_accept_message(self, owner, message):
        return False


def test_mailhandler_receive_prune_and_overwrite():
    var = VariableAgent("x1", domain=3)
    factor = FactorAgent(
        "f1",
        domain=3,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((3, 3)),
        param={},
    )

    handler = MailHandler(_domain_size=3)
    handler.set_pruning_policy(RejectAllPolicy())
    handler.receive_messages(Message(np.ones(3), sender=factor, recipient=var))
    assert handler.inbox == []  # pruned

    handler.set_pruning_policy(AcceptAllPolicy())
    message_a = Message(np.array([1.0, 2.0, 3.0]), sender=factor, recipient=var)
    handler.receive_messages(message_a)
    assert len(handler.inbox) == 1

    # Overwrite from same sender
    message_b = Message(np.array([0.0, -1.0, 5.0]), sender=factor, recipient=var)
    handler.receive_messages(message_b)
    assert len(handler.inbox) == 1
    np.testing.assert_allclose(handler.inbox[0].data, message_b.data)


def test_mailhandler_stage_send_and_clear():
    var = VariableAgent("x1", domain=2)
    factor = FactorAgent(
        "f1",
        domain=2,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((2, 2)),
        param={},
    )

    msg = Message(np.array([0.1, 0.2]), sender=var, recipient=factor)
    var.mailer.stage_sending([msg])
    assert len(var.mailer.outbox) == 1

    var.mailer.send()
    # Message should have been delivered to factor's inbox
    assert len(factor.mailer.inbox) == 1
    np.testing.assert_allclose(factor.mailer.inbox[0].data, msg.data)

    var.mailer.prepare()
    assert var.mailer.outbox == []
    factor.mailer.clear_inbox()
    assert factor.mailer.inbox == []


def test_mailhandler_set_first_message_and_inbox_setter():
    var = VariableAgent("x1", domain=3)
    factor = FactorAgent(
        "f1",
        domain=3,
        ct_creation_func=lambda *_args, **_kwargs: np.zeros((3, 3)),
        param={},
    )

    var.mailer.set_first_message(owner=var, neighbor=factor)
    assert len(var.mailer.inbox) == 1
    np.testing.assert_allclose(var.mailer.inbox[0].data, np.zeros(3))

    custom_messages = [
        Message(np.array([1.0, 0.0, 1.0]), sender=factor, recipient=var),
        Message(np.array([2.0, 2.0, 2.0]), sender=factor, recipient=var),
    ]
    var.mailer.inbox = custom_messages
    assert len(var.mailer.inbox) == 1  # keyed by sender, deduplicated
    np.testing.assert_allclose(var.mailer.inbox[0].data, custom_messages[-1].data)

    var.mailer.clear_outgoing()
    assert var.mailer.outbox == []
