import logging
import sys

import numpy as np
from bp_base.agents import VariableAgent
from bp_base.components import Message, MailHandler
from policies.damping import ConstantDampingPolicy

logger = logging.getLogger("test_constant_damping_policy")
logger.setLevel(logging.INFO)
logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.propagate = False


def test_constant_damping_policy_logs():
    # Create a VariableAgent with domain size 3
    var = VariableAgent("x1", domain=3)
    var.mailer = MailHandler(3)

    # Fake last_iteration and inbox messages
    last_messages = [Message(data=np.array([1.0, 2.0, 3.0]), sender=var, recipient=var)]
    inbox_messages = [
        Message(data=np.array([4.0, 5.0, 6.0]), sender=var, recipient=var)
    ]
    var.mailer._incoming = inbox_messages
    var._history = [last_messages]

    logger.info(
        f"Before damping: last_iteration={ [m.data for m in var.last_iteration] }, inbox={ [m.data for m in var.mailer.inbox] }"
    )

    # Apply constant damping policy
    policy = ConstantDampingPolicy(damping_value=0.5)
    damped = policy(var)

    logger.info(f"After damping: {damped}")

    # Check the result is as expected
    expected = [0.5 * last_messages[0].data + 0.5 * inbox_messages[0].data]
    assert np.array_equal(
        damped[0].data, expected[0]
    ), f"Expected {expected}, got {damped}"
