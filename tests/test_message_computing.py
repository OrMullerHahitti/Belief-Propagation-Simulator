# Tests for message computing
import pytest
import numpy as np
from bp_base.computators import MaxSumComputator
from bp_base.components import Message
from bp_base.agents import VariableAgent, FactorAgent
from DCOP_base import Agent  # For basic sender/recipient typing if needed


# Fixture for a MaxSumComputator
@pytest.fixture
def max_sum_computator():
    return MaxSumComputator()


# Fixture for dummy agents
@pytest.fixture
def dummy_sender():
    return Agent(name="DummySender", node_type="test")


@pytest.fixture
def dummy_recipient():
    return Agent(name="DummyRecipient", node_type="test")
