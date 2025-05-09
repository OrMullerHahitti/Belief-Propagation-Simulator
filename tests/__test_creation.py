# Tests for component creation
import pytest
import numpy as np
from DCOP_base import Agent, Computator  # Assuming Agent is in DCOP_base
from bp_base.agents import (
    VariableAgent,
    FactorAgent,
)  # Assuming these are in bp_base.agents
from bp_base.components import Message  # Assuming Message is in bp_base.components
from bp_base.computators import (
    MaxSumComputator,
)  # Assuming this is in bp_base.computators
from bp_base.factor_graph import FactorGraph  # Assuming this is in bp_base.factor_graph
from unittest.mock import (
    MagicMock,
)  # Ensure MagicMock is imported if not already at top level


# Test for basic Agent creation
def test_create_agent():
    agent = Agent(name="TestAgent", node_type="test_type")
    assert agent.name == "TestAgent"
    assert agent.type == "test_type"
    assert agent.computator is None
    assert agent.mailer is None  # Mailer might be initialized later or differently


# Test for VariableAgent creation
def test_create_variable_agent():
    var_agent = VariableAgent(name="V1", domain=2)
    assert var_agent.name == "V1"
    assert var_agent.domain == 2
    assert var_agent.type == "variable"
    assert var_agent.belief is None  # Initial belief


# Test for FactorAgent creation
def test_create_factor_agent():
    # Minimal creation, cost_table and connections would be set later
    factor_agent = FactorAgent(name="F1")
    assert factor_agent.name == "F1"
    assert factor_agent.type == "factor"
    assert factor_agent.cost_table is None


# Test for FactorAgent creation with a cost table
def test_create_factor_agent_with_cost_table():
    f_agent = FactorAgent(name="F_with_cost")
    cost_table_data = np.array([[0.1, 0.9], [0.8, 0.2]])
    # Assuming cost_table can be assigned directly or via a method
    f_agent.cost_table = cost_table_data
    assert np.array_equal(f_agent.cost_table, cost_table_data)


# Test for Message creation
def test_create_message():
    sender_agent = Agent(name="Sender", node_type="test")
    recipient_agent = Agent(name="Recipient", node_type="test")
    message_data = np.array([0.5, 0.5])
    msg = Message(data=message_data, sender=sender_agent, recipient=recipient_agent)
    assert np.array_equal(msg.data, message_data)
    assert msg.sender == sender_agent
    assert msg.recipient == recipient_agent


# Test for MaxSumComputator creation
def test_create_max_sum_computator():
    computator = MaxSumComputator()
    assert isinstance(
        computator, Computator
    )  # Check if it's an instance of the base Computator


# Test for FactorGraph creation (basic)
def test_create_factor_graph():
    pass


# Test for FactorGraph creation with nodes and edges
def test_create_factor_graph_with_elements():
    pass

    # assert v1 in fg.get_variable_agents()
    # assert f1 in fg.get_factor_agents()
    # assert len(fg.G.nodes) == 3
    #
    # fg.add_edge(v1, f1)
    # fg.add_edge(v2, f1)
    #
    # assert fg.G.has_edge(v1, f1)
    # assert fg.G.has_edge(v2, f1)


# Test FactorGraph creation with the constructor expecting lists and edge dict
def test_create_factor_graph_with_constructor_lists():
    v1 = VariableAgent(name="V1", domain_size=2)
    v2 = VariableAgent(name="V2", domain_size=2)
    f1 = FactorAgent(name="F1")

    # Mock initiate_cost_table and set_first_message to avoid side effects / complex setup
    f1.initiate_cost_table = lambda: None
    v1.mailer = MagicMock()
    v2.mailer = MagicMock()
    f1.mailer = MagicMock()  # Factors also have mailers

    variable_list = [v1, v2]
    factor_list = [f1]
    edges_dict = {f1: [v1, v2]}

    fg = FactorGraph(variable_li=variable_list, factor_li=factor_list, edges=edges_dict)

    assert len(fg.G.nodes) == 3
    assert v1 in fg.G.nodes
    assert v2 in fg.G.nodes
    assert f1 in fg.G.nodes

    assert fg.G.nodes[v1]["bipartite"] == 0
    assert fg.G.nodes[f1]["bipartite"] == 1

    assert fg.G.has_edge(f1, v1)
    assert fg.G.has_edge(f1, v2)

    assert f1.connection_number[v1] == 0
    assert f1.connection_number[v2] == 1


# Test FactorGraph creation with empty lists
def test_create_empty_factor_graph():
    fg = FactorGraph(variable_li=[], factor_li=[], edges={})
    assert len(fg.G.nodes) == 0
    assert len(fg.G.edges) == 0
    assert fg.diameter == 0
    assert len(fg.get_variable_agents()) == 0
    assert len(fg.get_factor_agents()) == 0


# Test FactorGraph creation with invalid edges (e.g., Var-Var)
def test_create_factor_graph_invalid_bipartite_edge():
    v1 = VariableAgent(name="V1", domain_size=2)
    v2 = VariableAgent(name="V2", domain_size=2)
    v3 = VariableAgent(name="V3", domain_size=2)  # Another variable agent
    f1 = FactorAgent(name="F1")

    # Mock methods as before
    f1.initiate_cost_table = lambda: None
    v1.mailer = MagicMock()
    v2.mailer = MagicMock()
    v3.mailer = MagicMock()
    f1.mailer = MagicMock()

    # Invalid edge: v1 connected to v3 (both are variable agents)
    f2 = FactorAgent(name="F2")
    f2.initiate_cost_table = lambda: None
    f2.mailer = MagicMock()

    with pytest.raises(
        ValueError, match="Edges must connect a factor node to a variable node"
    ):
        FactorGraph(
            variable_li=[v1], factor_li=[f1, f2], edges={f1: [f2]}
        )  # f2 is not a VariableAgent

    with pytest.raises(
        ValueError, match="Edges must connect a factor node to a variable node"
    ):
        FactorGraph(
            variable_li=[v1], factor_li=[f1], edges={f1: [f1]}
        )  # f1 (a factor) in variable list for edge


# Test get_variable_agents and get_factor_agents
def test_get_agents_from_factor_graph():
    v1 = VariableAgent(name="V1", domain_size=2)
    f1 = FactorAgent(name="F1")
    f1.initiate_cost_table = lambda: None
    v1.mailer = MagicMock()
    f1.mailer = MagicMock()

    fg = FactorGraph(variable_li=[v1], factor_li=[f1], edges={f1: [v1]})
    assert fg.get_variable_agents() == [v1]
    assert fg.get_factor_agents() == [f1]


# Test FactorGraph diameter property
def test_factor_graph_diameter():
    v1 = VariableAgent(name="V1", domain_size=2)
    v2 = VariableAgent(name="V2", domain_size=2)
    f1 = FactorAgent(name="F1")
    f1.initiate_cost_table = lambda: None
    v1.mailer = MagicMock()
    v2.mailer = MagicMock()
    f1.mailer = MagicMock()

    # Path: V1 - F1 - V2. Diameter should be 2.
    fg1 = FactorGraph(variable_li=[v1, v2], factor_li=[f1], edges={f1: [v1, v2]})
    assert fg1.diameter == 2

    # Disconnected graph: (V1-F1) and (V2). Diameter is of the largest component (V1-F1, diameter 1)
    v3 = VariableAgent(name="V3", domain_size=2)
    f2 = FactorAgent(name="F2")
    f2.initiate_cost_table = lambda: None
    v3.mailer = MagicMock()
    f2.mailer = MagicMock()

    fg2 = FactorGraph(
        variable_li=[v1, v3], factor_li=[f1], edges={f1: [v1]}
    )  # V3 is disconnected
    assert fg2.diameter == 1

    # Graph: V1 - F1, V2 - F1. Still diameter 2.
    v1_again = VariableAgent(name="V1_again", domain_size=2)
    v2_again = VariableAgent(name="V2_again", domain_size=2)
    f1_again = FactorAgent(name="F1_again")
    f1_again.initiate_cost_table = lambda: None
    v1_again.mailer = MagicMock()
    v2_again.mailer = MagicMock()
    f1_again.mailer = MagicMock()
    fg3 = FactorGraph(
        variable_li=[v1_again, v2_again],
        factor_li=[f1_again],
        edges={f1_again: [v1_again, v2_again]},
    )
    assert fg3.diameter == 2

    # Test FactorGraph.set_computator
    computator = MaxSumComputator()
    fg3.set_computator(computator)
    for node in fg3.G.nodes():
        assert node.computator == computator


# TODO: Add more specific creation tests as needed, e.g., FactorGraph with nodes/edges
