"""
Comprehensive tests for core agent functionality.

This module tests the core agents and components:
- FGAgent abstract base class
- VariableAgent implementation
- FactorAgent implementation
- Message class
- MailHandler functionality
- Agent interactions and message passing
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from propflow.core.agents import FGAgent, VariableAgent, FactorAgent
from propflow.core.components import Message, MailHandler, CostTable
from propflow.core.dcop_base import Agent


@pytest.mark.unit
class TestMessage:
    """Test the Message class functionality."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        sender = Mock(spec=Agent)
        sender.name = "sender_agent"
        sender.type = "variable"

        recipient = Mock(spec=Agent)
        recipient.name = "recipient_agent"
        recipient.type = "factor"

        return sender, recipient

    def test_message_initialization(self, sample_agents):
        """Test Message initialization."""
        sender, recipient = sample_agents
        data = np.array([1.0, 2.0, 3.0])

        message = Message(data, sender, recipient)

        assert np.array_equal(message.data, data)
        assert message.sender == sender
        assert message.recipient == recipient

    def test_message_copy(self, sample_agents):
        """Test Message copy functionality."""
        sender, recipient = sample_agents
        original_data = np.array([1.0, 2.0, 3.0])

        message = Message(original_data, sender, recipient)
        copied_message = message.copy()

        # Data should be equal but not the same object
        assert np.array_equal(copied_message.data, message.data)
        assert copied_message.data is not message.data

        # Agents should be the same objects (not copied)
        assert copied_message.sender is message.sender
        assert copied_message.recipient is message.recipient

    def test_message_equality(self, sample_agents):
        """Test Message equality based on sender and recipient names."""
        sender, recipient = sample_agents
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])

        message1 = Message(data1, sender, recipient)
        message2 = Message(data2, sender, recipient)  # Different data, same agents

        # Should be equal based on agent names, not data
        assert message1 == message2

    def test_message_inequality(self, sample_agents):
        """Test Message inequality with different agents."""
        sender, recipient = sample_agents

        # Create different recipient
        other_recipient = Mock(spec=Agent)
        other_recipient.name = "other_recipient"
        other_recipient.type = "variable"

        data = np.array([1.0, 2.0, 3.0])
        message1 = Message(data, sender, recipient)
        message2 = Message(data, sender, other_recipient)

        assert message1 != message2

    def test_message_hash(self, sample_agents):
        """Test Message hash based on agent names."""
        sender, recipient = sample_agents
        data = np.array([1.0, 2.0, 3.0])

        message1 = Message(data, sender, recipient)
        message2 = Message(data, sender, recipient)

        assert hash(message1) == hash(message2)

    def test_message_string_representation(self, sample_agents):
        """Test Message string and repr methods."""
        sender, recipient = sample_agents
        data = np.array([1.0, 2.0])

        message = Message(data, sender, recipient)

        str_repr = str(message)
        assert "sender_agent" in str_repr
        assert "recipient_agent" in str_repr
        assert "[1. 2.]" in str_repr

        assert repr(message) == str(message)


@pytest.mark.unit
class TestMailHandler:
    """Test the MailHandler class functionality."""

    @pytest.fixture
    def domain_size(self):
        return 3

    @pytest.fixture
    def mail_handler(self, domain_size):
        return MailHandler(domain_size)

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        agent1 = Mock(spec=Agent)
        agent1.name = "agent1"
        agent1.type = "variable"

        agent2 = Mock(spec=Agent)
        agent2.name = "agent2"
        agent2.type = "factor"

        return agent1, agent2

    def test_mailhandler_initialization(self, domain_size):
        """Test MailHandler initialization."""
        handler = MailHandler(domain_size)

        assert handler._message_domain_size == domain_size
        assert handler._incoming == {}
        assert handler._outgoing == []
        assert handler._clear_after_staging is True

    def test_make_key(self, mail_handler, sample_agents):
        """Test _make_key method for creating unique agent keys."""
        agent1, agent2 = sample_agents

        key1 = mail_handler._make_key(agent1)
        key2 = mail_handler._make_key(agent2)

        assert key1 == "agent1_variable"
        assert key2 == "agent2_factor"
        assert key1 != key2

    def test_set_first_message(self, mail_handler, sample_agents, domain_size):
        """Test set_first_message initialization."""
        agent1, agent2 = sample_agents

        # Mock FGAgent
        owner = Mock()
        owner.name = "owner"

        mail_handler.set_first_message(owner, agent1)

        key = mail_handler._make_key(agent1)
        assert key in mail_handler._incoming

        message = mail_handler._incoming[key]
        assert np.array_equal(message.data, np.zeros(domain_size))
        assert message.sender == agent1
        assert message.recipient == owner

    def test_receive_single_message(self, mail_handler, sample_agents):
        """Test receiving a single message."""
        agent1, agent2 = sample_agents
        data = np.array([1.0, 2.0, 3.0])

        message = Message(data, agent1, agent2)
        mail_handler.receive_messages(message)

        key = mail_handler._make_key(agent1)
        assert key in mail_handler._incoming
        assert mail_handler._incoming[key] == message

    def test_receive_multiple_messages(self, mail_handler, sample_agents):
        """Test receiving multiple messages."""
        agent1, agent2 = sample_agents

        # Create another agent
        agent3 = Mock(spec=Agent)
        agent3.name = "agent3"
        agent3.type = "variable"

        messages = [
            Message(np.array([1.0, 2.0, 3.0]), agent1, agent2),
            Message(np.array([4.0, 5.0, 6.0]), agent3, agent2),
        ]

        mail_handler.receive_messages(messages)

        key1 = mail_handler._make_key(agent1)
        key3 = mail_handler._make_key(agent3)

        assert key1 in mail_handler._incoming
        assert key3 in mail_handler._incoming
        assert len(mail_handler._incoming) == 2

    def test_message_overwriting(self, mail_handler, sample_agents):
        """Test that newer messages overwrite older ones from the same sender."""
        agent1, agent2 = sample_agents

        # Send first message
        message1 = Message(np.array([1.0, 2.0, 3.0]), agent1, agent2)
        mail_handler.receive_messages(message1)

        # Send second message from same sender
        message2 = Message(np.array([4.0, 5.0, 6.0]), agent1, agent2)
        mail_handler.receive_messages(message2)

        key = mail_handler._make_key(agent1)
        assert mail_handler._incoming[key] == message2  # Should be the newer message

    def test_inbox_property(self, mail_handler, sample_agents):
        """Test inbox property returns list of messages."""
        agent1, agent2 = sample_agents
        message = Message(np.array([1.0, 2.0, 3.0]), agent1, agent2)

        mail_handler.receive_messages(message)

        inbox = mail_handler.inbox
        assert isinstance(inbox, list)
        assert len(inbox) == 1
        assert inbox[0] == message

    def test_outbox_property(self, mail_handler):
        """Test outbox property."""
        assert mail_handler.outbox == []

        # Add message to outgoing
        message = Mock(spec=Message)
        mail_handler._outgoing.append(message)

        assert mail_handler.outbox == [message]


@pytest.mark.unit
class TestFGAgent:
    """Test the abstract FGAgent base class."""

    def test_fgagent_is_abstract(self):
        """Test that FGAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FGAgent("test", "variable", 3)

    def test_fgagent_inheritance(self):
        """Test that concrete subclasses can be instantiated."""
        # VariableAgent should inherit from FGAgent
        var_agent = VariableAgent("var1", 3)
        assert isinstance(var_agent, FGAgent)

        # FactorAgent should inherit from FGAgent
        factor_agent = FactorAgent("factor1", 3, lambda n, d: np.random.rand(d, d))
        assert isinstance(factor_agent, FGAgent)


@pytest.mark.unit
class TestVariableAgent:
    """Test VariableAgent functionality."""

    @pytest.fixture
    def domain_size(self):
        return 3

    @pytest.fixture
    def variable_agent(self, domain_size):
        return VariableAgent("var1", domain_size)

    def test_variable_agent_initialization(self, variable_agent, domain_size):
        """Test VariableAgent initialization."""
        assert variable_agent.name == "var1"
        assert variable_agent.type == "variable"
        assert variable_agent.domain == domain_size
        assert isinstance(variable_agent.mailer, MailHandler)
        assert variable_agent._history == []
        assert variable_agent._max_history == 10

    def test_variable_agent_str_repr(self, variable_agent):
        """Test string and repr methods."""
        assert str(variable_agent) == "VAR1"
        assert repr(variable_agent) == "VariableAgent(var1, domain=3)"

    def test_compute_messages_without_computator(self, variable_agent):
        """Test compute_messages when no computator is set."""
        # Should not raise exception
        variable_agent.compute_messages()

        # No messages should be staged
        assert len(variable_agent.outbox) == 0

    def test_compute_messages_with_computator(self, variable_agent):
        """Test compute_messages with computator."""
        # Mock computator
        mock_computator = Mock()
        mock_messages = [Mock(spec=Message), Mock(spec=Message)]
        mock_computator.compute_Q.return_value = mock_messages

        variable_agent.computator = mock_computator

        # Add some inbox messages
        sender = Mock(spec=Agent)
        sender.name = "sender"
        sender.type = "factor"

        message = Message(np.array([1.0, 2.0, 3.0]), sender, variable_agent)
        variable_agent.mailer.receive_messages(message)

        # Mock stage_sending
        variable_agent.mailer.stage_sending = Mock()

        variable_agent.compute_messages()

        mock_computator.compute_Q.assert_called_once_with(variable_agent.mailer.inbox)
        variable_agent.mailer.stage_sending.assert_called_once_with(mock_messages)

    def test_belief_without_computator(self, variable_agent, domain_size):
        """Test belief computation without computator."""
        # With empty inbox, should return uniform belief
        belief = variable_agent.belief
        expected = np.ones(domain_size) / domain_size
        assert np.allclose(belief, expected)

        # Add some messages to inbox
        sender = Mock(spec=Agent)
        sender.name = "sender"
        sender.type = "factor"

        message1 = Message(np.array([1.0, 2.0, 3.0]), sender, variable_agent)
        variable_agent.mailer.receive_messages(message1)

        belief = variable_agent.belief
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(belief, expected)

    def test_belief_with_computator(self, variable_agent):
        """Test belief computation with computator."""
        mock_computator = Mock()
        mock_belief = np.array([0.1, 0.7, 0.2])
        mock_computator.compute_belief.return_value = mock_belief

        variable_agent.computator = mock_computator

        belief = variable_agent.belief

        mock_computator.compute_belief.assert_called_once_with(
            variable_agent.inbox, variable_agent.domain
        )
        assert np.array_equal(belief, mock_belief)

    def test_curr_assignment_without_computator(self, variable_agent):
        """Test current assignment without computator."""
        # Mock belief to return specific values
        with patch.object(variable_agent, "belief", np.array([3.0, 1.0, 2.0])):
            assignment = variable_agent.curr_assignment
            assert assignment == 1  # Index of minimum value

    def test_curr_assignment_with_computator(self, variable_agent):
        """Test current assignment with computator."""
        mock_computator = Mock()
        mock_assignment = 2
        mock_computator.get_assignment.return_value = mock_assignment

        variable_agent.computator = mock_computator

        with patch.object(
            variable_agent, "belief", np.array([0.1, 0.2, 0.7])
        ) as mock_belief:
            assignment = variable_agent.curr_assignment

            mock_computator.get_assignment.assert_called_once_with(mock_belief)
            assert assignment == mock_assignment

    def test_message_history(self, variable_agent):
        """Test message history functionality."""
        # Initially empty
        assert variable_agent.last_iteration == []
        assert variable_agent.last_cycle() == []

        # Add some mock messages to outbox and save to history
        mock_message = Mock(spec=Message)
        mock_message.copy.return_value = mock_message
        variable_agent.mailer._outgoing = [mock_message]

        variable_agent.append_last_iteration()

        assert len(variable_agent._history) == 1
        assert variable_agent.last_iteration == [mock_message]

    def test_history_size_limit(self, variable_agent):
        """Test that history is limited to max_history size."""
        # Fill history beyond limit
        mock_message = Mock(spec=Message)
        mock_message.copy.return_value = mock_message

        for i in range(15):  # More than _max_history (10)
            variable_agent.mailer._outgoing = [mock_message]
            variable_agent.append_last_iteration()

        assert len(variable_agent._history) == variable_agent._max_history

    def test_last_cycle_with_diameter(self, variable_agent):
        """Test last_cycle with different diameter values."""
        mock_message = Mock(spec=Message)
        mock_message.copy.return_value = mock_message

        # Add multiple iterations
        for i in range(5):
            variable_agent.mailer._outgoing = [mock_message]
            variable_agent.append_last_iteration()

        # Test different diameters
        assert variable_agent.last_cycle(1) == [mock_message]
        assert variable_agent.last_cycle(3) == [mock_message]

        # Test diameter larger than history
        assert variable_agent.last_cycle(10) == []


@pytest.mark.unit
class TestFactorAgent:
    """Test FactorAgent functionality."""

    @pytest.fixture
    def domain_size(self):
        return 3

    @pytest.fixture
    def cost_table_func(self, domain_size):
        """Mock cost table creation function."""

        def create_table(num_vars, domain, **kwargs):
            return np.random.rand(*([domain] * num_vars))

        return create_table

    @pytest.fixture
    def factor_agent(self, domain_size, cost_table_func):
        return FactorAgent("factor1", domain_size, cost_table_func)

    def test_factor_agent_initialization(
        self, factor_agent, domain_size, cost_table_func
    ):
        """Test FactorAgent initialization."""
        assert factor_agent.name == "factor1"
        assert factor_agent.type == "factor"
        assert factor_agent.domain == domain_size
        assert factor_agent.cost_table is None  # Not created initially
        assert factor_agent.connection_number == {}
        assert factor_agent.ct_creation_func == cost_table_func
        assert factor_agent.ct_creation_params == {}
        assert factor_agent._original is None

    def test_factor_agent_with_cost_table(self, domain_size, cost_table_func):
        """Test FactorAgent initialization with existing cost table."""
        cost_table = np.array([[1, 2], [3, 4]])

        factor_agent = FactorAgent(
            "factor1", domain_size, cost_table_func, cost_table=cost_table
        )

        assert np.array_equal(factor_agent.cost_table, cost_table)
        assert factor_agent.cost_table is not cost_table  # Should be a copy

    def test_factor_agent_with_params(self, domain_size, cost_table_func):
        """Test FactorAgent initialization with creation parameters."""
        params = {"low": 0, "high": 10}

        factor_agent = FactorAgent(
            "factor1", domain_size, cost_table_func, param=params
        )

        assert factor_agent.ct_creation_params == params

    def test_create_from_cost_table(self, domain_size):
        """Test FactorAgent.create_from_cost_table class method."""
        cost_table = np.array([[1, 2, 3], [4, 5, 6]])

        factor_agent = FactorAgent.create_from_cost_table("factor1", cost_table)

        assert factor_agent.name == "factor1"
        assert factor_agent.domain == 3  # First dimension
        assert np.array_equal(factor_agent.cost_table, cost_table)

    def test_factor_agent_str_repr(self, factor_agent):
        """Test string and repr methods."""
        assert str(factor_agent) == "FACTOR1"
        assert repr(factor_agent) == "FactorAgent(factor1, connections=[])"

        # Add some connections
        factor_agent.connection_number = {"var1": 0, "var2": 1}
        assert "var1" in repr(factor_agent)
        assert "var2" in repr(factor_agent)

    def test_set_dim_for_variable(self, factor_agent):
        """Test setting dimension for variable connection."""
        var_agent = VariableAgent("var1", 3)

        factor_agent.set_dim_for_variable(var_agent, 0)

        assert factor_agent.connection_number["var1"] == 0

    def test_initiate_cost_table_success(self, factor_agent):
        """Test successful cost table creation."""
        # Set up connections
        factor_agent.connection_number = {"var1": 0, "var2": 1}

        factor_agent.initiate_cost_table()

        assert factor_agent.cost_table is not None
        assert factor_agent.cost_table.shape == (3, 3)  # domain x domain for 2 vars

    def test_initiate_cost_table_already_exists(self, factor_agent):
        """Test error when cost table already exists."""
        factor_agent.cost_table = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Cost table already exists"):
            factor_agent.initiate_cost_table()

    def test_initiate_cost_table_no_connections(self, factor_agent):
        """Test error when no connections are set."""
        with pytest.raises(ValueError, match="No connections set"):
            factor_agent.initiate_cost_table()

    def test_set_name_for_factor(self, factor_agent):
        """Test automatic name generation based on connections."""
        factor_agent.connection_number = {"x1": 0, "x2": 1, "x10": 2}

        factor_agent.set_name_for_factor()

        assert factor_agent.name == "f11012_"  # Should sort and concatenate indices

    def test_set_name_for_factor_no_connections(self, factor_agent):
        """Test error when setting name with no connections."""
        with pytest.raises(ValueError, match="No connections set"):
            factor_agent.set_name_for_factor()

    def test_save_original_cost_table(self, factor_agent):
        """Test saving original cost table."""
        cost_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = cost_table

        factor_agent.save_original()

        assert np.array_equal(factor_agent._original, cost_table)
        assert factor_agent._original is not cost_table  # Should be a copy

    def test_save_original_with_parameter(self, factor_agent):
        """Test saving original cost table with parameter."""
        original_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = np.array([[5, 6], [7, 8]])

        factor_agent.save_original(original_table)

        assert np.array_equal(factor_agent._original, original_table)

    def test_save_original_already_saved(self, factor_agent):
        """Test that original is not overwritten if already saved."""
        cost_table1 = np.array([[1, 2], [3, 4]])
        cost_table2 = np.array([[5, 6], [7, 8]])

        factor_agent.cost_table = cost_table1
        factor_agent.save_original()

        factor_agent.cost_table = cost_table2
        factor_agent.save_original()  # Should not overwrite

        assert np.array_equal(factor_agent._original, cost_table1)

    def test_mean_cost_property(self, factor_agent):
        """Test mean_cost property."""
        # With no cost table
        assert factor_agent.mean_cost == 0.0

        # With cost table
        cost_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = cost_table

        expected_mean = np.mean(cost_table)
        assert factor_agent.mean_cost == expected_mean

    def test_total_cost_property(self, factor_agent):
        """Test total_cost property."""
        # With no cost table
        assert factor_agent.total_cost == 0.0

        # With cost table
        cost_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = cost_table

        expected_sum = np.sum(cost_table)
        assert factor_agent.total_cost == expected_sum

    def test_original_cost_table_property(self, factor_agent):
        """Test original_cost_table property."""
        assert factor_agent.original_cost_table is None

        cost_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = cost_table
        factor_agent.save_original()

        assert np.array_equal(factor_agent.original_cost_table, cost_table)

    def test_compute_messages_without_computator(self, factor_agent):
        """Test compute_messages when no computator is set."""
        factor_agent.cost_table = np.array([[1, 2], [3, 4]])

        # Should not raise exception
        factor_agent.compute_messages()

        # No messages should be staged
        assert len(factor_agent.outbox) == 0

    def test_compute_messages_with_computator(self, factor_agent):
        """Test compute_messages with computator."""
        # Set up factor agent
        cost_table = np.array([[1, 2], [3, 4]])
        factor_agent.cost_table = cost_table

        # Mock computator
        mock_computator = Mock()
        mock_messages = [Mock(spec=Message), Mock(spec=Message)]
        mock_computator.compute_R.return_value = mock_messages

        factor_agent.computator = mock_computator

        # Add some inbox messages
        sender = Mock(spec=Agent)
        sender.name = "sender"
        sender.type = "variable"

        message = Message(np.array([1.0, 2.0]), sender, factor_agent)
        factor_agent.mailer.receive_messages(message)

        # Mock stage_sending
        factor_agent.mailer.stage_sending = Mock()

        factor_agent.compute_messages()

        mock_computator.compute_R.assert_called_once_with(
            cost_table=cost_table, incoming_messages=factor_agent.inbox
        )
        factor_agent.mailer.stage_sending.assert_called_once_with(mock_messages)


@pytest.mark.integration
class TestAgentInteractions:
    """Integration tests for agent interactions."""

    def test_variable_to_factor_message_flow(self):
        """Test message flow from variable to factor agent."""
        var_agent = VariableAgent("var1", 3)
        factor_agent = FactorAgent("factor1", 3, lambda n, d: np.ones((d, d)))

        # Create message from variable to factor
        message_data = np.array([1.0, 2.0, 3.0])
        message = Message(message_data, var_agent, factor_agent)

        # Factor receives message
        factor_agent.receive_message(message)

        # Check that message is in factor's inbox
        assert len(factor_agent.inbox) == 1
        assert factor_agent.inbox[0] == message

    def test_factor_to_variable_message_flow(self):
        """Test message flow from factor to variable agent."""
        var_agent = VariableAgent("var1", 3)
        factor_agent = FactorAgent("factor1", 3, lambda n, d: np.ones((d, d)))

        # Create message from factor to variable
        message_data = np.array([0.1, 0.5, 0.4])
        message = Message(message_data, factor_agent, var_agent)

        # Variable receives message
        var_agent.receive_message(message)

        # Check that message is in variable's inbox
        assert len(var_agent.inbox) == 1
        assert var_agent.inbox[0] == message

    def test_agent_mailbox_operations(self):
        """Test agent mailbox operations."""
        var_agent = VariableAgent("var1", 3)

        # Add message to inbox
        sender = Mock(spec=Agent)
        sender.name = "sender"
        sender.type = "factor"

        message = Message(np.array([1.0, 2.0, 3.0]), sender, var_agent)
        var_agent.receive_message(message)

        assert len(var_agent.inbox) == 1

        # Clear mailbox
        var_agent.empty_mailbox()
        assert len(var_agent.inbox) == 0

    def test_agent_history_tracking(self):
        """Test that agents properly track message history."""
        var_agent = VariableAgent("var1", 3)

        # Mock outgoing messages
        mock_message1 = Mock(spec=Message)
        mock_message1.copy.return_value = mock_message1
        mock_message2 = Mock(spec=Message)
        mock_message2.copy.return_value = mock_message2

        # First iteration
        var_agent.mailer._outgoing = [mock_message1]
        var_agent.append_last_iteration()

        # Second iteration
        var_agent.mailer._outgoing = [mock_message2]
        var_agent.append_last_iteration()

        # Check history
        assert len(var_agent._history) == 2
        assert var_agent.last_iteration == [mock_message2]
        assert var_agent.last_cycle(2) == [mock_message1]


@pytest.mark.unit
@pytest.mark.parametrize("domain_size", [2, 3, 5, 10])
class TestAgentParametrized:
    """Parametrized tests for agents with different domain sizes."""

    def test_variable_agent_with_different_domains(self, domain_size):
        """Test VariableAgent with different domain sizes."""
        var_agent = VariableAgent(f"var_{domain_size}", domain_size)

        assert var_agent.domain == domain_size
        assert var_agent.mailer._message_domain_size == domain_size

        # Test uniform belief
        uniform_belief = var_agent.belief
        expected = np.ones(domain_size) / domain_size
        assert np.allclose(uniform_belief, expected)

    def test_factor_agent_with_different_domains(self, domain_size):
        """Test FactorAgent with different domain sizes."""

        def create_table(n, d):
            return np.random.rand(*([d] * n))

        factor_agent = FactorAgent(f"factor_{domain_size}", domain_size, create_table)

        assert factor_agent.domain == domain_size
        assert factor_agent.mailer._message_domain_size == domain_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
