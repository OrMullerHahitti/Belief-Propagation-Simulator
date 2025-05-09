# Tests for BPEngine units (Step, Cycle, History)
import pytest
import numpy as np
from bp_base.bp_engine import Step, Cycle, History
from bp_base.components import Message # Assuming Message is in bp_base.components
from DCOP_base import Agent # Assuming Agent is in DCOP_base

# Test for Step creation and adding messages
def test_step_creation_and_add_message():
    step = Step(num=1)
    assert step.num == 1
    assert len(step.messages) == 0

    agent1 = Agent(name="Agent1", node_type="test")
    message1_data = np.array([0.1, 0.9])
    msg1 = Message(data=message1_data, sender=agent1, recipient=agent1) # Sender/recipient can be same for simplicity here

    step.add(agent1, msg1)
    assert len(step.messages) == 1
    assert agent1.name in step.messages
    assert len(step.messages[agent1.name]) == 1
    assert step.messages[agent1.name][0] == msg1

    agent2 = Agent(name="Agent2", node_type="test")
    message2_data = np.array([0.8, 0.2])
    msg2 = Message(data=message2_data, sender=agent2, recipient=agent2)
    step.add(agent2, msg2)
    assert len(step.messages) == 2
    assert agent2.name in step.messages

    message3_data = np.array([0.3, 0.7])
    msg3 = Message(data=message3_data, sender=agent1, recipient=agent1) # Another message for agent1
    step.add(agent1, msg3)
    assert len(step.messages[agent1.name]) == 2
    assert step.messages[agent1.name][1] == msg3

# Test for Cycle creation and adding steps
def test_cycle_creation_and_add_step():
    cycle = Cycle(number=1)
    assert cycle.number == 1
    assert len(cycle.steps) == 0
    assert cycle.global_cost is None

    step1 = Step(num=0)
    cycle.add(step1)
    assert len(cycle.steps) == 1
    assert cycle.steps[0] == step1

    step2 = Step(num=1)
    cycle.add(step2)
    assert len(cycle.steps) == 2
    assert cycle.steps[1] == step2

# Test for Cycle equality
def test_cycle_equality():
    agent = Agent(name="TestAgent", node_type="test")
    msg_data = np.array([0.5, 0.5])
    msg = Message(data=msg_data, sender=agent, recipient=agent)

    cycle1 = Cycle(number=1)
    step1_c1 = Step(num=0)
    step1_c1.add(agent, msg)
    cycle1.add(step1_c1)

    cycle2 = Cycle(number=1) # Same number, different content initially
    step1_c2 = Step(num=0)
    step1_c2.add(agent, msg)
    cycle2.add(step1_c2)

    assert cycle1 == cycle2

    cycle3 = Cycle(number=2) # Different number
    step1_c3 = Step(num=0)
    step1_c3.add(agent, msg)
    cycle3.add(step1_c3)
    assert cycle1 != cycle3

    cycle4 = Cycle(number=1)
    step1_c4 = Step(num=0) # Same step number but different message content
    msg_data_diff = np.array([0.6, 0.4])
    msg_diff = Message(data=msg_data_diff, sender=agent, recipient=agent)
    step1_c4.add(agent, msg_diff)
    cycle4.add(step1_c4)
    assert cycle1 != cycle4

    cycle5 = Cycle(number=1)
    step1_c5 = Step(num=0)
    step1_c5.add(agent, msg)
    cycle5.add(step1_c5)
    step2_c5 = Step(num=1) # cycle5 has an extra step
    cycle5.add(step2_c5)
    assert cycle1 != cycle5

# Test for History creation
def test_history_creation():
    history = History(factor_graph_name="TestFG", computator="TestComp")
    assert history.config["factor_graph_name"] == "TestFG"
    assert history.config["computator"] == "TestComp"
    assert len(history.iterations) == 0
    assert len(history.cycles) == 0
    assert len(history.beliefs) == 0
    assert len(history.assignments) == 0

# Test for History item getting/setting (iterations)
def test_history_iteration_item_access():
    history = History()
    step0 = Step(num=0)
    history[0] = step0 # Test __setitem__
    assert len(history.iterations) == 1
    assert history.iterations[0] == step0
    retrieved_step0 = history[0] # Test __getitem__
    assert retrieved_step0 == step0

# Test for History compare_last_two_iterations
def test_history_compare_last_two_iterations():
    history = History()
    assert not history.compare_last_two_iterations() # Not enough assignments

    history.assignments[0] = {"V1": 1, "V2": 0}
    assert not history.compare_last_two_iterations() # Still not enough

    history.assignments[1] = {"V1": 1, "V2": 0} # Same as previous
    assert history.compare_last_two_iterations()

    history.assignments[2] = {"V1": 0, "V2": 1} # Different from previous
    assert not history.compare_last_two_iterations()

    history.assignments[3] = {"V1": 1, "V2": 0} # Same as iteration 1, different from 2
    assert not history.compare_last_two_iterations()

    # Test with different order of keys but same values
    history.assignments[4] = {"V2": 0, "V1": 1} # Same values as iteration 3
    assert history.compare_last_two_iterations()

# Test for History.name property
def test_history_name_property():
    history1 = History(factor_graph_name="MyFG", computator="MyComp", policies_repr="PolicyA")
    assert history1.name == "MyFG_MyComp_PolicyA"

    history2 = History(factor_graph_name="AnotherFG")
    assert history2.name == "AnotherFG_UnknownComputator_" # Default computator and empty policies_repr

    history3 = History()
    assert history3.name == "UnknownFactorGraph_UnknownComputator_"

# Tests for History._normalize_for_json
def test_history_normalize_for_json():
    # Test with basic types (should remain unchanged)
    assert History._normalize_for_json(10) == 10
    assert History._normalize_for_json("test") == "test"
    assert History._normalize_for_json([1, "a"]) == [1, "a"]
    assert History._normalize_for_json({"key": "value"}) == {"key": "value"}

    # Test with numpy generic types
    assert History._normalize_for_json(np.int64(5)) == 5
    assert History._normalize_for_json(np.float32(3.14)) == pytest.approx(3.14, abs=1e-6)

    # Test with numpy arrays
    assert History._normalize_for_json(np.array([1, 2, 3])) == [1, 2, 3]
    assert History._normalize_for_json(np.array([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]

    # Test with a simple dataclass (like Step or Cycle, if they were simpler)
    @pytest.fixture # Using a fixture to define a simple dataclass for testing
    def simple_dataclass_instance():
        from dataclasses import dataclass
        @dataclass
        class DummyData:
            x: int
            y: str
        return DummyData(x=1, y="hello")

    # Test with the simple_dataclass_instance
    # This part of the test needs to be within a test function that can use the fixture.
    # We'll create a new test function specifically for this.

    # Test with nested structures
    nested_obj = {
        "a": np.array([1, 2]),
        "b": {"c": np.int64(3), "d": [4, np.float32(5.0)]},
    }
    expected_normalized_nested = {
        "a": [1, 2],
        "b": {"c": 3, "d": [4, 5.0]},
    }
    normalized_nested = History._normalize_for_json(nested_obj)
    assert normalized_nested["a"] == expected_normalized_nested["a"]
    assert normalized_nested["b"]["c"] == expected_normalized_nested["b"]["c"]
    assert normalized_nested["b"]["d"][0] == expected_normalized_nested["b"]["d"][0]
    assert normalized_nested["b"]["d"][1] == pytest.approx(expected_normalized_nested["b"]["d"][1])


def test_history_normalize_dataclass(simple_dataclass_instance):
    """Test normalization of a simple dataclass."""
    normalized_dc = History._normalize_for_json(simple_dataclass_instance)
    assert normalized_dc == {"x": 1, "y": "hello"}

    # Test with a Step object (which is a dataclass)
    step_instance = Step(num=1)
    agent_for_step = Agent(name="A1", node_type="var")
    msg_for_step = Message(data=np.array([0.5,0.5]), sender=agent_for_step, recipient=agent_for_step)
    step_instance.add(agent_for_step, msg_for_step)
    
    normalized_step = History._normalize_for_json(step_instance)
    assert normalized_step["num"] == 1
    assert "messages" in normalized_step
    assert agent_for_step.name in normalized_step["messages"]
    # Message itself might be complex, check its data part
    normalized_msg_data = normalized_step["messages"][agent_for_step.name][0]["data"]
    assert normalized_msg_data == [0.5, 0.5]


# Tests for History.save_results
from unittest.mock import patch, mock_open, MagicMock
import json
import os

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_history_save_results_default_filename(mock_json_dump, mock_file_open, mock_os_makedirs):
    history = History(factor_graph_name="TestFG", computator="TestComp", policies_repr="TestPolicies")
    history.iterations[0] = Step(num=0) # Add some data to ensure filename reflects it
    
    expected_filename = "TestFG_TestComp_TestPolicies_details_iter_1.json"
    
    returned_filename = history.save_results()

    mock_os_makedirs.assert_called_once_with(os.path.dirname(expected_filename) or ".", exist_ok=True)
    mock_file_open.assert_called_once_with(expected_filename, "w")
    
    # Check that json.dump was called with data that has been normalized
    # The first argument to json.dump is the data.
    # We need to ensure that the normalization logic was applied.
    # For simplicity, we'll check a few key aspects of the dumped data.
    
    args, kwargs = mock_json_dump.call_args
    dumped_data = args[0]

    assert dumped_data["name"] == "TestFG_TestComp_TestPolicies"
    assert "config_summary" in dumped_data
    assert "iterations_data" in dumped_data
    assert "0" in dumped_data["iterations_data"] # Iteration 0 should be there
    assert dumped_data["iterations_data"]["0"]["num"] == 0 # Step num normalized
    
    assert returned_filename == expected_filename

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_history_save_results_custom_filename(mock_json_dump, mock_file_open, mock_os_makedirs):
    history = History()
    custom_filename = "my_custom_history.json"
    
    returned_filename = history.save_results(filename=custom_filename)

    mock_os_makedirs.assert_called_once_with(os.path.dirname(custom_filename) or ".", exist_ok=True)
    mock_file_open.assert_called_once_with(custom_filename, "w")
    mock_json_dump.assert_called_once() # Check it was called
    assert returned_filename == custom_filename

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_history_save_results_with_complex_data(mock_json_dump, mock_file_open, mock_os_makedirs):
    history = History(factor_graph_name="ComplexFG")
    
    # Populate with more complex data including numpy arrays and dataclasses (Step/Message)
    step0 = Step(num=0)
    agent1 = Agent(name="V1", node_type="var")
    msg1_data = np.array([0.1, 0.9])
    msg1 = Message(data=msg1_data, sender=agent1, recipient=agent1)
    step0.add(agent1, msg1)
    history.iterations[0] = step0
    
    history.beliefs[0] = {"V1": np.array([0.5, 0.5]), "V2": np.array([1.0, 0.0])}
    history.assignments[0] = {"V1": np.int64(0), "V2": 1} # Use a numpy int

    # Mock the computator and factor_graph in config for more thorough testing of config serialization
    mock_computator = MagicMock()
    mock_computator.__class__.__name__ = "MockComputator"
    mock_fg = MagicMock()
    mock_fg.name = "MockFactorGraph"
    
    from bp_base.typing import PolicyType # Assuming PolicyType is an Enum
    # Mock a policy (needs a __class__.__name__)
    class MockPolicy: pass
    mock_policy_instance = MockPolicy()
    
    history.config['computator'] = mock_computator
    history.config['factor_graph'] = mock_fg
    history.config['policies'] = {PolicyType.MESSAGE: [mock_policy_instance]}


    filename = "complex_history.json"
    history.save_results(filename=filename)

    mock_file_open.assert_called_with(filename, "w")
    args, _ = mock_json_dump.call_args
    dumped_data = args[0]

    # Check config serialization
    assert dumped_data["config_summary"]["factor_graph_name"] == "ComplexFG" # from constructor
    assert dumped_data["config_summary"]["computator"] == "MockComputator"
    assert dumped_data["config_summary"]["factor_graph"] == "MockFactorGraph"
    assert dumped_data["config_summary"]["policies"][PolicyType.MESSAGE.value][0] == "MockPolicy"


    # Check iterations_data normalization
    assert dumped_data["iterations_data"]["0"]["num"] == 0
    assert dumped_data["iterations_data"]["0"]["messages"]["V1"][0]["data"] == [0.1, 0.9]
    
    # Check beliefs_per_iteration normalization
    assert dumped_data["beliefs_per_iteration"]["0"]["V1"] == [0.5, 0.5]
    
    # Check assignments_per_iteration normalization
    assert dumped_data["assignments_per_iteration"]["0"]["V1"] == 0 # np.int64 should be converted to int
