import pytest
import numpy as np
import os
from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.engines import (
    SplitEngine,
    DampingEngine,
    CostReductionOnceEngine,
    CostReductionAndDamping,
    DampingAndSplitting
)
from utils.splitting import split_all_factors
from utils.cost_reduction import cost_reduction_all_factors
from utils.damping import damp

# Helper function to create a simple factor graph for testing
def create_simple_factor_graph():
    """Create a simple factor graph for testing."""
    # Create variable agents
    var1 = VariableAgent(name="var1", domain=2)
    var2 = VariableAgent(name="var2", domain=2)

    # Create factor agent with a simple cost table
    cost_table = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = FactorAgent(name="factor", cost_table=cost_table)

    # Set up connection numbers
    factor.connection_number = {"var1": 0, "var2": 1}

    # Create the factor graph
    variable_li = [var1, var2]
    factor_li = [factor]
    edges = {factor: [var1, var2]}

    fg = FactorGraph(variable_li=variable_li, factor_li=factor_li, edges=edges)

    return fg

def test_split_engine():
    """Test that SplitEngine correctly applies splitting."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Get the original number of factors
    original_factor_count = len(fg.factors)

    # Create a SplitEngine with the factor graph
    p = 0.5
    engine = SplitEngine(factor_graph=fg, p=p)

    # Check that the number of factors has doubled (splitting was applied in post_init)
    assert len(fg.factors) == original_factor_count * 2, "SplitEngine should double the number of factors"

    # Check that the factors have the correct names
    assert any(f.name == "factor'" for f in fg.factors), "Split factor should exist"
    assert any(f.name == "factor''" for f in fg.factors), "Split factor should exist"

def test_cost_reduction_once_engine():
    """Test that CostReductionOnceEngine correctly applies cost reduction."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()

    # Create a CostReductionOnceEngine with the factor graph
    p = 0.5
    engine = CostReductionOnceEngine(factor_graph=fg, p=p)

    # Check that the cost table was reduced
    reduced_cost_table = fg.factors[0].cost_table
    np.testing.assert_array_almost_equal(reduced_cost_table, original_cost_table * p)

def test_damping_engine():
    """Test that DampingEngine correctly applies damping."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Create a DampingEngine with the factor graph
    damping_factor = 0.5
    engine = DampingEngine(factor_graph=fg, damping_factor=damping_factor)

    # Create messages for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))

    # Create a message from var to factor
    prev_msg = var.create_message(factor, np.array([1.0, 2.0]))
    var.last_iteration = [prev_msg]

    # Create a new message with different data
    curr_msg = var.create_message(factor, np.array([3.0, 4.0]))
    var.mailer.outbox = [curr_msg]

    # Call post_cycle to apply damping
    engine.post_cycle()

    # Check that the message was damped
    expected_data = (1 - damping_factor) * prev_msg.data + damping_factor * np.array([3.0, 4.0])
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)

def test_cost_reduction_and_damping_engine():
    """Test that CostReductionAndDamping engine correctly applies both operations."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()

    # Create a CostReductionAndDamping engine with the factor graph
    p = 0.5
    damping_factor = 0.5
    engine = CostReductionAndDamping(factor_graph=fg, p=p, damping_factor=damping_factor)

    # Check that the cost table was reduced
    reduced_cost_table = fg.factors[0].cost_table
    np.testing.assert_array_almost_equal(reduced_cost_table, original_cost_table * p)

    # Create messages for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))

    # Create a message from var to factor
    prev_msg = var.create_message(factor, np.array([1.0, 2.0]))
    var.last_iteration = [prev_msg]

    # Create a new message with different data
    curr_msg = var.create_message(factor, np.array([3.0, 4.0]))
    var.mailer.outbox = [curr_msg]

    # Call post_cycle to apply damping
    engine.post_cycle()

    # Check that the message was damped
    expected_data = (1 - damping_factor) * prev_msg.data + damping_factor * np.array([3.0, 4.0])
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)

def test_engine_csv_output():
    """Test that the engine correctly saves iteration:global cost to a CSV file."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Create engines of different types
    engines = [
        CostReductionOnceEngine(factor_graph=fg, p=0.5, name="test_csv_cr"),
        DampingEngine(factor_graph=fg, damping_factor=0.7, name="test_csv_damping"),
        SplitEngine(factor_graph=fg, p=0.3, name="test_csv_split"),
        CostReductionAndDamping(factor_graph=fg, p=0.4, damping_factor=0.6, name="test_csv_cr_damping")
    ]

    for engine in engines:
        # Run the engine for a few iterations
        engine.run(max_iter=3, save_csv=True)

        # Get the expected file path
        engine_type = engine.__class__.__name__
        config_name = engine._generate_config_name()
        csv_path = os.path.join("results", engine_type, f"{config_name}.csv")

        # Check that the CSV file was created
        assert os.path.exists(csv_path), f"CSV file {csv_path} should exist"

        # Read the CSV file and check its contents
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Check that there are at least 3 lines (one for each iteration)
        assert len(lines) >= 3, "CSV file should have at least 3 lines"

        # Check the format of each line
        for line in lines:
            parts = line.strip().split(",")
            assert len(parts) == 2, "Each line should have two values separated by a comma"

            # Check that the first value is an integer (iteration)
            try:
                int(parts[0])
            except ValueError:
                pytest.fail(f"First value '{parts[0]}' should be an integer (iteration)")

            # Check that the second value is a float (global cost)
            try:
                float(parts[1])
            except ValueError:
                pytest.fail(f"Second value '{parts[1]}' should be a float (global cost)")

        # Clean up the file
        os.remove(csv_path)

        # Clean up the directory if it's empty
        engine_dir = os.path.join("results", engine_type)
        if os.path.exists(engine_dir) and not os.listdir(engine_dir):
            os.rmdir(engine_dir)
