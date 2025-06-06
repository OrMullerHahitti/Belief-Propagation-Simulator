import numpy as np
import os
from bp_base.factor_graph import FactorGraph
from base_all.agents import VariableAgent, FactorAgent
from bp_base.engines_realizations import (
    SplitEngine,
    DampingEngine,
    CostReductionOnceEngine,
    DampingCROnceEngine,
)
from base_all.components import Message


# Helper function to create a simple factor graph for testing
def create_simple_factor_graph():
    """Create a simple factor graph for testing."""
    # Create variable agents
    var1 = VariableAgent(name="var1", domain=2)
    var2 = VariableAgent(name="var2", domain=2)

    # Create factor agent with a simple cost table
    def create_test_cost_table(num_vars=None, domain_size=None, **kwargs):
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    factor = FactorAgent(
        name="factor",
        domain=2,
        ct_creation_func=create_test_cost_table,
    )

    # Set up connection numbers
    factor.connection_number = {"var1": 0, "var2": 1}

    # Initialize the cost table
    # factor.initiate_cost_table()

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
    engine = SplitEngine(factor_graph=fg, split_factor=p)

    # Check that the number of factors has doubled (splitting was applied in post_init)
    assert (
        len(fg.factors) == original_factor_count * 2
    ), "SplitEngine should double the number of factors"

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
    engine = CostReductionOnceEngine(factor_graph=fg, reduction_factor=p)

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

    # Get a variable and factor for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))

    # Create a message from var to factor
    prev_msg = Message(
        data=np.array([1.0, 2.0]),
        sender=var,
        recipient=factor,
    )
    var._history = [[prev_msg]]  # Set last_iteration

    # Create a new message with different data
    curr_msg = Message(
        data=np.array([3.0, 4.0]),
        sender=var,
        recipient=factor,
    )
    var.mailer.outbox = [curr_msg]

    # Apply damping directly using the post_var_compute method
    engine.post_var_compute(var)

    # Check that the message was damped
    expected_data = damping_factor * prev_msg.data + (1 - damping_factor) * np.array(
        [3.0, 4.0]
    )
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)


def test_cost_reduction_and_damping_engine():
    """Test that DampingCROnceEngine correctly applies both cost reduction and damping."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()

    # Create a DampingCROnceEngine with the factor graph
    reduction_factor = 0.5
    damping_factor = 0.5
    engine = DampingCROnceEngine(
        factor_graph=fg, reduction_factor=reduction_factor, damping_factor=damping_factor
    )

    # Trigger cost reduction by simulating post_two_cycles
    engine.post_two_cycles()

    # Check that the cost table was reduced
    reduced_cost_table = fg.factors[0].cost_table
    np.testing.assert_array_almost_equal(reduced_cost_table, original_cost_table * reduction_factor)

    # Get a variable and factor for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))

    # Create a message from var to factor
    prev_msg = Message(
        data=np.array([1.0, 2.0]),
        sender=var,
        recipient=factor
    )
    var._history = [[prev_msg]]  # Set last_iteration

    # Create a new message with different data
    curr_msg = Message(
        data=np.array([3.0, 4.0]),
        sender=var,
        recipient=factor
    )
    var.mailer.outbox = [curr_msg]

    # Apply damping directly using the post_var_compute method
    engine.post_var_compute(var)

    # Check that the message was damped
    expected_data = damping_factor * prev_msg.data + (1 - damping_factor) * np.array([3.0, 4.0])
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)


def test_engine_csv_output():
    """Test that the engine correctly saves iteration:global cost to a CSV file."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Create engines of different types
    engines = [
        CostReductionOnceEngine(factor_graph=fg, reduction_factor=0.5, name="test_csv_cr"),
        DampingEngine(factor_graph=fg, damping_factor=0.7, name="test_csv_damping"),
        SplitEngine(factor_graph=fg, split_factor=0.3, name="test_csv_split"),
        DampingCROnceEngine(
            factor_graph=fg, reduction_factor=0.4, damping_factor=0.6, name="test_csv_cr_damping"
        ),
    ]

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    for engine in engines:
        # Create engine-specific directory
        engine_type = engine.__class__.__name__
        engine_dir = os.path.join("results", engine_type)
        os.makedirs(engine_dir, exist_ok=True)

        # Run the engine for a few iterations
        engine.run(max_iter=3, save_csv=True)

        # Get the expected file path
        config_name = engine._generate_config_name()
        csv_path = os.path.join("results", engine_type, f"{config_name}.csv")

        # Check that the CSV file was created
        assert os.path.exists(csv_path), f"CSV file {csv_path} should exist"

        # Read the CSV file and check its contents
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Check that there is at least 1 line (header)
        assert len(lines) >= 1, "CSV file should have at least 1 line"

        # Check the format of each line
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            # First line might be header in some CSV formats
            if i == 0 and not parts[0].isdigit():
                continue

            # Check that we have values for at least one run
            assert len(parts) >= 1, "Each line should have at least one value"

        # Clean up the file
        os.remove(csv_path)

        # Clean up the directory if it's empty
        if os.path.exists(engine_dir) and not os.listdir(engine_dir):
            os.rmdir(engine_dir)

    # Clean up results directory if it's empty
    if os.path.exists("results") and not os.listdir("results"):
        os.rmdir("results")
