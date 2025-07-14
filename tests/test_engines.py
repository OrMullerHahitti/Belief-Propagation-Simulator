import numpy as np
import sys
from src.propflow.bp.factor_graph import FactorGraph
from src.propflow.core import VariableAgent, FactorAgent
from src.propflow.bp.engines_realizations import (
    SplitEngine,
    DampingEngine,
    CostReductionOnceEngine,
    DampingCROnceEngine,
    DampingSCFGEngine,
    DiscountEngine,
    MessagePruningEngine,
    TDEngine,
    TDAndPruningEngine,
)
from src.propflow.core import Message

# Flag to enable verbose output during tests
VERBOSE = True


def verbose_print(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE:
        print(*args, **kwargs)
        sys.stdout.flush()  # Ensure output is flushed immediately


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
    verbose_print("\n=== Testing SplitEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print(f"Created factor graph with {len(fg.factors)} factors")

    # Get the original number of factors
    original_factor_count = len(fg.factors)

    # Create a SplitEngine with the factor graph
    p = 0.5
    verbose_print(f"Creating SplitEngine with split_factor={p}")
    engine = SplitEngine(factor_graph=fg, split_factor=p)
    verbose_print(f"Factor count after splitting: {len(fg.factors)}")

    # Check that the number of factors has doubled (splitting was applied in post_init)
    assert (
        len(fg.factors) == original_factor_count * 2
    ), "SplitEngine should double the number of factors"
    verbose_print("✓ Number of factors successfully doubled")

    # Check that the factors have the correct names
    assert any(f.name == "factor'" for f in fg.factors), "Split factor should exist"
    assert any(f.name == "factor''" for f in fg.factors), "Split factor should exist"
    verbose_print("✓ Split factors have correct naming")


def test_cost_reduction_once_engine():
    """Test that CostReductionOnceEngine correctly applies cost reduction and doubles messages."""
    verbose_print("\n=== Testing CostReductionOnceEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()
    verbose_print(f"Original cost table: \n{original_cost_table}")

    # Create a CostReductionOnceEngine with the factor graph
    p = 0.5
    verbose_print(f"Creating CostReductionOnceEngine with reduction_factor={p}")
    engine = CostReductionOnceEngine(factor_graph=fg, reduction_factor=p)

    # Check that the cost table was reduced
    reduced_cost_table = fg.factors[0].cost_table
    verbose_print(f"Reduced cost table: \n{reduced_cost_table}")
    np.testing.assert_array_almost_equal(reduced_cost_table, original_cost_table * p)
    verbose_print("✓ Cost table successfully reduced")

    # Test message doubling functionality after factor computation
    factor = fg.factors[0]

    # Setup initial messages from variables to factor
    var1 = fg.variables[0]
    var2 = fg.variables[1]

    # Create test messages from variables to factor (incoming to factor)
    var1_to_factor_msg = Message(
        data=np.array([0.5, 0.5]),  # Initial belief - uniform distribution
        sender=var1,
        recipient=factor,
    )
    var2_to_factor_msg = Message(
        data=np.array([0.5, 0.5]),  # Initial belief - uniform distribution
        sender=var2,
        recipient=factor,
    )

    # Add messages to factor's inbox
    factor.mailer.inbox = [var1_to_factor_msg, var2_to_factor_msg]
    verbose_print(f"Added messages to factor inbox from variables")

    # Compute factor messages using the engine's computator (this normally happens before post_factor_compute)
    verbose_print("Computing factor messages using engine's computator...")
    engine.pre_factor_compute(factor)
    factor.compute_messages()
    # Save the initially computed messages (before doubling)

    computed_messages = [msg.data.copy() for msg in factor.mailer.outbox]
    verbose_print(f"Factor computed {len(computed_messages)} messages")
    for i, msg_data in enumerate(computed_messages):
        verbose_print(f"Original computed message {i} data: {msg_data}")

    # Now apply the post_factor_compute method to double the messages
    verbose_print("Applying post_factor_compute to double message data...")

    # Check that the message data has been doubled
    for i, (original, current) in enumerate(
        zip(computed_messages, [msg.data for msg in factor.mailer.outbox])
    ):
        verbose_print(f"Message {i} before doubling: {original}")
        verbose_print(f"Message {i} after doubling: {current}")
        np.testing.assert_array_almost_equal(current, original * 2)

    verbose_print("✓ Message data successfully doubled after computation")

    # Verify the number of messages remains the same (double_messages modifies the data, not duplicates messages)
    verbose_print(f"Number of messages in outbox: {len(factor.mailer.outbox)}")
    assert len(factor.mailer.outbox) == len(
        computed_messages
    ), "The number of messages should remain the same"
    verbose_print("✓ Number of messages in outbox remained the same")


def test_damping_engine():
    """Test that DampingEngine correctly applies damping."""
    verbose_print("\n=== Testing DampingEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Create a DampingEngine with the factor graph
    damping_factor = 0.5
    verbose_print(f"Creating DampingEngine with damping_factor={damping_factor}")
    engine = DampingEngine(factor_graph=fg, damping_factor=damping_factor)

    # Get a variable and factor for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))
    verbose_print(f"Testing with variable: {var.name} and factor: {factor.name}")

    # Create a message from var to factor
    prev_msg = Message(
        data=np.array([1.0, 2.0]),
        sender=var,
        recipient=factor,
    )
    var._history = [[prev_msg]]  # Set last_iteration
    verbose_print(f"Previous message data: {prev_msg.data}")

    # Create a new message with different data
    curr_msg = Message(
        data=np.array([3.0, 4.0]),
        sender=var,
        recipient=factor,
    )
    var.mailer.outbox = [curr_msg]
    verbose_print(f"Current message data before damping: {curr_msg.data}")

    # Apply damping directly using the post_var_compute method
    verbose_print("Applying damping via post_var_compute...")
    engine.post_var_compute(var)

    # Check that the message was damped
    expected_data = damping_factor * prev_msg.data + (1 - damping_factor) * np.array(
        [3.0, 4.0]
    )
    verbose_print(f"Expected damped data: {expected_data}")
    verbose_print(f"Actual damped data: {curr_msg.data}")
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)
    verbose_print("✓ Message correctly damped")


def test_damping_scfg_engine():
    """Test that DampingSCFGEngine correctly applies both damping and splitting."""
    verbose_print("\n=== Testing DampingSCFGEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Get the original number of factors
    original_factor_count = len(fg.factors)
    verbose_print(f"Original factor count: {original_factor_count}")

    # Create a DampingSCFGEngine with the factor graph
    split_factor = 0.5
    damping_factor = 0.6
    verbose_print(
        f"Creating DampingSCFGEngine with split_factor={split_factor}, damping_factor={damping_factor}"
    )
    engine = DampingSCFGEngine(
        factor_graph=fg, split_factor=split_factor, damping_factor=damping_factor
    )
    verbose_print(f"New factor count: {len(fg.factors)}")

    # Check that the number of factors has doubled (splitting was applied in post_init)
    assert (
        len(fg.factors) == original_factor_count * 2
    ), "DampingSCFGEngine should double the number of factors"
    verbose_print("✓ Number of factors successfully doubled")

    # Check that the factors have the correct names
    assert any(f.name == "factor'" for f in fg.factors), "Split factor should exist"
    assert any(f.name == "factor''" for f in fg.factors), "Split factor should exist"
    verbose_print("✓ Split factors have correct naming")

    # Test damping functionality
    # Get a variable and factor for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))
    verbose_print(
        f"Testing damping with variable: {var.name} and factor: {factor.name}"
    )

    # Create a message from var to factor
    prev_msg = Message(
        data=np.array([1.0, 2.0]),
        sender=var,
        recipient=factor,
    )
    var._history = [[prev_msg]]  # Set last_iteration
    verbose_print(f"Previous message data: {prev_msg.data}")

    # Create a new message with different data
    curr_msg = Message(
        data=np.array([3.0, 4.0]),
        sender=var,
        recipient=factor,
    )
    var.mailer.outbox = [curr_msg]
    verbose_print(f"Current message data before damping: {curr_msg.data}")

    # Apply damping directly using the post_var_compute method
    verbose_print("Applying damping via post_var_compute...")
    engine.post_var_compute(var)

    # Check that the message was damped
    expected_data = damping_factor * prev_msg.data + (1 - damping_factor) * np.array(
        [3.0, 4.0]
    )
    verbose_print(f"Expected damped data: {expected_data}")
    verbose_print(f"Actual damped data: {curr_msg.data}")
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)
    verbose_print("✓ Message correctly damped")
    verbose_print("✓ DampingSCFGEngine correctly implements both splitting and damping")


def test_damping_cr_once_engine():
    """Test that DampingCROnceEngine correctly applies both cost reduction and damping."""
    verbose_print("\n=== Testing DampingCROnceEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()
    verbose_print(f"Original cost table: \n{original_cost_table}")

    # Create a DampingCROnceEngine with the factor graph
    reduction_factor = 0.5
    damping_factor = 0.5
    verbose_print(
        f"Creating DampingCROnceEngine with reduction_factor={reduction_factor}, damping_factor={damping_factor}"
    )
    engine = DampingCROnceEngine(
        factor_graph=fg,
        reduction_factor=reduction_factor,
        damping_factor=damping_factor,
    )

    # Check that the cost table was reduced
    reduced_cost_table = fg.factors[0].cost_table
    verbose_print(f"Reduced cost table: \n{reduced_cost_table}")
    np.testing.assert_array_almost_equal(
        reduced_cost_table, original_cost_table * reduction_factor
    )
    verbose_print("✓ Cost table successfully reduced")

    # Test damping functionality
    # Get a variable and factor for testing
    var = next(n for n in fg.G.nodes() if isinstance(n, VariableAgent))
    factor = next(n for n in fg.G.nodes() if isinstance(n, FactorAgent))
    verbose_print(
        f"Testing damping with variable: {var.name} and factor: {factor.name}"
    )

    # Create a message from var to factor
    prev_msg = Message(
        data=np.array([1.0, 2.0]),
        sender=var,
        recipient=factor,
    )
    var._history = [[prev_msg]]  # Set last_iteration
    verbose_print(f"Previous message data: {prev_msg.data}")

    # Create a new message with different data
    curr_msg = Message(
        data=np.array([3.0, 4.0]),
        sender=var,
        recipient=factor,
    )
    var.mailer.outbox = [curr_msg]
    verbose_print(f"Current message data before damping: {curr_msg.data}")

    # Apply damping directly using the post_var_compute method
    verbose_print("Applying damping via post_var_compute...")
    engine.post_var_compute(var)

    # Check that the message was damped
    expected_data = damping_factor * prev_msg.data + (1 - damping_factor) * np.array(
        [3.0, 4.0]
    )
    verbose_print(f"Expected damped data: {expected_data}")
    verbose_print(f"Actual damped data: {curr_msg.data}")
    np.testing.assert_array_almost_equal(curr_msg.data, expected_data)
    verbose_print("✓ Message correctly damped")
    verbose_print(
        "✓ DampingCROnceEngine correctly implements both cost reduction and damping"
    )


def test_discount_engine():
    """Test that DiscountEngine correctly initializes."""
    verbose_print("\n=== Testing DiscountEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Create a DiscountEngine with the factor graph
    verbose_print("Creating DiscountEngine")
    engine = DiscountEngine(factor_graph=fg)

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    verbose_print("✓ Engine initialized correctly")

    # Testing post_factor_cycle would require more complex setup to mock discount_attentive
    # Just ensure the method exists and can be called
    verbose_print("Calling post_factor_cycle method...")
    engine.post_factor_cycle()
    verbose_print("✓ post_factor_cycle method called successfully")


def test_td_engine():
    """Test that TDEngine correctly initializes and sets damping factor."""
    verbose_print("\n=== Testing TDEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Create a TDEngine with the factor graph
    damping_factor = 0.7
    verbose_print(f"Creating TDEngine with damping_factor={damping_factor}")
    engine = TDEngine(factor_graph=fg, damping_factor=damping_factor)

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    assert engine.damping_factor == damping_factor
    verbose_print("✓ Engine initialized correctly with proper damping factor")

    # Testing post_var_cycle would require more complex setup to mock TD
    # Just ensure the method exists and can be called
    verbose_print("Calling post_var_cycle method...")
    engine.post_var_cycle()
    verbose_print("✓ post_var_cycle method called successfully")


def test_message_pruning_engine():
    """Test that MessagePruningEngine correctly initializes with pruning parameters."""
    verbose_print("\n=== Testing MessagePruningEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Create a MessagePruningEngine with the factor graph
    prune_threshold = 1e-5
    min_iterations = 3
    adaptive_threshold = False
    verbose_print(
        f"Creating MessagePruningEngine with: \n  prune_threshold={prune_threshold} \n  min_iterations={min_iterations} \n  adaptive_threshold={adaptive_threshold}"
    )
    engine = MessagePruningEngine(
        factor_graph=fg,
        prune_threshold=prune_threshold,
        min_iterations=min_iterations,
        adaptive_threshold=adaptive_threshold,
    )

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    assert engine.prune_threshold == prune_threshold
    assert engine.min_iterations == min_iterations
    assert engine.adaptive_threshold == adaptive_threshold
    verbose_print("✓ Engine initialized correctly with proper pruning parameters")


def test_td_and_pruning_engine():
    """Test that TDAndPruningEngine correctly initializes with both TD and pruning parameters."""
    verbose_print("\n=== Testing TDAndPruningEngine ===")

    # Create a simple factor graph
    fg = create_simple_factor_graph()
    verbose_print("Created factor graph for testing")

    # Create a TDAndPruningEngine with the factor graph
    damping_factor = 0.8
    prune_threshold = 2e-4
    verbose_print(
        f"Creating TDAndPruningEngine with damping_factor={damping_factor}, prune_threshold={prune_threshold}"
    )
    engine = TDAndPruningEngine(
        factor_graph=fg, damping_factor=damping_factor, prune_threshold=prune_threshold
    )

    # Verify engine is initialized correctly
    assert engine is not None
    assert engine.graph == fg
    assert engine.damping_factor == damping_factor
    assert engine.prune_threshold == prune_threshold
    verbose_print(
        "✓ Engine initialized correctly with proper TD and pruning parameters"
    )

    # Ensure methods from both parent classes are present
    verbose_print("Calling post_var_cycle method from TDEngine parent...")
    engine.post_var_cycle()  # From TDEngine
    verbose_print("✓ Successfully called method from parent class")
    verbose_print(
        "✓ TDAndPruningEngine correctly combines functionality from both parent classes"
    )
