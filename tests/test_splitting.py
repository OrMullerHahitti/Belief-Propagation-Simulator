import pytest
import numpy as np
from propflow.bp.factor_graph import FactorGraph
from propflow.core import VariableAgent, FactorAgent
from propflow.policies import split_all_factors


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


def test_split_all_factors():
    """Test that split_all_factors correctly splits factors."""
    # Create a simple factor graph
    fg = create_simple_factor_graph()

    # Get the original number of factors
    original_factor_count = len(fg.factors)
    original_factor_names = [f.name for f in fg.factors]

    # Get the original cost table
    original_cost_table = fg.factors[0].cost_table.copy()

    # Apply the splitting
    p = 0.5
    split_all_factors(fg, p)

    # Check that the number of factors has doubled
    assert (
        len(fg.factors) == original_factor_count * 2
    ), "Number of factors should double after splitting"

    # Check that the original factor is removed
    for name in original_factor_names:
        assert not any(
            f.name == name for f in fg.factors
        ), f"Original factor {name} should be removed"

    # Check that the new factors have the correct names
    for name in original_factor_names:
        assert any(
            f.name == f"{name}'" for f in fg.factors
        ), f"New factor {name}' should exist"
        assert any(
            f.name == f"{name}''" for f in fg.factors
        ), f"New factor {name}'' should exist"

    # Check that the cost tables are correctly scaled
    for f in fg.factors:
        if f.name.endswith("'"):
            original_name = f.name[:-1]
            expected_cost = p * original_cost_table
            np.testing.assert_array_almost_equal(f.cost_table, expected_cost)
        elif f.name.endswith("''"):
            original_name = f.name[:-2]
            expected_cost = (1 - p) * original_cost_table
            np.testing.assert_array_almost_equal(f.cost_table, expected_cost)


def test_split_all_factors_invalid_p():
    """Test that split_all_factors raises an error for invalid p values."""
    fg = create_simple_factor_graph()

    # Test p = 0
    with pytest.raises(AssertionError):
        split_all_factors(fg, 0.0)

    # Test p = 1
    with pytest.raises(AssertionError):
        split_all_factors(fg, 1.0)

    # Test p < 0
    with pytest.raises(AssertionError):
        split_all_factors(fg, -0.1)

    # Test p > 1
    with pytest.raises(AssertionError):
        split_all_factors(fg, 1.1)
