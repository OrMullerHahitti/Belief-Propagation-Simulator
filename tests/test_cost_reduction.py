import pytest
import numpy as np
from bp_base.agents import FactorAgent
from utils.cost_reduction import cost_reduction_all_factors


def test_cost_reduction_all_factors():
    """Test that cost_reduction_all_factors correctly reduces costs."""
    # Create factor agents with cost tables
    cost_table1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    cost_table2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    factor1 = FactorAgent(name="factor1", cost_table=cost_table1.copy())
    factor2 = FactorAgent(name="factor2", cost_table=cost_table2.copy())
    factor3 = FactorAgent(name="factor3", cost_table=None)  # Test with None cost table

    factors = [factor1, factor2, factor3]

    # Apply cost reduction
    reduction_factor = 0.5
    cost_reduction_all_factors(factors, reduction_factor)

    # Check that cost tables are correctly scaled
    np.testing.assert_array_almost_equal(
        factor1.cost_table, cost_table1 * reduction_factor
    )
    np.testing.assert_array_almost_equal(
        factor2.cost_table, cost_table2 * reduction_factor
    )
    assert factor3.cost_table is None, "None cost table should remain None"


def test_cost_reduction_all_factors_zero():
    """Test cost reduction with a zero factor."""
    cost_table = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = FactorAgent(name="factor", cost_table=cost_table.copy())

    # Apply cost reduction with zero
    cost_reduction_all_factors([factor], 0.0)

    # Check that cost table is zeroed
    np.testing.assert_array_almost_equal(factor.cost_table, np.zeros_like(cost_table))


def test_cost_reduction_all_factors_negative():
    """Test cost reduction with a negative factor."""
    cost_table = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = FactorAgent(name="factor", cost_table=cost_table.copy())

    # Apply cost reduction with negative value
    cost_reduction_all_factors([factor], -1.0)

    # Check that cost table is negated
    np.testing.assert_array_almost_equal(factor.cost_table, -cost_table)


def test_cost_reduction_all_factors_empty_list():
    """Test cost reduction with an empty list of factors."""
    # This should not raise any errors
    cost_reduction_all_factors([], 0.5)
