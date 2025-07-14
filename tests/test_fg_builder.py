import pytest
import numpy as np
from src.propflow.utils import FGBuilder
from src.propflow.configs import (
    create_random_int_table,
    create_uniform_float_table,
    create_poisson_table,
)
from src.propflow.bp.factor_graph import FactorGraph
from src.propflow.core.agents import VariableAgent, FactorAgent


class TestFGBuilder:
    """Test suite for FGBuilder functionality."""

    @pytest.fixture(params=[2, 3, 4, 5])
    def domain_size(self, request):
        """Parameterize tests with different domain sizes."""
        return request.param

    @pytest.fixture(params=[3, 5, 7, 10])
    def num_vars(self, request):
        """Parameterize tests with different numbers of variables."""
        return request.param

    @pytest.fixture(params=[0.3, 0.5, 0.8])
    def density(self, request):
        """Parameterize tests with different graph densities."""
        return request.param

    @pytest.fixture(
        params=[
            (create_random_int_table, {"low": 1, "high": 10}),
            (create_poisson_table, {"strength": 2.0}),
        ]
    )
    def cost_table_config(self, request):
        """Parameterize tests with different cost table configurations."""
        return request.param

    def test_build_cycle_graph_basic(self, domain_size, num_vars, cost_table_config):
        """Test basic cycle graph construction."""
        ct_factory, ct_params = cost_table_config

        fg = FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory,
            ct_params=ct_params,
        )

        assert isinstance(fg, FactorGraph)
        assert len(fg.variables) == num_vars
        assert (
            len(fg.factors) == num_vars
        )  # Cycle has same number of factors as variables
        assert len(fg.edges) == num_vars

        # Check that all variables have correct domain
        for var in fg.variables:
            assert isinstance(var, VariableAgent)
            assert var.domain == domain_size

        # Check that all factors are properly created
        for factor in fg.factors:
            assert isinstance(factor, FactorAgent)
            assert factor.domain == domain_size

    def test_build_random_graph_basic(
        self, domain_size, num_vars, density, cost_table_config
    ):
        """Test basic random graph construction."""
        ct_factory, ct_params = cost_table_config

        fg = FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=ct_factory,
            ct_params=ct_params,
            density=density,
        )

        assert isinstance(fg, FactorGraph)
        assert len(fg.variables) == num_vars
        assert len(fg.factors) >= 0  # Number of factors depends on density
        assert len(fg.edges) == len(fg.factors)

        # Check that all variables have correct domain
        for var in fg.variables:
            assert isinstance(var, VariableAgent)
            assert var.domain == domain_size

    def test_cycle_graph_structure(self):
        """Test that cycle graph has correct structure."""
        num_vars = 4
        domain_size = 2

        fg = FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )

        # Check variable names
        var_names = [var.name for var in fg.variables]
        expected_names = [f"x{i}" for i in range(1, num_vars + 1)]
        assert set(var_names) == set(expected_names)

        # Check that each variable is connected to exactly 2 factors
        for var in fg.variables:
            connected_factors = [
                f for f, vars_list in fg.edges.items() if var in vars_list
            ]
            assert len(connected_factors) == 2

    def test_random_graph_density_effect(self):
        """Test that density affects the number of edges in random graphs."""
        num_vars = 6
        domain_size = 3

        # Low density
        fg_low = FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
            density=0.2,
        )

        # High density
        fg_high = FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
            density=0.8,
        )

        # Higher density should generally result in more factors
        # (this is probabilistic, so we'll be lenient)
        assert (
            len(fg_high.factors) >= len(fg_low.factors)
            or len(fg_high.factors) >= num_vars // 2
        )

    def test_cost_table_initialization(self):
        """Test that cost tables are properly initialized."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10},
        )

        for factor in fg.factors:
            assert hasattr(factor, "cost_table")
            assert factor.cost_table is not None
            assert factor.cost_table.shape == (2, 2)  # Binary factors with domain 2
            assert np.all(factor.cost_table >= 1)
            assert np.all(factor.cost_table <= 10)

    def test_edge_consistency(self):
        """Test that edges are consistent between factors and variables."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_attractive_table,
            ct_params={"strength": 1.5},
        )

        # Check that all edges reference valid variables
        for factor, variables in fg.edges.items():
            assert factor in fg.factors
            assert len(variables) == 2  # Binary factors
            for var in variables:
                assert var in fg.variables

    def test_different_cost_tables(self):
        """Test that different cost table factories produce different results."""
        num_vars = 3
        domain_size = 2

        # Integer cost table
        fg_int = FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )

        # Float cost table
        fg_float = FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_float_table,
            ct_params={"low": 0.1, "high": 2.0},
        )

        # Attractive cost table
        fg_attractive = FGBuilder.build_cycle_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_attractive_table,
            ct_params={"strength": 2.0},
        )

        # Check that integer tables contain integers
        for factor in fg_int.factors:
            assert factor.cost_table.dtype in [np.int32, np.int64]

        # Check that float tables contain floats
        for factor in fg_float.factors:
            assert factor.cost_table.dtype in [np.float32, np.float64]

        # Check that attractive tables have diagonal preference
        for factor in fg_attractive.factors:
            ct = factor.cost_table
            # Diagonal should be lower cost than off-diagonal
            assert ct[0, 0] < ct[0, 1]
            assert ct[1, 1] < ct[1, 0]

    def test_large_graph_construction(self):
        """Test construction of larger graphs."""
        fg = FGBuilder.build_random_graph(
            num_vars=20,
            domain_size=4,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 100},
            density=0.4,
        )

        assert len(fg.variables) == 20
        assert all(var.domain == 4 for var in fg.variables)
        assert len(fg.factors) > 0
        assert len(fg.edges) == len(fg.factors)

    def test_single_variable_graph(self):
        """Test edge case of single variable graph."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=1,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )

        assert len(fg.variables) == 1
        assert len(fg.factors) == 1
        # In a single variable cycle, the factor connects to itself
        factor = fg.factors[0]
        variables = fg.edges[factor]
        assert len(variables) == 2
        assert variables[0] == variables[1]  # Self-loop

    def test_factor_naming_convention(self):
        """Test that factors follow correct naming convention."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )

        factor_names = [f.name for f in fg.factors]
        # Should have factors like f12, f23, f34, f41
        expected_patterns = ["f12", "f23", "f34", "f41"]
        assert set(factor_names) == set(expected_patterns)

    def test_graph_networkx_structure(self):
        """Test that the underlying NetworkX graph is properly constructed."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )

        # Check that NetworkX graph exists and has correct structure
        assert hasattr(fg, "G")
        assert fg.G is not None
        assert len(fg.G.nodes()) == len(fg.variables) + len(fg.factors)
        assert len(fg.G.edges()) == sum(
            len(vars_list) for vars_list in fg.edges.values()
        )
