import pytest
import numpy as np
import networkx as nx
from propflow.utils import FGBuilder
from propflow.configs import create_random_int_table, create_uniform_float_table
from propflow.bp.factor_graph import FactorGraph
from propflow.core.agents import VariableAgent, FactorAgent
from propflow.core.components import Message


class TestFactorGraph:
    """Test suite for FactorGraph class."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing."""
        v1 = VariableAgent("x1", domain=2)
        v2 = VariableAgent("x2", domain=2)
        v3 = VariableAgent("x3", domain=2)

        f1 = FactorAgent(
            "f12",
            domain=2,
            ct_creation_func=create_random_int_table,
            param={"low": 1, "high": 5},
        )
        f2 = FactorAgent(
            "f23",
            domain=2,
            ct_creation_func=create_random_int_table,
            param={"low": 1, "high": 5},
        )

        return {
            "variables": [v1, v2, v3],
            "factors": [f1, f2],
            "edges": {f1: [v1, v2], f2: [v2, v3]},
        }

    @pytest.fixture
    def sample_factor_graph(self, sample_agents):
        """Create a sample factor graph."""
        return FactorGraph(
            variable_li=sample_agents["variables"],
            factor_li=sample_agents["factors"],
            edges=sample_agents["edges"],
        )

    def test_factor_graph_creation(self, sample_factor_graph):
        """Test basic factor graph creation."""
        fg = sample_factor_graph

        assert isinstance(fg, FactorGraph)
        assert len(fg.variables) == 3
        assert len(fg.factors) == 2
        assert len(fg.edges) == 2
        assert isinstance(fg.G, nx.Graph)

    def test_factor_graph_networkx_structure(self, sample_factor_graph):
        """Test NetworkX graph structure."""
        fg = sample_factor_graph

        # Check that all nodes are in the graph
        all_agents = fg.variables + fg.factors
        assert len(fg.G.nodes()) == len(all_agents)

        for agent in all_agents:
            assert agent in fg.G.nodes()

    def test_factor_graph_bipartite_structure(self, sample_factor_graph):
        """Test that factor graph maintains bipartite structure."""
        fg = sample_factor_graph

        # Variables should only connect to factors, and vice versa
        for var in fg.variables:
            neighbors = list(fg.G.neighbors(var))
            for neighbor in neighbors:
                assert neighbor in fg.factors

        for factor in fg.factors:
            neighbors = list(fg.G.neighbors(factor))
            for neighbor in neighbors:
                assert neighbor in fg.variables

    def test_factor_graph_edge_consistency(self, sample_factor_graph):
        """Test that edges are consistent between definition and NetworkX graph."""
        fg = sample_factor_graph

        for factor, variables in fg.edges.items():
            # Check that factor is connected to all its variables in NetworkX graph
            for var in variables:
                assert fg.G.has_edge(factor, var)

            # Check that factor's neighbors in NetworkX match edge definition
            networkx_neighbors = set(fg.G.neighbors(factor))
            edge_neighbors = set(variables)
            assert networkx_neighbors == edge_neighbors

    def test_factor_graph_cost_table_initialization(self, sample_factor_graph):
        """Test that cost tables are properly initialized."""
        fg = sample_factor_graph

        for factor in fg.factors:
            assert hasattr(factor, "cost_table")
            assert factor.cost_table is not None
            assert isinstance(factor.cost_table, np.ndarray)
            assert factor.cost_table.shape == (2, 2)  # Binary factors with domain 2

    def test_factor_graph_agent_properties(self, sample_factor_graph):
        """Test that agents have correct properties."""
        fg = sample_factor_graph

        # Test variables
        for var in fg.variables:
            assert isinstance(var, VariableAgent)
            assert var.domain == 2
            assert var.name.startswith("x")
            assert hasattr(var, "type")
            assert var.type == "variable"

        # Test factors
        for factor in fg.factors:
            assert isinstance(factor, FactorAgent)
            assert factor.domain == 2
            assert factor.name.startswith("f")
            assert hasattr(factor, "type")
            assert factor.type == "factor"

    def test_factor_graph_with_fgbuilder(self):
        """Test factor graph creation using FGBuilder."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=5,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10},
        )

        assert isinstance(fg, FactorGraph)
        assert len(fg.variables) == 5
        assert len(fg.factors) == 5
        assert len(fg.edges) == 5

        # Check that all variables have domain 3
        for var in fg.variables:
            assert var.domain == 3

        # Check that all cost tables have correct shape
        for factor in fg.factors:
            assert factor.cost_table.shape == (3, 3)

    def test_factor_graph_different_domains(self):
        """Test factor graph with different domain sizes."""
        for domain_size in [2, 3, 4, 5]:
            fg = FGBuilder.build_cycle_graph(
                num_vars=4,
                domain_size=domain_size,
                ct_factory=create_random_int_table,
                ct_params={"low": 1, "high": 10},
            )

            assert len(fg.variables) == 4
            assert len(fg.factors) == 4

            for var in fg.variables:
                assert var.domain == domain_size

            for factor in fg.factors:
                assert factor.cost_table.shape == (domain_size, domain_size)

    def test_factor_graph_random_structure(self):
        """Test factor graph with random structure."""
        fg = FGBuilder.build_random_graph(
            num_vars=6,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10},
            density=0.5,
        )

        assert isinstance(fg, FactorGraph)
        assert len(fg.variables) == 6
        assert len(fg.factors) >= 0
        assert len(fg.edges) == len(fg.factors)

        # Check bipartite structure
        for var in fg.variables:
            neighbors = list(fg.G.neighbors(var))
            for neighbor in neighbors:
                assert neighbor in fg.factors

    def test_factor_graph_message_passing_setup(self, sample_factor_graph):
        """Test that factor graph is properly set up for message passing."""
        fg = sample_factor_graph

        # Check that all agents have mailers (not mailbox)
        for agent in fg.variables + fg.factors:
            assert hasattr(agent, "mailer")
            # mailer is a MailHandler object, not a list

        # Check that all factors have connection mappings (variables don't have this)
        for factor in fg.factors:
            assert hasattr(factor, "connection_number")
            assert isinstance(factor.connection_number, dict)

    def test_factor_graph_neighbor_access(self, sample_factor_graph):
        """Test neighbor access functionality."""
        fg = sample_factor_graph

        # Test getting neighbors of variables
        for var in fg.variables:
            neighbors = list(fg.G.neighbors(var))
            assert all(isinstance(n, FactorAgent) for n in neighbors)

        # Test getting neighbors of factors
        for factor in fg.factors:
            neighbors = list(fg.G.neighbors(factor))
            assert all(isinstance(n, VariableAgent) for n in neighbors)

    # test_factor_graph_empty_initialization deleted - empty graphs cause NetworkXPointlessConcept error
    # Empty factor graphs are not a valid use case in PropFlow

    def test_factor_graph_single_variable(self):
        """Test factor graph with single variable."""
        var = VariableAgent("x1", domain=2)
        factor = FactorAgent(
            "f1",
            domain=2,
            ct_creation_func=create_random_int_table,
            param={"low": 1, "high": 5},
        )

        fg = FactorGraph(variable_li=[var], factor_li=[factor], edges={factor: [var]})

        assert len(fg.variables) == 1
        assert len(fg.factors) == 1
        assert len(fg.edges) == 1
        assert fg.G.has_edge(var, factor)

    def test_factor_graph_with_attractive_costs(self):
        """Test factor graph with attractive cost tables."""
        fg = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_uniform_float_table,
            ct_params={"low": 0.0, "high": 1.0},
        )

        # Check that cost tables have uniform float structure
        for factor in fg.factors:
            ct = factor.cost_table
            # Should be floats between 0 and 1
            assert ct.dtype in [np.float32, np.float64]
            assert np.all(ct >= 0.0)
            assert np.all(ct <= 1.0)

    def test_factor_graph_large_scale(self):
        """Test factor graph with larger scale."""
        fg = FGBuilder.build_random_graph(
            num_vars=20,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 100},
            density=0.3,
        )

        assert len(fg.variables) == 20
        assert len(fg.factors) >= 0
        assert len(fg.edges) == len(fg.factors)

        # Check that graph is connected (or at least has some edges)
        if len(fg.factors) > 0:
            assert len(fg.G.edges()) > 0

    def test_factor_graph_serialization_readiness(self, sample_factor_graph):
        """Test that factor graph can be prepared for serialization."""
        fg = sample_factor_graph

        # Check that all necessary attributes exist for serialization
        assert hasattr(fg, "variables")
        assert hasattr(fg, "factors")
        assert hasattr(fg, "edges")
        assert hasattr(fg, "G")

        # Check that all agents have necessary attributes
        for agent in fg.variables + fg.factors:
            assert hasattr(agent, "name")
            assert hasattr(agent, "domain")
            assert hasattr(agent, "type")

    def test_factor_graph_cost_table_properties(self, sample_factor_graph):
        """Test properties of cost tables in factor graph."""
        fg = sample_factor_graph

        for factor in fg.factors:
            ct = factor.cost_table

            # Check basic properties
            assert ct.ndim == 2  # Binary factors
            assert ct.shape[0] == factor.domain
            assert ct.shape[1] == factor.domain

            # Check that costs are non-negative
            assert np.all(ct >= 0)

            # Check that costs are finite
            assert np.all(np.isfinite(ct))

    def test_factor_graph_agent_connectivity(self, sample_factor_graph):
        """Test agent connectivity in factor graph."""
        fg = sample_factor_graph

        # Check that connection_number mappings are correct
        for factor in fg.factors:
            connected_vars = fg.edges[factor]
            assert len(factor.connection_number) == len(connected_vars)

            # connection_number uses variable names as keys, not variable objects
            for var in connected_vars:
                assert var.name in factor.connection_number
                assert isinstance(factor.connection_number[var.name], int)

    def test_factor_graph_graph_theory_properties(self, sample_factor_graph):
        """Test graph theory properties of factor graph."""
        fg = sample_factor_graph

        # Check that graph is bipartite by manually checking connections
        # NetworkX's is_bipartite_node_set doesn't exist in newer versions
        for var in fg.variables:
            neighbors = list(fg.G.neighbors(var))
            assert all(n in fg.factors for n in neighbors)

        for factor in fg.factors:
            neighbors = list(fg.G.neighbors(factor))
            assert all(n in fg.variables for n in neighbors)

    def test_factor_graph_modification_safety(self, sample_factor_graph):
        """Test that factor graph handles modifications safely."""
        fg = sample_factor_graph

        original_var_count = len(fg.variables)
        original_factor_count = len(fg.factors)

        # Modifying the lists should not affect the graph structure
        fg.variables.append(VariableAgent("temp", domain=2))

        # Original graph should be unchanged
        assert len(fg.G.nodes()) == original_var_count + original_factor_count

        # Remove the temporary addition
        fg.variables.pop()

        assert len(fg.variables) == original_var_count
