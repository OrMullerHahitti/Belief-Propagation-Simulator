"""
Shared pytest fixtures and utilities for the belief propagation simulator tests.

This module provides common fixtures, mock objects, and test utilities that
can be used across multiple test modules to reduce code duplication and
ensure consistent test data.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Callable
import pickle

# ---------------------------------------------------------------------------
# Ensure optional analyzer package is importable during tests.
# The published wheel ships the analyzer namespace (analyzer.*). When running
# the suite from source, add the local dist/ wheel to Python's path so imports
# like ``from analyzer.reporting import ...`` resolve without requiring an
# external install.
# ---------------------------------------------------------------------------
_DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
if _DIST_DIR.exists():
    for wheel_path in sorted(_DIST_DIR.glob("propflow-*.whl"), reverse=True):
        wheel_str = str(wheel_path)
        if wheel_path.is_file() and wheel_str not in sys.path:
            sys.path.append(wheel_str)
            break

# Import main modules for creating fixtures
from propflow.core.agents import VariableAgent, FactorAgent, FGAgent
from propflow.core.components import Message, MailHandler, CostTable
from propflow.bp.factor_graph import FactorGraph
from propflow.bp.computators import MinSumComputator, MaxSumComputator
from propflow.bp.engines import BPEngine
from propflow.policies.convergance import ConvergenceConfig, ConvergenceMonitor
from propflow.configs.global_config_mapping import (
    ENGINE_DEFAULTS,
    POLICY_DEFAULTS,
    CONVERGENCE_DEFAULTS,
    SIMULATOR_DEFAULTS,
    LOGGING_CONFIG,
)
from propflow.utils.fg_utils import FGBuilder
from propflow.configs import create_random_int_table


# ========================================================================================
# Test Configuration and Markers
# ========================================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "config: Configuration system tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line(
        "markers", "validation: Validation and error handling tests"
    )


# ========================================================================================
# Basic Data Fixtures
# ========================================================================================


@pytest.fixture(scope="session")
def domain_sizes():
    """Common domain sizes used across tests."""
    return [2, 3, 4, 5]


@pytest.fixture(scope="session")
def small_domain():
    """Small domain size for quick tests."""
    return 3


@pytest.fixture(scope="session")
def medium_domain():
    """Medium domain size for more comprehensive tests."""
    return 5


@pytest.fixture(scope="session")
def large_domain():
    """Large domain size for performance tests."""
    return 10


@pytest.fixture
def random_seed():
    """Set and return a random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_belief_vectors(small_domain):
    """Sample normalized belief vectors for testing."""
    beliefs = []
    for _ in range(5):
        belief = np.random.rand(small_domain)
        belief /= np.sum(belief)  # Normalize
        beliefs.append(belief)
    return beliefs


@pytest.fixture
def sample_cost_matrices(small_domain):
    """Sample cost matrices for testing."""
    matrices = []
    for shape in [
        (small_domain,),
        (small_domain, small_domain),
        (small_domain, small_domain, small_domain),
    ]:
        matrix = np.random.rand(*shape) * 10
        matrices.append(matrix)
    return matrices


# ========================================================================================
# Mock Objects and Agents
# ========================================================================================


@pytest.fixture
def mock_agent():
    """Create a basic mock agent."""
    agent = Mock()
    agent.name = "mock_agent"
    agent.type = "variable"
    agent.domain = 3
    return agent


@pytest.fixture
def mock_variable_agents(small_domain):
    """Create multiple mock variable agents."""
    agents = []
    for i in range(3):
        agent = Mock(spec=VariableAgent)
        agent.name = f"var{i}"
        agent.type = "variable"
        agent.domain = small_domain
        agents.append(agent)
    return agents


@pytest.fixture
def mock_factor_agents(small_domain):
    """Create multiple mock factor agents."""
    agents = []
    for i in range(2):
        agent = Mock(spec=FactorAgent)
        agent.name = f"factor{i}"
        agent.type = "factor"
        agent.domain = small_domain
        agent.cost_table = np.random.rand(small_domain, small_domain)
        agents.append(agent)
    return agents


@pytest.fixture
def mock_computator():
    """Create a mock computator with common methods."""
    computator = Mock()
    computator.compute_Q.return_value = []
    computator.compute_R.return_value = []
    computator.compute_belief.return_value = np.array([0.5, 0.3, 0.2])
    computator.get_assignment.return_value = 0
    return computator


@pytest.fixture
def mock_engine_class():
    """Create a mock engine class for testing."""
    mock_instance = Mock()
    mock_instance.run.return_value = None
    mock_instance.history.costs = [10.0, 8.0, 6.0, 5.0, 4.0]

    mock_class = Mock(return_value=mock_instance)
    return mock_class


# ========================================================================================
# Real Agent Fixtures
# ========================================================================================


@pytest.fixture
def variable_agent(small_domain):
    """Create a real VariableAgent for testing."""
    return VariableAgent("test_var", small_domain)


@pytest.fixture
def multiple_variable_agents(small_domain):
    """Create multiple real VariableAgents."""
    return [VariableAgent(f"var{i}", small_domain) for i in range(3)]


@pytest.fixture
def factor_agent(small_domain):
    """Create a real FactorAgent for testing."""

    def cost_table_func(n, d, **kwargs):
        return np.random.rand(*([d] * n))

    return FactorAgent("test_factor", small_domain, cost_table_func)


@pytest.fixture
def factor_agent_with_cost_table(small_domain):
    """Create a FactorAgent with pre-existing cost table."""
    cost_table = np.random.rand(small_domain, small_domain)
    return FactorAgent.create_from_cost_table("test_factor", cost_table)


# ========================================================================================
# Message and Communication Fixtures
# ========================================================================================


@pytest.fixture
def sample_message(variable_agent, factor_agent, small_domain):
    """Create a sample message between agents."""
    data = np.random.rand(small_domain)
    return Message(data, variable_agent, factor_agent)


@pytest.fixture
def message_batch(multiple_variable_agents, factor_agent, small_domain):
    """Create a batch of messages for testing."""
    messages = []
    for var_agent in multiple_variable_agents:
        data = np.random.rand(small_domain)
        message = Message(data, var_agent, factor_agent)
        messages.append(message)
    return messages


@pytest.fixture
def mail_handler(small_domain):
    """Create a MailHandler for testing."""
    return MailHandler(small_domain)


# ========================================================================================
# Factor Graph Fixtures
# ========================================================================================


@pytest.fixture
def simple_cycle_graph():
    """Create a simple cycle factor graph for testing."""
    return FGBuilder.build_cycle_graph(
        num_vars=4,
        domain_size=3,
        ct_factory=create_random_int_table,
        ct_params={"low": 1, "high": 10},
    )


@pytest.fixture
def small_random_graph():
    """Create a small random factor graph for testing."""
    return FGBuilder.build_random_graph(
        num_vars=4,
        domain_size=3,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 5},
        density=0.5,
    )


@pytest.fixture
def medium_random_graph():
    """Create a medium-sized random factor graph for testing."""
    return FGBuilder.build_random_graph(
        num_vars=8,
        domain_size=4,
        ct_factory=create_random_int_table,
        ct_params={"low": 0, "high": 10},
        density=0.3,
    )


@pytest.fixture
def multiple_test_graphs():
    """Create multiple test graphs for batch testing."""
    graphs = []

    # Cycle graph
    graphs.append(
        FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 5},
        )
    )

    # Small random graph
    graphs.append(
        FGBuilder.build_random_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 8},
            density=0.4,
        )
    )

    return graphs


@pytest.fixture
def simple_fg():
    """Two-variable factor graph with a unique MAP assignment (0, 0)."""
    var_x = VariableAgent("x1", domain=2)
    var_y = VariableAgent("x2", domain=2)

    cost_table = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=float)
    factor = FactorAgent.create_from_cost_table("f_xy", cost_table)

    edges = {factor: [var_x, var_y]}
    return FactorGraph(variable_li=[var_x, var_y], factor_li=[factor], edges=edges)


@pytest.fixture
def tree_fg():
    """Three-variable chain factor graph used for MAP consistency tests."""
    vars_list = [
        VariableAgent("x1", domain=2),
        VariableAgent("x2", domain=2),
        VariableAgent("x3", domain=2),
    ]

    cost_xy = np.array([[0.0, 1.5], [1.5, 3.0]], dtype=float)
    cost_yz = np.array([[0.0, 1.0], [1.0, 2.5]], dtype=float)

    factor_xy = FactorAgent.create_from_cost_table("f_xy", cost_xy)
    factor_yz = FactorAgent.create_from_cost_table("f_yz", cost_yz)

    edges = {
        factor_xy: [vars_list[0], vars_list[1]],
        factor_yz: [vars_list[1], vars_list[2]],
    }
    return FactorGraph(variable_li=vars_list, factor_li=[factor_xy, factor_yz], edges=edges)


# ========================================================================================
# Engine and Computator Fixtures
# ========================================================================================


@pytest.fixture
def min_sum_computator():
    """Create a MinSum computator."""
    return MinSumComputator()


@pytest.fixture
def max_sum_computator():
    """Create a MaxSum computator."""
    return MaxSumComputator()


@pytest.fixture
def bp_engine(simple_cycle_graph, min_sum_computator):
    """Create a basic BP engine for testing."""
    return BPEngine(
        factor_graph=simple_cycle_graph,
        computator=min_sum_computator,
        name="TestBPEngine",
    )


@pytest.fixture
def engine_configs():
    """Standard engine configurations for testing."""
    return {
        "BPEngine": {"class": BPEngine},
        "TestEngine": {
            "class": Mock,
            "normalize_messages": True,
            "monitor_performance": False,
        },
    }


# ========================================================================================
# Configuration Fixtures
# ========================================================================================


@pytest.fixture
def default_configs():
    """Access to all default configurations."""
    return {
        "engine": ENGINE_DEFAULTS,
        "policy": POLICY_DEFAULTS,
        "convergence": CONVERGENCE_DEFAULTS,
        "simulator": SIMULATOR_DEFAULTS,
        "logging": LOGGING_CONFIG,
    }


@pytest.fixture
def test_convergence_config():
    """Create a test convergence configuration."""
    return ConvergenceConfig(
        belief_threshold=1e-4, min_iterations=2, patience=3, use_relative_change=True
    )


@pytest.fixture
def convergence_monitor(test_convergence_config):
    """Create a convergence monitor for testing."""
    return ConvergenceMonitor(test_convergence_config)


@pytest.fixture
def custom_engine_config():
    """Custom engine configuration for testing overrides."""
    return {
        "max_iterations": 500,
        "timeout": 1800,
        "normalize_messages": False,
        "monitor_performance": True,
        "anytime": True,
        "use_bct_history": True,
    }


# ========================================================================================
# File System and Temporary Resources
# ========================================================================================


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_log_file(temp_directory):
    """Create a temporary log file."""
    log_file = os.path.join(temp_directory, "test.log")
    with open(log_file, "w") as f:
        f.write("Test log file\n")
    return log_file


@pytest.fixture
def pickled_graph(simple_cycle_graph, temp_directory):
    """Create a pickled factor graph for testing serialization."""
    pickle_file = os.path.join(temp_directory, "test_graph.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump(simple_cycle_graph, f)
    return pickle_file


# ========================================================================================
# Performance and Benchmarking Fixtures
# ========================================================================================


@pytest.fixture
def performance_timer():
    """Simple timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


@pytest.fixture
def benchmark_graphs():
    """Create graphs of various sizes for benchmarking."""
    sizes = [(5, 3), (10, 4), (15, 5)]  # (num_vars, domain_size)
    graphs = []

    for num_vars, domain_size in sizes:
        graph = FGBuilder.build_random_graph(
            num_vars=num_vars,
            domain_size=domain_size,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 10},
            density=0.3,
        )
        graphs.append((graph, num_vars, domain_size))

    return graphs


# ========================================================================================
# Test Data Generators
# ========================================================================================


@pytest.fixture
def belief_data_generator():
    """Generator function for creating belief test data."""

    def generate_beliefs(num_variables: int, domain_size: int, num_samples: int = 5):
        beliefs_list = []
        for _ in range(num_samples):
            beliefs = {}
            for i in range(num_variables):
                belief = np.random.dirichlet(
                    np.ones(domain_size)
                )  # Proper probability distribution
                beliefs[f"var{i}"] = belief
            beliefs_list.append(beliefs)
        return beliefs_list

    return generate_beliefs


@pytest.fixture
def assignment_data_generator():
    """Generator function for creating assignment test data."""

    def generate_assignments(
        num_variables: int, domain_size: int, num_samples: int = 5
    ):
        assignments_list = []
        for _ in range(num_samples):
            assignments = {}
            for i in range(num_variables):
                assignments[f"var{i}"] = np.random.randint(0, domain_size)
            assignments_list.append(assignments)
        return assignments_list

    return generate_assignments


@pytest.fixture
def cost_table_generator():
    """Generator function for creating cost table test data."""

    def generate_cost_tables(shapes: List[tuple], num_samples: int = 3):
        cost_tables = []
        for shape in shapes:
            tables = []
            for _ in range(num_samples):
                table = np.random.rand(*shape) * 10
                tables.append(table)
            cost_tables.append(tables)
        return cost_tables

    return generate_cost_tables


# ========================================================================================
# Mock Network and Communication
# ========================================================================================


@pytest.fixture
def mock_network_topology():
    """Create a mock network topology for testing distributed scenarios."""
    topology = {
        "variables": ["var1", "var2", "var3", "var4"],
        "factors": ["f12", "f23", "f34", "f41"],
        "connections": {
            "f12": ["var1", "var2"],
            "f23": ["var2", "var3"],
            "f34": ["var3", "var4"],
            "f41": ["var4", "var1"],
        },
    }
    return topology


# ========================================================================================
# Validation and Error Testing Utilities
# ========================================================================================


@pytest.fixture
def invalid_config_samples():
    """Sample invalid configurations for testing validation."""
    return {
        "engine": [
            {"max_iterations": -1},
            {"timeout": 0},
            {"max_iterations": "not_int"},
            {},  # Missing required keys
        ],
        "policy": [
            {"damping_factor": 0.0},
            {"damping_factor": 1.1},
            {"split_factor": 0.0},
            {"split_factor": 1.0},
            {"pruning_threshold": -0.1},
        ],
        "convergence": [
            {"belief_threshold": 0},
            {"belief_threshold": -1e-6},
            {"min_iterations": -1},
            {"patience": -1},
        ],
    }


# ========================================================================================
# Parametrized Test Data
# ========================================================================================


@pytest.fixture(params=[2, 3, 4, 5])
def parametrized_domain_size(request):
    """Parametrized domain sizes for comprehensive testing."""
    return request.param


@pytest.fixture(params=[1, 5, 10, 50])
def parametrized_max_iterations(request):
    """Parametrized max iterations for testing different scenarios."""
    return request.param


@pytest.fixture(params=["min-sum", "max-sum"])
def parametrized_computator_type(request):
    """Parametrized computator types."""
    if request.param == "min-sum":
        return MinSumComputator()
    else:
        return MaxSumComputator()


# ========================================================================================
# Test Utilities and Helpers
# ========================================================================================


@pytest.fixture
def assert_helpers():
    """Collection of assertion helpers for tests."""

    class AssertHelpers:
        @staticmethod
        def assert_probability_distribution(arr, tolerance=1e-10):
            """Assert that array is a valid probability distribution."""
            assert np.all(arr >= 0), "Probabilities must be non-negative"
            assert abs(np.sum(arr) - 1.0) < tolerance, "Probabilities must sum to 1"

        @staticmethod
        def assert_cost_table_properties(cost_table):
            """Assert basic properties of cost tables."""
            assert isinstance(cost_table, np.ndarray), "Cost table must be numpy array"
            assert cost_table.ndim >= 1, "Cost table must have at least 1 dimension"
            assert np.all(
                np.isfinite(cost_table)
            ), "Cost table must contain finite values"

        @staticmethod
        def assert_agent_properties(agent, expected_name=None, expected_domain=None):
            """Assert basic properties of agents."""
            assert hasattr(agent, "name"), "Agent must have name"
            assert hasattr(agent, "type"), "Agent must have type"
            assert hasattr(agent, "domain"), "Agent must have domain"

            if expected_name:
                assert agent.name == expected_name
            if expected_domain:
                assert agent.domain == expected_domain

        @staticmethod
        def assert_message_properties(
            message, expected_sender=None, expected_recipient=None
        ):
            """Assert basic properties of messages."""
            assert hasattr(message, "data"), "Message must have data"
            assert hasattr(message, "sender"), "Message must have sender"
            assert hasattr(message, "recipient"), "Message must have recipient"
            assert isinstance(
                message.data, np.ndarray
            ), "Message data must be numpy array"

            if expected_sender:
                assert message.sender == expected_sender
            if expected_recipient:
                assert message.recipient == expected_recipient

    return AssertHelpers()


@pytest.fixture
def test_data_validator():
    """Validator for test data integrity."""

    class TestDataValidator:
        @staticmethod
        def validate_beliefs(beliefs_dict, expected_domain_size=None):
            """Validate belief dictionary structure and values."""
            assert isinstance(beliefs_dict, dict), "Beliefs must be dictionary"

            for var_name, belief in beliefs_dict.items():
                assert isinstance(var_name, str), "Variable names must be strings"
                assert isinstance(belief, np.ndarray), "Beliefs must be numpy arrays"
                assert belief.ndim == 1, "Beliefs must be 1D arrays"

                if expected_domain_size:
                    assert len(belief) == expected_domain_size

        @staticmethod
        def validate_assignments(assignments_dict, expected_domain_size=None):
            """Validate assignment dictionary structure and values."""
            assert isinstance(assignments_dict, dict), "Assignments must be dictionary"

            for var_name, assignment in assignments_dict.items():
                assert isinstance(var_name, str), "Variable names must be strings"
                assert isinstance(
                    assignment, (int, np.integer)
                ), "Assignments must be integers"

                if expected_domain_size:
                    assert 0 <= assignment < expected_domain_size

    return TestDataValidator()


# ========================================================================================
# Session-scoped cleanup
# ========================================================================================


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Clean up any session-wide resources."""
    yield
    # Any session-wide cleanup can go here


if __name__ == "__main__":
    # This file should not be run directly
    print("conftest.py should not be run directly. Use pytest to run tests.")
