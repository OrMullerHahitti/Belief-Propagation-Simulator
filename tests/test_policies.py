import pytest
import numpy as np
from src.propflow.utils import FGBuilder
from src.propflow.configs import create_random_int_table
from src.propflow.policies.bp_policies import (
    DampingPolicy,
    CostReductionPolicy,
    SplittingPolicy,
)
from src.propflow.policies.convergance import ConvergenceConfig
from src.propflow.bp.engines_realizations import DampingEngine, SplittingEngine


class TestBasicPolicies:
    """Test suite for basic policy functionality."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10},
        )

    def test_damping_policy_creation(self):
        """Test damping policy creation."""
        policy = DampingPolicy(damping_factor=0.5)
        assert policy.damping_factor == 0.5

    def test_cost_reduction_policy_creation(self):
        """Test cost reduction policy creation."""
        policy = CostReductionPolicy(reduction_factor=0.3)
        assert policy.reduction_factor == 0.3

    def test_splitting_policy_creation(self):
        """Test splitting policy creation."""
        policy = SplittingPolicy(split_ratio=0.7)
        assert policy.split_ratio == 0.7

    def test_convergence_config_creation(self):
        """Test convergence configuration creation."""
        config = ConvergenceConfig(
            max_iterations=50,
            convergence_threshold=1e-6,
            time_limit=10.0,
            check_interval=5,
        )
        assert config.max_iterations == 50
        assert config.convergence_threshold == 1e-6
        assert config.time_limit == 10.0
        assert config.check_interval == 5

    def test_damping_engine_with_policy(self, sample_factor_graph):
        """Test damping engine with damping policy."""
        engine = DampingEngine(factor_graph=sample_factor_graph, damping_factor=0.5)

        # Run a few iterations
        engine.run(max_iter=3)

        # Check that engine completed without errors
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 3

    def test_splitting_engine_with_policy(self, sample_factor_graph):
        """Test splitting engine with splitting policy."""
        engine = SplittingEngine(factor_graph=sample_factor_graph, split_ratio=0.5)

        # Run a few iterations
        engine.run(max_iter=3)

        # Check that engine completed without errors
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 3

    @pytest.mark.parametrize("damping_factor", [0.1, 0.5, 0.9])
    def test_damping_policy_with_different_factors(
        self, sample_factor_graph, damping_factor
    ):
        """Test damping policy with different damping factors."""
        engine = DampingEngine(
            factor_graph=sample_factor_graph, damping_factor=damping_factor
        )

        # Run engine
        engine.run(max_iter=5)

        # Check that engine completed successfully
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 5

    @pytest.mark.parametrize("split_ratio", [0.3, 0.5, 0.7])
    def test_splitting_policy_with_different_ratios(
        self, sample_factor_graph, split_ratio
    ):
        """Test splitting policy with different split ratios."""
        engine = SplittingEngine(
            factor_graph=sample_factor_graph, split_ratio=split_ratio
        )

        # Run engine
        engine.run(max_iter=5)

        # Check that engine completed successfully
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 5

    def test_policy_parameter_validation(self):
        """Test that policies validate their parameters correctly."""
        # Test damping factor bounds
        with pytest.raises(ValueError):
            DampingPolicy(damping_factor=-0.1)  # Negative damping

        with pytest.raises(ValueError):
            DampingPolicy(damping_factor=1.1)  # Damping > 1

        # Test split ratio bounds
        with pytest.raises(ValueError):
            SplittingPolicy(split_ratio=0.0)  # Zero split ratio

        with pytest.raises(ValueError):
            SplittingPolicy(split_ratio=1.1)  # Split ratio > 1

    def test_convergence_config_validation(self):
        """Test convergence configuration validation."""
        # Test positive max_iterations
        with pytest.raises(ValueError):
            ConvergenceConfig(max_iterations=0)

        # Test positive convergence_threshold
        with pytest.raises(ValueError):
            ConvergenceConfig(convergence_threshold=-1e-6)

        # Test positive time_limit
        with pytest.raises(ValueError):
            ConvergenceConfig(time_limit=-1.0)


class TestPolicyIntegration:
    """Test suite for policy integration with bp."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10},
        )

    def test_engine_with_convergence_config(self, sample_factor_graph):
        """Test engine with convergence configuration."""
        convergence_config = ConvergenceConfig(
            max_iterations=20,
            convergence_threshold=1e-4,
            time_limit=5.0,
            check_interval=2,
        )

        engine = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5,
            convergence_config=convergence_config,
        )

        # Run engine
        engine.run(max_iter=20)

        # Check that engine respects convergence config
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 20

    def test_different_engines_same_graph(self, sample_factor_graph):
        """Test different bp on the same graph."""
        # Damping engine
        damping_engine = DampingEngine(
            factor_graph=sample_factor_graph, damping_factor=0.5
        )

        # Splitting engine
        splitting_engine = SplittingEngine(
            factor_graph=sample_factor_graph, split_ratio=0.5
        )

        # Both should run successfully
        damping_engine.run(max_iter=3)
        splitting_engine.run(max_iter=3)

        assert damping_engine.iteration_count > 0
        assert splitting_engine.iteration_count > 0

    def test_engine_performance_basic(self, sample_factor_graph):
        """Test basic engine performance characteristics."""
        import time

        engine = DampingEngine(factor_graph=sample_factor_graph, damping_factor=0.5)

        start_time = time.time()
        engine.run(max_iter=10)
        end_time = time.time()

        # Engine should complete in reasonable time
        assert end_time - start_time < 10.0  # Should take less than 10 seconds
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 10
