import pytest
import numpy as np
from src.propflow.utils import FGBuilder
from src.propflow.configs import create_random_int_table
from src.propflow.policies.damping import DampingPolicy
from src.propflow.policies.splitting import SplittingPolicy
from src.propflow.policies.cost_reduction import CostReductionPolicy
from src.propflow.policies.convergance import ConvergenceConfig, ConvergencePolicy
from src.propflow.policies.message_pruning import MessagePruningPolicy
from src.propflow.policies.normalize_cost import NormalizeCostPolicy
from src.propflow.base_models.components import Message
from src.propflow.bp_base.engines_realizations import DampingEngine, SplittingEngine


class TestDampingPolicy:
    """Test suite for damping policy."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10}
        )

    @pytest.fixture(params=[0.1, 0.5, 0.9])
    def damping_factor(self, request):
        """Parameterize tests with different damping factors."""
        return request.param

    def test_damping_policy_creation(self, damping_factor):
        """Test damping policy creation."""
        policy = DampingPolicy(damping_factor=damping_factor)
        assert policy.damping_factor == damping_factor

    def test_damping_policy_application(self, sample_factor_graph, damping_factor):
        """Test damping policy application in engine."""
        engine = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=damping_factor
        )
        
        # Run a few iterations
        engine.run(max_iter=3)
        
        # Check that engine completed without errors
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 3

    def test_damping_message_modification(self, damping_factor):
        """Test that damping actually modifies messages."""
        # Create sample messages
        old_message = Message(
            sender="test_sender",
            receiver="test_receiver",
            content=np.array([1.0, 2.0, 3.0])
        )
        
        new_message = Message(
            sender="test_sender",
            receiver="test_receiver",
            content=np.array([2.0, 3.0, 4.0])
        )
        
        policy = DampingPolicy(damping_factor=damping_factor)
        damped_message = policy.apply_damping(old_message, new_message)
        
        # Damped message should be between old and new
        expected_content = (1 - damping_factor) * old_message.content + damping_factor * new_message.content
        np.testing.assert_array_almost_equal(damped_message.content, expected_content)

    def test_damping_extreme_values(self, sample_factor_graph):
        """Test damping with extreme values."""
        # Test with damping factor 0 (no update)
        engine_no_damp = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.0
        )
        engine_no_damp.run(max_iter=1)
        
        # Test with damping factor 1 (full update)
        engine_full_damp = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=1.0
        )
        engine_full_damp.run(max_iter=1)
        
        # Both should complete without error
        assert engine_no_damp.iteration_count > 0
        assert engine_full_damp.iteration_count > 0


class TestSplittingPolicy:
    """Test suite for splitting policy."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10}
        )

    @pytest.fixture(params=[0.3, 0.5, 0.7])
    def split_ratio(self, request):
        """Parameterize tests with different split ratios."""
        return request.param

    def test_splitting_policy_creation(self, split_ratio):
        """Test splitting policy creation."""
        policy = SplittingPolicy(split_ratio=split_ratio)
        assert policy.split_ratio == split_ratio

    def test_splitting_policy_application(self, sample_factor_graph, split_ratio):
        """Test splitting policy application in engine."""
        engine = SplittingEngine(
            factor_graph=sample_factor_graph,
            split_ratio=split_ratio
        )
        
        # Run a few iterations
        engine.run(max_iter=3)
        
        # Check that engine completed without errors
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 3

    def test_factor_splitting(self, sample_factor_graph, split_ratio):
        """Test that factors are actually split."""
        policy = SplittingPolicy(split_ratio=split_ratio)
        
        # Get a factor from the graph
        factor = sample_factor_graph.factors[0]
        original_cost_table = factor.cost_table.copy()
        
        # Apply splitting
        split_factors = policy.split_factor(factor)
        
        # Check that we got multiple factors
        assert len(split_factors) >= 1
        
        # Check that split factors have correct properties
        for split_factor in split_factors:
            assert hasattr(split_factor, 'cost_table')
            assert split_factor.cost_table is not None


class TestCostReductionPolicy:
    """Test suite for cost reduction policy."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10}
        )

    def test_cost_reduction_policy_creation(self):
        """Test cost reduction policy creation."""
        policy = CostReductionPolicy()
        assert policy is not None

    def test_cost_reduction_application(self, sample_factor_graph):
        """Test cost reduction policy application."""
        policy = CostReductionPolicy()
        
        # Get a factor from the graph
        factor = sample_factor_graph.factors[0]
        original_cost_table = factor.cost_table.copy()
        
        # Apply cost reduction
        reduced_table = policy.reduce_costs(original_cost_table)
        
        # Check that costs are reduced
        assert reduced_table is not None
        assert reduced_table.shape == original_cost_table.shape
        
        # Reduced costs should be <= original costs
        assert np.all(reduced_table <= original_cost_table)

    def test_cost_reduction_preserves_structure(self, sample_factor_graph):
        """Test that cost reduction preserves table structure."""
        policy = CostReductionPolicy()
        
        for factor in sample_factor_graph.factors:
            original_shape = factor.cost_table.shape
            reduced_table = policy.reduce_costs(factor.cost_table)
            
            assert reduced_table.shape == original_shape
            assert reduced_table.dtype == factor.cost_table.dtype


class TestConvergencePolicy:
    """Test suite for convergence policy."""

    @pytest.fixture
    def convergence_config(self):
        """Create a convergence configuration."""
        return ConvergenceConfig(
            max_iterations=50,
            convergence_threshold=1e-6,
            time_limit=10.0,
            check_interval=5
        )

    def test_convergence_config_creation(self, convergence_config):
        """Test convergence configuration creation."""
        assert convergence_config.max_iterations == 50
        assert convergence_config.convergence_threshold == 1e-6
        assert convergence_config.time_limit == 10.0
        assert convergence_config.check_interval == 5

    def test_convergence_policy_creation(self, convergence_config):
        """Test convergence policy creation."""
        policy = ConvergencePolicy(convergence_config)
        assert policy.config == convergence_config

    def test_convergence_detection(self, convergence_config):
        """Test convergence detection."""
        policy = ConvergencePolicy(convergence_config)
        
        # Test with identical beliefs (should converge)
        beliefs1 = {"x1": np.array([0.5, 0.5]), "x2": np.array([0.3, 0.7])}
        beliefs2 = {"x1": np.array([0.5, 0.5]), "x2": np.array([0.3, 0.7])}
        
        assert policy.has_converged(beliefs1, beliefs2)
        
        # Test with different beliefs (should not converge)
        beliefs3 = {"x1": np.array([0.6, 0.4]), "x2": np.array([0.3, 0.7])}
        
        assert not policy.has_converged(beliefs1, beliefs3)

    def test_convergence_threshold_sensitivity(self):
        """Test convergence threshold sensitivity."""
        # Strict threshold
        strict_config = ConvergenceConfig(
            max_iterations=50,
            convergence_threshold=1e-10,
            time_limit=10.0,
            check_interval=5
        )
        strict_policy = ConvergencePolicy(strict_config)
        
        # Loose threshold
        loose_config = ConvergenceConfig(
            max_iterations=50,
            convergence_threshold=1e-2,
            time_limit=10.0,
            check_interval=5
        )
        loose_policy = ConvergencePolicy(loose_config)
        
        # Slightly different beliefs
        beliefs1 = {"x1": np.array([0.5, 0.5])}
        beliefs2 = {"x1": np.array([0.501, 0.499])}
        
        # Loose policy should converge, strict should not
        assert loose_policy.has_converged(beliefs1, beliefs2)
        assert not strict_policy.has_converged(beliefs1, beliefs2)


class TestMessagePruningPolicy:
    """Test suite for message pruning policy."""

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            Message("sender1", "receiver1", np.array([1.0, 2.0, 3.0])),
            Message("sender2", "receiver2", np.array([0.1, 0.2, 0.3])),
            Message("sender3", "receiver3", np.array([10.0, 20.0, 30.0])),
            Message("sender4", "receiver4", np.array([0.01, 0.02, 0.03]))
        ]

    def test_message_pruning_policy_creation(self):
        """Test message pruning policy creation."""
        policy = MessagePruningPolicy(threshold=0.5)
        assert policy.threshold == 0.5

    def test_message_pruning_application(self, sample_messages):
        """Test message pruning application."""
        policy = MessagePruningPolicy(threshold=1.0)
        
        pruned_messages = policy.prune_messages(sample_messages)
        
        # Check that pruning returns a list
        assert isinstance(pruned_messages, list)
        
        # Check that pruned messages are subset of original
        assert len(pruned_messages) <= len(sample_messages)

    def test_message_pruning_threshold_effect(self, sample_messages):
        """Test that pruning threshold affects results."""
        # Low threshold (prune more)
        low_policy = MessagePruningPolicy(threshold=0.1)
        low_pruned = low_policy.prune_messages(sample_messages)
        
        # High threshold (prune less)
        high_policy = MessagePruningPolicy(threshold=10.0)
        high_pruned = high_policy.prune_messages(sample_messages)
        
        # Higher threshold should retain more messages
        assert len(high_pruned) >= len(low_pruned)


class TestNormalizeCostPolicy:
    """Test suite for cost normalization policy."""

    @pytest.fixture
    def sample_cost_table(self):
        """Create a sample cost table for testing."""
        return np.array([
            [1.0, 5.0, 10.0],
            [2.0, 3.0, 8.0],
            [4.0, 6.0, 7.0]
        ])

    def test_normalize_cost_policy_creation(self):
        """Test cost normalization policy creation."""
        policy = NormalizeCostPolicy()
        assert policy is not None

    def test_cost_normalization_application(self, sample_cost_table):
        """Test cost normalization application."""
        policy = NormalizeCostPolicy()
        
        normalized_table = policy.normalize_costs(sample_cost_table)
        
        # Check that normalization preserves shape
        assert normalized_table.shape == sample_cost_table.shape
        
        # Check that costs are normalized (min should be 0)
        assert np.min(normalized_table) == 0.0
        
        # Check that relative ordering is preserved
        original_order = np.argsort(sample_cost_table.flatten())
        normalized_order = np.argsort(normalized_table.flatten())
        np.testing.assert_array_equal(original_order, normalized_order)

    def test_cost_normalization_methods(self, sample_cost_table):
        """Test different normalization methods."""
        policy = NormalizeCostPolicy()
        
        # Test min-max normalization
        normalized_minmax = policy.normalize_costs(sample_cost_table, method='minmax')
        assert np.min(normalized_minmax) == 0.0
        assert np.max(normalized_minmax) == 1.0
        
        # Test z-score normalization
        normalized_zscore = policy.normalize_costs(sample_cost_table, method='zscore')
        assert abs(np.mean(normalized_zscore)) < 1e-10  # Mean should be ~0
        assert abs(np.std(normalized_zscore) - 1.0) < 1e-10  # Std should be ~1

    def test_cost_normalization_edge_cases(self):
        """Test cost normalization with edge cases."""
        policy = NormalizeCostPolicy()
        
        # Test with all equal values
        equal_table = np.ones((3, 3))
        normalized_equal = policy.normalize_costs(equal_table)
        assert np.all(normalized_equal == 0.0)  # All should be 0 after normalization
        
        # Test with single value
        single_table = np.array([[5.0]])
        normalized_single = policy.normalize_costs(single_table)
        assert normalized_single[0, 0] == 0.0


class TestPolicyIntegration:
    """Test suite for policy integration and combinations."""

    @pytest.fixture
    def sample_factor_graph(self):
        """Create a sample factor graph for testing."""
        return FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 10}
        )

    def test_multiple_policies_combination(self, sample_factor_graph):
        """Test combining multiple policies."""
        # Create an engine with multiple policies
        engine = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5
        )
        
        # Add additional policies
        engine.add_policy(CostReductionPolicy())
        engine.add_policy(NormalizeCostPolicy())
        
        # Run engine
        engine.run(max_iter=5)
        
        # Check that engine completed successfully
        assert engine.iteration_count > 0
        assert engine.iteration_count <= 5

    def test_policy_order_independence(self, sample_factor_graph):
        """Test that policy order doesn't affect basic functionality."""
        # Create engines with different policy orders
        engine1 = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5
        )
        engine1.add_policy(CostReductionPolicy())
        engine1.add_policy(NormalizeCostPolicy())
        
        engine2 = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5
        )
        engine2.add_policy(NormalizeCostPolicy())
        engine2.add_policy(CostReductionPolicy())
        
        # Both should run without errors
        engine1.run(max_iter=3)
        engine2.run(max_iter=3)
        
        assert engine1.iteration_count > 0
        assert engine2.iteration_count > 0

    def test_policy_performance_impact(self, sample_factor_graph):
        """Test that policies don't cause performance regression."""
        import time
        
        # Engine without policies
        engine_simple = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5
        )
        
        start_time = time.time()
        engine_simple.run(max_iter=10)
        simple_time = time.time() - start_time
        
        # Engine with policies
        engine_complex = DampingEngine(
            factor_graph=sample_factor_graph,
            damping_factor=0.5
        )
        engine_complex.add_policy(CostReductionPolicy())
        engine_complex.add_policy(NormalizeCostPolicy())
        
        start_time = time.time()
        engine_complex.run(max_iter=10)
        complex_time = time.time() - start_time
        
        # Complex engine should not be excessively slower
        # (allowing 10x slowdown for policies)
        assert complex_time < simple_time * 10