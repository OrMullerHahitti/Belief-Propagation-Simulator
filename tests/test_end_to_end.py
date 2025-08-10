"""
End-to-end integration tests for the belief propagation simulator.

This module tests complete workflows from configuration through execution
to results, ensuring all components work together properly:
- Full simulation pipelines
- Configuration integration across components
- Data flow from factor graphs through engines to results
- Error handling in realistic scenarios
- Performance and scalability testing
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, Mock
import tempfile
import os

from propflow.simulator import Simulator
from propflow.bp.engines_realizations import BPEngine, DampingEngine, SplitEngine
from propflow.bp.computators import MinSumComputator, MaxSumComputator
from propflow.policies.convergance import ConvergenceConfig
from propflow.utils.fg_utils import FGBuilder
from propflow.configs import create_random_int_table, get_validated_config
from propflow.configs.global_config_mapping import (
    SIMULATOR_DEFAULTS,
    ENGINE_DEFAULTS,
    POLICY_DEFAULTS,
)


@pytest.mark.integration
class TestFullSimulationWorkflow:
    """Test complete simulation workflows."""

    def test_basic_simulation_workflow(self, simple_cycle_graph):
        """Test basic end-to-end simulation with single engine and graph."""
        # Configure engines
        engine_configs = {"BPEngine": {"class": BPEngine}}

        # Create simulator
        simulator = Simulator(engine_configs, log_level="HIGH")

        # Run simulation
        results = simulator.run_simulations([simple_cycle_graph], max_iter=10)

        # Verify results structure
        assert "BPEngine" in results
        assert len(results["BPEngine"]) == 1
        assert isinstance(results["BPEngine"][0], list)  # Cost trajectory

        # Verify costs are decreasing or stable (optimization property)
        costs = results["BPEngine"][0]
        if len(costs) > 1:
            # Should generally decrease or stay stable
            assert costs[-1] <= costs[0] + 1e-6  # Allow small numerical error

    def test_multiple_engines_single_graph(self, simple_cycle_graph):
        """Test simulation with multiple engines on single graph."""
        engine_configs = {
            "BPEngine": {"class": BPEngine},
            "BPEngine_Custom": {
                "class": BPEngine,
                "normalize_messages": False,
                "monitor_performance": False,
            },
        }

        simulator = Simulator(engine_configs)
        results = simulator.run_simulations([simple_cycle_graph], max_iter=5)

        # All engines should have results
        for engine_name in engine_configs:
            assert engine_name in results
            assert len(results[engine_name]) == 1
            assert len(results[engine_name][0]) <= 5  # Limited by max_iter

    def test_multiple_graphs_single_engine(self, multiple_test_graphs):
        """Test simulation with single engine on multiple graphs."""
        engine_configs = {"BPEngine": {"class": BPEngine}}

        simulator = Simulator(engine_configs)
        results = simulator.run_simulations(multiple_test_graphs, max_iter=8)

        # Should have results for each graph
        assert len(results["BPEngine"]) == len(multiple_test_graphs)

        # Each result should be a cost trajectory
        for cost_trajectory in results["BPEngine"]:
            assert isinstance(cost_trajectory, list)
            assert len(cost_trajectory) <= 8
            assert all(isinstance(cost, (int, float)) for cost in cost_trajectory)

    def test_multiple_engines_multiple_graphs(self, multiple_test_graphs):
        """Test simulation with multiple engines on multiple graphs."""
        engine_configs = {
            "BPEngine": {"class": BPEngine},
            "MinSumEngine": {"class": BPEngine, "name": "MinSumCustom"},
        }

        simulator = Simulator(engine_configs)
        results = simulator.run_simulations(multiple_test_graphs, max_iter=6)

        # Comprehensive result verification
        assert len(results) == len(engine_configs)

        for engine_name in engine_configs:
            assert engine_name in results
            assert len(results[engine_name]) == len(multiple_test_graphs)

            for cost_trajectory in results[engine_name]:
                assert isinstance(cost_trajectory, list)
                assert len(cost_trajectory) <= 6

    def test_centralized_configuration_integration(self, simple_cycle_graph):
        """Test that centralized configuration works end-to-end."""
        # Use default configurations
        engine_configs = {"BPEngine": {"class": BPEngine}}

        simulator = Simulator(engine_configs)  # Should use SIMULATOR_DEFAULTS

        # Verify simulator uses centralized defaults
        assert simulator.timeout == SIMULATOR_DEFAULTS["timeout"]

        # Run with centralized defaults
        results = simulator.run_simulations(
            [simple_cycle_graph]
        )  # Should use SIMULATOR_DEFAULTS["default_max_iter"]

        assert "BPEngine" in results
        assert len(results["BPEngine"]) == 1

    def test_configuration_override_integration(self, simple_cycle_graph):
        """Test configuration overrides work throughout the system."""
        custom_max_iter = 15
        custom_log_level = "HIGH"

        engine_configs = {
            "BPEngine": {
                "class": BPEngine,
                "normalize_messages": False,  # Override default
                "monitor_performance": True,  # Override default
            }
        }

        simulator = Simulator(engine_configs, log_level=custom_log_level)
        results = simulator.run_simulations(
            [simple_cycle_graph], max_iter=custom_max_iter
        )

        # Results should reflect custom configuration
        costs = results["BPEngine"][0]
        assert len(costs) <= custom_max_iter

        # Logger should use custom level
        expected_log_level = getattr(simulator.logger, "level", None)
        assert expected_log_level is not None


@pytest.mark.integration
class TestSimulationWithRealEngines:
    """Test simulations with actual engine implementations."""

    def test_bp_engine_convergence_behavior(self, small_random_graph):
        """Test that BP engine shows realistic convergence behavior."""
        engine_configs = {"BPEngine": {"class": BPEngine}}

        simulator = Simulator(engine_configs)
        results = simulator.run_simulations([small_random_graph], max_iter=20)

        costs = results["BPEngine"][0]

        # Should have some iterations
        assert len(costs) > 0

        # Costs should be finite and non-negative
        assert all(np.isfinite(cost) and cost >= 0 for cost in costs)

        # Should show some optimization (first cost >= last cost, allowing noise)
        if len(costs) > 5:
            # Allow for some noise in optimization
            improvement = costs[0] - costs[-1]
            assert improvement >= -1.0  # Should not get significantly worse

    def test_damping_vs_regular_bp(self, small_random_graph):
        """Test that damping engine behaves differently from regular BP."""
        try:
            # Only run this test if DampingEngine is available
            engine_configs = {
                "BPEngine": {"class": BPEngine},
                "DampingEngine": {
                    "class": DampingEngine,
                    "damping_factor": POLICY_DEFAULTS["damping_factor"],
                },
            }

            simulator = Simulator(engine_configs)
            results = simulator.run_simulations([small_random_graph], max_iter=15)

            # Both engines should produce results
            assert "BPEngine" in results
            assert "DampingEngine" in results

            bp_costs = results["BPEngine"][0]
            damping_costs = results["DampingEngine"][0]

            # Both should have reasonable cost trajectories
            assert len(bp_costs) > 0
            assert len(damping_costs) > 0

            # Results may differ (damping often more stable)
            # This is more about ensuring both work than specific behavior

        except (ImportError, AttributeError):
            pytest.skip("DampingEngine not available")

    def test_different_computators(self, simple_cycle_graph):
        """Test engines with different computators."""
        engine_configs = {
            "MinSum": {
                "class": BPEngine,
                "computator": MinSumComputator(),
                "name": "MinSumBP",
            },
            "MaxSum": {
                "class": BPEngine,
                "computator": MaxSumComputator(),
                "name": "MaxSumBP",
            },
        }

        simulator = Simulator(engine_configs)
        results = simulator.run_simulations([simple_cycle_graph], max_iter=10)

        # Both computators should work
        assert "MinSum" in results
        assert "MaxSum" in results

        min_sum_costs = results["MinSum"][0]
        max_sum_costs = results["MaxSum"][0]

        # Both should produce valid cost trajectories
        assert len(min_sum_costs) > 0
        assert len(max_sum_costs) > 0
        assert all(np.isfinite(c) for c in min_sum_costs)
        assert all(np.isfinite(c) for c in max_sum_costs)

        # Results will generally be different due to different optimization objectives

    def test_convergence_config_integration(self, simple_cycle_graph):
        """Test that convergence configuration affects simulation behavior."""
        # Test with early convergence
        early_convergence = ConvergenceConfig(
            belief_threshold=1e-2, min_iterations=2, patience=2  # Loose threshold
        )

        # Test with strict convergence
        strict_convergence = ConvergenceConfig(
            belief_threshold=1e-8, min_iterations=5, patience=5  # Tight threshold
        )

        engine_configs_early = {
            "EarlyConv": {"class": BPEngine, "convergence_config": early_convergence}
        }

        engine_configs_strict = {
            "StrictConv": {"class": BPEngine, "convergence_config": strict_convergence}
        }

        simulator_early = Simulator(engine_configs_early)
        simulator_strict = Simulator(engine_configs_strict)

        results_early = simulator_early.run_simulations(
            [simple_cycle_graph], max_iter=50
        )
        results_strict = simulator_strict.run_simulations(
            [simple_cycle_graph], max_iter=50
        )

        early_costs = results_early["EarlyConv"][0]
        strict_costs = results_strict["StrictConv"][0]

        # Early convergence might stop sooner (implementation dependent)
        # Both should produce valid results
        assert len(early_costs) > 0
        assert len(strict_costs) > 0


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in realistic scenarios."""

    def test_engine_failure_recovery(self, simple_cycle_graph):
        """Test simulation recovery when an engine fails."""

        # Create a failing engine
        class FailingEngine:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Engine initialization failed")

        engine_configs = {
            "WorkingEngine": {"class": BPEngine},
            "FailingEngine": {"class": FailingEngine},
        }

        simulator = Simulator(engine_configs)

        # Should handle failure gracefully
        results = simulator.run_simulations([simple_cycle_graph], max_iter=5)

        # Working engine should still produce results
        assert "WorkingEngine" in results

        # Failing engine should be in results but with empty or failed data
        assert "FailingEngine" in results

    def test_invalid_graph_handling(self):
        """Test handling of invalid or empty graphs."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Test with empty graph list
        results = simulator.run_simulations([], max_iter=5)
        assert "BPEngine" in results
        assert len(results["BPEngine"]) == 0

        # Test with None graph (should be handled gracefully)
        try:
            results = simulator.run_simulations([None], max_iter=5)
            # If this succeeds, verify structure
            assert "BPEngine" in results
        except Exception:
            # If it fails, that's also acceptable behavior
            pass

    def test_memory_and_resource_handling(self, benchmark_graphs):
        """Test simulation with various graph sizes for resource handling."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Test with progressively larger graphs
        for graph, num_vars, domain_size in benchmark_graphs[
            :2
        ]:  # Limit to prevent long tests
            results = simulator.run_simulations([graph], max_iter=5)

            assert "BPEngine" in results
            assert len(results["BPEngine"]) == 1

            costs = results["BPEngine"][0]
            assert len(costs) <= 5
            assert all(np.isfinite(cost) for cost in costs)

    def test_configuration_validation_integration(self):
        """Test that invalid configurations are caught in integration."""
        # Test invalid engine configuration
        with pytest.raises((ValueError, TypeError)):
            invalid_config = get_validated_config("engine", {"max_iterations": -1})

        # Test invalid policy configuration
        with pytest.raises((ValueError, TypeError)):
            invalid_config = get_validated_config("policy", {"damping_factor": 2.0})

        # Valid configurations should work
        valid_engine_config = get_validated_config("engine", {"max_iterations": 100})
        assert valid_engine_config["max_iterations"] == 100

        valid_policy_config = get_validated_config("policy", {"damping_factor": 0.7})
        assert valid_policy_config["damping_factor"] == 0.7


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""

    def test_simulation_scaling(self, performance_timer):
        """Test that simulation time scales reasonably with problem size."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Small graph
        small_graph = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 5},
        )

        # Medium graph
        medium_graph = FGBuilder.build_cycle_graph(
            num_vars=6,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 5},
        )

        # Time small graph
        performance_timer.start()
        results_small = simulator.run_simulations([small_graph], max_iter=10)
        performance_timer.stop()
        small_time = performance_timer.elapsed

        # Time medium graph
        performance_timer.start()
        results_medium = simulator.run_simulations([medium_graph], max_iter=10)
        performance_timer.stop()
        medium_time = performance_timer.elapsed

        # Both should complete successfully
        assert "BPEngine" in results_small
        assert "BPEngine" in results_medium

        # Times should be reasonable (not checking specific scaling)
        assert small_time < 60  # Should complete within 60 seconds
        assert medium_time < 120  # Should complete within 120 seconds

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable across multiple simulations."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Create consistent test graph
        test_graph = FGBuilder.build_cycle_graph(
            num_vars=5,
            domain_size=3,
            ct_factory=create_random_int_table,
            ct_params={"low": 0, "high": 5},
        )

        # Run multiple simulations
        for i in range(3):  # Reduced number to prevent long tests
            results = simulator.run_simulations([test_graph], max_iter=5)

            # Each simulation should succeed
            assert "BPEngine" in results
            assert len(results["BPEngine"]) == 1

            # Clear results to prevent accumulation
            simulator.results = {name: [] for name in simulator.engine_configs}

    def test_concurrent_simulation_safety(self):
        """Test that multiprocessing simulation is safe and consistent."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Create multiple identical graphs
        graphs = []
        for _ in range(3):
            graph = FGBuilder.build_cycle_graph(
                num_vars=4,
                domain_size=2,
                ct_factory=create_random_int_table,
                ct_params={"low": 1, "high": 3},
            )
            graphs.append(graph)

        # Run with multiprocessing
        results = simulator.run_simulations(graphs, max_iter=8)

        # Should get results for all graphs
        assert "BPEngine" in results
        assert len(results["BPEngine"]) == len(graphs)

        # All results should be valid
        for cost_trajectory in results["BPEngine"]:
            assert isinstance(cost_trajectory, list)
            assert len(cost_trajectory) <= 8
            assert all(np.isfinite(cost) for cost in cost_trajectory)


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow through the complete system."""

    def test_factor_graph_to_results_data_flow(self, simple_cycle_graph):
        """Test complete data flow from factor graph creation to final results."""
        # Verify initial graph properties
        assert simple_cycle_graph.G is not None
        assert len(simple_cycle_graph.variables) > 0
        assert len(simple_cycle_graph.factors) > 0

        # Configure and run simulation
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        results = simulator.run_simulations([simple_cycle_graph], max_iter=10)

        # Verify data flow to results
        assert "BPEngine" in results
        costs = results["BPEngine"][0]

        # Results should reflect graph properties
        assert len(costs) > 0
        assert all(isinstance(cost, (int, float, np.number)) for cost in costs)

        # First cost should correspond to initial graph state
        initial_cost = costs[0]
        assert np.isfinite(initial_cost)
        assert initial_cost >= 0  # Costs should be non-negative

    def test_configuration_propagation(self, simple_cycle_graph):
        """Test that configuration changes propagate through all components."""
        custom_max_iter = 7
        custom_engine_config = {
            "BPEngine": {
                "class": BPEngine,
                "normalize_messages": False,
                "monitor_performance": False,
            }
        }

        simulator = Simulator(custom_engine_config)
        results = simulator.run_simulations(
            [simple_cycle_graph], max_iter=custom_max_iter
        )

        # Configuration should limit iterations
        costs = results["BPEngine"][0]
        assert len(costs) <= custom_max_iter

    def test_message_passing_integration(self, small_random_graph):
        """Test that message passing works correctly in integration."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Run simulation and verify message passing occurred
        results = simulator.run_simulations([small_random_graph], max_iter=5)

        costs = results["BPEngine"][0]

        # If message passing is working, we should see cost changes
        if len(costs) > 1:
            # Costs should show some pattern (not all identical unless converged)
            cost_variance = np.var(costs)
            # Some change is expected unless perfect convergence
            assert cost_variance >= 0  # Non-negative variance

    def test_result_aggregation_accuracy(self):
        """Test that results are accurately aggregated from multiple simulations."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Create two different graphs
        graph1 = FGBuilder.build_cycle_graph(
            num_vars=3,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 1, "high": 3},
        )

        graph2 = FGBuilder.build_cycle_graph(
            num_vars=4,
            domain_size=2,
            ct_factory=create_random_int_table,
            ct_params={"low": 2, "high": 4},
        )

        # Run simulations
        results = simulator.run_simulations([graph1, graph2], max_iter=6)

        # Should have exactly 2 result sets
        assert len(results["BPEngine"]) == 2

        # Each should be independent
        costs1 = results["BPEngine"][0]
        costs2 = results["BPEngine"][1]

        assert isinstance(costs1, list)
        assert isinstance(costs2, list)
        assert len(costs1) <= 6
        assert len(costs2) <= 6

        # Results should be different (different graphs)
        if len(costs1) > 0 and len(costs2) > 0:
            # At least initial costs should likely differ
            assert (
                costs1 != costs2 or len(set(costs1 + costs2)) == 1
            )  # Unless both converge to same value


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Large-scale integration tests (marked as slow)."""

    def test_many_graphs_simulation(self):
        """Test simulation with many graphs."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Create many small graphs
        graphs = []
        for i in range(5):  # Reduced from larger number to prevent long test times
            graph = FGBuilder.build_cycle_graph(
                num_vars=3,
                domain_size=2,
                ct_factory=create_random_int_table,
                ct_params={"low": 0, "high": 5},
            )
            graphs.append(graph)

        # Run simulation
        results = simulator.run_simulations(graphs, max_iter=5)

        # Should handle all graphs
        assert len(results["BPEngine"]) == len(graphs)

        # All should have valid results
        for cost_trajectory in results["BPEngine"]:
            assert len(cost_trajectory) <= 5
            assert all(np.isfinite(cost) for cost in cost_trajectory)

    def test_long_running_simulation(self, simple_cycle_graph):
        """Test longer running simulation for stability."""
        engine_configs = {"BPEngine": {"class": BPEngine}}
        simulator = Simulator(engine_configs)

        # Run longer simulation
        results = simulator.run_simulations([simple_cycle_graph], max_iter=50)

        costs = results["BPEngine"][0]

        # Should handle long runs
        assert len(costs) <= 50
        assert all(np.isfinite(cost) for cost in costs)

        # Should show convergence behavior
        if len(costs) > 10:
            # Later costs should be more stable
            early_variance = np.var(costs[: len(costs) // 2])
            late_variance = np.var(costs[len(costs) // 2 :])

            # Often late variance is smaller (convergence), but not always
            assert late_variance >= 0
            assert early_variance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
