"""
Comprehensive tests for the Simulator class.

This module tests the main simulation orchestrator:
- Simulator initialization and configuration
- Multiprocessing simulation execution
- Result collection and aggregation
- Plot generation and visualization
- Error handling and fallback mechanisms
- Logging integration
"""

import pytest
import numpy as np
import tempfile
import pickle
import os
import sys
from unittest.mock import Mock, patch, MagicMock, call
from multiprocessing import Pool
from io import StringIO
import matplotlib.pyplot as plt

from propflow.simulator import Simulator, _setup_logger
from propflow.configs.global_config_mapping import (
    SIMULATOR_DEFAULTS,
    LOG_LEVELS,
    LOGGING_CONFIG,
)
from propflow.policies import ConvergenceConfig
from propflow.bp.engines_realizations import BPEngine


@pytest.mark.unit
class TestSetupLogger:
    """Test the _setup_logger helper function."""

    def test_setup_logger_with_default_level(self):
        """Test _setup_logger with default level."""
        logger = _setup_logger()

        expected_level = LOG_LEVELS[SIMULATOR_DEFAULTS["default_log_level"]]
        assert logger.level == expected_level

    def test_setup_logger_with_custom_level(self):
        """Test _setup_logger with custom level."""
        logger = _setup_logger("HIGH")

        expected_level = LOG_LEVELS["HIGH"]
        assert logger.level == expected_level

    def test_setup_logger_with_invalid_level(self):
        """Test _setup_logger with invalid level."""
        logger = _setup_logger("INVALID")

        # Should fall back to centralized default
        expected_level = LOGGING_CONFIG["default_level"]
        assert logger.level == expected_level

    def test_setup_logger_with_non_string_level(self):
        """Test _setup_logger with non-string level."""
        logger = _setup_logger(123)

        # Should fall back to centralized default
        expected_level = LOGGING_CONFIG["default_level"]
        assert logger.level == expected_level

    def test_setup_logger_uses_centralized_config(self):
        """Test that _setup_logger uses centralized configuration."""
        with patch(
            "propflow.simulator.SIMULATOR_DEFAULTS", {"default_log_level": "HIGH"}
        ):
            with patch("propflow.simulator.LOG_LEVELS", {"HIGH": 999}):
                logger = _setup_logger(None)
                assert logger.level == 999


@pytest.mark.unit
class TestSimulatorInitialization:
    """Test Simulator initialization and configuration."""

    @pytest.fixture
    def engine_configs(self):
        """Sample engine configurations for testing."""
        return {
            "BPEngine": {"class": Mock},
            "TestEngine": {"class": Mock, "param1": "value1"},
        }

    def test_simulator_default_initialization(self, engine_configs):
        """Test Simulator initialization with defaults."""
        simulator = Simulator(engine_configs)

        assert simulator.engine_configs == engine_configs
        assert simulator.timeout == SIMULATOR_DEFAULTS["timeout"]
        assert len(simulator.results) == len(engine_configs)
        for engine_name in engine_configs:
            assert simulator.results[engine_name] == []

    def test_simulator_with_custom_log_level(self, engine_configs):
        """Test Simulator initialization with custom log level."""
        simulator = Simulator(engine_configs, log_level="HIGH")

        expected_level = LOG_LEVELS["HIGH"]
        assert simulator.logger.level == expected_level

    def test_simulator_with_none_log_level(self, engine_configs):
        """Test Simulator initialization with None log level."""
        simulator = Simulator(engine_configs, log_level=None)

        # Should use centralized default
        expected_level = LOG_LEVELS[SIMULATOR_DEFAULTS["default_log_level"]]
        assert simulator.logger.level == expected_level

    def test_simulator_results_initialization(self, engine_configs):
        """Test that results dictionary is properly initialized."""
        simulator = Simulator(engine_configs)

        assert len(simulator.results) == len(engine_configs)
        for engine_name in engine_configs:
            assert engine_name in simulator.results
            assert simulator.results[engine_name] == []


@pytest.mark.unit
class TestSimulatorRunSimulations:
    """Test Simulator.run_simulations method."""

    @pytest.fixture
    def engine_configs(self):
        return {"TestEngine": {"class": Mock}}

    @pytest.fixture
    def simulator(self, engine_configs):
        return Simulator(engine_configs)

    @pytest.fixture
    def mock_graphs(self):
        """Create mock factor graphs for testing."""
        graphs = []
        for i in range(3):
            graph = Mock()
            graph.name = f"graph_{i}"
            graphs.append(graph)
        return graphs

    def test_run_simulations_with_default_max_iter(self, simulator, mock_graphs):
        """Test run_simulations with default max_iter."""
        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = []

            result = simulator.run_simulations(mock_graphs)

            # Should use default max_iter from centralized config
            assert result == simulator.results
            mock_batch.assert_called_once()

            # Check simulation args passed to batch processing
            call_args = mock_batch.call_args[0][0]
            for args in call_args:
                assert args[4] == SIMULATOR_DEFAULTS["default_max_iter"]  # max_iter

    def test_run_simulations_with_custom_max_iter(self, simulator, mock_graphs):
        """Test run_simulations with custom max_iter."""
        custom_max_iter = 2000

        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = []

            simulator.run_simulations(mock_graphs, max_iter=custom_max_iter)

            # Check that custom max_iter is used
            call_args = mock_batch.call_args[0][0]
            for args in call_args:
                assert args[4] == custom_max_iter

    def test_run_simulations_creates_correct_args(self, simulator, mock_graphs):
        """Test that run_simulations creates correct simulation arguments."""
        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = []

            simulator.run_simulations(mock_graphs, max_iter=1000)

            call_args = mock_batch.call_args[0][0]
            expected_num_sims = len(mock_graphs) * len(simulator.engine_configs)
            assert len(call_args) == expected_num_sims

            # Check structure of simulation arguments
            for args in call_args:
                assert len(args) == 6
                assert isinstance(args[0], int)  # graph_index
                assert isinstance(args[1], str)  # engine_name
                assert isinstance(args[2], dict)  # config
                assert isinstance(args[3], bytes)  # pickled graph_data
                assert args[4] == 1000  # max_iter
                assert isinstance(args[5], int)  # log_level

    def test_run_simulations_handles_results(self, simulator, mock_graphs):
        """Test that run_simulations properly handles and aggregates results."""
        mock_results = [
            (0, "TestEngine", [1.0, 2.0, 3.0]),
            (1, "TestEngine", [2.0, 3.0, 4.0]),
            (2, "TestEngine", [3.0, 4.0, 5.0]),
        ]

        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = mock_results

            result = simulator.run_simulations(mock_graphs)

            assert len(simulator.results["TestEngine"]) == 3
            assert simulator.results["TestEngine"][0] == [1.0, 2.0, 3.0]
            assert simulator.results["TestEngine"][1] == [2.0, 3.0, 4.0]
            assert simulator.results["TestEngine"][2] == [3.0, 4.0, 5.0]

    def test_run_simulations_logs_warnings(self, simulator, mock_graphs):
        """Test that run_simulations logs appropriate warning messages."""
        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = []

            with patch.object(simulator.logger, "warning") as mock_warning:
                simulator.run_simulations(mock_graphs)

                # Should log preparation and completion messages
                mock_warning.assert_any_call(
                    f"Preparing {len(mock_graphs) * len(simulator.engine_configs)} total simulations."
                )

    def test_run_simulations_handles_exceptions(self, simulator, mock_graphs):
        """Test that run_simulations handles exceptions gracefully."""
        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.side_effect = RuntimeError("Test error")

            with patch.object(simulator, "_sequential_fallback") as mock_fallback:
                mock_fallback.return_value = []

                with patch.object(simulator.logger, "error") as mock_error:
                    simulator.run_simulations(mock_graphs)

                    mock_error.assert_called()
                    mock_fallback.assert_called_once()


@pytest.mark.unit
class TestSimulatorRunSingleSimulation:
    """Test the _run_single_simulation static method."""

    def test_run_single_simulation_success(self):
        """Test successful single simulation execution."""
        # Create mock engine class and instance
        mock_engine_instance = Mock()
        mock_engine_instance.run.return_value = None
        mock_engine_instance.history.costs = [1.0, 2.0, 3.0]

        mock_engine_class = Mock(return_value=mock_engine_instance)

        # Create mock factor graph
        mock_graph = Mock()

        # Create simulation arguments
        args = (
            0,  # graph_index
            "TestEngine",  # engine_name
            {"class": mock_engine_class, "param1": "value1"},  # config
            pickle.dumps(mock_graph),  # graph_data
            1000,  # max_iter
            20,  # log_level
        )

        result = Simulator._run_single_simulation(args)

        assert result == (0, "TestEngine", [1.0, 2.0, 3.0])
        mock_engine_class.assert_called_once()
        mock_engine_instance.run.assert_called_once_with(max_iter=1000)

    def test_run_single_simulation_with_convergence_config(self):
        """Test single simulation with convergence config."""
        mock_engine_instance = Mock()
        mock_engine_instance.run.return_value = None
        mock_engine_instance.history.costs = [1.0, 2.0]

        mock_engine_class = Mock(return_value=mock_engine_instance)

        mock_graph = Mock()
        args = (
            0,
            "TestEngine",
            {"class": mock_engine_class},
            pickle.dumps(mock_graph),
            500,
            20,
        )

        with patch("propflow.simulator.ConvergenceConfig") as mock_conv_config:
            result = Simulator._run_single_simulation(args)

            # Should create ConvergenceConfig and pass to engine
            mock_conv_config.assert_called_once()
            mock_engine_class.assert_called_once()

            call_kwargs = mock_engine_class.call_args[1]
            assert "convergence_config" in call_kwargs

    def test_run_single_simulation_exception_handling(self):
        """Test that single simulation handles exceptions gracefully."""
        mock_engine_class = Mock(side_effect=RuntimeError("Test error"))

        mock_graph = Mock()
        args = (
            0,
            "TestEngine",
            {"class": mock_engine_class},
            pickle.dumps(mock_graph),
            1000,
            20,
        )

        result = Simulator._run_single_simulation(args)

        # Should return empty costs on failure
        assert result == (0, "TestEngine", [])

    def test_run_single_simulation_engine_params(self):
        """Test that engine parameters are correctly passed."""
        mock_engine_instance = Mock()
        mock_engine_instance.history.costs = [1.0]

        mock_engine_class = Mock(return_value=mock_engine_instance)

        config = {
            "class": mock_engine_class,
            "param1": "value1",
            "param2": 42,
            "param3": True,
        }

        mock_graph = Mock()
        args = (0, "TestEngine", config, pickle.dumps(mock_graph), 1000, 20)

        Simulator._run_single_simulation(args)

        # Check that engine was called with correct parameters
        call_kwargs = mock_engine_class.call_args[1]
        assert call_kwargs["param1"] == "value1"
        assert call_kwargs["param2"] == 42
        assert call_kwargs["param3"] == True
        assert "class" not in call_kwargs  # Should be filtered out


@pytest.mark.unit
class TestSimulatorPlotResults:
    """Test Simulator.plot_results method."""

    @pytest.fixture
    def simulator_with_results(self):
        """Create simulator with mock results for testing."""
        engine_configs = {"Engine1": {"class": Mock}, "Engine2": {"class": Mock}}
        simulator = Simulator(engine_configs)

        # Add mock results
        simulator.results = {
            "Engine1": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            "Engine2": [[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]],
        }

        return simulator

    def test_plot_results_with_default_max_iter(self, simulator_with_results):
        """Test plot_results with default max_iter."""
        with patch("matplotlib.pyplot.figure") as mock_figure:
            with patch("matplotlib.pyplot.plot") as mock_plot:
                with patch("matplotlib.pyplot.show") as mock_show:
                    simulator_with_results.plot_results()

                    mock_figure.assert_called_once_with(figsize=(12, 8))
                    assert mock_plot.call_count > 0  # Should make plotting calls
                    mock_show.assert_called_once()

    def test_plot_results_with_custom_max_iter(self, simulator_with_results):
        """Test plot_results with custom max_iter."""
        with patch("matplotlib.pyplot.figure"):
            with patch("matplotlib.pyplot.plot"):
                with patch("matplotlib.pyplot.show"):
                    simulator_with_results.plot_results(max_iter=2000)

                    # Should complete without errors

    def test_plot_results_verbose_mode(self, simulator_with_results):
        """Test plot_results in verbose mode."""
        with patch("matplotlib.pyplot.figure"):
            with patch("matplotlib.pyplot.plot") as mock_plot:
                with patch("matplotlib.pyplot.fill_between") as mock_fill:
                    with patch("matplotlib.pyplot.show"):
                        simulator_with_results.plot_results(verbose=True)

                        # In verbose mode, should make additional plotting calls
                        assert mock_plot.call_count > 2  # Individual runs + averages
                        assert mock_fill.call_count > 0  # Standard deviation bands

    def test_plot_results_empty_results(self):
        """Test plot_results with empty results."""
        engine_configs = {"Engine1": {"class": Mock}}
        simulator = Simulator(engine_configs)

        with patch("matplotlib.pyplot.figure"):
            with patch("matplotlib.pyplot.show"):
                with patch.object(simulator.logger, "error") as mock_error:
                    simulator.plot_results()

                    # Should log error for empty results
                    mock_error.assert_called()

    def test_plot_results_invalid_data(self):
        """Test plot_results with invalid data."""
        engine_configs = {"Engine1": {"class": Mock}}
        simulator = Simulator(engine_configs)
        simulator.results = {"Engine1": [None, [], "invalid"]}

        with patch("matplotlib.pyplot.figure"):
            with patch("matplotlib.pyplot.show"):
                with patch.object(simulator.logger, "error") as mock_error:
                    simulator.plot_results()

                    # Should handle invalid data gracefully
                    mock_error.assert_called()


@pytest.mark.unit
class TestSimulatorUtilityMethods:
    """Test Simulator utility methods."""

    @pytest.fixture
    def simulator(self):
        return Simulator({"Engine1": {"class": Mock}})

    def test_set_log_level_valid(self, simulator):
        """Test set_log_level with valid level."""
        with patch.object(simulator.logger, "setLevel") as mock_set_level:
            with patch.object(simulator.logger, "warning") as mock_warning:
                simulator.set_log_level("HIGH")

                expected_level = LOG_LEVELS["HIGH"]
                mock_set_level.assert_called_once_with(expected_level)
                mock_warning.assert_called_once_with("Log level set to HIGH")

    def test_set_log_level_invalid(self, simulator):
        """Test set_log_level with invalid level."""
        with patch.object(simulator.logger, "error") as mock_error:
            simulator.set_log_level("INVALID")

            mock_error.assert_called_once_with("Invalid log level: INVALID")

    def test_set_log_level_non_string(self, simulator):
        """Test set_log_level with non-string level."""
        with patch.object(simulator.logger, "error") as mock_error:
            simulator.set_log_level(123)

            mock_error.assert_called_once_with("Invalid log level: 123")

    def test_run_batch_safe_success(self, simulator):
        """Test _run_batch_safe successful execution."""
        mock_args = [("arg1",), ("arg2",), ("arg3",)]

        with patch("multiprocessing.Pool") as mock_pool_class:
            mock_pool = Mock()
            mock_result = Mock()
            mock_result.get.return_value = ["result1", "result2", "result3"]

            mock_pool.map_async.return_value = mock_result
            mock_pool_class.return_value.__enter__.return_value = mock_pool

            results = simulator._run_batch_safe(mock_args, max_workers=2)

            assert results == ["result1", "result2", "result3"]
            mock_pool_class.assert_called_once_with(processes=2)
            mock_pool.map_async.assert_called_once()
            mock_result.get.assert_called_once_with(timeout=simulator.timeout)

    def test_run_batch_safe_exception(self, simulator):
        """Test _run_batch_safe exception handling."""
        mock_args = [("arg1",), ("arg2",)]

        with patch("multiprocessing.Pool") as mock_pool_class:
            mock_pool_class.side_effect = RuntimeError("Pool creation failed")

            with patch.object(simulator, "_run_in_batches") as mock_batches:
                mock_batches.return_value = ["batch_result"]

                results = simulator._run_batch_safe(mock_args)

                assert results == ["batch_result"]
                mock_batches.assert_called_once()

    def test_sequential_fallback(self, simulator):
        """Test _sequential_fallback method."""
        mock_args = [("arg1",), ("arg2",)]

        with patch.object(Simulator, "_run_single_simulation") as mock_single:
            mock_single.side_effect = ["result1", "result2"]

            results = simulator._sequential_fallback(mock_args)

            assert results == ["result1", "result2"]
            assert mock_single.call_count == 2


@pytest.mark.integration
class TestSimulatorIntegration:
    """Integration tests for Simulator."""

    def test_simulator_full_workflow_mock(self):
        """Test complete simulation workflow with mocks."""
        # Create mock engine
        mock_engine_instance = Mock()
        mock_engine_instance.run.return_value = None
        mock_engine_instance.history.costs = [10.0, 5.0, 3.0]

        mock_engine_class = Mock(return_value=mock_engine_instance)

        engine_configs = {"MockEngine": {"class": mock_engine_class}}
        simulator = Simulator(engine_configs)

        # Create mock graphs
        mock_graphs = [Mock() for _ in range(2)]

        # Run simulation
        with patch("multiprocessing.cpu_count", return_value=2):
            results = simulator.run_simulations(mock_graphs, max_iter=100)

        # Check results
        assert "MockEngine" in results
        assert len(results["MockEngine"]) == 2
        assert all(len(cost_list) == 3 for cost_list in results["MockEngine"])

    def test_simulator_with_centralized_config(self):
        """Test that simulator properly uses centralized configuration."""
        engine_configs = {"Engine1": {"class": Mock}}

        # Test with default configuration
        simulator1 = Simulator(engine_configs)
        assert simulator1.timeout == SIMULATOR_DEFAULTS["timeout"]

        # Test with custom log level
        simulator2 = Simulator(engine_configs, log_level="HIGH")
        expected_level = LOG_LEVELS["HIGH"]
        assert simulator2.logger.level == expected_level

    def test_simulator_error_recovery(self):
        """Test simulator error recovery mechanisms."""
        # Create engine that fails
        mock_engine_class = Mock(side_effect=RuntimeError("Engine failed"))

        engine_configs = {"FailingEngine": {"class": mock_engine_class}}
        simulator = Simulator(engine_configs)

        mock_graphs = [Mock()]

        # Should handle failure gracefully
        with patch.object(simulator.logger, "error"):
            results = simulator.run_simulations(mock_graphs)

        # Should still return results structure
        assert "FailingEngine" in results

    def test_simulator_logging_integration(self):
        """Test that simulator properly integrates with logging system."""
        engine_configs = {"Engine1": {"class": Mock}}

        # Test with different log levels
        for level_name in LOG_LEVELS:
            simulator = Simulator(engine_configs, log_level=level_name)
            assert simulator.logger.level == LOG_LEVELS[level_name]


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_graphs,num_engines", [(1, 1), (2, 1), (1, 2), (3, 2), (5, 3)]
)
class TestSimulatorParametrized:
    """Parametrized tests for Simulator with different configurations."""

    def test_simulation_args_generation(self, num_graphs, num_engines):
        """Test that correct number of simulation args are generated."""
        engine_configs = {f"Engine{i}": {"class": Mock} for i in range(num_engines)}
        simulator = Simulator(engine_configs)

        mock_graphs = [Mock() for _ in range(num_graphs)]

        with patch.object(simulator, "_run_batch_safe") as mock_batch:
            mock_batch.return_value = []

            simulator.run_simulations(mock_graphs)

            # Check that correct number of simulation args were created
            call_args = mock_batch.call_args[0][0]
            expected_num_sims = num_graphs * num_engines
            assert len(call_args) == expected_num_sims

    def test_results_structure_with_different_configs(self, num_graphs, num_engines):
        """Test results structure with different numbers of graphs and engines."""
        engine_configs = {f"Engine{i}": {"class": Mock} for i in range(num_engines)}
        simulator = Simulator(engine_configs)

        # Check initial results structure
        assert len(simulator.results) == num_engines
        for i in range(num_engines):
            assert f"Engine{i}" in simulator.results
            assert simulator.results[f"Engine{i}"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
