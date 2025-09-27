"""A parallelized simulator for running and comparing multiple engine configurations.

This module provides a `Simulator` class that can run multiple belief propagation
engine configurations across a set of factor graphs in parallel. It uses Python's
`multiprocessing` module to distribute the simulation runs, collects the results,
and provides a simple plotting utility to visualize and compare the performance
of different engines.
"""
import logging
import time
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple

import colorlog
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import traceback

from .configs import Logger
from .configs.global_config_mapping import LOG_LEVELS, LOGGING_CONFIG, SIMULATOR_DEFAULTS
from .policies import ConvergenceConfig


def _setup_logger(level: Optional[str] = None) -> Logger:
    """Configures and returns a logger for the simulator."""
    safe_level = level if isinstance(level, str) else SIMULATOR_DEFAULTS["default_log_level"]
    log_level = LOG_LEVELS.get(safe_level.upper(), LOGGING_CONFIG["default_level"])
    logger = Logger("Simulator")
    logger.setLevel(log_level)

    if not logger.handlers:
        console = colorlog.StreamHandler(sys.stdout)
        console.setFormatter(
            colorlog.ColoredFormatter(LOGGING_CONFIG["console_format"], log_colors=LOGGING_CONFIG["console_colors"])
        )
        logger.addHandler(console)
    return logger


class Simulator:
    """Orchestrates parallel execution of multiple simulation configurations.

    This class takes a set of engine configurations and a list of factor graphs,
    runs each engine on each graph in parallel, collects the cost history from
    each run, and provides methods to visualize the aggregated results.

    Attributes:
        engine_configs (dict): A dictionary mapping engine names to their configurations.
        logger (Logger): A configured logger instance.
        results (dict): A dictionary to store the results of the simulations.
        timeout (int): The timeout in seconds for multiprocessing tasks.
    """
    def __init__(self, engine_configs: Dict[str, Any], log_level: Optional[str] = None):
        """Initializes the Simulator.

        Args:
            engine_configs: A dictionary where keys are descriptive engine names
                and values are configuration dictionaries for each engine.
            log_level: The logging level for the simulator (e.g., 'INFO', 'DEBUG').
        """
        self.engine_configs = engine_configs
        self.logger = _setup_logger(log_level)
        self.results: Dict[str, List[List[float]]] = {name: [] for name in engine_configs}
        self.timeout = SIMULATOR_DEFAULTS["timeout"]

    def run_simulations(self, graphs: List[Any], max_iter: Optional[int] = None) -> Dict[str, List[List[float]]]:
        """Runs all engine configurations on all provided graphs in parallel.

        Args:
            graphs: A list of factor graph objects to run simulations on.
            max_iter: The maximum number of iterations for each simulation run.

        Returns:
            A dictionary containing the collected results, where keys are engine
            names and values are lists of cost histories for each run.
        """
        max_iter = max_iter or SIMULATOR_DEFAULTS["default_max_iter"]
        self.logger.warning(f"Preparing {len(graphs) * len(self.engine_configs)} total simulations.")

        simulation_args = [
            (i, name, config, pickle.dumps(graph), max_iter, self.logger.level)
            for i, graph in enumerate(graphs) for name, config in self.engine_configs.items()
        ]

        start_time = time.time()
        try:
            all_results = self._run_batch_safe(simulation_args, max_workers=cpu_count())
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR - All multiprocessing strategies failed: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.warning("Falling back to sequential processing...")
            all_results = self._sequential_fallback(simulation_args)

        total_time = time.time() - start_time
        self.logger.warning(f"All simulations completed in {total_time:.2f} seconds.")

        if len(all_results) != len(simulation_args):
            self.logger.error(f"Expected {len(simulation_args)} results, but got {len(all_results)}")

        for _, engine_name, costs in all_results:
            self.results[engine_name].append(costs)

        for engine_name, costs_list in self.results.items():
            self.logger.warning(f"{engine_name}: {len(costs_list)} runs completed.")
        return self.results

    def plot_results(self, max_iter: Optional[int] = None, verbose: bool = False) -> None:
        """Plots the average cost convergence for each engine configuration.

        Args:
            max_iter: The maximum number of iterations to display on the plot.
            verbose: If True, plots individual simulation runs with transparency
                and standard deviation bands around the average.
        """
        max_iter = max_iter or SIMULATOR_DEFAULTS["default_max_iter"]
        self.logger.warning(f"Starting plotting... (Verbose: {verbose})")
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))

        for idx, (engine_name, costs_list) in enumerate(self.results.items()):
            valid_costs_list = [c for c in costs_list if c]
            if not valid_costs_list:
                self.logger.error(f"No valid cost data for {engine_name}")
                continue

            max_len = max(max_iter, max(len(c) for c in valid_costs_list))
            padded_costs = np.array([c + [c[-1]] * (max_len - len(c)) for c in valid_costs_list])
            avg_costs = np.mean(padded_costs, axis=0)
            color = colors[idx]

            if verbose:
                for i in range(padded_costs.shape[0]):
                    plt.plot(padded_costs[i, :], color=color, alpha=0.2, linewidth=0.5)
                std_costs = np.std(padded_costs, axis=0)
                plt.fill_between(range(max_len), avg_costs - std_costs, avg_costs + std_costs, color=color, alpha=0.1)

            plt.plot(avg_costs, label=f"{engine_name} (Avg)", color=color, linewidth=2)
            self.logger.warning(f"Plotted {engine_name}: avg final cost = {avg_costs[-1]:.2f}")

        plt.title(f"Average Costs over {len(self.results.get(list(self.results.keys())[0], []))} Runs", fontsize=14)
        plt.xlabel("Iteration", fontsize=12); plt.ylabel("Average Cost", fontsize=12)
        plt.legend(fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        self.logger.warning("Displaying plot.")

    def set_log_level(self, level: str) -> None:
        """Sets the logging level for the simulator's logger.

        Args:
            level: The desired logging level (e.g., 'INFO', 'DEBUG').
        """
        log_level = LOG_LEVELS.get(level.upper())
        if log_level:
            self.logger.setLevel(log_level)
            self.logger.warning(f"Log level set to {level.upper()}")
        else:
            self.logger.error(f"Invalid log level: {level}")

    @staticmethod
    def _run_single_simulation(args: Tuple) -> Tuple[int, str, List[float]]:
        """A static method to run a single simulation instance, designed for multiprocessing."""
        graph_index, engine_name, config, graph_data, max_iter, log_level = args
        logger = _setup_logger(str(log_level))
        try:
            fg_copy = pickle.loads(graph_data)
            engine_class = config["class"]
            engine_params = {k: v for k, v in config.items() if k != "class"}
            engine = engine_class(factor_graph=fg_copy, convergence_config=ConvergenceConfig(), **engine_params)
            engine.run(max_iter=max_iter)
            costs = engine.history.costs
            logger.info(f"Finished: graph {graph_index}, engine {engine_name}. Final cost: {costs[-1] if costs else 'N/A'}")
            return (graph_index, engine_name, costs)
        except Exception as e:
            logger.error(f"Exception in child process for graph {graph_index}, engine {engine_name}: {e}\n{traceback.format_exc()}")
            return (graph_index, engine_name, [])

    def _run_batch_safe(self, simulation_args: List[Tuple], max_workers: int) -> List[Tuple]:
        """Runs simulations in parallel with a timeout, falling back to batching."""
        self.logger.warning(f"Attempting full multiprocessing with {max_workers} processes...")
        try:
            with Pool(processes=max_workers) as pool:
                result = pool.map_async(self._run_single_simulation, simulation_args)
                return result.get(timeout=self.timeout)
        except Exception as e:
            self.logger.error(f"Full multiprocessing failed: {e}")
            self.logger.warning("Trying batch processing...")
            return self._run_in_batches(simulation_args, max_workers=max(1, max_workers // 2))

    def _run_in_batches(self, simulation_args: List[Tuple], batch_size: int = 8, max_workers: int = 4) -> List[Tuple]:
        """Runs simulations in smaller parallel batches as a fallback."""
        self.logger.warning(f"Starting batch processing with batch_size={batch_size} and max_workers={max_workers}")
        all_results = []
        for i in range(0, len(simulation_args), batch_size):
            batch = simulation_args[i:i + batch_size]
            self.logger.warning(f"Running batch {i // batch_size + 1}/{len(simulation_args) // batch_size + 1}...")
            try:
                with Pool(processes=min(max_workers, len(batch))) as pool:
                    all_results.extend(pool.map(self._run_single_simulation, batch))
            except Exception as e:
                self.logger.error(f"Batch failed: {e}. Running sequentially as fallback.")
                all_results.extend(self._sequential_fallback(batch))
        return all_results

    def _sequential_fallback(self, simulation_args: List[Tuple]) -> List[Tuple]:
        """Runs all simulations sequentially as a final fallback."""
        self.logger.warning("Running all simulations sequentially as a last resort.")
        return [self._run_single_simulation(args) for args in simulation_args]
