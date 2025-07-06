import logging
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import colorlog
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pickle
import psutil
import traceback

from bp_base.factor_graph import FactorGraph
from bp_base.engine_base import BPEngine
from bp_base.engines_realizations import (
    DampingEngine,
    DampingSCFGEngine,
    SplitEngine,
    DampingCROnceEngine,
    CostReductionOnceEngine,
)
from utils.fg_utils import FGBuilder
from configs.global_config_mapping import CT_FACTORIES
from utils.path_utils import find_project_root
from policies.convergance import ConvergenceConfig

# --- Logging Setup ---
LOG_LEVELS = {
    "VERBOSE": logging.DEBUG,
    "MILD": logging.INFO,
    "INFORMATIVE": logging.WARNING,
    "HIGH": logging.ERROR,
}


def _setup_logger(level='INFORMATIVE'):
    # Ensure level is a string before calling .upper()
    safe_level = level if isinstance(level, str) else 'INFORMATIVE'
    log_level = LOG_LEVELS.get(safe_level.upper(), logging.WARNING)
    logger = logging.getLogger('Simulator')
    logger.setLevel(log_level)

    # Avoid adding handlers if they already exist
    if not logger.handlers:

        console = colorlog.StreamHandler(sys.stdout)
        console.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        logger.addHandler(console)


    return logger

class Simulator:
    def __init__(self, engine_configs, log_level='INFORMATIVE'):
        self.engine_configs = engine_configs
        self.logger = _setup_logger(log_level)
        self.results = {engine_name: [] for engine_name in self.engine_configs.keys()}

    def set_log_level(self, level):
        # Ensure level is a string before calling .upper()
        safe_level = level if isinstance(level, str) else 'INFORMATIVE'
        log_level = LOG_LEVELS.get(safe_level.upper())
        if log_level is not None:
            self.logger.setLevel(log_level)
            self.logger.warning(f"Log level set to {safe_level.upper()}")
        else:
            self.logger.error(f"Invalid log level: {level}")

    def run_simulations(self, graphs, max_iter=5000):
        """Run simulations for all graphs and engine configurations."""
        self.logger.warning(f"Preparing {len(graphs) * len(self.engine_configs)} total simulations.")

        simulation_args = self._prepare_simulation_arguments(graphs, max_iter)
        all_results = self._execute_simulations(simulation_args)
        self._process_simulation_results(all_results, simulation_args)

        return self.results

    def _prepare_simulation_arguments(self, graphs, max_iter):
        """Prepare arguments for all simulation combinations."""
        simulation_args = []
        for i, graph in enumerate(graphs):
            graph_data = pickle.dumps(graph)
            for engine_name, config in self.engine_configs.items():
                args = (i, engine_name, config, graph_data, max_iter, self.logger.level)
                simulation_args.append(args)
        return simulation_args

    def _execute_simulations(self, simulation_args):
        """Execute all simulations with fallback strategies."""
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
        return all_results

    def _process_simulation_results(self, all_results, simulation_args):
        """Process and organize simulation results."""
        if len(all_results) != len(simulation_args):
            self.logger.error(f"Expected {len(simulation_args)} results, but got {len(all_results)}")

        for graph_index, engine_name, costs in all_results:
            self.results[engine_name].append(costs)

        for engine_name, costs_list in self.results.items():
            self.logger.warning(f"{engine_name}: {len(costs_list)} runs completed.")

    def _run_batch_safe(self, simulation_args, max_workers=None):
        if max_workers is None:
            max_workers = cpu_count()

        self.logger.warning(f"Attempting full multiprocessing with {max_workers} processes...")

        try:
            with Pool(processes=max_workers) as pool:
                result = pool.map_async(self._run_single_simulation, simulation_args)

                # Wait with timeout
                all_results = result.get(timeout=600)

            self.logger.warning(f"SUCCESS - Full multiprocessing completed.")
            return all_results
        except Exception as e:
            self.logger.error(f"Full multiprocessing failed: {e}")
            self.logger.warning("Trying batch processing...")
            return self._run_in_batches(simulation_args, max_workers=max(1, max_workers // 2))

    def _run_in_batches(self, simulation_args, batch_size=None, max_workers=None):
        """Run simulations in batches with fallback strategies."""
        batch_config = self._configure_batch_processing(batch_size, max_workers)
        self.logger.warning(f"Starting batch processing with batch_size={batch_config['batch_size']} and max_workers={batch_config['max_workers']}")

        all_results = []
        batches = self._create_batches(simulation_args, batch_config['batch_size'])
        
        for batch_num, batch in enumerate(batches, 1):
            batch_results = self._process_single_batch(batch, batch_num, len(batches), batch_config['max_workers'])
            all_results.extend(batch_results)

        return all_results

    def _configure_batch_processing(self, batch_size, max_workers):
        """Configure batch processing parameters."""
        if max_workers is None:
            max_workers = min(6, cpu_count())
        if batch_size is None:
            batch_size = max(8, max_workers * 2)
        return {'batch_size': batch_size, 'max_workers': max_workers}

    def _create_batches(self, simulation_args, batch_size):
        """Create batches from simulation arguments."""
        return [simulation_args[i:i + batch_size] 
                for i in range(0, len(simulation_args), batch_size)]

    def _process_single_batch(self, batch, batch_num, total_batches, max_workers):
        """Process a single batch with multiprocessing and fallback."""
        self.logger.warning(f"Running batch {batch_num}/{total_batches}...")
        
        try:
            return self._run_batch_multiprocessing(batch, max_workers)
        except Exception as e:
            self.logger.error(f"Batch {batch_num} failed: {e}. Running sequentially as fallback.")
            return self._run_batch_sequential_fallback(batch, batch_num)

    def _run_batch_multiprocessing(self, batch, max_workers):
        """Run batch using multiprocessing."""
        with Pool(processes=min(max_workers, len(batch))) as pool:
            return pool.map(self._run_single_simulation, batch)

    def _run_batch_sequential_fallback(self, batch, batch_num):
        """Run batch sequentially as fallback."""
        results = []
        for args in batch:
            try:
                results.append(self._run_single_simulation(args))
            except Exception as seq_e:
                self.logger.error(f"Sequential item failed in batch {batch_num}: {seq_e}")
        return results

    def _sequential_fallback(self, simulation_args):
        self.logger.warning("Running all simulations sequentially as a last resort.")
        return [self._run_single_simulation(args) for args in simulation_args]

    @staticmethod
    def _run_single_simulation(args):
        graph_index, engine_name, config, graph_data, max_iter, log_level = args

        # Re-create a logger for the child process
        logger = logging.getLogger(f'Simulator-p{os.getpid()}')
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f'[%(asctime)s] [%(levelname)s] [p%(process)d] %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        try:
            logger.info(f"Starting simulation for graph {graph_index}, engine {engine_name}")

            fg_copy = pickle.loads(graph_data)

            engine_class = config["class"]
            engine_params = {k: v for k, v in config.items() if k != "class"}
            engine = engine_class(
                factor_graph=fg_copy,
                convergence_config=ConvergenceConfig(),
                **engine_params,
            )

            engine.run(max_iter=max_iter)

            costs = engine.history.costs
            logger.info(f"Finished simulation for graph {graph_index}, engine {engine_name}. Final cost: {costs[-1] if costs else 'N/A'}")

            return (graph_index, engine_name, costs)
        except Exception as e:
            logger.error(f"Exception in simulation for graph {graph_index}, engine {engine_name}: {e}")
            logger.error(traceback.format_exc())
            return (graph_index, engine_name, []) # Return empty costs on failure

    def plot_results(self, max_iter=5000, verbose=False):
        """Plot simulation results with optional verbose mode."""
        self.logger.warning(f"Starting plotting... (Verbose: {verbose})")
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for idx, (engine_name, costs_list) in enumerate(self.results.items()):
            processed_costs = self._validate_and_process_costs(engine_name, costs_list, max_iter)
            if processed_costs is None:
                continue
                
            color = colors[idx % len(colors)]
            self._plot_engine_results(engine_name, processed_costs, color, verbose)

        self._finalize_plot(verbose)

    def _validate_and_process_costs(self, engine_name, costs_list, max_iter):
        """Validate and process cost data for plotting."""
        if not costs_list:
            self.logger.error(f"No results for {engine_name}")
            return None

        valid_costs_list = [c for c in costs_list if c]
        if not valid_costs_list:
            self.logger.error(f"No valid cost data for {engine_name}")
            return None

        max_len = max(max(max_iter, len(c)) for c in valid_costs_list)
        padded_costs = [c + [c[-1]] * (max_len - len(c)) for c in valid_costs_list]

        if not padded_costs:
            self.logger.error(f"No valid costs to plot for {engine_name}")
            return None

        return np.array(padded_costs)

    def _plot_engine_results(self, engine_name, all_costs_np, color, verbose):
        """Plot results for a single engine."""
        avg_costs = np.average(all_costs_np, axis=0)
        max_len = all_costs_np.shape[1]

        if verbose:
            self._plot_individual_runs(all_costs_np, color)

        # Plot average line
        plt.plot(avg_costs, label=f'{engine_name} (Avg)', color=color, linewidth=2)

        if verbose:
            self._plot_uncertainty_bands(avg_costs, all_costs_np, max_len, color)

        self.logger.warning(f"Plotted {engine_name}: avg final cost = {avg_costs[-1]:.2f}")

    def _plot_individual_runs(self, all_costs_np, color):
        """Plot individual simulation runs with transparency."""
        for i in range(all_costs_np.shape[0]):
            plt.plot(all_costs_np[i, :], color=color, alpha=0.2, linewidth=0.5)

    def _plot_uncertainty_bands(self, avg_costs, all_costs_np, max_len, color):
        """Plot standard deviation bands around average."""
        std_costs = np.std(all_costs_np, axis=0)
        plt.fill_between(range(max_len), avg_costs - std_costs, 
                        avg_costs + std_costs, color=color, alpha=0.1)

    def _finalize_plot(self, verbose):
        """Add final plot elements and display."""
        title = "Average Costs over Runs on Random Factor Graphs"
        if verbose:
            title = "Verbose " + title
        plt.title(title, fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Average Cost", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        self.logger.warning("Displaying plot.")
