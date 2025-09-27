"""A script to run a baseline experiment comparing the performance of the
standard `BPEngine` with the `MessagePruningEngine`.

This script automates the process of:
1. Creating a sample factor graph configuration.
2. Building the factor graph from the configuration.
3. Running a simulation with the standard `BPEngine`.
4. Running a simulation with the `MessagePruningEngine` on the same graph.
5. Printing a summary that compares the performance metrics of both engines,
   such as total messages processed, execution time, and pruning rate.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ..bp.engines import BPEngine, MessagePruningEngine
from .create.create_factor_graph_config import ConfigCreator
from .create.create_factor_graphs_from_config import (
    FactorGraphBuilder,
)
from ..bp.computators import MinSumComputator
import logging

logging.basicConfig(level=logging.INFO)


def run_baseline_comparison() -> dict:
    """Runs a side-by-side comparison of a regular BP engine and a pruning BP engine.

    This function creates a test graph, runs both engines on it for a fixed
    number of iterations, and collects performance and pruning statistics.
    The results of the comparison are printed to the console.

    Returns:
        A dictionary containing the performance summaries for both the "regular"
        and "pruning" engine runs.
    """

    # Create test graph config
    creator = ConfigCreator()
    config_path = creator.create_graph_config(
        graph_type="cycle",
        num_variables=10,
        domain_size=3,
        ct_factory="random_int",
        ct_params={"low": 0, "high": 10},
    )

    # Build factor graph
    builder = FactorGraphBuilder()
    fg_path = builder.build_and_save(config_path)
    factor_graph = builder.load_graph(fg_path)

    results = {}

    # Test regular engine
    print("Testing regular BPEngine...")
    regular_engine = BPEngine(
        factor_graph=factor_graph,
        computator=MinSumComputator(),
        monitor_performance=True,
    )
    regular_engine.run(max_iter=50, save_csv=False)
    regular_summary = regular_engine.performance_monitor.get_summary()
    results["regular"] = regular_summary

    # Test pruning engine
    print("Testing MessagePruningEngine...")
    pruning_engine = MessagePruningEngine(
        factor_graph=factor_graph,
        computator=MinSumComputator(),
        prune_threshold=1e-4,
        monitor_performance=True,
    )
    pruning_engine.run(max_iter=50, save_csv=False)
    pruning_summary = pruning_engine.performance_monitor.get_summary()
    # The `pruning_policy` attribute is set in the engine's `post_init` hook.
    pruning_stats = pruning_engine.pruning_policy.get_stats()
    results["pruning"] = {**pruning_summary, **pruning_stats}

    # Print comparison
    print("\n=== BASELINE COMPARISON ===")
    print(f"Regular Engine:")
    print(f"  Total messages: {results['regular'].get('total_messages', 'N/A')}")
    print(f"  Total time: {results['regular'].get('total_time', 0.0):.3f}s")
    print(f"  Avg memory: {results['regular'].get('avg_memory_mb', 'N/A')} MB")

    print(f"\nPruning Engine:")
    print(f"  Total messages: {results['pruning'].get('total_messages', 'N/A')}")
    print(f"  Pruned messages: {results['pruning'].get('pruned_messages', 'N/A')}")
    print(f"  Pruning rate: {results['pruning'].get('pruning_rate', 0.0):.2%}")
    print(f"  Total time: {results['pruning'].get('total_time', 0.0):.3f}s")
    print(f"  Avg memory: {results['pruning'].get('avg_memory_mb', 'N/A')} MB")

    # Calculate improvements
    regular_messages = results["regular"].get("total_messages", 0)
    if regular_messages > 0:
        pruned_messages = results["pruning"].get("total_messages", 0)
        msg_reduction = (regular_messages - pruned_messages) / regular_messages
        print(f"\nMessage reduction: {msg_reduction:.2%}")

    return results


if __name__ == "__main__":
    run_baseline_comparison()
