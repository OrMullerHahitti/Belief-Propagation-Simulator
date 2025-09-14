#!/usr/bin/env python3
"""Baseline experiment to test message pruning effectiveness."""

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


def run_baseline_comparison():
    """Compare regular BP vs pruning BP on same graph."""

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
    pruning_stats = pruning_engine.message_pruning_policy.get_stats()
    results["pruning"] = {**pruning_summary, **pruning_stats}

    # Print comparison
    print("\n=== BASELINE COMPARISON ===")
    print(f"Regular Engine:")
    print(f"  Total messages: {results['regular']['total_messages']}")
    print(f"  Total time: {results['regular']['total_time']:.3f}s")
    print(f"  Avg memory: {results['regular'].get('avg_memory_mb', 'N/A')} MB")

    print(f"\nPruning Engine:")
    print(f"  Total messages: {results['pruning']['total_messages']}")
    print(f"  Pruned messages: {results['pruning']['pruned_messages']}")
    print(f"  Pruning rate: {results['pruning']['pruning_rate']:.2%}")
    print(f"  Total time: {results['pruning']['total_time']:.3f}s")
    print(f"  Avg memory: {results['pruning'].get('avg_memory_mb', 'N/A')} MB")

    # Calculate improvements
    if results["regular"]["total_messages"] > 0:
        msg_reduction = (
            results["regular"]["total_messages"] - results["pruning"]["total_messages"]
        ) / results["regular"]["total_messages"]
        print(f"\nMessage reduction: {msg_reduction:.2%}")

    return results


if __name__ == "__main__":
    run_baseline_comparison()
