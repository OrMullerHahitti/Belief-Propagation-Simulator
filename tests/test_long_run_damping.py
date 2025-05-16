import os
import sys
import pytest
import logging
import numpy as np
import json
import time
from pathlib import Path

from bp_base.engines_realizations import SplitEngine, DampingEngine, DampingAndSplitting
from configs.global_config_mapping import PROJECT_ROOT

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Now import from utils
from utils.path_utils import load_pickle

from bp_base.factor_graph import FactorGraph
from bp_base.agents import VariableAgent, FactorAgent
from bp_base.components import Message
from bp_base.bp_engine_base import BPEngine
from configs.loggers import Logger

log_dir = "test_logs"
logger = Logger(__name__, file=True)
logger.setLevel(level=logging.DEBUG)


@pytest.fixture
def simple_factor_graph():
    # Use a smaller factor graph for testing
    pickle_path = os.path.join(
        PROJECT_ROOT,
        "configs",
        "factor_graphs",
        "factor-graph-random-30-random_intlow1,high1000.25-number107.pkl",  # If this exists, otherwise keep the original
    )
    if not os.path.exists(pickle_path):
        # Fallback to original graph
        pickle_path = os.path.join(
            PROJECT_ROOT,
            "configs",
            "factor_graphs",
            "test-factor.pkl",
        )

    logger.info(f"Loading factor graph from: {pickle_path}")
    start_time = time.time()
    fg = load_pickle(pickle_path)
    logger.info(f"Graph loaded in {time.time() - start_time:.2f} seconds")

    # Log basic graph statistics
    logger.info(
        f"Graph has {len(fg.variables)} variables and {len(fg.factors)} factors"
    )

    return fg


def test_bp_engine_long_run(simple_factor_graph):
    fg = simple_factor_graph
    logger.info("Creating BPEngine...")
    start_time = time.time()
    logger.debug(f"Factor graph: {len(fg.factors)}")
    # engine = DampingAndSplitting(factor_graph=fg)
    engine = DampingEngine(factor_graph=fg)
    engine = DampingAndDiscount(factor_graph=fg)
    logger.info(f"BPEngine initialized in {time.time() - start_time:.2f} seconds")

    # Run just 1 or 2 iterations with timing
    logger.info("Starting BP Engine run (just 1 iteration)...")
    start_time = time.time()
    engine.run(max_iter=100, save_json=False, save_csv=True)
    logger.debug(f"Factor graph: {len(fg.factors)}")

    logger.info(
        f"BP Engine completed 1 iteration in {time.time() - start_time:.2f} seconds"
    )
